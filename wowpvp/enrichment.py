from __future__ import annotations

import json
import random
import threading
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
import time
from typing import Any

import pandas as pd
import requests
from requests.exceptions import RequestException

from wowpvp.constants import SPEC_ID_TO_INFO
from wowpvp.storage import PLAYER_COLUMNS, write_players_to_database
from wowpvp.utils import ensure_dirs, normalize_character_name, slugify_realm


WEB_PVP_CACHE_COLUMNS = [
    "player_key",
    "region",
    "realm_slug",
    "character_name",
    "status_code",
    "payload_json",
    "fetched_at",
]
RATING_COLUMNS = ["shuffle_rating", "blitz_rating", "rating_2v2", "rating_3v3", "rating_rbg"]
SPEC_MODE_TO_COLUMN = {
    "shuffle": "shuffle_rating",
    "blitz": "blitz_rating",
}
RETRYABLE_ERROR_MARKERS = (
    "too many 429",
    "max retries exceeded",
    "temporarily unavailable",
    "timed out",
    "timeout",
    "connection reset",
    "connection aborted",
    "remote disconnected",
    "name resolution",
    "getaddrinfo failed",
    "503",
    "504",
)
WOW_PROFILE_ORIGIN = "https://worldofwarcraft.blizzard.com"
WOW_PROFILE_HEADERS = {
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36 WoWPvPData/1.0"
    ),
}
WEB_RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
_thread_local = threading.local()
_request_lock = threading.Lock()
_next_request_at = 0.0


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def load_cache(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        df = pd.read_parquet(path)
    else:
        df = pd.DataFrame(columns=columns)
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    return df[columns]


def save_cache(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def append_cache(
    cache: pd.DataFrame,
    records: list[dict[str, Any]],
    columns: list[str],
    key_columns: list[str],
) -> pd.DataFrame:
    if not records:
        return cache
    new_rows = pd.DataFrame(records)
    for column in columns:
        if column not in new_rows.columns:
            new_rows[column] = pd.NA
    combined = pd.concat([cache, new_rows[columns]], ignore_index=True)
    return combined.drop_duplicates(key_columns, keep="last").reset_index(drop=True)


def player_identities(df: pd.DataFrame) -> pd.DataFrame:
    identities = df[["player_key", "region", "realm_slug", "character_name"]].copy()
    identities["region"] = identities["region"].astype(str).str.lower()
    identities["realm_slug"] = identities["realm_slug"].map(slugify_realm)
    identities["character_name"] = identities["character_name"].map(normalize_character_name)
    return identities.drop_duplicates("player_key").reset_index(drop=True)


def run_fetch_pool(
    label: str,
    tasks: list[Any],
    fetch_one: Any,
    max_workers: int,
    flush_every: int,
    cache: pd.DataFrame,
    cache_path: Path,
    columns: list[str],
    key_columns: list[str],
    max_task_attempts: int = 25,
    retry_delay_seconds: float = 20.0,
    max_retry_delay_seconds: float = 600.0,
    heartbeat_seconds: float = 15.0,
) -> pd.DataFrame:
    if not tasks:
        return cache

    max_task_attempts = max(1, int(max_task_attempts))
    retry_delay_seconds = max(1.0, float(retry_delay_seconds))
    max_retry_delay_seconds = max(retry_delay_seconds, float(max_retry_delay_seconds))
    heartbeat_seconds = max(1.0, float(heartbeat_seconds))

    def is_retryable_error(exc: Exception) -> bool:
        if isinstance(exc, RequestException):
            return True
        message = str(exc).lower()
        return any(marker in message for marker in RETRYABLE_ERROR_MARKERS)

    task_entries = list(enumerate(tasks))
    attempts: dict[int, int] = {index: 0 for index, _ in task_entries}
    pending = task_entries
    total = len(task_entries)
    completed = 0
    records: list[dict[str, Any]] = []
    round_index = 0
    permanent_failures = 0
    transient_failures = 0
    status_counts: Counter[str] = Counter()
    started_at = time.monotonic()
    last_heartbeat_at = started_at
    last_save_at = started_at

    def print_progress(force: bool = False) -> None:
        nonlocal last_heartbeat_at
        now = time.monotonic()
        if not force and now - last_heartbeat_at < heartbeat_seconds:
            return
        elapsed = max(0.001, now - started_at)
        rate_per_minute = completed / elapsed * 60
        remaining = max(0, total - completed)
        eta_minutes = remaining / rate_per_minute if rate_per_minute > 0 else float("inf")
        eta_text = f"{eta_minutes:.1f}m" if eta_minutes != float("inf") else "unknown"
        statuses = ", ".join(f"{key}:{value}" for key, value in status_counts.most_common(6))
        statuses = statuses or "none"
        print(
            f"{label}: progress {completed}/{total} ok, "
            f"retryable={transient_failures}, permanent={permanent_failures}, "
            f"statuses={statuses}, cache_rows={len(cache) + len(records)}, "
            f"speed={rate_per_minute:.0f}/min, eta={eta_text}",
            flush=True,
        )
        last_heartbeat_at = now

    while pending:
        round_index += 1
        worker_count = max(1, min(max_workers, len(pending)))
        retry_queue: list[tuple[int, Any]] = []

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures: dict[Any, tuple[int, Any]] = {}
            pending_index = 0

            def submit_available() -> None:
                nonlocal pending_index
                while len(futures) < worker_count and pending_index < len(pending):
                    task_index, task = pending[pending_index]
                    pending_index += 1
                    futures[executor.submit(fetch_one, task)] = (task_index, task)

            submit_available()
            while futures:
                done, _ = wait(
                    futures,
                    timeout=heartbeat_seconds,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    print_progress(force=True)
                    continue

                for future in done:
                    task_index, task = futures.pop(future)
                    try:
                        record = future.result()
                        records.append(record)
                        status_counts[str(record.get("status_code", "ok"))] += 1
                        completed += 1
                    except Exception as exc:
                        attempts[task_index] += 1
                        if is_retryable_error(exc) and attempts[task_index] < max_task_attempts:
                            transient_failures += 1
                            retry_queue.append((task_index, task))
                        else:
                            permanent_failures += 1
                            print(
                                f"{label}: task failed permanently after {attempts[task_index]} attempt(s): {exc}",
                                flush=True,
                            )
                    now = time.monotonic()
                    if len(records) >= flush_every or (records and now - last_save_at >= 30):
                        cache = append_cache(cache, records, columns, key_columns)
                        save_cache(cache_path, cache)
                        records.clear()
                        last_save_at = now
                        print_progress(force=True)

                submit_available()
                print_progress()

        cache = append_cache(cache, records, columns, key_columns)
        save_cache(cache_path, cache)
        records.clear()
        last_save_at = time.monotonic()
        print_progress(force=True)

        if retry_queue:
            delay = min(max_retry_delay_seconds, retry_delay_seconds * (2 ** (round_index - 1)))
            print(
                f"{label}: retrying {len(retry_queue)} transient failures in {delay:.0f}s "
                f"(retry round {round_index + 1})",
                flush=True,
            )
            time.sleep(delay)
        pending = retry_queue

    print_progress(force=True)
    print(f"{label}: fetched {completed}/{total}", flush=True)
    if permanent_failures:
        print(f"{label}: permanent failures={permanent_failures}", flush=True)
    return cache


def normalize_player_ratings(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=PLAYER_COLUMNS)

    normalized = df.copy()
    for column in PLAYER_COLUMNS:
        if column not in normalized:
            normalized[column] = "" if column not in RATING_COLUMNS else pd.NA
    for column in [column for column in PLAYER_COLUMNS if column not in RATING_COLUMNS]:
        normalized[column] = normalized[column].fillna("").astype(str)
    for column in RATING_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("Int64")
    return normalized[PLAYER_COLUMNS]


def incremental_refresh_player_keys(current: pd.DataFrame, previous: pd.DataFrame | None) -> set[str] | None:
    previous = normalize_player_ratings(previous)
    if previous.empty:
        print("WoW profile pvp.json enrichment: no previous dataset; applying cached enrichment only")
        return set()

    current = normalize_player_ratings(current)
    missing = current[RATING_COLUMNS].isna().any(axis=1)
    missing_keys = set(current.loc[missing, "player_key"].astype(str))
    if not missing_keys:
        return set()

    identity_columns = ["player_key", "class_name", "spec_name"]
    previous_ratings = (
        previous.sort_values(identity_columns, kind="mergesort")
        .drop_duplicates(identity_columns, keep="last")
        .loc[:, identity_columns + RATING_COLUMNS]
    )
    current_ratings = (
        current.sort_values(identity_columns, kind="mergesort")
        .drop_duplicates(identity_columns, keep="last")
        .loc[:, identity_columns + RATING_COLUMNS]
    )
    merged = current_ratings.merge(
        previous_ratings,
        on=identity_columns,
        how="left",
        suffixes=("_current", "_previous"),
        indicator=True,
    )

    changed = merged["_merge"].eq("left_only")
    for column in RATING_COLUMNS:
        current_column = f"{column}_current"
        previous_column = f"{column}_previous"
        current_known = merged[current_column].notna()
        previous_known = merged[previous_column].notna()
        changed |= current_known & (~previous_known | merged[current_column].ne(merged[previous_column]))

    changed_keys = set(merged.loc[changed, "player_key"].astype(str))
    refresh_keys = missing_keys & changed_keys
    print(
        "WoW profile pvp.json enrichment: incremental refresh players="
        f"{len(refresh_keys)} (players with missing ratings={len(missing_keys)}, changed/new={len(changed_keys)})"
    )
    return refresh_keys


def get_web_session() -> requests.Session:
    session = getattr(_thread_local, "wow_profile_session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(WOW_PROFILE_HEADERS)
        _thread_local.wow_profile_session = session
    return session


def wait_for_request_slot(delay_seconds: float, jitter_seconds: float) -> None:
    global _next_request_at
    delay_seconds = max(0.0, float(delay_seconds))
    jitter_seconds = max(0.0, float(jitter_seconds))
    if delay_seconds <= 0 and jitter_seconds <= 0:
        return

    with _request_lock:
        now = time.monotonic()
        wait_seconds = max(0.0, _next_request_at - now)
        spacing = delay_seconds + random.uniform(0.0, jitter_seconds)
        _next_request_at = max(now, _next_request_at) + spacing

    if wait_seconds > 0:
        time.sleep(wait_seconds)


def wow_profile_pvp_url(region: str, realm_slug: str, character_name: str) -> str:
    return (
        f"{WOW_PROFILE_ORIGIN}/en-us/character/"
        f"{region.lower()}/{slugify_realm(realm_slug)}/{normalize_character_name(character_name)}/pvp.json"
    )


def fetch_web_pvp_profile(
    row: pd.Series,
    request_delay_seconds: float,
    request_jitter_seconds: float,
) -> dict[str, Any]:
    region = str(row["region"]).lower()
    realm_slug = slugify_realm(str(row["realm_slug"]))
    character_name = normalize_character_name(str(row["character_name"]))
    wait_for_request_slot(request_delay_seconds, request_jitter_seconds)
    response = get_web_session().get(
        wow_profile_pvp_url(region, realm_slug, character_name),
        timeout=45,
    )
    if response.status_code in WEB_RETRYABLE_STATUS_CODES:
        raise RequestException(f"WoW profile pvp.json returned status {response.status_code}")

    record: dict[str, Any] = {
        "player_key": row["player_key"],
        "region": region,
        "realm_slug": realm_slug,
        "character_name": character_name,
        "status_code": response.status_code,
        "payload_json": "",
        "fetched_at": now_iso(),
    }
    if response.status_code == 200:
        record["payload_json"] = response.text
    return record


def pending_web_profile_players(
    df: pd.DataFrame,
    identities: pd.DataFrame,
    web_cache: pd.DataFrame,
    force: bool,
    refresh_player_keys: set[str] | None = None,
) -> pd.DataFrame:
    missing = df[RATING_COLUMNS].isna().any(axis=1)
    player_keys = set(df.loc[missing, "player_key"].astype(str))
    if refresh_player_keys is not None:
        player_keys &= refresh_player_keys
    if not force and refresh_player_keys is None and not web_cache.empty:
        cached = web_cache[
            web_cache["status_code"].isin([200, 404])
        ]["player_key"].astype(str)
        player_keys -= set(cached)
    return identities[identities["player_key"].isin(player_keys)].reset_index(drop=True)


def carry_forward_previous_ratings(current: pd.DataFrame, previous: pd.DataFrame | None) -> pd.DataFrame:
    previous = normalize_player_ratings(previous)
    current = normalize_player_ratings(current)
    if previous.empty or current.empty:
        return current

    identity_columns = ["player_key", "class_name", "spec_name"]
    previous = previous[previous["player_key"].isin(set(current["player_key"].astype(str)))].copy()
    previous = (
        previous.sort_values(identity_columns, kind="mergesort")
        .drop_duplicates(identity_columns, keep="last")
    )
    current = (
        current.sort_values(identity_columns, kind="mergesort")
        .drop_duplicates(identity_columns, keep="last")
    )

    previous_by_identity = previous.set_index(identity_columns)
    current_by_identity = current.set_index(identity_columns)
    for column in RATING_COLUMNS:
        previous_values = previous_by_identity[column].reindex(current_by_identity.index)
        current_by_identity[column] = current_by_identity[column].where(
            current_by_identity[column].notna(),
            previous_values,
        )

    previous_only = previous_by_identity.loc[
        ~previous_by_identity.index.isin(current_by_identity.index)
    ]
    combined = pd.concat(
        [current_by_identity.reset_index(), previous_only.reset_index()],
        ignore_index=True,
    )
    return combined[PLAYER_COLUMNS]


def web_rating_value(data: dict[str, Any], key: str) -> int | None:
    item = data.get(key)
    if not isinstance(item, dict):
        return None
    rating = item.get("rating")
    if rating is None:
        return None
    return int(rating)


def target_row_index(enriched: pd.DataFrame, player_key: str, shuffle_ratings: dict[tuple[str, str], int]) -> int | None:
    player_rows = enriched[enriched["player_key"].eq(player_key)]
    if player_rows.empty:
        return None

    non_spec_known = player_rows[["blitz_rating", "rating_2v2", "rating_3v3", "rating_rbg"]].notna().any(axis=1)
    if non_spec_known.any():
        return int(player_rows[non_spec_known].index[0])

    best_shuffle: tuple[int, int] | None = None
    for index, row in player_rows.iterrows():
        identity = (str(row["class_name"]), str(row["spec_name"]))
        rating = shuffle_ratings.get(identity)
        if rating is not None and (best_shuffle is None or rating > best_shuffle[0]):
            best_shuffle = (rating, int(index))
    if best_shuffle is not None:
        return best_shuffle[1]

    ratings = player_rows[RATING_COLUMNS].max(axis=1, skipna=True).fillna(-1)
    return int(ratings.sort_values(ascending=False, kind="mergesort").index[0])


def parse_web_shuffle_ratings(payload: dict[str, Any]) -> dict[tuple[str, str], int]:
    ratings = payload.get("ratings") or {}
    shuffle = ratings.get("shuffle") or {}
    specs = shuffle.get("specs") or []
    result: dict[tuple[str, str], int] = {}
    for item in specs:
        if not isinstance(item, dict):
            continue
        specialization = item.get("specialization") or {}
        spec_id = specialization.get("id")
        class_spec = SPEC_ID_TO_INFO.get(spec_id)
        if not class_spec:
            spec_name = specialization.get("name")
            if not spec_name:
                continue
            class_spec = ("", str(spec_name))
        rating = int(item.get("rating") or 0)
        result[(class_spec[0], class_spec[1])] = rating
    return result


def apply_web_pvp_enrichment(df: pd.DataFrame, web_cache: pd.DataFrame) -> pd.DataFrame:
    enriched = normalize_player_ratings(df)
    if web_cache.empty:
        return finalize_enriched_players(enriched)

    identity_by_key = (
        enriched.sort_values("player_key", kind="mergesort")
        .drop_duplicates("player_key")
        .set_index("player_key")[["region", "character_name", "realm", "realm_slug"]]
        .to_dict("index")
    )

    successful = web_cache[web_cache["status_code"].eq(200)].copy()
    successful = successful.drop_duplicates("player_key", keep="last")
    for row in successful.itertuples(index=False):
        player_key_value = str(row.player_key)
        if player_key_value not in identity_by_key:
            continue
        try:
            payload = json.loads(str(row.payload_json or "{}"))
        except json.JSONDecodeError:
            continue

        ratings = payload.get("ratings") or {}
        shuffle_ratings = parse_web_shuffle_ratings(payload)
        for (class_name, spec_name), rating in shuffle_ratings.items():
            if not class_name or not spec_name:
                continue
            mask = (
                enriched["player_key"].eq(player_key_value)
                & enriched["class_name"].eq(class_name)
                & enriched["spec_name"].eq(spec_name)
            )
            if not mask.any() and rating <= 0:
                continue
            enriched = ensure_player_spec_row(
                enriched,
                identity_by_key,
                player_key_value,
                class_name,
                spec_name,
            )
            mask = (
                enriched["player_key"].eq(player_key_value)
                & enriched["class_name"].eq(class_name)
                & enriched["spec_name"].eq(spec_name)
            )
            enriched.loc[mask, "shuffle_rating"] = rating

        non_spec_ratings = {
            "rating_2v2": web_rating_value(ratings, "2v2"),
            "rating_3v3": web_rating_value(ratings, "3v3"),
            "rating_rbg": web_rating_value(ratings, "battlegrounds"),
            "blitz_rating": web_rating_value(ratings, "blitz"),
        }
        non_spec_ratings = {
            column: value for column, value in non_spec_ratings.items() if value is not None
        }
        if non_spec_ratings:
            index = target_row_index(enriched, player_key_value, shuffle_ratings)
            if index is not None:
                for column, value in non_spec_ratings.items():
                    enriched.at[index, column] = value

    return finalize_enriched_players(enriched)


def finalize_enriched_players(df: pd.DataFrame) -> pd.DataFrame:
    enriched = normalize_player_ratings(df)
    for column in RATING_COLUMNS:
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce").astype("Int64")
    for mode in SPEC_MODE_TO_COLUMN:
        rating_column = f"{mode}_rating"
        enriched[f"{mode}_class_name"] = enriched["class_name"].where(enriched[rating_column].notna(), "")
        enriched[f"{mode}_spec_name"] = enriched["spec_name"].where(enriched[rating_column].notna(), "")

    return enriched[PLAYER_COLUMNS].sort_values(
        ["rating_3v3", "shuffle_rating", "blitz_rating", "rating_2v2", "rating_rbg"],
        ascending=False,
        na_position="last",
    )


def ensure_player_spec_row(
    df: pd.DataFrame,
    identity_by_key: dict[str, dict[str, Any]],
    player_key: str,
    class_name: str,
    spec_name: str,
) -> pd.DataFrame:
    mask = (
        df["player_key"].eq(player_key)
        & df["class_name"].eq(class_name)
        & df["spec_name"].eq(spec_name)
    )
    if mask.any():
        return df

    identity = identity_by_key[player_key]
    row = {column: pd.NA for column in PLAYER_COLUMNS}
    row.update(
        {
            "player_key": player_key,
            "region": identity["region"],
            "character_name": identity["character_name"],
            "realm": identity["realm"],
            "realm_slug": identity["realm_slug"],
            "class_name": class_name,
            "spec_name": spec_name,
            "shuffle_class_name": "",
            "shuffle_spec_name": "",
            "blitz_class_name": "",
            "blitz_spec_name": "",
        }
    )
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def enrich_processed_players(
    client: Any,
    data_dir: Path,
    max_workers: int = 16,
    force: bool = False,
    incremental: bool = False,
    previous_players: pd.DataFrame | None = None,
    max_players: int | None = None,
    max_brackets: int | None = None,
    flush_every: int = 1000,
    retry_attempts: int = 25,
    retry_delay_seconds: float = 20.0,
    max_retry_delay_seconds: float = 600.0,
    request_delay_seconds: float = 0.03,
    request_jitter_seconds: float = 0.02,
    write_csv: bool = True,
    write_database: bool = True,
) -> Path:
    del client
    ensure_dirs(data_dir)
    processed_path = data_dir / "processed" / "pvp_players.parquet"
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed dataset: {processed_path}")

    df = pd.read_parquet(processed_path)
    identities = player_identities(df)
    refresh_player_keys = incremental_refresh_player_keys(df, previous_players) if incremental else None
    web_cache_path = data_dir / "raw" / "worldofwarcraft_pvp_profiles.parquet"
    web_cache = load_cache(web_cache_path, WEB_PVP_CACHE_COLUMNS)

    profile_tasks = pending_web_profile_players(df, identities, web_cache, force, refresh_player_keys)
    if max_players is not None:
        profile_tasks = profile_tasks.head(max_players)
    if (force or incremental) and not web_cache.empty and not profile_tasks.empty:
        task_keys = set(profile_tasks["player_key"].astype(str))
        web_cache = web_cache[~web_cache["player_key"].astype(str).isin(task_keys)].copy()
    print(f"WoW profile pvp.json enrichment: pending profiles={len(profile_tasks)}")
    web_cache = run_fetch_pool(
        "WoW profile pvp.json",
        list(profile_tasks.itertuples(index=False)),
        lambda row: fetch_web_pvp_profile(
            pd.Series(row._asdict()),
            request_delay_seconds,
            request_jitter_seconds,
        ),
        max_workers,
        flush_every,
        web_cache,
        web_cache_path,
        WEB_PVP_CACHE_COLUMNS,
        ["player_key"],
        retry_attempts,
        retry_delay_seconds,
        max_retry_delay_seconds,
    )

    if max_brackets is not None:
        print("WoW profile pvp.json enrichment: --max-enrichment-brackets is ignored for pvp.json")

    base = carry_forward_previous_ratings(df, previous_players) if incremental else df
    cache_for_apply = web_cache
    if incremental and refresh_player_keys is not None:
        cache_for_apply = web_cache[web_cache["player_key"].astype(str).isin(refresh_player_keys)].copy()
    enriched = apply_web_pvp_enrichment(base, cache_for_apply)
    enriched.to_parquet(processed_path, index=False)
    csv_path = data_dir / "processed" / "pvp_players.csv"
    if write_csv:
        enriched.to_csv(csv_path, index=False, encoding="utf-8-sig")
    elif csv_path.exists():
        csv_path.unlink()
    if write_database:
        write_players_to_database(enriched)
    print(f"Saved enriched PvP dataset: {processed_path}")
    return processed_path
