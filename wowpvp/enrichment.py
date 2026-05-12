from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from wowpvp.blizzard import BlizzardClient, parse_spec_leaderboard_slug
from wowpvp.constants import CLASS_SLUG_TO_NAME, SPEC_ID_TO_INFO, SPEC_SLUG_TO_NAME
from wowpvp.storage import PLAYER_COLUMNS, write_players_to_database
from wowpvp.utils import ensure_dirs, normalize_character_name, player_key, slugify_realm


SUMMARY_CACHE_COLUMNS = [
    "player_key",
    "region",
    "realm_slug",
    "character_name",
    "status_code",
    "brackets_json",
    "fetched_at",
]
BRACKET_CACHE_COLUMNS = [
    "player_key",
    "region",
    "realm_slug",
    "character_name",
    "bracket",
    "status_code",
    "rating",
    "class_name",
    "spec_name",
    "season_id",
    "played",
    "won",
    "lost",
    "fetched_at",
]
CHARACTER_CACHE_COLUMNS = [
    "player_key",
    "region",
    "realm_slug",
    "character_name",
    "status_code",
    "active_class_name",
    "active_spec_name",
    "fetched_at",
]
RATING_COLUMNS = ["shuffle_rating", "blitz_rating", "rating_2v2", "rating_3v3", "rating_rbg"]
NON_SPEC_BRACKET_TO_COLUMN = {
    "2v2": "rating_2v2",
    "3v3": "rating_3v3",
    "rbg": "rating_rbg",
}
SPEC_MODE_TO_COLUMN = {
    "shuffle": "shuffle_rating",
    "blitz": "blitz_rating",
}
CLASS_NAME_TO_SLUG = {value: key for key, value in CLASS_SLUG_TO_NAME.items()}
SPEC_NAME_TO_SLUG = {value: key for key, value in SPEC_SLUG_TO_NAME.items()}


def class_spec_bracket(mode: str, class_name: str, spec_name: str) -> str | None:
    class_slug = CLASS_NAME_TO_SLUG.get(str(class_name or ""))
    spec_slug = SPEC_NAME_TO_SLUG.get(str(spec_name or ""))
    if not class_slug or not spec_slug:
        return None
    return f"{mode}-{class_slug}-{spec_slug}"


def bracket_from_href(href: str) -> str:
    return Path(urlparse(href).path).name


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


def fetch_pvp_summary(client: BlizzardClient, row: pd.Series) -> dict[str, Any]:
    region = str(row["region"])
    realm_slug = str(row["realm_slug"])
    character_name = str(row["character_name"])
    response = client.get_profile_response(region, f"character/{realm_slug}/{character_name}/pvp-summary")
    record: dict[str, Any] = {
        "player_key": row["player_key"],
        "region": region,
        "realm_slug": realm_slug,
        "character_name": character_name,
        "status_code": response.status_code,
        "brackets_json": "[]",
        "fetched_at": now_iso(),
    }
    if response.status_code == 200:
        data = response.json()
        brackets = [bracket_from_href(item.get("href", "")) for item in data.get("brackets", [])]
        record["brackets_json"] = json.dumps(sorted(bracket for bracket in brackets if bracket))
    return record


def fetch_character_profile(client: BlizzardClient, row: pd.Series) -> dict[str, Any]:
    region = str(row["region"])
    realm_slug = str(row["realm_slug"])
    character_name = str(row["character_name"])
    response = client.get_profile_response(region, f"character/{realm_slug}/{character_name}")
    record: dict[str, Any] = {
        "player_key": row["player_key"],
        "region": region,
        "realm_slug": realm_slug,
        "character_name": character_name,
        "status_code": response.status_code,
        "active_class_name": "",
        "active_spec_name": "",
        "fetched_at": now_iso(),
    }
    if response.status_code == 200:
        data = response.json()
        active_spec = data.get("active_spec") or {}
        spec_info = SPEC_ID_TO_INFO.get(active_spec.get("id"))
        if spec_info:
            record["active_class_name"], record["active_spec_name"] = spec_info
        else:
            record["active_class_name"] = (data.get("character_class") or {}).get("name", "")
            record["active_spec_name"] = active_spec.get("name", "")
    return record


def fetch_pvp_bracket(client: BlizzardClient, row: pd.Series, bracket: str) -> dict[str, Any]:
    region = str(row["region"])
    realm_slug = str(row["realm_slug"])
    character_name = str(row["character_name"])
    response = client.get_profile_response(
        region,
        f"character/{realm_slug}/{character_name}/pvp-bracket/{bracket}",
    )
    record: dict[str, Any] = {
        "player_key": row["player_key"],
        "region": region,
        "realm_slug": realm_slug,
        "character_name": character_name,
        "bracket": bracket,
        "status_code": response.status_code,
        "rating": pd.NA,
        "class_name": "",
        "spec_name": "",
        "season_id": pd.NA,
        "played": pd.NA,
        "won": pd.NA,
        "lost": pd.NA,
        "fetched_at": now_iso(),
    }
    if response.status_code == 200:
        data = response.json()
        record["rating"] = data.get("rating")
        record["season_id"] = (data.get("season") or {}).get("id")
        stats = data.get("season_match_statistics") or {}
        record["played"] = stats.get("played")
        record["won"] = stats.get("won")
        record["lost"] = stats.get("lost")
        if bracket.startswith(("shuffle-", "blitz-")):
            _, class_name, spec_name = parse_spec_leaderboard_slug(bracket)
            record["class_name"] = class_name
            record["spec_name"] = spec_name
    return record


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
) -> pd.DataFrame:
    if not tasks:
        return cache

    records: list[dict[str, Any]] = []
    worker_count = max(1, min(max_workers, len(tasks)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(fetch_one, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                records.append(future.result())
            except Exception as exc:
                print(f"{label}: task failed: {exc}")
            if len(records) >= flush_every:
                cache = append_cache(cache, records, columns, key_columns)
                save_cache(cache_path, cache)
                records.clear()
                print(f"{label}: fetched {index}/{len(tasks)}")

    cache = append_cache(cache, records, columns, key_columns)
    save_cache(cache_path, cache)
    print(f"{label}: fetched {len(tasks)}/{len(tasks)}")
    return cache


def summary_brackets(summary_cache: pd.DataFrame) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for row in summary_cache.itertuples(index=False):
        status_code = getattr(row, "status_code")
        if pd.isna(status_code) or int(status_code) != 200:
            continue
        try:
            brackets = set(json.loads(getattr(row, "brackets_json") or "[]"))
        except json.JSONDecodeError:
            brackets = set()
        mapping[getattr(row, "player_key")] = brackets
    return mapping


def pending_summary_players(
    df: pd.DataFrame,
    identities: pd.DataFrame,
    summary_cache: pd.DataFrame,
    force: bool,
) -> pd.DataFrame:
    missing = df[RATING_COLUMNS].isna().any(axis=1)
    player_keys = set(df.loc[missing, "player_key"])
    if not force:
        cached_keys = set(summary_cache["player_key"].astype(str))
        player_keys -= cached_keys
    return identities[identities["player_key"].isin(player_keys)].reset_index(drop=True)


def desired_bracket_tasks(
    df: pd.DataFrame,
    identities: pd.DataFrame,
    summary_cache: pd.DataFrame,
    bracket_cache: pd.DataFrame,
    force: bool,
) -> list[tuple[pd.Series, str]]:
    brackets_by_player = summary_brackets(summary_cache)
    identity_by_key = identities.set_index("player_key")
    cached = set()
    if not force and not bracket_cache.empty:
        cached = set(map(tuple, bracket_cache[["player_key", "bracket"]].astype(str).itertuples(index=False, name=None)))

    wanted: set[tuple[str, str]] = set()
    for row in df.itertuples(index=False):
        player_brackets = brackets_by_player.get(row.player_key)
        if not player_brackets:
            continue

        for mode, rating_column in SPEC_MODE_TO_COLUMN.items():
            if pd.isna(getattr(row, rating_column)):
                bracket = class_spec_bracket(mode, row.class_name, row.spec_name)
                if bracket and bracket in player_brackets:
                    wanted.add((row.player_key, bracket))

        for bracket in NON_SPEC_BRACKET_TO_COLUMN:
            if pd.isna(getattr(row, NON_SPEC_BRACKET_TO_COLUMN[bracket])) and bracket in player_brackets:
                wanted.add((row.player_key, bracket))

        for bracket in player_brackets:
            if bracket.startswith(("shuffle-", "blitz-")):
                wanted.add((row.player_key, bracket))

    tasks: list[tuple[pd.Series, str]] = []
    for player_key, bracket in sorted(wanted):
        if (player_key, bracket) in cached:
            continue
        if player_key in identity_by_key.index:
            identity = identity_by_key.loc[player_key].copy()
            identity["player_key"] = player_key
            tasks.append((identity, bracket))
    return tasks


def pending_character_players(
    identities: pd.DataFrame,
    bracket_cache: pd.DataFrame,
    character_cache: pd.DataFrame,
    force: bool,
) -> pd.DataFrame:
    non_spec_success = bracket_cache[
        bracket_cache["bracket"].isin(NON_SPEC_BRACKET_TO_COLUMN)
        & bracket_cache["status_code"].eq(200)
        & bracket_cache["rating"].notna()
    ]
    player_keys = set(non_spec_success["player_key"].astype(str))
    if not force:
        player_keys -= set(character_cache["player_key"].astype(str))
    return identities[identities["player_key"].isin(player_keys)].reset_index(drop=True)


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


def apply_enrichment(
    df: pd.DataFrame,
    summary_cache: pd.DataFrame,
    bracket_cache: pd.DataFrame,
    character_cache: pd.DataFrame,
) -> pd.DataFrame:
    enriched = df.copy()
    for column in RATING_COLUMNS:
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce").astype("Int64")

    identity_by_key = (
        enriched.sort_values("player_key", kind="mergesort")
        .drop_duplicates("player_key")
        .set_index("player_key")[["region", "character_name", "realm", "realm_slug"]]
        .to_dict("index")
    )
    brackets_by_player = summary_brackets(summary_cache)

    for row in enriched.itertuples(index=True):
        player_brackets = brackets_by_player.get(row.player_key)
        if player_brackets is None:
            continue
        for mode, rating_column in SPEC_MODE_TO_COLUMN.items():
            bracket = class_spec_bracket(mode, row.class_name, row.spec_name)
            if pd.isna(getattr(row, rating_column)) and bracket and bracket not in player_brackets:
                enriched.at[row.Index, rating_column] = 0

    successful_brackets = bracket_cache[
        bracket_cache["status_code"].eq(200) & bracket_cache["rating"].notna()
    ].copy()
    for row in successful_brackets.itertuples(index=False):
        player_key = str(row.player_key)
        bracket = str(row.bracket)
        if bracket.startswith(("shuffle-", "blitz-")):
            mode, class_name, spec_name = parse_spec_leaderboard_slug(bracket)
            enriched = ensure_player_spec_row(enriched, identity_by_key, player_key, class_name, spec_name)
            mask = (
                enriched["player_key"].eq(player_key)
                & enriched["class_name"].eq(class_name)
                & enriched["spec_name"].eq(spec_name)
            )
            rating_column = SPEC_MODE_TO_COLUMN[mode]
            enriched.loc[mask, rating_column] = int(row.rating)
            enriched.loc[mask, f"{mode}_class_name"] = class_name
            enriched.loc[mask, f"{mode}_spec_name"] = spec_name

    active_specs = character_cache[
        character_cache["status_code"].eq(200)
        & character_cache["active_class_name"].astype(str).ne("")
        & character_cache["active_spec_name"].astype(str).ne("")
    ].set_index("player_key")
    for row in successful_brackets.itertuples(index=False):
        player_key = str(row.player_key)
        bracket = str(row.bracket)
        rating_column = NON_SPEC_BRACKET_TO_COLUMN.get(bracket)
        if not rating_column or player_key not in active_specs.index:
            continue
        active = active_specs.loc[player_key]
        class_name = str(active["active_class_name"])
        spec_name = str(active["active_spec_name"])
        enriched = ensure_player_spec_row(enriched, identity_by_key, player_key, class_name, spec_name)
        mask = (
            enriched["player_key"].eq(player_key)
            & enriched["class_name"].eq(class_name)
            & enriched["spec_name"].eq(spec_name)
        )
        enriched.loc[mask, rating_column] = int(row.rating)

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


def enrich_processed_players(
    client: BlizzardClient,
    data_dir: Path,
    max_workers: int = 16,
    force: bool = False,
    max_players: int | None = None,
    max_brackets: int | None = None,
    flush_every: int = 1000,
    write_csv: bool = True,
    write_database: bool = True,
) -> Path:
    ensure_dirs(data_dir)
    processed_path = data_dir / "processed" / "pvp_players.parquet"
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed dataset: {processed_path}")

    df = pd.read_parquet(processed_path)
    identities = player_identities(df)
    summary_path = data_dir / "raw" / "blizzard_profile_pvp_summaries.parquet"
    bracket_path = data_dir / "raw" / "blizzard_profile_pvp_brackets.parquet"
    character_path = data_dir / "raw" / "blizzard_profile_characters.parquet"
    summary_cache = load_cache(summary_path, SUMMARY_CACHE_COLUMNS)
    bracket_cache = load_cache(bracket_path, BRACKET_CACHE_COLUMNS)
    character_cache = load_cache(character_path, CHARACTER_CACHE_COLUMNS)

    summary_tasks = pending_summary_players(df, identities, summary_cache, force)
    if max_players is not None:
        summary_tasks = summary_tasks.head(max_players)
    print(f"Blizzard profile enrichment: pending PvP summaries={len(summary_tasks)}")
    summary_cache = run_fetch_pool(
        "Blizzard profile PvP summaries",
        list(summary_tasks.itertuples(index=False)),
        lambda row: fetch_pvp_summary(client, pd.Series(row._asdict())),
        max_workers,
        flush_every,
        summary_cache,
        summary_path,
        SUMMARY_CACHE_COLUMNS,
        ["player_key"],
    )

    bracket_tasks = desired_bracket_tasks(df, identities, summary_cache, bracket_cache, force)
    if max_brackets is not None:
        bracket_tasks = bracket_tasks[:max_brackets]
    print(f"Blizzard profile enrichment: pending PvP brackets={len(bracket_tasks)}")
    bracket_cache = run_fetch_pool(
        "Blizzard profile PvP brackets",
        bracket_tasks,
        lambda task: fetch_pvp_bracket(client, task[0], task[1]),
        max_workers,
        flush_every,
        bracket_cache,
        bracket_path,
        BRACKET_CACHE_COLUMNS,
        ["player_key", "bracket"],
    )

    character_tasks = pending_character_players(identities, bracket_cache, character_cache, force)
    if max_players is not None:
        character_tasks = character_tasks.head(max_players)
    print(f"Blizzard profile enrichment: pending character profiles={len(character_tasks)}")
    character_cache = run_fetch_pool(
        "Blizzard character profiles",
        list(character_tasks.itertuples(index=False)),
        lambda row: fetch_character_profile(client, pd.Series(row._asdict())),
        max_workers,
        flush_every,
        character_cache,
        character_path,
        CHARACTER_CACHE_COLUMNS,
        ["player_key"],
    )

    enriched = apply_enrichment(df, summary_cache, bracket_cache, character_cache)
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
