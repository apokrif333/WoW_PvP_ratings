from __future__ import annotations

from pathlib import Path

import pandas as pd

from wowpvp.cleaning import deduplicate_checkpvp_players
from wowpvp.constants import CLASS_ID_TO_NAME, SPEC_ID_TO_INFO
from wowpvp.storage import INTEGER_COLUMNS, PLAYER_COLUMNS, TEXT_COLUMNS, write_players_to_database
from wowpvp.utils import ensure_dirs, player_key, slugify_realm


FINAL_COLUMNS = PLAYER_COLUMNS
BLIZZARD_MODE_PREFIXES = {"shuffle": "shuffle", "blitz": "blitz"}
BLIZZARD_GLOBAL_RATING_COLUMNS = {
    "2v2": "rating_2v2",
    "3v3": "rating_3v3",
    "rbg": "rating_rbg",
}
BLIZZARD_IDENTITY_COLUMNS = [
    "player_key",
    "region",
    "character_name",
    "realm",
    "realm_slug",
    "class_name",
    "spec_name",
]
BLIZZARD_GLOBAL_IDENTITY_COLUMNS = [
    "player_key",
    "region",
    "character_name",
    "realm",
    "realm_slug",
]
BLIZZARD_PROFILE_COLUMNS = [
    *BLIZZARD_GLOBAL_IDENTITY_COLUMNS,
    "class_name",
    "spec_name",
]
BLIZZARD_GLOBAL_COLUMNS = [
    *BLIZZARD_PROFILE_COLUMNS,
    *BLIZZARD_GLOBAL_RATING_COLUMNS.values(),
]
BLIZZARD_COLUMNS = [
    *BLIZZARD_IDENTITY_COLUMNS,
    "shuffle_rating",
    "blitz_rating",
    "shuffle_class_name",
    "shuffle_spec_name",
    "blitz_class_name",
    "blitz_spec_name",
]
CHECKPVP_COLUMNS = [
    *BLIZZARD_IDENTITY_COLUMNS,
    "rating_2v2",
    "rating_3v3",
    "rating_rbg",
]


def load_raw_blizzard(data_dir: Path) -> pd.DataFrame:
    paths = sorted((data_dir / "raw").glob("blizzard_*_season_*.parquet"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def load_raw_checkpvp(data_dir: Path) -> pd.DataFrame:
    paths = sorted((data_dir / "raw").glob("checkpvp_*.parquet"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def load_raw_blizzard_profile_specs(data_dir: Path) -> pd.DataFrame:
    paths = sorted((data_dir / "raw").glob("blizzard_profile_specs_*.parquet"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def prepare_blizzard(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=BLIZZARD_COLUMNS)

    df = df.copy()
    df = df[df["mode"].isin(BLIZZARD_MODE_PREFIXES)].copy()
    if df.empty:
        return pd.DataFrame(columns=BLIZZARD_COLUMNS)

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0).astype(int)
    df = (
        df.sort_values(
            ["player_key", "mode", "class_name", "spec_name", "rating"],
            ascending=[True, True, True, True, False],
            kind="mergesort",
        )
        .drop_duplicates(["player_key", "mode", "class_name", "spec_name"], keep="first")
        .reset_index(drop=True)
    )

    ratings = (
        df.pivot_table(
            index=BLIZZARD_IDENTITY_COLUMNS,
            columns="mode",
            values="rating",
            aggfunc="max",
        )
        .rename(columns={"shuffle": "shuffle_rating", "blitz": "blitz_rating"})
        .reset_index()
    )

    result = ratings
    for column in ["shuffle_rating", "blitz_rating"]:
        if column not in result:
            result[column] = pd.NA

    for mode, prefix in BLIZZARD_MODE_PREFIXES.items():
        rating_column = f"{prefix}_rating"
        has_mode_rating = result[rating_column].notna()
        result[f"{prefix}_class_name"] = result["class_name"].where(has_mode_rating, "")
        result[f"{prefix}_spec_name"] = result["spec_name"].where(has_mode_rating, "")

    return result[BLIZZARD_COLUMNS]


def prepare_blizzard_profile_specs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=BLIZZARD_PROFILE_COLUMNS)

    result = df.copy()
    for column in BLIZZARD_PROFILE_COLUMNS:
        if column not in result:
            result[column] = ""
        result[column] = result[column].fillna("").astype(str)
    result = result[BLIZZARD_PROFILE_COLUMNS]
    result["_has_spec"] = result["class_name"].ne("") & result["spec_name"].ne("")
    result = (
        result.sort_values(["player_key", "_has_spec"], ascending=[True, False], kind="mergesort")
        .drop_duplicates("player_key", keep="first")
        .drop(columns="_has_spec")
        .reset_index(drop=True)
    )
    return result


def prepare_blizzard_global(
    df: pd.DataFrame,
    profile_specs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    columns = BLIZZARD_GLOBAL_COLUMNS
    if df.empty:
        return pd.DataFrame(columns=columns)

    df = df.copy()
    df = df[df["mode"].isin(BLIZZARD_GLOBAL_RATING_COLUMNS)].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("Int64")
    df = (
        df.sort_values(
            ["player_key", "mode", "rating"],
            ascending=[True, True, False],
            kind="mergesort",
        )
        .drop_duplicates(["player_key", "mode"], keep="first")
        .reset_index(drop=True)
    )
    ratings = (
        df.pivot_table(
            index=BLIZZARD_GLOBAL_IDENTITY_COLUMNS,
            columns="mode",
            values="rating",
            aggfunc="max",
        )
        .rename(columns=BLIZZARD_GLOBAL_RATING_COLUMNS)
        .reset_index()
    )
    profiles = prepare_blizzard_profile_specs(
        profile_specs if profile_specs is not None else pd.DataFrame()
    )
    if profiles.empty:
        ratings["class_name"] = ""
        ratings["spec_name"] = ""
    else:
        ratings = ratings.merge(
            profiles[["player_key", "class_name", "spec_name"]],
            on="player_key",
            how="left",
        )
        ratings["class_name"] = ratings["class_name"].fillna("")
        ratings["spec_name"] = ratings["spec_name"].fillna("")

    for column in BLIZZARD_GLOBAL_RATING_COLUMNS.values():
        if column not in ratings:
            ratings[column] = pd.NA
        ratings[column] = pd.to_numeric(ratings[column], errors="coerce").astype("Int64")
    return ratings[columns]


def prepare_checkpvp(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=CHECKPVP_COLUMNS)

    result = deduplicate_checkpvp_players(df)
    result["region"] = result["region"].astype(str).str.lower()
    result["realm_slug"] = result["realm"].map(slugify_realm)
    result["player_key"] = result.apply(
        lambda row: player_key(row["region"], row["realm_slug"], row["name"]),
        axis=1,
    )
    result["character_name"] = result["name"]
    result["class_name"] = result["class"].map(CLASS_ID_TO_NAME).fillna("")
    spec_info = result["activeSpecId"].map(SPEC_ID_TO_INFO)
    result["spec_name"] = spec_info.map(lambda item: item[1] if isinstance(item, tuple) else "")
    result.loc[result["class_name"].eq("") & spec_info.notna(), "class_name"] = spec_info.map(
        lambda item: item[0] if isinstance(item, tuple) else ""
    )

    result = result.rename(
        columns={
            "rateatm2v2": "rating_2v2",
            "rateatm3v3": "rating_3v3",
            "rateatmrbg": "rating_rbg",
        }
    )
    for column in ["rating_2v2", "rating_3v3", "rating_rbg"]:
        result[column] = result.get(column, 0).fillna(0).astype(int)

    result = (
        result.sort_values(
            ["player_key", "rating_3v3", "rating_2v2", "rating_rbg"],
            ascending=[True, False, False, False],
            kind="mergesort",
        )
        .drop_duplicates("player_key")
        .loc[
            :,
            [
                "player_key",
                "region",
                "character_name",
                "realm",
                "realm_slug",
                "class_name",
                "spec_name",
                "rating_2v2",
                "rating_3v3",
                "rating_rbg",
            ],
        ]
    )
    return result


def first_non_empty(left: pd.Series, right: pd.Series) -> pd.Series:
    left = left.fillna("")
    right = right.fillna("")
    return left.where(left.astype(str).ne(""), right)


def optional_series(df: pd.DataFrame, column: str, default: str | int = "") -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def normalize_final_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in FINAL_COLUMNS:
        if column not in normalized:
            normalized[column] = "" if column in TEXT_COLUMNS else pd.NA
    normalized = normalized[FINAL_COLUMNS]
    for column in TEXT_COLUMNS:
        normalized[column] = normalized[column].fillna("").astype(str)
    for column in INTEGER_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("Int64")
    return normalized


def choose_global_rating_target(
    final: pd.DataFrame,
    indices: list[int],
    class_name: str = "",
    spec_name: str = "",
) -> int:
    group = final.loc[indices]
    if class_name and spec_name:
        matches_profile_spec = group["class_name"].eq(class_name) & group["spec_name"].eq(spec_name)
        if matches_profile_spec.any():
            return int(matches_profile_spec[matches_profile_spec].index[0])

    checkpvp_columns = list(BLIZZARD_GLOBAL_RATING_COLUMNS.values())
    has_checkpvp_rating = group[checkpvp_columns].notna().any(axis=1)
    if has_checkpvp_rating.any():
        return int(has_checkpvp_rating[has_checkpvp_rating].index[0])

    mode_scores = (
        group[["shuffle_rating", "blitz_rating"]]
        .apply(lambda column: pd.to_numeric(column, errors="coerce"))
        .max(axis=1)
        .fillna(-1)
    )
    has_spec = group["class_name"].astype(str).ne("") & group["spec_name"].astype(str).ne("")
    candidates = pd.DataFrame(
        {
            "has_spec": has_spec.astype(int),
            "mode_score": mode_scores,
        },
        index=group.index,
    )
    return int(
        candidates.sort_values(
            ["has_spec", "mode_score"],
            ascending=[False, False],
            kind="mergesort",
        ).index[0]
    )


def apply_blizzard_global_ratings(final: pd.DataFrame, blizzard_global: pd.DataFrame) -> pd.DataFrame:
    if blizzard_global.empty:
        return final

    final = final.copy()
    global_ratings = blizzard_global.copy()
    for column in BLIZZARD_GLOBAL_RATING_COLUMNS.values():
        global_ratings[column] = pd.to_numeric(global_ratings[column], errors="coerce").astype("Int64")

    indices_by_player = {
        str(player_key_value): list(indices)
        for player_key_value, indices in final.groupby("player_key", sort=False).indices.items()
    }
    new_rows: list[dict[str, object]] = []

    for row in global_ratings.itertuples(index=False):
        row_data = row._asdict()
        key = str(row_data["player_key"])
        indices = indices_by_player.get(key)
        if indices:
            target_index = choose_global_rating_target(
                final,
                indices,
                str(row_data.get("class_name") or ""),
                str(row_data.get("spec_name") or ""),
            )
            for column in BLIZZARD_GLOBAL_RATING_COLUMNS.values():
                value = row_data.get(column)
                if pd.notna(value):
                    final.at[target_index, column] = int(value)
            continue

        new_row: dict[str, object] = {column: pd.NA for column in FINAL_COLUMNS}
        for column in BLIZZARD_GLOBAL_IDENTITY_COLUMNS:
            new_row[column] = row_data.get(column, "")
        new_row["class_name"] = row_data.get("class_name", "") or ""
        new_row["spec_name"] = row_data.get("spec_name", "") or ""
        new_row["shuffle_class_name"] = ""
        new_row["shuffle_spec_name"] = ""
        new_row["blitz_class_name"] = ""
        new_row["blitz_spec_name"] = ""
        for column in BLIZZARD_GLOBAL_RATING_COLUMNS.values():
            value = row_data.get(column)
            if pd.notna(value):
                new_row[column] = int(value)
        new_rows.append(new_row)

    if new_rows:
        final = pd.concat([final, pd.DataFrame(new_rows)], ignore_index=True)
    return final


def blizzard_global_profile_candidates(
    data_dir: Path,
    regions: list[str] | None = None,
) -> pd.DataFrame:
    raw_blizzard = load_raw_blizzard(data_dir)
    if raw_blizzard.empty or "mode" not in raw_blizzard:
        return pd.DataFrame(columns=BLIZZARD_GLOBAL_IDENTITY_COLUMNS)

    raw_blizzard = raw_blizzard.copy()
    raw_blizzard["mode"] = raw_blizzard["mode"].astype(str)
    global_players = raw_blizzard[raw_blizzard["mode"].isin(BLIZZARD_GLOBAL_RATING_COLUMNS)].copy()
    if regions:
        allowed_regions = {region.lower() for region in regions}
        global_players = global_players[global_players["region"].astype(str).str.lower().isin(allowed_regions)]
    if global_players.empty:
        return pd.DataFrame(columns=BLIZZARD_GLOBAL_IDENTITY_COLUMNS)

    known_keys: set[str] = set()
    spec_rows = raw_blizzard[raw_blizzard["mode"].isin(BLIZZARD_MODE_PREFIXES)]
    if {"class_name", "spec_name"}.issubset(spec_rows.columns):
        has_spec = spec_rows["class_name"].fillna("").ne("") & spec_rows["spec_name"].fillna("").ne("")
        known_keys.update(spec_rows.loc[has_spec, "player_key"].astype(str))

    checkpvp = prepare_checkpvp(load_raw_checkpvp(data_dir))
    if not checkpvp.empty:
        has_spec = checkpvp["class_name"].fillna("").ne("") & checkpvp["spec_name"].fillna("").ne("")
        known_keys.update(checkpvp.loc[has_spec, "player_key"].astype(str))

    result = (
        global_players[~global_players["player_key"].astype(str).isin(known_keys)]
        .sort_values(["region", "player_key"], kind="mergesort")
        .drop_duplicates("player_key", keep="first")
    )
    return result[BLIZZARD_GLOBAL_IDENTITY_COLUMNS].reset_index(drop=True)


def build_final_dataset(
    data_dir: Path,
    write_csv: bool = True,
    write_database: bool = True,
) -> Path:
    ensure_dirs(data_dir)
    raw_blizzard = load_raw_blizzard(data_dir)
    blizzard = prepare_blizzard(raw_blizzard)
    profile_specs = prepare_blizzard_profile_specs(load_raw_blizzard_profile_specs(data_dir))
    blizzard_global = prepare_blizzard_global(raw_blizzard, profile_specs)
    checkpvp = prepare_checkpvp(load_raw_checkpvp(data_dir))

    merge_keys = ["player_key", "class_name", "spec_name"]
    merged = checkpvp.merge(
        blizzard,
        on=merge_keys,
        how="outer",
        suffixes=("_checkpvp", "_blizzard"),
    )

    final = pd.DataFrame()
    final["player_key"] = merged["player_key"]
    final["class_name"] = merged["class_name"].fillna("")
    final["spec_name"] = merged["spec_name"].fillna("")
    for column in ["region", "character_name", "realm", "realm_slug"]:
        final[column] = first_non_empty(
            optional_series(merged, f"{column}_checkpvp"),
            optional_series(merged, f"{column}_blizzard"),
        )

    for column in ["shuffle_rating", "blitz_rating", "rating_2v2", "rating_3v3", "rating_rbg"]:
        final[column] = pd.to_numeric(optional_series(merged, column, pd.NA), errors="coerce").astype("Int64")
    for prefix in BLIZZARD_MODE_PREFIXES.values():
        final[f"{prefix}_class_name"] = optional_series(merged, f"{prefix}_class_name").fillna("")
        final[f"{prefix}_spec_name"] = optional_series(merged, f"{prefix}_spec_name").fillna("")

    final = apply_blizzard_global_ratings(final, blizzard_global)
    final = normalize_final_dataframe(final).sort_values(
        ["rating_3v3", "shuffle_rating", "blitz_rating", "rating_2v2", "rating_rbg"],
        ascending=False,
    )

    output_path = data_dir / "processed" / "pvp_players.parquet"
    csv_path = data_dir / "processed" / "pvp_players.csv"
    final.to_parquet(output_path, index=False)
    if write_csv:
        final.to_csv(csv_path, index=False, encoding="utf-8-sig")
    elif csv_path.exists():
        csv_path.unlink()
    if write_database:
        write_players_to_database(final)
    return output_path
