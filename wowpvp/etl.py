from __future__ import annotations

from pathlib import Path

import pandas as pd

from wowpvp.constants import CLASS_ID_TO_NAME, SPEC_ID_TO_INFO
from wowpvp.storage import PLAYER_COLUMNS, write_players_to_database
from wowpvp.utils import ensure_dirs, player_key, slugify_realm


FINAL_COLUMNS = PLAYER_COLUMNS
BLIZZARD_MODE_PREFIXES = {"shuffle": "shuffle", "blitz": "blitz"}


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


def prepare_blizzard(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=FINAL_COLUMNS)

    ratings = (
        df.pivot_table(
            index="player_key",
            columns="mode",
            values="rating",
            aggfunc="max",
            fill_value=0,
        )
        .rename(columns={"shuffle": "shuffle_rating", "blitz": "blitz_rating"})
        .reset_index()
    )

    identities = (
        df.sort_values(["player_key", "rating"], ascending=[True, False])
        .drop_duplicates("player_key")
        .loc[:, ["player_key", "region", "character_name", "realm", "realm_slug", "class_name", "spec_name"]]
    )

    result = identities.merge(ratings, on="player_key", how="left")
    for column in ["shuffle_rating", "blitz_rating"]:
        if column not in result:
            result[column] = 0

    mode_identities = (
        df.sort_values(["player_key", "mode", "rating"], ascending=[True, True, False])
        .drop_duplicates(["player_key", "mode"])
        .loc[:, ["player_key", "mode", "class_name", "spec_name"]]
    )
    for mode, prefix in BLIZZARD_MODE_PREFIXES.items():
        mode_columns = (
            mode_identities[mode_identities["mode"].eq(mode)]
            .drop(columns=["mode"])
            .rename(
                columns={
                    "class_name": f"{prefix}_class_name",
                    "spec_name": f"{prefix}_spec_name",
                }
            )
        )
        result = result.merge(mode_columns, on="player_key", how="left")
        result[f"{prefix}_class_name"] = result[f"{prefix}_class_name"].fillna("")
        result[f"{prefix}_spec_name"] = result[f"{prefix}_spec_name"].fillna("")

    return result


def prepare_checkpvp(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=FINAL_COLUMNS)

    result = df.copy()
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
        result.sort_values(["player_key", "rating_3v3", "rating_2v2", "rating_rbg"], ascending=False)
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


def build_final_dataset(
    data_dir: Path,
    write_csv: bool = True,
    write_database: bool = True,
) -> Path:
    ensure_dirs(data_dir)
    blizzard = prepare_blizzard(load_raw_blizzard(data_dir))
    checkpvp = prepare_checkpvp(load_raw_checkpvp(data_dir))

    merged = checkpvp.merge(
        blizzard,
        on="player_key",
        how="outer",
        suffixes=("_checkpvp", "_blizzard"),
    )

    final = pd.DataFrame()
    final["player_key"] = merged["player_key"]
    for column in ["region", "character_name", "realm", "realm_slug", "class_name", "spec_name"]:
        final[column] = first_non_empty(
            optional_series(merged, f"{column}_checkpvp"),
            optional_series(merged, f"{column}_blizzard"),
        )

    for column in ["shuffle_rating", "blitz_rating"]:
        final[column] = optional_series(merged, column, 0).fillna(0).astype(int)
    for column in ["rating_2v2", "rating_3v3", "rating_rbg"]:
        final[column] = optional_series(merged, column, 0).fillna(0).astype(int)
    for prefix in BLIZZARD_MODE_PREFIXES.values():
        final[f"{prefix}_class_name"] = first_non_empty(
            optional_series(merged, f"{prefix}_class_name"),
            final["class_name"],
        )
        final[f"{prefix}_spec_name"] = first_non_empty(
            optional_series(merged, f"{prefix}_spec_name"),
            final["spec_name"],
        )

    final = final[FINAL_COLUMNS].sort_values(
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
