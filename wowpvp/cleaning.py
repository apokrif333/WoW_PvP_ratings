from __future__ import annotations

import pandas as pd

from wowpvp.utils import player_key


CHECKPVP_RATING_SORT_COLUMNS = ["rateatm3v3", "rateatm2v2", "rateatmrbg"]
CHECKPVP_KEY_COLUMNS = {"region", "realm", "name"}


def deduplicate_checkpvp_players(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not CHECKPVP_KEY_COLUMNS.issubset(df.columns):
        return df.copy()

    clean = df.copy()
    clean["_player_key"] = clean.apply(
        lambda row: player_key(str(row["region"]), str(row["realm"]), str(row["name"])),
        axis=1,
    )

    sort_columns = ["_player_key"]
    ascending = [True]
    for column in CHECKPVP_RATING_SORT_COLUMNS:
        if column in clean.columns:
            clean[column] = pd.to_numeric(clean[column], errors="coerce").fillna(0)
            sort_columns.append(column)
            ascending.append(False)

    clean = (
        clean.sort_values(sort_columns, ascending=ascending, kind="mergesort")
        .drop_duplicates("_player_key", keep="first")
        .drop(columns=["_player_key"])
        .reset_index(drop=True)
    )
    return clean
