from __future__ import annotations

import io
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


PLAYER_COLUMNS = [
    "player_key",
    "region",
    "character_name",
    "realm",
    "realm_slug",
    "class_name",
    "spec_name",
    "shuffle_rating",
    "blitz_rating",
    "rating_2v2",
    "rating_3v3",
    "rating_rbg",
]

TEXT_COLUMNS = {
    "player_key",
    "region",
    "character_name",
    "realm",
    "realm_slug",
    "class_name",
    "spec_name",
}
INTEGER_COLUMNS = set(PLAYER_COLUMNS) - TEXT_COLUMNS


def database_url() -> str:
    return os.getenv("DATABASE_URL", "").strip()


def _connect() -> Any:
    import psycopg

    return psycopg.connect(database_url())


def has_database_store() -> bool:
    return bool(database_url())


def _players_table_sql(table_name: str) -> str:
    columns = []
    for column in PLAYER_COLUMNS:
        if column in TEXT_COLUMNS:
            columns.append(f"{column} text NOT NULL DEFAULT ''")
        else:
            columns.append(f"{column} integer NOT NULL DEFAULT 0")
    return f"CREATE TABLE {table_name} ({', '.join(columns)})"


def _prepare_players(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    for column in PLAYER_COLUMNS:
        if column not in clean.columns:
            clean[column] = "" if column in TEXT_COLUMNS else 0
    clean = clean[PLAYER_COLUMNS]
    for column in TEXT_COLUMNS:
        clean[column] = clean[column].fillna("").astype(str)
    for column in INTEGER_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce").fillna(0).astype(int)
    return clean


def write_players_to_database(df: pd.DataFrame) -> None:
    if not has_database_store():
        return

    clean = _prepare_players(df)
    csv_buffer = io.StringIO()
    clean.to_csv(csv_buffer, index=False, header=False, lineterminator="\n")
    csv_buffer.seek(0)

    updated_at = datetime.now(UTC).isoformat()
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS pvp_players_new")
            cur.execute(_players_table_sql("pvp_players_new"))
            with cur.copy(
                f"COPY pvp_players_new ({', '.join(PLAYER_COLUMNS)}) FROM STDIN WITH (FORMAT CSV)"
            ) as copy:
                copy.write(csv_buffer.getvalue())

            cur.execute("DROP TABLE IF EXISTS pvp_players_old")
            cur.execute("ALTER TABLE IF EXISTS pvp_players RENAME TO pvp_players_old")
            cur.execute("ALTER TABLE pvp_players_new RENAME TO pvp_players")
            cur.execute("DROP TABLE IF EXISTS pvp_players_old")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_pvp_players_region ON pvp_players(region)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_pvp_players_class ON pvp_players(class_name)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_pvp_players_spec ON pvp_players(spec_name)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS pvp_dataset_metadata (
                    id integer PRIMARY KEY DEFAULT 1,
                    updated_at text NOT NULL,
                    row_count integer NOT NULL
                )
                """
            )
            cur.execute(
                """
                INSERT INTO pvp_dataset_metadata (id, updated_at, row_count)
                VALUES (1, %s, %s)
                ON CONFLICT (id)
                DO UPDATE SET updated_at = EXCLUDED.updated_at, row_count = EXCLUDED.row_count
                """,
                (updated_at, len(clean)),
            )


def read_players_from_database() -> pd.DataFrame:
    if not has_database_store():
        return pd.DataFrame(columns=PLAYER_COLUMNS)

    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT {', '.join(PLAYER_COLUMNS)} FROM pvp_players")
                rows = cur.fetchall()
    except Exception as exc:
        print(f"Postgres dataset is not available yet: {exc}")
        return pd.DataFrame(columns=PLAYER_COLUMNS)

    return pd.DataFrame(rows, columns=PLAYER_COLUMNS)


def database_dataset_version() -> str | None:
    if not has_database_store():
        return None

    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT updated_at, row_count FROM pvp_dataset_metadata WHERE id = 1")
                row = cur.fetchone()
    except Exception:
        return None

    if not row:
        return None
    return f"{row[0]}:{row[1]}"


def local_dataset_version(path: Path) -> str | None:
    if not path.exists():
        return None
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def dataset_version(path: Path) -> str | None:
    return database_dataset_version() if has_database_store() else local_dataset_version(path)


def read_processed_players(path: Path) -> pd.DataFrame:
    if has_database_store():
        df = read_players_from_database()
        if not df.empty:
            return df

    if not path.exists():
        return pd.DataFrame(columns=PLAYER_COLUMNS)
    return pd.read_parquet(path)
