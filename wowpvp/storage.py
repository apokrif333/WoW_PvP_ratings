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
    "shuffle_class_name",
    "shuffle_spec_name",
    "blitz_class_name",
    "blitz_spec_name",
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
    "shuffle_class_name",
    "shuffle_spec_name",
    "blitz_class_name",
    "blitz_spec_name",
}
INTEGER_COLUMNS = set(PLAYER_COLUMNS) - TEXT_COLUMNS
TEXT_PLAYER_COLUMNS = [column for column in PLAYER_COLUMNS if column in TEXT_COLUMNS]


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
            columns.append(f"{column} integer")
    return f"CREATE TABLE {table_name} ({', '.join(columns)})"


def _prepare_players(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    for column in PLAYER_COLUMNS:
        if column not in clean.columns:
            clean[column] = "" if column in TEXT_COLUMNS else pd.NA
    clean = clean[PLAYER_COLUMNS]
    for column in TEXT_COLUMNS:
        clean[column] = clean[column].fillna("").astype(str)
    for column in INTEGER_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce").astype("Int64")
    return clean


def _ensure_public_read_policy(cur: Any, table_name: str, policy_name: str) -> None:
    cur.execute(f"ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY")
    cur.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_policies
                WHERE schemaname = 'public'
                  AND tablename = %s
                  AND policyname = %s
            ) THEN
                EXECUTE format(
                    'CREATE POLICY %I ON %I FOR SELECT USING (true)',
                    %s,
                    %s
                );
            END IF;
        END
        $$;
        """,
        (table_name, policy_name, policy_name, table_name),
    )


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
            text_columns_sql = ", ".join(TEXT_PLAYER_COLUMNS)
            with cur.copy(
                f"COPY pvp_players_new ({', '.join(PLAYER_COLUMNS)}) "
                f"FROM STDIN WITH (FORMAT CSV, FORCE_NOT_NULL ({text_columns_sql}))"
            ) as copy:
                copy.write(csv_buffer.getvalue())

            cur.execute("DROP TABLE IF EXISTS pvp_players_old")
            cur.execute("ALTER TABLE IF EXISTS pvp_players RENAME TO pvp_players_old")
            cur.execute("ALTER TABLE pvp_players_new RENAME TO pvp_players")
            cur.execute("DROP TABLE IF EXISTS pvp_players_old")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_pvp_players_region ON pvp_players(region)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_pvp_players_class ON pvp_players(class_name)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_pvp_players_spec ON pvp_players(spec_name)")
            _ensure_public_read_policy(
                cur,
                table_name="pvp_players",
                policy_name="pvp_players_public_read",
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS pvp_dataset_metadata (
                    id integer PRIMARY KEY DEFAULT 1,
                    updated_at text NOT NULL,
                    row_count integer NOT NULL
                )
                """
            )
            _ensure_public_read_policy(
                cur,
                table_name="pvp_dataset_metadata",
                policy_name="pvp_dataset_metadata_public_read",
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


def _select_player_columns(columns: list[str] | tuple[str, ...] | None = None) -> list[str]:
    if not columns:
        return PLAYER_COLUMNS
    selected = [column for column in columns if column in PLAYER_COLUMNS]
    return selected or PLAYER_COLUMNS


def _csv_read_dtypes(columns: list[str]) -> dict[str, str]:
    return {column: "category" for column in columns if column in TEXT_COLUMNS}


def _write_copy_chunks_to_buffer(copy: Any, buffer: io.BytesIO) -> None:
    while True:
        chunk = copy.read()
        if not chunk:
            break
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        elif isinstance(chunk, memoryview):
            chunk = chunk.tobytes()
        buffer.write(chunk)


def read_players_from_database(columns: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    selected_columns = _select_player_columns(columns)
    if not has_database_store():
        return pd.DataFrame(columns=selected_columns)

    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'pvp_players'
                    """
                )
                available_columns = {row[0] for row in cur.fetchall()}
                existing_columns = [
                    column for column in selected_columns if column in available_columns
                ]
                if not existing_columns:
                    return pd.DataFrame(columns=selected_columns)

                csv_buffer = io.BytesIO()
                column_sql = ", ".join(existing_columns)
                with cur.copy(
                    f"COPY (SELECT {column_sql} FROM pvp_players) "
                    "TO STDOUT WITH (FORMAT CSV, HEADER TRUE)"
                ) as copy:
                    _write_copy_chunks_to_buffer(copy, csv_buffer)
    except Exception as exc:
        print(f"Postgres dataset is not available yet: {exc}")
        return pd.DataFrame(columns=selected_columns)

    csv_buffer.seek(0)
    df = pd.read_csv(
        csv_buffer,
        dtype=_csv_read_dtypes(existing_columns),
        keep_default_na=False,
    )
    for column in selected_columns:
        if column not in df:
            df[column] = "" if column in TEXT_COLUMNS else pd.NA
    for column in selected_columns:
        if column in INTEGER_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    return df[selected_columns]


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


def read_processed_players(
    path: Path,
    columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    selected_columns = _select_player_columns(columns)
    if has_database_store():
        df = read_players_from_database(selected_columns)
        if not df.empty:
            return df

    if not path.exists():
        return pd.DataFrame(columns=selected_columns)

    import pyarrow.parquet as pq

    available_columns = set(pq.read_schema(path).names)
    existing_columns = [column for column in selected_columns if column in available_columns]
    if existing_columns:
        df = pd.read_parquet(path, columns=existing_columns)
    else:
        df = pd.DataFrame(index=pd.RangeIndex(0))
    for column in selected_columns:
        if column not in df:
            df[column] = "" if column in TEXT_COLUMNS else pd.NA
    return df[selected_columns]
