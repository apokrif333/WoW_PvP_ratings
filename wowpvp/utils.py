from __future__ import annotations

import re
import shutil
import unicodedata
from pathlib import Path


GENERATED_DATA_DIRS = ("cache", "raw", "processed")
PRESERVED_RAW_FILES = {
    "worldofwarcraft_pvp_profiles.parquet",
}


def ensure_dirs(data_dir: Path) -> None:
    for path in [
        data_dir,
        data_dir / "cache" / "checkpvp",
        data_dir / "raw",
        data_dir / "processed",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def reset_generated_data(data_dir: Path) -> None:
    data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    for directory_name in GENERATED_DATA_DIRS:
        target = (data_dir / directory_name).resolve()
        if data_dir not in target.parents:
            raise RuntimeError(f"Refusing to delete path outside data dir: {target}")
        if target.exists():
            if directory_name == "raw":
                for child in target.iterdir():
                    if child.name in PRESERVED_RAW_FILES:
                        continue
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
                continue
            shutil.rmtree(target)

    ensure_dirs(data_dir)


def normalize_character_name(name: str) -> str:
    return (name or "").strip().casefold()


def slugify_realm(value: str) -> str:
    value = (value or "").strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[()']", "", value)
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def player_key(region: str, realm_slug: str, character_name: str) -> str:
    return "|".join(
        [
            region.lower().strip(),
            slugify_realm(realm_slug),
            normalize_character_name(character_name),
        ]
    )
