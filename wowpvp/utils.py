from __future__ import annotations

import re
import unicodedata
from pathlib import Path


def ensure_dirs(data_dir: Path) -> None:
    for path in [
        data_dir,
        data_dir / "cache" / "checkpvp",
        data_dir / "raw",
        data_dir / "processed",
    ]:
        path.mkdir(parents=True, exist_ok=True)


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
