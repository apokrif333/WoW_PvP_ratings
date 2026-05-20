from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    blizzard_client_id: str
    blizzard_client_secret: str
    data_dir: Path

    @classmethod
    def from_env(cls, data_dir: Path = Path("data")) -> "Settings":
        load_dotenv(encoding="utf-8-sig")
        client_id = os.getenv("BLIZZARD_CLIENT_ID", "").strip()
        client_secret = os.getenv("BLIZZARD_CLIENT_SECRET", "").strip()
        if not client_id or not client_secret:
            raise RuntimeError(
                "Missing BLIZZARD_CLIENT_ID or BLIZZARD_CLIENT_SECRET. "
                "Create a ..env file from ..env.example."
            )
        return cls(
            blizzard_client_id=client_id,
            blizzard_client_secret=client_secret,
            data_dir=data_dir,
        )
