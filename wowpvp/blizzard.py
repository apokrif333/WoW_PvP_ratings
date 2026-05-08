from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests import Session
from requests.auth import HTTPBasicAuth
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from wowpvp.constants import CLASS_SLUG_TO_NAME, SPEC_SLUG_TO_NAME
from wowpvp.utils import ensure_dirs, player_key, slugify_realm


@dataclass
class BlizzardClient:
    client_id: str
    client_secret: str
    timeout: int = 45

    def __post_init__(self) -> None:
        self.session = self._make_session()
        self._tokens: dict[str, str] = {}

    def _make_session(self) -> Session:
        session = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=0.8,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": "WoWPvPData/1.0"})
        return session

    def get_access_token(self, region: str) -> str:
        region = region.lower()
        if region in self._tokens:
            return self._tokens[region]

        response = self.session.post(
            "https://oauth.battle.net/token",
            auth=HTTPBasicAuth(self.client_id, self.client_secret),
            data={"grant_type": "client_credentials"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        token = response.json()["access_token"]
        self._tokens[region] = token
        return token

    def get(self, region: str, path: str, namespace: str | None = None) -> dict[str, Any]:
        token = self.get_access_token(region)
        namespace = namespace or f"dynamic-{region}"
        response = self.session.get(
            f"https://{region}.api.blizzard.com/data/wow/{path.lstrip('/')}",
            params={"namespace": namespace, "locale": "en_US"},
            headers={"Authorization": f"Bearer {token}"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_current_pvp_season_id(self, region: str) -> int:
        data = self.get(region, "pvp-season/index")
        seasons = data.get("seasons", [])
        if not seasons:
            raise RuntimeError(f"No PvP seasons returned for region {region}.")
        return max(int(season["id"]) for season in seasons)


def parse_spec_leaderboard_slug(slug: str) -> tuple[str, str, str]:
    mode, class_slug, spec_slug = slug.split("-", 2)
    class_name = CLASS_SLUG_TO_NAME.get(class_slug, class_slug.replace("-", " ").title())
    spec_name = SPEC_SLUG_TO_NAME.get(spec_slug, spec_slug.replace("-", " ").title())
    return mode, class_name, spec_name


def extract_leaderboard_rows(
    region: str,
    season_id: int,
    leaderboard_slug: str,
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mode, class_name, spec_name = parse_spec_leaderboard_slug(leaderboard_slug)
    rows: list[dict[str, Any]] = []

    for entry in entries:
        character = entry.get("character") or {}
        realm = character.get("realm") or {}
        name = character.get("name") or ""
        realm_slug = slugify_realm(realm.get("slug") or "")
        if not name or not realm_slug:
            continue

        rows.append(
            {
                "player_key": player_key(region, realm_slug, name),
                "region": region,
                "realm_slug": realm_slug,
                "realm": realm_slug,
                "character_name": name,
                "class_name": class_name,
                "spec_name": spec_name,
                "mode": mode,
                "rating": int(entry.get("rating") or 0),
                "rank": int(entry.get("rank") or 0),
                "season_id": season_id,
                "leaderboard": leaderboard_slug,
                "played": int((entry.get("season_match_statistics") or {}).get("played") or 0),
                "won": int((entry.get("season_match_statistics") or {}).get("won") or 0),
                "lost": int((entry.get("season_match_statistics") or {}).get("lost") or 0),
            }
        )
    return rows


def fetch_blizzard_pvp_data(
    client: BlizzardClient,
    regions: list[str],
    data_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    ensure_dirs(data_dir)
    all_rows: list[dict[str, Any]] = []

    for region in regions:
        region = region.lower()
        season_id = client.get_current_pvp_season_id(region)
        raw_path = data_dir / "raw" / f"blizzard_{region}_season_{season_id}.parquet"
        meta_path = data_dir / "raw" / f"blizzard_{region}_season_{season_id}.json"

        if raw_path.exists() and not force:
            print(f"Blizzard {region}: using cached {raw_path}")
            all_rows.extend(pd.read_parquet(raw_path).to_dict("records"))
            continue

        leaderboard_index = client.get(region, f"pvp-season/{season_id}/pvp-leaderboard/index")
        leaderboards = [
            item["name"]
            for item in leaderboard_index.get("leaderboards", [])
            if item.get("name", "").startswith(("shuffle-", "blitz-"))
            and not item.get("name", "").endswith("-overall")
        ]
        print(f"Blizzard {region}: season {season_id}, {len(leaderboards)} spec leaderboards")

        region_rows: list[dict[str, Any]] = []
        for index, leaderboard_slug in enumerate(leaderboards, start=1):
            data = client.get(region, f"pvp-season/{season_id}/pvp-leaderboard/{leaderboard_slug}")
            entries = data.get("entries", [])
            rows = extract_leaderboard_rows(region, season_id, leaderboard_slug, entries)
            region_rows.extend(rows)
            print(
                f"Blizzard {region}: {index}/{len(leaderboards)} "
                f"{leaderboard_slug} rows={len(rows)}"
            )
            time.sleep(0.05)

        region_df = pd.DataFrame(region_rows)
        region_df.to_parquet(raw_path, index=False)
        meta_path.write_text(
            json.dumps(
                {
                    "region": region,
                    "season_id": season_id,
                    "leaderboards": len(leaderboards),
                    "rows": len(region_df),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        all_rows.extend(region_rows)

    return pd.DataFrame(all_rows)
