from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from py_mini_racer import py_mini_racer
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from wowpvp.utils import ensure_dirs


CHECKPVP_ORIGIN = "https://check-pvp.fr"
RANKING_PATH = (
    "characters/ranking?classes=all&region={region}"
    "&sort=rateatm3v3&order=desc&minRating=0&maxRating=4100{page}"
)


class CheckPvpClient:
    def __init__(self, data_dir: Path, timeout: int = 45) -> None:
        self.data_dir = data_dir
        self.timeout = timeout
        self.session = self._make_session()
        self.signer = self._load_signer()

    def _make_session(self) -> Session:
        session = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
                )
            }
        )
        return session

    def _load_signer(self) -> py_mini_racer.MiniRacer:
        cache_dir = self.data_dir / "cache" / "checkpvp"
        cache_dir.mkdir(parents=True, exist_ok=True)

        html = self.session.get(f"{CHECKPVP_ORIGIN}/ranking", timeout=self.timeout).text
        match = re.search(r'<script src="(scripts\.[^"]+\.js)"', html)
        if not match:
            raise RuntimeError("Could not find check-pvp signing script in /ranking HTML.")

        script_name = match.group(1)
        script_path = cache_dir / script_name
        if script_path.exists():
            script = script_path.read_text(encoding="utf-8")
        else:
            response = self.session.get(f"{CHECKPVP_ORIGIN}/{script_name}", timeout=self.timeout)
            response.raise_for_status()
            script = response.text
            script_path.write_text(script, encoding="utf-8")

        ctx = py_mini_racer.MiniRacer()
        ctx.eval(
            """
            var window = this;
            var self = this;
            var globalThis = this;
            var console = {error: function(){}, log: function(){}};
            var crypto = {
              getRandomValues: function(arr) {
                for (var i = 0; i < arr.length; i++) {
                  arr[i] = Math.floor(Math.random() * 4294967296);
                }
                return arr;
              }
            };
            """
        )
        ctx.eval(script)
        if ctx.eval("typeof _0x4f2a") != "function":
            raise RuntimeError("check-pvp signing function was not initialized.")
        return ctx

    def get_json(self, path: str) -> dict[str, Any]:
        signature = self.signer.eval(f"_0x4f2a({path!r},0,0)")
        response = self.session.get(
            f"{CHECKPVP_ORIGIN}/api/{path}",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": "Bearer null",
                "delay": "0",
                "function": signature,
                "Referer": f"{CHECKPVP_ORIGIN}/ranking",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()


def ranking_path(region: str, page: int | None = None) -> str:
    page_query = f"&page={page}" if page and page > 1 else ""
    return RANKING_PATH.format(region=region, page=page_query)


def fetch_region_rankings(
    client: CheckPvpClient,
    region: str,
    data_dir: Path,
    delay_seconds: float,
    max_pages: int | None,
    force: bool,
) -> pd.DataFrame:
    cache_dir = data_dir / "cache" / "checkpvp"
    raw_path = data_dir / "raw" / f"checkpvp_{region}.parquet"
    meta_path = data_dir / "raw" / f"checkpvp_{region}.json"

    if raw_path.exists() and meta_path.exists() and not force and max_pages is None:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("complete"):
            print(f"check-pvp {region}: using cached {raw_path}")
            return pd.read_parquet(raw_path)

    first_page = client.get_json(ranking_path(region))
    total = int(first_page.get("total") or 0)
    page_size = len(first_page.get("characters") or []) or 50
    page_count = math.ceil(total / page_size)
    if max_pages:
        page_count = min(page_count, max_pages)
    print(f"check-pvp {region}: total={total}, pages={page_count}, page_size={page_size}")

    rows: list[dict[str, Any]] = []
    for page in range(1, page_count + 1):
        page_file = cache_dir / f"ranking_{region}_page_{page}.json"
        if page == 1:
            data = first_page
            page_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        elif page_file.exists() and not force:
            data = json.loads(page_file.read_text(encoding="utf-8"))
        else:
            data = client.get_json(ranking_path(region, page))
            page_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            time.sleep(delay_seconds)

        characters = data.get("characters") or []
        rows.extend(characters)
        print(f"check-pvp {region}: page {page}/{page_count} rows={len(characters)}")

    df = pd.DataFrame(rows)
    df.to_parquet(raw_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "region": region,
                "total": total,
                "page_size": page_size,
                "pages_fetched": page_count,
                "rows": len(df),
                "complete": max_pages is None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return df


def fetch_checkpvp_rankings(
    client: CheckPvpClient,
    regions: list[str],
    data_dir: Path,
    delay_seconds: float = 0.20,
    max_pages: int | None = None,
    force: bool = False,
) -> pd.DataFrame:
    ensure_dirs(data_dir)
    frames = [
        fetch_region_rankings(
            client=client,
            region=region.lower(),
            data_dir=data_dir,
            delay_seconds=delay_seconds,
            max_pages=max_pages,
            force=force,
        )
        for region in regions
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
