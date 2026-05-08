from __future__ import annotations

import time
from pathlib import Path

import requests

from wowpvp.blizzard import BlizzardClient
from wowpvp.constants import CLASS_ID_TO_NAME, SPEC_ID_TO_INFO


def icon_slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in value).strip("-")


def _icon_url(media: dict) -> str:
    for asset in media.get("assets", []):
        if asset.get("key") == "icon" and asset.get("value"):
            return str(asset["value"])
    raise RuntimeError(f"No icon asset returned: {media}")


def _download_icon(url: str, path: Path, force: bool = False) -> None:
    if path.exists() and not force:
        return

    response = requests.get(url, timeout=45, headers={"User-Agent": "WoWPvPData/1.0"})
    response.raise_for_status()
    path.write_bytes(response.content)


def fetch_blizzard_icons(
    client: BlizzardClient,
    output_dir: Path = Path("assets/icons"),
    region: str = "eu",
    force: bool = False,
) -> None:
    class_dir = output_dir / "class"
    spec_dir = output_dir / "spec"
    class_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    for class_id, class_name in sorted(CLASS_ID_TO_NAME.items()):
        media = client.get(region, f"media/playable-class/{class_id}", namespace=f"static-{region}")
        url = _icon_url(media)
        _download_icon(url, class_dir / f"{icon_slug(class_name)}.jpg", force=force)
        time.sleep(0.03)

    for spec_id, (class_name, spec_name) in sorted(SPEC_ID_TO_INFO.items()):
        media = client.get(
            region,
            f"media/playable-specialization/{spec_id}",
            namespace=f"static-{region}",
        )
        url = _icon_url(media)
        _download_icon(
            url,
            spec_dir / f"{icon_slug(class_name)}-{icon_slug(spec_name)}.jpg",
            force=force,
        )
        time.sleep(0.03)
