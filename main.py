from __future__ import annotations

import argparse
from pathlib import Path

from wowpvp.blizzard import BlizzardClient, fetch_blizzard_pvp_data
from wowpvp.checkpvp import CheckPvpClient, fetch_checkpvp_rankings
from wowpvp.config import Settings
from wowpvp.etl import build_final_dataset
from wowpvp.icons import fetch_blizzard_icons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and merge World of Warcraft PvP data from Blizzard and check-pvp."
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["eu", "us"],
        choices=["eu", "us"],
        help="Regions to fetch.",
    )
    parser.add_argument(
        "--skip-blizzard",
        action="store_true",
        help="Use existing local Blizzard raw data.",
    )
    parser.add_argument(
        "--skip-checkpvp",
        action="store_true",
        help="Use existing local check-pvp raw data.",
    )
    parser.add_argument(
        "--skip-icons",
        action="store_true",
        help="Do not refresh local Blizzard class/spec icons.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached HTTP/page data where supported.",
    )
    parser.add_argument(
        "--max-checkpvp-pages",
        type=int,
        default=None,
        help="Debug limit for check-pvp pages per region.",
    )
    parser.add_argument(
        "--checkpvp-delay",
        type=float,
        default=0.20,
        help="Delay between check-pvp page requests in seconds.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Local data directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env(data_dir=args.data_dir)
    regions = [region.lower() for region in args.regions]
    blizzard = BlizzardClient(settings.blizzard_client_id, settings.blizzard_client_secret)

    if not args.skip_icons:
        fetch_blizzard_icons(blizzard, force=args.force)

    if not args.skip_blizzard:
        fetch_blizzard_pvp_data(
            client=blizzard,
            regions=regions,
            data_dir=settings.data_dir,
            force=args.force,
        )

    if not args.skip_checkpvp:
        checkpvp = CheckPvpClient(settings.data_dir)
        fetch_checkpvp_rankings(
            client=checkpvp,
            regions=regions,
            data_dir=settings.data_dir,
            delay_seconds=args.checkpvp_delay,
            max_pages=args.max_checkpvp_pages,
            force=args.force,
        )

    output_path = build_final_dataset(settings.data_dir)
    print(f"Saved merged PvP dataset: {output_path}")


if __name__ == "__main__":
    main()
