from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from wowpvp.blizzard import BlizzardClient, fetch_blizzard_profile_specs, fetch_blizzard_pvp_data
from wowpvp.checkpvp import CheckPvpClient, fetch_checkpvp_rankings
from wowpvp.config import Settings
from wowpvp.etl import blizzard_global_profile_candidates, build_final_dataset
from wowpvp.icons import fetch_blizzard_icons
from wowpvp.utils import reset_generated_data


def default_parallel_workers(cpu_target: float) -> int:
    cpu_count = os.cpu_count() or 1
    cap = int(os.getenv("WOWPVP_MAX_WORKERS_CAP", "16"))
    return max(1, min(cap, math.floor(cpu_count * cpu_target)))


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
        "--skip-blizzard-profiles",
        action="store_true",
        help="Do not fetch character profiles for Blizzard-only 2v2/3v3/RBG players.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached HTTP/page data where supported.",
    )
    parser.add_argument(
        "--reset-data",
        action="store_true",
        help="Remove generated cache/raw/processed data before fetching fresh data.",
    )
    parser.add_argument(
        "--cpu-target",
        type=float,
        default=float(os.getenv("WOWPVP_CPU_TARGET", "0.90")),
        help="Target CPU share used to choose the default parallel worker count.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        help=(
            "Maximum parallel HTTP workers. Defaults to floor(cpu_count * cpu_target), "
            "capped by WOWPVP_MAX_WORKERS_CAP."
        ),
    )
    parser.add_argument(
        "--blizzard-profile-workers",
        type=int,
        default=None,
        help=(
            "Parallel workers for targeted Blizzard character profile lookups. "
            "Defaults to --parallel-workers and is capped internally."
        ),
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
    parser.add_argument(
        "--skip-csv",
        action="store_true",
        help="Only write parquet/Postgres output, not the large CSV export.",
    )
    parser.add_argument(
        "--skip-database",
        action="store_true",
        help="Do not write the processed dataset to DATABASE_URL/Postgres.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reset_data:
        reset_generated_data(args.data_dir)

    settings = Settings.from_env(data_dir=args.data_dir)
    regions = [region.lower() for region in args.regions]
    parallel_workers = args.parallel_workers or default_parallel_workers(args.cpu_target)
    print(
        f"Parallel workers: {parallel_workers} "
        f"(cpu_target={args.cpu_target:.0%}, cpu_count={os.cpu_count() or 1})"
    )
    blizzard = BlizzardClient(settings.blizzard_client_id, settings.blizzard_client_secret)

    if not args.skip_icons:
        fetch_blizzard_icons(blizzard, force=args.force)

    if not args.skip_blizzard:
        fetch_blizzard_pvp_data(
            client=blizzard,
            regions=regions,
            data_dir=settings.data_dir,
            force=args.force,
            max_workers=parallel_workers,
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
            max_workers=parallel_workers,
        )

    if not args.skip_blizzard and not args.skip_blizzard_profiles:
        profile_candidates = blizzard_global_profile_candidates(settings.data_dir, regions=regions)
        if not profile_candidates.empty:
            fetch_blizzard_profile_specs(
                client=blizzard,
                players=profile_candidates,
                data_dir=settings.data_dir,
                force=args.force,
                max_workers=args.blizzard_profile_workers or parallel_workers,
            )
        else:
            print("Blizzard profile specs: no Blizzard-only 2v2/3v3/RBG players")

    output_path = build_final_dataset(
        settings.data_dir,
        write_csv=not args.skip_csv,
        write_database=not args.skip_database,
    )
    print(f"Saved merged PvP dataset: {output_path}")


if __name__ == "__main__":
    main()
