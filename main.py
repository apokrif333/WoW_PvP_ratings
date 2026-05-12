from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Sequence

from wowpvp.blizzard import BlizzardClient, fetch_blizzard_pvp_data
from wowpvp.checkpvp import CheckPvpClient, fetch_checkpvp_rankings
from wowpvp.config import Settings
from wowpvp.enrichment import enrich_processed_players
from wowpvp.etl import build_final_dataset
from wowpvp.icons import fetch_blizzard_icons
from wowpvp.storage import read_processed_players
from wowpvp.utils import reset_generated_data


def default_parallel_workers(cpu_target: float) -> int:
    cpu_count = os.cpu_count() or 1
    cap = int(os.getenv("WOWPVP_MAX_WORKERS_CAP", "16"))
    return max(1, min(cap, math.floor(cpu_count * cpu_target)))


def _default_parser_args() -> Sequence[str] | None:
    """Avoid parsing Jupyter's own kernel arguments when main() is called in a notebook."""
    kernel_args = sys.argv[1:]
    has_kernel_file_arg = (
        "-f" in kernel_args and any(arg.endswith("connection.json") for arg in kernel_args)
    )
    has_ipykernel_runtime_arg = any(
        arg.startswith("--IPKernelApp.") or arg.startswith("--IPCompleter.")
        for arg in kernel_args
    )
    if (
        Path(sys.argv[0]).name == "ipykernel_launcher.py"
        or has_kernel_file_arg
        or has_ipykernel_runtime_arg
    ):
        return []
    return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--enrich-blizzard-profile",
        "--enrich-wow-profile",
        dest="enrich_blizzard_profile",
        action="store_true",
        help="Fill unknown ratings from World of Warcraft character pvp.json profiles.",
    )
    parser.add_argument(
        "--force-enrichment",
        action="store_true",
        help="Ignore cached World of Warcraft pvp.json enrichment data.",
    )
    parser.add_argument(
        "--incremental-enrichment",
        action="store_true",
        help="Only refresh pvp.json enrichment for players changed since the previous dataset.",
    )
    parser.add_argument(
        "--enrichment-workers",
        type=int,
        default=None,
        help="Maximum parallel workers for pvp.json enrichment.",
    )
    parser.add_argument(
        "--max-enrichment-players",
        type=int,
        default=None,
        help="Debug limit for World of Warcraft pvp.json profile requests.",
    )
    parser.add_argument(
        "--max-enrichment-brackets",
        type=int,
        default=None,
        help="Deprecated debug option kept for compatibility; pvp.json has no bracket fan-out.",
    )
    parser.add_argument(
        "--enrichment-retry-attempts",
        type=int,
        default=25,
        help="Maximum retry attempts per failed enrichment request.",
    )
    parser.add_argument(
        "--enrichment-retry-delay",
        type=float,
        default=20.0,
        help="Initial backoff delay in seconds for failed enrichment requests.",
    )
    parser.add_argument(
        "--enrichment-retry-max-delay",
        type=float,
        default=600.0,
        help="Maximum backoff delay in seconds for failed enrichment requests.",
    )
    parser.add_argument(
        "--enrichment-request-delay",
        type=float,
        default=0.03,
        help="Minimum global delay in seconds between World of Warcraft pvp.json requests.",
    )
    parser.add_argument(
        "--enrichment-request-jitter",
        type=float,
        default=0.02,
        help="Random extra delay in seconds added between pvp.json requests.",
    )
    return parser.parse_args(_default_parser_args() if argv is None else argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    previous_players = None
    if args.enrich_blizzard_profile and args.incremental_enrichment:
        previous_players = read_processed_players(args.data_dir / "processed" / "pvp_players.parquet")
        if previous_players.empty:
            previous_players = None

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

    output_path = build_final_dataset(
        settings.data_dir,
        write_csv=not args.skip_csv and not args.enrich_blizzard_profile,
        write_database=not args.skip_database and not args.enrich_blizzard_profile,
    )
    if args.enrich_blizzard_profile:
        enrichment_workers = args.enrichment_workers or parallel_workers
        print(f"Enrichment workers: {enrichment_workers}")
        output_path = enrich_processed_players(
            client=blizzard,
            data_dir=settings.data_dir,
            max_workers=enrichment_workers,
            force=args.force_enrichment,
            incremental=args.incremental_enrichment,
            previous_players=previous_players,
            max_players=args.max_enrichment_players,
            max_brackets=args.max_enrichment_brackets,
            retry_attempts=args.enrichment_retry_attempts,
            retry_delay_seconds=args.enrichment_retry_delay,
            max_retry_delay_seconds=args.enrichment_retry_max_delay,
            request_delay_seconds=args.enrichment_request_delay,
            request_jitter_seconds=args.enrichment_request_jitter,
            write_csv=not args.skip_csv,
            write_database=not args.skip_database,
        )
    print(f"Saved merged PvP dataset: {output_path}")


if __name__ == "__main__":
    main()
