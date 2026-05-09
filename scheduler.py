from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta


def refresh_command() -> list[str]:
    extra_args = os.getenv("WOWPVP_REFRESH_EXTRA_ARGS", "").split()
    return [
        sys.executable,
        "main.py",
        "--force",
        "--reset-data",
        "--skip-icons",
        "--skip-csv",
        *extra_args,
    ]


def run_refresh() -> None:
    started_at = datetime.now(UTC)
    print(f"Starting WoW PvP refresh at {started_at.isoformat()}", flush=True)
    subprocess.run(refresh_command(), check=True)
    finished_at = datetime.now(UTC)
    print(
        f"Finished WoW PvP refresh at {finished_at.isoformat()} "
        f"after {finished_at - started_at}",
        flush=True,
    )


def main() -> None:
    interval_hours = float(os.getenv("WOWPVP_REFRESH_INTERVAL_HOURS", "24"))
    run_on_start = os.getenv("WOWPVP_REFRESH_ON_START", "1").lower() not in {"0", "false", "no"}

    next_run = datetime.now(UTC) if run_on_start else datetime.now(UTC) + timedelta(hours=interval_hours)
    while True:
        now = datetime.now(UTC)
        if now >= next_run:
            try:
                run_refresh()
            except Exception as exc:
                print(f"WoW PvP refresh failed: {exc}", flush=True)
            next_run = datetime.now(UTC) + timedelta(hours=interval_hours)

        sleep_seconds = max(30.0, min(300.0, (next_run - datetime.now(UTC)).total_seconds()))
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
