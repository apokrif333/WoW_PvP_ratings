# WoW PvP Data

Local ETL and Dash dashboard for World of Warcraft PvP ratings.

## Setup

Create `.env`:

```env
BLIZZARD_CLIENT_ID=...
BLIZZARD_CLIENT_SECRET=...
```

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Fetch Data

Full EU + US fetch:

```powershell
.\.venv\Scripts\python.exe main.py
```

Fresh overwrite fetch with parallel downloads:

```powershell
.\.venv\Scripts\python.exe main.py --force --reset-data
```

Useful debug run:

```powershell
.\.venv\Scripts\python.exe main.py --regions eu --max-checkpvp-pages 2 --force
```

Optional Blizzard profile enrichment for unknown ratings:

```powershell
.\.venv\Scripts\python.exe main.py --skip-blizzard --skip-checkpvp --skip-icons --enrich-blizzard-profile
```

The enrichment step reads `data/processed/pvp_players.parquet`, checks Blizzard character profile PvP endpoints for rows where a mode rating is unknown, and writes the same processed parquet/CSV/database outputs. It caches profile summaries, bracket responses, and character profiles under `data/raw/blizzard_profile_*.parquet`, so interrupted runs can resume. Use `--max-enrichment-players` and `--max-enrichment-brackets` for small test runs.

The final merged files are written to:

- `data/processed/pvp_players.parquet`
- `data/processed/pvp_players.csv`

`--reset-data` removes generated `data/cache`, `data/raw`, and `data/processed` before fetching so old pages and old season files do not accumulate. Parallel workers default to `floor(cpu_count * 0.90)`, capped by `WOWPVP_MAX_WORKERS_CAP=16`, and can be overridden with `--parallel-workers` or `WOWPVP_CPU_TARGET`.

## Dashboard

```powershell
.\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:8050`.

The dashboard uses server-side filtering, sorting, and pagination so the browser only receives the current page. It includes:

- player table with global sorting, multi-select string filters, rating range filters, and 10/20/50/100 page sizes
- spec summary table by game mode and region with Blizzard class/spec icons, fixed core columns, optional extra stat columns, rating bands, 1800+ stats, 1800+ lift, and mode-specific P80 lift

## Free Internet Deployment

Recommended free setup:

- Render Free Web Service runs the Dash app.
- Supabase Free Postgres stores the latest dataset through `DATABASE_URL`.
- GitHub Actions runs the daily refresh and rewrites the database.

### Supabase

Create a Supabase project and copy the Postgres connection string. Use a connection string with SSL, for example:

```env
DATABASE_URL=postgresql://user:password@host:5432/postgres?sslmode=require
```

### GitHub Secrets

Add these repository secrets in GitHub:

- `BLIZZARD_CLIENT_ID`
- `BLIZZARD_CLIENT_SECRET`
- `DATABASE_URL`

The workflow in `.github/workflows/refresh-data.yml` can be started manually with `workflow_dispatch` and also runs daily at 08:00 UTC.

### Render

Create a new Render Blueprint from this repository. `render.yaml` defines a free Python web service with:

- build command: `pip install -r requirements.txt`
- start command: `gunicorn app:server --workers 1 --threads 4 --timeout 120`

Set these Render environment variables:

- `BLIZZARD_CLIENT_ID`
- `BLIZZARD_CLIENT_SECRET`
- `DATABASE_URL`

The Render web service reads the current dataset from Supabase. If the database has not been populated yet, run the GitHub workflow once manually.
