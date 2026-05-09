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

## Heroku

The app supports Heroku Postgres through `DATABASE_URL`. This is required for daily refreshes on Heroku because dyno filesystems are ephemeral.

```powershell
heroku git:remote -a wowpvp
heroku addons:create heroku-postgresql:essential-0 -a wowpvp
heroku config:set BLIZZARD_CLIENT_ID=... BLIZZARD_CLIENT_SECRET=... -a wowpvp
git push heroku master
heroku ps:scale web=1 clock=1 -a wowpvp
```

The `clock` dyno runs `scheduler.py`, which executes:

```powershell
python main.py --force --reset-data --skip-icons --skip-csv
```

once every 24 hours by default. Change the interval with `WOWPVP_REFRESH_INTERVAL_HOURS`.
