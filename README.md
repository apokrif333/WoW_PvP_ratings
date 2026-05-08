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

Useful debug run:

```powershell
.\.venv\Scripts\python.exe main.py --regions eu --max-checkpvp-pages 2 --force
```

The final merged files are written to:

- `data/processed/pvp_players.parquet`
- `data/processed/pvp_players.csv`

## Dashboard

```powershell
.\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:8050`.

The dashboard uses server-side filtering, sorting, and pagination so the browser only receives the current page. It includes:

- player table with global sorting, multi-select string filters, rating range filters, and 10/20/50/100 page sizes
- spec summary table by game mode and region with Blizzard class/spec icons, fixed core columns, optional extra stat columns, rating bands, 1800+ stats, and lift
