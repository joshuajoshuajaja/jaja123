# soccer-model — data mirror

This repo exists for one reason: to get football data somewhere my modelling
session can actually read it.

The modelling sandbox has a locked-down egress policy — every football data
host (football-data.co.uk, Understat, FBref, the odds APIs) is blocked. GitHub
is not. So a GitHub Action runs on GitHub's own servers, which have open
internet, pulls the data, and commits it here. The model then reads it over
`raw.githubusercontent.com`.

## Setup

1. Create a **private** repo and drop these files in at the root.
2. Settings → Secrets and variables → Actions → New repository secret:
   - `ODDS_API_KEY` — from https://the-odds-api.com (free tier is fine to start)
   - Skip this and the fetch still runs; it just won't pull live odds.
3. Settings → Actions → General → Workflow permissions → **Read and write**.
4. Actions tab → `daily-fetch` → **Run workflow** to test it now.

The first run takes ~5–10 minutes (it pulls 12 seasons across ~35 leagues).
Every run after that is fast.

## What lands in `data/raw/`

| File | What it is |
|---|---|
| `matches.parquet` / `matches.csv.gz` | Every result + bookmaker closing odds, football-data.co.uk |
| `clubelo.csv` | Today's ClubElo ratings — cross-league strength prior |
| `xg.parquet` | Understat xG for the top 5 leagues |
| `odds_history.csv.gz` | Append-only odds snapshots, 90-day window, for closing-line-value tracking |
| `MANIFEST.txt` | Row counts and last run time |

## Schedule

`09:15 UTC` daily = `17:15 SGT`, 45 minutes ahead of the 18:00 SGT dashboard.
