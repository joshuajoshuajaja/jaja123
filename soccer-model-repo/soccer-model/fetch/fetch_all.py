#!/usr/bin/env python3
"""
Daily data pull for the soccer model.

Runs on GitHub Actions (open internet), writes raw files into data/raw/,
and the workflow commits them. The modelling session then reads them over
raw.githubusercontent.com.

Nothing here is clever. It just gets bytes onto disk, reliably, and fails
loudly on the sources that matter.
"""
from __future__ import annotations

import io
import os
import sys
import time
import datetime as dt
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (compatible; soccer-model/1.0)"}
TIMEOUT = 60

# ---------------------------------------------------------------------------
# football-data.co.uk  --  results + closing odds, the backbone of the project
# ---------------------------------------------------------------------------

# Main-file divisions. Key = football-data division code.
MAIN_DIVS = [
    "E0", "E1", "E2", "E3", "EC",          # England: PL -> National League
    "SC0", "SC1", "SC2", "SC3",            # Scotland
    "D1", "D2",                            # Germany
    "I1", "I2",                            # Italy
    "SP1", "SP2",                          # Spain
    "F1", "F2",                            # France
    "N1",                                  # Netherlands
    "B1",                                  # Belgium
    "P1",                                  # Portugal
    "T1",                                  # Turkey
    "G1",                                  # Greece
]

# "New leagues" single-file-per-country (one file covers all seasons).
EXTRA_COUNTRIES = [
    "ARG", "AUT", "BRA", "CHN", "DNK", "FIN", "IRL", "JPN",
    "MEX", "NOR", "POL", "ROU", "RUS", "SWE", "SWZ", "USA",
]

# How many seasons of history to pull for the main divisions.
# 12 seasons is plenty: older data is down-weighted to nothing by the
# time-decay term anyway, and team identity drifts.
N_SEASONS = 12


def season_codes(n: int = N_SEASONS) -> list[str]:
    """['2526', '2425', ...] anchored on the current European season."""
    today = dt.date.today()
    start_year = today.year if today.month >= 7 else today.year - 1
    out = []
    for i in range(n):
        y = start_year - i
        out.append(f"{y % 100:02d}{(y + 1) % 100:02d}")
    return out


def _get(url: str, tries: int = 3) -> bytes | None:
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=TIMEOUT)
            if r.status_code == 200 and r.content:
                return r.content
            if r.status_code == 404:
                return None  # season/division simply doesn't exist
        except requests.RequestException as e:
            print(f"    retry {attempt + 1}: {e}", file=sys.stderr)
        time.sleep(2 * (attempt + 1))
    print(f"    FAILED {url}", file=sys.stderr)
    return None


def _read_csv(blob: bytes) -> pd.DataFrame | None:
    """football-data files are latin-1 and occasionally have ragged tails."""
    for enc in ("latin-1", "utf-8"):
        try:
            df = pd.read_csv(io.BytesIO(blob), encoding=enc,
                             on_bad_lines="skip", low_memory=False)
            df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
            # trailing junk rows have no date
            if "Date" in df.columns:
                df = df[df["Date"].notna()]
            return df if len(df) else None
        except Exception:
            continue
    return None


def fetch_football_data() -> pd.DataFrame:
    frames = []
    seasons = season_codes()
    print(f"football-data.co.uk: {len(MAIN_DIVS)} divisions x {len(seasons)} seasons")
    for div in MAIN_DIVS:
        got = 0
        for season in seasons:
            blob = _get(f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv")
            if not blob:
                continue
            df = _read_csv(blob)
            if df is None:
                continue
            df["Div"] = div
            df["SeasonCode"] = season
            frames.append(df)
            got += 1
        print(f"  {div:<4} {got}/{len(seasons)} seasons")

    print(f"football-data.co.uk extra: {len(EXTRA_COUNTRIES)} countries")
    for ctry in EXTRA_COUNTRIES:
        blob = _get(f"https://www.football-data.co.uk/new/{ctry}.csv")
        if not blob:
            continue
        df = _read_csv(blob)
        if df is None:
            continue
        # Harmonise the extra-file schema onto the main one.
        df = df.rename(columns={
            "Home": "HomeTeam", "Away": "AwayTeam",
            "HG": "FTHG", "AG": "FTAG", "Res": "FTR",
        })
        if "League" in df.columns:
            df["Div"] = df["League"].astype(str)
        df["Source"] = "extra"
        frames.append(df)
        print(f"  {ctry:<4} {len(df)} rows")

    if not frames:
        raise SystemExit("football-data.co.uk returned nothing - aborting")

    out = pd.concat(frames, ignore_index=True, sort=False)
    out["Source"] = out.get("Source", pd.Series(index=out.index)).fillna("main")
    return out


# ---------------------------------------------------------------------------
# ClubElo -- free cross-league strength prior, useful for promoted teams
# ---------------------------------------------------------------------------

def fetch_clubelo() -> pd.DataFrame | None:
    today = dt.date.today().isoformat()
    blob = _get(f"http://api.clubelo.com/{today}")
    if not blob:
        return None
    try:
        return pd.read_csv(io.BytesIO(blob))
    except Exception as e:
        print(f"clubelo parse failed: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# xG -- FBref / Understat via the `soccerdata` package (top leagues only)
# ---------------------------------------------------------------------------

def fetch_xg() -> pd.DataFrame | None:
    try:
        import soccerdata as sd
    except ImportError:
        print("soccerdata not installed - skipping xG", file=sys.stderr)
        return None

    current = dt.date.today()
    start_year = current.year if current.month >= 7 else current.year - 1
    seasons = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(start_year - 5, start_year + 1)]

    try:
        us = sd.Understat(leagues="ENG-Premier League", seasons=seasons)
        del us  # touch-test the constructor before doing the slow pull
    except Exception as e:
        print(f"understat unavailable: {e}", file=sys.stderr)

    frames = []
    for league in ["ENG-Premier League", "ESP-La Liga", "ITA-Serie A",
                   "GER-Bundesliga", "FRA-Ligue 1"]:
        try:
            us = sd.Understat(leagues=league, seasons=seasons)
            df = us.read_team_match_stats()
            df = df.reset_index()
            df["league_key"] = league
            frames.append(df)
            print(f"  xG {league}: {len(df)} rows")
        except Exception as e:
            print(f"  xG {league} failed: {e}", file=sys.stderr)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True, sort=False)


# ---------------------------------------------------------------------------
# Odds snapshot -- The Odds API, if a key is present
# ---------------------------------------------------------------------------

ODDS_SPORTS = [
    "soccer_epl", "soccer_efl_champ", "soccer_england_league1",
    "soccer_england_league2", "soccer_spain_la_liga", "soccer_spain_segunda_division",
    "soccer_italy_serie_a", "soccer_italy_serie_b", "soccer_germany_bundesliga",
    "soccer_germany_bundesliga2", "soccer_france_ligue_one", "soccer_france_ligue_two",
    "soccer_netherlands_eredivisie", "soccer_belgium_first_div",
    "soccer_portugal_primeira_liga", "soccer_turkey_super_league",
    "soccer_greece_super_league", "soccer_brazil_campeonato",
    "soccer_argentina_primera_division", "soccer_japan_j_league",
    "soccer_usa_mls", "soccer_mexico_ligamx",
    "soccer_uefa_champs_league", "soccer_uefa_europa_league",
]


def fetch_odds() -> pd.DataFrame | None:
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        print("ODDS_API_KEY not set - skipping live odds")
        return None

    rows, used = [], 0
    for sport in ODDS_SPORTS:
        url = (f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
               f"?apiKey={key}&regions=eu,uk&markets=h2h,totals,spreads"
               f"&oddsFormat=decimal&dateFormat=iso")
        try:
            r = requests.get(url, timeout=TIMEOUT)
        except requests.RequestException as e:
            print(f"  odds {sport}: {e}", file=sys.stderr)
            continue
        if r.status_code == 422:
            continue  # sport not in season / not offered
        if r.status_code != 200:
            print(f"  odds {sport}: HTTP {r.status_code} {r.text[:120]}", file=sys.stderr)
            if r.status_code == 401:
                break
            continue
        used = r.headers.get("x-requests-used", used)
        for ev in r.json():
            for bk in ev.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    for oc in mk.get("outcomes", []):
                        rows.append({
                            "snapshot_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                            "sport_key": ev.get("sport_key"),
                            "event_id": ev.get("id"),
                            "commence_time": ev.get("commence_time"),
                            "home_team": ev.get("home_team"),
                            "away_team": ev.get("away_team"),
                            "book": bk.get("key"),
                            "book_updated": bk.get("last_update"),
                            "market": mk.get("key"),
                            "outcome": oc.get("name"),
                            "point": oc.get("point"),
                            "price": oc.get("price"),
                        })
    print(f"odds rows: {len(rows)}   credits used this month: {used}")
    return pd.DataFrame(rows) if rows else None


# ---------------------------------------------------------------------------

def main() -> None:
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    matches = fetch_football_data()
    matches.to_parquet(RAW / "matches.parquet", index=False)
    matches.to_csv(RAW / "matches.csv.gz", index=False, compression="gzip")
    print(f"\nmatches: {len(matches):,} rows, {matches['Div'].nunique()} divisions")

    elo = fetch_clubelo()
    if elo is not None:
        elo.to_csv(RAW / "clubelo.csv", index=False)
        print(f"clubelo: {len(elo):,} clubs")

    xg = fetch_xg()
    if xg is not None:
        xg.to_parquet(RAW / "xg.parquet", index=False)
        print(f"xg: {len(xg):,} rows")

    odds = fetch_odds()
    if odds is not None:
        # append-only history so we can measure closing line value later
        hist = RAW / "odds_history.csv.gz"
        if hist.exists():
            prev = pd.read_csv(hist)
            odds = pd.concat([prev, odds], ignore_index=True, sort=False)
            # keep the file from growing without bound: 90 days is enough for CLV
            odds["_ct"] = pd.to_datetime(odds["commence_time"], errors="coerce", utc=True)
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=90)
            odds = odds[odds["_ct"].isna() | (odds["_ct"] >= cutoff)].drop(columns=["_ct"])
        odds.to_csv(hist, index=False, compression="gzip")
        print(f"odds history: {len(odds):,} rows")

    (RAW / "MANIFEST.txt").write_text(
        f"last_run_utc: {stamp}\n"
        f"matches_rows: {len(matches)}\n"
        f"divisions: {matches['Div'].nunique()}\n"
        f"clubelo: {'yes' if elo is not None else 'no'}\n"
        f"xg: {'yes' if xg is not None else 'no'}\n"
        f"odds: {'yes' if odds is not None else 'no'}\n"
    )
    print(f"\ndone {stamp}")


if __name__ == "__main__":
    main()
