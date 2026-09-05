#!/usr/bin/env python3
"""
Daily data pull for the soccer model.  v2 - hardened.

Design rule: football-data.co.uk is the only source allowed to fail the run.
Everything else is best-effort and reports itself in the log. Whatever landed
gets committed either way.
"""
from __future__ import annotations

import io
import os
import re
import sys
import time
import traceback
import datetime as dt
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (compatible; soccer-model/2.0)"}
TIMEOUT = 45
WARNINGS: list[str] = []


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  !! {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

MAIN_DIVS = [
    "E0", "E1", "E2", "E3", "EC",
    "SC0", "SC1", "SC2", "SC3",
    "D1", "D2", "I1", "I2", "SP1", "SP2", "F1", "F2",
    "N1", "B1", "P1", "T1", "G1",
]

EXTRA_COUNTRIES = [
    "ARG", "AUT", "BRA", "CHN", "DNK", "FIN", "IRL", "JPN",
    "MEX", "NOR", "POL", "ROU", "RUS", "SWE", "SWZ", "USA",
]

N_SEASONS = 12

# Columns worth keeping. Everything else is one of ~30 individual bookmakers
# we don't need - dropping them keeps the committed file small enough that
# GitHub doesn't complain.
CORE_COLS = {
    "Div", "Date", "Time", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
    "SeasonCode", "Source", "Country", "League", "Season",
}

# B365/Pinnacle/BetFair + market max & average, opening and closing ("C").
ODDS_RE = re.compile(
    r"^(B365|PS|P|BF|Max|Avg)C?("
    r"H|D|A"                       # 1X2
    r"|>2\.5|<2\.5"                # totals
    r"|AHH|AHA"                    # asian handicap prices
    r")$"
)
LINE_COLS = {"AHh", "AHCh", "B365AHh", "PAHh", "MaxAHh", "AvgAHh"}

NUMERIC_HINT = re.compile(r"^(FT|HT)(HG|AG)$|^(HS|AS|HST|AST|HC|AC|HF|AF|HY|AY|HR|AR)$")


def season_codes(n: int = N_SEASONS) -> list[str]:
    today = dt.date.today()
    start_year = today.year if today.month >= 7 else today.year - 1
    return [f"{(start_year - i) % 100:02d}{(start_year - i + 1) % 100:02d}" for i in range(n)]


def _get(url: str, tries: int = 3) -> bytes | None:
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=TIMEOUT)
            if r.status_code == 200 and r.content:
                return r.content
            if r.status_code == 404:
                return None
        except requests.RequestException as e:
            if attempt == tries - 1:
                warn(f"{url}: {e}")
        time.sleep(1.5 * (attempt + 1))
    return None


def _read_csv(blob: bytes) -> pd.DataFrame | None:
    # utf-8-sig FIRST: these files carry a UTF-8 BOM, and decoding them as
    # latin-1 turns the first column name into "ï»¿Country",
    # which silently loses the Country column on the extra-league files.
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(blob), encoding=enc,
                             on_bad_lines="skip", low_memory=False)
        except Exception:
            continue
        df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
        df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
        if "Date" in df.columns:
            df = df[df["Date"].notna()]
        return df if len(df) else None
    return None


def _trim(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns
            if c in CORE_COLS or c in LINE_COLS or ODDS_RE.match(str(c))]
    return df[keep] if keep else df


def fetch_football_data() -> pd.DataFrame:
    frames, seasons = [], season_codes()
    print(f"football-data.co.uk main: {len(MAIN_DIVS)} divisions x {len(seasons)} seasons",
          flush=True)
    for div in MAIN_DIVS:
        got = 0
        for season in seasons:
            blob = _get(f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv")
            if not blob:
                continue
            df = _read_csv(blob)
            if df is None:
                continue
            df = _trim(df)
            df["Div"] = div
            df["SeasonCode"] = season
            df["Source"] = "main"
            frames.append(df)
            got += 1
        print(f"  {div:<4} {got}/{len(seasons)} seasons", flush=True)

    print(f"football-data.co.uk extra: {len(EXTRA_COUNTRIES)} countries", flush=True)
    for ctry in EXTRA_COUNTRIES:
        blob = _get(f"https://www.football-data.co.uk/new/{ctry}.csv")
        if not blob:
            print(f"  {ctry:<4} unavailable", flush=True)
            continue
        df = _read_csv(blob)
        if df is None:
            continue
        df = df.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam",
                                "HG": "FTHG", "AG": "FTAG", "Res": "FTR"})
        # League names collide across countries ("Super League" is both China
        # and Switzerland; "Superliga" is Denmark and Romania) and arrive with
        # stray whitespace, so key on country + league, always.
        league = df["League"].astype(str).str.strip() if "League" in df.columns else ctry
        country = df["Country"].astype(str).str.strip() if "Country" in df.columns else ctry
        df["Div"] = country.str.upper().str.slice(0, 3) + ": " + league
        df = _trim(df)
        df["Source"] = "extra"
        frames.append(df)
        print(f"  {ctry:<4} {len(df)} rows", flush=True)

    if not frames:
        raise RuntimeError("football-data.co.uk returned nothing at all")

    out = pd.concat(frames, ignore_index=True, sort=False)

    # --- normalise dtypes. This is the step that used to blow the run up:
    # concatenating 12 seasons of slightly different schemas leaves object
    # columns holding a mix of floats and stray strings, which parquet refuses.
    for col in out.columns:
        if col in {"Div", "HomeTeam", "AwayTeam", "FTR", "HTR", "Time",
                   "Date", "SeasonCode", "Source", "Country", "League", "Season"}:
            out[col] = out[col].astype("string")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # parsed ISO date - football-data uses both dd/mm/yy and dd/mm/yyyy
    d = pd.to_datetime(out["Date"], format="%d/%m/%Y", errors="coerce")
    d2 = pd.to_datetime(out["Date"], format="%d/%m/%y", errors="coerce")
    out["date"] = d.fillna(d2)

    before = len(out)
    out = out.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    out = out.drop_duplicates(subset=["Div", "date", "HomeTeam", "AwayTeam"], keep="last")
    print(f"\ncleaned: {before:,} -> {len(out):,} rows "
          f"({out['Div'].nunique()} divisions, "
          f"{out['date'].min():%Y-%m-%d} to {out['date'].max():%Y-%m-%d})", flush=True)
    return out.sort_values("date").reset_index(drop=True)


def fetch_clubelo() -> pd.DataFrame | None:
    for scheme in ("https", "http"):
        for days_back in (0, 1, 2):
            day = (dt.date.today() - dt.timedelta(days=days_back)).isoformat()
            blob = _get(f"{scheme}://api.clubelo.com/{day}", tries=2)
            if not blob:
                continue
            try:
                df = pd.read_csv(io.BytesIO(blob))
                if len(df) > 100:
                    return df
            except Exception as e:
                warn(f"clubelo parse ({scheme}): {e}")
    warn("clubelo unavailable")
    return None


# ---------------------------------------------------------------------------
# Understat - match-level expected goals for the top five leagues
# ---------------------------------------------------------------------------
# Understat embeds its data as a hex-escaped JSON string inside a <script> tag
# rather than serving an API, so we pull the page and unescape it. No scraping
# library needed, and nothing to break when a package version moves.

UNDERSTAT_LEAGUES = ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1", "RFPL"]
UNDERSTAT_FIRST_SEASON = 2014

_US_RE = re.compile(r"var\s+datesData\s*=\s*JSON\.parse\('(.*?)'\)\s*;", re.S)


def _understat_season_years() -> list[int]:
    today = dt.date.today()
    latest = today.year if today.month >= 7 else today.year - 1
    return list(range(UNDERSTAT_FIRST_SEASON, latest + 1))


def fetch_xg() -> pd.DataFrame | None:
    import json

    rows = []
    for league in UNDERSTAT_LEAGUES:
        got = 0
        for year in _understat_season_years():
            blob = _get(f"https://understat.com/league/{league}/{year}", tries=2)
            if not blob:
                continue
            m = _US_RE.search(blob.decode("utf-8", errors="replace"))
            if not m:
                continue
            try:
                # \xHH escapes are UTF-8 *bytes*, so unicode_escape alone turns
                # "Munchen" with an umlaut into mojibake - round-trip through
                # latin-1 to recover the real characters.
                s = m.group(1).encode("utf-8").decode("unicode_escape")
                s = s.encode("latin-1", "replace").decode("utf-8", "replace")
                data = json.loads(s)
            except Exception as e:
                warn(f"understat {league} {year}: parse failed ({e})")
                continue
            for g in data:
                if not g.get("isResult"):
                    continue          # fixture not played yet
                try:
                    rows.append({
                        "league": league,
                        "season": year,
                        "datetime": g.get("datetime"),
                        "home": g["h"]["title"],
                        "away": g["a"]["title"],
                        "hg": int(g["goals"]["h"]),
                        "ag": int(g["goals"]["a"]),
                        "hxg": float(g["xG"]["h"]),
                        "axg": float(g["xG"]["a"]),
                    })
                except (KeyError, TypeError, ValueError):
                    continue
            got += 1
            time.sleep(0.8)           # be polite; the site is a free resource
        print(f"  xG {league:<12} {got}/{len(_understat_season_years())} seasons", flush=True)

    if not rows:
        warn("understat returned nothing - xG unavailable this run")
        return None

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).drop_duplicates(
        subset=["league", "date", "home", "away"], keep="last")
    return df.sort_values("date").reset_index(drop=True)


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
        print("ODDS_API_KEY not set - skipping live odds (expected for now)", flush=True)
        return None

    rows, used, snap = [], "?", dt.datetime.now(dt.timezone.utc).isoformat()
    for sport in ODDS_SPORTS:
        url = (f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
               f"?apiKey={key}&regions=eu,uk&markets=h2h,totals,spreads"
               f"&oddsFormat=decimal&dateFormat=iso")
        try:
            r = requests.get(url, timeout=TIMEOUT)
        except requests.RequestException as e:
            warn(f"odds {sport}: {e}")
            continue
        if r.status_code == 401:
            warn("odds API rejected the key - stopping")
            break
        if r.status_code != 200:
            continue
        used = r.headers.get("x-requests-used", used)
        for ev in r.json():
            for bk in ev.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    for oc in mk.get("outcomes", []):
                        rows.append({
                            "snapshot_utc": snap,
                            "sport_key": ev.get("sport_key"),
                            "event_id": ev.get("id"),
                            "commence_time": ev.get("commence_time"),
                            "home_team": ev.get("home_team"),
                            "away_team": ev.get("away_team"),
                            "book": bk.get("key"),
                            "market": mk.get("key"),
                            "outcome": oc.get("name"),
                            "point": oc.get("point"),
                            "price": oc.get("price"),
                        })
    print(f"odds rows: {len(rows)}   credits used this month: {used}", flush=True)
    return pd.DataFrame(rows) if rows else None


# ---------------------------------------------------------------------------

def main() -> int:
    started = dt.datetime.now(dt.timezone.utc)
    print(f"=== fetch started {started:%Y-%m-%d %H:%M:%S} UTC ===\n", flush=True)

    # --- the one thing that must work -------------------------------------
    try:
        matches = fetch_football_data()
    except Exception:
        traceback.print_exc()
        print("\nFATAL: could not build the match table.", file=sys.stderr)
        return 1

    # CSV first - it cannot fail on dtypes, so the run always leaves something behind.
    csv_path = RAW / "matches.csv.gz"
    matches.to_csv(csv_path, index=False, compression="gzip")
    print(f"wrote {csv_path.name}  ({csv_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    try:
        matches.to_parquet(RAW / "matches.parquet", index=False)
        print(f"wrote matches.parquet ({(RAW / 'matches.parquet').stat().st_size / 1e6:.1f} MB)",
              flush=True)
    except Exception as e:
        warn(f"parquet write skipped: {e}")

    # --- best effort from here on -----------------------------------------
    elo = None
    try:
        elo = fetch_clubelo()
        if elo is not None:
            elo.to_csv(RAW / "clubelo.csv", index=False)
            print(f"wrote clubelo.csv ({len(elo):,} clubs)", flush=True)
    except Exception as e:
        warn(f"clubelo step: {e}")

    xg = None
    try:
        print("\nunderstat xG:", flush=True)
        xg = fetch_xg()
        if xg is not None:
            xg.to_csv(RAW / "xg.csv.gz", index=False, compression="gzip")
            # distinct team names, so the modelling side can build the mapping
            # onto football-data's names without another round trip
            names = (pd.concat([xg[["league", "home"]].rename(columns={"home": "team"}),
                                xg[["league", "away"]].rename(columns={"away": "team"})])
                     .drop_duplicates().sort_values(["league", "team"]))
            names.to_csv(RAW / "xg_teams.csv", index=False)
            print(f"wrote xg.csv.gz ({len(xg):,} matches, "
                  f"{xg['date'].min():%Y-%m-%d}..{xg['date'].max():%Y-%m-%d}, "
                  f"{len(names)} team-league pairs)", flush=True)
    except Exception as e:
        warn(f"xg step: {e}")

    odds = None
    try:
        odds = fetch_odds()
        if odds is not None:
            hist = RAW / "odds_history.csv.gz"
            if hist.exists():
                prev = pd.read_csv(hist)
                odds = pd.concat([prev, odds], ignore_index=True, sort=False)
                ct = pd.to_datetime(odds["commence_time"], errors="coerce", utc=True)
                cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=90)
                odds = odds[ct.isna() | (ct >= cutoff)]
            odds.to_csv(hist, index=False, compression="gzip")
            print(f"wrote odds_history.csv.gz ({len(odds):,} rows)", flush=True)
    except Exception as e:
        warn(f"odds step: {e}")

    per_div = matches.groupby("Div").size().sort_values(ascending=False)
    (RAW / "MANIFEST.txt").write_text(
        f"last_run_utc: {started.isoformat(timespec='seconds')}\n"
        f"matches_rows: {len(matches)}\n"
        f"divisions: {matches['Div'].nunique()}\n"
        f"date_range: {matches['date'].min():%Y-%m-%d} .. {matches['date'].max():%Y-%m-%d}\n"
        f"clubelo: {'yes' if elo is not None else 'no'}\n"
        f"xg_matches: {0 if xg is None else len(xg)}\n"
        f"odds: {'yes' if odds is not None else 'no'}\n"
        f"warnings: {len(WARNINGS)}\n"
        + "".join(f"  - {w}\n" for w in WARNINGS)
        + "\nrows per division:\n"
        + "".join(f"  {d:<28} {n}\n" for d, n in per_div.items())
    )

    took = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
    print(f"\n=== done in {took:.0f}s, {len(WARNINGS)} warning(s) ===", flush=True)
    for w in WARNINGS:
        print(f"  warning: {w}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

