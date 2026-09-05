"""
MLB Board — shared core.
De-vig, market consensus, HR model, team/park tables, name matching.
No I/O here beyond plain HTTP helpers, so it can be unit-tested offline.
"""
import csv
import difflib
import json
import math
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timedelta, timezone

import requests

UA = {"User-Agent": "mlb-board/1.0 (+github actions)"}


# ---------------------------------------------------------------- http

def get_json(url, params=None, tries=3, timeout=30, headers=None):
    """GET returning (json_or_None, status_code, response_headers)."""
    h = dict(UA)
    if headers:
        h.update(headers)
    last = 0
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=h, timeout=timeout)
            last = r.status_code
            if r.status_code == 200:
                return r.json(), 200, r.headers
            # 4xx other than 429 will not fix themselves
            if 400 <= r.status_code < 500 and r.status_code != 429:
                print(f"    HTTP {r.status_code} {url} :: {r.text[:220]}")
                return None, r.status_code, r.headers
            print(f"    HTTP {r.status_code} {url} (attempt {i+1}/{tries})")
        except Exception as e:
            print(f"    ERR {type(e).__name__}: {e} {url} (attempt {i+1}/{tries})")
            last = -1
        time.sleep(1.5 * (i + 1))
    return None, last, {}


# ---------------------------------------------------------------- de-vig

def shin_devig(quoted):
    """
    Shin (1993) de-vig for an n-outcome market.
    `quoted` = list of implied probabilities (1/decimal_odds), booksum > 1.
    Returns fair probabilities summing to 1.

    Shin models the overround as protection against insider trading, which
    puts proportionally more of the margin on longshots. Proportional
    de-vig (just dividing by the booksum) overstates longshot fair value —
    which matters a lot here, because every home-run prop is a longshot.
    """
    q = [max(x, 1e-9) for x in quoted]
    s = sum(q)
    if s <= 1.0 or len(q) < 2:
        return [x / s for x in q]

    def probs(z):
        z = min(max(z, 1e-9), 0.9999)
        out = []
        for qi in q:
            root = math.sqrt(max(z * z + 4.0 * (1.0 - z) * qi * qi / s, 0.0))
            out.append((root - z) / (2.0 * (1.0 - z)))
        return out

    lo, hi = 0.0, 0.9
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if sum(probs(mid)) > 1.0:
            lo = mid
        else:
            hi = mid
    p = probs((lo + hi) / 2.0)
    tot = sum(p)
    return [x / tot for x in p] if tot > 0 else [x / s for x in q]


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def american(dec):
    """Decimal -> American, for display."""
    if dec is None:
        return ""
    if dec >= 2.0:
        return f"+{round((dec - 1) * 100):d}"
    return f"-{round(100 / (dec - 1)):d}"


# ---------------------------------------------------------------- teams / parks

# MLB Stats API team id -> (abbrev, Odds API display name)
TEAMS = {
    108: ("LAA", "Los Angeles Angels"),
    109: ("ARI", "Arizona Diamondbacks"),
    110: ("BAL", "Baltimore Orioles"),
    111: ("BOS", "Boston Red Sox"),
    112: ("CHC", "Chicago Cubs"),
    113: ("CIN", "Cincinnati Reds"),
    114: ("CLE", "Cleveland Guardians"),
    115: ("COL", "Colorado Rockies"),
    116: ("DET", "Detroit Tigers"),
    117: ("HOU", "Houston Astros"),
    118: ("KC", "Kansas City Royals"),
    119: ("LAD", "Los Angeles Dodgers"),
    120: ("WSH", "Washington Nationals"),
    121: ("NYM", "New York Mets"),
    133: ("ATH", "Athletics"),
    134: ("PIT", "Pittsburgh Pirates"),
    135: ("SD", "San Diego Padres"),
    136: ("SEA", "Seattle Mariners"),
    137: ("SF", "San Francisco Giants"),
    138: ("STL", "St. Louis Cardinals"),
    139: ("TB", "Tampa Bay Rays"),
    140: ("TEX", "Texas Rangers"),
    141: ("TOR", "Toronto Blue Jays"),
    142: ("MIN", "Minnesota Twins"),
    143: ("PHI", "Philadelphia Phillies"),
    144: ("ATL", "Atlanta Braves"),
    145: ("CWS", "Chicago White Sox"),
    146: ("MIA", "Miami Marlins"),
    147: ("NYY", "New York Yankees"),
    158: ("MIL", "Milwaukee Brewers"),
}

# Approximate multi-year Statcast HOME RUN park factors, 100 = league average.
# These feed the reference model only, never the picks. Refresh yearly from
# baseballsavant.mlb.com/leaderboard/statcast-park-factors (stat = HR).
PARK_HR = {
    "CIN": 117, "NYY": 114, "COL": 112, "PHI": 112, "CWS": 111, "ATH": 110,
    "MIL": 108, "LAD": 107, "TEX": 106, "TOR": 105, "LAA": 104, "ARI": 103,
    "CHC": 103, "ATL": 103, "WSH": 103, "BAL": 102, "HOU": 101, "NYM": 100,
    "CLE": 100, "MIN": 99, "BOS": 98, "PIT": 96, "DET": 95, "SEA": 95,
    "SD": 95, "STL": 95, "TB": 99, "MIA": 93, "KC": 92, "SF": 90,
}


def norm_team(s):
    s = (s or "").lower()
    s = re.sub(r"[^a-z ]", "", s)
    return " ".join(s.split())


TEAM_BY_NAME = {}
for _tid, (_ab, _nm) in TEAMS.items():
    TEAM_BY_NAME[norm_team(_nm)] = _tid
# Odds API / statsapi spelling drift
for _alias, _tid in {
    "oakland athletics": 133, "athletics": 133, "las vegas athletics": 133,
    "st louis cardinals": 138, "saint louis cardinals": 138,
    "chicago white sox": 145, "washington nationals": 120,
    "arizona diamondbacks": 109, "cleveland guardians": 114,
}.items():
    TEAM_BY_NAME[norm_team(_alias)] = _tid


def team_id_from_name(name):
    n = norm_team(name)
    if n in TEAM_BY_NAME:
        return TEAM_BY_NAME[n]
    # last-word match ("Yankees", "Blue Jays")
    best, score = None, 0.0
    for k, tid in TEAM_BY_NAME.items():
        r = difflib.SequenceMatcher(None, n, k).ratio()
        if r > score:
            best, score = tid, r
    return best if score >= 0.62 else None


# ---------------------------------------------------------------- player names

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_person(name):
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace(".", " ").replace("'", "").replace("-", " ")
    s = re.sub(r"[^a-z ]", " ", s)
    parts = [p for p in s.split() if p and p not in SUFFIXES]
    return " ".join(parts)


def match_player(name, candidates):
    """
    candidates: dict normalised_name -> player_id, restricted to the two
    rosters actually in this game. Small candidate pool is what makes a
    string match safe here.
    """
    n = norm_person(name)
    if n in candidates:
        return candidates[n], 1.0
    # last name + first initial
    parts = n.split()
    if len(parts) >= 2:
        key = parts[0][0] + " " + parts[-1]
        for cn, pid in candidates.items():
            cp = cn.split()
            if len(cp) >= 2 and cp[0][0] + " " + cp[-1] == key:
                return pid, 0.9
    best, score = None, 0.0
    for cn, pid in candidates.items():
        r = difflib.SequenceMatcher(None, n, cn).ratio()
        if r > score:
            best, score = pid, r
    return (best, score) if score >= 0.82 else (None, score)


# ---------------------------------------------------------------- HR model

LG_HR_PER_PA_FALLBACK = 0.0310   # ~3.1% of PA end in a home run
BAT_SHRINK_PA = 250.0
PIT_SHRINK_BF = 300.0
STARTER_SHARE = 0.62             # share of a lineup's PA that face the starter
SLOT_PA = {1: 4.62, 2: 4.52, 3: 4.42, 4: 4.32, 5: 4.20,
           6: 4.08, 7: 3.96, 8: 3.84, 9: 3.72}
DEFAULT_PA = 4.15


def hr_model_prob(bat_hr, bat_pa, pit_hr, pit_bf, park_idx, lg_rate, slot=None):
    """
    P(batter hits >= 1 HR) via a shrunk log5 rate combination.

    Deliberately simple. Its job is to give the board a second opinion and
    flag disagreement, not to price anything. Every backtest in the football
    build said the same thing: the model loses to the closing line, so the
    closing line prices the board.
    """
    lg = lg_rate or LG_HR_PER_PA_FALLBACK
    rb = (bat_hr + BAT_SHRINK_PA * lg) / (bat_pa + BAT_SHRINK_PA)
    rp = (pit_hr + PIT_SHRINK_BF * lg) / (pit_bf + PIT_SHRINK_BF)
    # starter for ~62% of PA, league-average relief for the rest
    rp_eff = STARTER_SHARE * rp + (1.0 - STARTER_SHARE) * lg
    rate = (rb * rp_eff / lg) * (park_idx / 100.0)
    rate = min(max(rate, 0.0005), 0.25)
    pa = SLOT_PA.get(slot, DEFAULT_PA)
    return 1.0 - (1.0 - rate) ** pa


# ---------------------------------------------------------------- misc

def eastern_date(dt_utc=None):
    """US Eastern calendar date. MLB's 'game day' is an Eastern date."""
    dt = dt_utc or datetime.now(timezone.utc)
    # EDT (Mar-Nov) is UTC-4; EST is UTC-5. Season runs inside EDT.
    off = -4 if 3 <= dt.month <= 11 else -5
    return (dt + timedelta(hours=off)).date()


def date_from_iso(s):
    from datetime import date as _d
    y, m, d = (s or "1970-01-01").split("-")[:3]
    return _d(int(y), int(m), int(d))


def ensure_csv(path, header):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def append_csv(path, header, rows):
    ensure_csv(path, header)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow([r.get(h, "") for h in header])


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, header, rows):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(h, "") for h in header])
