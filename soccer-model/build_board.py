#!/usr/bin/env python3
"""Build the daily board from the current odds snapshot."""
from __future__ import annotations

import json
import sys
from datetime import timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, "soccer-model")
from model.db import load, SystemData, DIV_NAMES          # noqa: E402
from model.dc import DixonColes, score_matrix             # noqa: E402
from model.live import (SPORT_TO_DIV, map_teams, consensus_1x2,     # noqa: E402
                        consensus_totals, implied_matrix)

HALF_LIFE, RIDGE = 330.0, 0.25

# "Tonight" is anchored to a fixed end-of-slate, not a rolling window, so a
# re-run later in the evening covers the same night rather than creeping into
# tomorrow. 10am Singapore sits after the last European night kick-off.
SGT_TZ = timezone(timedelta(hours=8))
SLATE_END_HOUR = 10
LEAD_MINUTES = 10      # need time to actually place the bet
MAX_HOURS = 20         # safety cap


def slate_window(now=None):
    """(start, end) in UTC: games not yet kicked off, up to end of tonight."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    local = now.tz_convert(SGT_TZ)
    end_local = local.normalize() + pd.Timedelta(hours=SLATE_END_HOUR)
    if end_local <= local:
        end_local += pd.Timedelta(days=1)
    start = now + pd.Timedelta(minutes=LEAD_MINUTES)
    end = min(end_local.tz_convert("UTC"), now + pd.Timedelta(hours=MAX_HOURS))
    return start, end
K = np.arange(11)
GD = np.subtract.outer(K, K)
TOT = np.add.outer(K, K)

BOOK_NAMES = {
    "onexbet": "1xBet", "betfair_ex_eu": "Betfair Exchange", "pinnacle": "Pinnacle",
    "williamhill": "William Hill", "betclic_fr": "Betclic", "winamax_fr": "Winamax",
    "winamax_de": "Winamax DE", "marathonbet": "Marathonbet", "unibet_eu": "Unibet",
    "nordicbet": "NordicBet", "betsson": "Betsson", "coolbet": "Coolbet",
    "leovegas_se": "LeoVegas", "matchbook": "Matchbook", "everygame": "Everygame",
    "betonlineag": "BetOnline", "mybookieag": "MyBookie", "gtbets": "GTbets",
    "betanysports": "BetAnySports", "codere_it": "Codere", "suprabets": "Suprabets",
    "tipico_de": "Tipico", "sport888": "888sport", "casumo": "Casumo",
}
book = lambda b: BOOK_NAMES.get(b, str(b).replace("_", " ").title())


def fair_price(p: float) -> float:
    return float("inf") if p <= 0 else 1.0 / p


def ah_split(M, h):
    q = round(h * 4)
    if q % 2 != 0:
        a = ah_split(M, (q - 1) / 4.0)
        b = ah_split(M, (q + 1) / 4.0)
        return tuple(0.5 * (x + y) for x, y in zip(a, b))
    adj = GD + h
    return (float(M[adj > 0].sum()), float(M[np.isclose(adj, 0)].sum()),
            float(M[adj < 0].sum()))


def _find(*names):
    import os
    for n in names:
        if os.path.exists(n):
            return n
    raise FileNotFoundError(names[0])


def main() -> None:
    matches = load(_find("data/raw/matches.parquet", "data/matches.parquet"))
    odds = pd.read_csv(_find("data/raw/odds_history.csv.gz", "data/odds_history.csv.gz"))
    odds["ct"] = pd.to_datetime(odds.commence_time, utc=True, errors="coerce")

    # keep only the most recent snapshot per event/book/market/outcome
    odds = (odds.sort_values("snapshot_utc")
                .drop_duplicates(subset=["event_id", "book", "market", "outcome", "point"],
                                 keep="last"))
    now = pd.Timestamp.now(tz="UTC")
    start, end = slate_window(now)
    started = odds[odds.ct < start].event_id.nunique()
    live = odds[(odds.ct >= start) & (odds.ct <= end)]
    print(f"slate window {start.tz_convert(SGT_TZ):%a %d %b %H:%M} -> "
          f"{end.tz_convert(SGT_TZ):%a %d %b %H:%M} SGT")
    print(f"{live.event_id.nunique()} fixtures still to kick off "
          f"({started} already started or finished, skipped)")

    recent = matches[matches["date"] >= matches["date"].max() - pd.Timedelta(days=450)]
    fitted: dict[str, tuple] = {}
    fixtures = []

    for sport, gs in live.groupby("sport_key"):
        div = SPORT_TO_DIV.get(sport)
        if div is None:
            continue
        system = matches.loc[matches.Div == div, "system"].dropna()
        if system.empty:
            continue
        system = system.iat[0]

        if system not in fitted:
            sd = SystemData(matches[matches.system == system])
            asof = np.datetime64(pd.Timestamp.now().date())
            train = sd.dates < asof
            m = DixonColes(ridge=RIDGE).fit(
                sd.hi[train], sd.ai[train], sd.di[train],
                sd.hg[train], sd.ag[train], sd.weights(asof, HALF_LIFE)[train],
                len(sd.teams), len(sd.divs))
            fitted[system] = (sd, m)
            print(f"  fitted {system}: {train.sum():,} matches, rho {m.rho_:+.3f}")
        sd, m = fitted[system]

        sub = recent[recent.Div == div]
        cands = sorted(set(sub.HomeTeam) | set(sub.AwayTeam))
        feed = sorted(set(gs.home_team) | set(gs.away_team))
        tmap, unmapped = map_teams(feed, cands)
        if unmapped:
            print(f"  ! {sport}: unmapped {unmapped}")

        for eid, g in gs.groupby("event_id"):
            h_feed, a_feed = g.home_team.iat[0], g.away_team.iat[0]
            hn, an = tmap.get(h_feed), tmap.get(a_feed)
            if hn is None or an is None:
                continue
            fair, best, nbooks = consensus_1x2(g, h_feed, a_feed)
            if fair is None:
                continue
            tot = consensus_totals(g)
            M = implied_matrix(fair, tot, rho=float(m.rho_))
            if M is None:
                continue

            # our own model's view, for the disagreement column
            ti = sd.teams.get_indexer([hn, an])
            di = sd.divs.get_indexer([div])[0] if div in sd.divs else 0
            if ti[0] < 0 or ti[1] < 0 or di < 0:
                own = None
            else:
                lh, la = m.rates(np.array([ti[0]]), np.array([ti[1]]), np.array([di]))
                own = score_matrix(float(lh[0]), float(la[0]), float(m.rho_))

            fixtures.append(dict(
                event_id=eid, sport=sport, div=div,
                league=DIV_NAMES.get(div, div),
                kickoff=g.ct.iat[0].isoformat(),
                home=h_feed, away=a_feed, home_fd=hn, away_fd=an,
                nbooks=nbooks, M=M, own=own, fair=fair, best=best, totals=tot))

    fixtures.sort(key=lambda f: f["kickoff"])
    print(f"priced {len(fixtures)} fixtures")
    return fixtures


if __name__ == "__main__":
    fx = main()
    import pickle
    with open("fixtures.pkl", "wb") as fh:
        pickle.dump(fx, fh)
    print("saved fixtures.pkl")

