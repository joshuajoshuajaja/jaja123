"""
MLB Board — closing line snapshot.

Runs 22:45 UTC (6:45pm US Eastern), minutes before the bulk of the night
slate goes off. Re-prices everything logged on today's board and records
closing line value against it.

Why this exists: the football build could measure everything except whether
the prices it took were actually good, because nothing ever captured the
close. CLV is the only honest read on a betting system's edge, and it is
readable in weeks rather than the years a profit/loss sample needs.

CLV here = best_price_taken x closing_fair_probability - 1.
Positive means the price beat where the market finished. Day games have
already started by run time and simply don't get a closing price.
"""
import os
import sys
from datetime import datetime, timezone

import core as C
import feeds as F
import board as B

PICKS = B.PICKS        # data/mlb/picks.csv


def closing_hr_probs(od, ev_id):
    """{player_name: fair P(HR)} from the current market."""
    eo = od.hr_props(ev_id)
    if not eo:
        return {}
    quotes = {}
    for bk in eo.get("bookmakers", []) or []:
        for mk in bk.get("markets", []) or []:
            if mk.get("key") != "batter_home_runs":
                continue
            for oc in mk.get("outcomes", []) or []:
                if abs(float(oc.get("point") or 0.5) - 0.5) > 1e-6:
                    continue
                who, side = oc.get("description"), (oc.get("name") or "").lower()
                if who and side in ("over", "under") and oc.get("price"):
                    (quotes.setdefault(who, {}).setdefault(bk.get("key"), {})
                     )[side] = float(oc["price"])
    out = {}
    for who, books in quotes.items():
        two = [(b["over"], b["under"]) for b in books.values()
               if "over" in b and "under" in b]
        overs = [b["over"] for b in books.values() if "over" in b]
        p = B.consensus_two_way(two) if len(two) >= 3 else None
        if p is None and overs:
            p = 0.93 / C.median(overs)      # same margin assumption as the board
        if p:
            out[C.norm_person(who)] = p
    return out


def closing_ml_probs(od):
    """{(home_name, away_name, team_name): fair win prob} from the current market."""
    out = {}
    for ev in od.moneylines():
        h, a = ev.get("home_team"), ev.get("away_team")
        per = []
        for bk in ev.get("bookmakers", []) or []:
            m = next((m for m in bk.get("markets", []) or []
                      if m.get("key") == "h2h"), None)
            if not m:
                continue
            o = {x.get("name"): float(x.get("price") or 0)
                 for x in m.get("outcomes", []) or []}
            if o.get(h, 0) > 1 and o.get(a, 0) > 1:
                fair = C.shin_devig([1.0 / o[h], 1.0 / o[a]])
                per.append(fair[0])
        if not per:
            continue
        ph = C.median(per)
        hid, aid = C.team_id_from_name(h), C.team_id_from_name(a)
        if hid:
            out[C.TEAMS[hid][0]] = out.get(C.TEAMS[hid][0], []) + [ph]
        if aid:
            out[C.TEAMS[aid][0]] = out.get(C.TEAMS[aid][0], []) + [1 - ph]
    return {k: v[0] for k, v in out.items()}


def main():
    api = os.environ.get("ODDS_API_KEY", "").strip()
    if not api:
        print("FATAL: ODDS_API_KEY is not set")
        sys.exit(1)

    rows = C.read_csv(PICKS)
    if not rows:
        print("no picks logged yet — nothing to snapshot")
        return
    ds = C.eastern_date().isoformat()
    todo = [r for r in rows if r.get("board_date") == ds
            and not r.get("closing_price")]
    print(f"== closing snapshot for {ds}: {len(todo)} open picks ==")
    if not todo:
        print("nothing open — either no board today, or already snapshotted")
        return

    od = F.Odds(api)
    events = od.events()
    games = F.schedule(ds)
    linked = F.link_events_to_games(events, games)
    ev_for_game = {str(g["gamePk"]): ev for ev, g in linked}
    now = datetime.now(timezone.utc)

    # moneylines: one call covers the whole slate
    ml_needed = any(r["market"] == "ML" for r in todo)
    ml_fair = closing_ml_probs(od) if ml_needed else {}
    print(f"  moneyline closing prices for {len(ml_fair)} teams")

    # home runs: one call per game that still has open legs and hasn't started
    hr_games = sorted({r["game_pk"] for r in todo if r["market"] == "HR"})
    hr_fair, started = {}, 0
    for pk in hr_games:
        ev = ev_for_game.get(pk)
        if not ev:
            continue
        if F.event_start(ev) <= now:
            started += 1
            continue                       # already underway, no close to read
        hr_fair[pk] = closing_hr_probs(od, ev["id"])
    print(f"  home run closing prices for {len(hr_fair)} games "
          f"({started} already started)")

    done = missed = 0
    clvs = []
    for r in rows:
        if r.get("board_date") != ds or r.get("closing_price"):
            continue
        p = None
        if r["market"] == "ML":
            p = ml_fair.get(r.get("team"))
        else:
            p = hr_fair.get(r.get("game_pk"), {}).get(
                C.norm_person(r.get("selection", "")))
        if not p or p <= 0:
            missed += 1
            continue
        taken = float(r["best_price"])
        clv = taken * p - 1.0
        r["closing_price"] = f"{1.0 / p:.3f}"
        r["clv_pct"] = f"{clv * 100:.2f}"
        clvs.append(clv)
        done += 1

    hdr = B.PICK_HEADER + ["selection_id"]
    C.write_csv(PICKS, hdr, rows)

    print(f"  priced {done}, missed {missed}")
    if clvs:
        avg = sum(clvs) / len(clvs)
        beat = sum(1 for c in clvs if c > 0)
        print(f"  average CLV {avg*100:+.2f}% · beat the close on "
              f"{beat}/{len(clvs)} ({beat/len(clvs)*100:.0f}%)")

    # running total across every snapshot so far
    hist = [float(r["clv_pct"]) / 100 for r in rows
            if r.get("clv_pct") not in (None, "")]
    if hist:
        b = sum(1 for c in hist if c > 0)
        print(f"  LIFETIME: {len(hist)} priced picks, average CLV "
              f"{sum(hist)/len(hist)*100:+.2f}%, beat the close "
              f"{b}/{len(hist)} ({b/len(hist)*100:.0f}%)")
        print("  A system with a real edge shows persistently positive CLV. "
              "If this sits negative after a few hundred picks, the board is "
              "not beating the market and the honest move is line shopping "
              "alone.")
    print(f"  odds credits remaining: {od.remaining}")


if __name__ == "__main__":
    main()
