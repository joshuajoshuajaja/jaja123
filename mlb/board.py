"""
MLB Board — daily build.

Runs 14:00 UTC = 10pm Singapore = 10am US Eastern, ahead of that day's slate.

Output
  2 x 3-leg home-run parlays  (the likeliest bats)
  2 x 4-leg home-run parlays  (reaching further down for price)
  Moneyline pick for every game on the slate

Pricing philosophy, carried over from the football build: the board is
priced off market consensus, not off the model. The model is computed,
printed next to every pick, and never allowed to choose one.
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import core as C
import feeds as F

DATA = "data/mlb"      # namespaced: the football board owns data/
PICKS = f"{DATA}/picks.csv"
RESULTS = f"{DATA}/results.csv"
BOARD_MD = f"{DATA}/board_latest.md"

PICK_HEADER = [
    "board_date", "logged_at", "market", "game_pk", "matchup", "selection",
    "team", "opp_sp", "best_price", "best_book", "fair_prob", "model_prob",
    "n_books", "edge_pct", "parlay", "closing_price", "clv_pct", "result",
]
RESULT_HEADER = ["board_date", "market", "matchup", "selection",
                 "best_price", "fair_prob", "result", "graded_at"]

MIN_BOOKS = 4
MIN_PRICE = 2.20
MAX_PRICE = 15.00
PARLAY_SHAPE = [("Parlay 1", 3), ("Parlay 2", 3), ("Parlay 3", 4), ("Parlay 4", 4)]


# ================================================================ consensus

def consensus_two_way(pairs):
    """
    pairs: list of (over_decimal, under_decimal) per book, both sides present.
    Returns median Shin-de-vigged P(over).
    """
    ps = []
    for o, u in pairs:
        if not o or not u or o <= 1.0 or u <= 1.0:
            continue
        fair = C.shin_devig([1.0 / o, 1.0 / u])
        ps.append(fair[0])
    return C.median(ps) if ps else None


def collect_hr_selections(ev_odds, game, rosters, hitting, pitching, lg_rate):
    """One dict per player with a home-run price in this game."""
    if not ev_odds:
        return []
    park = C.PARK_HR.get(C.TEAMS.get(game["home_id"], ("", ""))[0], 100)

    # lineup slots, when MLB has posted them (often not yet at 10am ET)
    slots = {}
    for side in ("home", "away"):
        for i, p in enumerate((game.get("lineups") or {})
                              .get(f"{side}Players", []) or []):
            if p.get("id"):
                slots[p["id"]] = i + 1

    # player -> {book: {"over": price, "under": price}}
    quotes = {}
    for bk in ev_odds.get("bookmakers", []) or []:
        bkey = bk.get("key", "?")
        for mk in bk.get("markets", []) or []:
            if mk.get("key") != "batter_home_runs":
                continue
            for oc in mk.get("outcomes", []) or []:
                if abs(float(oc.get("point") or 0.5) - 0.5) > 1e-6:
                    continue          # skip alt lines (1.5 HR etc)
                who = oc.get("description")
                side = (oc.get("name") or "").lower()
                price = oc.get("price")
                if not who or side not in ("over", "under") or not price:
                    continue
                quotes.setdefault(who, {}).setdefault(bkey, {})[side] = float(price)

    home_ab = C.TEAMS.get(game["home_id"], ("?",))[0]
    away_ab = C.TEAMS.get(game["away_id"], ("?",))[0]
    cand = dict(rosters.get(game["home_id"], {}))
    cand.update(rosters.get(game["away_id"], {}))
    home_ids = set(rosters.get(game["home_id"], {}).values())

    out = []
    for who, books in quotes.items():
        overs = [b["over"] for b in books.values() if "over" in b]
        two_way = [(b["over"], b["under"]) for b in books.values()
                   if "over" in b and "under" in b]
        if len(overs) < MIN_BOOKS:
            continue
        best_price = max(overs)
        best_book = max(books.items(), key=lambda kv: kv[1].get("over", 0))[0]
        fair = consensus_two_way(two_way) if len(two_way) >= 3 else None

        pid, conf = C.match_player(who, cand)
        if pid is None:
            print(f"    unmatched batter '{who}' in {away_ab}@{home_ab} "
                  f"(best ratio {conf:.2f})")
        is_home = pid in home_ids if pid else None
        team = home_ab if is_home else away_ab
        opp_sp = game["away_sp"] if is_home else game["home_sp"]
        opp_sp_name = (game["away_sp_name"] if is_home
                       else game["home_sp_name"]) or "TBD"

        bs = hitting.get(pid, {}) if pid else {}
        ps = pitching.get(opp_sp, {}) if opp_sp else {}
        model = C.hr_model_prob(
            bat_hr=float(bs.get("homeRuns") or 0),
            bat_pa=float(bs.get("plateAppearances") or 0),
            pit_hr=float(ps.get("homeRuns") or 0),
            pit_bf=float(ps.get("battersFaced") or 0),
            park_idx=park,
            lg_rate=lg_rate,
            slot=slots.get(pid),
        )
        out.append({
            "player": who, "player_id": pid, "team": team,
            "matchup": f"{away_ab} @ {home_ab}",
            "game_pk": game["gamePk"], "opp_sp": opp_sp_name,
            "best_price": best_price, "best_book": best_book,
            "n_books": len(overs), "two_way": len(two_way),
            "median_over": C.median(overs),
            "fair": fair, "fair_estimated": False,
            "model": model, "park": park,
            "slot": slots.get(pid),
        })
    return out


def fill_estimated_fair(sels):
    """
    Books that quote home runs one-sided can't be de-vigged directly.
    Measure the day's own typical fair/implied ratio from the two-sided
    selections and apply it to the rest, rather than guessing a margin.
    """
    ratios = []
    for s in sels:
        if s["fair"] and s["median_over"]:
            ratios.append(s["fair"] * s["median_over"])
    ratio = C.median(ratios) or 0.93
    n = 0
    for s in sels:
        if s["fair"] is None and s["median_over"]:
            s["fair"] = min(0.98, ratio / s["median_over"])
            s["fair_estimated"] = True
            n += 1
    print(f"  fair-value margin ratio {ratio:.4f}; "
          f"estimated for {n} one-sided selections")
    return ratio


# ================================================================ parlays

def build_parlays(pool):
    """
    Rank by consensus hit probability, then slice.

    Ranking by *edge* would silently fill the board with longshots — a 3%
    edge is far easier to find at 12.00 than at 3.50. Ranking by hit
    probability keeps the 3-leggers genuinely the likeliest bats, which is
    what they're for.
    """
    ranked = sorted(pool, key=lambda s: -s["fair"])
    parlays, seen, cur = [], set(), 0
    for name, n in PARLAY_SHAPE:
        legs, games, wrapped = [], set(), False
        while len(legs) < n:
            if cur >= len(ranked):
                if wrapped:
                    break                # genuinely nothing left to take
                cur, wrapped = 0, True   # reuse strong bats rather than drop legs
            s = ranked[cur]
            cur += 1
            if s["game_pk"] in games:
                continue                 # never two legs from one game
            legs.append(s)
            games.add(s["game_pk"])
        key = frozenset(id(s) for s in legs)
        if not legs or key in seen:
            continue                     # a thin slate is better served by
        seen.add(key)                    # fewer tickets than identical ones
        parlays.append((name, legs))
    return parlays


def parlay_price(legs):
    p = 1.0
    for s in legs:
        p *= s["best_price"]
    return p


def parlay_prob(legs):
    p = 1.0
    for s in legs:
        p *= s["fair"]
    return p


# ================================================================ moneyline

def moneyline_picks(ml_events, linked_by_event):
    picks = []
    for ev in ml_events:
        g = linked_by_event.get(ev.get("id"))
        if not g:
            continue
        home, away = ev.get("home_team"), ev.get("away_team")
        per_book, best = [], {}
        for bk in ev.get("bookmakers", []) or []:
            m = next((m for m in bk.get("markets", []) or []
                      if m.get("key") == "h2h"), None)
            if not m:
                continue
            o = {x.get("name"): float(x.get("price") or 0)
                 for x in m.get("outcomes", []) or []}
            if o.get(home, 0) > 1 and o.get(away, 0) > 1:
                fair = C.shin_devig([1.0 / o[home], 1.0 / o[away]])
                per_book.append({home: fair[0], away: fair[1]})
            for side, price in o.items():
                if price > best.get(side, 0):
                    best[side] = price
                    best[side + "|book"] = bk.get("key")
        if not per_book or not best:
            continue
        fair_home = C.median([b[home] for b in per_book])
        fair_away = 1.0 - fair_home
        sel = home if fair_home >= fair_away else away
        fp = max(fair_home, fair_away)
        bp = best.get(sel, 0)
        if bp <= 1.0:
            continue
        dog = away if sel == home else home
        dog_id = g["away_id"] if sel == home else g["home_id"]
        dog_fp, dog_bp = 1.0 - fp, best.get(dog, 0)
        picks.append({
            "matchup": f"{C.TEAMS.get(g['away_id'],('?',))[0]} @ "
                       f"{C.TEAMS.get(g['home_id'],('?',))[0]}",
            "game_pk": g["gamePk"], "selection": sel,
            "sel_abbr": C.TEAMS.get(
                g["home_id"] if sel == home else g["away_id"], ("?",))[0],
            "fair": fp, "best_price": bp,
            "best_book": best.get(sel + "|book", "?"),
            "n_books": len(per_book),
            "edge": bp * fp - 1.0,
            "dog": dog, "dog_abbr": C.TEAMS.get(dog_id, ("?",))[0],
            "dog_edge": (dog_bp * dog_fp - 1.0) if dog_bp else -1,
            "dog_price": dog_bp, "dog_book": best.get(dog + "|book", "?"),
            "sp": f"{g['away_sp_name'] or 'TBD'} / {g['home_sp_name'] or 'TBD'}",
            "start": ev.get("commence_time"),
            "start_dt": F.event_start(ev) if ev.get("commence_time") else None,
        })
    picks.sort(key=lambda p: -p["fair"])
    return picks


# ================================================================ render

def edge_threshold(sels):
    """
    Adaptive flag. Book dispersion on home-run props is far wider than on
    moneylines, so a fixed 2% would light up every leg and mean nothing.
    Flag the genuinely wide prices: the day's own 80th percentile, floored
    at 4% so a flat slate doesn't manufacture signal.
    """
    e = sorted(s["best_price"] * s["fair"] - 1 for s in sels if s.get("fair"))
    if not e:
        return 0.04
    q80 = e[min(len(e) - 1, int(len(e) * 0.80))]
    return max(0.04, q80)


def tier(p):
    if p >= 0.620:
        return "STRONG"
    if p >= 0.555:
        return "LEAN"
    return "COIN FLIP"


SGT = timezone(timedelta(hours=8))
TIER_ORDER = ["STRONG", "LEAN", "COIN FLIP"]


def sgt_time(dt):
    """Start time in the user's own clock, since the slate runs overnight."""
    if not dt:
        return "?"
    return dt.astimezone(SGT).strftime("%-I:%M%p").lower()


def render(board_date, parlays, mls, sels, credits, graded):
    A, B = [], []
    thr = edge_threshold(sels)
    A.append(f"<b>MLB HOME RUN BOARD — {board_date:%a %d %b %Y}</b>")
    A.append(f"{len(mls)} games · {len(sels)} priced bats · "
             f"built {datetime.now(timezone.utc):%H:%M} UTC")
    A.append("")
    for name, legs in parlays:
        price = parlay_price(legs)
        prob = parlay_prob(legs)
        A.append(f"<b>━━ {name} · {len(legs)} legs ━━</b>")
        for s in legs:
            star = " ⚡" if s["best_price"] * s["fair"] - 1 > thr else ""
            A.append(f"• <b>{s['player']}</b> ({s['team']}) — {s['matchup']} — "
                     f"{s['best_price']:.2f} ({C.american(s['best_price'])})"
                     f"{star}")
        A.append(f"  <b>Combined {price:.2f} ({C.american(price)})</b> · "
                 f"hit chance {prob*100:.2f}%")
        A.append("")
    A.append(f"<i>Legs ranked by how likely they are to land, never two from "
             f"the same game. ⚡ = the best price out there is unusually far "
             f"above what the rest of the market is offering.</i>")

    B.append(f"<b>MONEYLINE — every game, {board_date:%a %d %b}</b>")
    for label in TIER_ORDER:
        group = [m for m in mls if tier(m["fair"]) == label]
        if not group:
            continue
        group.sort(key=lambda m: (m["start_dt"] is None, m["start_dt"]))
        B.append("")
        B.append(f"<b>━━ {label} ━━</b>")
        for m in group:
            star = " ⚡" if m["edge"] > 0.02 else ""
            B.append(f"<b>{m['sel_abbr']}</b> {m['matchup']} · "
                     f"{m['best_price']:.2f} ({C.american(m['best_price'])}) · "
                     f"fair {m['fair']*100:.0f}% · "
                     f"{sgt_time(m['start_dt'])}{star}")
            if m["dog_edge"] > 0.02:
                B.append(f"   ↳ value on the underdog: {m['dog_abbr']} "
                         f"{m['dog_price']:.2f} ({m['dog_edge']*100:+.1f}%)")
    B.append("")
    if graded:
        B.append(f"<b>Yesterday:</b> {graded}")
    B.append(f"<i>Sorted strongest first, then by start time (Singapore). "
             f"Prices are the best available across "
             f"{max((m['n_books'] for m in mls), default=0)} books. "
             f"⚡ = that price beats where the market has the game — the one "
             f"edge the football backtest ever found. "
             f"Credits left: {credits}.</i>")
    return "\n".join(A), "\n".join(B)


def discover_chat_id(token):
    """
    Telegram will not let a bot message you until you have messaged it first,
    and the chat id is not shown anywhere in the app. This reads it off the
    bot's own update feed so the first run can tell you what to paste.
    """
    import requests
    try:
        r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates",
                         timeout=30)
        ids = []
        for u in (r.json().get("result") or []):
            for k in ("message", "channel_post", "edited_message"):
                c = (u.get(k) or {}).get("chat") or {}
                if c.get("id") and c["id"] not in [i for i, _ in ids]:
                    ids.append((c["id"],
                                c.get("title") or c.get("username")
                                or c.get("first_name") or "?"))
        return ids
    except Exception as e:
        print(f"  getUpdates failed: {e}")
        return []


def telegram(token, chat_id, text):
    import requests
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text,
              "parse_mode": "HTML", "disable_web_page_preview": True},
        timeout=45,
    )
    ok = r.status_code == 200 and r.json().get("ok")
    print(f"  telegram HTTP {r.status_code} ok={ok}")
    if not ok:
        print(f"  telegram body: {r.text[:400]}")
    return ok, r.status_code, r.text[:400]


# ================================================================ grading

STALE_DAYS = 5


def grade_previous(board_date):
    """
    Settle everything logged on an earlier board.

    A scratched hitter never appears in the boxscore at all. Books void a
    home-run prop when the player takes no plate appearance, so that is what
    gets recorded — treating it as a loss would quietly understate the hit
    rate, and leaving it blank would re-fetch the same boxscore every day
    forever.
    """
    rows = C.read_csv(PICKS)
    today = board_date.isoformat()
    todo = [r for r in rows if not r.get("result")
            and r.get("board_date") and r["board_date"] != today]
    if not todo:
        return "", rows, False

    finals = {}
    for d in sorted({r["board_date"] for r in todo}):
        for g in F.schedule(d):
            finals[str(g["gamePk"])] = g

    want_box = {r["game_pk"] for r in todo if r["market"] == "HR"
                and finals.get(r["game_pk"], {}).get("state") == "Final"}
    boxes = {}
    for pk in sorted(want_box):
        b = F.boxscore_hr(int(pk))
        if b is not None:
            boxes[pk] = {str(k): v for k, v in b.items()}

    hit = tot = ml_hit = ml_tot = void = 0
    changed = False
    for r in rows:
        if r.get("result") or not r.get("board_date") or r["board_date"] == today:
            continue
        pk = r.get("game_pk")
        g = finals.get(pk)
        stale = (board_date - C.date_from_iso(r["board_date"])).days > STALE_DAYS

        if not g or g.get("state") != "Final":
            if stale:                       # postponed or never played
                r["result"] = "VOID"
                changed = True
                void += 1
            continue

        if r["market"] == "HR":
            box = boxes.get(pk)
            if box is None:
                if stale:
                    r["result"] = "VOID"
                    changed = True
                    void += 1
                continue
            n = box.get(str(r.get("selection_id") or ""))
            if n is None:                   # did not bat -> book voids it
                r["result"] = "VOID"
                void += 1
            else:
                r["result"] = "WIN" if n >= 1 else "LOSS"
                tot += 1
                hit += 1 if n >= 1 else 0
            changed = True
        else:
            if g.get("home_score") is None:
                continue
            winner = (C.TEAMS.get(g["home_id"], ("?",))[0]
                      if g["home_score"] > g["away_score"]
                      else C.TEAMS.get(g["away_id"], ("?",))[0])
            r["result"] = "WIN" if r.get("team") == winner else "LOSS"
            changed = True
            ml_tot += 1
            ml_hit += 1 if r["result"] == "WIN" else 0

    bits = []
    if tot:
        bits.append(f"HR legs {hit}/{tot}")
    if ml_tot:
        bits.append(f"moneylines {ml_hit}/{ml_tot} ({ml_hit/ml_tot*100:.0f}%)")
    if void:
        bits.append(f"{void} void")
    return " · ".join(bits), rows, changed


# ================================================================ main

def main():
    token = os.environ.get("TELEGRAM_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    api = os.environ.get("ODDS_API_KEY", "").strip()
    if not api:
        print("FATAL: ODDS_API_KEY is not set")
        sys.exit(1)
    if not token:
        print("FATAL: TELEGRAM_TOKEN is not set")
        sys.exit(1)
    if not chat:
        found = discover_chat_id(token)
        print("")
        print("=" * 62)
        if found:
            print("TELEGRAM_CHAT_ID is not set. Your bot has heard from:")
            for cid, who in found:
                print(f"    {cid}    ({who})")
            print("Add the one you want as the TELEGRAM_CHAT_ID repository "
                  "secret, then re-run.")
        else:
            print("TELEGRAM_CHAT_ID is not set, and your bot has received no "
                  "messages. Open the bot in Telegram, send it /start, then "
                  "re-run this workflow — the chat id will be printed here.")
        sys.exit(1)

    os.makedirs(DATA, exist_ok=True)
    board_date = C.eastern_date()
    ds = board_date.isoformat()
    # deliverable file first, so a bad run still leaves something behind
    with open(BOARD_MD, "w", encoding="utf-8") as f:
        f.write(f"# MLB board {ds}\n\nrun started "
                f"{datetime.now(timezone.utc).isoformat()}\n")

    print(f"== MLB board for US Eastern date {ds} ==")

    print("- grading previous picks")
    graded, all_rows, changed = grade_previous(board_date)
    if changed:
        hdr = PICK_HEADER + ["selection_id"]
        C.write_csv(PICKS, hdr, all_rows)
    print(f"  {graded or 'nothing to grade'}")

    print("- season rates")
    season = board_date.year
    hitting = F.season_stats(season, "hitting")
    pitching = F.season_stats(season, "pitching")
    lg = F.league_hr_rate(hitting)
    print(f"  league HR/PA {lg if lg else 'fallback'}")

    print("- schedule")
    games = [g for g in F.schedule(ds) if g["state"] == "Preview"]
    if not games:
        print("FATAL: no upcoming games on the slate")
        sys.exit(1)

    print("- odds")
    od = F.Odds(api)
    events = od.events()
    linked = F.link_events_to_games(events, games)
    linked_by_event = {ev["id"]: g for ev, g in linked}
    print(f"  linked {len(linked)} of {len(events)} events to {len(games)} games")

    ml_events = od.moneylines()
    mls = moneyline_picks(ml_events, linked_by_event)
    print(f"  {len(mls)} moneyline picks")

    print("- rosters")
    rosters = {}
    for g in games:
        for tid in (g["home_id"], g["away_id"]):
            if tid not in rosters:
                rosters[tid] = F.roster(tid)

    print("- home run props")
    sels = []
    for ev, g in linked:
        eo = od.hr_props(ev["id"])
        got = collect_hr_selections(eo, g, rosters, hitting, pitching, lg)
        sels.extend(got)
        print(f"  {C.TEAMS.get(g['away_id'],('?',))[0]}@"
              f"{C.TEAMS.get(g['home_id'],('?',))[0]}: {len(got)} bats")
    if not sels:
        print("FATAL: no home run prices returned — check plan level "
              "and that batter_home_runs is covered for this slate")
        sys.exit(1)

    fill_estimated_fair(sels)
    pool = [s for s in sels
            if s["fair"] and MIN_PRICE <= s["best_price"] <= MAX_PRICE]
    print(f"  {len(pool)} of {len(sels)} bats eligible "
          f"(>= {MIN_BOOKS} books, {MIN_PRICE}-{MAX_PRICE})")
    if len(pool) < 3:
        print("FATAL: eligible pool too small to build a parlay")
        sys.exit(1)

    parlays = build_parlays(pool)
    msg_a, msg_b = render(board_date, parlays, mls, sels, od.remaining, graded)

    with open(BOARD_MD, "w", encoding="utf-8") as f:
        f.write(f"# MLB board {ds}\n\n")
        f.write(msg_a.replace("<b>", "**").replace("</b>", "**")
                .replace("<i>", "_").replace("</i>", "_"))
        f.write("\n\n")
        f.write(msg_b.replace("<b>", "**").replace("</b>", "**")
                .replace("<i>", "_").replace("</i>", "_"))
        f.write("\n")

    print("- logging picks")
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for name, legs in parlays:
        for s in legs:
            rows.append({
                "board_date": ds, "logged_at": now, "market": "HR",
                "game_pk": s["game_pk"], "matchup": s["matchup"],
                "selection": s["player"], "selection_id": s["player_id"] or "",
                "team": s["team"], "opp_sp": s["opp_sp"],
                "best_price": f"{s['best_price']:.2f}",
                "best_book": s["best_book"],
                "fair_prob": f"{s['fair']:.4f}",
                "model_prob": f"{s['model']:.4f}",
                "n_books": s["n_books"],
                "edge_pct": f"{(s['best_price']*s['fair']-1)*100:.2f}",
                "parlay": name,
            })
    for m in mls:
        rows.append({
            "board_date": ds, "logged_at": now, "market": "ML",
            "game_pk": m["game_pk"], "matchup": m["matchup"],
            "selection": m["selection"], "selection_id": "",
            "team": m["sel_abbr"], "opp_sp": m["sp"],
            "best_price": f"{m['best_price']:.2f}",
            "best_book": m["best_book"],
            "fair_prob": f"{m['fair']:.4f}", "model_prob": "",
            "n_books": m["n_books"],
            "edge_pct": f"{m['edge']*100:.2f}", "parlay": "",
        })
    C.append_csv(PICKS, PICK_HEADER + ["selection_id"], rows)
    print(f"  logged {len(rows)} picks")

    print("- telegram")
    ok1, c1, b1 = telegram(token, chat, msg_a)
    ok2, c2, b2 = telegram(token, chat, msg_b)
    if not (ok1 and ok2):
        print("")
        print("=" * 62)
        print("DELIVERY FAILED — the board was built and committed, but "
              "Telegram did not accept it.")
        print(f"  message 1: HTTP {c1} :: {b1}")
        print(f"  message 2: HTTP {c2} :: {b2}")
        if "chat not found" in (b1 + b2).lower():
            print("  DIAGNOSIS: TELEGRAM_CHAT_ID is wrong, or you have not "
                  "sent your bot a message yet. Open the bot in Telegram, "
                  "send it /start, then re-run.")
        elif "unauthorized" in (b1 + b2).lower():
            print("  DIAGNOSIS: TELEGRAM_TOKEN is wrong or revoked.")
        else:
            print("  DIAGNOSIS: see the HTTP body above.")
        sys.exit(1)

    print("")
    print(f"OK — {len(parlays)} parlays and {len(mls)} moneylines sent. "
          f"Odds credits remaining: {od.remaining}")


if __name__ == "__main__":
    main()
