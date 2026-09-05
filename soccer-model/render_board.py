#!/usr/bin/env python3
"""Render the daily board to HTML and write the picks log."""
from __future__ import annotations

import html
import json
import pickle
import sys
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, "soccer-model")
from build_board import ah_split, book, GD, TOT, K   # noqa: E402

SGT = timezone(timedelta(hours=8))
EXCHANGES = {"betfair_ex_eu", "matchbook", "betfair_ex_uk"}

# J's rule: nothing below 1.50 decimal, and among what clears that bar, rank by
# the chance of it actually landing rather than by how good the price is.
MIN_ODDS = 1.50
ACCA_LEGS = 4


def kick(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone(SGT).strftime("%a %-d %b, %-I:%M%p")


def build(fx):
    winners, totals, handicaps, scores = [], [], [], []

    for f in fx:
        M, fair = f["M"], f["fair"]
        for i, side in enumerate(("home", "draw", "away")):
            price, bk = f["best"][side]
            if not price:
                continue
            name = {"home": f["home"], "draw": "Draw", "away": f["away"]}[side]
            winners.append(dict(
                fixture=f, pick=name, side=side, p=float(fair[i]),
                price=float(price), bookie=bk, edge=float(price * fair[i] - 1)))

        for line, (fo, bo, bu, n) in sorted(f["totals"].items()):
            for label, p, (price, bk) in (("Over", fo, bo), ("Under", 1 - fo, bu)):
                if not price:
                    continue
                totals.append(dict(
                    fixture=f, pick=f"{label} {line:g}", p=float(p),
                    price=float(price), bookie=bk, edge=float(price * p - 1)))

        # the fairest handicap: the line closest to a coin flip
        best_h, best_gap = None, 9
        for q in range(-10, 11):
            h = q / 4.0
            w, pu, l = ah_split(M, h)
            live = w + l
            if live < 0.2:
                continue
            gap = abs(w / live - 0.5)
            if gap < best_gap:
                best_gap, best_h = gap, (h, w, pu, l)
        if best_h:
            h, w, pu, l = best_h
            handicaps.append(dict(
                fixture=f, line=h, w=w, push=pu, l=l,
                price_home=1 + l / w if w > 0 else float("inf"),
                price_away=1 + w / l if l > 0 else float("inf")))

        flat = [(float(M[i, j]), i, j) for i in range(6) for j in range(6)]
        flat.sort(reverse=True)
        scores.append(dict(fixture=f, top=flat[:3]))

    # ONE pick per game. Take everything clearing the odds floor, rank by hit
    # chance, then keep only the best selection for each fixture - so a match
    # can never turn up twice with two different bets on it.
    pool = [r for r in winners + totals if r["price"] >= MIN_ODDS]
    pool.sort(key=lambda r: -r["p"])
    picks, seen = [], set()
    for r in pool:
        eid = r["fixture"]["event_id"]
        if eid in seen:
            continue
        picks.append(r)
        seen.add(eid)

    winners = [r for r in picks if r.get("side")]
    totals = [r for r in picks if not r.get("side")]
    scores.sort(key=lambda r: -r["top"][0][0])
    parlay = scores[:4]
    acca = picks[:ACCA_LEGS]
    return winners, totals, handicaps, scores, parlay, acca


def write_log(winners, totals, parlay, acca=(), path="picks_log.csv"):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for r in winners + totals:
        if r["price"] < MIN_ODDS:
            continue
        f = r["fixture"]
        rows.append(dict(logged_utc=now, kickoff=f["kickoff"], league=f["league"],
                         home=f["home"], away=f["away"], market="1X2/Totals",
                         pick=r["pick"], model_prob=round(r["p"], 5),
                         price_taken=r["price"], bookie=r["bookie"],
                         edge=round(r["edge"], 5)))
    for s in parlay:
        f = s["fixture"]; p, i, j = s["top"][0]
        rows.append(dict(logged_utc=now, kickoff=f["kickoff"], league=f["league"],
                         home=f["home"], away=f["away"], market="Correct Score",
                         pick=f"{i}-{j}", model_prob=round(p, 5),
                         price_taken="", bookie="", edge=""))
    df = pd.DataFrame(rows)
    try:
        prev = pd.read_csv(path)
        df = pd.concat([prev, df], ignore_index=True)
    except FileNotFoundError:
        pass
    df.to_csv(path, index=False)
    return len(rows)


# ---------------------------------------------------------------------------

def row_fixture(f):
    return (f'<div class="fx"><span class="fx-t">{html.escape(f["home"])}'
            f'<span class="v">v</span>{html.escape(f["away"])}</span>'
            f'<span class="fx-m">{html.escape(f["league"])} · {kick(f["kickoff"])}</span></div>')


def render(fx, winners, totals, handicaps, scores, parlay, acca) -> str:
    today = datetime.now(SGT).strftime("%A %-d %B")
    leagues = sorted({f["league"] for f in fx})
    pos_w = [w for w in winners if w["price"] >= MIN_ODDS]
    pos_t = [t for t in totals if t["price"] >= MIN_ODDS]
    acca_p = float(np.prod([r["p"] for r in acca])) if acca else 0.0
    acca_pr = float(np.prod([r["price"] for r in acca])) if acca else 0.0
    comb = float(np.prod([s["top"][0][0] for s in parlay])) if parlay else 0.0

    def edge_cell(e):
        cls = "up" if e > 0.02 else ("mid" if e > 0 else "flat")
        return f'<td class="n {cls}">{e*100:+.1f}%</td>'

    def price_cell(price, bk):
        ex = ' <span class="ex">exch</span>' if bk in EXCHANGES else ""
        return (f'<td class="n big">{price:.2f}</td>'
                f'<td class="bk">{html.escape(book(bk))}{ex}</td>')

    legs = "".join(
        f'<div class="leg"><div class="leg-n">{i+1}</div>'
        f'<div class="leg-b"><div class="leg-s">{s["top"][0][1]}&ndash;{s["top"][0][2]}</div>'
        f'<div class="leg-f">{html.escape(s["fixture"]["home"])} v {html.escape(s["fixture"]["away"])}</div>'
        f'<div class="leg-m">{html.escape(s["fixture"]["league"])} · {kick(s["fixture"]["kickoff"])} '
        f'</div></div></div>'
        for i, s in enumerate(parlay))

    win_rows = "".join(
        f'<tr><td>{row_fixture(w["fixture"])}</td>'
        f'<td class="pick">{html.escape(w["pick"])}</td>'
        f'<td class="n chance">{w["p"]*100:.0f}%</td>'
        f'{price_cell(w["price"], w["bookie"])}'
        f'<td class="n dim">{1/w["p"]:.2f}</td></tr>'
        for w in pos_w[:18])

    tot_rows = "".join(
        f'<tr><td>{row_fixture(t["fixture"])}</td>'
        f'<td class="pick">{html.escape(t["pick"])}</td>'
        f'<td class="n chance">{t["p"]*100:.0f}%</td>'
        f'{price_cell(t["price"], t["bookie"])}'
        f'<td class="n dim">{1/t["p"]:.2f}</td></tr>'
        for t in pos_t[:14])

    acca_rows = "".join(
        f'<div class="leg"><div class="leg-n">{i+1}</div><div class="leg-b">'
        f'<div class="leg-s acc">{r["price"]:.2f}</div>'
        f'<div class="leg-f">{html.escape(r["pick"])}</div>'
        f'<div class="leg-m">{html.escape(r["fixture"]["home"])} v {html.escape(r["fixture"]["away"])}<br>'
        f'{html.escape(book(r["bookie"]))} · {kick(r["fixture"]["kickoff"])}</div>'
        f'</div></div>'
        for i, r in enumerate(acca))

    hcp_rows = "".join(
        f'<tr><td>{row_fixture(h["fixture"])}</td>'
        f'<td class="pick">{html.escape(h["fixture"]["home"])} {h["line"]:+g}</td>'
        f'<td class="n big">{h["price_home"]:.2f}</td>'
        f'<td class="pick">{html.escape(h["fixture"]["away"])} {-h["line"]:+g}</td>'
        f'<td class="n big">{h["price_away"]:.2f}</td>'
        f'<td class="n dim">{h["push"]*100:.0f}%</td></tr>'
        for h in sorted(handicaps, key=lambda x: x["fixture"]["kickoff"])[:16])

    cs_rows = "".join(
        f'<tr><td>{row_fixture(s["fixture"])}</td>' +
        "".join(f'<td class="cs"><b>{i}&ndash;{j}</b><span>{1/p:.1f}</span></td>'
                for p, i, j in s["top"]) + "</tr>"
        for s in scores[:16])

    return f"""<title>Matchday Board</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=JetBrains+Mono:wght@400;600;700&display=swap">
<style>
:root{{
  --paper:#F3F4F1;--surface:#FBFBF9;--sunk:#E9EBE6;
  --ink:#15191B;--ink-2:#4A5457;--ink-3:#7C8688;
  --rule:#D5D9D2;--rule-2:#C2C8BF;
  --pitch:#1E6B55;--pitch-soft:#DCE9E3;
  --amber:#9A6612;--amber-soft:#F0E6D2;
  --claret:#8E3247;
  --shadow:0 1px 2px rgba(21,25,27,.06),0 10px 30px -20px rgba(21,25,27,.5);
}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
  --paper:#101416;--surface:#171C1F;--sunk:#1D2326;
  --ink:#E4E7E3;--ink-2:#A2ACAD;--ink-3:#727C7E;
  --rule:#2A3134;--rule-2:#3A4347;
  --pitch:#4FA88C;--pitch-soft:#16302A;
  --amber:#D9A03C;--amber-soft:#332715;--claret:#C4677F;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -20px rgba(0,0,0,.9);
}}}}
:root[data-theme="dark"]{{
  --paper:#101416;--surface:#171C1F;--sunk:#1D2326;
  --ink:#E4E7E3;--ink-2:#A2ACAD;--ink-3:#727C7E;
  --rule:#2A3134;--rule-2:#3A4347;
  --pitch:#4FA88C;--pitch-soft:#16302A;
  --amber:#D9A03C;--amber-soft:#332715;--claret:#C4677F;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -20px rgba(0,0,0,.9);
}}
*{{box-sizing:border-box}}
body{{background:var(--paper);color:var(--ink);font-family:"Source Serif 4",Georgia,serif;
  font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1080px;margin:0 auto;padding:0 24px 80px}}
h1,h2,h3,th,.pick,.fx-t,.leg-s,.leg-f,.bk,.eyebrow,.lbl{{font-family:Archivo,Helvetica,Arial,sans-serif}}
.n,.mono,.cs{{font-family:"JetBrains Mono",ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}}

.mast{{padding:56px 0 26px;border-bottom:2px solid var(--ink)}}
.eyebrow{{font-size:11px;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:var(--pitch)}}
h1{{font-size:clamp(38px,6.5vw,64px);font-weight:700;letter-spacing:-.035em;line-height:.96;margin:16px 0 0}}
.sub{{color:var(--ink-2);margin:14px 0 0;font-size:17px}}

section{{padding:44px 0 0}}
h2{{font-size:12px;font-weight:600;letter-spacing:.15em;text-transform:uppercase;color:var(--ink-3);
  margin:0 0 4px;padding-bottom:9px;border-bottom:1px solid var(--rule)}}
.hint{{font-size:14.5px;color:var(--ink-2);margin:14px 0 0;max-width:600px}}

.parlay{{margin:26px 0 0;background:var(--surface);border:1px solid var(--rule);box-shadow:var(--shadow)}}
.parlay-h{{display:flex;justify-content:space-between;align-items:baseline;gap:16px;flex-wrap:wrap;
  padding:20px 24px;border-bottom:1px solid var(--rule)}}
.parlay-t{{font-family:Archivo,sans-serif;font-weight:600;font-size:15px}}
.parlay-p{{font-family:"JetBrains Mono",monospace;font-weight:700;font-size:26px;color:var(--pitch);letter-spacing:-.02em}}
.legs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:1px;background:var(--rule)}}
.leg{{background:var(--surface);padding:18px 20px;display:flex;gap:14px;align-items:flex-start}}
.leg-n{{width:22px;height:22px;flex:none;border-radius:50%;border:1px solid var(--rule-2);
  display:flex;align-items:center;justify-content:center;font-family:"JetBrains Mono",monospace;
  font-size:10px;color:var(--ink-3);margin-top:3px}}
.leg-s{{font-family:"JetBrains Mono",monospace;font-weight:700;font-size:22px;letter-spacing:-.02em;line-height:1}}
.leg-f{{font-size:14px;font-weight:600;margin-top:6px;line-height:1.3}}
.leg-m{{font-size:12px;color:var(--ink-3);margin-top:4px;line-height:1.35}}

.tw{{overflow-x:auto;margin:22px 0 0;border:1px solid var(--rule);background:var(--surface)}}
table{{border-collapse:collapse;width:100%;min-width:640px;font-size:14px}}
th{{font-size:10px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3);
  text-align:left;padding:11px 14px;border-bottom:1px solid var(--rule-2);white-space:nowrap}}
th.r,td.n{{text-align:right}}
td{{padding:10px 14px;border-bottom:1px solid var(--rule);vertical-align:middle}}
tr:last-child td{{border-bottom:0}}
.fx-t{{font-weight:600;font-size:14px;display:block;white-space:nowrap}}
.v{{color:var(--ink-3);font-weight:400;margin:0 7px;font-size:12px}}
.fx-m{{font-size:11.5px;color:var(--ink-3);display:block;margin-top:2px;white-space:nowrap}}
.pick{{font-weight:600;font-size:13.5px;white-space:nowrap}}
.n.big{{font-weight:600;font-size:15px}}
.n.dim{{color:var(--ink-3);font-size:12.5px}}
.bk{{font-size:12px;color:var(--ink-2);white-space:nowrap}}
.ex{{font-size:9px;letter-spacing:.08em;text-transform:uppercase;color:var(--amber);
  background:var(--amber-soft);padding:1px 4px;border-radius:2px;margin-left:3px}}
.up{{color:var(--pitch);font-weight:700}}
.mid{{color:var(--ink-2)}}
.flat{{color:var(--ink-3)}}
.n.chance{{font-weight:700;font-size:15px;color:var(--pitch)}}
.leg-s.acc{{color:var(--pitch)}}
.cs{{text-align:center;white-space:nowrap;padding:8px 10px}}
.cs b{{display:block;font-size:15px;font-weight:700}}
.cs span{{display:block;font-size:11px;color:var(--ink-3);margin-top:1px}}

.note{{margin:26px 0 0;padding:18px 22px;background:var(--surface);border-left:3px solid var(--amber);
  max-width:640px;font-size:15px;color:var(--ink-2)}}
.note b{{color:var(--ink)}}
.foot{{margin-top:52px;padding-top:20px;border-top:2px solid var(--ink);font-size:13.5px;color:var(--ink-3);max-width:640px}}
</style>

<div class="wrap">
<header class="mast">
  <div class="eyebrow">Matchday board &middot; tonight</div>
  <h1>{today}</h1>
  <p class="sub">{len(fx)} matches across {len(leagues)} leagues. {len(pos_w)} winners and
  {len(pos_t)} goals bets, tonight only. One pick per game, likeliest first.</p>
</header>

<section>
  <h2>The accumulator</h2>
  <div class="parlay">
    <div class="parlay-h">
      <span class="parlay-t">Four legs, every one paying 1.50 or better</span>
      <span class="parlay-p">{acca_pr:.0f} to 1</span>
    </div>
    <div class="legs">{acca_rows}</div>
  </div>
  <p class="hint">The four likeliest picks on tonight's board that still pay 1.50 or better,
  one per match.</p>
</section>

<section>
  <h2>The lottery ticket</h2>
  <div class="parlay">
    <div class="parlay-h">
      <span class="parlay-t">Four correct scores, four different matches</span>
      <span class="parlay-p">{1/comb:,.0f} to 1</span>
    </div>
    <div class="legs">{legs}</div>
  </div>
  <p class="hint">Four correct scores from tonight's games. The long shot &mdash; the
  accumulator above is the one built to land.</p>
</section>

<section>
  <h2>Winners</h2>
  <p class="hint">One pick per game, everything paying 1.50 or better, likeliest first.
  Price is the best I could find across the books in the feed.</p>
  <div class="tw"><table>
    <thead><tr><th>Match</th><th>Pick</th><th class="r">Price</th><th>Where</th></tr></thead>
    <tbody>{win_rows}</tbody>
  </table></div>
</section>

<section>
  <h2>Goals</h2>
  <div class="tw"><table>
    <thead><tr><th>Match</th><th>Pick</th><th class="r">Price</th><th>Where</th></tr></thead>
    <tbody>{tot_rows}</tbody>
  </table></div>
</section>

<section>
  <h2>Handicaps</h2>
  <p class="hint">The line that makes each match closest to a coin flip, and what each side is
  worth at it. No bookmaker prices in the feed for these yet &mdash; compare against Pools yourself.</p>
  <div class="tw"><table>
    <thead><tr><th>Match</th><th>Home</th><th class="r">Worth</th><th>Away</th>
    <th class="r">Worth</th><th class="r">Push</th></tr></thead>
    <tbody>{hcp_rows}</tbody>
  </table></div>
</section>

<section>
  <h2>Correct scores</h2>
  <p class="hint">Three likeliest scorelines per match.</p>
  <div class="tw"><table>
    <thead><tr><th>Match</th><th class="r">1st</th><th class="r">2nd</th><th class="r">3rd</th></tr></thead>
    <tbody>{cs_rows}</tbody>
  </table></div>
</section>

<div class="note">
  <b>How this is picked.</b> One selection per game, nothing paying under 1.50, ordered by
  how likely it is to land. Tonight's kick-offs only &mdash; nothing that runs into tomorrow.
  Anything marked <span class="ex">exch</span> is an exchange, so knock a few percent off
  for commission.
</div>

<div class="foot">
  Built {datetime.now(SGT).strftime("%-d %b %Y, %-I:%M%p")} Singapore time from
  {sum(f["nbooks"] for f in fx)} bookmaker price sets. Every pick above is logged with the price
  and the time, so in a few months there'll be a real record of how this went rather than
  a memory of the good ones.
</div>
</div>
"""


if __name__ == "__main__":
    fx = pickle.load(open("fixtures.pkl", "rb"))
    w, t, h, s, p, acca = build(fx)
    n = write_log(w, t, p, acca)
    out = "matchday_board.html"
    with open(out, "w") as fh:
        fh.write(render(fx, w, t, h, s, p, acca))
    print(f"wrote {out} | {len(w)} winner lines, {len(t)} totals, "
          f"{len(h)} handicaps, {len(s)} score sets | logged {n} picks")

