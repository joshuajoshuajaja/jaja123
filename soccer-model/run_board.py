#!/usr/bin/env python3
"""
Daily board: read the committed data, price today's fixtures, write the page,
send the summary to Telegram. Runs on GitHub Actions - no laptop required.
"""
from __future__ import annotations

import os
import sys
import html as _html
from datetime import datetime, timezone, timedelta

import numpy as np
import requests

sys.path.insert(0, "soccer-model")
sys.path.insert(0, ".")

SGT = timezone(timedelta(hours=8))
TG = "https://api.telegram.org"


def chat_id(token: str) -> str | None:
    """Use the configured chat, or find it from whoever messaged the bot last."""
    cid = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if cid:
        return cid
    try:
        r = requests.get(f"{TG}/bot{token}/getUpdates", timeout=30)
        for u in reversed(r.json().get("result", [])):
            c = (u.get("message") or u.get("channel_post") or {}).get("chat", {})
            if c.get("id"):
                print(f"discovered chat id {c['id']} - add it as TELEGRAM_CHAT_ID "
                      f"so it keeps working once old messages expire")
                return str(c["id"])
    except Exception as e:
        print(f"could not discover chat id: {e}", file=sys.stderr)
    return None


def send(text: str) -> bool:
    token = os.environ.get("TELEGRAM_TOKEN", "").strip()
    if not token:
        print("TELEGRAM_TOKEN not set - printing instead of sending\n")
        print(text)
        return False
    cid = chat_id(token)
    if not cid:
        print("no chat id - message the bot once, then rerun", file=sys.stderr)
        return False
    try:
        r = requests.post(f"{TG}/bot{token}/sendMessage", timeout=30, json={
            "chat_id": cid, "text": text, "parse_mode": "HTML",
            "disable_web_page_preview": True})
        if r.status_code != 200:
            print(f"telegram said {r.status_code}: {r.text[:300]}", file=sys.stderr)
            return False
        print("sent to telegram")
        return True
    except Exception as e:
        print(f"telegram failed: {e}", file=sys.stderr)
        return False


def summary(fx, winners, totals, parlay, board_url: str | None) -> str:
    e = _html.escape
    day = datetime.now(SGT).strftime("%a %-d %b")
    comb = float(np.prod([s["top"][0][0] for s in parlay])) if parlay else 0.0
    L = [f"<b>Matchday board - {day}</b>",
         f"{len(fx)} matches, next 48h", ""]

    if parlay:
        L.append(f"<b>PARLAY</b>  ~{1/comb:,.0f} to 1 if priced fairly")
        for s in parlay:
            f = s["fixture"]; p, i, j = s["top"][0]
            L.append(f"  <b>{i}-{j}</b>  {e(f['home'])} v {e(f['away'])}  <i>{p*100:.0f}%</i>")
        L.append("")

    pw = [w for w in winners if w["edge"] > 0][:6]
    if pw:
        L.append("<b>WINNERS</b>  <i>pick / fair price / how good the best price is</i>")
        for w in pw:
            f = w["fixture"]
            L.append(f"  {e(w['pick'])} <i>({e(f['home'])} v {e(f['away'])})</i>")
            L.append(f"     fair {1/w['p']:.2f} · best {w['price']:.2f} · +{w['edge']*100:.1f}%")
        L.append("")

    pt = [t for t in totals if t["edge"] > 0][:4]
    if pt:
        L.append("<b>GOALS</b>")
        for t in pt:
            f = t["fixture"]
            L.append(f"  {e(t['pick'])} <i>({e(f['home'])} v {e(f['away'])})</i>")
            L.append(f"     fair {1/t['p']:.2f} · best {t['price']:.2f} · +{t['edge']*100:.1f}%")
        L.append("")

    L.append("<i>Fair = what the bet is worth with the bookmaker's cut removed. "
             "If Pools pays less than that, it's a bad price - not a bad bet.</i>")
    if board_url:
        L.append(f'\n<a href="{board_url}">Full board</a>')
    text = "\n".join(L)
    return text[:4000]


def main() -> int:
    import pickle
    from build_board import main as price_fixtures
    from render_board import build, render, write_log

    fx = price_fixtures()
    if not fx:
        send("<b>Matchday board</b>\nNo fixtures priced today - the odds feed came back empty.")
        return 0

    winners, totals, handicaps, scores, parlay = build(fx)
    n = write_log(winners, totals, parlay, path="data/picks_log.csv")

    with open("board.html", "w") as fh:
        fh.write(render(fx, winners, totals, handicaps, scores, parlay))
    with open("fixtures.pkl", "wb") as fh:
        pickle.dump(fx, fh)

    url = os.environ.get("BOARD_URL", "").strip() or None
    send(summary(fx, winners, totals, parlay, url))
    print(f"board built: {len(fx)} fixtures, {n} picks logged")
    return 0


if __name__ == "__main__":
    sys.exit(main())

