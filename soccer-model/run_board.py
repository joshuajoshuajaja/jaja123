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


def preflight(token: str) -> tuple[bool, str | None, str]:
    """
    Work out exactly what state Telegram is in, and say so in plain words.
    Returns (ok, chat_id, message).
    """
    if not token:
        return False, None, (
            "TELEGRAM_TOKEN is missing or empty.\n"
            "  Fix: repo Settings -> Secrets and variables -> Actions -> New repository\n"
            "  secret, named exactly TELEGRAM_TOKEN, value = the token BotFather gave you.")

    try:
        me = requests.get(f"{TG}/bot{token}/getMe", timeout=30)
    except Exception as e:
        return False, None, (
            f"Could not reach Telegram at all ({type(e).__name__}).\n"
            "  GitHub's servers may be blocked from api.telegram.org. Tell Claude and\n"
            "  we will send the summary another way.")

    if me.status_code == 401:
        return False, None, (
            "Telegram rejected the token (401).\n"
            "  Fix: the value is wrong or truncated. Copy the whole thing from BotFather,\n"
            "  it looks like 1234567890:AAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    if me.status_code != 200:
        return False, None, f"Telegram getMe returned {me.status_code}: {me.text[:200]}"

    uname = me.json().get("result", {}).get("username", "?")

    cid = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if cid:
        return True, cid, f"bot @{uname} ok, using configured chat {cid}"

    try:
        r = requests.get(f"{TG}/bot{token}/getUpdates", timeout=30)
        for u in reversed(r.json().get("result", [])):
            c = (u.get("message") or u.get("channel_post") or {}).get("chat", {})
            if c.get("id"):
                return True, str(c["id"]), (
                    f"bot @{uname} ok, found chat {c['id']}.\n"
                    f"  Add {c['id']} as a secret named TELEGRAM_CHAT_ID so this keeps\n"
                    "  working after old messages expire.")
    except Exception as e:
        return False, None, f"bot @{uname} ok but getUpdates failed: {e}"

    return False, None, (
        f"The bot @{uname} is alive, but nobody has ever messaged it, so it has\n"
        "  nowhere to send to.\n"
        f"  Fix: open Telegram, search @{uname}, open the chat and press Start\n"
        "  (or just send it 'hi'), then run this workflow again.")


def send(text: str) -> bool:
    token = os.environ.get("TELEGRAM_TOKEN", "").strip()
    ok, cid, note = preflight(token)
    print(f"telegram: {note}", flush=True)
    if not ok:
        print("\n--- the summary that would have been sent ---\n")
        print(text)
        return False
    try:
        r = requests.post(f"{TG}/bot{token}/sendMessage", timeout=30, json={
            "chat_id": cid, "text": text, "parse_mode": "HTML",
            "disable_web_page_preview": True})
    except Exception as e:
        print(f"telegram send failed: {e}", file=sys.stderr)
        return False
    if r.status_code != 200:
        print(f"telegram refused the message ({r.status_code}): {r.text[:400]}",
              file=sys.stderr)
        return False
    print("sent to telegram", flush=True)
    return True


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
    delivered = send(summary(fx, winners, totals, parlay, url))
    print(f"board built: {len(fx)} fixtures, {n} picks logged")
    if not delivered:
        print("\nBOARD BUILT BUT NOT DELIVERED - see the telegram line above.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

