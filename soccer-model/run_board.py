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
    if not ok:
        # diagnosis LAST, so it is the final thing in the log rather than
        # buried above the summary dump
        print("\n--- the summary that would have been sent ---\n")
        print(text)
        print("\n" + "=" * 66)
        print("TELEGRAM DID NOT SEND")
        print("=" * 66)
        print(note)
        print("=" * 66, flush=True)
        return False
    print(f"telegram: {note}", flush=True)
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


def summary(fx, winners, totals, parlay, acca, board_url: str | None) -> str:
    e = _html.escape
    day = datetime.now(SGT).strftime("%a %-d %b")
    comb = float(np.prod([s["top"][0][0] for s in parlay])) if parlay else 0.0
    L = [f"<b>Matchday board - {day}</b>",
         f"{len(fx)} matches, next 48h", ""]

    if acca:
        ap = float(np.prod([r["p"] for r in acca]))
        apr = float(np.prod([r["price"] for r in acca]))
        L.append(f"<b>ACCUMULATOR</b>  {apr:.0f} to 1  ·  lands {ap*100:.1f}% of the time")
        for r in acca:
            f = r["fixture"]
            L.append(f"  <b>{r['p']*100:.0f}%</b>  {e(r['pick'])}  <i>@{r['price']:.2f}</i>")
            L.append(f"       {e(f['home'])} v {e(f['away'])}")
        L.append("")

    from render_board import MIN_ODDS
    pw = [w for w in winners if w["price"] >= MIN_ODDS][:6]
    if pw:
        L.append("<b>WINNERS</b>  <i>most likely first, all 1.50 or better</i>")
        for w in pw:
            f = w["fixture"]
            L.append(f"  <b>{w['p']*100:.0f}%</b>  {e(w['pick'])}  <i>@{w['price']:.2f}</i>"
                     f"  <i>(fair {1/w['p']:.2f})</i>")
            L.append(f"       {e(f['home'])} v {e(f['away'])}")
        L.append("")

    pt = [t for t in totals if t["price"] >= MIN_ODDS][:4]
    if pt:
        L.append("<b>GOALS</b>")
        for t in pt:
            f = t["fixture"]
            L.append(f"  <b>{t['p']*100:.0f}%</b>  {e(t['pick'])}  <i>@{t['price']:.2f}</i>"
                     f"  <i>(fair {1/t['p']:.2f})</i>")
            L.append(f"       {e(f['home'])} v {e(f['away'])}")
        L.append("")

    if parlay:
        L.append(f"<b>LOTTERY TICKET</b>  ~{1/comb:,.0f} to 1  ·  {comb*100:.2f}%")
        L.append("  " + "  ".join(f"{s['top'][0][1]}-{s['top'][0][2]}" for s in parlay))
        L.append("")

    L.append("<i>Everything here pays 1.50 or better, likeliest first. Fair = what it's worth "
             "with the bookmaker's cut removed; if Pools pays under that, take a different one "
             "off the list.</i>")
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

    winners, totals, handicaps, scores, parlay, acca = build(fx)
    n = write_log(winners, totals, parlay, acca, path="data/picks_log.csv")

    with open("board.html", "w") as fh:
        fh.write(render(fx, winners, totals, handicaps, scores, parlay, acca))
    with open("fixtures.pkl", "wb") as fh:
        pickle.dump(fx, fh)

    url = os.environ.get("BOARD_URL", "").strip() or None
    delivered = send(summary(fx, winners, totals, parlay, acca, url))
    print(f"board built: {len(fx)} fixtures, {n} picks logged")
    if not delivered:
        print("\nBOARD BUILT BUT NOT DELIVERED - see the telegram line above.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

