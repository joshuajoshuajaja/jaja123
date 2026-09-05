"""Turn today's odds feed into priced fixtures."""
from __future__ import annotations

import re
import unicodedata
import numpy as np
import pandas as pd
from scipy.special import gammaln

SPORT_TO_DIV = {
    "soccer_epl": "E0", "soccer_efl_champ": "E1",
    "soccer_england_league1": "E2", "soccer_england_league2": "E3",
    "soccer_spain_la_liga": "SP1", "soccer_spain_segunda_division": "SP2",
    "soccer_italy_serie_a": "I1", "soccer_italy_serie_b": "I2",
    "soccer_germany_bundesliga": "D1", "soccer_germany_bundesliga2": "D2",
    "soccer_france_ligue_one": "F1", "soccer_france_ligue_two": "F2",
    "soccer_netherlands_eredivisie": "N1", "soccer_belgium_first_div": "B1",
    "soccer_portugal_primeira_liga": "P1", "soccer_turkey_super_league": "T1",
    "soccer_greece_super_league": "G1",
    "soccer_brazil_campeonato": "BRA: Serie A",
    "soccer_usa_mls": "USA: MLS", "soccer_mexico_ligamx": "MEX: Liga MX",
    "soccer_japan_j_league": "JAP: J1 League",
    "soccer_argentina_primera_division": "ARG: Liga Profesional",
}

_NOISE = re.compile(r"\b(fc|cf|afc|ac|as|ss|ssc|sc|cd|ud|rcd|club|calcio|"
                    r"football|deportivo|de|the|united|city|town|athletic|"
                    r"wanderers|rovers|albion|county|hotspur)\b")


def norm(s: str, strip_noise: bool = True) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    if strip_noise:
        s = _NOISE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def _acronym(s: str) -> str:
    return "".join(w[0] for w in norm(s, False).split() if w)


def _score(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    fa, fb = norm(a, False), norm(b, False)
    na, nb = norm(a), norm(b)
    if fa == fb:
        return 3.0
    if na and na == nb:
        return 2.5
    # football-data abbreviates: "QPR" for Queens Park Rangers, "Sp Gijon"
    if fa.replace(" ", "") == _acronym(b) or fb.replace(" ", "") == _acronym(a):
        return 2.4
    ta, tb = set(na.split()), set(nb.split())
    if not ta or not tb:
        return 0.0
    jac = len(ta & tb) / len(ta | tb)
    # one name being a prefix of the other is the common case:
    # "Wolverhampton Wanderers" vs "Wolves", "Newcastle United" vs "Newcastle"
    pre = 0.6 if (fa.startswith(fb[:5]) or fb.startswith(fa[:5])) else 0.0
    sub = 0.5 if (na and nb and (na in nb or nb in na)) else 0.0
    # character-level similarity catches the contractions football-data likes:
    # "Nott'm Forest", "M'gladbach", "Ein Frankfurt"
    chars = SequenceMatcher(None, fa, fb).ratio()
    # ...and compare the distinctive words directly, so "Monchengladbach"
    # still finds "M'gladbach" even with "Borussia" padding one side out
    tok = 0.0
    for wa in (w for w in fa.split() if len(w) >= 5):
        for wb in (w for w in fb.split() if len(w) >= 5):
            tok = max(tok, SequenceMatcher(None, wa, wb).ratio())
    # a close match on the distinctive word is strong evidence by itself:
    # "Monchengladbach" vs "gladbach" is the same club and nothing else in the
    # division comes near it
    tok_bonus = 0.6 if tok >= 0.65 else 0.0
    return (jac + pre + sub + 1.2 * max(0.0, chars - 0.35)
            + 1.0 * max(0.0, tok - 0.55) + tok_bonus)


def map_teams(feed_names: list[str], candidates: list[str],
              min_score: float = 0.5) -> tuple[dict, list]:
    """
    Greedy 1:1 assignment - each feed name takes its best free candidate,
    strongest pairs first, so a confident match can't be stolen by a weak one.
    """
    pairs = sorted(((_score(f, c), f, c) for f in feed_names for c in candidates),
                   key=lambda t: -t[0])
    used_f, used_c, out = set(), set(), {}
    for sc, f, c in pairs:
        if sc < min_score or f in used_f or c in used_c:
            continue
        out[f] = c
        used_f.add(f); used_c.add(c)
    return out, [f for f in feed_names if f not in out]


# ---------------------------------------------------------------------------
# consensus fair prices from many books
# ---------------------------------------------------------------------------

def devig_shin_row(odds: np.ndarray, iters: int = 60) -> np.ndarray:
    q = 1.0 / odds
    s = q.sum()
    z = float(np.clip((s - 1.0) / 2.0, 0.0, 0.35))
    for _ in range(iters):
        root = np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / s)
        p = (root - z) / (2 * (1 - z))
        z = float(np.clip(z + 0.55 * (p.sum() - 1.0), 0.0, 0.35))
    root = np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / s)
    p = (root - z) / (2 * (1 - z))
    return p / p.sum()


def consensus_1x2(g: pd.DataFrame, home: str, away: str):
    """Median de-vigged view across books, plus the best price on each side."""
    rows, best = [], {"home": (0, None), "draw": (0, None), "away": (0, None)}
    for book, gb in g[g.market == "h2h"].groupby("book"):
        px = {}
        for _, r in gb.iterrows():
            if r.outcome == home:
                px["home"] = r.price
            elif r.outcome == away:
                px["away"] = r.price
            elif str(r.outcome).lower() == "draw":
                px["draw"] = r.price
        if len(px) != 3 or min(px.values()) <= 1.0:
            continue
        rows.append(devig_shin_row(np.array([px["home"], px["draw"], px["away"]])))
        for k in ("home", "draw", "away"):
            if px[k] > best[k][0]:
                best[k] = (px[k], book)
    if not rows:
        return None, best, 0
    fair = np.median(np.vstack(rows), axis=0)
    return fair / fair.sum(), best, len(rows)


def consensus_totals(g: pd.DataFrame):
    """{line: (fair_over, best_over_price, book, best_under_price, book)}"""
    out = {}
    t = g[g.market == "totals"]
    for line, gl in t.groupby("point"):
        rows, bo, bu = [], (0, None), (0, None)
        for book, gb in gl.groupby("book"):
            o = gb[gb.outcome.str.lower() == "over"].price
            u = gb[gb.outcome.str.lower() == "under"].price
            if not len(o) or not len(u):
                continue
            o, u = float(o.iloc[0]), float(u.iloc[0])
            if o <= 1 or u <= 1:
                continue
            rows.append(devig_shin_row(np.array([o, u])))
            if o > bo[0]:
                bo = (o, book)
            if u > bu[0]:
                bu = (u, book)
        if rows:
            fair = np.median(np.vstack(rows), axis=0)
            fair = fair / fair.sum()
            out[float(line)] = (float(fair[0]), bo, bu, len(rows))
    return out


# ---------------------------------------------------------------------------
# the scoreline matrix the market itself implies
# ---------------------------------------------------------------------------

MAX = 10
_K = np.arange(MAX + 1)
_LF = gammaln(_K + 1)


def _poisson_matrix(lh: float, la: float) -> np.ndarray:
    ph = np.exp(-lh + _K * np.log(lh) - _LF)
    pa = np.exp(-la + _K * np.log(la) - _LF)
    M = np.outer(ph, pa)
    return M / M.sum()


def _p_over(T: float, line: float) -> float:
    n = int(np.floor(line))
    k = np.arange(n + 1)
    return float(1.0 - np.exp(-T + k * np.log(T) - gammaln(k + 1)).sum())


def implied_matrix(fair_1x2, totals, rho: float = -0.03) -> np.ndarray | None:
    """
    Books build correct-score off a supremacy and a total. So do we - it beat
    our own model on real scorelines by a clear margin, so this is the better
    shape to price from.
    """
    if fair_1x2 is None:
        return None
    # total: prefer the 2.5 line, else whichever is closest to it
    if totals:
        line = min(totals, key=lambda L: abs(L - 2.5))
        target_over = totals[line][0]
        lo, hi = 0.05, 9.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if _p_over(mid, line) < target_over:
                lo = mid
            else:
                hi = mid
        T = (lo + hi) / 2
    else:
        T = 2.6
    target = fair_1x2[0] - fair_1x2[2]
    lo, hi = -T * 0.95, T * 0.95
    for _ in range(60):
        mid = (lo + hi) / 2
        M = _poisson_matrix(max((T + mid) / 2, 1e-6), max((T - mid) / 2, 1e-6))
        gd = np.subtract.outer(_K, _K)
        if float(M[gd > 0].sum() - M[gd < 0].sum()) < target:
            lo = mid
        else:
            hi = mid
    S = (lo + hi) / 2
    lh, la = max((T + S) / 2, 1e-6), max((T - S) / 2, 1e-6)
    M = _poisson_matrix(lh, la)
    M[0, 0] *= 1 - lh * la * rho
    M[0, 1] *= 1 + lh * rho
    M[1, 0] *= 1 + la * rho
    M[1, 1] *= 1 - rho
    M = np.clip(M, 1e-15, None)
    return M / M.sum()

