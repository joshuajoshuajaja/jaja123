"""
Time-decayed Dixon-Coles with a shared attack/defence pool across a country's
divisions, so a promoted or relegated club carries its rating with it.

For one "system" (a country's pyramid):

    log lam_home = mu[d] + hfa[d] + att[i] - dfn[j]
    log lam_away = mu[d]          + att[j] - dfn[i]

with the Dixon-Coles tau correction on the four lowest scorelines, an
exponential time-decay weight on each historical match, and a ridge penalty on
att/dfn that both fixes identifiability and does the shrinkage: a club with
eight matches of history gets pulled most of the way to its division's mean.

Fitted by L-BFGS-B with an analytic gradient - fast enough to refit inside a
walk-forward backtest a thousand times over.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

MAXGOALS = 10


# ---------------------------------------------------------------------------
# tau: the Dixon-Coles low-score correction
# ---------------------------------------------------------------------------

def _tau_parts(x, y, lh, la, rho):
    """tau and its partials, evaluated only where it isn't 1."""
    tau = np.ones_like(lh)
    dl_h = np.zeros_like(lh)   # d log tau / d lam_home
    dl_a = np.zeros_like(lh)
    dl_r = np.zeros_like(lh)   # d log tau / d rho

    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    if m00.any():
        t = 1.0 - lh[m00] * la[m00] * rho
        tau[m00] = t
        dl_h[m00] = -la[m00] * rho / t
        dl_a[m00] = -lh[m00] * rho / t
        dl_r[m00] = -lh[m00] * la[m00] / t
    if m01.any():
        t = 1.0 + lh[m01] * rho
        tau[m01] = t
        dl_h[m01] = rho / t
        dl_r[m01] = lh[m01] / t
    if m10.any():
        t = 1.0 + la[m10] * rho
        tau[m10] = t
        dl_a[m10] = rho / t
        dl_r[m10] = la[m10] / t
    if m11.any():
        t = 1.0 - rho
        tau[m11] = t
        dl_r[m11] = -1.0 / t
    return tau, dl_h, dl_a, dl_r


# ---------------------------------------------------------------------------

class DixonColes:
    def __init__(self, ridge: float = 0.02, rho_bound: float = 0.18,
                 fix_rho: float | None = None):
        """
        fix_rho: pin the low-score correction instead of estimating it. Needed
        when the response is a continuous goals/xG blend - tau's masks are
        defined on integer scorelines, so it can only be estimated from real
        goals. Fit rho once on goals, then pass it here.
        """
        self.ridge = ridge
        self.rho_bound = rho_bound
        self.fix_rho = fix_rho

    # -- packing -----------------------------------------------------------
    def _unpack(self, p):
        n, d = self.n_teams, self.n_divs
        att = p[:n]
        dfn = p[n:2 * n]
        mu = p[2 * n:2 * n + d]
        hfa = p[2 * n + d:2 * n + 2 * d]
        rho = p[-1]
        return att, dfn, mu, hfa, rho

    # -- objective ---------------------------------------------------------
    def _nll(self, p):
        att, dfn, mu, hfa, rho = self._unpack(p)
        hi, ai, di, x, y, w = self.hi, self.ai, self.di, self.x, self.y, self.w

        log_lh = mu[di] + hfa[di] + att[hi] - dfn[ai]
        log_la = mu[di] + att[ai] - dfn[hi]
        lh, la = np.exp(log_lh), np.exp(log_la)

        tau, dl_h, dl_a, dl_r = _tau_parts(x, y, lh, la, rho)
        if np.any(tau <= 1e-9):
            return 1e12, np.zeros_like(p)

        ll = w * (np.log(tau) + x * log_lh - lh + y * log_la - la)
        nll = -ll.sum() + self.ridge * (att @ att + dfn @ dfn)

        # gradient
        gh = w * ((x - lh) + lh * dl_h)     # d/d log_lam_home
        ga = w * ((y - la) + la * dl_a)

        n, d = self.n_teams, self.n_divs
        g_att = np.bincount(hi, gh, n) + np.bincount(ai, ga, n)
        g_dfn = -(np.bincount(ai, gh, n) + np.bincount(hi, ga, n))
        g_mu = np.bincount(di, gh + ga, d)
        g_hfa = np.bincount(di, gh, d)
        g_rho = float((w * dl_r).sum())

        grad = np.concatenate([
            -g_att + 2 * self.ridge * att,
            -g_dfn + 2 * self.ridge * dfn,
            -g_mu, -g_hfa, [-g_rho],
        ])
        return nll, grad

    # -- fit ---------------------------------------------------------------
    def fit(self, home_idx, away_idx, div_idx, hg, ag, weights,
            n_teams, n_divs, x0=None):
        self.hi = np.ascontiguousarray(home_idx, dtype=np.intp)
        self.ai = np.ascontiguousarray(away_idx, dtype=np.intp)
        self.di = np.ascontiguousarray(div_idx, dtype=np.intp)
        self.x = np.ascontiguousarray(hg, dtype=float)
        self.y = np.ascontiguousarray(ag, dtype=float)
        self.w = np.ascontiguousarray(weights, dtype=float)
        self.n_teams, self.n_divs = n_teams, n_divs

        if x0 is None:
            x0 = np.concatenate([
                np.zeros(n_teams), np.zeros(n_teams),
                np.full(n_divs, np.log(1.35)), np.full(n_divs, 0.25), [0.0],
            ])
        if self.fix_rho is None:
            rb = (-self.rho_bound, self.rho_bound)
        else:
            rb = (self.fix_rho, self.fix_rho)
            x0 = np.asarray(x0, dtype=float).copy()
            x0[-1] = self.fix_rho
        bounds = ([(-3, 3)] * (2 * n_teams)
                  + [(-2, 2)] * n_divs
                  + [(-1, 1)] * n_divs
                  + [rb])

        res = minimize(self._nll, x0, jac=True, method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": 400, "ftol": 1e-10})
        self.params_ = res.x
        self.att_, self.dfn_, self.mu_, self.hfa_, self.rho_ = self._unpack(res.x)
        self.result_ = res
        return self

    # -- predict -----------------------------------------------------------
    def rates(self, home_idx, away_idx, div_idx):
        lh = np.exp(self.mu_[div_idx] + self.hfa_[div_idx]
                    + self.att_[home_idx] - self.dfn_[away_idx])
        la = np.exp(self.mu_[div_idx] + self.att_[away_idx] - self.dfn_[home_idx])
        return lh, la


# ---------------------------------------------------------------------------
# scoreline matrix and everything derived from it
# ---------------------------------------------------------------------------

_K = np.arange(MAXGOALS + 1)
_LOGFACT = gammaln(_K + 1)


def score_matrix(lh: float, la: float, rho: float, maxgoals: int = MAXGOALS):
    """Joint P(home=x, away=y) as a (maxgoals+1, maxgoals+1) array."""
    k = _K[:maxgoals + 1]
    lf = _LOGFACT[:maxgoals + 1]
    ph = np.exp(-lh + k * np.log(lh) - lf)
    pa = np.exp(-la + k * np.log(la) - lf)
    M = np.outer(ph, pa)
    M[0, 0] *= 1.0 - lh * la * rho
    M[0, 1] *= 1.0 + lh * rho
    M[1, 0] *= 1.0 + la * rho
    M[1, 1] *= 1.0 - rho
    M = np.clip(M, 1e-15, None)
    return M / M.sum()


def markets(M: np.ndarray) -> dict:
    """Every market we price, read off one matrix."""
    n = M.shape[0]
    gd = np.subtract.outer(np.arange(n), np.arange(n))   # home - away
    tot = np.add.outer(np.arange(n), np.arange(n))

    out = {
        "home": float(M[gd > 0].sum()),
        "draw": float(np.trace(M)),
        "away": float(M[gd < 0].sum()),
        "btts": float(M[1:, 1:].sum()),
    }
    for line in (0.5, 1.5, 2.5, 3.5, 4.5):
        out[f"over{line}"] = float(M[tot > line].sum())
        out[f"under{line}"] = float(M[tot < line].sum())

    # Asian handicaps, home side, including quarter lines.
    for h in (-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
              0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0):
        out[f"ah{h:+g}"] = ah_prob(M, gd, h)
    return out


def ah_prob(M, gd, handicap: float) -> dict:
    """Win / push / lose for the home side at an Asian line (quarters split)."""
    if abs(handicap * 4 - round(handicap * 4)) > 1e-9:
        raise ValueError("handicap must be a multiple of 0.25")
    q = round(handicap * 4)
    if q % 2 != 0:                      # quarter line: half stake on each side
        lo, hi = (q - 1) / 4.0, (q + 1) / 4.0
        a, b = ah_prob(M, gd, lo), ah_prob(M, gd, hi)
        return {k: 0.5 * (a[k] + b[k]) for k in a}
    adj = gd + handicap
    return {"win": float(M[adj > 0].sum()),
            "push": float(M[np.isclose(adj, 0)].sum()),
            "lose": float(M[adj < 0].sum())}


def ah_fair_odds(p: dict) -> float:
    """Fair decimal price for an Asian line, pushes returned not lost."""
    live = p["win"] + p["lose"]
    return float("inf") if live <= 0 or p["win"] <= 0 else 1.0 + (p["lose"] / p["win"])

