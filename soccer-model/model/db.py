"""Load the raw match dump and shape it into fittable systems."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Divisions that share a promotion/relegation pyramid are fitted together, so a
# club that goes up or down carries its attack/defence rating across.
SYSTEMS: dict[str, list[str]] = {
    "ENG": ["E0", "E1", "E2", "E3", "EC"],
    "SCO": ["SC0", "SC1", "SC2", "SC3"],
    "ESP": ["SP1", "SP2"],
    "ITA": ["I1", "I2"],
    "GER": ["D1", "D2"],
    "FRA": ["F1", "F2"],
    "NED": ["N1"],
    "BEL": ["B1"],
    "POR": ["P1"],
    "TUR": ["T1"],
    "GRE": ["G1"],
}

DIV_NAMES = {
    "E0": "Premier League", "E1": "Championship", "E2": "League One",
    "E3": "League Two", "EC": "National League",
    "SC0": "Scottish Prem", "SC1": "Scottish Champ", "SC2": "Scottish L1",
    "SC3": "Scottish L2",
    "SP1": "La Liga", "SP2": "La Liga 2", "I1": "Serie A", "I2": "Serie B",
    "D1": "Bundesliga", "D2": "2. Bundesliga", "F1": "Ligue 1", "F2": "Ligue 2",
    "N1": "Eredivisie", "B1": "Belgian Pro", "P1": "Primeira Liga",
    "T1": "Super Lig", "G1": "Super League Greece",
}

# 1X2 price columns, best first. C = closing.
CLOSE_1X2 = [("PSCH", "PSCD", "PSCA"), ("MaxCH", "MaxCD", "MaxCA"),
             ("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA")]
OPEN_1X2 = [("MaxH", "MaxD", "MaxA"), ("PSH", "PSD", "PSA"),
            ("B365H", "B365D", "B365A"), ("AvgH", "AvgD", "AvgA")]


def load(path="data/matches.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df["date"].notna()].copy()
    df["Div"] = df["Div"].astype(str)
    for c in ("HomeTeam", "AwayTeam"):
        df[c] = df[c].astype(str).str.strip()
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df["system"] = df["Div"].map({d: s for s, ds in SYSTEMS.items() for d in ds})

    # Extra-league divisions arrive as "ARG: Liga Profesional". Everything from
    # one country is one system, so clubs moving between a league and its cup
    # competition keep a single rating.
    extra = df["system"].isna() & df["Div"].str.contains(":", regex=False)
    df.loc[extra, "system"] = df.loc[extra, "Div"].str.split(":").str[0].str.strip()
    return df.sort_values("date").reset_index(drop=True)


def first_available(df: pd.DataFrame, groups) -> tuple[np.ndarray, np.ndarray]:
    """Pick, per row, the first price triple whose three columns are all present."""
    out = np.full((len(df), 3), np.nan)
    which = np.full(len(df), -1)
    for gi, cols in enumerate(groups):
        if not all(c in df.columns for c in cols):
            continue
        vals = df[list(cols)].to_numpy(dtype=float)
        ok = np.isfinite(vals).all(axis=1) & (vals > 1.0).all(axis=1) & (which < 0)
        out[ok] = vals[ok]
        which[ok] = gi
    return out, which


def devig_shin(odds: np.ndarray, iters: int = 60) -> np.ndarray:
    """
    Shin (1993) de-vig. Assumes the overround comes from a proportion z of
    insider money, which removes more margin from longshots than from
    favourites - the right shape for football, where the naive proportional
    method leaves draws and away sides systematically overpriced.
    """
    q = 1.0 / odds
    s = q.sum(axis=1, keepdims=True)
    z = np.clip((s - 1.0) / 2.0, 0.0, 0.35)
    for _ in range(iters):
        root = np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / s)
        p = (root - z) / (2 * (1 - z))
        tot = p.sum(axis=1, keepdims=True)
        z = np.clip(z + 0.55 * (tot - 1.0), 0.0, 0.35)
    root = np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / s)
    p = (root - z) / (2 * (1 - z))
    return p / p.sum(axis=1, keepdims=True)


def devig_proportional(odds: np.ndarray) -> np.ndarray:
    q = 1.0 / odds
    return q / q.sum(axis=1, keepdims=True)


class SystemData:
    """Integer-indexed view of one pyramid, ready to hand to DixonColes."""

    def __init__(self, df: pd.DataFrame):
        # keep a handle on the caller's row labels so predictions can be joined
        # back to the full table for odds
        self.orig_index = df.index.to_numpy()
        self.df = df.reset_index(drop=True)
        teams = pd.Index(sorted(set(self.df.HomeTeam) | set(self.df.AwayTeam)))
        divs = pd.Index(sorted(self.df.Div.unique()))
        self.teams, self.divs = teams, divs
        self.hi = teams.get_indexer(self.df.HomeTeam).astype(np.intp)
        self.ai = teams.get_indexer(self.df.AwayTeam).astype(np.intp)
        self.di = divs.get_indexer(self.df.Div).astype(np.intp)
        self.hg = self.df.FTHG.to_numpy(float)
        self.ag = self.df.FTAG.to_numpy(float)
        self.dates = self.df["date"].to_numpy("datetime64[D]")

        # shot counts - the raw material for a chance-quality proxy
        def col(c):
            return (self.df[c].to_numpy(float) if c in self.df.columns
                    else np.full(len(self.df), np.nan))
        self.hs, self.as_ = col("HS"), col("AS")
        self.hst, self.ast = col("HST"), col("AST")
        self.has_shots = (np.isfinite(self.hs) & np.isfinite(self.as_)
                          & np.isfinite(self.hst) & np.isfinite(self.ast))

    def chance_quality(self, train_mask: np.ndarray, weight: float
                       ) -> tuple[np.ndarray, np.ndarray]:
        """
        A cheap stand-in for xG, fitted only on matches the model is allowed
        to see: goals ~ a*(on target) + b*(off target), by least squares, then
        blended with actual goals.

        It is cruder than real xG - no shot locations, no chance quality - but
        it separates chances created from finishing, which is the whole reason
        xG predicts better than goals do. If this moves nothing, real xG very
        likely wouldn't either.
        """
        m = train_mask & self.has_shots
        if m.sum() < 200 or weight <= 0:
            return self.hg.copy(), self.ag.copy()

        # stack home and away rows into one regression
        on = np.concatenate([self.hst[m], self.ast[m]])
        off = np.concatenate([self.hs[m] - self.hst[m], self.as_[m] - self.ast[m]])
        y = np.concatenate([self.hg[m], self.ag[m]])
        X = np.column_stack([on, off])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, b = float(coef[0]), float(coef[1])

        ph = a * self.hst + b * (self.hs - self.hst)
        pa = a * self.ast + b * (self.as_ - self.ast)
        ph = np.clip(ph, 0.02, 8.0)
        pa = np.clip(pa, 0.02, 8.0)
        use = self.has_shots
        return (np.where(use, weight * ph + (1 - weight) * self.hg, self.hg),
                np.where(use, weight * pa + (1 - weight) * self.ag, self.ag))

    def weights(self, asof: np.datetime64, half_life_days: float) -> np.ndarray:
        age = (asof - self.dates).astype("timedelta64[D]").astype(float)
        xi = np.log(2.0) / half_life_days
        return np.exp(-xi * np.maximum(age, 0.0))

    def appearances(self, mask: np.ndarray) -> np.ndarray:
        """Matches played per team within `mask` - drives the confidence gate."""
        n = len(self.teams)
        return np.bincount(self.hi[mask], minlength=n) + np.bincount(self.ai[mask], minlength=n)

