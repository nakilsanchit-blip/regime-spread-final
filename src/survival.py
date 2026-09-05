"""Distress model on an EDGAR company-year panel.

Input: data/edgar/panel.csv, one row per company per 10-K year:
  cik, name, ticker, year, assets, liabilities, equity, cash, revenue,
  net_income, ocf, current_assets, current_liabilities, going_concern,
  outcome, outcome_known

`outcome` is what happened within the next two years: filing, acquired,
bankrupt, deregistered. `outcome_known` is False for the most recent cohorts
where two years haven't passed yet (those rows are scored, never trained on).

Two models, both walk-forward by cohort year:
  * State model: binary factors -> 2^k states -> shrunk outcome histogram.
    The Markov-matrix idea, with "what did the filings look like" as the state
    and "still here / acquired / bankrupt / deregistered" as the outcome.
  * Logistic model: continuous features -> P(bad exit). The boring baseline
    every distress paper starts from. If the state model can't beat it, the
    states aren't adding anything.

Design rule that matters more than the models: features for cohort year Y
come only from the Y filing. The model never sees who died. Cohorts are only
trained on once their two-year window has closed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOMES = ["filing", "acquired", "bankrupt", "deregistered"]
BAD = {"bankrupt", "deregistered"}


# --------------------------------------------------------------------------- #
# Features and states
# --------------------------------------------------------------------------- #

def features(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    eps = 1e-9
    a = p["assets"].clip(lower=1.0)
    p["log_assets"] = np.log10(a)
    p["leverage"] = p["liabilities"] / a
    p["cash_ratio"] = p["cash"] / a
    p["roa"] = p["net_income"] / a
    p["ocf_assets"] = p["ocf"] / a
    p["current_ratio"] = p["current_assets"] / p["current_liabilities"].clip(lower=1.0)
    p["neg_equity"] = p["equity"] < 0
    burn = (-p["ocf"]).clip(lower=0)
    p["runway_years"] = np.where(burn > 0, p["cash"] / (burn + eps), np.inf)
    p["going_concern"] = p["going_concern"].fillna(0).astype(int)
    return p


FACTORS = {
    # name: (function -> bool Series, (label_true, label_false))
    "burning": (lambda p: p["ocf"] < 0, ("Burning", "CashPositive")),
    "short_runway": (lambda p: p["runway_years"] < 1.0, ("Runway<1y", "Runway>1y")),
    "neg_equity": (lambda p: p["neg_equity"], ("NegEquity", "PosEquity")),
    "high_leverage": (lambda p: p["leverage"] > 0.8, ("HighLev", "LowLev")),
    "going_concern": (lambda p: p["going_concern"] == 1, ("GoingConcern", "NoGC")),
    "micro": (lambda p: p["assets"] < 50e6, ("Micro", "NotMicro")),
    "loss": (lambda p: p["net_income"] < 0, ("Loss", "Profit")),
}

CONTINUOUS = ["log_assets", "leverage", "cash_ratio", "roa", "ocf_assets", "current_ratio", "going_concern"]


def assign_states(p: pd.DataFrame, factor_names: list[str]) -> pd.Series:
    labels = []
    for f in factor_names:
        fn, (yes, no) = FACTORS[f]
        labels.append(fn(p).map({True: yes, False: no}))
    return pd.concat(labels, axis=1).agg("/".join, axis=1)


def state_space(factor_names: list[str]) -> list[str]:
    from itertools import product
    return ["/".join(c) for c in product(*[FACTORS[f][1] for f in factor_names])]


# --------------------------------------------------------------------------- #
# State model (categorical outcomes, shrunk toward the pooled histogram)
# --------------------------------------------------------------------------- #

class StateOutcomeModel:
    def __init__(self, states: list[str], prior_strength: float = 20.0):
        self.states = states
        self.alpha = prior_strength

    def fit(self, state: pd.Series, outcome: pd.Series):
        S, K = len(self.states), len(OUTCOMES)
        si = state.map({s: i for i, s in enumerate(self.states)}).values.astype(int)
        oi = outcome.map({o: i for i, o in enumerate(OUTCOMES)}).values.astype(int)
        counts = np.bincount(si * K + oi, minlength=S * K).reshape(S, K).astype(float)
        self.counts = counts
        self.n = counts.sum(axis=1)
        self.q = counts.sum(axis=0) / counts.sum()
        p = np.zeros_like(counts)
        for i in range(S):
            emp = counts[i] / self.n[i] if self.n[i] > 0 else self.q
            w = self.n[i] / (self.n[i] + self.alpha)     # company-years are ~independent here
            p[i] = w * emp + (1 - w) * self.q
        self.p = p
        # transition between states year to year is computed by the caller
        return self

    def predict(self, state: pd.Series) -> np.ndarray:
        idx = state.map({s: i for i, s in enumerate(self.states)})
        out = np.tile(self.q, (len(state), 1))
        ok = idx.notna().values
        out[ok] = self.p[idx[ok].values.astype(int)]
        return out


def state_transitions(p: pd.DataFrame, states: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Year-over-year state transitions within a company, plus exits.
    Returns (matrix S x (S+1), counts) where the last column is 'exited'."""
    S = len(states)
    sidx = {s: i for i, s in enumerate(states)}
    T = np.zeros((S, S + 1))
    p = p.sort_values(["cik", "year"])
    for cik, g in p.groupby("cik"):
        yrs = g["year"].values
        sts = g["state"].values
        outs = g["outcome"].values
        known = g["outcome_known"].values
        for j in range(len(g)):
            a = sidx[sts[j]]
            nxt = np.where(yrs == yrs[j] + 1)[0]
            if len(nxt):
                T[a, sidx[sts[nxt[0]]]] += 1
            elif known[j] and outs[j] != "filing":
                T[a, S] += 1
    rows = T.sum(axis=1, keepdims=True)
    P = np.divide(T, rows, out=np.zeros_like(T), where=rows > 0)
    return P, T


# --------------------------------------------------------------------------- #
# Logistic model on continuous features (numpy, L2, Newton steps)
# --------------------------------------------------------------------------- #

class Logistic:
    def __init__(self, l2: float = 1.0):
        self.l2 = l2

    def _prep(self, X: np.ndarray, fit: bool):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if fit:
            self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        Z = (X - self.mu) / self.sd
        Z = np.clip(Z, -5, 5)
        return np.hstack([np.ones((len(Z), 1)), Z])

    def fit(self, X: np.ndarray, y: np.ndarray, iters: int = 25):
        Z = self._prep(X, fit=True)
        w = np.zeros(Z.shape[1])
        reg = np.full(Z.shape[1], self.l2); reg[0] = 0
        for _ in range(iters):
            pr = 1 / (1 + np.exp(-Z @ w))
            g = Z.T @ (pr - y) + reg * w
            H = (Z * (pr * (1 - pr))[:, None]).T @ Z + np.diag(reg)
            w -= np.linalg.solve(H, g)
        self.w = w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = self._prep(X, fit=False)
        return 1 / (1 + np.exp(-Z @ self.w))


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank AUC: P(random bad-exit scores higher than random survivor)."""
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# --------------------------------------------------------------------------- #
# Walk-forward by cohort year
# --------------------------------------------------------------------------- #

def walk_forward(p: pd.DataFrame, factor_names: list[str], prior_strength: float = 20.0,
                 min_train_years: int = 3, horizon_years: int = 2) -> dict:
    states = state_space(factor_names)
    p = p.copy()
    p["state"] = assign_states(p, factor_names)
    years = sorted(p["year"].unique())
    rows = []
    for Y in years:
        train = p[(p["year"] <= Y - horizon_years) & p["outcome_known"]]
        test = p[(p["year"] == Y) & p["outcome_known"]]
        if train["year"].nunique() < min_train_years or len(test) == 0:
            continue
        sm = StateOutcomeModel(states, prior_strength).fit(train["state"], train["outcome"])
        P = sm.predict(test["state"])
        lg = Logistic().fit(train[CONTINUOUS].values.astype(float),
                            train["outcome"].isin(BAD).values.astype(float))
        p_bad_lg = lg.predict(test[CONTINUOUS].values.astype(float))
        oi = test["outcome"].map({o: i for i, o in enumerate(OUTCOMES)}).values.astype(int)
        y_bad = test["outcome"].isin(BAD).values.astype(int)
        bad_idx = [OUTCOMES.index(o) for o in BAD]
        for j in range(len(test)):
            rows.append({
                "year": Y, "cik": test["cik"].iloc[j], "state": test["state"].iloc[j],
                "outcome": test["outcome"].iloc[j],
                "ll_state": -np.log(max(P[j, oi[j]], 1e-12)),
                "ll_base": -np.log(max(sm.q[oi[j]], 1e-12)),
                "p_bad_state": float(P[j, bad_idx].sum()),
                "p_bad_base": float(sm.q[bad_idx].sum()),
                "p_bad_logit": float(p_bad_lg[j]),
                "y_bad": int(y_bad[j]),
            })
    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("Not enough cohort years with known outcomes to backtest.")

    # Score. Bootstrap over cohort years (the natural independent unit).
    yrs = res["year"].unique()
    gain = (res["ll_base"] - res["ll_state"]).values
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        pick = rng.choice(yrs, size=len(yrs))
        boots.append(np.mean(np.concatenate([gain[res["year"].values == y] for y in pick])))
    lo, hi = np.quantile(boots, [0.05, 0.95])

    by_year = []
    for y in yrs:
        s = res[res["year"] == y]
        by_year.append({
            "year": int(y), "n": int(len(s)), "bad_rate": round(float(s["y_bad"].mean()), 4),
            "auc_state": round(auc(s["p_bad_state"].values, s["y_bad"].values), 3),
            "auc_logit": round(auc(s["p_bad_logit"].values, s["y_bad"].values), 3),
        })
    summary = {
        "rows": int(len(res)), "years": [int(y) for y in yrs],
        "logloss_state": round(float(res["ll_state"].mean()), 4),
        "logloss_baseline": round(float(res["ll_base"].mean()), 4),
        "logloss_gain": round(float(gain.mean()), 4),
        "logloss_gain_ci90": [round(float(lo), 4), round(float(hi), 4)],
        "auc_state": round(auc(res["p_bad_state"].values, res["y_bad"].values), 3),
        "auc_logit": round(auc(res["p_bad_logit"].values, res["y_bad"].values), 3),
        "auc_baseline": 0.5,
        "bad_rate": round(float(res["y_bad"].mean()), 4),
        "by_year": by_year,
        "calibration": _calibration(res),
    }
    if lo > 0:
        summary["verdict"] = ("The filing states predict exits better than the base rate, out of sample, "
                              "and the interval excludes zero.")
    elif hi < 0:
        summary["verdict"] = "The filing states do worse than the base rate. The state split is adding noise."
    else:
        summary["verdict"] = "The filing states' gain over the base rate is indistinguishable from zero."
    return {"summary": summary, "states": states, "results": res}


def _calibration(res: pd.DataFrame, n_bins: int = 5) -> list[dict]:
    edges = np.quantile(res["p_bad_state"], np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = res[(res["p_bad_state"] > lo) & (res["p_bad_state"] <= hi)]
        if len(s):
            out.append({"predicted": round(float(s["p_bad_state"].mean()), 4),
                        "realized": round(float(s["y_bad"].mean()), 4), "n": int(len(s))})
    return out


# --------------------------------------------------------------------------- #
# Final fit + per-company scoring for the latest cohort
# --------------------------------------------------------------------------- #

def final_fit(p: pd.DataFrame, factor_names: list[str], prior_strength: float = 20.0) -> dict:
    states = state_space(factor_names)
    p = p.copy()
    p["state"] = assign_states(p, factor_names)
    known = p[p["outcome_known"]]
    sm = StateOutcomeModel(states, prior_strength).fit(known["state"], known["outcome"])
    lg = Logistic().fit(known[CONTINUOUS].values.astype(float), known["outcome"].isin(BAD).values.astype(float))
    trans, trans_counts = state_transitions(p, states)

    latest_year = int(p["year"].max())
    latest = p[p["year"] == latest_year].copy()
    P = sm.predict(latest["state"])
    latest["p_bad_logit"] = lg.predict(latest[CONTINUOUS].values.astype(float))
    bad_idx = [OUTCOMES.index(o) for o in BAD]
    companies = []
    for j, (_, r) in enumerate(latest.iterrows()):
        companies.append({
            "cik": int(r["cik"]), "name": r["name"],
            "ticker": r["ticker"] if isinstance(r.get("ticker"), str) else "",
            "year": latest_year, "state": r["state"],
            "probs": {o: round(float(P[j, i]), 4) for i, o in enumerate(OUTCOMES)},
            "p_bad_state": round(float(P[j, bad_idx].sum()), 4),
            "p_bad_logit": round(float(r["p_bad_logit"]), 4),
            "features": {k: (None if not np.isfinite(r[k]) else round(float(r[k]), 4))
                         for k in ["log_assets", "leverage", "cash_ratio", "roa", "ocf_assets",
                                   "current_ratio", "runway_years"]},
            "going_concern": int(r["going_concern"]),
        })
    state_table = [{
        "state": s, "n": int(sm.n[i]),
        "probs": {o: round(float(sm.p[i][k]), 4) for k, o in enumerate(OUTCOMES)},
        "p_bad": round(float(sm.p[i][bad_idx].sum()), 4),
    } for i, s in enumerate(states)]
    return {
        "states": states, "outcomes": OUTCOMES,
        "baseline": {o: round(float(sm.q[k]), 4) for k, o in enumerate(OUTCOMES)},
        "state_table": state_table,
        "transition": np.round(trans, 3).tolist(),
        "transition_counts": trans_counts.astype(int).tolist(),
        "latest_year": latest_year,
        "companies": companies,
        "logit_weights": {k: round(float(w), 3) for k, w in zip(["intercept"] + CONTINUOUS, lg.w)},
    }
