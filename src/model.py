"""State-conditional outcome model.

Given today's discrete state s, estimate P(next-H-day move lands in bucket k | s).

This is the "when the world looked like this, what happened next" idea, with
two guards against fooling yourself:

1. Shrinkage. Each state's empirical histogram is blended with the
   unconditional histogram. The blend weight depends on how many
   *independent* observations the state has, not raw row counts, because
   with a 21-day horizon and daily rows, 21 consecutive rows are basically
   one observation. A state with 3 independent observations gets almost no
   say; a state with 300 gets nearly full say.

2. Nothing here ever sees the future. `fit` is handed only rows the caller
   has already purged (see backtest.py).

Also builds the state-to-state transition matrix at horizon spacing, which is
the literal "Markov matrix" from the conversation. It's reported for
inspection; the outcome distribution above doesn't need it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class BucketSpec:
    def __init__(self, edges_pct: list[float]):
        e = sorted(edges_pct)
        self.edges = np.array([-np.inf] + [x / 100 for x in e] + [np.inf])
        self.n = len(self.edges) - 1

    def assign(self, ret: pd.Series) -> np.ndarray:
        """Bucket index for each return. NaN -> -1."""
        idx = np.searchsorted(self.edges, ret.values, side="right") - 1
        idx = np.where(np.isnan(ret.values), -1, idx)
        return idx

    def labels_pct(self) -> list[str]:
        out = []
        for lo, hi in zip(self.edges[:-1], self.edges[1:]):
            if lo == -np.inf:
                out.append(f"below {hi:+.0%}")
            elif hi == np.inf:
                out.append(f"above {lo:+.0%}")
            else:
                out.append(f"{lo:+.0%} to {hi:+.0%}")
        return out

    def labels_dollars(self, price: float) -> list[dict]:
        out = []
        for lo, hi in zip(self.edges[:-1], self.edges[1:]):
            out.append({
                "lo": None if lo == -np.inf else round(price * lo, 2),
                "hi": None if hi == np.inf else round(price * hi, 2),
                "lo_pct": None if lo == -np.inf else lo,
                "hi_pct": None if hi == np.inf else hi,
            })
        return out

    def up_mask(self) -> np.ndarray:
        return self.edges[:-1] >= 0


class StateConditionalModel:
    def __init__(self, buckets: BucketSpec, states: list[str], horizon: int, prior_strength: float,
                 cluster: float = 1.0):
        self.b = buckets
        self.states = states
        self.h = horizon
        self.alpha = prior_strength
        # Rows per independent observation = horizon x cluster. Cluster > 1
        # when many stocks share the same dates and their outcomes co-move
        # (pooled universe in plain mode). In relative mode the shared market
        # move is subtracted out, so cluster stays ~1.
        self.cluster = cluster
        self.q = None            # unconditional bucket probs
        self.counts = None       # raw row counts per state x bucket
        self.p = None            # shrunk conditional probs per state
        self.n_rows = None
        self.transition = None

    # ------------------------------------------------------------------ #
    def fit(self, state: pd.Series, fwd_ret: pd.Series, ticker: pd.Series | None = None) -> "StateConditionalModel":
        ok = (state.notna() & fwd_ret.notna()).values
        sidx = {name: i for i, name in enumerate(self.states)}
        S, K = len(self.states), self.b.n
        si = state.map(sidx).values
        k = self.b.assign(fwd_ret)
        r = fwd_ret.values

        si_ok = si[ok].astype(int)
        k_ok = k[ok].astype(int)
        r_ok = r[ok]
        flat = si_ok * K + k_ok
        counts = np.bincount(flat, minlength=S * K).reshape(S, K).astype(float)
        sum_r = np.bincount(flat, weights=r_ok, minlength=S * K).reshape(S, K)
        sum_r2 = np.bincount(flat, weights=r_ok * r_ok, minlength=S * K).reshape(S, K)

        total = counts.sum()
        if total == 0:
            raise ValueError("No training rows after purge; extend the start date.")
        self.q = counts.sum(axis=0) / total
        self.counts = counts
        self.n_rows = counts.sum(axis=1)

        # Unconditional within-bucket moments (what a "-5% to -2%" move
        # actually averaged, rather than assuming the midpoint).
        cb = counts.sum(axis=0)
        self.bucket_mean = np.divide(sum_r.sum(axis=0), cb, out=np.zeros(K), where=cb > 0)
        self.bucket_m2 = np.divide(sum_r2.sum(axis=0), cb, out=np.zeros(K), where=cb > 0)
        self._sum_r, self._sum_r2 = sum_r, sum_r2

        # Shrink each state's histogram toward q, weighting by independent obs.
        eff_n = self.n_rows / (self.h * self.cluster)
        p = np.zeros_like(counts)
        for i in range(S):
            emp = counts[i] / self.n_rows[i] if self.n_rows[i] > 0 else self.q
            w = eff_n[i] / (eff_n[i] + self.alpha)
            p[i] = w * emp + (1 - w) * self.q
        self.p = p

        self._fit_transition(state, ticker)
        return self

    def _fit_transition(self, state: pd.Series, ticker: pd.Series | None):
        """P(state at t+H | state at t). With a ticker column, the H-step-ahead
        state is taken within the same ticker (pooled rows are interleaved by
        date, so a plain shift would pair different companies)."""
        S = len(self.states)
        sidx = {name: i for i, name in enumerate(self.states)}
        cur = state.map(sidx)
        nxt = cur.groupby(ticker.values).shift(-self.h) if ticker is not None else cur.shift(-self.h)
        ok = cur.notna() & nxt.notna()
        a = cur[ok].values.astype(int)
        bb = nxt[ok].values.astype(int)
        T = np.bincount(a * S + bb, minlength=S * S).reshape(S, S).astype(float)
        marg = T.sum(axis=0)
        marg = marg / marg.sum() if marg.sum() > 0 else np.ones(S) / S
        rows = T.sum(axis=1, keepdims=True)
        eff = rows / (self.h * self.cluster)
        w = eff / (eff + self.alpha)
        emp = np.divide(T, rows, out=np.tile(marg, (S, 1)), where=rows > 0)
        self.transition = w * emp + (1 - w) * marg
        self.transition_counts = T

    # ------------------------------------------------------------------ #
    def moments(self, i: int | None) -> dict:
        """EV, std, and Kelly fraction for state i (None = unconditional),
        using the same shrinkage as the bucket probabilities. Everything is in
        return units (0.02 = 2%); the report converts to dollars."""
        if i is None:
            p, m, m2, w = self.q, self.bucket_mean, self.bucket_m2, 0.0
        else:
            p = self.p[i]
            eff_n = self.n_rows[i] / (self.h * self.cluster)
            w = eff_n / (eff_n + self.alpha)
            c = self.counts[i]
            emp_m = np.divide(self._sum_r[i], c, out=self.bucket_mean.copy(), where=c > 0)
            emp_m2 = np.divide(self._sum_r2[i], c, out=self.bucket_m2.copy(), where=c > 0)
            m = w * emp_m + (1 - w) * self.bucket_mean
            m2 = w * emp_m2 + (1 - w) * self.bucket_m2
        ev = float((p * m).sum())
        var = max(float((p * m2).sum()) - ev ** 2, 1e-12)
        std = var ** 0.5
        kelly = ev / var
        return {
            "ev": ev,
            "std": std,
            "ev_over_std": ev / std,
            "kelly_full": kelly,
            "kelly_half": kelly / 2,
            "bucket_mean": m.tolist(),          # avg move inside each bucket
            "bucket_contrib": (p * m).tolist(), # prob x avg move = EV piece
        }

    def predict(self, state: str) -> dict:
        i = self.states.index(state)
        eff_n = self.n_rows[i] / (self.h * self.cluster)
        w = eff_n / (eff_n + self.alpha)
        return {
            "moments": self.moments(i),
            "moments_baseline": self.moments(None),
            "state": state,
            "probs": self.p[i].tolist(),
            "baseline": self.q.tolist(),
            "rows": int(self.n_rows[i]),
            "independent_obs": round(float(eff_n), 1),
            "weight_on_state": round(float(w), 3),
            "p_up": float(self.p[i][self.b.up_mask()].sum()),
            "p_up_baseline": float(self.q[self.b.up_mask()].sum()),
        }

    def predict_matrix(self, states: pd.Series) -> np.ndarray:
        """Row of probs for each state in a series (NaN state -> baseline)."""
        sidx = {name: i for i, name in enumerate(self.states)}
        out = np.tile(self.q, (len(states), 1))
        for j, st in enumerate(states.values):
            if isinstance(st, str):
                out[j] = self.p[sidx[st]]
        return out
