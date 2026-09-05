"""Walk-forward evaluation. This is the part that tells you whether the
factors actually add anything, and it's built to make it hard to lie to
yourself:

* Expanding window, refit every `refit_every` days. On refit date T the
  model only sees rows whose *outcome* was known by T, i.e. rows with
  t + horizon <= T. Without that purge, the last `horizon` training rows
  share their future with the test rows and the score is inflated.

* The last `holdout_fraction` of dates is locked. The dev-period score is
  what you look at while tinkering; the holdout is what you look at once,
  at the end, with `--unlock-holdout`. If you tune until the holdout looks
  good, it stops being a holdout.

* Scores are proper scoring rules (log loss, Brier) on the full bucket
  distribution, compared against the unconditional baseline. Beating the
  baseline is the bar. "Accuracy" alone is not reported because a model
  that always says the most common bucket gets high accuracy for free.

* Because daily rows with a 21-day horizon overlap, the "model beats
  baseline" gap is bootstrapped in blocks of `horizon` days to get an
  honest confidence interval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import BucketSpec, StateConditionalModel


def cluster_factor(feat: pd.DataFrame, cfg: dict) -> float:
    """Rows per date, unless relative mode already removed the shared move."""
    if "ticker" not in feat or cfg.get("relative"):
        return 1.0
    return float(max(1.0, feat.groupby(level=0).size().mean()))


def walk_forward(feat: pd.DataFrame, cfg: dict, states: list[str]) -> pd.DataFrame:
    h = cfg["horizon"]
    b = BucketSpec(cfg["bucket_edges_pct"])
    known = feat["fwd_ret"].notna() & feat["state"].notna()
    dates = feat.index
    cluster = cluster_factor(feat, cfg)
    has_ticker = "ticker" in feat

    uniq = dates.unique().sort_values()          # pooled frames repeat dates
    first_fit = uniq[0] + pd.Timedelta(days=int(cfg["min_train_years"] * 365))
    refit_dates = [d for d in uniq[::cfg["refit_every"]] if d >= first_fit]
    outcome_known_at = dates + pd.tseries.offsets.BDay(h)   # computed once, not per refit
    if not refit_dates:
        raise ValueError("Not enough history for min_train_years; lower it or use an earlier start.")

    rows = []
    for i, T in enumerate(refit_dates):
        T_next = refit_dates[i + 1] if i + 1 < len(refit_dates) else uniq[-1] + pd.Timedelta(days=1)

        # Purged training set: outcome fully observed by T.
        train_mask = (outcome_known_at <= T) & known
        train = feat[train_mask]
        if train["state"].nunique() < 2:
            continue
        m = StateConditionalModel(b, states, h, cfg["prior_strength"], cluster).fit(
            train["state"], train["fwd_ret"], train["ticker"] if has_ticker else None)

        test_mask = (dates >= T) & (dates < T_next) & known
        test = feat[test_mask]
        if len(test) == 0:
            continue
        P = m.predict_matrix(test["state"])
        k = b.assign(test["fwd_ret"])
        up_mask = b.up_mask()
        ev_by_state = {st: m.moments(i)["ev"] for i, st in enumerate(states)}
        ev_base = m.moments(None)["ev"]

        for j, (d, kk) in enumerate(zip(test.index, k)):
            pm, pb = P[j], m.q
            onehot = np.zeros(b.n); onehot[kk] = 1
            rows.append({
                "date": d,
                "ticker": test["ticker"].iloc[j] if has_ticker else None,
                "state": test["state"].iloc[j],
                "bucket": int(kk),
                "ll_model": -np.log(max(pm[kk], 1e-12)),
                "ll_base": -np.log(max(pb[kk], 1e-12)),
                "brier_model": float(((pm - onehot) ** 2).sum()),
                "brier_base": float(((pb - onehot) ** 2).sum()),
                "p_up_model": float(pm[up_mask].sum()),
                "p_up_base": float(pb[up_mask].sum()),
                "went_up": bool(test["fwd_ret"].iloc[j] > 0),
                "ev_model": float(ev_by_state.get(test["state"].iloc[j], ev_base)),
                "ev_base": float(ev_base),
                "realized": float(test["fwd_ret"].iloc[j]),
            })
    if not rows:
        raise ValueError("Not enough history to score any out-of-sample rows.")
    res = pd.DataFrame(rows).set_index("date")
    cut = res.index[int(len(res) * (1 - cfg["holdout_fraction"]))]
    res["period"] = np.where(res.index < cut, "dev", "holdout")
    return res


def block_bootstrap_ci(x: np.ndarray, block: int, n_boot: int = 2000, seed: int = 0,
                       dates: np.ndarray | None = None) -> tuple[float, float]:
    """90% CI for the mean of an autocorrelated series via moving-block bootstrap.
    With `dates`, blocks are runs of consecutive *dates* and every row on those
    dates comes along, so pooled stocks sharing a month aren't treated as
    independent evidence."""
    rng = np.random.default_rng(seed)
    if dates is None:
        groups = [np.array([i]) for i in range(len(x))]
    else:
        order = np.argsort(dates, kind="stable")
        x, dates = x[order], dates[order]
        _, starts_idx = np.unique(dates, return_index=True)
        ends_idx = np.append(starts_idx[1:], len(x))
        groups = [np.arange(a, b) for a, b in zip(starts_idx, ends_idx)]
    n = len(groups)
    if n < 2 * block:
        return (float("nan"), float("nan"))
    n_blocks = int(np.ceil(n / block))
    starts = np.arange(0, n - block + 1)
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(starts, size=n_blocks)
        idx = np.concatenate([np.concatenate(groups[s:s + block]) for s in pick])
        means[i] = x[idx].mean()
    return (float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95)))


def calibration(res: pd.DataFrame, n_bins: int = 5) -> list[dict]:
    """Does 'P(up)=0.6' actually go up 60% of the time?"""
    if len(res) == 0:
        return []
    edges = np.quantile(res["p_up_model"], np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = res[(res["p_up_model"] > lo) & (res["p_up_model"] <= hi)]
        if len(sel) == 0:
            continue
        out.append({
            "predicted_p_up": round(float(sel["p_up_model"].mean()), 3),
            "realized_up_rate": round(float(sel["went_up"].mean()), 3),
            "rows": int(len(sel)),
        })
    return out


def track_record(res: pd.DataFrame, h: int) -> dict:
    """Non-overlapping months: on each date, what EV the model would have
    computed with data available then, and what actually happened."""
    sub = res.iloc[::h]
    rows = []
    for d, r in sub.iterrows():
        rows.append({
            "date": str(d.date()),
            "state": r["state"],
            "ev_model": round(float(r["ev_model"]), 4),
            "ev_base": round(float(r["ev_base"]), 4),
            "realized": round(float(r["realized"]), 4),
            "hit": bool((r["ev_model"] > 0) == (r["realized"] > 0)),
            "period": r["period"],
        })
    dev = [r for r in rows if r["period"] == "dev"]
    pos = [r for r in dev if r["ev_model"] > 0]
    neg = [r for r in dev if r["ev_model"] <= 0]
    def rate(rs): return round(sum(r["hit"] for r in rs) / len(rs), 3) if rs else None
    return {
        "months": rows,
        "summary": {
            "n": len(dev),
            "n_ev_positive": len(pos),
            "hit_when_positive": rate(pos),
            "hit_when_negative": rate(neg),
            "hit_overall": rate(dev),
            "always_up": round(sum(r["realized"] > 0 for r in dev) / len(dev), 3) if dev else None,
        },
    }


def summarize(res: pd.DataFrame, cfg: dict, unlock_holdout: bool, cluster: float = 1.0,
              focal: str | None = None) -> dict:
    h = cfg["horizon"]

    def score(sub: pd.DataFrame) -> dict:
        if len(sub) == 0:
            return {}
        gain_ll = (sub["ll_base"] - sub["ll_model"]).values
        lo, hi = block_bootstrap_ci(gain_ll, block=h, dates=sub.index.values)
        return {
            "rows": int(len(sub)),
            "independent_obs": round(len(sub) / (h * cluster), 1),
            "start": str(sub.index[0].date()),
            "end": str(sub.index[-1].date()),
            "logloss_model": round(float(sub["ll_model"].mean()), 4),
            "logloss_baseline": round(float(sub["ll_base"].mean()), 4),
            "logloss_gain": round(float(gain_ll.mean()), 4),
            "logloss_gain_ci90": [round(lo, 4), round(hi, 4)],
            "brier_model": round(float(sub["brier_model"].mean()), 4),
            "brier_baseline": round(float(sub["brier_base"].mean()), 4),
            "direction_hit_model": round(float(((sub["p_up_model"] > 0.5) == sub["went_up"]).mean()), 3),
            "direction_hit_baseline": round(float(((sub["p_up_base"] > 0.5) == sub["went_up"]).mean()), 3),
            "calibration": calibration(sub),
        }

    dev = res[res["period"] == "dev"]
    out = {"dev": score(dev)}
    if unlock_holdout:
        out["holdout"] = score(res[res["period"] == "holdout"])
    else:
        hold = res[res["period"] == "holdout"]
        out["holdout"] = {"locked": True,
                          "rows": int(len(hold)),
                          "start": str(hold.index[0].date()) if len(hold) else None,
                          "end": str(hold.index[-1].date()) if len(hold) else None}

    tr_res = res[res["ticker"] == focal] if focal and "ticker" in res and res["ticker"].notna().any() else res
    out["track_record"] = track_record(tr_res, h)

    # Plain-language verdict, based on dev period only.
    d = out["dev"]
    if d:
        lo, hi = d["logloss_gain_ci90"]
        if np.isnan(lo):
            verdict = "Too little out-of-sample data to say anything."
        elif lo > 0:
            verdict = ("The factors beat the unconditional baseline out of sample, and the 90% "
                       "interval on that gain excludes zero. A real but probably small edge.")
        elif hi < 0:
            verdict = ("The factors do worse than just using the unconditional distribution. "
                       "Right now the state split is adding noise, not information.")
        else:
            verdict = ("The factors' gain over baseline is indistinguishable from zero. "
                       "Treat the conditional distribution as roughly the unconditional one.")
        out["verdict"] = verdict
    return out
