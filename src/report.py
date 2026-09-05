"""Assemble everything into one JSON blob and bake it into the dashboard."""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from .model import BucketSpec, StateConditionalModel

ROOT = os.path.dirname(os.path.dirname(__file__))


def build_results(feat: pd.DataFrame, cfg: dict, states: list[str], bt_summary: dict, source: str,
                  cluster: float = 1.0, focal: pd.DataFrame | None = None, universe: dict | None = None) -> dict:
    h = cfg["horizon"]
    b = BucketSpec(cfg["bucket_edges_pct"])

    # Final model for "today": trained on every row whose outcome is known.
    known = feat["fwd_ret"].notna() & feat["state"].notna()
    final = StateConditionalModel(b, states, h, cfg["prior_strength"], cluster).fit(
        feat.loc[known, "state"], feat.loc[known, "fwd_ret"], feat.loc[known, "ticker"] if "ticker" in feat else None)

    # "Today" is the focal stock's latest row (its own history in the single
    # stock case; the pooled model applied to it in the universe case).
    src = focal if focal is not None else feat
    today = src.index[-1]
    today_state = src["state"].iloc[-1]
    price = float(src["close"].iloc[-1])
    pred = final.predict(today_state) if isinstance(today_state, str) else None

    state_table = []
    for i, s in enumerate(states):
        state_table.append({
            "state": s,
            "rows": int(final.n_rows[i]),
            "independent_obs": round(float(final.n_rows[i] / (h * cluster)), 1),
            "p_up": round(float(final.p[i][b.up_mask()].sum()), 3),
            "ev": round(final.moments(i)["ev"], 4),
            "std": round(final.moments(i)["std"], 4),
        })

    factor_values = {f: bool(src[f].iloc[-1]) for f in cfg["factors"]}

    # Survivorship check: refit on the names that are still trading and compare.
    surv = None
    if universe and "ticker" in feat and universe.get("active_tickers"):
        keep = known & feat["ticker"].isin(universe["active_tickers"])
        if keep.sum() > 10 * h:
            sm = StateConditionalModel(b, states, h, cfg["prior_strength"], cluster).fit(
                feat.loc[keep, "state"], feat.loc[keep, "fwd_ret"], feat.loc[keep, "ticker"])
            up = b.up_mask()
            surv = {
                "baseline": {"survivors": {"p_up": float(sm.q[up].sum()), "ev": sm.moments(None)["ev"], "rows": int(sm.n_rows.sum())},
                             "all": {"p_up": float(final.q[up].sum()), "ev": final.moments(None)["ev"], "rows": int(final.n_rows.sum())}},
                "states": [{"state": st,
                            "survivors": {"p_up": float(sm.p[i][up].sum()), "ev": sm.moments(i)["ev"]},
                            "all": {"p_up": float(final.p[i][up].sum()), "ev": final.moments(i)["ev"]}}
                           for i, st in enumerate(states)],
            }

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source": source,
        "ticker": cfg["ticker"],
        "sector": cfg["sector"],
        "horizon": h,
        "relative": bool(cfg.get("relative", False)),
        "factors": cfg["factors"],
        "factor_values": factor_values,
        "as_of": str(today.date()),
        "price": round(price, 2),
        "data_start": str(feat.index[0].date()),
        "data_end": str(today.date()),
        "universe": universe,
        "survivorship_check": surv,
        "cluster": cluster,
        "buckets_pct": b.labels_pct(),
        "buckets_dollars": b.labels_dollars(price),
        "today": pred,
        "unconditional": final.q.tolist(),
        "unconditional_moments": final.moments(None),
        "states": states,
        "state_table": state_table,
        "transition": np.round(final.transition, 3).tolist(),
        "transition_counts": final.transition_counts.astype(int).tolist(),
        "backtest": bt_summary,
        "config": {k: v for k, v in cfg.items() if k not in ("ticker", "sector")},
    }


def write(results: dict, out_dir: str | None = None) -> tuple[str, str]:
    out_dir = out_dir or os.path.join(ROOT, "output")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)

    tpl_path = os.path.join(ROOT, "web", "template.html")
    with open(tpl_path) as f:
        html = f.read()
    payload = json.dumps(results, default=_json_default).replace("</", "<\\/")
    html = html.replace("/*__RESULTS__*/null", payload)
    html_path = os.path.join(out_dir, "dashboard.html")
    with open(html_path, "w") as f:
        f.write(html)
    return json_path, html_path


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    raise TypeError(str(type(o)))
