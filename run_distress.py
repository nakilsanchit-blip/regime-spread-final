"""Usage:
    python fetch_edgar.py --synthetic          (or the live version, see that file)
    python run_distress.py
    python run_distress.py --factors burning,neg_equity,going_concern,micro

Writes output/distress.json, which build_terminal.py picks up for the Distress tab.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from src import survival

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=os.path.join(ROOT, "data", "edgar", "panel.csv"))
    ap.add_argument("--factors", default="burning,short_runway,neg_equity,going_concern")
    ap.add_argument("--prior", type=float, default=20.0)
    a = ap.parse_args()

    if not os.path.exists(a.panel):
        raise SystemExit("No panel. Run fetch_edgar.py first.")
    panel = pd.read_csv(a.panel)
    meta_path = os.path.join(os.path.dirname(a.panel), "_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"source": "unknown"}
    factors = [f.strip() for f in a.factors.split(",")]
    if "going_concern" in factors and panel["going_concern"].isna().all():
        print("panel has no going-concern flag (fetch with --going-concern); dropping that factor")
        factors = [f for f in factors if f != "going_concern"]

    p = survival.features(panel)
    p = p.dropna(subset=["assets", "liabilities", "cash", "ocf"])
    print(f"{meta['source']}: {len(p)} company-years, {p['cik'].nunique()} companies, "
          f"{int(p['year'].min())}-{int(p['year'].max())}, factors {factors}")

    bt = survival.walk_forward(p, factors, a.prior)
    s = bt["summary"]
    print(f"\nWalk-forward by cohort ({s['years'][0]}-{s['years'][-1]}, {s['rows']} company-years):")
    print(f"  bad-exit base rate {s['bad_rate']:.1%}")
    print(f"  log loss   states {s['logloss_state']:.4f}  vs base rate {s['logloss_baseline']:.4f}"
          f"   gain {s['logloss_gain']:+.4f}  90% CI {s['logloss_gain_ci90']}")
    print(f"  AUC        states {s['auc_state']:.3f}   logistic {s['auc_logit']:.3f}   (0.5 = coin flip)")
    print(f"  -> {s['verdict']}")

    final = survival.final_fit(p, factors, a.prior)
    print(f"\nLatest cohort {final['latest_year']}: {len(final['companies'])} companies scored")
    worst = sorted(final["companies"], key=lambda c: -c["p_bad_state"])[:5]
    for c in worst:
        print(f"  {c['name'][:32]:32s} {c['state']:40s} P(bad exit 2y) {c['p_bad_state']:.1%}")

    out = {"source": meta["source"], "factors": factors, "backtest": s, **final}
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
    path = os.path.join(ROOT, "output", "distress.json")
    with open(path, "w") as f:
        json.dump(out, f, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
