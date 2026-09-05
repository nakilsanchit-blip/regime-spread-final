"""Two sanity checks for the backtest. Run with `pytest` or `python -m tests.test_pipeline`.

1. Plant a strong state-dependent signal -> the walk-forward must find it.
2. Pure noise -> the walk-forward must NOT claim an edge.

If (2) ever fails, something is leaking the future into training.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import backtest, data, features  # noqa: E402

CFG = json.load(open(os.path.join(os.path.dirname(__file__), "..", "config.json")))
CFG["factors"] = ["vix_regime"]


def _feat():
    df = data.synthetic(n_days=3000, seed=3)
    feat = features.build(df, CFG)
    return feat, features.state_space(CFG)


def test_detects_planted_signal():
    feat, states = _feat()
    rng = np.random.default_rng(1)
    hi = feat["vix_regime"].values
    feat["fwd_ret"] = np.where(hi, -0.06, 0.04) + 0.05 * rng.standard_normal(len(feat))
    feat.iloc[-CFG["horizon"]:, feat.columns.get_loc("fwd_ret")] = np.nan
    s = backtest.summarize(backtest.walk_forward(feat, CFG, states), CFG, True)
    assert s["dev"]["logloss_gain"] > 0.05, s["dev"]
    assert s["dev"]["logloss_gain_ci90"][0] > 0, s["dev"]
    assert s["dev"]["direction_hit_model"] > 0.7


def test_no_edge_on_noise():
    feat, states = _feat()
    rng = np.random.default_rng(2)
    feat["fwd_ret"] = 0.08 * rng.standard_normal(len(feat))
    feat.iloc[-CFG["horizon"]:, feat.columns.get_loc("fwd_ret")] = np.nan
    s = backtest.summarize(backtest.walk_forward(feat, CFG, states), CFG, True)
    lo, hi = s["dev"]["logloss_gain_ci90"]
    assert lo <= 0 <= hi or abs(s["dev"]["logloss_gain"]) < 0.01, s["dev"]


if __name__ == "__main__":
    test_detects_planted_signal()
    test_no_edge_on_noise()
    print("ok")
