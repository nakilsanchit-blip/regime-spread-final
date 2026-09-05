"""Distress model sanity check on the synthetic panel: it should find the planted hazard."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import survival
import fetch_edgar


def test_finds_planted_hazard():
    p = survival.features(fetch_edgar.synthetic(n_companies=1500, y0=2010, y1=2022))
    bt = survival.walk_forward(p, ["burning", "short_runway", "neg_equity", "going_concern"])
    s = bt["summary"]
    assert s["auc_state"] > 0.55, s
    assert s["logloss_gain_ci90"][0] > 0, s


if __name__ == "__main__":
    test_finds_planted_hazard(); print("ok")
