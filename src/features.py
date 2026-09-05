"""Turn raw prices into (a) a discrete market *state* per day and (b) the
forward return we're trying to put a distribution on.

Lookahead rule: every factor on day t uses only data up to and including
day t. Rolling windows in pandas are trailing by default, so that holds as
long as nobody adds `.shift(-k)` or centred windows. The forward return is
the *only* thing that peeks ahead, and it is only ever used as a training
target (see backtest.py for how it's purged near the train/test boundary).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Each factor: name -> (function(df, cfg) -> boolean Series, (label_if_true, label_if_false))
# True/False is deliberately binary. More levels per factor = more states =
# fewer samples per state = overfitting. Add levels only with a lot of data.


def _realized_vol(close: pd.Series, window: int) -> pd.Series:
    return np.log(close).diff().rolling(window).std()


def stock_momentum(df, cfg):
    return df["close"].pct_change(cfg["momentum_lookback"]) > 0


def sector_momentum(df, cfg):
    return df["sector_close"].pct_change(cfg["momentum_lookback"]) > 0


def vol_regime(df, cfg):
    rv = _realized_vol(df["close"], cfg["vol_window"])
    return rv > rv.rolling(cfg["regime_window"]).median()


def vix_regime(df, cfg):
    return df["vix"] > df["vix"].rolling(cfg["regime_window"]).median()


def rate_trend(df, cfg):
    return df["tnx"].diff(cfg["momentum_lookback"]) > 0


def volume_regime(df, cfg):
    v = df["volume"].rolling(cfg["vol_window"]).mean()
    return v > v.rolling(cfg["regime_window"]).median()


FACTORS = {
    "stock_momentum": (stock_momentum, ("Up", "Down")),
    "sector_momentum": (sector_momentum, ("Up", "Down")),
    "vol_regime": (vol_regime, ("HighVol", "LowVol")),
    "vix_regime": (vix_regime, ("HighVIX", "LowVIX")),
    "rate_trend": (rate_trend, ("RatesUp", "RatesDown")),
    "volume_regime": (volume_regime, ("HiVolume", "LoVolume")),
}


def build(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Returns a DataFrame with one boolean column per active factor, a
    human-readable `state` string, and `fwd_ret` (NaN in the last `horizon`
    rows because the future hasn't happened yet)."""
    active = cfg["factors"]
    unknown = [f for f in active if f not in FACTORS]
    if unknown:
        raise ValueError(f"Unknown factors {unknown}. Choose from {list(FACTORS)}")

    out = pd.DataFrame(index=df.index)
    labels = []
    for name in active:
        fn, (yes, no) = FACTORS[name]
        flag = fn(df, cfg)
        out[name] = flag
        labels.append(flag.map({True: yes, False: no}))

    # Rows where any factor hasn't warmed up yet (rolling windows) are unusable.
    warm = pd.concat([df["close"].pct_change(cfg["momentum_lookback"]),
                      _realized_vol(df["close"], cfg["vol_window"]).rolling(cfg["regime_window"]).median()],
                     axis=1).notna().all(axis=1)
    out["state"] = pd.concat(labels, axis=1).agg("/".join, axis=1)
    out.loc[~warm, "state"] = np.nan

    h = cfg["horizon"]
    out["fwd_ret"] = df["close"].shift(-h) / df["close"] - 1
    if cfg.get("relative", False):
        # Excess return over the sector ETF. Removes the market/sector drift so
        # a long-run winner's up-tilt stops being "the answer".
        out["fwd_ret"] = out["fwd_ret"] - (df["sector_close"].shift(-h) / df["sector_close"] - 1)
    out["close"] = df["close"]
    return out


def state_space(cfg: dict) -> list[str]:
    """Every possible state string, in a fixed order, so matrices line up."""
    from itertools import product
    label_sets = [FACTORS[f][1] for f in cfg["factors"]]
    return ["/".join(combo) for combo in product(*label_sets)]
