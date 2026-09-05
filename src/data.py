"""Price and factor data.

Two sources:
  * Yahoo Finance via `yfinance` (cached to data/ so you don't hammer the API).
  * A synthetic regime-switching generator, used by `--demo`, so the whole
    pipeline runs with no internet and so you can sanity-check the code on
    data whose true structure you know.

Everything downstream expects a daily DataFrame with these columns:
  close, volume, sector_close, vix, tnx
indexed by a DatetimeIndex, oldest first.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


# --------------------------------------------------------------------------- #
# Yahoo Finance
# --------------------------------------------------------------------------- #

def _extract(df: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    """yfinance changes its column layout between versions; handle both."""
    if isinstance(df.columns, pd.MultiIndex):
        if (field, ticker) in df.columns:
            return df[(field, ticker)]
        if (ticker, field) in df.columns:
            return df[(ticker, field)]
        raise KeyError(f"{field}/{ticker} not in download")
    return df[field]


def fetch_yahoo(ticker: str, sector: str, start: str, use_cache: bool = True) -> pd.DataFrame:
    cache_path = os.path.join(CACHE_DIR, f"{ticker}_{sector}_{start}.csv")
    if use_cache and os.path.exists(cache_path):
        age_days = (datetime.now().timestamp() - os.path.getmtime(cache_path)) / 86400
        if age_days < 1:
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    try:
        import yfinance as yf
    except ImportError as e:
        raise SystemExit("pip install yfinance   (or run with --demo)") from e

    symbols = [ticker, sector, "^VIX", "^TNX"]
    raw = yf.download(symbols, start=start, auto_adjust=True, progress=False, group_by="column")
    if raw is None or len(raw) == 0:
        raise SystemExit(f"No data returned for {symbols}. Check the tickers.")

    out = pd.DataFrame({
        "close": _extract(raw, "Close", ticker),
        "volume": _extract(raw, "Volume", ticker),
        "sector_close": _extract(raw, "Close", sector),
        "vix": _extract(raw, "Close", "^VIX"),
        "tnx": _extract(raw, "Close", "^TNX"),
    })
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.sort_index()
    # VIX/TNX occasionally have gaps on days the stock trades; carry forward,
    # which only ever uses *past* values so it doesn't leak the future.
    out[["vix", "tnx", "sector_close"]] = out[["vix", "tnx", "sector_close"]].ffill()
    out = out.dropna(subset=["close"])

    os.makedirs(CACHE_DIR, exist_ok=True)
    out.to_csv(cache_path)
    return out


# --------------------------------------------------------------------------- #
# Synthetic demo data
# --------------------------------------------------------------------------- #

def synthetic(n_days: int = 4000, seed: int = 7) -> pd.DataFrame:
    """Two hidden regimes (calm / stressed) drive the sector; the stock is the
    sector plus idiosyncratic noise. VIX tracks the true volatility with lag
    and noise. Rates are a slow random walk unrelated to anything.

    The stressed regime has a slightly negative drift, so a model that finds
    *some* directional edge here is behaving correctly, and the size of that
    edge tells you what "working" realistically looks like."""
    rng = np.random.default_rng(seed)

    # Regime chain. Calm lasts ~3 months on average, stressed ~5 weeks,
    # which is roughly what real equity regimes look like.
    P = np.array([[0.985, 0.015],
                  [0.040, 0.960]])
    regime = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        regime[t] = rng.choice(2, p=P[regime[t - 1]])

    mu = np.where(regime == 0, 0.0006, -0.0012)
    sig = np.where(regime == 0, 0.011, 0.028)
    sector_r = mu + sig * rng.standard_normal(n_days)
    idio = 0.012 * rng.standard_normal(n_days)
    stock_r = 1.2 * sector_r + idio

    sector = 100 * np.exp(np.cumsum(sector_r))
    stock = 100 * np.exp(np.cumsum(stock_r))

    # VIX: annualised vol of a leaky estimate of current sigma, plus noise.
    ewm_sig = pd.Series(sig).ewm(span=10).mean().values
    vix = ewm_sig * np.sqrt(252) * 100 * np.exp(0.08 * rng.standard_normal(n_days))

    tnx = 3.0 + np.cumsum(0.02 * rng.standard_normal(n_days))
    tnx = np.clip(tnx, 0.5, 8.0)

    volume = np.exp(np.log(5e7) + 0.4 * rng.standard_normal(n_days) + 0.8 * (regime == 1))

    idx = pd.bdate_range(end=pd.offsets.BDay().rollback(pd.Timestamp.today().normalize()), periods=n_days)
    return pd.DataFrame({
        "close": stock,
        "volume": volume,
        "sector_close": sector,
        "vix": vix,
        "tnx": tnx,
    }, index=idx)
