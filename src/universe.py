"""Load a multi-ticker universe from data/universe and build pooled features.

Each ticker gets the same per-ticker factors as the single-stock path
(features.build), using its own price series, plus the shared market series
(sector ETF, VIX, 10y). Rows are then stacked with a `ticker` column.

A stock that stopped trading contributes rows up to the point where its
forward return is still computable. Its last `horizon` days have no forward
return and are dropped, so the very last leg of a collapse is not in the
target. If the price at delisting is far above zero this understates the
loss; a fuller treatment would append the delisting return, which needs the
terminal price from the data vendor.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from . import features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIVERSE_DIR = os.path.join(ROOT, "data", "universe")


def load(universe_dir: str = UNIVERSE_DIR) -> tuple[dict[str, pd.DataFrame], dict, pd.DataFrame | None]:
    meta_path = os.path.join(universe_dir, "_meta.json")
    if not os.path.exists(meta_path):
        raise SystemExit("No universe on disk. Run build_universe.py first.")
    meta = json.load(open(meta_path))
    tickers = {}
    for fp in glob.glob(os.path.join(universe_dir, "*.csv")):
        sym = os.path.basename(fp)[:-4]
        if sym.startswith("_"):
            continue
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        if len(df) >= 300:
            tickers[sym] = df
    market_path = os.path.join(universe_dir, "_market.csv")
    market = pd.read_csv(market_path, index_col=0, parse_dates=True) if os.path.exists(market_path) else None
    return tickers, meta, market


def fetch_market(sector: str, start: str) -> pd.DataFrame:
    """Sector ETF, VIX, 10y from Yahoo, for universes that don't ship their own."""
    try:
        import yfinance as yf
    except ImportError as e:
        raise SystemExit("pip install yfinance") from e
    raw = yf.download([sector, "^VIX", "^TNX"], start=start, auto_adjust=True, progress=False, group_by="column")
    from .data import _extract
    m = pd.DataFrame({
        "sector_close": _extract(raw, "Close", sector),
        "vix": _extract(raw, "Close", "^VIX"),
        "tnx": _extract(raw, "Close", "^TNX"),
    })
    m.index = pd.to_datetime(m.index).tz_localize(None)
    return m.sort_index().ffill()


def build_pooled(tickers: dict[str, pd.DataFrame], market: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    frames = []
    for sym, px in tickers.items():
        df = px.join(market, how="left")
        df[["sector_close", "vix", "tnx"]] = df[["sector_close", "vix", "tnx"]].ffill()
        df = df.dropna(subset=["close", "sector_close", "vix"])
        if len(df) < cfg["regime_window"] + cfg["horizon"] + 50:
            continue
        f = features.build(df, cfg)
        f["ticker"] = sym
        frames.append(f)
    if not frames:
        raise SystemExit("No ticker had enough overlap with the market series.")
    pooled = pd.concat(frames).sort_index(kind="stable")
    return pooled


def universe_stats(tickers: dict[str, pd.DataFrame], meta: dict) -> dict:
    fates = {}
    for sym in tickers:
        fate = meta.get("tickers", {}).get(sym, {}).get("fate", "unknown")
        fates[fate] = fates.get(fate, 0) + 1
    # how many names were alive each year
    years = {}
    for sym, df in tickers.items():
        for y in range(df.index[0].year, df.index[-1].year + 1):
            years[y] = years.get(y, 0) + 1
    return {
        "source": meta.get("source"),
        "n_tickers": len(tickers),
        "fates": fates,
        "active_tickers": [s for s in tickers if meta.get("tickers", {}).get(s, {}).get("fate") == "active"],
        "alive_by_year": [{"year": y, "n": n} for y, n in sorted(years.items())],
        "first_date": str(min(df.index[0] for df in tickers.values()).date()),
        "last_date": str(max(df.index[-1] for df in tickers.values()).date()),
    }
