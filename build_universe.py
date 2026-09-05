"""Build a multi-stock universe on disk, including names that no longer trade.

    python build_universe.py --source synthetic                   # fake 60-stock universe with deaths
    python build_universe.py --source kaggle --path ~/Downloads/Stocks --tickers universe.txt
    python build_universe.py --source fmp --tickers universe.txt --delisted 150 --start 2010-01-01

Output: data/universe/<TICKER>.csv  (date, close, volume)  +  data/universe/_meta.json

Then:  python run.py --universe --ticker NVDA --sector SPY --relative --serve

--- Sources ---

synthetic  No internet. Stocks have different lifetimes; some go into distress
           and get delisted, some get acquired at a premium. Use it to check the
           pipeline before spending API calls.

kaggle     The "Huge Stock Market Dataset" (Boris Marjanovic) on Kaggle: a
           folder of files like Stocks/aapl.us.txt with Date,Open,High,Low,Close,
           Volume,OpenInt. It covers ~7,000 US tickers to Nov 2017, and any file
           whose last date is well before the end of the dataset is a stock that
           stopped trading. Free, one download, no limits, but ends in 2017.

fmp        Financial Modeling Prep, free plan: 250 requests/day, no card.
           Set FMP_API_KEY in your environment. The delisted-companies endpoint
           gives symbol + delisting date; price history is fetched per ticker,
           one request each, and cached, so you can stop and rerun tomorrow.
           Free-plan history depth varies; the script reports what came back.

--- Ticker list file ---
One symbol per line. Lines starting with # are ignored. For fmp, delisted names
found via the API are added on top of this list. For kaggle, if you pass no
list, every file in the folder is used (that's ~7,000 tickers; fine).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "universe")


def read_ticker_file(path: str | None) -> list[str]:
    if not path:
        return []
    with open(path) as f:
        return [ln.strip().upper() for ln in f if ln.strip() and not ln.startswith("#")]


def write_ticker(sym: str, df: pd.DataFrame):
    df = df[["close", "volume"]].dropna(subset=["close"]).sort_index()
    df.index.name = "date"
    df.to_csv(os.path.join(OUT, f"{sym}.csv"))


def write_meta(source: str, tickers: dict):
    with open(os.path.join(OUT, "_meta.json"), "w") as f:
        json.dump({"source": source, "built": time.strftime("%Y-%m-%d"), "tickers": tickers}, f, indent=1)


# --------------------------------------------------------------------------- #
def build_synthetic(n_stocks: int = 80, n_days: int = 4000, seed: int = 11):
    """Same two-regime market as data.synthetic, but many stocks, each with its
    own start date, and a real hazard: each year a stock has a chance of
    entering distress (negative drift, high vol) and being delisted 3-9 months
    later. A few get acquired at a premium instead. Stocks in distress tend to
    show falling momentum and high realized vol first, so a pooled model that
    finds a genuine edge in Down/HighVol states is behaving correctly here."""
    rng = np.random.default_rng(seed)
    P = np.array([[0.985, 0.015], [0.040, 0.960]])
    regime = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        regime[t] = rng.choice(2, p=P[regime[t - 1]])
    mu = np.where(regime == 0, 0.0004, -0.0010)
    sig = np.where(regime == 0, 0.010, 0.025)
    market_r = mu + sig * rng.standard_normal(n_days)
    idx = pd.bdate_range(end=pd.offsets.BDay().rollback(pd.Timestamp.today().normalize()), periods=n_days)

    meta = {}
    for i in range(n_stocks):
        sym = f"S{i:03d}"
        start = int(rng.integers(0, n_days // 2)) if i > 10 else 0
        r = np.full(n_days, np.nan)
        beta = rng.uniform(0.7, 1.5)
        idio = rng.uniform(0.010, 0.020)
        alive = True
        t = start
        distress_at = None
        end = n_days
        # hazard: ~9%/yr of entering distress, ~3%/yr of being acquired.
        # Over 16 years that removes most of the starting names, which is in
        # line with real US delisting rates (roughly 75% gone after 10 years).
        while t < n_days:
            r[t] = beta * market_r[t] + idio * rng.standard_normal()
            if distress_at is None and rng.random() < 0.09 / 252:
                distress_at = t
                dur = int(rng.integers(120, 260))
                end = min(n_days, t + dur)
            elif distress_at is None and rng.random() < 0.03 / 252:
                r[t] += rng.uniform(0.15, 0.35)  # acquisition premium
                end = t + 1
                meta[sym] = {"fate": "acquired"}
                break
            if distress_at is not None and t >= distress_at:
                r[t] += -0.004 + 0.03 * rng.standard_normal()
            t += 1
            if t >= end:
                break
        if sym not in meta:
            meta[sym] = {"fate": "delisted" if distress_at is not None else "active"}
        px = 50 * np.exp(np.nancumsum(np.nan_to_num(r)))
        px[:start] = np.nan
        px[end:] = np.nan
        vol = np.exp(np.log(2e6) + 0.5 * rng.standard_normal(n_days))
        df = pd.DataFrame({"close": px, "volume": vol}, index=idx)
        write_ticker(sym, df)
        meta[sym]["first"] = str(idx[start].date())
        meta[sym]["last"] = str(idx[min(end, n_days) - 1].date())

    # A sector index everyone can be measured against: equal-weight of the
    # survivors AND the dead at each date (that's what a real index does).
    closes = pd.concat([pd.read_csv(os.path.join(OUT, f"{s}.csv"), index_col=0, parse_dates=True)["close"]
                        for s in meta], axis=1)
    rets = closes.pct_change()
    sector = 100 * np.exp(np.log1p(rets.mean(axis=1).fillna(0)).cumsum())
    ewm = pd.Series(sig).ewm(span=10).mean().values
    vix = ewm * np.sqrt(252) * 100 * np.exp(0.08 * rng.standard_normal(n_days))
    tnx = np.clip(3 + np.cumsum(0.02 * rng.standard_normal(n_days)), 0.5, 8)
    pd.DataFrame({"sector_close": sector.values, "vix": vix, "tnx": tnx}, index=idx).to_csv(
        os.path.join(OUT, "_market.csv"), index_label="date")
    write_meta("synthetic", meta)
    n_dead = sum(m["fate"] != "active" for m in meta.values())
    print(f"synthetic: {n_stocks} stocks, {n_dead} delisted or acquired -> {OUT}")


# --------------------------------------------------------------------------- #
def build_kaggle(path: str, tickers: list[str]):
    files = glob.glob(os.path.join(path, "*.txt")) + glob.glob(os.path.join(path, "*.csv"))
    if not files:
        sys.exit(f"No .txt/.csv files found in {path}")
    want = set(tickers)
    meta, last_dates = {}, []
    for fp in files:
        sym = os.path.basename(fp).split(".")[0].upper()
        if want and sym not in want:
            continue
        try:
            df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date")
        except Exception:
            continue
        if len(df) < 300 or "Close" not in df:
            continue
        df = df.rename(columns={"Close": "close", "Volume": "volume"})
        write_ticker(sym, df)
        meta[sym] = {"first": str(df.index[0].date()), "last": str(df.index[-1].date())}
        last_dates.append(df.index[-1])
    if not meta:
        sys.exit("No usable tickers.")
    dataset_end = max(last_dates)
    for sym, m in meta.items():
        gap = (dataset_end - pd.Timestamp(m["last"])).days
        m["fate"] = "delisted" if gap > 30 else "active"
    write_meta("kaggle", meta)
    n_dead = sum(m["fate"] == "delisted" for m in meta.values())
    print(f"kaggle: {len(meta)} tickers, {n_dead} stopped trading before {dataset_end.date()} -> {OUT}")
    print("Market series (sector ETF, VIX, TNX) will come from Yahoo at run time.")


# --------------------------------------------------------------------------- #
def build_fmp(tickers: list[str], n_delisted: int, start: str, exchanges: tuple[str, ...]):
    import requests
    key = os.environ.get("FMP_API_KEY")
    if not key:
        sys.exit("Set FMP_API_KEY (free at financialmodelingprep.com).")
    base = "https://financialmodelingprep.com/stable"
    calls = 0

    class Paywalled(Exception):
        pass

    def get(url, **params):
        nonlocal calls
        params["apikey"] = key
        r = requests.get(url, params=params, timeout=30)
        calls += 1
        if r.status_code == 429 or "Limit Reach" in r.text[:200]:
            raise RuntimeError("Daily limit hit. Rerun tomorrow; cached tickers are skipped.")
        if r.status_code == 402:
            raise Paywalled(url.split("/")[-1])
        if r.status_code == 401:
            raise SystemExit("401 from FMP: the API key is wrong. Check FMP_API_KEY.")
        r.raise_for_status()
        return r.json()

    meta_path = os.path.join(OUT, "_meta.json")
    meta = json.load(open(meta_path))["tickers"] if os.path.exists(meta_path) else {}

    # 1. delisted names, newest first, a few pages
    delisted = []
    page = 0
    while len(delisted) < n_delisted and page < 20:
        try:
            rows = get(f"{base}/delisted-companies", page=page, limit=100)
        except Paywalled:
            print(f"delisted-companies page {page} is paywalled on the free plan; using pages 0..{page - 1}")
            break
        if not rows:
            break
        for r in rows:
            if r.get("exchange", "") in exchanges and str(r.get("delistedDate", "")) >= start:
                delisted.append((r["symbol"], r.get("delistedDate")))
        page += 1
    delisted = delisted[:n_delisted]
    print(f"{len(delisted)} delisted names from {page} pages ({calls} calls)")

    # 2. prices, one call each, cached
    todo = [(s, None) for s in tickers] + delisted
    for sym, dl in todo:
        if os.path.exists(os.path.join(OUT, f"{sym}.csv")):
            continue
        try:
            rows = get(f"{base}/historical-price-eod/full", symbol=sym, **{"from": start})
        except RuntimeError as e:
            print(e); break
        except Paywalled:
            print(f"{sym}: history paywalled on the free plan, skipped")
            meta[sym] = {"fate": "paywalled"}
            continue
        except Exception as e:
            print(f"{sym}: {e}"); continue
        if not rows:
            print(f"{sym}: no history"); continue
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df["close"] = df["adjClose"] if "adjClose" in df else df["close"]
        write_ticker(sym, df)
        meta[sym] = {"first": str(df.index[0].date()), "last": str(df.index[-1].date()),
                     "fate": "delisted" if dl else "active", "delisted_date": dl}
        print(f"{sym}: {len(df)} rows {df.index[0].date()}..{df.index[-1].date()}  (calls used: {calls})")
    meta = {k: v for k, v in meta.items() if v.get("fate") != "paywalled"}
    write_meta("fmp", meta)
    n_dead = sum(m.get("fate") == "delisted" for m in meta.values())
    print(f"done: {len(meta)} tickers on disk ({n_dead} delisted), {calls} calls this run")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "kaggle", "fmp"], required=True)
    ap.add_argument("--path", help="kaggle: folder of per-ticker files")
    ap.add_argument("--tickers", help="text file, one symbol per line")
    ap.add_argument("--delisted", type=int, default=150, help="fmp: how many delisted names to add")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--exchanges", default="NASDAQ,NYSE")
    ap.add_argument("--clean", action="store_true", help="wipe data/universe first")
    a = ap.parse_args()

    if a.clean and os.path.isdir(OUT):
        for f in glob.glob(os.path.join(OUT, "*")):
            os.remove(f)
    os.makedirs(OUT, exist_ok=True)

    if a.source == "synthetic":
        build_synthetic()
    elif a.source == "kaggle":
        if not a.path:
            sys.exit("--path required for kaggle")
        build_kaggle(a.path, read_ticker_file(a.tickers))
    else:
        build_fmp(read_ticker_file(a.tickers), a.delisted, a.start, tuple(a.exchanges.split(",")))
