"""Build the terminal UI: output/terminal.html

    python build_terminal.py --demo                      # 40 synthetic tickers, no internet
    python build_terminal.py                             # tickers from terminal_tickers.txt via Yahoo
    python build_terminal.py --tickers my_list.txt --sector SPY --serve

One page, every ticker in the sidebar. Click one: price, today's state, the
probability spread, the Markov matrix, EV / Kelly, the backtest verdicts, and
(if run_distress.py has been run) the EDGAR distress read.

What's computed:
  * One pooled model across every ticker in the list, relative to --sector.
    That's the distribution shown on the Spread tab, because a single stock's
    own 8-way split has ~20 observations per state and is mostly noise.
  * Per ticker: its own state today, its own transition matrix, its own quick
    walk-forward verdict (refit every 63 days) so you can see whether *its*
    history disagrees with the pool.
  * Pooled walk-forward verdict once.

The ticker list is a list of survivors. The pooled distribution therefore
carries survivorship bias; the Distress tab is the part built on data that
includes the dead.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from src import backtest, data, features
from src.model import BucketSpec, StateConditionalModel

ROOT = os.path.dirname(os.path.abspath(__file__))


def read_tickers(path: str) -> list[tuple[str, str]]:
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = [x.strip() for x in ln.split(",", 1)]
            out.append((parts[0].upper(), parts[1] if len(parts) > 1 else parts[0].upper()))
    return out


# --------------------------------------------------------------------------- #
def load_yahoo(tickers: list[str], sector: str, start: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    import yfinance as yf
    from src.data import _extract
    syms = sorted(set(tickers + [sector, "^VIX", "^TNX"]))
    raw = yf.download(syms, start=start, auto_adjust=True, progress=False, group_by="column", threads=True)
    close = raw["Close"] if "Close" in raw else raw.xs("Close", axis=1, level=0)
    vol = raw["Volume"] if "Volume" in raw else raw.xs("Volume", axis=1, level=0)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    vol.index = close.index
    market = pd.DataFrame({"sector_close": close[sector], "vix": close["^VIX"], "tnx": close["^TNX"]}).ffill()
    out = {}
    for t in tickers:
        if t not in close or close[t].dropna().shape[0] < 400:
            print(f"  skip {t}: not enough data")
            continue
        out[t] = pd.DataFrame({"close": close[t], "volume": vol[t]}).dropna(subset=["close"])
    return out, market


def load_demo(n: int = 40, seed: int = 3) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[tuple[str, str]]]:
    rng = np.random.default_rng(seed)
    base = data.synthetic(n_days=3000, seed=seed)
    market = base[["sector_close", "vix", "tnx"]]
    mkt_r = np.log(base["sector_close"]).diff().fillna(0).values
    out, names = {}, []
    for i in range(n):
        beta = rng.uniform(0.6, 1.6)
        r = beta * mkt_r + rng.uniform(0.008, 0.025) * rng.standard_normal(len(base)) + rng.uniform(-0.0003, 0.0006)
        px = rng.uniform(20, 400) * np.exp(np.cumsum(r))
        sym = f"DM{i:02d}"
        out[sym] = pd.DataFrame({"close": px, "volume": 1e6 * np.exp(0.4 * rng.standard_normal(len(base)))}, index=base.index)
        names.append((sym, f"Demo Company {i:02d}"))
    return out, market, names


# --------------------------------------------------------------------------- #
def per_ticker(sym: str, px: pd.DataFrame, market: pd.DataFrame, cfg: dict, pooled: StateConditionalModel,
               states: list[str], b: BucketSpec) -> dict | None:
    df = px.join(market, how="left").ffill().dropna(subset=["close", "sector_close", "vix"])
    if len(df) < cfg["regime_window"] + cfg["horizon"] + 300:
        return None
    feat = features.build(df, cfg)
    known = feat["fwd_ret"].notna() & feat["state"].notna()
    if known.sum() < 300 or not isinstance(feat["state"].iloc[-1], str):
        return None

    own = StateConditionalModel(b, states, cfg["horizon"], cfg["prior_strength"]).fit(
        feat.loc[known, "state"], feat.loc[known, "fwd_ret"])
    today_state = feat["state"].iloc[-1]
    i = states.index(today_state)
    price = float(df["close"].iloc[-1])
    pooled_pred = pooled.predict(today_state)
    own_pred = own.predict(today_state)

    # quick own backtest
    qcfg = dict(cfg); qcfg["refit_every"] = 63
    try:
        res = backtest.walk_forward(feat, qcfg, states)
        summ = backtest.summarize(res, qcfg, False)
        own_bt = {"verdict": summ["verdict"], **{k: summ["dev"][k] for k in
                  ("logloss_gain", "logloss_gain_ci90", "direction_hit_model", "direction_hit_baseline", "rows")}}
    except (Exception, SystemExit) as e:
        own_bt = {"verdict": f"Too little history for a walk-forward on this ticker alone ({e})."}

    closes = df["close"]
    def chg(days):
        return float(closes.iloc[-1] / closes.iloc[-1 - days] - 1) if len(closes) > days else None
    hist = closes.iloc[-756:]                     # ~3 years daily
    ann_vol = float(np.log(closes).diff().iloc[-252:].std() * np.sqrt(252))
    return {
        "ticker": sym,
        "price": round(price, 2),
        "as_of": str(df.index[-1].date()),
        "chg_1d": chg(1), "chg_1m": chg(21), "chg_1y": chg(252),
        "ann_vol": round(ann_vol, 4),
        "state": today_state,
        "factor_values": {f: bool(feat[f].iloc[-1]) for f in cfg["factors"]},
        "history": {"dates": [str(d.date()) for d in hist.index], "close": [round(float(x), 2) for x in hist.values]},
        "pooled": {"probs": pooled_pred["probs"], "p_up": pooled_pred["p_up"], "p_up_baseline": pooled_pred["p_up_baseline"],
                   "moments": pooled_pred["moments"], "moments_baseline": pooled_pred["moments_baseline"],
                   "independent_obs": pooled_pred["independent_obs"]},
        "own": {"probs": own_pred["probs"], "baseline": own_pred["baseline"], "p_up": own_pred["p_up"],
                "p_up_baseline": own_pred["p_up_baseline"], "moments": own_pred["moments"],
                "moments_baseline": own_pred["moments_baseline"], "independent_obs": own_pred["independent_obs"],
                "weight_on_state": own_pred["weight_on_state"],
                "transition": np.round(own.transition, 3).tolist(),
                "transition_counts": own.transition_counts.astype(int).tolist(),
                "state_rows": own.n_rows.astype(int).tolist(),
                "state_p_up": [float(own.p[k][b.up_mask()].sum()) for k in range(len(states))],
                "backtest": own_bt},
        "buckets_dollars": b.labels_dollars(price),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", default=os.path.join(ROOT, "terminal_tickers.txt"))
    ap.add_argument("--sector", default="SPY")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.json"))
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    cfg["relative"] = True
    cfg["sector"] = a.sector
    t0 = time.time()

    if a.demo:
        prices, market, names = load_demo()
        source = "synthetic"
    else:
        names = read_tickers(a.tickers)
        print(f"downloading {len(names)} tickers + {a.sector}, ^VIX, ^TNX from Yahoo…")
        prices, market = load_yahoo([t for t, _ in names], a.sector, a.start)
        source = "Yahoo Finance"
    name_of = dict(names)
    print(f"{len(prices)} tickers with prices ({time.time() - t0:.0f}s)")

    # ---- pooled model + pooled backtest
    states = features.state_space(cfg)
    b = BucketSpec(cfg["bucket_edges_pct"])
    frames = []
    for sym, px in prices.items():
        df = px.join(market, how="left").ffill().dropna(subset=["close", "sector_close", "vix"])
        if len(df) < cfg["regime_window"] + cfg["horizon"] + 300:
            continue
        f = features.build(df, cfg); f["ticker"] = sym
        frames.append(f)
    pooled_feat = pd.concat(frames).sort_index(kind="stable")
    known = pooled_feat["fwd_ret"].notna() & pooled_feat["state"].notna()
    pooled = StateConditionalModel(b, states, cfg["horizon"], cfg["prior_strength"]).fit(
        pooled_feat.loc[known, "state"], pooled_feat.loc[known, "fwd_ret"], pooled_feat.loc[known, "ticker"])
    print(f"pooled model: {int(known.sum())} rows across {len(frames)} tickers")
    res = backtest.walk_forward(pooled_feat, cfg, states)
    pooled_bt = backtest.summarize(res, cfg, False)
    print(f"pooled backtest: gain {pooled_bt['dev']['logloss_gain']:+.4f} CI {pooled_bt['dev']['logloss_gain_ci90']}  "
          f"-> {pooled_bt['verdict']}  ({time.time() - t0:.0f}s)")

    # ---- per ticker
    tick = {}
    for k, (sym, px) in enumerate(prices.items()):
        r = per_ticker(sym, px, market, cfg, pooled, states, b)
        if r:
            r["name"] = name_of.get(sym, sym)
            tick[sym] = r
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(prices)} tickers ({time.time() - t0:.0f}s)")
    print(f"{len(tick)} tickers scored ({time.time() - t0:.0f}s)")

    # ---- distress join
    distress = None
    dpath = os.path.join(ROOT, "output", "distress.json")
    if os.path.exists(dpath):
        d = json.load(open(dpath))
        by_ticker = {c["ticker"]: c for c in d["companies"] if isinstance(c.get("ticker"), str) and c["ticker"]}
        if not by_ticker and (a.demo or d.get("source") == "synthetic"):
            # synthetic panel has no tickers: map demo tickers onto synthetic companies for illustration
            rng = np.random.default_rng(1)
            pick = rng.choice(len(d["companies"]), size=len(tick), replace=False)
            by_ticker = {sym: d["companies"][int(i)] for sym, i in zip(tick, pick)}
            d["mapping_note"] = "demo: tickers mapped to random synthetic companies"
        for sym in tick:
            tick[sym]["distress"] = by_ticker.get(sym)
        distress = {k: d[k] for k in ("source", "factors", "backtest", "states", "outcomes", "baseline",
                                      "state_table", "transition", "latest_year", "logit_weights")}
        distress["mapping_note"] = d.get("mapping_note")
        print(f"distress: {sum(1 for s in tick if tick[s]['distress'])}/{len(tick)} tickers matched to EDGAR panel")

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M"),
        "source": source, "sector": a.sector, "horizon": cfg["horizon"], "factors": cfg["factors"],
        "states": states, "buckets_pct": b.labels_pct(),
        "pooled_backtest": pooled_bt, "pooled_unconditional": pooled.q.tolist(),
        "pooled_state_p_up": [float(pooled.p[k][b.up_mask()].sum()) for k in range(len(states))],
        "pooled_state_rows": pooled.n_rows.astype(int).tolist(),
        "pooled_transition": np.round(pooled.transition, 3).tolist(),
        "n_tickers": len(tick),
        "tickers": tick,
        "distress": distress,
        "config": cfg,
    }
    tpl = open(os.path.join(ROOT, "web", "terminal.html")).read()
    js = json.dumps(payload, default=lambda o: None if isinstance(o, float) and np.isnan(o) else float(o)).replace("</", "<\\/")
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
    out = os.path.join(ROOT, "output", "terminal.html")
    with open(out, "w") as f:
        f.write(tpl.replace("/*__DATA__*/null", js))
    print(f"Wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")

    if a.serve:
        import http.server, socketserver, threading, webbrowser
        os.chdir(os.path.join(ROOT, "output"))
        handler = http.server.SimpleHTTPRequestHandler
        handler.log_message = lambda *x, **k: None
        socketserver.TCPServer.allow_reuse_address = True
        for port in range(8765, 8790):
            try:
                httpd = socketserver.TCPServer(("127.0.0.1", port), handler); break
            except OSError:
                continue
        url = f"http://127.0.0.1:{port}/terminal.html"
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        print(f"Serving on {url}  (Ctrl-C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()


if __name__ == "__main__":
    main()
