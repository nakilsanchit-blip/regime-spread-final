"""Usage:

    python run.py --demo                      # synthetic data, no internet needed
    python run.py --ticker NVDA --sector SMH  # real data via yfinance
    python run.py --ticker AAPL --sector XLK --horizon 63 --unlock-holdout

Then open output/dashboard.html in a browser (or: python run.py --serve).

Everything else lives in config.json. Command-line flags override it.
"""

from __future__ import annotations

import argparse
import json
import os
import webbrowser

import pandas as pd

from src import backtest, data, features, report, universe as uni

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(ROOT, "config.json"))
    ap.add_argument("--ticker")
    ap.add_argument("--sector", help="sector/industry ETF used for sector momentum, e.g. SMH, XLK, XLF")
    ap.add_argument("--horizon", type=int, help="trading days ahead (21 ≈ 1 month, 63 ≈ 1 quarter)")
    ap.add_argument("--start", help="history start date, YYYY-MM-DD")
    ap.add_argument("--factors", help="comma-separated, e.g. stock_momentum,sector_momentum,vix_regime")
    ap.add_argument("--relative", action="store_true",
                    help="bucket the stock's return MINUS the sector ETF's return (strips survivor/market drift)")
    ap.add_argument("--demo", action="store_true", help="use synthetic data")
    ap.add_argument("--universe", action="store_true",
                    help="pool every ticker in data/universe (built by build_universe.py); --ticker is the one to display")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--unlock-holdout", action="store_true",
                    help="score the locked final period. Do this once, at the end.")
    ap.add_argument("--serve", action="store_true", help="serve output/ on localhost and open the dashboard")
    args = ap.parse_args()

    cfg = load_config(args.config)
    for k in ("ticker", "sector", "horizon", "start"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.factors:
        cfg["factors"] = [f.strip() for f in args.factors.split(",")]
    if args.relative:
        cfg["relative"] = True

    # ---- data ----
    universe_stats = None
    focal = None
    if args.universe:
        tickers, meta, market = uni.load()
        if market is None:
            market = uni.fetch_market(cfg["sector"], cfg["start"])
            source = f"{meta.get('source')} universe + Yahoo market series"
        else:
            cfg["sector"] = "UNIVERSE-EW" if meta.get("source") == "synthetic" else cfg["sector"]
            source = f"{meta.get('source')} universe"
        universe_stats = uni.universe_stats(tickers, meta)
        print(f"universe: {universe_stats['n_tickers']} tickers, fates {universe_stats['fates']}, "
              f"{universe_stats['first_date']} to {universe_stats['last_date']}")
        if not cfg.get("relative"):
            print("note: pooled universes work best with --relative; plain mode treats co-moving stocks as one loud observation")
        feat = uni.build_pooled(tickers, market, cfg)
        if cfg["ticker"] in tickers:
            focal = feat[feat["ticker"] == cfg["ticker"]]
        else:
            # focal stock not in the universe: build its own features from Yahoo, score it with the pooled model
            if meta.get("source") == "synthetic":
                alive = [t for t in sorted(tickers) if meta["tickers"].get(t, {}).get("fate") == "active"]
                cfg["ticker"] = alive[0]
                focal = feat[feat["ticker"] == cfg["ticker"]]
            else:
                own = data.fetch_yahoo(cfg["ticker"], cfg["sector"], cfg["start"], use_cache=not args.no_cache)
                focal = features.build(own, cfg)
                focal["ticker"] = cfg["ticker"]
        states = features.state_space(cfg)
        cluster = backtest.cluster_factor(feat, cfg)
        n_usable = int((feat["state"].notna() & feat["fwd_ret"].notna()).sum())
        print(f"{len(states)} states, {n_usable} pooled rows (~{int(n_usable / (cfg['horizon'] * cluster))} independent obs, cluster factor {cluster:.1f})")
    elif args.demo:
        cfg["ticker"], cfg["sector"] = "DEMO", "DEMOSECTOR"
        df = data.synthetic()
        source = "synthetic regime-switching simulation"
    else:
        df = data.fetch_yahoo(cfg["ticker"], cfg["sector"], cfg["start"], use_cache=not args.no_cache)
        source = "Yahoo Finance (adjusted close)"
    # ---- features / states ----
    if not args.universe:
        print(f"{cfg['ticker']}: {len(df)} trading days, {df.index[0].date()} to {df.index[-1].date()}")
        feat = features.build(df, cfg)
        states = features.state_space(cfg)
        cluster = 1.0
        n_usable = int((feat["state"].notna() & feat["fwd_ret"].notna()).sum())
        print(f"{len(states)} states, {n_usable} usable rows (~{n_usable // cfg['horizon']} independent obs)")

    # ---- walk-forward backtest ----
    res = backtest.walk_forward(feat, cfg, states)
    summary = backtest.summarize(res, cfg, args.unlock_holdout, cluster, focal=cfg["ticker"] if args.universe else None)
    d = summary["dev"]
    print(f"\nOut-of-sample (dev period {d['start']} to {d['end']}, {d['rows']} rows):")
    print(f"  log loss   model {d['logloss_model']:.4f}  vs baseline {d['logloss_baseline']:.4f}"
          f"   gain {d['logloss_gain']:+.4f}  90% CI {d['logloss_gain_ci90']}")
    print(f"  direction  model {d['direction_hit_model']:.1%}  vs baseline {d['direction_hit_baseline']:.1%}")
    print(f"  -> {summary['verdict']}")
    tr = summary["track_record"]["summary"]
    print(f"  month by month: EV was positive in {tr['n_ev_positive']}/{tr['n']} months; "
          f"when positive the stock followed {tr['hit_when_positive']:.0%} of the time"
          + (f", when negative {tr['hit_when_negative']:.0%}" if tr['hit_when_negative'] is not None else "")
          + f"; always-up would score {tr['always_up']:.0%}")
    if args.unlock_holdout:
        hh = summary["holdout"]
        print(f"\nHOLDOUT ({hh['start']} to {hh['end']}): gain {hh['logloss_gain']:+.4f} "
              f"CI {hh['logloss_gain_ci90']}")

    # ---- today's distribution + dashboard ----
    results = report.build_results(feat, cfg, states, summary, source, cluster, focal, universe_stats)
    json_path, html_path = report.write(results)
    if results["today"]:
        t = results["today"]
        print(f"\nToday ({results['as_of']}, ${results['price']}): state = {t['state']}")
        what = f"beats {cfg['sector']}" if cfg.get("relative") else "up"
        print(f"  P({what} over {cfg['horizon']}d) = {t['p_up']:.1%}   baseline {t['p_up_baseline']:.1%}"
              f"   ({t['independent_obs']} independent obs in this state)")
        mo, mb = t["moments"], t["moments_baseline"]
        print(f"  EV {mo['ev']:+.2%} (${mo['ev']*results['price']:+.2f})   std {mo['std']:.1%}"
              f"   EV/std {mo['ev_over_std']:+.2f}   half-Kelly {mo['kelly_half']:.0%} of bankroll"
              f"   [baseline EV {mb['ev']:+.2%}, std {mb['std']:.1%}]")
    print(f"\nWrote {html_path}")

    if args.serve:
        import http.server, socketserver, threading
        os.chdir(os.path.join(ROOT, "output"))
        handler = http.server.SimpleHTTPRequestHandler
        handler.log_message = lambda *a, **k: None   # quiet
        socketserver.TCPServer.allow_reuse_address = True
        for port in range(8765, 8790):
            try:
                httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
                break
            except OSError:
                continue
        else:
            raise SystemExit("No free port between 8765 and 8789; open output/dashboard.html directly.")
        url = f"http://127.0.0.1:{port}/dashboard.html"
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        print(f"Serving on {url}  (Ctrl-C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
        finally:
            httpd.server_close()


if __name__ == "__main__":
    main()
