"""Build data/edgar/panel.csv: one row per company per 10-K year, including
companies that later went bankrupt, were acquired, or went dark.

    python fetch_edgar.py --synthetic                          # fake panel, no internet
    export EDGAR_USER_AGENT="Your Name your@email.com"          # SEC requires this
    python fetch_edgar.py --years 2012 2024                    # real panel
    python fetch_edgar.py --years 2012 2024 --going-concern    # + going-concern flag (slower)

Why EDGAR fixes the survivorship problem: a company's filings don't vanish
when it dies. Every 10-K it filed is still there, and so is the 8-K that
announced the bankruptcy. So the cohort "everyone who filed a 10-K in 2015"
is complete, dead ones included, at zero cost.

How the panel is built (all free SEC endpoints, ~10 req/s allowed):
  1. Cohorts: quarterly form indexes list every filing. Every CIK with a 10-K
     in year Y is in cohort Y.
  2. Features: the XBRL "frames" API returns ONE concept for EVERY filer in
     one call (e.g. all companies' Assets at end of CY2015). ~12 concepts x
     N years = a few hundred calls for the whole market.
  3. Outcomes: a cohort-Y company that has no 10-K in year Y+2 has exited.
     For exits, the per-company submissions JSON says how: 8-K item 1.03 ->
     bankrupt; Form 15 -> deregistered; merger proxy (DEFM14A) -> acquired.
  4. Optional: full-text search for "substantial doubt" in 10-Ks -> going
     concern flag.

Caveats you should know before believing the output:
  * Frames coverage starts ~2009 (XBRL mandate). Nothing earlier.
  * Concept names vary between companies; fallbacks are tried, but some rows
    will have gaps. Rows missing assets are dropped.
  * "No 10-K in Y+2" also catches late filers. Some "exits" are just slow.
  * Fiscal years that don't end in December get squeezed into calendar
    frames. Good enough for a distress signal, not for accounting research.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "edgar")
RAW = os.path.join(OUT, "raw")

CONCEPTS = {
    # panel column: (list of us-gaap concept fallbacks, "instant" or "duration")
    "assets": (["Assets"], "instant"),
    "liabilities": (["Liabilities"], "instant"),
    "equity": (["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"], "instant"),
    "cash": (["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "Cash"], "instant"),
    "current_assets": (["AssetsCurrent"], "instant"),
    "current_liabilities": (["LiabilitiesCurrent"], "instant"),
    "revenue": (["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"], "duration"),
    "net_income": (["NetIncomeLoss", "ProfitLoss"], "duration"),
    "ocf": (["NetCashProvidedByUsedInOperatingActivities"], "duration"),
}


# --------------------------------------------------------------------------- #
# Synthetic panel
# --------------------------------------------------------------------------- #

def synthetic(n_companies: int = 3000, y0: int = 2010, y1: int = 2024, seed: int = 5) -> pd.DataFrame:
    """Companies with persistent financial health that drifts; a yearly hazard
    of bankruptcy/deregistration that rises sharply when they burn cash with
    little runway, negative equity, or a going-concern flag; a flat hazard of
    acquisition. Outcomes for the last two cohorts are not yet known."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_companies):
        cik = 100000 + c
        name = f"Synthetic Co {c:04d}"
        start = int(rng.integers(y0, y1 - 2))
        size = 10 ** rng.uniform(6.5, 10.5)          # $3M .. $30B
        health = rng.normal(0, 1)                    # latent
        alive = True
        for y in range(start, y1 + 1):
            if not alive:
                break
            health = 0.8 * health + 0.6 * rng.normal()
            assets = size * np.exp(0.1 * health + 0.05 * rng.normal())
            margin = 0.10 * health + 0.05 * rng.normal()
            ni = margin * assets * 0.6
            ocf = ni + 0.05 * assets * rng.normal()
            lev = np.clip(0.55 - 0.15 * health + 0.1 * rng.normal(), 0.05, 1.5)
            liab = lev * assets
            eq = assets - liab
            cash = np.clip(assets * (0.15 + 0.08 * health + 0.05 * rng.normal()), 0.001 * assets, None)
            ca, cl = assets * 0.4, liab * 0.5
            gc = int(health < -1.6 and rng.random() < 0.6)
            burn = max(-ocf, 0)
            runway = cash / burn if burn > 0 else np.inf
            distress = (ocf < 0) + (runway < 1) + (eq < 0) + 1.5 * gc + 0.5 * (assets < 50e6)
            h_bad = 0.01 + 0.06 * distress ** 1.7 / 4
            h_acq = 0.025
            u = rng.random()
            if u < h_bad:
                outcome = "bankrupt" if rng.random() < 0.55 else "deregistered"
                alive = False
            elif u < h_bad + h_acq:
                outcome = "acquired"; alive = False
            else:
                outcome = "filing"
            # outcome is "within 2 years": if it dies in y+1 instead, still counts
            rows.append(dict(cik=cik, name=name, ticker="", year=y, assets=assets, liabilities=liab,
                             equity=eq, cash=cash, revenue=assets * 0.8, net_income=ni, ocf=ocf,
                             current_assets=ca, current_liabilities=cl, going_concern=gc,
                             outcome=outcome))
    p = pd.DataFrame(rows).sort_values(["cik", "year"]).reset_index(drop=True)
    # push a death in year y+1 back onto year y's label (2-year window)
    nxt = p.groupby("cik")["outcome"].shift(-1)
    same_next = p.groupby("cik")["year"].shift(-1) == p["year"] + 1
    p.loc[(p["outcome"] == "filing") & same_next & nxt.isin(["bankrupt", "deregistered", "acquired"]), "outcome"] = \
        nxt[(p["outcome"] == "filing") & same_next & nxt.isin(["bankrupt", "deregistered", "acquired"])]
    p["outcome_known"] = p["year"] <= y1 - 2
    return p


# --------------------------------------------------------------------------- #
# Live EDGAR
# --------------------------------------------------------------------------- #

class Edgar:
    def __init__(self):
        ua = os.environ.get("EDGAR_USER_AGENT")
        if not ua or "@" not in ua:
            sys.exit('Set EDGAR_USER_AGENT="Your Name you@example.com" (SEC requires a contact).')
        import requests
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
        self.last = 0.0
        os.makedirs(RAW, exist_ok=True)

    def get(self, url: str, cache: str, text: bool = False):
        path = os.path.join(RAW, cache)
        if os.path.exists(path):
            with open(path) as f:
                return f.read() if text else json.load(f)
        wait = 0.12 - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        r = self.s.get(url, timeout=60)
        self.last = time.time()
        if r.status_code == 404:
            return None
        r.raise_for_status()
        body = r.text if text else r.json()
        with open(path, "w") as f:
            f.write(body if text else json.dumps(body))
        return body

    # -- 1. cohorts ---------------------------------------------------------
    def tenk_filers(self, year: int) -> dict[int, tuple[str, str]]:
        """{cik: (name, first 10-K date)} for every 10-K filed in `year`."""
        out = {}
        for q in (1, 2, 3, 4):
            txt = self.get(f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/form.idx",
                           f"form_{year}_Q{q}.idx", text=True)
            if not txt:
                continue
            for line in txt.splitlines():
                if not line.startswith(("10-K ", "10-K405 ", "10-KSB ")):
                    continue
                # fixed width: Form Type | Company Name | CIK | Date Filed | File Name
                parts = line.split()
                # walk from the right: filename, date, cik; the rest is name
                try:
                    fname, date, cik = parts[-1], parts[-2], int(parts[-3])
                except (ValueError, IndexError):
                    continue
                name = " ".join(parts[1:-3])
                if cik not in out:
                    out[cik] = (name, date)
        return out

    # -- 2. features --------------------------------------------------------
    def frame(self, concept: str, year: int, kind: str) -> dict[int, float]:
        period = f"CY{year}Q4I" if kind == "instant" else f"CY{year}"
        js = self.get(f"https://data.sec.gov/api/xbrl/frames/us-gaap/{concept}/USD/{period}.json",
                      f"frame_{concept}_{period}.json")
        if not js:
            return {}
        return {int(d["cik"]): float(d["val"]) for d in js.get("data", []) if d.get("val") is not None}

    # -- 3. outcomes --------------------------------------------------------
    def classify_exit(self, cik: int, after: str) -> str:
        js = self.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", f"sub_{cik}.json")
        if not js:
            return "deregistered"
        rec = js.get("filings", {}).get("recent", {})
        forms, dates, items = rec.get("form", []), rec.get("filingDate", []), rec.get("items", [])
        bankrupt = acquired = dereg = False
        for f, d, it in zip(forms, dates, items or [""] * len(forms)):
            if d < after:
                continue
            if f.startswith("8-K") and "1.03" in (it or ""):
                bankrupt = True
            if f in ("DEFM14A", "DEFM14C", "SC TO-T", "SC 13E3"):
                acquired = True
            if f.startswith("15-") or f == "15":
                dereg = True
        if bankrupt:
            return "bankrupt"
        if acquired:
            return "acquired"
        return "deregistered" if dereg else "deregistered"

    # -- 4. going concern ---------------------------------------------------
    def going_concern_ciks(self, year: int) -> set[int]:
        ciks = set()
        start = 0
        while start < 10000:
            js = self.get(
                f'https://efts.sec.gov/LATEST/search-index?q=%22substantial%20doubt%22%20%22going%20concern%22'
                f'&forms=10-K&dateRange=custom&startdt={year}-01-01&enddt={year}-12-31&from={start}',
                f"fts_gc_{year}_{start}.json")
            hits = (js or {}).get("hits", {}).get("hits", [])
            if not hits:
                break
            for h in hits:
                for c in h.get("_source", {}).get("ciks", []):
                    try:
                        ciks.add(int(c))
                    except ValueError:
                        pass
            start += len(hits)
        return ciks


def build_live(y0: int, y1: int, going_concern: bool) -> pd.DataFrame:
    e = Edgar()
    this_year = pd.Timestamp.today().year
    cohorts = {y: e.tenk_filers(y) for y in range(y0, min(y1, this_year) + 3)}
    print({y: len(c) for y, c in cohorts.items()})

    # current ticker map (survivors only, that's fine: it's only for display)
    tick = e.get("https://www.sec.gov/files/company_tickers.json", "company_tickers.json") or {}
    cik2ticker = {int(v["cik_str"]): v["ticker"] for v in tick.values()} if isinstance(tick, dict) else {}

    rows = []
    for y in range(y0, min(y1, this_year) + 1):
        filers = cohorts.get(y, {})
        if not filers:
            continue
        fy = y - 1                                  # 10-K filed in y reports fiscal y-1
        vals = {}
        for col, (concepts, kind) in CONCEPTS.items():
            merged = {}
            for c in concepts:
                for cik, v in e.frame(c, fy, kind).items():
                    merged.setdefault(cik, v)
            vals[col] = merged
        gc = e.going_concern_ciks(y) if going_concern else set()
        known = (y + 2) in cohorts and (y + 2) < this_year
        alive_next = set(cohorts.get(y + 2, {})) | (set(cohorts.get(y + 3, {})) if (y + 3) in cohorts else set())
        n_exit = 0
        for cik, (name, date) in filers.items():
            a = vals["assets"].get(cik)
            if a is None or a <= 0:
                continue
            if not known:
                outcome = "unknown"
            elif cik in alive_next:
                outcome = "filing"
            else:
                outcome = e.classify_exit(cik, date); n_exit += 1
            rows.append(dict(cik=cik, name=name, ticker=cik2ticker.get(cik, ""), year=y, assets=a,
                             liabilities=vals["liabilities"].get(cik, np.nan),
                             equity=vals["equity"].get(cik, np.nan), cash=vals["cash"].get(cik, np.nan),
                             revenue=vals["revenue"].get(cik, np.nan), net_income=vals["net_income"].get(cik, np.nan),
                             ocf=vals["ocf"].get(cik, np.nan), current_assets=vals["current_assets"].get(cik, np.nan),
                             current_liabilities=vals["current_liabilities"].get(cik, np.nan),
                             going_concern=int(cik in gc) if going_concern else np.nan,
                             outcome=outcome, outcome_known=known))
        print(f"{y}: {sum(r['year'] == y for r in rows)} rows with assets, {n_exit} exits classified")
    p = pd.DataFrame(rows)
    p = p[p["outcome"] != "unknown"].copy() if p["outcome_known"].all() else p
    p.loc[p["outcome"] == "unknown", "outcome"] = "filing"      # placeholder; outcome_known=False masks it
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--years", nargs=2, type=int, default=[2012, 2024], metavar=("FIRST", "LAST"))
    ap.add_argument("--going-concern", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.synthetic:
        p = synthetic(y0=a.years[0], y1=a.years[1])
        src = "synthetic"
    else:
        p = build_live(a.years[0], a.years[1], a.going_concern)
        src = "edgar"
    path = os.path.join(OUT, "panel.csv")
    p.to_csv(path, index=False)
    with open(os.path.join(OUT, "_meta.json"), "w") as f:
        json.dump({"source": src, "built": time.strftime("%Y-%m-%d"), "rows": int(len(p)),
                   "companies": int(p["cik"].nunique()), "years": [int(p["year"].min()), int(p["year"].max())]}, f)
    print(f"{src}: {len(p)} company-years, {p['cik'].nunique()} companies -> {path}")
    print(p[p["outcome_known"]]["outcome"].value_counts().to_dict())
