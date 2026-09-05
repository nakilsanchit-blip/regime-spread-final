# Regime spread

Three pieces, all sharing one harness (purged walk-forward, locked holdout,
shrinkage, bootstrap intervals):

1. **Single-stock dashboard** (`run.py`): probability spread for one ticker.
2. **Pooled universe** (`build_universe.py` + `run.py --universe`): the same
   thing across many tickers.
3. **Terminal** (`build_terminal.py`): one page, ~200 tickers in a sidebar,
   price / spread / Markov / EV / backtest / distress per ticker. Black, white,
   green.
4. **Distress model** (`fetch_edgar.py` + `run_distress.py`): SEC EDGAR
   filings, dead companies included, predicting bankruptcy / deregistration
   within two years. Feeds the terminal's Distress tab.

## Terminal quick start

```
python fetch_edgar.py --synthetic && python run_distress.py     # optional, for the Distress tab
python build_terminal.py --demo --serve                         # 40 fake tickers, no internet
python build_terminal.py --serve                                # ~200 real tickers from terminal_tickers.txt
```

The real build downloads ~200 tickers from Yahoo in one batch, fits one
pooled model (relative to SPY), runs the pooled walk-forward once, then scores
each ticker (own state, own transition matrix, own quick walk-forward). Expect
5 to 10 minutes. Edit `terminal_tickers.txt` to change the list.

## Hosting it (free, no server)

The terminal is one static HTML file, so the only thing that has to run is
the nightly build. `.github/workflows/build.yml` does that on GitHub Actions
and publishes to GitHub Pages:

1. Create a GitHub repo, push this folder to it (branch `main`).
2. Repo Settings -> Pages -> Source: **GitHub Actions**.
3. Settings -> Secrets and variables -> Actions -> New repository secret:
   `EDGAR_USER_AGENT` = `Your Name you@example.com`.
4. Actions tab -> build-terminal -> Run workflow (or just wait for the
   weekday 5:45pm ET schedule).

The site lands at `https://<you>.github.io/<repo>/`. Builds take 10 to 15
minutes; the free tier allows 2,000 minutes a month, so daily is fine. The
EDGAR panel is fetched on Mondays and cached between runs.

If Yahoo starts refusing the runner's IP (it happens), the build fails
loudly and the last good site stays up. Fallback is to run
`build_terminal.py` locally and push `output/terminal.html` into a `site/`
folder by hand.

## Distress model (EDGAR)

```
export EDGAR_USER_AGENT="Your Name you@example.com"    # SEC requires a contact string
python fetch_edgar.py --years 2012 2024                # every 10-K filer per year, ~300 requests
python fetch_edgar.py --years 2012 2024 --going-concern   # + "substantial doubt" flag from full-text search
python run_distress.py
python build_terminal.py --serve                       # picks up output/distress.json
```

Why EDGAR: a company's filings don't vanish when it dies. The cohort "everyone
who filed a 10-K in 2015" is complete, bankruptcies included, for free. The
XBRL frames API returns one concept for every filer in one call, so the whole
market is a few hundred requests.

Design that matters: features for cohort year Y come only from the Y filing;
the outcome is whether the company still filed a 10-K two years later and, if
not, how it left (8-K item 1.03 = bankrupt, Form 15 = deregistered, merger
proxy = acquired). Cohorts are only trained on once their window has closed.
The model never sees who died.

The live fetcher was written against the documented SEC endpoints but has not
been run end to end from this side; if an endpoint's shape has changed, the
error will say which. `--synthetic` exercises the whole pipeline offline.


Given today's market state, what's the probability distribution of where a stock
lands in N trading days? Not "will it go up," but "here's the spread, here's the
same spread ignoring the state, and here's whether the state has ever actually
helped out of sample."

It's the idea from the Markov-matrix conversation, built so it's hard to fool
yourself with it.

## Setup

```
cd regime-spread
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac / Linux
pip install -r requirements.txt
```

Python 3.10+. Open the folder in VS Code; there are two run configurations
under Run and Debug (F5).

## Run it

```
python run.py --demo                                  # synthetic data, no internet
python run.py --ticker NVDA --sector SMH              # real data
python run.py --ticker AAPL --sector XLK --horizon 63 # one quarter ahead
python run.py --ticker NVDA --sector SMH --serve      # also opens the dashboard
python run.py --ticker NVDA --sector SMH --relative   # target = stock return minus sector return
```

Output goes to `output/dashboard.html` (double-click it, it's self-contained)
and `output/results.json` (same numbers, for your own use).

The `--sector` argument is an ETF used for sector momentum. Pick whatever
actually tracks the stock's group: SMH for semis, XLK for tech, XLF for
financials, XLE for energy, XBI for biotech, and so on.

## What the dashboard shows

**The spread.** Solid bars are P(bucket | today's state). Hollow outlines are
P(bucket) over all days. Buckets are defined in percent and shown in dollars at
the current price. The whole point of the design is that the two are always
drawn together: if they're nearly the same height, the state isn't telling you
much, and that's a legitimate answer.

**Does knowing the state actually help?** A walk-forward test. Every
`refit_every` days, the model is refit on history whose outcomes were already
known, then scored on the following days against the lazy baseline (the
all-days distribution). Scoring is log loss and Brier on the full bucket
distribution, plus direction hit rate. The gain over baseline gets a 90%
block-bootstrap interval. The verdict sentence is generated from that interval.

**Expected value and how much to bet.** EV is each bucket's probability times
the average move history produced inside that bucket, summed (not bucket
midpoints). Standard deviation is the width of the same distribution. Kelly
fraction is EV / variance; half Kelly is the practical version. All three are
shown for today's state and for all days, and in `--relative` mode they
describe a long-stock / short-sector pair, not the stock alone. The Kelly
numbers assume the distribution is right, which is exactly what the backtest
section tests, so read them together.

**Month by month, in hindsight.** Every H trading days, using only data
available on that date: the EV the tool would have shown, and what the stock
then did. Hit rate when EV was positive, when negative, and the "always up"
bar. If EV was positive in nearly every month, the tool never made a call; it
was repeating the stock's average, and that's a survivor's history talking.

**The universe** (with `--universe`). How many names were trading each year,
what happened to them, and the survivors-only vs everyone comparison.

**The states.** How much history each state has, in independent observations
(daily rows divided by horizon, because 21 overlapping 21-day windows are
roughly one observation).

**Transitions.** The literal Markov matrix from the conversation: given the
state on day t, how often each state appeared H days later.

## How it works

```
data.py       prices, sector ETF, VIX, 10y yield  ->  daily DataFrame
features.py   trailing-window factors  ->  binary flags  ->  state string
              plus fwd_ret, the H-day-ahead return (the only forward-looking column)
model.py      P(bucket | state), shrunk toward P(bucket); transition matrix
backtest.py   purged expanding-window walk-forward, locked holdout, block bootstrap
report.py     results.json  ->  dashboard.html
```

Factors are deliberately binary. Three binary factors is 8 states; four is 16.
With a 21-day horizon and 15 years of data you have about 180 independent
observations total, so 16 states is already ~11 per state. More levels per
factor is the fastest route to a model that looks great in-sample and means
nothing.

Available factors (`config.json` -> `factors`):

| name | true when |
|---|---|
| `stock_momentum` | stock is up over `momentum_lookback` days |
| `sector_momentum` | sector ETF is up over `momentum_lookback` days |
| `vol_regime` | realized vol above its trailing `regime_window` median |
| `vix_regime` | VIX above its trailing `regime_window` median |
| `rate_trend` | 10-year yield up over `momentum_lookback` days |
| `volume_regime` | average volume above its trailing median |

## What's protecting you from overfitting

1. **Shrinkage.** Each state's histogram is blended with the unconditional
   histogram, weighted by independent observations: `w = n_eff / (n_eff + prior_strength)`.
   A state seen 5 times gets ~20% say; seen 100 times, ~83%. Same for the
   transition matrix, so thin rows don't show fake 0% / 100% cells.

2. **Purged walk-forward.** On refit date T, training uses only rows with
   `t + horizon <= T`. Without this, the last H training rows share their
   future with the test rows and the score is inflated. This is the most
   common leak in this kind of model and it's silent.

3. **Locked holdout.** The last `holdout_fraction` of dates isn't scored unless
   you pass `--unlock-holdout`. Tinker against the development score; look at
   the holdout once at the end. If you tune until the holdout looks good, you
   no longer have a holdout.

4. **Proper scoring rules.** Log loss and Brier, not accuracy. A model that
   always predicts the most common bucket gets good accuracy for free.

5. **Block bootstrap.** Overlapping windows are autocorrelated. The interval on
   the gain resamples in blocks of `horizon` days.

6. **Tests.** `python -m tests.test_pipeline` plants a strong signal and checks
   the walk-forward finds it, then feeds pure noise and checks it doesn't
   claim an edge. If the second one ever fails, something is leaking.

What none of this protects you from: changing lookbacks, factors, and bucket
edges until the development score looks good. That's overfitting done by hand,
and no code can stop it. Decide the config, run it, believe the result.

## Survivorship bias, specifically

You can't remove it from a single stock's history with code; it's in the data.
What you can do is change the question so the survivor's drift stops being
the answer:

- **`--relative`** buckets the stock's return *minus* the sector ETF's return.
  The market's long-run uptrend and the sector's drift are gone, so the state
  has to predict out- or under-performance. A 62% "up" rate for a 300x
  winner becomes a much more honest "beats the sector 50-ish% of the time".
- **Run on the ETF** (`--ticker SMH --sector SPY`). Indices are reconstituted:
  the losers were in them at the time and dropped out later.
- **Pool a universe that includes the dead.** The real fix. Every name that was
  in the group at each date, delisted ones included, tagged with the same
  states. Needs a survivorship-bias-free source (Norgate, Sharadar, CRSP).
  Yahoo Finance silently drops delisted tickers, so this project can't do it
  on free data. If you get such a source, `features.build` and
  `StateConditionalModel.fit` work unchanged on a stacked multi-ticker frame.

Why it matters: you picked the ticker because it exists and you've heard of it. Every company
that was in the same state and then went to zero is missing from the history.
For a single-name model this is partly acceptable, because the question really
is "what does *this* stock do," but keep in mind:

- The stock's own unconditional distribution is flattered by the fact that it
  survived. "P(up) = 57%" for a stock that 10x'd is not a forecast; it's a
  description of a winner.
- Any claim like "stocks in this state tend to..." would need a universe that
  includes delisted names (CRSP, or a survivorship-bias-free vendor). Yahoo
  Finance does not give you that.
- A useful sanity check: run the same config on the sector ETF and on SPY. If
  your single stock shows an "edge" the ETF doesn't, it's more likely
  stock-specific noise than a market structure.

## Pooled universe (the survivorship fix)

`--universe` pools every ticker in `data/universe/` and uses one focal ticker
for the "today" chart. Build the universe first:

```
python build_universe.py --source synthetic           # 80 fake stocks, most of them die; no internet
python run.py --universe --relative --serve

python build_universe.py --source kaggle --path ~/Downloads/Stocks --tickers universe.txt
python run.py --universe --relative --ticker NVDA --sector SMH --serve

export FMP_API_KEY=...                                 # free key, 250 requests/day
python build_universe.py --source fmp --tickers universe.txt --delisted 150
python run.py --universe --relative --ticker NVDA --sector SMH --serve
```

- **kaggle**: the "Huge Stock Market Dataset" (Boris Marjanovic). Download,
  unzip, point `--path` at the `Stocks` folder. About 7,000 US tickers to
  November 2017; any file that ends well before then is a stock that stopped
  trading. No limits, one download, but nothing after 2017.
- **fmp**: Financial Modeling Prep's free plan. Pulls the delisted-companies
  list, adds up to `--delisted` names that traded on NASDAQ/NYSE, then fetches
  price history one request per ticker. Results are cached, so when the daily
  cap hits, rerun tomorrow and it continues. Free-plan history depth varies by
  ticker; the script prints what came back.
- `universe.txt` is the active-name list. Edit it for whatever group you're
  studying. Pass `--sector` as the matching ETF.

What changes in the maths when pooling:

- Rows from different stocks on the same date share the market move. In
  plain mode the model treats rows-per-date as one observation (the "cluster
  factor" printed at startup). In `--relative` mode the shared move is
  subtracted, so stocks count separately. Use `--relative`.
- The bootstrap resamples blocks of *dates*, taking every stock on those
  dates along, so it can't count 50 stocks in March 2020 as 50 pieces of
  evidence.
- Transitions are computed within each ticker, never across.
- The dashboard's "The universe" panel fits the model twice, on survivors only
  and on everyone, and shows both. The gap is survivorship bias, measured.

What's still missing even with a universe: a delisted stock's final leg (the
last `horizon` days before it stopped trading) has no forward return and is
dropped, so the terminal loss is understated. Vendors that supply a delisting
return fix this; free sources don't. And a hand-made ticker list is not a
point-in-time index constituent list.

## Extending it

- **Add a factor:** write a function in `features.py` that returns a boolean
  Series using only trailing data, register it in `FACTORS`, add it to
  `config.json`.
- **Hidden regimes instead of observable ones:** fit a 2- or 3-state Gaussian
  HMM on returns and realized vol (`hmmlearn`) inside the walk-forward loop,
  and use the filtered regime probability as a factor. Watch for label
  switching between refits.
- **Time-varying transitions:** the cleaner version of "factors as an effect on
  the Markov model" is `statsmodels.tsa.regime_switching.MarkovRegression`
  with `exog_tvtp` set to your factors. Fewer states, smoother estimates,
  more math.

## Not advice

This is a modeling exercise. Nothing here is a recommendation to buy or sell
anything.
