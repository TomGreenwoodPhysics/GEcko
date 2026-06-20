# GEcko — Statistical Arbitrage on the OSRS Grand Exchange

A pairs-trading system built and rigorously backtested on Old School RuneScape's
in-game economy, not a real financial market. That's deliberate: the Grand
Exchange (GE) has lower fees than most real-world venues, openly-available
multi-year price history, and theoretically motivated pairs — raw materials
and the items they're converted into — making it an interesting,
under-explored testbed for the same techniques used in real statistical
arbitrage. The point of this project isn't pretending OSRS is a real market;
it's applying real market-neutral methodology somewhere most people haven't,
and being honest about every place that methodology breaks down.

**Full methodology, all rejected pairs, and every number behind these
headlines: [docs/METHODOLOGY.md](docs/METHODOLOGY.md).**

## Headline result

After screening 8 candidate pairs, the **Cowhide → Leather** relationship
survived every stress test applied to it:

| Check | Result |
|---|---|
| Cointegration (Engle-Granger + Johansen, 11y daily data) | Both reject the null; spread half-life ≈ 13 days |
| Stable across time? (rolling 2-year windows) | Cointegrated in 99% of 111 windows, hedge ratio 0.80–1.07 |
| Survives a *measured* bid-ask spread, not just the GE's 2% sell tax | Sharpe drops 1.16 → 0.80, still clearly positive |
| Corrected for having screened 8 pairs (Deflated Sharpe Ratio) | **89.8%** probability the edge is real, not multiple-testing luck |
| Independently re-derived from raw data, not just unit-tested | 0 lookahead violations across 3,328 days; 8/8 sampled refits matched exactly |
| Survives a GE buy-limit capacity check | **No** — capped at ~728,000 gp before execution becomes unrealistic (see below) |
| Beats an ML enhancement layer benchmarked identically? | **No** — both logistic regression and gradient boosted trees underperformed the statistical baseline |

That last two rows aren't failures of the project — they're the project working
as designed. A backtest that never finds a limit isn't rigorous, it's
incomplete.

## Why OSRS, specifically

Three deliberate choices, made before any data was touched:

- **Cost structure is simple and known.** The GE charges a flat 2% tax on
  *sells only* (raised from 1% in May 2025) — no commissions, no financing
  costs, no shorting (which is also why the strategy here is a long-only
  *rotation* between two legs, not a textbook long/short spread).
- **The pairs are mechanically motivated, not just statistically convenient.**
  Ore converts to bar via a fixed-ratio skill; cowhide converts to leather via
  a roughly fixed tanning fee. The cointegration isn't an accident of curve-
  fitting — there's a real production relationship underneath it, which is
  exactly what classical pairs-trading theory wants.
- **Novelty.** Crypto and real-equity pairs-trading projects are the most
  oversaturated category in self-taught quant portfolios. A video game
  economy is not — and the data quality and market structure turn out to be
  good enough to support rigorous work, not just a gimmick.

## Pipeline

```
Wiki real-time API + Weird Gloop history API
        |  (data_pull.py)
        v
raw price dumps, per leg, per source  --------------------------------> data/raw/
        |  (build_clean_panels.py -- causal, capped gap-filling, never
        |   interpolates using future prices)
        v
clean aligned panels (8 candidate pairs)  -----------------------------> data/clean/
        |  (screen_pairs.py -- ADF / quick cointegration screen, ranked)
        v
shortlist: hide, iron, rune
        |  (run_cointegration.py, run_rolling_diagnostics.py -- full
        |   Engle-Granger + Johansen, rolling-window stability check)
        v
iron dropped (52% of windows cointegrated -- not a stable relationship)
        |  (run_ou_signal.py -- walk-forward OU fit + z-score,
        |   refit every 30 days on a trailing 2-year window, never
        |   using data from after the date being scored)
        v
hide + rune both produce a tradeable signal
        |  (run_naive_backtest.py -- long-only hysteresis rotation;
        |   initial rotation backtest; realistic spread costs added next)
        v
rune dropped (lost money net of tax -- beta instability was real, not
just a diagnostic curiosity)
        |  (deflated_sharpe.py, run_lookahead_audit.py,
        |   run_buy_limit_check.py -- multiple-testing correction,
        |   independent re-derivation audit, GE capacity check)
        v
hide: a validated, audited, capacity-constrained statistical edge
        |  (run_ml_signal.py -- logistic regression + gradient boosted
        |   trees, identical backtest engine and costs as the baseline)
        v
Both ML models underperform the statistical baseline. Final answer: the
OU/z-score approach on hide, reported with its real capacity limit.
```

## What actually happened, in order

**The cointegration screen worked as a filter, not a rubber stamp.** Of 8
ore/bar-style pairs tested, two were statistical traps that looked fine on a
single full-sample test but fell apart under a rolling-window check (steel had
a negative return correlation hiding behind a "significant" p-value; mithril
and adamant both had ongoing structural breaks that inflated their apparent
cointegration). Gold was a clean, boring rejection — exactly what a negative
control should look like.

**Iron — the strongest candidate on paper, on-brand for the ore/bar theme —
didn't survive scrutiny.** It looked cointegrated on the full 11-year sample
(p=0.0026), but a rolling 2-year window showed it was cointegrated only 52% of
the time, with the hedge ratio swinging from 0.33 to 0.95. That's a coin flip,
not a stable relationship, and it explained an earlier puzzle (iron's two legs
had mismatched integration orders — a methodological red flag the full-sample
test alone wouldn't have caught).

**Rune looked like the best candidate for a while, then lost money.** Its
hedge ratio was just as unstable as iron's, but a walk-forward refit (instead
of one fixed beta) absorbed a late regime break that would otherwise have
wrecked it. The diagnostics — short half-lives, healthy entry frequency —
looked good. It still lost money once the 2% tax was applied (Sharpe -0.53
over 7+ years). Good diagnostics are not the same as a profitable strategy;
this is the cleanest reminder of that in the whole project.

**Hide held up at every stage**, including ones that killed the other
candidates: stable across 9 years of rolling windows, survives a measured
real bid-ask spread (not just tax), and passes a Deflated Sharpe correction
for having screened 8 pairs (89.8% — not just "better than zero," better than
the best of 8 random strategies by chance).

**The GE buy-limit check found hide's actual capacity ceiling: ~728,000 gp.**
Beyond that, every historical rotation would have needed multiple 4-hour
windows to execute — at 50M gp, the worst case is 69 windows (11+ days) just
to enter one position, by which point the mean-reversion opportunity has
almost certainly already resolved. The statistical edge is real; it's also
small. That's a more interesting and more honest finding than either "free
money" or "doesn't work" would have been, and it's a plausible explanation for
*why* the inefficiency survives at all — a hard structural constraint caps how
much capital can arbitrage it away.

**The ML layer lost, and lost informatively.** Logistic regression
underperformed the OU baseline on every metric. Gradient boosted trees did
far worse — 20x the trading frequency of the baseline and a complete capital
wipeout, a textbook overfitting failure: more model capacity found more
spurious patterns in noisy daily data, and traded on every one of them at real
cost. Both models were scored with the identical backtest engine, costs, and
date range as the baseline — this is a legitimate finding, not a rigged
comparison.

## What this project does and doesn't claim

**Does:** a statistically validated, walk-forward-fit, independently audited
edge on one OSRS item pair, with its real capacity limit measured, not assumed.

**Doesn't:** claim this is investable at any serious scale (see the buy-limit
ceiling), claim the realistic-spread estimate holds across the full 9-year
backtest (it's measured from ~3 months of real bid/ask data and applied
retroactively, stated openly as an approximation), or claim the ML layer adds
value (it doesn't, on this dataset).

## Repository structure

```
gecko/
  data/clean.py              # causal gap-filling, leg alignment
  stats/cointegration.py     # ADF, Engle-Granger, Johansen, rolling screen
  stats/ou.py                # OU fitting, walk-forward z-score signal
  backtest/naive.py          # long-only hysteresis rotation backtest
  backtest/deflated_sharpe.py# multiple-testing-corrected Sharpe
  backtest/audit.py          # independent lookahead-bias audit
  backtest/feasibility.py    # GE buy-limit capacity check
  backtest/risk_metrics.py   # Sortino, Calmar
  ml/features.py             # causal feature engineering
  ml/walkforward_classifier.py  # generic walk-forward sklearn harness

data_pull.py, screen_pairs.py, build_clean_panels.py,
run_cointegration.py, run_rolling_diagnostics.py, run_ou_signal.py,
run_naive_backtest.py, run_lookahead_audit.py, run_buy_limit_check.py,
run_ml_signal.py             # top-level pipeline scripts, run in this order

tests/                       # pytest suite, one file per module
data/raw/, data/clean/       # pipeline outputs
figures/                     # generated plots
docs/METHODOLOGY.md          # full tables, all rejected pairs, audit detail
```

## Running it

```bash
pip install -r requirements.txt
python data_pull.py
python screen_pairs.py
python build_clean_panels.py
python run_cointegration.py
python run_rolling_diagnostics.py
python run_ou_signal.py
python run_naive_backtest.py
python run_lookahead_audit.py
python run_buy_limit_check.py
python run_ml_signal.py
pytest tests/ -v
```

## Stack

Python, pandas/numpy, statsmodels (ADF, Engle-Granger, Johansen),
scikit-learn (logistic regression, gradient boosted trees), pytest. No
exotic dependencies — the rigor is in the methodology, not the tooling.