# GEcko - Methodology

Full detail behind every claim in the main [README](../README.md): the
8-pair screen, the rolling stability checks, the realistic-cost stress test,
the deflated Sharpe correction, the lookahead audit, the buy-limit check, and
the ML benchmark. Numbers here are taken directly from real runs of the
pipeline, not illustrative examples.

## 1. Data

Two sources are combined:

- **OSRS Wiki real-time prices API** -- realistic tradeable `avgHighPrice` /
  `avgLowPrice` and volumes, RuneLite-sampled (a sample of real trades, not
  the whole GE). Only available from ~March 2021, and the `/timeseries`
  endpoint caps at 365 points per call -- at 6-hour resolution, ~3 months per
  pull.
- **Weird Gloop history API** -- the official GE *guide* price, one point per
  day, back to March 2015 (~11 years). Slower-moving than a real tradeable
  price, and **volume is entirely empty** in this series -- a real property
  of the source, not a bug in the cleaning pipeline.

Cointegration statistics run on the long daily series (need years of history
for the tests to mean anything); realistic-cost stress testing runs on the
short, real-bid/ask series.

**Cleaning policy (`gecko/data/clean.py`):** gaps are forward-filled, capped
at 2 missing steps, using only past values -- never interpolated between a
past and future point. Rows where a gap exceeds the cap are dropped, with the
count reported, not silently absorbed. Completeness is judged on price
columns only; volume gaps don't block an otherwise-good price observation.

## 2. The 8-pair cointegration screen

Candidate pairs were chosen from OSRS's smithing/tanning production chains --
raw material to processed good, theoretically motivated rather than purely
statistically convenient. Quick screen: ADF on `log(processed) - log(raw)`
(hedge ratio assumed = 1, a fast first pass; the real hedge ratio is fitted
properly for the survivors in section 3), plus an AR(1)-implied half-life.

| Pair | n (days) | Ratio CV | Return corr | ADF p-value | Half-life (d) | Note |
|---|---|---|---|---|---|---|
| **hide** | 4014 | 9.0% | 0.224 | 0.000 | 13.1 | tanning fee ~= fixed gp cost |
| silver | 4017 | 27.8% | 0.130 | 0.000 | 53.1 | 1:1, no secondary input |
| steel | 4013 | 21.4% | **-0.347** | 0.004 | 51.2 | needs iron+coal; trap, see below |
| iron | 4018 | 29.7% | 0.080 | 0.000 | 67.0 | 1:1, no secondary input |
| rune | 4020 | **2.7%** | 0.124 | 0.161 | 32.3 | tightest ratio of any pair |
| mithril | 4014 | 28.0% | -0.087 | 0.009 | 151.9 | trap, see below |
| adamant | 4038 | 19.8% | -0.135 | 0.521 | 115.3 | ongoing structural break |
| gold | 4029 | 47.7% | 0.025 | 0.335 | 643.5 | clean rejection |

**Two pairs are statistical traps, not just weak candidates.** Steel's
"significant" ADF p-value (0.004) sits next to a strongly *negative* return
correlation (-0.347) -- coal and steel bar trending in opposite directions
over the sample, not short-run mean reversion. Mithril's "significant"
p=0.009 sits on top of a visible multi-year level shift (the ratio plot shows
a flat band through 2019, then a sustained jump to a new, higher band from
2020 onward) -- a structural break that a full-sample ADF test can spuriously
pass through. Adamant shows the same pattern, openly: its ADF test correctly
*fails* (p=0.521) during a break that was still ongoing at data collection
time.

Gold is a clean, boring rejection (p=0.335, 643-day half-life) -- exactly
what a negative control should look like.

## 3. Full cointegration treatment (Engle-Granger + Johansen)

The three most promising pairs (hide, iron, rune) got the full treatment:
both legs' individual integration order, Engle-Granger in both directions
(with correct MacKinnon critical values via `statsmodels.coint`, not naive
ADF on OLS residuals), and Johansen as an independent cross-check.

| Pair | n | Raw order | Proc order | EG p (proc~raw) | EG beta | Johansen rejects r=0? | Johansen beta | Half-life |
|---|---|---|---|---|---|---|---|---|
| hide | 4058 | I(0) | I(0) | 6.8e-9 | 0.925 | Yes (264.1 vs 15.5) | 0.954 | 13.1d |
| iron | 4058 | I(1) | I(0) | 0.0026 | 0.616 | Yes (65.6 vs 15.5) | 0.755 | 62.7d |
| rune | 4057 | I(1) | I(1) | 0.340 (not coint.) | 0.676 | Yes (76.6 vs 15.5) | 0.889 | 35.7d |

Two methodological caveats, stated rather than glossed over:

- **Hide's legs are both I(0)** -- already individually stationary. That's
  *not* the classical I(1)+I(1)->I(0) cointegration setup the test assumes.
  The spread is still meaningfully tighter than either leg alone (which is
  the actual tradeable property), but "cointegration" is the wrong precise
  word for it.
- **Iron's legs have mismatched integration orders** (I(1) vs I(0)). A linear
  combination of an I(1) and an I(0) series is generically I(1) unless the
  I(1) coefficient is ~0 -- so a "cointegrated" verdict here is suspect on
  theoretical grounds alone, before even looking at the rolling-window result
  below.
- **Rune's EG and Johansen tests disagree** (not cointegrated vs. rejects
  r=0). The rolling-window analysis below explains why.

## 4. Rolling-window stability (the real filter)

A single full-sample test can't distinguish "stable cointegration" from
"looked cointegrated on average across very different regimes." A 2-year
rolling window, stepped monthly (111 windows), can:

| Pair | % windows cointegrated | Hedge ratio range |
|---|---|---|
| **hide** | **99%** | 0.80 - 1.07 |
| rune | 79% | -0.00 - 0.96 |
| iron | **52%** | 0.33 - 0.95 |

Iron's 52% hit rate is a coin flip, not a stable relationship -- this is the
rolling-window evidence that explains its mismatched-integration-order flag
above. **Iron was dropped at this stage.**

Rune's p-value time series shows the full-sample "not cointegrated" verdict
was driven by two things: an erratic 2017-2018 burn-in period (illiquid
early GE data) and a late regime break (2024 onward) -- in between,
2018-2024 was robustly cointegrated almost continuously. But the hedge ratio
is nearly as unstable as iron's, which matters for what happens next.

## 5. OU fit + walk-forward z-score signal

For hide and rune, beta and the OU parameters (theta, mu, sigma) are refit
every 30 days on a trailing 2-year window -- never on data that includes or
postdates the day being scored. Verified two ways: a synthetic two-regime
test (beta=1 in the first half, beta=3 in the second; the harness recovers
~1.00 early and ~3.00 late, never the wrong one), and an independent audit on
the *real* saved output (section 7).

| Pair | Scored days | Refits | Beta range | Half-life range | % days \|z\|>2 |
|---|---|---|---|---|---|
| hide | 3328 | 111 | 0.80 - 1.07 | 3.9 - 27.3d | 9.4% |
| rune | 3327 | 111 | -0.00 - 0.96 | 3.3 - 22.5d | 9.6% |

The walk-forward refit visibly absorbs rune's late regime break (the
fixed-beta spread shows a sustained post-2024 level shift; the walk-forward
z-score does not) -- but the underlying beta instability doesn't go away,
it's just no longer catastrophic for the signal.

## 6. Initial backtest and realistic costs

Long-only **hysteresis rotation**: hold whichever leg looks cheap relative to
the fitted spread, switch when z crosses +/-1.5, hold otherwise (avoids
tax-incurring whipsaw near z=0). GE has no shorting, so this replaces the
textbook long/short spread construction.

**Tax-only result** (2% GE sell tax, instant execution at daily guide
price):

| Pair | Total return | Ann. return | Sharpe | Max DD | Rotations |
|---|---|---|---|---|---|
| hide | +4078.8% | +50.6% | 1.16 | -57.8% | 34 |
| rune (from 2019, excl. burn-in) | -- | -3.7% | -0.53 | -30.2% | 25 |

**Rune lost money net of tax and was dropped here** -- the beta instability
flagged in section 4 was real, not just a diagnostic curiosity.

**Realistic cost stress test on hide**: bid-ask half-spread measured from the
real ~3-month realtime data (raw/Cowhide=0.85%, proc/Leather=2.05%), charged
on every transaction in addition to the 2% sell tax:

| Cost model | Total return | Ann. return | Sharpe | Sortino | Calmar | Max DD |
|---|---|---|---|---|---|---|
| Tax only | +4078.8% | +50.6% | 1.16 | 1.73 | 0.87 | -57.8% |
| + measured spread | **+1455.1%** | **+35.1%** | **0.80** | **1.18** | **0.59** | -59.0% |

The +4078.8% number is **not** reported as a result anywhere in this project
-- it's a frictionless-execution artifact. The realistic-cost number (+1455%,
Sharpe 0.80) is what survives.

## 7. Deflated Sharpe Ratio and the lookahead audit

**Deflated Sharpe** (Bailey & Lopez de Prado): corrects the Sharpe ratio for
non-normal returns (skew, kurtosis) *and* for the fact that hide was the
best-looking result out of 8 screened pairs -- asking whether 0.80 clears the
expected maximum Sharpe from 8 random/null strategies, not just zero.

> DSR = **89.8%** (n_trials=8, T=3327, skew=-0.122, kurtosis=4.30,
> sr0_benchmark=0.0254 vs sr_hat=0.0475 daily)

`n_trials=8` is a documented judgment call -- the number of pairs actually
screened, which is the source of the multiple-testing exposure being
corrected for. A different, defensible choice would count parameter
configurations too; this choice is the more conservative direction (fewer
trials = an easier bar to clear), stated rather than hidden.

**Lookahead audit** (`gecko/backtest/audit.py`) -- two checks against the
real saved pipeline output, not just synthetic unit tests:

- *Structural*: every scored row's `fit_end_date` must be strictly before
  its own date. **0 violations across 3328 rows (hide) and 3327 rows (rune).**
- *Recomputation* (the stronger check): independently re-derive beta and mu
  from the raw price history for a random sample of refit windows, and
  confirm they match what was saved. **8/8 sampled refits matched exactly,
  for both pairs.** This catches what the structural check can't -- a bug
  that fits on the wrong window while still recording a fit_end_date that
  looks fine.

## 8. GE buy-limit feasibility

The backtest assumes every rotation executes instantly, in full, at the
daily guide price. The GE's buy limit (units per rolling 4 hours, per item)
makes that assumption false above some capital scale.

- **Safe capital ceiling** (every historical entry fits in one 4-hour
  window): **728,000 gp**. Binding constraint: the raw leg (Cowhide) on
  2018-05-12, capacity 728,000 gp; the proc leg (Leather) bound at 988,000 gp
  on 2018-04-24.
- At larger notional capital, the picture degrades fast:

| Capital | % of entries needing >1 window | Worst case |
|---|---|---|
| 1,000,000 gp | 12% | 2 windows |
| 10,000,000 gp | 100% | 14 windows |
| 50,000,000 gp | 100% | 69 windows (~11.5 days) |
| 100,000,000 gp | 100% | 138 windows (~23 days) |

By the time a 50-100M gp position is fully phased in, the mean-reversion
opportunity it was trying to capture has almost certainly already resolved.
**This is the project's real capacity constraint** -- the statistical edge
is genuine but small, plausibly *because* this exact constraint is what
stops it being arbitraged away by larger capital.

## 9. ML signal layer

Two models (logistic regression, gradient boosted trees), benchmarked against
the OU baseline using the **identical backtest engine, cost model, and date
range** -- the only thing that differs is where the entry/exit signal comes
from. Walk-forward fit with the same discipline
as the OU layer (same `fit_end_date` audit convention; the structural
lookahead check built for the OU layer works on the ML output unmodified).

**Features** (causal, no lookahead): today's z-score, spread momentum
(1-day and 5-day), trailing spread volatility, current half-life,
day-of-week (cyclical encoding), and 1-day/5-day trailing returns on both
legs. **Label**: does raw outperform proc from day t to t+1 (computed
ex-post for training, never fed back as a feature).

| Signal | Ann. return | Sharpe | Sortino | Max DD | Rotations | DSR |
|---|---|---|---|---|---|---|
| OU z-score (baseline) | +39.1% | 0.887 | 1.297 | -59.0% | 25 | **86.7%** |
| Logistic regression | +17.0% | 0.368 | 0.518 | -74.3% | 49 | 52.3% |
| Gradient boosted trees | **-90.8%** | -1.422 | -1.636 | **-100.0%** | **501** | 0.0% |

Both ML models underperform. Logistic regression loses on every metric
while trading 2x more often. Gradient boosted trees fails far more
severely -- **20x the baseline's trading frequency and a complete capital
wipeout** -- a textbook overfitting signature: more model flexibility found
more spurious patterns in noisy daily data and traded on every one of them
at real cost.

`n_trials=8` is used for both ML models' DSR, matching the baseline's
convention. If a result here had been reported as a final headline number,
the honest trial count should arguably be higher (8 pairs + 2 ML models
tried) -- flagged here because it matters in principle, even though in
practice both ML results are clearly negative regardless of how the
correction is applied.

LSTM was considered and not pursued: two models already gave a consistent, decisive result, and a third
confirmation was judged not worth the added implementation risk (LSTMs are
notably easy to leak lookahead into via windowing, and prone to the same
overfitting failure mode already observed in GBT, at higher implementation
cost).

## 10. Summary of rejected pairs

| Pair | Why rejected | Stage |
|---|---|---|
| gold | Clean negative result (ADF p=0.335, 643-day half-life) | Initial screen |
| steel | Negative return correlation (-0.347) behind a "significant" p-value | Initial screen |
| mithril | Structural break inflating apparent cointegration | Initial screen |
| adamant | Ongoing structural break, ADF correctly fails (p=0.521) | Initial screen |
| iron | 52% rolling-window cointegration rate; mismatched integration orders | Rolling diagnostic |
| rune | Unstable hedge ratio; lost money net of tax in the actual backtest | Naive backtest |
| silver | Never carried past the initial screen (not among the top candidates) | Initial screen |

Every rejection here is diagnosed, not just discarded -- each one taught
something the final result depends on (the rolling-window check exists
*because* the full-sample test missed iron's instability; the realistic-cost
stress test exists *because* the tax-only backtest's headline number was
implausible).