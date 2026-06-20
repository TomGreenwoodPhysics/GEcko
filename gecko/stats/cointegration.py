"""
Cointegration test battery.

Given a clean two-leg daily panel, determines whether the series are
cointegrated, estimates the hedge ratio, and measures how fast the spread
mean-reverts.

Notes:

  * Tests run on log prices. The production relationship is multiplicative
    (bar ~ k * ore), which is additive in logs (log bar ~ log k + log ore),
    so a clean proportional pair should give a hedge ratio near 1 in log
    space -- a check worth making per pair.

  * Engle-Granger is asymmetric: regressing proc on raw vs raw on proc can
    give different residuals and verdicts, so both directions are run and
    reported.

  * The test uses statsmodels.coint(), not a plain ADF on the OLS residuals.
    Ordinary ADF p-values are invalid on regression residuals because the OLS
    step absorbs some of the non-stationarity; coint() applies the correct
    MacKinnon critical values.

  * Johansen is included as a multivariate cross-check that needs no choice of
    dependent variable. Agreement with Engle-Granger is stronger evidence than
    either test alone.

statsmodels is a required dependency; the module raises on import if it's
missing rather than degrading to NaNs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller, coint
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gecko.stats.cointegration requires statsmodels. "
        "Install it with:  pip install statsmodels"
    ) from exc


# ----------------------------------------------------------------------------
# Stationarity / integration order
# ----------------------------------------------------------------------------

def adf(series: pd.Series, regression: str = "c") -> dict:
    """Augmented Dickey-Fuller test. H0: a unit root is present (non-stationary).
    p < 0.05 => reject H0 => evidence the series is stationary.
    """
    s = pd.Series(series).dropna().astype(float)
    stat, pvalue, usedlag, nobs, crit, _ = adfuller(
        s.values, regression=regression, autolag="AIC")
    return {
        "adf_stat": float(stat),
        "pvalue": float(pvalue),
        "used_lag": int(usedlag),
        "nobs": int(nobs),
        "crit_5pct": float(crit["5%"]),
        "stationary_at_5pct": bool(pvalue < 0.05),
    }


def integration_order(series: pd.Series) -> dict:
    """Classify a series as roughly I(0), I(1), or ambiguous by ADF on the
    levels and on the first difference. Cointegration is the right framework
    only when the individual series are I(1) (non-stationary in levels,
    stationary once differenced).
    """
    s = pd.Series(series).dropna().astype(float)
    lvl = adf(s)
    dif = adf(s.diff().dropna())
    if (not lvl["stationary_at_5pct"]) and dif["stationary_at_5pct"]:
        verdict = "I(1)"
    elif lvl["stationary_at_5pct"]:
        verdict = "I(0)"
    else:
        verdict = "ambiguous"
    return {"levels": lvl, "first_diff": dif, "verdict": verdict}


# ----------------------------------------------------------------------------
# Hedge ratio via OLS (numpy -- no statsmodels needed, fully unit-tested)
# ----------------------------------------------------------------------------

def ols_hedge_ratio(y: pd.Series, x: pd.Series) -> dict:
    """OLS  y = alpha + beta * x.  Returns alpha, beta, and residual series.
    Pure numpy so it is testable without statsmodels.
    """
    y = pd.Series(y).astype(float)
    x = pd.Series(x).astype(float)
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    X = np.column_stack([np.ones(len(df)), df["x"].values])
    coef, *_ = np.linalg.lstsq(X, df["y"].values, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    resid = df["y"].values - (alpha + beta * df["x"].values)
    return {"alpha": alpha, "beta": beta,
            "resid": pd.Series(resid, index=df.index)}


def build_spread(log_raw: pd.Series, log_proc: pd.Series,
                 beta: float, alpha: float = 0.0) -> pd.Series:
    """Spread = log_proc - (alpha + beta * log_raw). This is the series whose
    stationarity IS the cointegration, and which the z-score strategy trades.
    """
    df = pd.concat([log_proc.rename("p"), log_raw.rename("r")], axis=1).dropna()
    return (df["p"] - (alpha + beta * df["r"])).rename("spread")


def spread_half_life(spread: pd.Series) -> float:
    """Half-life of mean reversion (in observations) from an AR(1) fit:
    spread_t = a + phi * spread_{t-1} + e.  half-life = -ln(2)/ln(phi).
    This equals the discrete OU half-life. Full continuous-time OU fitting
    (theta, mu, sigma_eq for the z-score bands) is in gecko.stats.ou.

    IMPORTANT: this is a DESCRIPTIVE speed measure, not a stationarity test.
    A non-cointegrated near-random-walk series has phi just under 1 (finite-
    sample bias), so it returns a very LARGE finite half-life, not NaN. Use the
    cointegration test (coint p-value / Johansen) to decide whether mean
    reversion exists; use this only to describe how fast it is once it does.
    Returns NaN only for the degenerate non-mean-reverting cases phi<=0 or
    phi>=1.
    """
    s = pd.Series(spread).dropna().astype(float)
    s_lag, s_now = s.values[:-1], s.values[1:]
    X = np.column_stack([np.ones_like(s_lag), s_lag])
    coef, *_ = np.linalg.lstsq(X, s_now, rcond=None)
    phi = coef[1]
    if 0 < phi < 1:
        return float(-np.log(2) / np.log(phi))
    return float("nan")


# ----------------------------------------------------------------------------
# Engle-Granger (both directions, correct critical values)
# ----------------------------------------------------------------------------

def engle_granger(log_raw: pd.Series, log_proc: pd.Series) -> dict:
    """Engle-Granger in both directions. Uses statsmodels.coint() for the
    test statistic/p-value (correct EG critical values) and a separate OLS for
    the hedge ratio. The 'proc~raw' direction is the economically natural one
    for a raw->processed pair (processed price set by raw cost + conversion),
    so we mark it as recommended.
    """
    df = pd.concat([log_raw.rename("r"), log_proc.rename("p")], axis=1).dropna()
    r, p = df["r"], df["p"]

    # statsmodels.coint(y0, y1): tests whether residual of y0 on y1 is stationary
    t_pr, pval_pr, crit_pr = coint(p, r, trend="c")   # proc ~ raw
    t_rp, pval_rp, crit_rp = coint(r, p, trend="c")   # raw ~ proc

    hr_pr = ols_hedge_ratio(p, r)   # proc = a + beta*raw
    hr_rp = ols_hedge_ratio(r, p)   # raw  = a + beta*proc

    return {
        "proc_on_raw": {
            "direction": "log_proc ~ log_raw (recommended)",
            "coint_t": float(t_pr), "pvalue": float(pval_pr),
            "crit_5pct": float(crit_pr[1]),
            "alpha": hr_pr["alpha"], "beta": hr_pr["beta"],
            "cointegrated_at_5pct": bool(pval_pr < 0.05),
        },
        "raw_on_proc": {
            "direction": "log_raw ~ log_proc",
            "coint_t": float(t_rp), "pvalue": float(pval_rp),
            "crit_5pct": float(crit_rp[1]),
            "alpha": hr_rp["alpha"], "beta": hr_rp["beta"],
            "cointegrated_at_5pct": bool(pval_rp < 0.05),
        },
        "recommended_beta": hr_pr["beta"],
        "recommended_alpha": hr_pr["alpha"],
    }


# ----------------------------------------------------------------------------
# Johansen (multivariate cross-check)
# ----------------------------------------------------------------------------

def johansen(log_raw: pd.Series, log_proc: pd.Series,
             det_order: int = 0, k_ar_diff: int = 1) -> dict:
    """Johansen cointegration test on [log_raw, log_proc].

    det_order: -1 no deterministic term, 0 constant (usual for a spread with a
               non-zero mean), 1 linear trend.
    k_ar_diff: number of lagged differences in the VECM (1 is a common default;
               could be selected by AIC on a VAR -- left as a knob).

    Compares the trace and max-eigenvalue statistics for rank r=0 against their
    95% critical values. Rejecting r=0 is evidence of (at least) one
    cointegrating relation. The hedge ratio is read off the first cointegrating
    eigenvector, normalised on the raw leg.
    """
    df = pd.concat([log_raw.rename("r"), log_proc.rename("p")], axis=1).dropna()
    res = coint_johansen(df[["r", "p"]].values, det_order, k_ar_diff)

    # .lr1: trace statistics, one per hypothesised rank r = 0, 1, ...
    # .cvt: trace critical values, shape (n_vars, 3) = [90%, 95%, 99%]
    trace_stat_r0 = float(res.lr1[0])
    trace_crit_95_r0 = float(res.cvt[0, 1])
    maxeig_stat_r0 = float(res.lr2[0])
    maxeig_crit_95_r0 = float(res.cvm[0, 1])

    # First cointegrating vector = first column of .evec; normalise on raw leg
    # so the relation reads  raw + (v_p/v_r)*proc ~ stationary, i.e. the hedge
    # ratio on raw is beta = -v_p/v_r when written  proc = beta*raw.
    v = res.evec[:, 0]
    v_r, v_p = float(v[0]), float(v[1])
    beta_johansen = -v_r / v_p if v_p != 0 else float("nan")

    return {
        "trace_stat_r0": trace_stat_r0,
        "trace_crit_95_r0": trace_crit_95_r0,
        "trace_rejects_r0_at_5pct": bool(trace_stat_r0 > trace_crit_95_r0),
        "maxeig_stat_r0": maxeig_stat_r0,
        "maxeig_crit_95_r0": maxeig_crit_95_r0,
        "maxeig_rejects_r0_at_5pct": bool(maxeig_stat_r0 > maxeig_crit_95_r0),
        "beta_johansen": beta_johansen,
        "det_order": det_order,
        "k_ar_diff": k_ar_diff,
    }


def integration_caveat(raw_verdict: str, proc_verdict: str) -> str | None:
    """The standard cointegration framework assumes both legs are I(1). Flag
    when that assumption is violated, so a misleading 'cointegrated' verdict
    doesn't get reported without context.
    """
    if raw_verdict == "I(1)" and proc_verdict == "I(1)":
        return None
    if raw_verdict == "I(0)" and proc_verdict == "I(0)":
        return ("both legs are already individually stationary (I(0)) -- this "
                "is NOT the classical I(1)+I(1)->I(0) cointegration setup. "
                "Report the spread's own tightness, not 'cointegration'.")
    return (f"mismatched integration orders (raw {raw_verdict}, proc "
            f"{proc_verdict}) -- a linear combination of an I(1) and an I(0) "
            f"series is generically I(1) unless beta on the I(1) leg is ~0. "
            f"Treat any 'cointegrated' verdict here as suspect until the "
            f"integration-order classification is double-checked (try "
            f"different ADF lag specs / inspect the level series directly).")


# ----------------------------------------------------------------------------
# Rolling-window Engle-Granger: WHEN does cointegration hold, not just whether
# ----------------------------------------------------------------------------

def rolling_engle_granger(log_raw: pd.Series, log_proc: pd.Series,
                          window_days: int = 730, step_days: int = 30
                          ) -> pd.DataFrame:
    """Run Engle-Granger (proc~raw) on a trailing window, slid forward by
    step_days, over the whole series. Each row's date is the window's END
    date, and only uses data up to and including that date (no lookahead).

    Returns a DataFrame indexed by window-end date with columns:
    pvalue, beta, cointegrated_at_5pct, n_obs.
    """
    df = pd.concat([log_raw.rename("r"), log_proc.rename("p")], axis=1).dropna()
    if len(df) < window_days:
        raise ValueError(f"series has {len(df)} obs, shorter than window_days="
                         f"{window_days}")

    rows = []
    end_positions = range(window_days, len(df) + 1, step_days)
    for end_pos in end_positions:
        start_pos = end_pos - window_days
        sub = df.iloc[start_pos:end_pos]
        try:
            t, pval, _ = coint(sub["p"], sub["r"], trend="c")
            hr = ols_hedge_ratio(sub["p"], sub["r"])
            rows.append({
                "date": sub.index[-1],
                "pvalue": float(pval),
                "beta": hr["beta"],
                "cointegrated_at_5pct": bool(pval < 0.05),
                "n_obs": len(sub),
            })
        except Exception:  # noqa: BLE001 -- a single bad window shouldn't kill the scan
            rows.append({"date": sub.index[-1], "pvalue": float("nan"),
                        "beta": float("nan"), "cointegrated_at_5pct": False,
                        "n_obs": len(sub)})

    return pd.DataFrame(rows).set_index("date")

def run_pair_cointegration(panel: pd.DataFrame,
                           raw_col: str = "raw_price",
                           proc_col: str = "proc_price") -> dict:
    """Run the whole battery on one clean daily panel and return a tidy nested
    report. Prices are converted to logs internally.
    """
    df = panel[[raw_col, proc_col]].dropna()
    df = df[(df[raw_col] > 0) & (df[proc_col] > 0)]
    log_raw = np.log(df[raw_col])
    log_proc = np.log(df[proc_col])

    eg = engle_granger(log_raw, log_proc)
    joh = johansen(log_raw, log_proc)

    # Spread + half-life using the recommended (proc~raw) hedge ratio
    spread = build_spread(log_raw, log_proc,
                          beta=eg["recommended_beta"],
                          alpha=eg["recommended_alpha"])
    hl = spread_half_life(spread)

    int_raw = integration_order(log_raw)
    int_proc = integration_order(log_proc)
    caveat = integration_caveat(int_raw["verdict"], int_proc["verdict"])

    return {
        "n_obs": int(len(df)),
        "date_start": str(df.index.min().date()) if hasattr(df.index.min(), "date") else str(df.index.min()),
        "date_end": str(df.index.max().date()) if hasattr(df.index.max(), "date") else str(df.index.max()),
        "integration_raw": int_raw,
        "integration_proc": int_proc,
        "integration_caveat": caveat,
        "engle_granger": eg,
        "johansen": joh,
        "recommended_beta": eg["recommended_beta"],
        "spread_half_life_days": hl,
        "spread": spread,  # the series itself, for plotting and OU fitting
    }