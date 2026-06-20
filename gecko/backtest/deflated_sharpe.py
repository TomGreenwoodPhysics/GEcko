"""
Probabilistic and Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2012/2014).

A naive Sharpe ratio overstates confidence when returns are non-normal and
when the strategy was selected as the best of several trials. Both apply here:
the winning pair came from a screen of 8 candidates, on relatively few
independent bets. This module corrects for both effects.

  1. Probabilistic Sharpe Ratio (PSR): adjusts the Sharpe ratio's standard
     error for the return series' skew and kurtosis, giving P(true SR >
     benchmark) rather than a point estimate.

  2. Deflated Sharpe Ratio (DSR): sets the benchmark to the expected maximum
     Sharpe across N independent trials under the null of no skill
     (extreme-value approximation), then asks whether the observed Sharpe
     clears that bar rather than just zero. More trials -> higher bar.

Choosing N is a judgment call. It defaults to 8, the number of pairs screened
(the source of the multiple-testing exposure). Counting parameter settings too
would raise N; using the pair count alone is the more conservative direction
(fewer trials -> an easier bar).

Moments are computed at the native sampling frequency (daily). Skew and
kurtosis don't scale cleanly under time-aggregation, so SR_hat here is a daily
Sharpe -- pass in daily returns, not annualised figures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def _moments(returns: pd.Series) -> dict:
    """Sample mean/std/skew/raw-kurtosis (normal -> kurtosis=3, NOT excess)."""
    r = pd.Series(returns).dropna().astype(float).values
    mu = r.mean()
    sigma = r.std(ddof=1)
    centered = r - mu
    skew = np.mean(centered ** 3) / sigma ** 3
    kurtosis_raw = np.mean(centered ** 4) / sigma ** 4  # normal -> ~3.0
    return {"mean": float(mu), "std": float(sigma), "skew": float(skew),
            "kurtosis": float(kurtosis_raw), "T": len(r)}


def sharpe_variance(sr_hat: float, skew: float, kurtosis: float, T: int) -> float:
    """Var(SR_hat) adjusted for non-normality (Mertens 2002 / Bailey-LdP).
    kurtosis here is RAW (normal=3), not excess. Reduces to the classical
    (1 + 0.5*SR^2)/(T-1) when skew=0, kurtosis=3.
    """
    return (1 - skew * sr_hat + ((kurtosis - 1) / 4) * sr_hat ** 2) / (T - 1)


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sr: float = 0.0) -> dict:
    """PSR(benchmark) = P(true SR > benchmark), adjusting for the return
    series' own skew/kurtosis. Returns dict with psr, sr_hat, and the moments
    used, so the calculation is auditable.
    """
    m = _moments(returns)
    sr_hat = m["mean"] / m["std"]
    var_sr = sharpe_variance(sr_hat, m["skew"], m["kurtosis"], m["T"])
    z = (sr_hat - benchmark_sr) / np.sqrt(var_sr)
    psr = float(norm.cdf(z))
    return {"psr": psr, "sr_hat": float(sr_hat), "benchmark_sr": float(benchmark_sr),
            "var_sr": float(var_sr), **m}


def expected_max_sharpe_under_null(sr_std: float, n_trials: int) -> float:
    """Expected maximum Sharpe ratio across n_trials independent trials, each
    drawn from a null distribution with std dev sr_std (mean 0 -- no real
    skill). Extreme-value (Gumbel) approximation from Bailey & Lopez de Prado.
    Monotonically increasing in n_trials by construction: searching more
    strategies raises the bar a real result has to clear.
    """
    if n_trials <= 1:
        return 0.0
    z1 = norm.ppf(1 - 1 / n_trials)
    z2 = norm.ppf(1 - 1 / (n_trials * np.e))
    return float(sr_std * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int = 8) -> dict:
    """DSR = PSR evaluated against the expected-max-Sharpe-under-null
    benchmark for n_trials trials, instead of against 0.

    n_trials default (8) = the number of candidate pairs screened
    (see module docstring) -- the actual source of multiple-testing exposure
    in this project. Override if you've also varied other parameters
    (entry_threshold, fit_window_days, etc.) across documented trials.
    """
    m = _moments(returns)
    sr_hat = m["mean"] / m["std"]
    var_sr = sharpe_variance(sr_hat, m["skew"], m["kurtosis"], m["T"])
    sr_std = np.sqrt(var_sr)

    sr0 = expected_max_sharpe_under_null(sr_std, n_trials)
    psr_report = probabilistic_sharpe_ratio(returns, benchmark_sr=sr0)

    return {
        "dsr": psr_report["psr"],
        "sr_hat_daily": float(sr_hat),
        "sr0_benchmark": sr0,
        "n_trials": n_trials,
        "T": m["T"],
        "skew": m["skew"],
        "kurtosis": m["kurtosis"],
        "interpretation": (
            f"P(true skill exceeds what the best of {n_trials} random "
            f"strategies would show by chance) = {psr_report['psr']:.1%}"
        ),
    }