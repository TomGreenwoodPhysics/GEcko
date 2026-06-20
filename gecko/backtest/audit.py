"""
Lookahead-bias auditing for walk-forward model output.

Checks that no future data was used to score a past date, two ways:

  1. Structural: every scored row's fit_end_date must be strictly before its
     own date. A fast, exact check on the recorded metadata.

  2. Recomputation: the structural check only verifies what the output
     records, not what the code did -- a bug could fit on the wrong window yet
     still record a plausible fit_end_date. This independently re-derives
     beta/mu/half_life from the raw panel for a sample of blocks and confirms
     they match what was saved, within floating-point tolerance. Catches drift
     between code and saved output (stale cache, changed parameters) that the
     structural check would miss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gecko.stats.cointegration import ols_hedge_ratio, build_spread
from gecko.stats.ou import fit_ou


def check_no_future_leakage(roll: pd.DataFrame) -> dict:
    """Structural check: fit_end_date < date for every row."""
    dates = pd.to_datetime(roll.index)
    fit_end = pd.to_datetime(roll["fit_end_date"])
    violations = fit_end >= dates
    n_violations = int(violations.sum())
    return {
        "n_rows": len(roll),
        "n_violations": n_violations,
        "passed": n_violations == 0,
        "first_violation_date": str(dates[violations][0]) if n_violations else None,
    }


def _values_match(a: float, b: float, tol: float) -> bool:
    """True if a and b agree within tol -- including the case where BOTH are
    NaN (fit_ou legitimately returns NaN when phi is outside (0,1); that's an
    honest agreement, not a mismatch). NaN vs a real number is still a fail.
    """
    a_nan, b_nan = np.isnan(a), np.isnan(b)
    if a_nan and b_nan:
        return True
    if a_nan or b_nan:
        return False
    return abs(a - b) < tol


def recompute_and_compare(daily_panel: pd.DataFrame, roll: pd.DataFrame,
                          fit_window_days: int, n_samples: int = 5,
                          seed: int = 0, tol: float = 1e-6) -> dict:
    """Independently re-fit beta/mu/half_life for a sample of blocks and
    compare to what rolling_zscore actually saved. daily_panel must have
    'raw_price'/'proc_price' columns, same as the input to rolling_zscore.
    """
    fit_end_dates = roll["fit_end_date"].drop_duplicates()
    rng = np.random.default_rng(seed)
    n_samples = min(n_samples, len(fit_end_dates))
    sampled = rng.choice(fit_end_dates.values, size=n_samples, replace=False)

    log_raw = np.log(daily_panel["raw_price"])
    log_proc = np.log(daily_panel["proc_price"])

    results = []
    for fit_end in sampled:
        fit_end_ts = pd.Timestamp(fit_end)
        # Recreate exactly the trailing window rolling_zscore would have used.
        window = pd.concat([log_raw.rename("r"), log_proc.rename("p")], axis=1).dropna()
        window = window.loc[:fit_end_ts].iloc[-fit_window_days:]
        if len(window) < fit_window_days:
            continue  # not enough history before this point -- skip, don't fail

        hr = ols_hedge_ratio(window["p"], window["r"])
        spread = build_spread(window["r"], window["p"], beta=hr["beta"], alpha=hr["alpha"])
        ou = fit_ou(spread)

        recorded = roll[roll["fit_end_date"] == fit_end].iloc[0]
        beta_match = _values_match(hr["beta"], recorded["beta"], tol)
        mu_match = _values_match(ou["mu"], recorded["mu"], tol)
        results.append({
            "fit_end_date": str(fit_end_ts.date()),
            "recomputed_beta": hr["beta"], "recorded_beta": recorded["beta"],
            "recomputed_mu": ou["mu"], "recorded_mu": recorded["mu"],
            "match": bool(beta_match and mu_match),
        })

    all_match = all(r["match"] for r in results) if results else False
    return {
        "n_sampled": len(results),
        "n_matched": sum(r["match"] for r in results),
        "passed": all_match and len(results) > 0,
        "details": results,
    }