"""
Ornstein-Uhlenbeck fit for a spread, and the z-score signal derived from it.

The OU process:  dX = theta*(mu - X)dt + sigma*dW

Exact discretisation over a unit time step (dt=1, e.g. one day) gives an AR(1):
  X_{t+1} = c + phi*X_t + eps_t ,  phi = exp(-theta),  c = mu*(1-phi)
  Var(eps) = sigma^2 * (1-phi^2) / (2*theta)

Fitting the AR(1) by OLS and inverting these relations recovers theta, mu, and
the stationary std dev sigma_eq = sqrt(Var(eps)/(1-phi^2)). sigma_eq is the
correct scale for the z-score (not the raw residual std), since it accounts for
the variance the mean-reversion itself removes.

rolling_zscore() refits beta and the OU params on a trailing window only, then
applies them to the following block. This matters because the hedge ratio is
not stable for every pair (rune's beta ranges from ~0 to ~0.96 across 2-year
windows), so a single full-history beta would assume a relationship that
doesn't hold throughout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gecko.stats.cointegration import ols_hedge_ratio, build_spread


# ----------------------------------------------------------------------------
# Core OU fit (pure numpy, exact closed-form inversion of the AR(1) fit)
# ----------------------------------------------------------------------------

def fit_ou(spread: pd.Series, dt: float = 1.0) -> dict:
    """Fit OU parameters to a spread series via its AR(1) representation.

    Returns dict with: phi, c (the raw AR(1) coefficients), theta (mean-
    reversion speed), mu (long-run mean), sigma (instantaneous vol), sigma_eq
    (stationary std dev -- use this for z-scoring), half_life (in the same
    time units as dt), resid_std (raw AR(1) residual std, diagnostic only).

    theta/sigma_eq are NaN if phi is outside (0, 1) -- not mean-reverting.

    CAVEAT: for a true random walk, finite-sample bias pulls the ESTIMATED
    phi to just under 1, not exactly 1 -- so this returns a tiny-but-positive
    theta and a huge (economically useless) half-life rather than NaN. This
    function does not test for mean reversion; it describes its speed IF it
    exists. Use the cointegration tests (engle_granger / johansen) to decide
    that question first.
    """
    s = pd.Series(spread).dropna().astype(float)
    s_lag, s_now = s.values[:-1], s.values[1:]
    X = np.column_stack([np.ones_like(s_lag), s_lag])
    coef, *_ = np.linalg.lstsq(X, s_now, rcond=None)
    c, phi = float(coef[0]), float(coef[1])
    resid = s_now - (c + phi * s_lag)
    resid_var = float(np.var(resid, ddof=2))

    if not (0 < phi < 1):
        return {"phi": phi, "c": c, "theta": float("nan"), "mu": float("nan"),
                "sigma": float("nan"), "sigma_eq": float("nan"),
                "half_life": float("nan"), "resid_std": float(np.sqrt(resid_var))}

    theta = -np.log(phi) / dt
    mu = c / (1 - phi)
    sigma_eq = np.sqrt(resid_var / (1 - phi ** 2))
    sigma = np.sqrt(resid_var * 2 * theta / (1 - phi ** 2))
    half_life = np.log(2) / theta

    return {"phi": phi, "c": c, "theta": float(theta), "mu": float(mu),
            "sigma": float(sigma), "sigma_eq": float(sigma_eq),
            "half_life": float(half_life), "resid_std": float(np.sqrt(resid_var))}


def z_score(spread: pd.Series, mu: float, sigma_eq: float) -> pd.Series:
    """z = (spread - mu) / sigma_eq, using OU-implied (not naive rolling)
    mean and stationary std dev."""
    return ((pd.Series(spread).astype(float) - mu) / sigma_eq).rename("z")


# ----------------------------------------------------------------------------
# Walk-forward version: re-fit beta + OU params on trailing data only
# ----------------------------------------------------------------------------

def rolling_zscore(log_raw: pd.Series, log_proc: pd.Series,
                   fit_window_days: int = 730, refit_step_days: int = 30,
                   ) -> pd.DataFrame:
    """For each refit point, fit beta (OLS) and OU params on the trailing
    fit_window_days, then apply those FROZEN parameters to compute spread and
    z-score for the following refit_step_days. Slide forward and repeat.

    This never uses a parameter fitted on data that includes or postdates the
    point being scored -- the same no-lookahead discipline as the data
    cleaning step, just applied to model fitting instead of gap-filling.

    Returns a DataFrame indexed by date with columns: spread, z, beta, mu,
    sigma_eq, half_life (the last four constant within each application
    block, so you can see exactly which fit produced which z-score).
    """
    df = pd.concat([log_raw.rename("r"), log_proc.rename("p")], axis=1).dropna()
    n = len(df)
    if n < fit_window_days + refit_step_days:
        raise ValueError(f"series has {n} obs, too short for fit_window_days="
                         f"{fit_window_days} + refit_step_days={refit_step_days}")

    blocks = []
    fit_end = fit_window_days
    while fit_end < n:
        apply_end = min(fit_end + refit_step_days, n)
        fit_slice = df.iloc[fit_end - fit_window_days: fit_end]
        apply_slice = df.iloc[fit_end: apply_end]
        if len(apply_slice) == 0:
            break

        hr = ols_hedge_ratio(fit_slice["p"], fit_slice["r"])
        fit_spread = build_spread(fit_slice["r"], fit_slice["p"],
                                  beta=hr["beta"], alpha=hr["alpha"])
        ou = fit_ou(fit_spread)

        apply_spread = build_spread(apply_slice["r"], apply_slice["p"],
                                    beta=hr["beta"], alpha=hr["alpha"])
        apply_z = z_score(apply_spread, ou["mu"], ou["sigma_eq"])

        block = pd.DataFrame({"spread": apply_spread, "z": apply_z})
        block["beta"] = hr["beta"]
        block["mu"] = ou["mu"]
        block["sigma_eq"] = ou["sigma_eq"]
        block["half_life"] = ou["half_life"]
        block["fit_end_date"] = fit_slice.index[-1]  # explicit, auditable:
        # every row in this block must have fit_end_date < its own date,
        # or the parameters used to score it weren't truly causal.
        blocks.append(block)

        fit_end = apply_end

    return pd.concat(blocks)