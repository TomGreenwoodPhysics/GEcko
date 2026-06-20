import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from gecko.stats.ou import fit_ou, z_score, rolling_zscore


def simulate_ou(theta, mu, sigma, n=20000, dt=1.0, seed=0):
    """Exact OU discretisation -- not an Euler approximation -- so a
    successful parameter recovery test is a real check of the math, not just
    of numerical noise tolerance."""
    rng = np.random.default_rng(seed)
    phi = np.exp(-theta * dt)
    eps_std = sigma * np.sqrt((1 - phi**2) / (2 * theta))
    x = np.zeros(n)
    x[0] = mu
    for t in range(1, n):
        x[t] = mu + (x[t-1] - mu) * phi + rng.normal(scale=eps_std)
    return pd.Series(x)


def test_fit_ou_recovers_known_parameters():
    true_theta, true_mu, true_sigma = 0.05, 2.0, 0.3
    s = simulate_ou(true_theta, true_mu, true_sigma, n=50000)
    fit = fit_ou(s)
    assert abs(fit["theta"] - true_theta) / true_theta < 0.1
    assert abs(fit["mu"] - true_mu) < 0.05
    assert abs(fit["sigma"] - true_sigma) / true_sigma < 0.1
    print(f"test_fit_ou_recovers_known_parameters: PASS "
          f"(theta {fit['theta']:.4f} vs {true_theta}, mu {fit['mu']:.4f} vs "
          f"{true_mu}, sigma {fit['sigma']:.4f} vs {true_sigma})")


def test_fit_ou_half_life_matches_formula():
    # half-life = ln(2)/theta by construction; check the fit's internal
    # consistency, not just against simulated data.
    s = simulate_ou(theta=0.1, mu=0.0, sigma=1.0, n=30000)
    fit = fit_ou(s)
    expected_hl = np.log(2) / fit["theta"]
    assert abs(fit["half_life"] - expected_hl) < 1e-9
    print("test_fit_ou_half_life_matches_formula: PASS")


def test_fit_ou_useless_for_non_mean_reverting():
    # A random walk has phi just under 1 (finite-sample bias), so theta comes
    # back tiny-but-positive rather than NaN -- same documented behavior as
    # spread_half_life(). The contract here is that the result is USELESS as
    # a trading signal (huge half-life, blown-up sigma_eq), not that it's
    # literally NaN. This is exactly why the cointegration test, not this fit,
    # decides whether mean reversion exists at all.
    rng = np.random.default_rng(1)
    s = pd.Series(rng.normal(size=5000).cumsum())
    fit = fit_ou(s)
    assert np.isnan(fit["half_life"]) or fit["half_life"] > 200
    print(f"test_fit_ou_useless_for_non_mean_reverting: PASS "
          f"(half_life={fit['half_life']:.0f}, phi={fit['phi']:.4f})")


def test_z_score_zero_mean_unit_scale():
    # By construction: if mu/sigma_eq are the TRUE generating parameters,
    # z should have ~zero mean and ~unit variance over a long simulation.
    true_theta, true_mu, true_sigma = 0.05, 1.5, 0.2
    s = simulate_ou(true_theta, true_mu, true_sigma, n=50000)
    fit = fit_ou(s)
    z = z_score(s, fit["mu"], fit["sigma_eq"])
    assert abs(z.mean()) < 0.05
    assert abs(z.std() - 1.0) < 0.05
    print(f"test_z_score_zero_mean_unit_scale: PASS (mean={z.mean():.3f}, "
          f"std={z.std():.3f})")


def test_rolling_zscore_never_uses_future_fit():
    # Construct a series with a clean BREAK: beta=1 for the first half,
    # beta=3 for the second half. A window straddling the break should fit
    # something in between; a window fully inside one regime should recover
    # that regime's beta. Crucially, blocks from before the break must not
    # reflect the after-break beta (that would mean the fit peeked ahead).
    rng = np.random.default_rng(2)
    n = 2200
    log_raw = pd.Series(rng.normal(scale=0.01, size=n).cumsum() + 4)
    beta_true = np.where(np.arange(n) < n // 2, 1.0, 3.0)
    log_proc = pd.Series(beta_true * log_raw.values + rng.normal(scale=0.005, size=n))

    roll = rolling_zscore(log_raw, log_proc, fit_window_days=500, refit_step_days=100)

    early_betas = roll["beta"].iloc[:200].unique()
    late_betas = roll["beta"].iloc[-200:].unique()
    assert early_betas.max() < 2.0, "early blocks must not reflect the later beta=3 regime"
    assert late_betas.min() > 1.5, "late blocks should have picked up the beta=3 regime by now"
    print(f"test_rolling_zscore_never_uses_future_fit: PASS "
          f"(early beta~{early_betas[-1]:.2f}, late beta~{late_betas[-1]:.2f})")


def test_rolling_zscore_output_shape():
    rng = np.random.default_rng(3)
    n = 1000
    log_raw = pd.Series(rng.normal(scale=0.01, size=n).cumsum() + 4)
    log_proc = pd.Series(1.0 * log_raw.values + rng.normal(scale=0.01, size=n))
    roll = rolling_zscore(log_raw, log_proc, fit_window_days=400, refit_step_days=50)
    assert set(roll.columns) == {"spread", "z", "beta", "mu", "sigma_eq",
                                 "half_life", "fit_end_date"}
    assert len(roll) == n - 400
    print("test_rolling_zscore_output_shape: PASS")


if __name__ == "__main__":
    test_fit_ou_recovers_known_parameters()
    test_fit_ou_half_life_matches_formula()
    test_fit_ou_useless_for_non_mean_reverting()
    test_z_score_zero_mean_unit_scale()
    test_rolling_zscore_never_uses_future_fit()
    test_rolling_zscore_output_shape()
    print("\nAll OU tests passed.")