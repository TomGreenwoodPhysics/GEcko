import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from gecko.backtest.deflated_sharpe import (
    probabilistic_sharpe_ratio, expected_max_sharpe_under_null,
    deflated_sharpe_ratio, sharpe_variance,
)


def test_sharpe_variance_matches_classical_for_normal():
    # skew=0, kurtosis=3 (normal) should reduce to (1+0.5*SR^2)/(T-1)
    sr, T = 0.05, 1000
    var = sharpe_variance(sr_hat=sr, skew=0.0, kurtosis=3.0, T=T)
    classical = (1 + 0.5 * sr ** 2) / (T - 1)
    assert abs(var - classical) < 1e-12
    print("test_sharpe_variance_matches_classical_for_normal: PASS")


def test_psr_near_half_for_zero_mean_returns():
    # True SR=0: averaged over many seeds, PSR(0) should center near 0.5 --
    # no systematic evidence either way when there's genuinely nothing there.
    rng = np.random.default_rng(0)
    psrs = []
    for seed in range(200):
        r = pd.Series(rng.normal(loc=0.0, scale=1.0, size=500))
        psrs.append(probabilistic_sharpe_ratio(r, benchmark_sr=0.0)["psr"])
    avg_psr = np.mean(psrs)
    assert abs(avg_psr - 0.5) < 0.05, f"avg PSR {avg_psr} should be near 0.5"
    print(f"test_psr_near_half_for_zero_mean_returns: PASS (avg PSR={avg_psr:.3f})")


def test_psr_high_for_strong_genuine_signal():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(loc=0.08, scale=1.0, size=2000))  # larger true SR, comfortable margin
    res = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
    assert res["psr"] > 0.95
    print(f"test_psr_high_for_strong_genuine_signal: PASS (psr={res['psr']:.4f})")


def test_expected_max_increases_with_n_trials():
    sr_std = 0.05
    vals = [expected_max_sharpe_under_null(sr_std, n) for n in [1, 5, 20, 100, 1000]]
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    assert vals[0] == 0.0  # n_trials=1 -> no multiple-testing inflation
    print(f"test_expected_max_increases_with_n_trials: PASS {[round(v,4) for v in vals]}")


def test_deflated_sharpe_penalizes_more_trials():
    # Same returns, more trials searched -> LOWER confidence (DSR), since the
    # bar to clear is higher. This is the entire point of the correction.
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(loc=0.03, scale=1.0, size=1000))
    dsr_few = deflated_sharpe_ratio(r, n_trials=1)["dsr"]
    dsr_many = deflated_sharpe_ratio(r, n_trials=1000)["dsr"]
    assert dsr_many < dsr_few
    print(f"test_deflated_sharpe_penalizes_more_trials: PASS "
          f"(n=1: {dsr_few:.3f}, n=1000: {dsr_many:.3f})")


def test_deflated_sharpe_report_fields_present():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(loc=0.02, scale=1.0, size=500))
    rep = deflated_sharpe_ratio(r, n_trials=8)
    for key in ["dsr", "sr_hat_daily", "sr0_benchmark", "n_trials", "T",
               "skew", "kurtosis", "interpretation"]:
        assert key in rep
    print("test_deflated_sharpe_report_fields_present: PASS")


if __name__ == "__main__":
    test_sharpe_variance_matches_classical_for_normal()
    test_psr_near_half_for_zero_mean_returns()
    test_psr_high_for_strong_genuine_signal()
    test_expected_max_increases_with_n_trials()
    test_deflated_sharpe_penalizes_more_trials()
    test_deflated_sharpe_report_fields_present()
    print("\nAll deflated Sharpe tests passed.")