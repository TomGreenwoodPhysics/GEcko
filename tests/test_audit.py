import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

statsmodels_available = True
try:
    from gecko.stats.ou import rolling_zscore
    from gecko.backtest.audit import check_no_future_leakage, recompute_and_compare
except ImportError:
    statsmodels_available = False


def _make_synthetic_panel(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n)
    log_raw = pd.Series(rng.normal(scale=0.01, size=n).cumsum() + 4, index=idx)
    log_proc = pd.Series(1.0 * log_raw.values + rng.normal(scale=0.01, size=n), index=idx)
    raw_price = np.exp(log_raw)
    proc_price = np.exp(log_proc)
    panel = pd.DataFrame({"raw_price": raw_price, "proc_price": proc_price}, index=idx)
    return panel, log_raw, log_proc


def test_clean_rolling_output_passes_structural_check():
    panel, log_raw, log_proc = _make_synthetic_panel()
    roll = rolling_zscore(log_raw, log_proc, fit_window_days=500, refit_step_days=50)
    rep = check_no_future_leakage(roll)
    assert rep["passed"]
    assert rep["n_violations"] == 0
    print(f"test_clean_rolling_output_passes_structural_check: PASS ({rep['n_rows']} rows)")


def test_corrupted_fit_end_date_fails_structural_check():
    # Tamper with fit_end_date to simulate a leakage bug: set it AFTER the
    # row's own date for a block. The check must catch this, not pass anyway.
    panel, log_raw, log_proc = _make_synthetic_panel()
    roll = rolling_zscore(log_raw, log_proc, fit_window_days=500, refit_step_days=50)
    corrupted = roll.copy()
    bad_idx = corrupted.index[10]
    corrupted.loc[bad_idx, "fit_end_date"] = corrupted.index[-1]  # clearly in the future

    rep = check_no_future_leakage(corrupted)
    assert not rep["passed"]
    assert rep["n_violations"] >= 1
    print(f"test_corrupted_fit_end_date_fails_structural_check: PASS "
          f"(caught {rep['n_violations']} violation(s), as it should)")


def test_recompute_matches_recorded_for_clean_output():
    panel, log_raw, log_proc = _make_synthetic_panel()
    roll = rolling_zscore(log_raw, log_proc, fit_window_days=500, refit_step_days=50)
    rep = recompute_and_compare(panel, roll, fit_window_days=500, n_samples=5, seed=1)
    assert rep["passed"]
    assert rep["n_matched"] == rep["n_sampled"]
    print(f"test_recompute_matches_recorded_for_clean_output: PASS "
          f"({rep['n_matched']}/{rep['n_sampled']} blocks matched independently)")


def test_recompute_catches_a_wrong_recorded_value():
    # Tamper with a recorded beta so it no longer matches what an honest
    # recomputation from the raw data would produce. Must fail, not pass.
    panel, log_raw, log_proc = _make_synthetic_panel()
    roll = rolling_zscore(log_raw, log_proc, fit_window_days=500, refit_step_days=50)
    corrupted = roll.copy()
    corrupted["beta"] = corrupted["beta"] + 5.0  # obviously wrong now

    rep = recompute_and_compare(panel, corrupted, fit_window_days=500, n_samples=5, seed=1)
    assert not rep["passed"]
    assert rep["n_matched"] == 0
    print(f"test_recompute_catches_a_wrong_recorded_value: PASS "
          f"(correctly flagged 0/{rep['n_sampled']} as matching)")


if __name__ == "__main__":
    if not statsmodels_available:
        print("statsmodels not importable; cannot run.")
        sys.exit(0)
    test_clean_rolling_output_passes_structural_check()
    test_corrupted_fit_end_date_fails_structural_check()
    test_recompute_matches_recorded_for_clean_output()
    test_recompute_catches_a_wrong_recorded_value()
    print("\nAll audit tests passed.")