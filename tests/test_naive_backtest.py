import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from gecko.backtest.naive import run_naive_backtest


def test_hand_traced_rotation():
    """6 days, hand-calculated expected equity at every step. If this fails,
    the rotation/tax/mark-to-market logic has a real bug -- these numbers are
    not approximate, they're traced by hand from the same rules the code
    implements.

      day0: z=0.0   raw=100 proc=50   -> no position yet, equity=1.0
      day1: z=2.0   raw=100 proc=50   -> enter raw @100,        equity=1.0
      day2: z=1.8   raw=110 proc=52   -> hold raw,              equity=1.10
      day3: z=0.0   raw=105 proc=53   -> hold raw (in band),    equity=1.05
      day4: z=-2.0  raw=108 proc=50   -> exit raw@108 (tax 2%), enter proc@50
                                          capital = 1.05238...(via day1 base)
                                          exact: 1.0*(108/100)*(1-0.02)=1.0584
                                          equity=1.0584
      day5: z=-1.0  raw=107 proc=55   -> hold proc,             equity=1.0584*55/50=1.16424
    """
    idx = pd.date_range("2024-01-01", periods=6)
    z = pd.Series([0.0, 2.0, 1.8, 0.0, -2.0, -1.0], index=idx)
    raw = pd.Series([100, 100, 110, 105, 108, 107], index=idx, dtype=float)
    proc = pd.Series([50, 50, 52, 53, 50, 55], index=idx, dtype=float)

    res = run_naive_backtest(z, raw, proc, entry_threshold=1.5, tax=0.02,
                             initial_capital=1.0)
    eq = res["equity_curve"]

    expected = [1.0, 1.0, 1.10, 1.05, 1.0584, 1.16424]
    for got, exp in zip(eq.values, expected):
        assert abs(got - exp) < 1e-9, f"got {got}, expected {exp}"

    # exactly one rotation occurred (day1 entry, day4 exit+enter)
    assert res["summary"]["n_rotations"] == 2  # enter_raw (day1) + enter_proc (day4)
    print("test_hand_traced_rotation: PASS (all 6 equity points match exactly)")


def test_no_threshold_crossed_stays_flat():
    idx = pd.date_range("2024-01-01", periods=5)
    z = pd.Series([0.1, -0.2, 0.3, -0.1, 0.0], index=idx)  # never crosses 1.5
    raw = pd.Series([100.0] * 5, index=idx)
    proc = pd.Series([50.0] * 5, index=idx)

    res = run_naive_backtest(z, raw, proc, entry_threshold=1.5)
    assert (res["equity_curve"] == 1.0).all()
    assert res["summary"]["n_rotations"] == 0
    assert np.isnan(res["summary"]["sharpe_naive"])  # zero vol -> undefined, not crash
    print("test_no_threshold_crossed_stays_flat: PASS")


def test_tax_applied_only_on_exit_not_entry():
    # A single rotation with prices unchanged isolates the tax's effect:
    # capital should shrink by EXACTLY (1-tax) on the day of exit.
    idx = pd.date_range("2024-01-01", periods=3)
    z = pd.Series([2.0, 2.0, -2.0], index=idx)
    raw = pd.Series([100.0, 100.0, 100.0], index=idx)
    proc = pd.Series([50.0, 50.0, 50.0], index=idx)

    res = run_naive_backtest(z, raw, proc, entry_threshold=1.5, tax=0.02)
    eq = res["equity_curve"]
    assert abs(eq.iloc[1] - 1.0) < 1e-9       # held raw, price unchanged
    assert abs(eq.iloc[2] - 0.98) < 1e-9      # exit raw -> tax only, no price move
    print("test_tax_applied_only_on_exit_not_entry: PASS")


def test_max_drawdown_sign_and_magnitude():
    idx = pd.date_range("2024-01-01", periods=4)
    equity_like_z = pd.Series([2.0, 2.0, 2.0, 2.0], index=idx)
    raw = pd.Series([100.0, 80.0, 60.0, 90.0], index=idx)  # down 40%, then up
    proc = pd.Series([50.0, 50.0, 50.0, 50.0], index=idx)

    res = run_naive_backtest(equity_like_z, raw, proc, entry_threshold=1.5, tax=0.0)
    # peak 1.0 (day0), trough 0.6 (day2) -> drawdown = 0.6/1.0 - 1 = -0.4
    assert abs(res["summary"]["max_drawdown"] - (-0.4)) < 1e-9
    print("test_max_drawdown_sign_and_magnitude: PASS")


def test_default_half_spread_matches_original_behavior():
    # half_spread_raw/proc default to 0.0 -- must reproduce the tax-only
    # numbers exactly, or the spread-cost extension silently changed old
    # results.
    idx = pd.date_range("2024-01-01", periods=6)
    z = pd.Series([0.0, 2.0, 1.8, 0.0, -2.0, -1.0], index=idx)
    raw = pd.Series([100, 100, 110, 105, 108, 107], index=idx, dtype=float)
    proc = pd.Series([50, 50, 52, 53, 50, 55], index=idx, dtype=float)
    res = run_naive_backtest(z, raw, proc, entry_threshold=1.5, tax=0.02)
    expected = [1.0, 1.0, 1.10, 1.05, 1.0584, 1.16424]
    for got, exp in zip(res["equity_curve"].values, expected):
        assert abs(got - exp) < 1e-9
    print("test_default_half_spread_matches_original_behavior: PASS")


def test_hand_traced_with_spread_costs():
    """Same shape as the tax-only trace, but with half_spread_raw=0.01 and
    half_spread_proc=0.02, charged on every entry AND exit. Hand-computed:

      day0: z=2.0  raw=100 proc=50 -> enter raw, haircut 1%: capital=0.99
      day1: z=2.0  raw=110 proc=52 -> hold:                  equity=0.99*1.10=1.089
      day2: z=-2.0 raw=108 proc=50 -> exit raw (haircut+tax), enter proc (haircut):
              capital = 0.99*1.08 = 1.0692
                      *0.99 (exit spread) = 1.058508
                      *0.98 (tax)         = 1.03733784
                      *0.98 (entry spread)= 1.0165910832
              equity = 1.0165910832
      day3: z=-1.0 raw=107 proc=55 -> hold: equity = 1.0165910832*1.1 = 1.11825019152
    """
    idx = pd.date_range("2024-01-01", periods=4)
    z = pd.Series([2.0, 2.0, -2.0, -1.0], index=idx)
    raw = pd.Series([100.0, 110.0, 108.0, 107.0], index=idx)
    proc = pd.Series([50.0, 52.0, 50.0, 55.0], index=idx)

    res = run_naive_backtest(z, raw, proc, entry_threshold=1.5, tax=0.02,
                             half_spread_raw=0.01, half_spread_proc=0.02)
    expected = [0.99, 1.089, 1.0165910832, 1.11825019152]
    for got, exp in zip(res["equity_curve"].values, expected):
        assert abs(got - exp) < 1e-9, f"got {got}, expected {exp}"
    print("test_hand_traced_with_spread_costs: PASS (all 4 points match exactly)")


if __name__ == "__main__":
    test_hand_traced_rotation()
    test_no_threshold_crossed_stays_flat()
    test_tax_applied_only_on_exit_not_entry()
    test_max_drawdown_sign_and_magnitude()
    test_default_half_spread_matches_original_behavior()
    test_hand_traced_with_spread_costs()
    print("\nAll naive backtest tests passed.")