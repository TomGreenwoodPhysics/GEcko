import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from gecko.backtest.risk_metrics import downside_deviation, sortino_ratio, calmar_ratio


def test_downside_deviation_hand_traced():
    # returns: [0.02, -0.01, 0.03, -0.02, 0.0]  target=0
    # shortfalls (only negative, else 0): [0, -0.01, 0, -0.02, 0]
    # squared: [0, 0.0001, 0, 0.0004, 0] -> mean = 0.0001 -> sqrt = 0.01
    r = pd.Series([0.02, -0.01, 0.03, -0.02, 0.0])
    dd = downside_deviation(r, target=0.0)
    assert abs(dd - 0.01) < 1e-12
    print(f"test_downside_deviation_hand_traced: PASS (dd={dd})")


def test_downside_deviation_ignores_pure_upside():
    r = pd.Series([0.01, 0.02, 0.03, 0.04])  # all positive, no downside at all
    dd = downside_deviation(r, target=0.0)
    assert dd == 0.0
    print("test_downside_deviation_ignores_pure_upside: PASS")


def test_sortino_exceeds_sharpe_for_positive_skew():
    # Frequent small losses, rare large gains (positive skew): total vol is
    # inflated by the upside jumps, but downside vol only sees the small
    # losses -- Sortino should come out higher than the naive Sharpe.
    rng = np.random.default_rng(0)
    n = 3000
    r = rng.normal(loc=-0.001, scale=0.01, size=n)
    jump_idx = rng.choice(n, size=30, replace=False)
    r[jump_idx] += 0.15  # rare big positive jumps
    r = pd.Series(r)

    sharpe_naive = (r.mean() / r.std()) * np.sqrt(365)
    ann_return = (1 + r.mean()) ** 365 - 1  # rough, just for this comparison
    sortino = sortino_ratio(r, ann_return)

    assert sortino > sharpe_naive
    print(f"test_sortino_exceeds_sharpe_for_positive_skew: PASS "
          f"(sortino={sortino:.2f} > sharpe={sharpe_naive:.2f})")


def test_calmar_hand_traced():
    ann_return, max_dd = 0.30, -0.15
    calmar = calmar_ratio(ann_return, max_dd)
    assert abs(calmar - 2.0) < 1e-12
    print(f"test_calmar_hand_traced: PASS (calmar={calmar})")


def test_calmar_nan_for_zero_drawdown():
    calmar = calmar_ratio(ann_return=0.10, max_drawdown=0.0)
    assert np.isnan(calmar)
    print("test_calmar_nan_for_zero_drawdown: PASS")


def test_sortino_nan_for_zero_downside_deviation():
    r = pd.Series([0.01, 0.02, 0.03])  # no downside at all
    sortino = sortino_ratio(r, ann_return=0.50)
    assert np.isnan(sortino)
    print("test_sortino_nan_for_zero_downside_deviation: PASS")


if __name__ == "__main__":
    test_downside_deviation_hand_traced()
    test_downside_deviation_ignores_pure_upside()
    test_sortino_exceeds_sharpe_for_positive_skew()
    test_calmar_hand_traced()
    test_calmar_nan_for_zero_drawdown()
    test_sortino_nan_for_zero_downside_deviation()
    print("\nAll risk metrics tests passed.")