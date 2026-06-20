import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from gecko.backtest.feasibility import (
    safe_capital_ceiling, windows_needed, feasibility_at_capital,
)


def _toy_data():
    """3 entries: 2 into raw, 1 into proc, with known prices and limits.
      raw limit=1000 units; entries at raw price 100 and 200
        -> per-window capacity: 1000*100=100,000 gp and 1000*200=200,000 gp
        -> the BINDING (worst) one is the smaller: 100,000 gp at price=100
      proc limit=500 units; one entry at proc price 50
        -> per-window capacity: 500*50=25,000 gp
      overall ceiling = min(100_000, 25_000) = 25_000 gp
    """
    trades = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10"]),
        "action": ["enter_raw", "enter_raw", "enter_proc"],
    })
    idx = pd.date_range("2024-01-01", periods=10)
    prices = pd.DataFrame({
        "raw_price": [100.0] * 10,
        "proc_price": [50.0] * 10,
    }, index=idx)
    # day 5 (index position 4) has raw price 200 instead
    prices.loc["2024-01-05", "raw_price"] = 200.0
    limits = {"raw": 1000, "proc": 500}
    return trades, prices, limits


def test_safe_capital_ceiling_hand_traced():
    trades, prices, limits = _toy_data()
    rep = safe_capital_ceiling(trades, prices, limits)
    assert abs(rep["raw_ceiling_gp"] - 100_000) < 1e-6
    assert abs(rep["proc_ceiling_gp"] - 25_000) < 1e-6
    assert abs(rep["overall_ceiling_gp"] - 25_000) < 1e-6
    assert rep["raw_worst_date"] == "2024-01-01"  # the 100gp-price entry binds, not the 200gp one
    assert rep["n_entries"] == 3
    print(f"test_safe_capital_ceiling_hand_traced: PASS {rep}")


def test_windows_needed_hand_traced():
    # 250,000 gp into an item priced 100 with buy limit 1000 units:
    # capacity per window = 100,000 gp -> need ceil(250000/100000) = 3 windows
    assert windows_needed(capital_gp=250_000, price=100, buy_limit_units=1000) == 3
    # exactly at the boundary: 100,000 gp needs exactly 1 window
    assert windows_needed(capital_gp=100_000, price=100, buy_limit_units=1000) == 1
    # just over the boundary: needs 2
    assert windows_needed(capital_gp=100_001, price=100, buy_limit_units=1000) == 2
    print("test_windows_needed_hand_traced: PASS")


def test_feasibility_at_capital_below_ceiling_is_clean():
    trades, prices, limits = _toy_data()
    rep = feasibility_at_capital(trades, prices, limits, capital_gp=20_000)
    assert rep["pct_needing_multiple_windows"] == 0.0
    assert rep["max_windows_needed"] == 1
    print("test_feasibility_at_capital_below_ceiling_is_clean: PASS")


def test_feasibility_at_capital_above_ceiling_shows_friction():
    trades, prices, limits = _toy_data()
    # 200,000 gp: raw entries (100k, 200k capacity) -- one needs 2 windows,
    # one fits in 1; proc entry (25k capacity) needs ceil(200000/25000)=8
    rep = feasibility_at_capital(trades, prices, limits, capital_gp=200_000)
    assert rep["max_windows_needed"] == 8
    assert rep["pct_needing_multiple_windows"] > 0
    print(f"test_feasibility_at_capital_above_ceiling_shows_friction: PASS {rep}")


def test_safe_capital_ceiling_empty_trades():
    empty_trades = pd.DataFrame({"date": [], "action": []})
    _, prices, limits = _toy_data()
    rep = safe_capital_ceiling(empty_trades, prices, limits)
    assert rep["n_entries"] == 0
    assert np.isnan(rep["overall_ceiling_gp"])
    print("test_safe_capital_ceiling_empty_trades: PASS (no crash on empty input)")


if __name__ == "__main__":
    test_safe_capital_ceiling_hand_traced()
    test_windows_needed_hand_traced()
    test_feasibility_at_capital_below_ceiling_is_clean()
    test_feasibility_at_capital_above_ceiling_shows_friction()
    test_safe_capital_ceiling_empty_trades()
    print("\nAll feasibility tests passed.")