"""
Tests for gecko.data.clean — small, hand-built cases where the right answer
is known by construction, so a wrong gap policy shows up as a failing
assertion, not a confusing number three modules downstream.

Run with:  pytest tests/test_clean.py -v
(or just:  python tests/test_clean.py   -- runs the same checks without pytest)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))  # so `import gecko` works
from gecko.data.clean import fill_grid, align_pair, clean_realtime_pair, clean_daily_pair


def test_fill_grid_fills_short_gap():
    # 5 daily points with a single missing day (day 2) -- a 1-day gap,
    # well within max_gap=2, so it should be forward-filled, not dropped.
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"])
    df = pd.DataFrame({"price": [10, 11, 13, 14]}, index=idx)

    filled, report = fill_grid(df, freq="1D", max_gap=2, required_cols=["price"])

    assert report["n_grid_points"] == 5  # 01-01 .. 01-05 inclusive
    assert report["n_missing_before_fill"] == 1  # 01-03 was missing
    assert report["n_missing_after_fill"] == 0  # filled, within cap
    assert filled.loc["2024-01-03", "price"] == 11  # forward-filled from 01-02
    print("test_fill_grid_fills_short_gap: PASS")


def test_fill_grid_respects_cap():
    # Same shape, but max_gap=0 -- no filling allowed, gap must remain NaN.
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"])
    df = pd.DataFrame({"price": [10, 11, 13, 14]}, index=idx)

    filled, report = fill_grid(df, freq="1D", max_gap=0, required_cols=["price"])

    assert report["n_missing_after_fill"] == 1
    assert pd.isna(filled.loc["2024-01-03", "price"])
    print("test_fill_grid_respects_cap: PASS")


def test_fill_grid_never_uses_future_values():
    # The gap sits between 5 and 50 -- if filling looked forward it would
    # produce something near the average (~27); causal ffill must produce
    # exactly 5 (the prior value), never anything informed by 50.
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"])
    df = pd.DataFrame({"price": [5, 5, 50]}, index=idx)

    filled, _ = fill_grid(df, freq="1D", max_gap=2, required_cols=["price"])

    assert filled.loc["2024-01-03", "price"] == 5
    print("test_fill_grid_never_uses_future_values: PASS")


def test_volume_does_not_gate_completeness():
    # Volume is NaN but price is present -- this row must NOT count as
    # missing, because only 'price' is in required_cols.
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    df = pd.DataFrame({"price": [10, 11, 12], "volume": [100, np.nan, 100]}, index=idx)

    _, report = fill_grid(df, freq="1D", max_gap=0, required_cols=["price"])

    assert report["n_missing_before_fill"] == 0
    assert report["n_missing_after_fill"] == 0
    print("test_volume_does_not_gate_completeness: PASS")


def test_align_pair_drops_unfillable_rows():
    # raw has a gap on day 3 that exceeds its cap; proc is complete.
    # The joined panel must drop day 3 and report exactly 1 drop.
    idx_raw = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"])
    idx_proc = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    raw = pd.DataFrame({"price": [1, 1, 1, 1]}, index=idx_raw)
    proc = pd.DataFrame({"price": [2, 2, 2, 2, 2]}, index=idx_proc)

    raw_f, _ = fill_grid(raw, freq="1D", max_gap=0, required_cols=["price"])
    proc_f, _ = fill_grid(proc, freq="1D", max_gap=0, required_cols=["price"])
    panel, report = align_pair(raw_f, proc_f, required_raw=["price"], required_proc=["price"])

    assert report["n_dropped_incomplete"] == 1
    assert "2024-01-03" not in panel.index.astype(str).tolist()
    assert len(panel) == 4
    print("test_align_pair_drops_unfillable_rows: PASS")


def test_clean_realtime_pair_end_to_end():
    # Sanity check the convenience wrapper runs and returns a sane shape.
    ts = pd.date_range("2026-01-01", periods=20, freq="6h")
    raw = pd.DataFrame({
        "avgHighPrice": np.linspace(100, 110, 20),
        "avgLowPrice": np.linspace(95, 105, 20),
        "highPriceVolume": np.full(20, 500),
        "lowPriceVolume": np.full(20, 500),
    }, index=ts)
    proc = raw.copy() * 2  # trivially "cointegrated" by construction

    panel, report = clean_realtime_pair(raw, proc, max_gap_buckets=2)

    assert len(panel) == 20
    assert "raw_avgHighPrice" in panel.columns
    assert "proc_avgLowPrice" in panel.columns
    assert report["join"]["n_dropped_incomplete"] == 0
    print("test_clean_realtime_pair_end_to_end: PASS")


def test_clean_daily_pair_end_to_end():
    days = pd.date_range("2020-01-01", periods=100, freq="D")
    raw = pd.DataFrame({"price": np.linspace(50, 60, 100),
                        "volume": np.full(100, 1000)}, index=days)
    proc = pd.DataFrame({"price": np.linspace(100, 120, 100),
                         "volume": np.full(100, 1000)}, index=days)

    panel, report = clean_daily_pair(raw, proc, max_gap_days=2)

    assert len(panel) == 100
    assert report["join"]["n_dropped_incomplete"] == 0
    print("test_clean_daily_pair_end_to_end: PASS")


if __name__ == "__main__":
    test_fill_grid_fills_short_gap()
    test_fill_grid_respects_cap()
    test_fill_grid_never_uses_future_values()
    test_volume_does_not_gate_completeness()
    test_align_pair_drops_unfillable_rows()
    test_clean_realtime_pair_end_to_end()
    test_clean_daily_pair_end_to_end()
    print("\nAll tests passed.")