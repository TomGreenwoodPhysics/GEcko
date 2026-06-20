"""
gecko.data.clean — turn raw per-leg price dumps into one clean, aligned panel.

Design rules (these matter for the project's lookahead-bias story later,
so they're enforced here, not bolted on in Week 9):

  1. Gap-filling is CAUSAL ONLY. We forward-fill (use the last known value),
     never interpolate between a past and a future point. A backtest that
     accidentally used an interpolated value would be peeking at the future.

  2. Forward-fill is CAPPED. A short gap (one or two missed buckets) is
     plausibly just a quiet market and forward-filling is a reasonable
     approximation. A long gap means we don't actually know the price, and
     pretending otherwise (by filling it anyway) would manufacture data.
     Past the cap, the row stays incomplete and is dropped at the join step,
     with the drop count reported, not silently swallowed.

  3. "Completeness" is judged on the columns you actually need (the price
     columns), not on every column. Volume is occasionally missing in the
     source data for reasons unrelated to price (see data_pull.py); that
     shouldn't force a perfectly good price observation to be dropped.

Everything here is a pure function: dataframe in, (dataframe, report) out.
No file I/O, no network — easy to unit test (see tests/test_clean.py).
"""

from __future__ import annotations

import pandas as pd


# ----------------------------------------------------------------------------
# Single-leg: regularize onto an exact-frequency grid, capped causal fill
# ----------------------------------------------------------------------------

def fill_grid(df: pd.DataFrame, freq: str, max_gap: int,
              required_cols: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Reindex df (already timestamp-indexed, deduplicated) onto a regular
    `freq` grid spanning its own min..max, then forward-fill up to `max_gap`
    consecutive missing steps.

    required_cols: columns that must be non-null for a row to count as
    "complete" (default: all columns). Use this to stop optional fields
    like volume from gating completeness of the price columns.

    Returns (filled_df, report). filled_df still has NaNs in required_cols
    wherever a gap exceeded max_gap — those rows are NOT silently dropped
    here; that decision belongs to the caller (see align_pair), so a
    single-leg report is honest about what this leg alone could supply.
    """
    if df.index.has_duplicates:
        raise ValueError("fill_grid expects a deduplicated index")
    if required_cols is None:
        required_cols = list(df.columns)

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    out = df.reindex(full_idx)

    n_missing_before = int(out[required_cols].isna().any(axis=1).sum())
    filled = out.ffill(limit=max_gap) if max_gap > 0 else out.copy()
    n_missing_after = int(filled[required_cols].isna().any(axis=1).sum())

    report = {
        "n_grid_points": len(full_idx),
        "n_missing_before_fill": n_missing_before,
        "n_missing_after_fill": n_missing_after,
        "n_filled": n_missing_before - n_missing_after,
        "max_gap_allowed": max_gap,
    }
    return filled, report


# ----------------------------------------------------------------------------
# Two legs -> one aligned panel
# ----------------------------------------------------------------------------

def align_pair(raw_filled: pd.DataFrame, proc_filled: pd.DataFrame,
               required_raw: list[str], required_proc: list[str],
               raw_prefix: str = "raw_", proc_prefix: str = "proc_"
               ) -> tuple[pd.DataFrame, dict]:
    """Join two already-grid-filled legs on their common timestamps, then
    drop any row where a required column on either leg is still NaN (i.e.
    the gap on that leg exceeded its fill cap). Every drop is counted.
    """
    raw_r = raw_filled.add_prefix(raw_prefix)
    proc_r = proc_filled.add_prefix(proc_prefix)
    req_cols = [f"{raw_prefix}{c}" for c in required_raw] + \
               [f"{proc_prefix}{c}" for c in required_proc]

    joined = raw_r.join(proc_r, how="inner")
    n_after_join = len(joined)
    complete = joined.dropna(subset=req_cols)
    n_complete = len(complete)

    report = {
        "n_after_join": n_after_join,
        "n_complete_rows": n_complete,
        "n_dropped_incomplete": n_after_join - n_complete,
    }
    return complete, report


# ----------------------------------------------------------------------------
# Pair-level convenience wrappers for the two data sources used in GEcko
# ----------------------------------------------------------------------------

def clean_realtime_pair(raw_df: pd.DataFrame, proc_df: pd.DataFrame,
                         max_gap_buckets: int = 2) -> tuple[pd.DataFrame, dict]:
    """raw_df/proc_df: timestamp-indexed, columns include avgHighPrice,
    avgLowPrice (required) and *Volume columns (optional, not gating).
    6-hourly cadence assumed (matches data_pull.get_timeseries timestep).
    """
    price_cols = ["avgHighPrice", "avgLowPrice"]
    raw_f, raw_report = fill_grid(raw_df, freq="6h", max_gap=max_gap_buckets,
                                   required_cols=price_cols)
    proc_f, proc_report = fill_grid(proc_df, freq="6h", max_gap=max_gap_buckets,
                                     required_cols=price_cols)
    panel, join_report = align_pair(raw_f, proc_f,
                                     required_raw=price_cols,
                                     required_proc=price_cols)
    report = {"raw_leg": raw_report, "proc_leg": proc_report, "join": join_report}
    return panel, report


def clean_daily_pair(raw_df: pd.DataFrame, proc_df: pd.DataFrame,
                      max_gap_days: int = 2) -> tuple[pd.DataFrame, dict]:
    """raw_df/proc_df: timestamp-indexed (daily), columns include 'price'
    (required) and 'volume' (optional, not gating).
    """
    price_cols = ["price"]
    raw_f, raw_report = fill_grid(raw_df, freq="1D", max_gap=max_gap_days,
                                   required_cols=price_cols)
    proc_f, proc_report = fill_grid(proc_df, freq="1D", max_gap=max_gap_days,
                                     required_cols=price_cols)
    panel, join_report = align_pair(raw_f, proc_f,
                                     required_raw=price_cols,
                                     required_proc=price_cols)
    report = {"raw_leg": raw_report, "proc_leg": proc_report, "join": join_report}
    return panel, report