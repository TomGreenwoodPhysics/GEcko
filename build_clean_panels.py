"""
Build clean, aligned panels for the shortlisted pairs.

Shortlist (from the multi-pair screen):
  hide  - Cowhide / Leather       (tightest, fast-reverting, clean cost story)
  iron  - Iron ore / Iron bar     (1:1 ore/bar reference case)
  rune  - Runite ore / Runite bar (tight co-movement with a recent regime
                                    break; carried forward for the full
                                    cointegration treatment)

For each pair, for both data sources (realtime 6h + daily guide price):
  1. Load raw data -- from data/raw/ cache if present, else fetch fresh.
  2. Clean via gecko.data.clean (causal capped fill, then align legs).
  3. Save the clean panel to data/clean/.
  4. Record a data-quality report row (gaps, fills, drops) per pair/source.

Run from the project root:   python build_clean_panels.py
Outputs:
  data/clean/<label>_realtime_clean.csv
  data/clean/<label>_daily_clean.csv
  data/clean/data_quality_report.csv   <- gap/fill/drop counts per pair
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pull import _session, get_timeseries, get_wg_history, USER_AGENT  # noqa: F401
from screen_pairs import load_or_fetch_mapping, resolve_id_safe
from gecko.data.clean import clean_realtime_pair, clean_daily_pair

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

SHORTLIST = {
    "hide": ("Cowhide", "Leather"),
    "iron": ("Iron ore", "Iron bar"),
    "rune": ("Runite ore", "Runite bar"),
}

# Explicit per-leg cache filename stems. iron reuses the stems from the
# initial pull (data_pull.py's DEFAULT_PAIR) to avoid re-fetching what's
# already on disk; hide/rune get distinct stems per leg so the two legs
# can never collide on the same cache file.
CACHE_NAMES = {
    "hide": {"raw": "hide_raw", "proc": "hide_proc"},
    "iron": {"raw": "iron_ore", "proc": "iron_bar"},
    "rune": {"raw": "rune_raw", "proc": "rune_proc"},
}


# ----------------------------------------------------------------------------
# Cached fetch: read from data/raw/ if a prior pull saved it, else hit the API
# ----------------------------------------------------------------------------

def load_or_fetch_realtime(session, item_id: int, cache_name: str) -> pd.DataFrame:
    path = RAW_DIR / f"{cache_name}_realtime_6h.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["timestamp"])
    else:
        df = get_timeseries(session, item_id, timestep="6h")
        df.to_csv(path, index=False)
    return df.set_index("timestamp").sort_index()


def load_or_fetch_daily(session, item_id: int, cache_name: str) -> pd.DataFrame:
    path = RAW_DIR / f"{cache_name}_daily_all.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["timestamp"])
    else:
        df = get_wg_history(session, item_id, span="all")
        df.to_csv(path, index=False)
    df = df.set_index("timestamp").sort_index()
    df.index = df.index.normalize()
    return df[~df.index.duplicated(keep="last")]


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def build_all() -> pd.DataFrame:
    session = _session()
    mapping = load_or_fetch_mapping(session)
    quality_rows = []

    for label, (raw_name, proc_name) in SHORTLIST.items():
        print(f"\n=== {label}: {raw_name} / {proc_name} ===")
        raw_id = resolve_id_safe(mapping, raw_name)
        proc_id = resolve_id_safe(mapping, proc_name)
        if raw_id is None or proc_id is None:
            print(f"  ! skipping {label}, could not resolve item ids")
            continue

        names = CACHE_NAMES[label]

        # --- realtime (6h, tradeable prices) ---
        raw_rt = load_or_fetch_realtime(session, raw_id, names["raw"])
        proc_rt = load_or_fetch_realtime(session, proc_id, names["proc"])
        rt_panel, rt_report = clean_realtime_pair(raw_rt, proc_rt, max_gap_buckets=2)
        rt_panel.to_csv(CLEAN_DIR / f"{label}_realtime_clean.csv")
        print(f"  realtime: {rt_report['join']['n_complete_rows']} rows kept, "
              f"{rt_report['join']['n_dropped_incomplete']} dropped "
              f"(raw filled {rt_report['raw_leg']['n_filled']}, "
              f"proc filled {rt_report['proc_leg']['n_filled']})")
        quality_rows.append({"label": label, "source": "realtime", **_flat(rt_report)})

        # --- daily (long guide-price history) ---
        raw_d = load_or_fetch_daily(session, raw_id, names["raw"])
        proc_d = load_or_fetch_daily(session, proc_id, names["proc"])
        d_panel, d_report = clean_daily_pair(raw_d, proc_d, max_gap_days=2)
        d_panel.to_csv(CLEAN_DIR / f"{label}_daily_clean.csv")
        print(f"  daily:    {d_report['join']['n_complete_rows']} rows kept, "
              f"{d_report['join']['n_dropped_incomplete']} dropped "
              f"(raw filled {d_report['raw_leg']['n_filled']}, "
              f"proc filled {d_report['proc_leg']['n_filled']})")
        quality_rows.append({"label": label, "source": "daily", **_flat(d_report)})

    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(CLEAN_DIR / "data_quality_report.csv", index=False)
    print(f"\nSaved {CLEAN_DIR / 'data_quality_report.csv'}")
    return quality_df


def _flat(report: dict) -> dict:
    """Flatten the nested {raw_leg, proc_leg, join} report into flat columns
    for a tidy one-row-per-pair-per-source CSV."""
    return {
        "raw_filled": report["raw_leg"]["n_filled"],
        "raw_still_missing": report["raw_leg"]["n_missing_after_fill"],
        "proc_filled": report["proc_leg"]["n_filled"],
        "proc_still_missing": report["proc_leg"]["n_missing_after_fill"],
        "rows_kept": report["join"]["n_complete_rows"],
        "rows_dropped": report["join"]["n_dropped_incomplete"],
    }


if __name__ == "__main__":
    summary = build_all()
    print("\n" + "=" * 70)
    print("DATA QUALITY SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))