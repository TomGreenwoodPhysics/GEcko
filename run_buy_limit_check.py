"""
GE buy-limit feasibility check.

Uses hide's trade log from the realistic-cost backtest and the GE buy limits
from mapping.csv to find the capital scale at which the backtest's instant,
full-execution assumption stops being realistic.

Run AFTER run_naive_backtest.py (needs its trade log).
Run from the project root:   python run_buy_limit_check.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gecko.backtest.naive import run_naive_backtest, estimate_half_spread
from gecko.backtest.feasibility import safe_capital_ceiling, feasibility_at_capital

CLEAN_DIR = Path("data/clean")
RAW_DIR = Path("data/raw")

LABEL = "hide"
CACHE_STEM = "hide"
ITEM_NAMES = {"raw": "Cowhide", "proc": "Leather"}
REFERENCE_CAPITALS_GP = [1_000_000, 10_000_000, 50_000_000, 100_000_000]


def load_buy_limits() -> dict[str, float]:
    mapping = pd.read_csv(RAW_DIR / "mapping.csv")
    limits = {}
    for leg, name in ITEM_NAMES.items():
        row = mapping[mapping["name"].str.lower() == name.lower()]
        if row.empty:
            print(f"  ! could not find '{name}' in mapping.csv")
            continue
        limits[leg] = float(row["limit"].iloc[0])
    return limits


def main():
    z_path = CLEAN_DIR / f"{LABEL}_zscore.csv"
    panel_path = CLEAN_DIR / f"{LABEL}_daily_clean.csv"
    if not z_path.exists() or not panel_path.exists():
        print(f"  ! missing inputs for {LABEL}, run the pipeline up to "
              f"run_ou_signal.py first")
        return

    zdf = pd.read_csv(z_path, index_col=0, parse_dates=True)
    panel = pd.read_csv(panel_path, index_col=0, parse_dates=True)
    merged = zdf[["z"]].join(panel[["raw_price", "proc_price"]], how="inner")

    hs_raw = estimate_half_spread(RAW_DIR / f"{CACHE_STEM}_raw_realtime_6h.csv")
    hs_proc = estimate_half_spread(RAW_DIR / f"{CACHE_STEM}_proc_realtime_6h.csv")
    res = run_naive_backtest(merged["z"], merged["raw_price"], merged["proc_price"],
                             entry_threshold=1.5, tax=0.02,
                             half_spread_raw=hs_raw or 0.0, half_spread_proc=hs_proc or 0.0)

    limits = load_buy_limits()
    print(f"\n=== {LABEL} buy-limit feasibility ===")
    print(f"  GE 4-hour buy limits: {ITEM_NAMES['raw']}={limits.get('raw')}, "
          f"{ITEM_NAMES['proc']}={limits.get('proc')}")

    ceiling = safe_capital_ceiling(res["trades"], panel, limits)
    print(f"\n  Safe capital ceiling (every historical entry fits in 1 window): "
          f"{ceiling['overall_ceiling_gp']:,.0f} gp")
    print(f"    raw leg binding date: {ceiling['raw_worst_date']}  "
          f"({ceiling['raw_ceiling_gp']:,.0f} gp capacity)")
    print(f"    proc leg binding date: {ceiling['proc_worst_date']}  "
          f"({ceiling['proc_ceiling_gp']:,.0f} gp capacity)")
    print(f"    -> below this capital, the backtest's instant-execution "
          f"assumption held for every real historical entry.")
    print(f"    -> above it, at least one entry would have needed >1 window "
          f"to fill, which this backtest does not model.")

    print(f"\n  At illustrative capital levels:")
    for cap in REFERENCE_CAPITALS_GP:
        rep = feasibility_at_capital(res["trades"], panel, limits, cap)
        print(f"    {cap:>12,.0f} gp: {rep['pct_needing_multiple_windows']:.0%} of "
              f"entries need >1 window, worst case {rep['max_windows_needed']} windows")


if __name__ == "__main__":
    main()