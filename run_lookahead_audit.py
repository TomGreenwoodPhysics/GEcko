"""
Lookahead-bias audit against saved pipeline output.

Re-runs the structural and recomputation checks from gecko.backtest.audit
against the REAL saved zscore/panel files, not synthetic test data.

IMPORTANT: rerun run_ou_signal.py first if your zscore CSVs predate this
change -- they won't have the fit_end_date column the audit needs.

Run from the project root:   python run_lookahead_audit.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gecko.backtest.audit import check_no_future_leakage, recompute_and_compare

CLEAN_DIR = Path("data/clean")

# label -> fit_window_days used when that pair's zscore.csv was generated
# (must match run_ou_signal.py's FIT_WINDOW_DAYS for the recompute check to
# be comparing apples to apples).
PAIRS = {"hide": 730, "rune": 730}


def main():
    for label, fit_window_days in PAIRS.items():
        z_path = CLEAN_DIR / f"{label}_zscore.csv"
        panel_path = CLEAN_DIR / f"{label}_daily_clean.csv"
        if not z_path.exists() or not panel_path.exists():
            print(f"  ! missing inputs for {label}, skipping")
            continue

        roll = pd.read_csv(z_path, index_col=0, parse_dates=True)
        if "fit_end_date" not in roll.columns:
            print(f"  ! {label}_zscore.csv has no fit_end_date column -- "
                  f"rerun run_ou_signal.py to regenerate it, then retry.")
            continue
        panel = pd.read_csv(panel_path, index_col=0, parse_dates=True)

        print(f"\n=== {label} ===")
        structural = check_no_future_leakage(roll)
        print(f"  structural check: {structural['n_violations']} violations "
              f"out of {structural['n_rows']} rows -- "
              f"{'PASS' if structural['passed'] else 'FAIL'}")
        if not structural["passed"]:
            print(f"    first violation at {structural['first_violation_date']}")

        recompute = recompute_and_compare(panel, roll, fit_window_days=fit_window_days,
                                          n_samples=8, seed=0)
        print(f"  recomputation check: {recompute['n_matched']}/"
              f"{recompute['n_sampled']} sampled refits independently matched -- "
              f"{'PASS' if recompute['passed'] else 'FAIL'}")
        if not recompute["passed"]:
            for d in recompute["details"]:
                if not d["match"]:
                    print(f"    mismatch at {d['fit_end_date']}: "
                          f"recomputed beta={d['recomputed_beta']:.4f} vs "
                          f"recorded={d['recorded_beta']:.4f}")


if __name__ == "__main__":
    main()