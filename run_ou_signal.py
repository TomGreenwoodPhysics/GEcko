"""
Walk-forward OU fit + z-score signal.

Pairs: hide (beta stable 0.80-1.07 in the rolling screen) and rune (beta
unstable -0.00-0.96, hence the walk-forward fit rather than a fixed beta).
iron is excluded: only 52% of rolling windows showed cointegration, with beta
ranging 0.33-0.95 -- no stable relationship to trade. See
gecko/stats/cointegration.py and the rolling diagnostic for the evidence.

Run from the project root:   python run_ou_signal.py
Outputs:
  data/clean/<label>_zscore.csv
  figures/zscore_<label>.png   -- z-score with +/-2 entry bands marked
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gecko.stats.ou import rolling_zscore

CLEAN_DIR = Path("data/clean")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

PAIRS = ["hide", "rune"]   # iron excluded -- see module docstring
FIT_WINDOW_DAYS = 730
REFIT_STEP_DAYS = 30


def main():
    for label in PAIRS:
        path = CLEAN_DIR / f"{label}_daily_clean.csv"
        if not path.exists():
            print(f"  ! {path} missing, skipping")
            continue
        panel = pd.read_csv(path, index_col=0, parse_dates=True)
        df = panel[["raw_price", "proc_price"]].dropna()
        df = df[(df["raw_price"] > 0) & (df["proc_price"] > 0)]
        log_raw, log_proc = np.log(df["raw_price"]), np.log(df["proc_price"])

        print(f"\n=== {label} ===")
        roll = rolling_zscore(log_raw, log_proc,
                              fit_window_days=FIT_WINDOW_DAYS,
                              refit_step_days=REFIT_STEP_DAYS)
        roll.to_csv(CLEAN_DIR / f"{label}_zscore.csv")

        print(f"  {len(roll)} scored days, {roll['beta'].nunique()} refits")
        print(f"  beta range across refits: {roll['beta'].min():.2f} to "
              f"{roll['beta'].max():.2f}")
        print(f"  half-life range: {roll['half_life'].min():.1f} to "
              f"{roll['half_life'].max():.1f} days")
        pct_extreme = (roll["z"].abs() > 2).mean()
        print(f"  |z|>2 (a plausible entry threshold): {pct_extreme:.1%} of days")

        plot_zscore(label, roll)

    print(f"\nSaved per-pair CSVs in data/clean/ and figures in figures/")


def plot_zscore(label: str, roll: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1]})
    ax1.plot(roll.index, roll["spread"], lw=0.7, color="tab:blue")
    ax1.plot(roll.index, roll["mu"], lw=0.8, color="k", ls="--",
             label="trailing-fit mu (steps as it refits)")
    ax1.set_title(f"{label} - spread vs walk-forward fitted mean", fontsize=10)
    ax1.legend(fontsize=8, loc="upper right")

    ax2.plot(roll.index, roll["z"], lw=0.7, color="tab:purple")
    ax2.axhline(2, color="r", ls="--", lw=0.8)
    ax2.axhline(-2, color="r", ls="--", lw=0.8)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_title(f"{label} - z-score (red lines = naive +/-2 entry threshold)",
                  fontsize=10)

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"zscore_{label}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()