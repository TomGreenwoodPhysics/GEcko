"""
Rolling cointegration diagnostic.

The full-sample Engle-Granger/Johansen tests answer "is this pair cointegrated
on average over 11 years". That single verdict can hide a lot: iron's
multi-year cycling, rune's recent regime break. This runs Engle-Granger on a
sliding 2-year window (step 1 month) so you can see WHEN each relationship
held, not just whether it did overall.

Run from the project root:   python run_rolling_diagnostics.py
Outputs:
  data/clean/<label>_rolling_eg.csv
  figures/rolling_cointegration.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gecko.stats.cointegration import rolling_engle_granger

CLEAN_DIR = Path("data/clean")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

PAIRS = ["hide", "iron", "rune"]
WINDOW_DAYS = 730   # 2 years
STEP_DAYS = 30      # monthly


def main():
    results = {}
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
        roll = rolling_engle_granger(log_raw, log_proc,
                                     window_days=WINDOW_DAYS, step_days=STEP_DAYS)
        roll.to_csv(CLEAN_DIR / f"{label}_rolling_eg.csv")
        pct_coint = roll["cointegrated_at_5pct"].mean()
        print(f"  {len(roll)} windows, cointegrated in {pct_coint:.0%} of them")
        print(f"  beta range across windows: {roll['beta'].min():.2f} to "
              f"{roll['beta'].max():.2f}")
        results[label] = roll

    plot_rolling(results)
    print(f"\nSaved figures/rolling_cointegration.png and per-pair CSVs in data/clean/")


def plot_rolling(results: dict[str, pd.DataFrame]):
    n = len(results)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.0 * n), squeeze=False)
    for ax, (label, roll) in zip(axes.flat, results.items()):
        ax.plot(roll.index, roll["pvalue"], color="tab:blue", lw=1.0)
        ax.axhline(0.05, color="r", ls="--", lw=0.8, label="p=0.05")
        ax.set_ylabel("EG p-value", color="tab:blue")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{label} - rolling {WINDOW_DAYS//365}y Engle-Granger "
                     "p-value (below red line = cointegrated in that window)",
                     fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rolling_cointegration.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()