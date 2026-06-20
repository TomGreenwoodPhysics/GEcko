"""
Run the cointegration battery on the shortlisted pairs.

Loads the clean DAILY panels (the long history; daily is the right series for
cointegration inference -- the 3-month realtime series is far too short), runs
the full battery per pair, prints a comparison table, and saves each pair's
spread series for the OU fitting / z-score step.

Run from the project root:   python run_cointegration.py
Outputs:
  data/clean/<label>_spread.csv          - the spread series per pair
  data/clean/cointegration_summary.csv   - one row per pair, the headline table
  figures/spreads.png                    - spread of each pair, eyeball check
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gecko.stats.cointegration import run_pair_cointegration

CLEAN_DIR = Path("data/clean")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

PAIRS = ["hide", "iron", "rune"]


def main() -> pd.DataFrame:
    rows = []
    spreads = {}

    for label in PAIRS:
        path = CLEAN_DIR / f"{label}_daily_clean.csv"
        if not path.exists():
            print(f"  ! {path} missing -- run build_clean_panels.py first; skipping")
            continue
        panel = pd.read_csv(path, index_col=0, parse_dates=True)
        print(f"\n=== {label} ===")
        rep = run_pair_cointegration(panel)

        eg_pr = rep["engle_granger"]["proc_on_raw"]
        eg_rp = rep["engle_granger"]["raw_on_proc"]
        joh = rep["johansen"]

        spreads[label] = rep["spread"]
        rep["spread"].to_csv(CLEAN_DIR / f"{label}_spread.csv")

        print(f"  n={rep['n_obs']}  {rep['date_start']} -> {rep['date_end']}")
        print(f"  integration: raw {rep['integration_raw']['verdict']}, "
              f"proc {rep['integration_proc']['verdict']}")
        print(f"  Engle-Granger proc~raw:  p={eg_pr['pvalue']:.4f}  "
              f"beta={eg_pr['beta']:.3f}  "
              f"{'COINTEGRATED' if eg_pr['cointegrated_at_5pct'] else 'not cointegrated'}")
        print(f"  Engle-Granger raw~proc:  p={eg_rp['pvalue']:.4f}  "
              f"beta={eg_rp['beta']:.3f}")
        print(f"  Johansen trace: stat={joh['trace_stat_r0']:.2f} vs "
              f"crit95={joh['trace_crit_95_r0']:.2f}  "
              f"{'rejects r=0 (COINTEGRATED)' if joh['trace_rejects_r0_at_5pct'] else 'cannot reject r=0'}"
              f"  beta_joh={joh['beta_johansen']:.3f}")
        print(f"  spread half-life: {rep['spread_half_life_days']:.0f} days")

        rows.append({
            "label": label,
            "n_obs": rep["n_obs"],
            "raw_order": rep["integration_raw"]["verdict"],
            "proc_order": rep["integration_proc"]["verdict"],
            "eg_pr_pvalue": eg_pr["pvalue"],
            "eg_pr_beta": eg_pr["beta"],
            "eg_pr_coint": eg_pr["cointegrated_at_5pct"],
            "eg_rp_pvalue": eg_rp["pvalue"],
            "johansen_trace_reject": joh["trace_rejects_r0_at_5pct"],
            "johansen_beta": joh["beta_johansen"],
            "half_life_days": rep["spread_half_life_days"],
        })

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary.to_csv(CLEAN_DIR / "cointegration_summary.csv", index=False)
        plot_spreads(spreads)
        print("\n" + "=" * 78)
        print("COINTEGRATION SUMMARY")
        print("=" * 78)
        print(summary.to_string(index=False))
        print(f"\nSaved {CLEAN_DIR / 'cointegration_summary.csv'} and "
              f"{FIG_DIR / 'spreads.png'}")
    return summary


def plot_spreads(spreads: dict[str, pd.Series]):
    n = len(spreads)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.8 * n), squeeze=False)
    for ax, (label, sp) in zip(axes.flat, spreads.items()):
        ax.plot(sp.index, sp.values, lw=0.7)
        ax.axhline(sp.mean(), color="k", ls="--", lw=0.8)
        ax.set_title(f"{label} - log-spread (mean dashed). "
                     "Flat band around mean = tradeable; drift/break = caution",
                     fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "spreads.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()