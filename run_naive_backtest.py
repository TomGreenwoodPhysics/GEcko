"""
Run the rotation backtest on hide and rune.

rune's pre-2019 z-score is excluded -- that window is an early-market
liquidity artifact (extreme, erratic z-scores in the rolling diagnostic and
the OU walk-forward plot), not tradeable signal. Including it would let
performance be driven by noise rather than the relationship the rest of the
series shows.

Run from the project root:   python run_naive_backtest.py
Outputs:
  figures/naive_backtest_<label>.png
  data/clean/naive_backtest_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gecko.backtest.naive import run_naive_backtest, estimate_half_spread
from gecko.backtest.deflated_sharpe import deflated_sharpe_ratio

CLEAN_DIR = Path("data/clean")
RAW_DIR = Path("data/raw")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# Number of candidate pairs screened -- the trial count
# the deflated Sharpe correction needs. See deflated_sharpe.py docstring for
# why this is the defensible (and conservative-direction) choice.
N_TRIALS_SCREENED = 8

# label -> (start_date or None, realtime cache stem or None).
# rune is excluded: it lost money even on the tax-only model (Sharpe -0.53),
# so a stricter cost model can only do worse.
PAIRS = {
    "hide": (None, "hide"),
}

ENTRY_THRESHOLD = 1.5
TAX = 0.02


def main():
    rows = []
    for label, (start, cache_stem) in PAIRS.items():
        z_path = CLEAN_DIR / f"{label}_zscore.csv"
        panel_path = CLEAN_DIR / f"{label}_daily_clean.csv"
        if not z_path.exists() or not panel_path.exists():
            print(f"  ! missing inputs for {label}, skipping")
            continue

        zdf = pd.read_csv(z_path, index_col=0, parse_dates=True)
        panel = pd.read_csv(panel_path, index_col=0, parse_dates=True)
        merged = zdf[["z"]].join(panel[["raw_price", "proc_price"]], how="inner")
        if start:
            merged = merged.loc[start:]

        print(f"\n=== {label} ===  ({len(merged)} days"
              f"{f', from {start}' if start else ''})")

        # --- tax-only, for direct comparison ---
        res_tax_only = run_naive_backtest(merged["z"], merged["raw_price"],
                                          merged["proc_price"],
                                          entry_threshold=ENTRY_THRESHOLD, tax=TAX)
        s0 = res_tax_only["summary"]
        print(f"  [tax only]      total return: {s0['total_return']:+.1%}   "
              f"ann. return: {s0['ann_return']:+.1%}   "
              f"Sharpe: {s0['sharpe_naive']:.2f}   "
              f"max DD: {s0['max_drawdown']:.1%}")
        rows.append({"label": label, "model": "tax_only", **s0})

        # --- tax + realistic bid-ask spread, calibrated from real data ---
        hs_raw = estimate_half_spread(RAW_DIR / f"{cache_stem}_raw_realtime_6h.csv")
        hs_proc = estimate_half_spread(RAW_DIR / f"{cache_stem}_proc_realtime_6h.csv")
        if hs_raw is None or hs_proc is None:
            print(f"  ! no cached realtime data for {label} -- run "
                  f"build_clean_panels.py first to enable the realistic-cost "
                  f"comparison. Skipping.")
            continue

        print(f"  measured half-spread: raw={hs_raw:.2%}  proc={hs_proc:.2%} "
              f"(from realtime bid/ask, last ~3 months)")
        res_realistic = run_naive_backtest(merged["z"], merged["raw_price"],
                                           merged["proc_price"],
                                           entry_threshold=ENTRY_THRESHOLD, tax=TAX,
                                           half_spread_raw=hs_raw,
                                           half_spread_proc=hs_proc)
        s1 = res_realistic["summary"]
        print(f"  [+ real spread] total return: {s1['total_return']:+.1%}   "
              f"ann. return: {s1['ann_return']:+.1%}   "
              f"Sharpe: {s1['sharpe_naive']:.2f}   "
              f"Sortino: {s1['sortino_naive']:.2f}   "
              f"Calmar: {s1['calmar']:.2f}   "
              f"max DD: {s1['max_drawdown']:.1%}")
        rows.append({"label": label, "model": "tax_plus_spread", **s1})

        # --- deflated Sharpe: corrects for non-normal returns AND the fact
        # that hide was the best-looking result out of 8 screened pairs ---
        daily_rets = res_realistic["equity_curve"].pct_change().dropna()
        dsr_rep = deflated_sharpe_ratio(daily_rets, n_trials=N_TRIALS_SCREENED)
        print(f"  [deflated]      DSR: {dsr_rep['dsr']:.1%}   "
              f"(naive Sharpe {s1['sharpe_naive']:.2f} vs expected-max-by-chance "
              f"benchmark from {N_TRIALS_SCREENED} trials)")
        print(f"  {dsr_rep['interpretation']}")
        rows[-1].update({f"dsr_{k}": v for k, v in dsr_rep.items()
                         if k != "interpretation"})

        plot_comparison(label, res_tax_only, res_realistic)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary.to_csv(CLEAN_DIR / "naive_backtest_summary.csv", index=False)
        print("\n" + "=" * 78)
        print("NAIVE BACKTEST SUMMARY -- tax-only vs realistic-spread comparison")
        print("=" * 78)
        print(summary.to_string(index=False))


def plot_comparison(label: str, res_tax_only: dict, res_realistic: dict):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(res_tax_only["equity_curve"].index, res_tax_only["equity_curve"].values,
            lw=1.0, color="tab:orange", label="tax only")
    ax.plot(res_realistic["equity_curve"].index, res_realistic["equity_curve"].values,
            lw=1.0, color="tab:green", label="tax + measured realistic spread")
    ax.set_yscale("log")
    ax.set_title(f"{label} - how much of the tax-only return survives a "
                 "realistic spread cost? (log scale)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"naive_backtest_{label}_cost_comparison.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()