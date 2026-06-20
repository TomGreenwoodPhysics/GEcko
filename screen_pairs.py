"""
Multi-pair cointegration screen.

Screens several candidate raw->processed pairs on the same diagnostics, to
rank them before committing one to the full pipeline.

For each pair, on the long daily guide-price history:
  - common date range after alignment
  - price ratio (processed/raw): mean, std, coefficient of variation
  - daily log-return correlation
  - spread = log(processed) - log(raw)   [hedge ratio assumed = 1 for this
    quick screen; the real Engle-Granger hedge ratio is fitted later for the
    survivors. Assuming 1 is fine here -- in log space, for a fast comparable
    first pass across many pairs.]
  - ADF test on the spread (requires statsmodels) -> quick cointegration screen
  - AR(1) half-life of the spread, in days, as a tradeability check: a
    "cointegrated" pair that reverts over months is a different prospect from
    one that reverts in two weeks.

A screen, not the final analysis -- it ranks candidates before the full
Engle-Granger/Johansen + OU fitting on the survivors.

Requires: requests, pandas, numpy, matplotlib, statsmodels
  pip install statsmodels

Run from the project root:   python screen_pairs.py
Outputs:
  data/screen/pair_screen_summary.csv   - ranked table, one row per pair
  figures/pair_screen_ratios.png        - small-multiples ratio plot
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data_pull import USER_AGENT, _session, get_mapping, get_wg_history  # noqa: F401

try:
    from statsmodels.tsa.stattools import adfuller
    HAVE_STATSMODELS = True
except ImportError:
    HAVE_STATSMODELS = False
    warnings.warn(
        "statsmodels not installed -- ADF p-values will be NaN. "
        "Run: pip install statsmodels", stacklevel=2)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW_DIR = Path("data/raw")
SCREEN_DIR = Path("data/screen")
FIG_DIR = Path("figures")
SCREEN_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# label -> (raw item name, processed item name, note)
# Names must match the OSRS Wiki /mapping 'name' field exactly (case-insensitive).
CANDIDATE_PAIRS = {
    "iron":    ("Iron ore",       "Iron bar",   "1:1, no secondary input"),
    "silver":  ("Silver ore",     "Silver bar", "1:1, no secondary input"),
    "gold":    ("Gold ore",       "Gold bar",   "1:1, no secondary input"),
    "steel":   ("Coal",           "Steel bar",  "needs iron+coal; screened vs coal only"),
    "mithril": ("Mithril ore",    "Mithril bar","needs ore+4 coal; ore-only screen"),
    "adamant": ("Adamantite ore", "Adamantite bar","needs ore+6 coal; ore-only screen"),
    "rune":    ("Runite ore",     "Runite bar",  "needs ore+8 coal; ore-only screen"),
    "hide":    ("Cowhide",        "Leather",    "tanning fee is ~fixed gp, not free"),
}


# ----------------------------------------------------------------------------
# Mapping (cached)
# ----------------------------------------------------------------------------

def load_or_fetch_mapping(session) -> pd.DataFrame:
    path = RAW_DIR / "mapping.csv"
    if path.exists():
        return pd.read_csv(path)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    mapping = get_mapping(session)
    mapping.to_csv(path, index=False)
    return mapping


def resolve_id_safe(mapping: pd.DataFrame, name: str) -> int | None:
    """Case-insensitive exact match; on failure, print close suggestions and
    return None instead of raising, so one bad name doesn't kill the screen.
    """
    hit = mapping.loc[mapping["name"].str.lower() == name.lower(), "id"]
    if not hit.empty:
        return int(hit.iloc[0])
    matches = mapping.loc[
        mapping["name"].str.lower().str.contains(name.lower().split()[0]),
        "name",
    ].sort_values()
    suggestions = matches.head(10).tolist()
    print(f"  ! could not resolve '{name}'. Close matches "
          f"({len(matches)} total): {suggestions}")
    return None


# ----------------------------------------------------------------------------
# Per-pair fetch + diagnostics
# ----------------------------------------------------------------------------

def fetch_pair_daily(session, mapping: pd.DataFrame,
                      raw_name: str, processed_name: str) -> pd.DataFrame | None:
    """Returns an aligned daily df with columns ['raw', 'processed'], or None
    if either leg couldn't be resolved/fetched.
    """
    raw_id = resolve_id_safe(mapping, raw_name)
    proc_id = resolve_id_safe(mapping, processed_name)
    if raw_id is None or proc_id is None:
        return None

    raw_df = get_wg_history(session, raw_id, span="all")
    time.sleep(0.4)  # be polite to a free community API
    proc_df = get_wg_history(session, proc_id, span="all")
    time.sleep(0.4)

    if raw_df.empty or proc_df.empty:
        print(f"  ! empty history for {raw_name} or {processed_name}")
        return None

    for df in (raw_df, proc_df):
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.normalize()
        df.set_index("timestamp", inplace=True)

    aligned = (raw_df[["price"]].rename(columns={"price": "raw"})
               .join(proc_df[["price"]].rename(columns={"price": "processed"}),
                     how="inner"))
    aligned = aligned[(aligned["raw"] > 0) & (aligned["processed"] > 0)]
    return aligned


def compute_diagnostics(aligned: pd.DataFrame) -> dict:
    """Pure function: aligned df -> diagnostics dict. No I/O, easy to test."""
    ratio = aligned["processed"] / aligned["raw"]
    spread = np.log(aligned["processed"]) - np.log(aligned["raw"])

    log_rets = np.log(aligned).diff().dropna()
    ret_corr = log_rets["raw"].corr(log_rets["processed"])

    # AR(1) half-life on the spread: spread_t = a + phi * spread_{t-1} + e
    s = spread.dropna()
    s_lag, s_now = s.iloc[:-1].values, s.iloc[1:].values
    X = np.column_stack([np.ones_like(s_lag), s_lag])
    coef, *_ = np.linalg.lstsq(X, s_now, rcond=None)
    phi = coef[1]
    if 0 < phi < 1:
        half_life = -np.log(2) / np.log(phi)
    else:
        half_life = np.nan  # non mean-reverting (phi>=1) or oscillatory/degenerate (phi<=0)

    adf_p = np.nan
    if HAVE_STATSMODELS:
        try:
            adf_p = adfuller(s.values, autolag="AIC")[1]
        except Exception as exc:  # noqa: BLE001
            print(f"  ! ADF failed: {exc}")

    return {
        "n_days": len(aligned),
        "start": aligned.index.min().date(),
        "end": aligned.index.max().date(),
        "ratio_mean": ratio.mean(),
        "ratio_cv": ratio.std() / ratio.mean(),
        "ret_corr": ret_corr,
        "adf_pvalue": adf_p,
        "half_life_days": half_life,
    }


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def run_screen() -> pd.DataFrame:
    session = _session()
    print("Loading item mapping...")
    mapping = load_or_fetch_mapping(session)

    rows = []
    series_for_plot = {}

    for label, (raw_name, proc_name, note) in CANDIDATE_PAIRS.items():
        print(f"\n{label}: {raw_name} / {proc_name}  ({note})")
        aligned = fetch_pair_daily(session, mapping, raw_name, proc_name)
        if aligned is None or len(aligned) < 60:
            print("  skipped (insufficient data)")
            continue
        diag = compute_diagnostics(aligned)
        diag["label"] = label
        diag["note"] = note
        rows.append(diag)
        series_for_plot[label] = aligned["processed"] / aligned["raw"]
        print(f"  n={diag['n_days']} days, ratio CV={diag['ratio_cv']:.1%}, "
              f"ret_corr={diag['ret_corr']:.2f}, "
              f"ADF p={diag['adf_pvalue']:.3f}, "
              f"half-life={diag['half_life_days']:.0f}d"
              if pd.notna(diag['half_life_days']) else
              f"  n={diag['n_days']} days, ratio CV={diag['ratio_cv']:.1%}, "
              f"ret_corr={diag['ret_corr']:.2f}, ADF p={diag['adf_pvalue']:.3f}, "
              f"half-life=n/a (no mean reversion detected)")

    summary = pd.DataFrame(rows).set_index("label")
    # Rank: lower ADF p-value (stronger cointegration evidence) and shorter
    # half-life (more tradeable in a 14-week project) are both good.
    # If ADF is unavailable (statsmodels missing) or all-NaN, fall back to
    # ranking on half-life alone rather than silently producing NaN scores.
    hl_rank = summary["half_life_days"].rank()
    if summary["adf_pvalue"].notna().any():
        adf_rank = summary["adf_pvalue"].rank()
        summary["rank_score"] = adf_rank.fillna(adf_rank.max() + 1) + hl_rank
    else:
        print("\n  ! all ADF p-values are NaN (statsmodels missing or all "
              "tests failed) -- ranking by half-life only.")
        summary["rank_score"] = hl_rank
    summary = summary.sort_values("rank_score")
    summary.to_csv(SCREEN_DIR / "pair_screen_summary.csv")

    print("\n" + "=" * 70)
    print("RANKED SUMMARY (best candidates first)")
    print("=" * 70)
    cols = ["n_days", "ratio_cv", "ret_corr", "adf_pvalue", "half_life_days", "note"]
    with pd.option_context("display.width", 120, "display.max_colwidth", 30):
        print(summary[cols].to_string(float_format=lambda x: f"{x:.3f}"))
    print("=" * 70)

    plot_ratios(series_for_plot, summary)
    return summary


def plot_ratios(series_for_plot: dict[str, pd.Series], summary: pd.DataFrame):
    n = len(series_for_plot)
    if n == 0:
        return
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.6 * nrows), squeeze=False)
    for ax, (label, ratio) in zip(axes.flat, series_for_plot.items()):
        ax.plot(ratio.index, ratio.values, lw=0.7)
        ax.axhline(ratio.mean(), color="k", ls="--", lw=0.8)
        hl = summary.loc[label, "half_life_days"]
        hl_txt = f"{hl:.0f}d" if pd.notna(hl) else "n/a"
        ax.set_title(f"{label}  (half-life {hl_txt})", fontsize=10)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("Processed/raw price ratio by candidate pair", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pair_screen_ratios.png", dpi=130)
    plt.close(fig)
    print(f"\nsaved {FIG_DIR / 'pair_screen_ratios.png'}")


if __name__ == "__main__":
    run_screen()