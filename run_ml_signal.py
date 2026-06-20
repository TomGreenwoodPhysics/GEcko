"""
ML signal layer, benchmarked against the OU z-score baseline.

The comparison holds everything fixed except the signal source: same backtest
engine (gecko.backtest.naive), same costs (tax + measured bid-ask spread),
same pair (hide). Only the entry/exit signal differs -- the OU z-score, or a
walk-forward classifier.

The model predicts P(raw outperforms proc over the next day) from causal
features (z, spread momentum/vol, half-life, day-of-week, recent returns --
see gecko/ml/features.py). The probability is centered to [-0.5, +0.5] and fed
into run_naive_backtest like a z-score; entry_threshold=0.10 means acting only
above ~60% confidence.

Swapping classifiers is a one-line change: pass a different model_factory to
walkforward_classify.

Run AFTER run_ou_signal.py (needs hide's zscore.csv for the z/spread/half_life
features). Run from the project root:   python run_ml_signal.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gecko.ml.features import build_features_and_label, FEATURE_COLUMNS
from gecko.ml.walkforward_classifier import walkforward_classify
from gecko.backtest.audit import check_no_future_leakage
from gecko.backtest.naive import run_naive_backtest, estimate_half_spread
from gecko.backtest.deflated_sharpe import deflated_sharpe_ratio

CLEAN_DIR = Path("data/clean")
RAW_DIR = Path("data/raw")

LABEL = "hide"
FIT_WINDOW_DAYS = 730
REFIT_STEP_DAYS = 30
ML_ENTRY_THRESHOLD = 0.10
N_TRIALS_SCREENED = 8

# Both models are reported, not just the better one. HistGB natively tolerates
# NaN features (see allow_nan_features in walkforward_classifier.py);
# LogisticRegression does not, so it scores fewer days.
def _model_registry():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    return {
        "Logistic regression": {
            "factory": lambda: LogisticRegression(max_iter=1000),
            "allow_nan_features": False,
        },
        "Gradient boosted trees": {
            "factory": lambda: HistGradientBoostingClassifier(max_iter=100, max_depth=3),
            "allow_nan_features": True,
        },
    }


def run_one_model(name: str, factory, allow_nan_features: bool,
                  feat: pd.DataFrame, panel: pd.DataFrame,
                  hs_raw: float, hs_proc: float) -> dict:
    print(f"\n=== {name} (fit {FIT_WINDOW_DAYS}d, refit every {REFIT_STEP_DAYS}d) ===")
    ml_roll = walkforward_classify(feat, feature_cols=FEATURE_COLUMNS,
                                   label_col="label",
                                   fit_window_days=FIT_WINDOW_DAYS,
                                   refit_step_days=REFIT_STEP_DAYS,
                                   model_factory=factory,
                                   allow_nan_features=allow_nan_features)
    n_scored = ml_roll["proba_positive"].notna().sum()
    print(f"  {len(ml_roll)} scored days, {n_scored} with a usable prediction")

    audit = check_no_future_leakage(ml_roll)
    print(f"  lookahead audit: {audit['n_violations']} violations -- "
          f"{'PASS' if audit['passed'] else 'FAIL'}")

    ml_signal = (ml_roll["proba_positive"] - 0.5).rename("z")
    merged = ml_signal.to_frame().join(panel[["raw_price", "proc_price"]], how="inner")
    res = run_naive_backtest(merged["z"], merged["raw_price"], merged["proc_price"],
                             entry_threshold=ML_ENTRY_THRESHOLD, tax=0.02,
                             half_spread_raw=hs_raw, half_spread_proc=hs_proc)
    s = res["summary"]
    print(f"  ann. return: {s['ann_return']:+.1%}   Sharpe: {s['sharpe_naive']:.2f}   "
          f"Sortino: {s['sortino_naive']:.2f}   max DD: {s['max_drawdown']:.1%}   "
          f"rotations: {s['n_rotations']}")

    daily_rets = res["equity_curve"].pct_change().dropna()
    dsr = deflated_sharpe_ratio(daily_rets, n_trials=N_TRIALS_SCREENED)
    print(f"  DSR: {dsr['dsr']:.1%}")

    return {"signal": name, "ann_return": s["ann_return"], "sharpe": s["sharpe_naive"],
            "sortino": s["sortino_naive"], "max_dd": s["max_drawdown"],
            "rotations": s["n_rotations"], "dsr": dsr["dsr"],
            "date_range": (merged.index.min(), merged.index.max())}


def main():
    z_path = CLEAN_DIR / f"{LABEL}_zscore.csv"
    panel_path = CLEAN_DIR / f"{LABEL}_daily_clean.csv"
    if not z_path.exists() or not panel_path.exists():
        print(f"  ! missing inputs for {LABEL} -- run the pipeline up to "
              f"run_ou_signal.py first")
        return

    zdf = pd.read_csv(z_path, index_col=0, parse_dates=True)
    panel = pd.read_csv(panel_path, index_col=0, parse_dates=True)

    print(f"=== {LABEL}: building causal features ===")
    feat = build_features_and_label(panel, zdf)
    print(f"  {len(feat)} rows, {feat['label'].notna().sum()} with a valid "
          f"next-day label")

    hs_raw = estimate_half_spread(RAW_DIR / f"{LABEL}_raw_realtime_6h.csv") or 0.0
    hs_proc = estimate_half_spread(RAW_DIR / f"{LABEL}_proc_realtime_6h.csv") or 0.0

    rows = []
    earliest_start, latest_start = None, None
    for name, cfg in _model_registry().items():
        r = run_one_model(name, cfg["factory"], cfg["allow_nan_features"],
                          feat, panel, hs_raw, hs_proc)
        start, end = r.pop("date_range")
        rows.append(r)
        latest_start = start if latest_start is None else max(latest_start, start)

    # OU baseline, evaluated on the SAME date range as the ML models (the
    # later of their two start dates, so every model -- including whichever
    # needed more history to produce a first prediction -- is compared fairly)
    ou_merged = zdf[["z"]].join(panel[["raw_price", "proc_price"]], how="inner")
    ou_merged = ou_merged.loc[latest_start:]
    res_ou = run_naive_backtest(ou_merged["z"], ou_merged["raw_price"], ou_merged["proc_price"],
                                entry_threshold=1.5, tax=0.02,
                                half_spread_raw=hs_raw, half_spread_proc=hs_proc)
    s_ou = res_ou["summary"]
    dsr_ou = deflated_sharpe_ratio(res_ou["equity_curve"].pct_change().dropna(),
                                   n_trials=N_TRIALS_SCREENED)
    rows.insert(0, {"signal": "OU z-score (baseline)", "ann_return": s_ou["ann_return"],
                    "sharpe": s_ou["sharpe_naive"], "sortino": s_ou["sortino_naive"],
                    "max_dd": s_ou["max_drawdown"], "rotations": s_ou["n_rotations"],
                    "dsr": dsr_ou["dsr"]})

    print("\n" + "=" * 78)
    print("Comparison: same pair, date range, costs, and engine")
    print("=" * 78)
    comparison = pd.DataFrame(rows)
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    comparison.to_csv(CLEAN_DIR / "ml_vs_ou_comparison.csv", index=False)


if __name__ == "__main__":
    main()