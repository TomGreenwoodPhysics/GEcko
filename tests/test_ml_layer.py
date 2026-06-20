import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from gecko.ml.features import build_features_and_label, FEATURE_COLUMNS
from gecko.ml.walkforward_classifier import walkforward_classify
from gecko.backtest.audit import check_no_future_leakage


def test_features_hand_traced():
    idx = pd.date_range("2024-01-01", periods=8)
    panel = pd.DataFrame({
        "raw_price": [100, 102, 101, 105, 103, 108, 106, 110],
        "proc_price": [50, 50, 51, 50, 52, 51, 53, 52],
    }, index=idx)
    zscore_df = pd.DataFrame({
        "z": [0.1, 0.2, -0.1, 0.3, 0.0, 0.4, -0.2, 0.1],
        "spread": [1.0, 1.05, 1.02, 1.10, 1.08, 1.15, 1.11, 1.20],
        "half_life": [10.0] * 8,
    }, index=idx)

    feat = build_features_and_label(panel, zscore_df)

    assert abs(feat["spread_mom_1"].iloc[1] - 0.05) < 1e-9
    assert abs(feat["raw_ret_1"].iloc[1] - np.log(102 / 100)) < 1e-9
    assert feat["label"].iloc[0] == 1.0
    assert feat["label"].iloc[2] == 1.0
    assert np.isnan(feat["label"].iloc[-1])
    print("test_features_hand_traced: PASS")


def test_features_columns_present():
    idx = pd.date_range("2024-01-01", periods=20)
    rng = np.random.default_rng(0)
    panel = pd.DataFrame({"raw_price": 100 + rng.normal(size=20).cumsum(),
                         "proc_price": 50 + rng.normal(size=20).cumsum()}, index=idx)
    zscore_df = pd.DataFrame({"z": rng.normal(size=20), "spread": rng.normal(size=20),
                              "half_life": np.full(20, 5.0)}, index=idx)
    feat = build_features_and_label(panel, zscore_df)
    for c in FEATURE_COLUMNS:
        assert c in feat.columns
    print("test_features_columns_present: PASS")


def _regime_flip_features(n=2200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n)
    x = rng.normal(size=n)
    first_half = np.arange(n) < n // 2
    label = np.where(first_half, (x > 0).astype(float), (x < 0).astype(float))
    df = pd.DataFrame({"x": x, "label": label}, index=idx)
    return df


def test_walkforward_classify_never_uses_future_regime():
    df = _regime_flip_features()
    roll = walkforward_classify(df, feature_cols=["x"], label_col="label",
                                fit_window_days=500, refit_step_days=100)

    early = roll.iloc[:200].join(df[["x"]])
    late = roll.iloc[-200:].join(df[["x"]])
    early_corr = early["proba_positive"].corr(early["x"])
    late_corr = late["proba_positive"].corr(late["x"])

    assert early_corr > 0.3, f"early corr {early_corr} should be clearly positive"
    assert late_corr < -0.3, f"late corr {late_corr} should be clearly negative"
    print(f"test_walkforward_classify_never_uses_future_regime: PASS "
          f"(early corr={early_corr:.2f}, late corr={late_corr:.2f})")


def test_walkforward_classify_passes_lookahead_audit():
    df = _regime_flip_features()
    roll = walkforward_classify(df, feature_cols=["x"], label_col="label",
                                fit_window_days=500, refit_step_days=100)
    rep = check_no_future_leakage(roll)
    assert rep["passed"]
    print(f"test_walkforward_classify_passes_lookahead_audit: PASS "
          f"({rep['n_rows']} rows, 0 violations)")


def test_walkforward_classify_handles_insufficient_data():
    idx = pd.date_range("2018-01-01", periods=900)
    df = pd.DataFrame({"x": np.random.default_rng(1).normal(size=900),
                       "label": np.zeros(900)}, index=idx)
    roll = walkforward_classify(df, feature_cols=["x"], label_col="label",
                                fit_window_days=500, refit_step_days=100)
    assert roll["proba_positive"].isna().all()
    print("test_walkforward_classify_handles_insufficient_data: PASS")


def test_walkforward_classify_nan_features_yield_nan_proba():
    df = _regime_flip_features(n=900)
    df.loc[df.index[600], "x"] = np.nan
    roll = walkforward_classify(df, feature_cols=["x"], label_col="label",
                                fit_window_days=500, refit_step_days=100)
    assert pd.isna(roll.loc[df.index[600], "proba_positive"])
    assert pd.notna(roll.loc[df.index[601], "proba_positive"])
    print("test_walkforward_classify_nan_features_yield_nan_proba: PASS")


def test_allow_nan_features_scores_rows_default_path_masks():
    # Default (allow_nan_features=False): a NaN-feature row gets NaN proba,
    # even with a model that COULD handle it -- confirms the flag genuinely
    # gates the behavior rather than the model's capability alone deciding.
    from sklearn.ensemble import HistGradientBoostingClassifier
    df = _regime_flip_features(n=900)
    df.loc[df.index[600], "x"] = np.nan

    roll_masked = walkforward_classify(
        df, feature_cols=["x"], label_col="label",
        fit_window_days=500, refit_step_days=100,
        model_factory=lambda: HistGradientBoostingClassifier(max_iter=50),
        allow_nan_features=False)
    assert pd.isna(roll_masked.loc[df.index[600], "proba_positive"])

    roll_unmasked = walkforward_classify(
        df, feature_cols=["x"], label_col="label",
        fit_window_days=500, refit_step_days=100,
        model_factory=lambda: HistGradientBoostingClassifier(max_iter=50),
        allow_nan_features=True)
    assert pd.notna(roll_unmasked.loc[df.index[600], "proba_positive"])
    print("test_allow_nan_features_scores_rows_default_path_masks: PASS")


if __name__ == "__main__":
    test_features_hand_traced()
    test_features_columns_present()
    test_walkforward_classify_never_uses_future_regime()
    test_walkforward_classify_passes_lookahead_audit()
    test_walkforward_classify_handles_insufficient_data()
    test_walkforward_classify_nan_features_yield_nan_proba()
    test_allow_nan_features_scores_rows_default_path_masks()
    print("\nAll ML layer tests passed.")