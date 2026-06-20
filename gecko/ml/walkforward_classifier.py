"""
Walk-forward fitting for any sklearn-compatible classifier.

Uses the same fit-trailing-window / predict-forward-block / refit / slide
pattern as gecko.stats.ou.rolling_zscore, and writes the same fit_end_date
column, so gecko.backtest.audit.check_no_future_leakage works on this output
unmodified.

model_factory: a zero-arg callable returning a fresh, unfit classifier with
.fit(X, y) and .predict_proba(X). Defaults to LogisticRegression; pass a
different factory to swap in GradientBoostingClassifier or any other sklearn
classifier.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def _default_model_factory():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000)


def walkforward_classify(features: pd.DataFrame, feature_cols: list[str],
                         label_col: str = "label",
                         fit_window_days: int = 730, refit_step_days: int = 30,
                         model_factory: Callable | None = None,
                         min_fit_rows: int = 30,
                         allow_nan_features: bool = False) -> pd.DataFrame:
    """Returns a DataFrame indexed by date with columns:
      proba_positive  -- P(label=1) for that date, NaN where the window
                         couldn't be fit (too little data, only one class
                         present, or -- unless allow_nan_features=True -- the
                         row's own features contain NaN)
      fit_end_date    -- last date INCLUDED in the fit window (for the audit)

    allow_nan_features: training rows with NaN are always dropped. This flag
    only affects PREDICTION: set True for models that natively tolerate missing
    features (e.g. HistGradientBoostingClassifier), so they aren't masked by a
    rule that exists for models like LogisticRegression that can't accept NaN.
    Note that with it set, such a model will score more rows than a masked one.
    """
    model_factory = model_factory or _default_model_factory
    df = features[feature_cols + [label_col]].copy()
    n = len(df)
    if n < fit_window_days + refit_step_days:
        raise ValueError(f"series has {n} obs, too short for fit_window_days="
                         f"{fit_window_days} + refit_step_days={refit_step_days}")

    blocks = []
    fit_end = fit_window_days
    while fit_end < n:
        apply_end = min(fit_end + refit_step_days, n)
        fit_slice = df.iloc[fit_end - fit_window_days: fit_end].dropna()
        apply_slice = df.iloc[fit_end: apply_end]
        if len(apply_slice) == 0:
            break

        fit_end_date = df.index[fit_end - 1]
        proba = np.full(len(apply_slice), np.nan)

        enough_data = len(fit_slice) >= min_fit_rows
        both_classes = fit_slice[label_col].nunique() == 2 if enough_data else False

        if enough_data and both_classes:
            model = model_factory()
            model.fit(fit_slice[feature_cols].values, fit_slice[label_col].values)

            X_apply = apply_slice[feature_cols].values
            if allow_nan_features:
                valid = np.ones(len(apply_slice), dtype=bool)
            else:
                valid = ~np.isnan(X_apply).any(axis=1)
            if valid.any():
                # predict_proba column order follows model.classes_; locate class 1
                classes = list(model.classes_)
                col = classes.index(1.0) if 1.0 in classes else classes.index(1)
                proba[valid] = model.predict_proba(X_apply[valid])[:, col]

        block = pd.DataFrame({"proba_positive": proba}, index=apply_slice.index)
        block["fit_end_date"] = fit_end_date
        blocks.append(block)
        fit_end = apply_end

    return pd.concat(blocks)