"""
Causal feature engineering for the ML signal layer.

Operates on the OU layer's outputs (z, spread, half_life from
gecko.stats.ou.rolling_zscore) plus raw prices, so it has no statsmodels
dependency.

Every feature column uses only data up to and including day t. The label uses
day t+1 (the thing being predicted), is never fed back as a feature, and is
NaN on the final row where no t+1 exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "z", "spread_mom_1", "spread_mom_5", "spread_vol",
    "half_life", "dow_sin", "dow_cos",
    "raw_ret_1", "proc_ret_1", "raw_ret_5", "proc_ret_5",
]


def build_features_and_label(panel: pd.DataFrame, zscore_df: pd.DataFrame,
                             vol_window: int = 10) -> pd.DataFrame:
    """panel: raw_price/proc_price, DatetimeIndex.
    zscore_df: z/spread/half_life columns from rolling_zscore, same index.

    Returns a DataFrame with FEATURE_COLUMNS plus 'label':
      label = 1 if raw outperforms proc from day t to day t+1, else 0.
    """
    df = panel[["raw_price", "proc_price"]].join(
        zscore_df[["z", "spread", "half_life"]], how="inner")
    log_raw = np.log(df["raw_price"])
    log_proc = np.log(df["proc_price"])

    feat = pd.DataFrame(index=df.index)
    feat["z"] = df["z"]
    feat["spread_mom_1"] = df["spread"].diff(1)
    feat["spread_mom_5"] = df["spread"].diff(5)
    feat["spread_vol"] = df["spread"].rolling(vol_window).std()
    feat["half_life"] = df["half_life"]

    dow = df.index.dayofweek
    feat["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    feat["raw_ret_1"] = log_raw.diff(1)
    feat["proc_ret_1"] = log_proc.diff(1)
    feat["raw_ret_5"] = log_raw.diff(5)
    feat["proc_ret_5"] = log_proc.diff(5)

    raw_ret_fwd = log_raw.shift(-1) - log_raw
    proc_ret_fwd = log_proc.shift(-1) - log_proc
    label = (raw_ret_fwd > proc_ret_fwd).astype(float)
    label.iloc[-1] = np.nan  # no t+1 for the final observation
    feat["label"] = label

    return feat