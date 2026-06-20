"""
Sortino and Calmar ratios (Sharpe and max drawdown live in gecko.backtest.naive).

Sortino penalises only downside deviation, not total volatility, so a strategy
with frequent small losses and rare large gains (positive skew) scores higher
under Sortino than Sharpe. Reporting both surfaces that asymmetry.

Calmar = annualised return / max drawdown -- a worst-case risk-adjusted return,
complementary to Sharpe's volatility-based view.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def downside_deviation(returns: pd.Series, target: float = 0.0) -> float:
    """Full-method downside deviation: sqrt(mean of squared shortfalls below
    target), averaged over ALL observations (not just the negative ones) --
    the standard convention, since it correctly reflects that a strategy
    spending more time above target has objectively LESS downside risk.
    """
    r = pd.Series(returns).dropna().astype(float)
    shortfall = (r - target).clip(upper=0.0)
    return float(np.sqrt((shortfall ** 2).mean()))


def sortino_ratio(daily_returns: pd.Series, ann_return: float,
                  target: float = 0.0, periods_per_year: int = 365) -> float:
    """ann_return: the already-computed annualised return (geometric), so
    this stays consistent with however the rest of the backtest annualises
    returns rather than re-deriving it differently here.
    """
    dd_daily = downside_deviation(daily_returns, target)
    if dd_daily == 0:
        return float("nan")
    dd_ann = dd_daily * np.sqrt(periods_per_year)
    return float((ann_return - target * periods_per_year) / dd_ann)


def calmar_ratio(ann_return: float, max_drawdown: float) -> float:
    """ann_return / |max_drawdown|. NaN if there was no drawdown at all
    (division by zero would otherwise silently produce inf)."""
    if max_drawdown == 0:
        return float("nan")
    return float(ann_return / abs(max_drawdown))