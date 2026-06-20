"""
GE buy-limit feasibility check.

The backtest assumes every rotation executes instantly and in full at the
day's guide price. The GE buy limit (units per rolling 4 hours, per item)
breaks that assumption above some capital scale: a large position into a
limited-supply item can't fill in one window.

From the backtest's trade log, this computes the largest notional capital for
which every historical rotation would still have fit inside a single 4-hour
window -- a feasibility ceiling, not a cost. Below it the instant-execution
assumption holds; above it, execution would need to be phased across multiple
windows (and likely wouldn't fill at the quoted guide price).
"""

from __future__ import annotations

import math

import pandas as pd


def safe_capital_ceiling(trades: pd.DataFrame, prices_panel: pd.DataFrame,
                         item_limits: dict[str, float]) -> dict:
    """trades: a backtest's trade log (res['trades'] from run_naive_backtest),
    with 'date' and 'action' columns (actions like 'enter_raw', 'enter_proc').
    prices_panel: daily panel with raw_price/proc_price, same dates as trades.
    item_limits: {'raw': buy_limit_units, 'proc': buy_limit_units} -- the GE's
    4-hour rolling buy limit for each leg's item.

    Returns the per-leg and overall safe capital ceiling (gp), plus which
    date was the binding (worst-case) constraint for each leg.
    """
    if trades.empty:
        return {"raw_ceiling_gp": float("nan"), "proc_ceiling_gp": float("nan"),
                "overall_ceiling_gp": float("nan"), "n_entries": 0}
    entries = trades[trades["action"].str.startswith("enter_")].copy()
    if entries.empty:
        return {"raw_ceiling_gp": float("nan"), "proc_ceiling_gp": float("nan"),
                "overall_ceiling_gp": float("nan"), "n_entries": 0}

    entries["leg"] = entries["action"].str.replace("enter_", "", regex=False)
    entries["date"] = pd.to_datetime(entries["date"])
    price_col = {"raw": "raw_price", "proc": "proc_price"}

    per_leg_ceiling = {}
    per_leg_worst_date = {}
    for leg in ("raw", "proc"):
        leg_entries = entries[entries["leg"] == leg]
        if leg_entries.empty or leg not in item_limits:
            continue
        prices = prices_panel.loc[leg_entries["date"], price_col[leg]]
        max_per_window = item_limits[leg] * prices  # gp tradeable in one 4h window
        per_leg_ceiling[leg] = float(max_per_window.min())
        per_leg_worst_date[leg] = str(max_per_window.idxmin().date())

    overall = min(per_leg_ceiling.values()) if per_leg_ceiling else float("nan")

    return {
        "raw_ceiling_gp": per_leg_ceiling.get("raw", float("nan")),
        "proc_ceiling_gp": per_leg_ceiling.get("proc", float("nan")),
        "raw_worst_date": per_leg_worst_date.get("raw"),
        "proc_worst_date": per_leg_worst_date.get("proc"),
        "overall_ceiling_gp": overall,
        "n_entries": len(entries),
    }


def windows_needed(capital_gp: float, price: float, buy_limit_units: float) -> int:
    """How many 4-hour windows to fully execute a capital_gp entry into an
    item with this price and buy limit, assuming (optimistically) the price
    doesn't move against you while phasing in."""
    max_per_window = buy_limit_units * price
    if max_per_window <= 0:
        return float("inf")
    return math.ceil(capital_gp / max_per_window)


def feasibility_at_capital(trades: pd.DataFrame, prices_panel: pd.DataFrame,
                           item_limits: dict[str, float],
                           capital_gp: float) -> dict:
    """For a specific notional capital, what fraction of the backtest's
    actual historical entries would have needed more than one 4-hour window?
    """
    if trades.empty:
        return {"capital_gp": capital_gp, "n_entries": 0,
               "pct_needing_multiple_windows": float("nan"), "max_windows_needed": 0}
    entries = trades[trades["action"].str.startswith("enter_")].copy()
    if entries.empty:
        return {"capital_gp": capital_gp, "n_entries": 0,
               "pct_needing_multiple_windows": float("nan"), "max_windows_needed": 0}

    entries["leg"] = entries["action"].str.replace("enter_", "", regex=False)
    entries["date"] = pd.to_datetime(entries["date"])
    price_col = {"raw": "raw_price", "proc": "proc_price"}

    needed = []
    for _, row in entries.iterrows():
        leg = row["leg"]
        if leg not in item_limits:
            continue
        price = prices_panel.loc[row["date"], price_col[leg]]
        needed.append(windows_needed(capital_gp, price, item_limits[leg]))

    if not needed:
        return {"capital_gp": capital_gp, "n_entries": 0,
               "pct_needing_multiple_windows": float("nan"), "max_windows_needed": 0}

    pct_multi = sum(1 for w in needed if w > 1) / len(needed)
    return {
        "capital_gp": capital_gp,
        "n_entries": len(needed),
        "pct_needing_multiple_windows": pct_multi,
        "max_windows_needed": max(needed),
    }