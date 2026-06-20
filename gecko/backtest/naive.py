"""
Long-only rotation backtest for a cointegrated pair.

The GE has no short selling, so this isn't a long/short spread trade. It's a
rotation: hold whichever leg is cheap relative to the fitted spread, switch to
the other when the z-score crosses the threshold, and pay the GE's 2% sell tax
on each rotation (buying is untaxed).

Position logic uses hysteresis rather than separate entry/exit thresholds:
  z >  +entry_threshold  -> hold 'raw'  (proc rich, raw cheap)
  z <  -entry_threshold  -> hold 'proc' (raw rich, proc cheap)
  otherwise              -> hold the current position
This avoids whipsawing in and out near z=0, where each switch costs tax.

Simplifications, addressed by other modules:
  - daily guide prices, not realistic bid/ask (spread cost added separately
    via the half_spread args below)
  - 2% sell tax only; buy limits handled in gecko.backtest.feasibility
  - annualisation uses 365: the GE trades every day, not ~252 days/year
The z-score input is already fit walk-forward (gecko.stats.ou), so there's no
lookahead introduced here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gecko.backtest.risk_metrics import sortino_ratio, calmar_ratio


def run_naive_backtest(z: pd.Series, raw_price: pd.Series, proc_price: pd.Series,
                       entry_threshold: float = 1.5, tax: float = 0.02,
                       half_spread_raw: float = 0.0, half_spread_proc: float = 0.0,
                       initial_capital: float = 1.0) -> dict:
    """Run the hysteresis rotation strategy. Returns a dict with:
      equity_curve: pd.Series of portfolio value over time
      trades: pd.DataFrame log of every enter/exit event
      summary: dict of total_return, ann_return, ann_vol, sharpe, max_drawdown,
               n_trades

    half_spread_raw/half_spread_proc: per-leg bid-ask half-spread, charged as
    a capital haircut on EVERY transaction in that leg (both buying into it
    and selling out of it) -- crossing the spread happens on each trade, the
    2% GE tax only on selling. Default 0.0 gives the tax-only model.
    """
    df = pd.concat([z.rename("z"), raw_price.rename("raw"),
                    proc_price.rename("proc")], axis=1).dropna()
    if df.empty:
        raise ValueError("no overlapping data between z, raw_price, proc_price")
    half_spread = {"raw": half_spread_raw, "proc": half_spread_proc}

    position = None       # 'raw', 'proc', or None (no position taken yet)
    capital = float(initial_capital)
    entry_price = None
    equity_rows = []
    trade_log = []

    for date, row in df.iterrows():
        if row["z"] > entry_threshold:
            desired = "raw"
        elif row["z"] < -entry_threshold:
            desired = "proc"
        else:
            desired = position  # hold current position, do nothing

        if desired != position:
            if position is not None:
                # mark current holding to today's price, cross the spread
                # selling, then pay the GE sell tax
                capital *= row[position] / entry_price
                capital *= (1 - half_spread[position])
                capital *= (1 - tax)
                trade_log.append({"date": date, "action": f"exit_{position}",
                                  "capital_after": capital})
            if desired is not None:
                # cross the spread buying into the new leg
                capital *= (1 - half_spread[desired])
                entry_price = row[desired]
                trade_log.append({"date": date, "action": f"enter_{desired}",
                                  "capital_after": capital})
            position = desired

        mtm = capital * (row[position] / entry_price) if position is not None else capital
        equity_rows.append({"date": date, "equity": mtm, "position": position})

    equity_df = pd.DataFrame(equity_rows).set_index("date")
    equity = equity_df["equity"]
    trades = pd.DataFrame(trade_log)

    summary = _summarize(equity, trades)
    return {"equity_curve": equity, "trades": trades,
            "position_log": equity_df["position"], "summary": summary}


def estimate_half_spread(realtime_csv_path) -> float | None:
    """Estimate a realistic half-spread from cached realtime price data
    (columns avgHighPrice, avgLowPrice -- see data_pull.get_timeseries).
    half-spread = median((high-low)/mid) / 2.

    Returns None (not 0.0) if the file is missing, so a missing-data case is
    distinguishable from a measured zero spread.
    """
    path = Path(realtime_csv_path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    mid = (df["avgHighPrice"] + df["avgLowPrice"]) / 2
    full_spread_pct = (df["avgHighPrice"] - df["avgLowPrice"]) / mid
    return float(full_spread_pct.median() / 2)


def _summarize(equity: pd.Series, trades: pd.DataFrame) -> dict:
    n_days = len(equity)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    daily_rets = equity.pct_change().dropna()

    ann_return = float((1 + total_return) ** (365 / n_days) - 1) if n_days > 0 else float("nan")
    ann_vol = float(daily_rets.std() * np.sqrt(365)) if len(daily_rets) > 1 else float("nan")
    sharpe = float(ann_return / ann_vol) if ann_vol and ann_vol > 0 else float("nan")

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = float(drawdown.min())

    n_rotations = int((trades["action"].str.startswith("enter_")).sum()) if not trades.empty else 0

    sortino = sortino_ratio(daily_rets, ann_return) if len(daily_rets) > 1 else float("nan")
    calmar = calmar_ratio(ann_return, max_drawdown)

    return {
        "n_days": n_days,
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe_naive": sharpe,
        "sortino_naive": sortino,
        "calmar": calmar,
        "max_drawdown": max_drawdown,
        "n_rotations": n_rotations,
    }