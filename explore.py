"""
First look at the iron ore / iron bar data.

Loads the four CSVs from data/raw/, prints a sanity report, and saves two
figures to figures/. Exploratory only -- the reusable logic lives in the
gecko/ package.

Run from the project root:   python explore.py
Then open figures/iron_overview.png and figures/iron_provenance.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save PNGs; no interactive window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW = Path("data/raw")
FIG = Path("figures")
FIG.mkdir(exist_ok=True)

ORE, BAR = "iron_ore", "iron_bar"


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_daily(leg: str) -> pd.DataFrame:
    """Long daily guide-price series: columns price, volume; daily index."""
    df = pd.read_csv(RAW / f"{leg}_daily_all.csv", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df.index = df.index.normalize()          # collapse to the date
    df = df[~df.index.duplicated(keep="last")]
    return df


def load_realtime(leg: str) -> pd.DataFrame:
    """Realistic 6h series. Adds a 'mid' = mean(avgHigh, avgLow)."""
    df = pd.read_csv(RAW / f"{leg}_realtime_6h.csv", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df["mid"] = df[["avgHighPrice", "avgLowPrice"]].mean(axis=1)
    return df


# ----------------------------------------------------------------------------
# Sanity report
# ----------------------------------------------------------------------------

def sanity_report(ore_d, bar_d, ore_r, bar_r) -> pd.DataFrame:
    print("=" * 64)
    print("SANITY REPORT")
    print("=" * 64)

    # --- daily coverage + alignment ---
    print("\nDaily guide series:")
    for leg, d in (("ore", ore_d), ("bar", bar_d)):
        print(f"  {leg}: {len(d):>5} rows  "
              f"{d.index.min().date()} -> {d.index.max().date()}")

    aligned = (ore_d[["price"]].rename(columns={"price": "ore"})
               .join(bar_d[["price"]].rename(columns={"price": "bar"}),
                     how="inner"))
    dropped = max(len(ore_d), len(bar_d)) - len(aligned)
    print(f"  common dates after inner join: {len(aligned)}  "
          f"(dropped {dropped} unmatched)")

    # --- ratio as a crude mean-reversion eyeball (NOT a cointegration test) ---
    ratio = aligned["bar"] / aligned["ore"]
    print("\nbar/ore daily price ratio:")
    print(f"  mean {ratio.mean():.3f}   std {ratio.std():.3f}   "
          f"min {ratio.min():.3f}   max {ratio.max():.3f}")
    print(f"  coefficient of variation: {ratio.std()/ratio.mean():.3%}  "
          "(smaller = tighter co-movement)")

    # --- daily log-return correlation ---
    rets = np.log(aligned[aligned > 0]).diff().dropna()
    corr = rets["ore"].corr(rets["bar"])
    print(f"\ndaily log-return correlation (ore vs bar): {corr:.3f}")

    # --- realtime data quality + the cost reality ---
    print("\nRealtime 6h series (the prices you'd actually trade):")
    for leg, r in (("ore", ore_r), ("bar", bar_r)):
        miss = r["mid"].isna().mean()
        spread_pct = ((r["avgHighPrice"] - r["avgLowPrice"]) / r["mid"]).median()
        print(f"  {leg}: {len(r)} buckets, {miss:.1%} empty, "
              f"median bid-ask spread {spread_pct:.2%} of price")
    print("  ^ compare that spread to the 2% sell tax: both must be cleared "
          "to profit.")
    print("=" * 64 + "\n")

    return aligned


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def plot_overview(ore_d, bar_d, ore_r, bar_r, aligned):
    fig, axes = plt.subplots(3, 1, figsize=(11, 11))

    # (1) full daily history, twin axes (different price scales)
    ax = axes[0]
    ax.plot(ore_d.index, ore_d["price"], color="tab:blue", lw=0.8, label="ore")
    ax.set_ylabel("ore price (gp)", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(bar_d.index, bar_d["price"], color="tab:orange", lw=0.8, label="bar")
    ax2.set_ylabel("bar price (gp)", color="tab:orange")
    ax.set_title("Daily guide price, full history — do they co-move?")

    # (2) the ratio (the whole pairs-trading thesis in one line)
    ax = axes[1]
    ratio = aligned["bar"] / aligned["ore"]
    ax.plot(ratio.index, ratio, color="tab:green", lw=0.8)
    ax.axhline(ratio.mean(), color="k", ls="--", lw=1,
               label=f"mean {ratio.mean():.2f}")
    ax.set_ylabel("bar / ore")
    ax.set_title("Price ratio — a flat-ish, mean-reverting line is what you want")
    ax.legend(loc="upper right")

    # (3) realtime window zoom, realistic mid prices
    ax = axes[2]
    ax.plot(ore_r.index, ore_r["mid"], color="tab:blue", lw=0.9, label="ore mid")
    ax3 = ax.twinx()
    ax3.plot(bar_r.index, bar_r["mid"], color="tab:orange", lw=0.9, label="bar mid")
    ax.set_ylabel("ore mid (gp)", color="tab:blue")
    ax3.set_ylabel("bar mid (gp)", color="tab:orange")
    ax.set_title("Realistic 6h prices (~last 3 months)")

    fig.tight_layout()
    fig.savefig(FIG / "iron_overview.png", dpi=130)
    plt.close(fig)
    print(f"saved {FIG / 'iron_overview.png'}")


def plot_provenance(ore_d, ore_r):
    """Same item, two sources: guide price vs realistic mid, over the overlap."""
    start = ore_r.index.min()
    guide = ore_d.loc[ore_d.index >= start.normalize(), "price"]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(guide.index, guide.values, color="tab:red", lw=1.2,
            label="daily guide price")
    ax.plot(ore_r.index, ore_r["mid"], color="tab:blue", lw=0.9, alpha=0.8,
            label="realtime mid (realistic)")
    ax.set_title("Iron ore: guide price vs realistic price — how much do they "
                 "disagree?")
    ax.set_ylabel("price (gp)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "iron_provenance.png", dpi=130)
    plt.close(fig)
    print(f"saved {FIG / 'iron_provenance.png'}")


# ----------------------------------------------------------------------------

def main():
    ore_d, bar_d = load_daily(ORE), load_daily(BAR)
    ore_r, bar_r = load_realtime(ORE), load_realtime(BAR)
    aligned = sanity_report(ore_d, bar_d, ore_r, bar_r)
    plot_overview(ore_d, bar_d, ore_r, bar_r, aligned)
    plot_provenance(ore_d, ore_r)
    print("\nOpen the two PNGs in figures/ to see the data.")


if __name__ == "__main__":
    main()