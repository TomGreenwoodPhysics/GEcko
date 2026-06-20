"""
GEcko — Phase 1 data acquisition.

Pulls OSRS Grand Exchange price data from two complementary sources:

  1. OSRS Wiki real-time prices API  (prices.runescape.wiki)
       - Realistic tradeable high/low prices + traded volumes.
       - RuneLite-sampled, so a SAMPLE of real trades, not the whole GE.
       - Only goes back to ~March 2021. /timeseries returns <=365 points/call,
         timesteps in {5m, 1h, 6h} only (no daily step here).

  2. Weird Gloop history API  (api.weirdgloop.org)
       - Official GE *guide* price, one point per day, long history.
       - Slow-moving (guide price, not realistic bid/ask), but deep.

Pulling both lets you benchmark "long but coarse" vs "shallow but realistic"
and decide which feeds cointegration vs which feeds the backtest fill model.

Requires: requests, pandas   (pip install requests pandas)

IMPORTANT: edit USER_AGENT below to include a real contact. The Wiki API asks
for an identifying User-Agent; requests without one may be throttled or blocked.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

USER_AGENT = "GEcko stat-arb research project - https://github.com/TomGreenwoodPhysics/GEcko"

WIKI_BASE = "https://prices.runescape.wiki/api/v1/osrs"
WG_BASE = "https://api.weirdgloop.org/exchange/history/osrs"

OUT_DIR = Path(__file__).parent / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# A first cointegrating pair: iron ore -> iron bar (1:1 via Smithing).
# Swap/extend these once the pipeline works. (ids from the /mapping endpoint.)
DEFAULT_PAIR = {"iron_ore": 440, "iron_bar": 2351}


# ----------------------------------------------------------------------------
# HTTP plumbing
# ----------------------------------------------------------------------------

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _get(session: requests.Session, url: str, params: dict | None = None,
         retries: int = 4, backoff: float = 1.5) -> dict:
    """GET with simple exponential backoff. Returns parsed JSON."""
    last_exc = None
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:  # noqa: PERF203
            last_exc = exc
            sleep = backoff ** attempt
            print(f"  request failed ({exc}); retrying in {sleep:.1f}s")
            time.sleep(sleep)
    raise RuntimeError(f"GET {url} failed after {retries} attempts") from last_exc


# ----------------------------------------------------------------------------
# OSRS Wiki real-time prices API
# ----------------------------------------------------------------------------

def get_mapping(session: requests.Session) -> pd.DataFrame:
    """All items: id, name, GE buy limit, members flag, alch/value fields.

    The 'limit' column is the 4-hour rolling GE buy limit you MUST honour in
    the backtest cost/position model.
    """
    data = _get(session, f"{WIKI_BASE}/mapping")
    df = pd.DataFrame(data)
    keep = [c for c in ("id", "name", "limit", "members", "value",
                        "highalch", "lowalch") if c in df.columns]
    return df[keep].sort_values("id").reset_index(drop=True)


def resolve_id(mapping: pd.DataFrame, name: str) -> int:
    """Case-insensitive exact name -> item id."""
    hit = mapping.loc[mapping["name"].str.lower() == name.lower(), "id"]
    if hit.empty:
        raise KeyError(f"item not found in mapping: {name!r}")
    return int(hit.iloc[0])


def get_timeseries(session: requests.Session, item_id: int,
                   timestep: str = "6h") -> pd.DataFrame:
    """Realistic price history for one item (<=365 points).

    timestep in {'5m', '1h', '6h'}. Use '6h' for the longest window per call
    (~91 days); '1h' (~15 days) or '5m' (~30h) for finer microstructure.

    Columns returned by the API: avgHighPrice (instabuy), avgLowPrice
    (instasell), highPriceVolume, lowPriceVolume, timestamp (unix seconds).
    Note the high/low are the realistic instabuy/instasell sides — your
    effective spread to cross is roughly (avgHigh - avgLow).
    """
    if timestep not in {"5m", "1h", "6h"}:
        raise ValueError("timestep must be one of 5m, 1h, 6h")
    payload = _get(session, f"{WIKI_BASE}/timeseries",
                   params={"id": item_id, "timestep": timestep})
    df = pd.DataFrame(payload.get("data", []))
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Weird Gloop history API (long daily guide-price series)
# ----------------------------------------------------------------------------

def get_wg_history(session: requests.Session, item_id: int,
                   span: str = "all") -> pd.DataFrame:
    """Daily guide-price history for one item.

    span in {'all', 'last90d', 'sample', 'latest'}. 'all' = full history.
    Columns: timestamp (datetime, UTC), price, volume (volume may be missing
    for older points).
    """
    payload = _get(session, f"{WG_BASE}/{span}", params={"id": item_id})
    # payload is keyed by the item id as a string -> list of records.
    records = payload.get(str(item_id), [])
    df = pd.DataFrame(records)
    if df.empty:
        return df
    # Weird Gloop timestamps are unix milliseconds.
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    cols = [c for c in ("timestamp", "price", "volume") if c in df.columns]
    return df[cols].sort_values("timestamp").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def pull_pair(pair: dict[str, int] | None = None) -> None:
    pair = pair or DEFAULT_PAIR
    session = _session()

    if "REPLACE_WITH_YOUR_CONTACT" in USER_AGENT:
        print("WARNING: set a real contact in USER_AGENT before heavy use.\n")

    print("Fetching item mapping (ids, names, buy limits)...")
    mapping = get_mapping(session)
    mapping.to_csv(OUT_DIR / "mapping.csv", index=False)
    print(f"  saved mapping.csv ({len(mapping)} items)")

    for label, item_id in pair.items():
        limit_row = mapping.loc[mapping["id"] == item_id, "limit"]
        has_limit = not limit_row.empty and pd.notna(limit_row.iloc[0])
        limit = int(limit_row.iloc[0]) if has_limit else None
        print(f"\n{label} (id={item_id}, 4h buy limit={limit})")

        ts = get_timeseries(session, item_id, timestep="6h")
        ts.to_csv(OUT_DIR / f"{label}_realtime_6h.csv", index=False)
        span = (f"{ts['timestamp'].min().date()} -> {ts['timestamp'].max().date()}"
                if not ts.empty else "no data")
        print(f"  realtime 6h: {len(ts)} rows ({span})")

        wg = get_wg_history(session, item_id, span="all")
        wg.to_csv(OUT_DIR / f"{label}_daily_all.csv", index=False)
        wspan = (f"{wg['timestamp'].min().date()} -> {wg['timestamp'].max().date()}"
                 if not wg.empty else "no data")
        print(f"  daily guide: {len(wg)} rows ({wspan})")

    print(f"\nDone. Raw files in: {OUT_DIR}")


if __name__ == "__main__":
    pull_pair()