"""Refresh `data/XAUUSD_H4_long.csv` from the MT5 broker feed.

WHY THIS EXISTS. `refresh_d1_panel.py` refreshes the DAILY panel from Yahoo, but the champion's
gold sleeve is driven by an H4 series (`xau_lab.load_h4`) that Yahoo does not offer at that
interval — and nothing in the repo refreshed it. It was 16 days stale on the VPS
(last bar 2026-08-31) while the book executor was about to be switched on, which would have
sized the largest sleeve in the book off two-week-old prices.

SCHEMA, matched exactly to the existing file and to `load_h4`'s expectations:

    time,open,high,low,close,tick_volume,spread,real_volume

`spread` is MT5's value in POINTS, because `load_h4` converts it with
`spread_px = max(spread, median) * 0.1`. Writing price units here would understate gold's cost
by 10x, so the raw MT5 integer is what goes in.

MERGE RULE, same as the D1 refresher: existing rows are preserved and freshly fetched rows win
on overlap, so a short fetch can never silently truncate eleven years of history.

CLOSED BARS ONLY. `copy_rates_from_pos(..,0,n)` returns the in-progress bar at position 0;
writing it would put a bar into the signal history that can still change.

    python scripts/refresh_xau_h4.py --port 18814
    python scripts/refresh_xau_h4.py --port 18814 --dry
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CSV = ROOT / "data" / "XAUUSD_H4_long.csv"
COLS = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
TF_H4 = 16388


def fetch(mt5, symbol: str, n: int) -> pd.DataFrame:
    mt5.symbol_select(symbol, True)
    r = None
    for _ in range(6):
        r = mt5.copy_rates_from_pos(symbol, TF_H4, 0, n)
        if r is not None and len(r) >= min(500, n):
            break
        time.sleep(2)
    if r is None or len(r) < 500:
        raise SystemExit(f"could not fetch {symbol} H4: {mt5.last_error()}")
    rows = [(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7] if len(x) > 7 else np.nan)
            for x in r]
    d = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close",
                                    "tick_volume", "spread", "real_volume"])
    d["time"] = pd.to_datetime(d["ts"], unit="s")
    d = d.drop(columns=["ts"]).set_index("time").sort_index()
    # drop the in-progress bar
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return d.drop(index=d.index[d.index + pd.Timedelta(hours=4) > now], errors="ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18814)
    ap.add_argument("--symbol", default="XAUUSD")
    # FTMO caps H4 requests: 5,000 succeeds, 20,000 returns
    # (-1, 'Terminal: Call failed'). Measured on bridge 18814 2026-09-16. 5,000 H4 bars is
    # ~2.3 years, far more overlap than a top-up needs, and the merge preserves the existing
    # eleven years of history regardless.
    ap.add_argument("--bars", type=int, default=5000)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    from mt5linux import MetaTrader5
    mt5 = MetaTrader5(host="localhost", port=a.port)
    if not mt5.initialize():
        raise SystemExit(f"bridge {a.port} init failed: {mt5.last_error()}")
    new = fetch(mt5, a.symbol, a.bars)
    mt5.shutdown()

    if CSV.exists():
        old = pd.read_csv(CSV, parse_dates=["time"]).set_index("time").sort_index()
        before_last, before_n = old.index.max(), len(old)
    else:
        old = pd.DataFrame(columns=COLS)
        before_last, before_n = None, 0

    # fresh rows win on overlap; history is preserved
    merged = pd.concat([old, new])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    for c in COLS:
        if c not in merged.columns:
            merged[c] = np.nan
    merged = merged[COLS]

    print(f"existing: {before_n:,} rows, last {before_last}")
    print(f"fetched : {len(new):,} closed H4 bars, {new.index.min()} -> {new.index.max()}")
    print(f"merged  : {len(merged):,} rows, last {merged.index.max()} "
          f"(+{len(merged) - before_n:,} new)")
    gap = (pd.Timestamp.now(tz='UTC').tz_localize(None) - merged.index.max())
    print(f"staleness after refresh: {gap.total_seconds()/3600:.1f}h")
    print(f"spread: median {merged['spread'].median()} (POINTS — load_h4 multiplies by 0.1)")

    if a.dry:
        print("\n(dry — nothing written)")
        return
    tmp = CSV.with_suffix(".csv.tmp")
    merged.to_csv(tmp, index_label="time")
    tmp.replace(CSV)        # atomic: a crash mid-write cannot leave a truncated panel
    print(f"\nwrote {CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
