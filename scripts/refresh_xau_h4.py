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

ANY TIMEFRAME, added 2026-09-19. The advisor's shipped cell is a SOURCE arm, so it needs M15
intrabar features, and `data/XAUUSD_M15_long.csv` was a month staler than the H4 file
(2026-06-15 vs 2026-07-16) with nothing in the repo able to refresh it. `--tf` covers that
without a second copy of this logic. The default is H4 with the H4 path, so every recorded
result and every existing timer keeps working unchanged.

The in-progress-bar drop is derived from the timeframe's own duration rather than hard-coded to
four hours — with `--tf M15` a 4-hour rule would discard the most recent 16 closed bars, which
on a 5-minute service is the difference between a live reading and a stale one.

    python scripts/refresh_xau_h4.py --port 18814                 # H4, as before
    python scripts/refresh_xau_h4.py --port 18814 --tf M15
    python scripts/refresh_xau_h4.py --port 18814 --tf M15 --dry
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

COLS = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]

# MT5 TIMEFRAME_* constants and each one's bar duration in minutes
TFS = {"M5": (5, 5), "M15": (15, 15), "M30": (30, 30),
       "H1": (16385, 60), "H4": (16388, 240), "D1": (16408, 1440)}
TF_H4 = TFS["H4"][0]                      # kept: imported by name elsewhere


def fetch(mt5, symbol: str, n: int, tf: str = "H4") -> pd.DataFrame:
    code, minutes = TFS[tf]
    mt5.symbol_select(symbol, True)
    r = None
    for _ in range(6):
        r = mt5.copy_rates_from_pos(symbol, code, 0, n)
        if r is not None and len(r) >= min(500, n):
            break
        time.sleep(2)
    if r is None or len(r) < 500:
        raise SystemExit(f"could not fetch {symbol} {tf}: {mt5.last_error()}")
    rows = [(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7] if len(x) > 7 else np.nan)
            for x in r]
    d = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close",
                                    "tick_volume", "spread", "real_volume"])
    d["time"] = pd.to_datetime(d["ts"], unit="s")
    d = d.drop(columns=["ts"]).set_index("time").sort_index()
    # drop the in-progress bar — by THIS timeframe's duration, not H4's
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return d.drop(index=d.index[d.index + pd.Timedelta(minutes=minutes) > now], errors="ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18814)
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--tf", default="H4", choices=sorted(TFS))
    ap.add_argument("--csv", default=None,
                    help="target file; defaults to data/<symbol>_<tf>_long.csv")
    # FTMO caps H4 requests: 5,000 succeeds, 20,000 returns
    # (-1, 'Terminal: Call failed'). Measured on bridge 18814 2026-09-16. 5,000 H4 bars is
    # ~2.3 years, far more overlap than a top-up needs, and the merge preserves the existing
    # eleven years of history regardless.
    ap.add_argument("--bars", type=int, default=5000)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    csv = Path(a.csv) if a.csv else ROOT / "data" / f"{a.symbol}_{a.tf}_long.csv"

    from mt5linux import MetaTrader5
    mt5 = MetaTrader5(host="localhost", port=a.port)
    if not mt5.initialize():
        raise SystemExit(f"bridge {a.port} init failed: {mt5.last_error()}")
    new = fetch(mt5, a.symbol, a.bars, a.tf)
    mt5.shutdown()

    if csv.exists():
        old = pd.read_csv(csv, parse_dates=["time"]).set_index("time").sort_index()
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

    print(f"target  : {csv.relative_to(ROOT)}")
    print(f"existing: {before_n:,} rows, last {before_last}")
    print(f"fetched : {len(new):,} closed {a.tf} bars, {new.index.min()} -> {new.index.max()}")
    print(f"merged  : {len(merged):,} rows, last {merged.index.max()} "
          f"(+{len(merged) - before_n:,} new)")
    gap = (pd.Timestamp.now(tz='UTC').tz_localize(None) - merged.index.max())
    print(f"staleness after refresh: {gap.total_seconds()/3600:.1f}h")
    print(f"spread: median {merged['spread'].median()} (POINTS — load_h4 multiplies by 0.1)")

    if a.dry:
        print("\n(dry — nothing written)")
        return
    tmp = csv.with_suffix(".csv.tmp")
    merged.to_csv(tmp, index_label="time")
    tmp.replace(csv)        # atomic: a crash mid-write cannot leave a truncated panel
    print(f"\nwrote {csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
