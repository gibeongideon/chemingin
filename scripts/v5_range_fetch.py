"""Fetch fresh multi-timeframe XAUUSD from the broker feed for the range/extremes study.

WHY NOT THE CSVs. data/XAUUSD_H1_long.csv is >2,000 hours stale and the D1 panel is Yahoo
(GC=F futures, not the spot CFD actually traded). This study is about the SHAPE of the daily
and weekly bar — highs, lows, where in the session they land — so it must use the same feed
the account trades, at full resolution. One feed, no splice.

M15 is the resolution that decides the question the user actually raised ("even in a day the
bottom and top seem almost the same"): you cannot see WHEN the high and low of a day happen
from daily bars, only from intraday ones.

    python scripts/v5_range_fetch.py --port 18814
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v5_runs" / "range_study"
TF = {"M15": 15, "M30": 30, "H1": 16385, "H4": 16388, "D1": 16408, "W1": 32769}
# FTMO caps large requests per timeframe: H4 fails at 20k but works at 5k (measured 2026-09-16).
WANT = {"M15": 60000, "M30": 40000, "H1": 30000, "H4": 5000, "D1": 3000, "W1": 600}


def fetch(mt5, sym: str, tf: str, n: int) -> pd.DataFrame:
    mt5.symbol_select(sym, True)
    got = None
    for req in (n, n // 2, n // 4, 5000, 2000):
        for _ in range(3):
            r = mt5.copy_rates_from_pos(sym, TF[tf], 0, req)
            if r is not None and len(r) > 100:
                got = r
                break
            time.sleep(1.5)
        if got is not None:
            break
    if got is None:
        raise SystemExit(f"could not fetch {sym} {tf}: {mt5.last_error()}")
    rows = [(x[0], x[1], x[2], x[3], x[4], x[5], x[6]) for x in got]
    d = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close",
                                    "tick_volume", "spread"])
    d["time"] = pd.to_datetime(d["ts"], unit="s")
    d = d.drop(columns=["ts"]).set_index("time").sort_index()
    d = d[~d.index.duplicated(keep="last")]
    # drop the in-progress bar
    mins = {"M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440, "W1": 10080}[tf]
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return d.drop(index=d.index[d.index + pd.Timedelta(minutes=mins) > now], errors="ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18814)
    ap.add_argument("--symbol", default="XAUUSD")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from mt5linux import MetaTrader5
    mt5 = MetaTrader5(host="localhost", port=a.port)
    if not mt5.initialize():
        raise SystemExit(f"bridge {a.port} init failed: {mt5.last_error()}")
    print(f"{'tf':5s} {'bars':>7s}  {'from':<19s} {'to':<19s} {'years':>6s}")
    for tf, n in WANT.items():
        d = fetch(mt5, a.symbol, tf, n)
        yrs = (d.index[-1] - d.index[0]).days / 365.25
        d.to_csv(OUT / f"{a.symbol}_{tf}.csv", index_label="time")
        print(f"{tf:5s} {len(d):7d}  {str(d.index[0])[:19]:<19s} "
              f"{str(d.index[-1])[:19]:<19s} {yrs:6.2f}")
    mt5.shutdown()
    print(f"\nwrote -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
