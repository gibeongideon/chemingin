#!/usr/bin/env python3
"""Fetch Switch US-share CFD M5 history (bid bars + spread + tick volume) for the ORB study.

Gentle on the SHARED bridge 18815: one symbol at a time, a short pause between symbols, never
shutdown() (it drops the session for every Switch job, see the 2026-11-04 incident). Resumable:
symbols already on disk are skipped. A symbol that returns no bars on the first ask is retried
once after a pause (the terminal downloads history on demand).

    python scripts/v5_us_m5_fetch.py --port 18815
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "us_m5"


def grab(m, s):
    for k in (60000, 47000, 30000, 20000, 10000):
        r = m.copy_rates_from_pos(s, 5, 1, k)            # TIMEFRAME_M5, pos 1 = closed bars
        if r is not None and len(r):
            return r
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18815)
    a = ap.parse_args()
    from mt5linux import MetaTrader5
    m = MetaTrader5(host="localhost", port=a.port)
    if not m.initialize():
        raise SystemExit(f"init failed {m.last_error()}")
    OUT.mkdir(parents=True, exist_ok=True)
    syms = sorted(s.name for s in m.symbols_get()
                  if "Classic US Shares" in s.path and s.trade_mode == 4)
    print(len(syms), "tradeable US shares", flush=True)
    done = fail = 0
    for s in syms:
        f = OUT / f"{s}.csv.gz"
        if f.exists():
            continue
        m.symbol_select(s, True)
        r = grab(m, s)
        if r is None:
            time.sleep(3)
            r = grab(m, s)
        if r is None:
            fail += 1
            continue
        a_ = np.array([tuple(x)[:7] for x in r], float)
        d = pd.DataFrame(a_[:, 1:], columns=["o", "h", "l", "c", "tv", "sp"],
                         index=pd.to_datetime(a_[:, 0], unit="s"))
        d.index.name = "t"
        d.to_csv(f, compression="gzip")
        done += 1
        if done % 50 == 0:
            print(f"{done} saved, {fail} empty", flush=True)
        time.sleep(0.3)
    info = {s.name: (s.point, s.trade_contract_size) for s in m.symbols_get()
            if "Classic US Shares" in s.path}
    pd.DataFrame(info, index=["point", "contract"]).T.to_csv(OUT / "_meta.csv")
    print(f"DONE {done} saved, {fail} empty", flush=True)
    # NO m.shutdown(): bridge 18815 is shared.


if __name__ == "__main__":
    main()
