#!/usr/bin/env python3
"""CME Globex (GLBX.MDP3) hourly bars from Databento — ONE provenance, 2010-06 .. now.

Why: the broker CSVs are spliced across brokers (memory: multi-broker dataset = leak) and the FX D1
files leak forward-stamped returns. Exchange futures bars are single-source and exchange-stamped
in UTC (ts_event = bar OPEN).

Continuous symbol `ROOT.v.0` = front contract by VOLUME, NOT back-adjusted. Each row keeps its
`instrument_id`, and `adjust()` builds a back-adjusted close by zeroing the return on every bar
where the contract changes. Use `ret_adj` / `close_adj` for any multi-day study.

Key: ~/.databento_key (chmod 600, never in git). Output: data/databento/ohlcv_1h/{ROOT}.csv.gz
    python scripts/fetch_databento_ohlcv.py            # default list
    python scripts/fetch_databento_ohlcv.py GC SI      # specific roots
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "databento" / "ohlcv_1h"
DATASET, START, END = "GLBX.MDP3", "2010-06-06", "2026-11-07"
DEFAULT = ["GC", "SI", "HG", "PL", "CL", "NG", "ES", "NQ", "YM", "RTY", "ZN", "ZB", "ZF",
           "6E", "6J", "6B", "6A", "6C", "6S", "BTC", "ETH", "ZC", "ZS", "ZW", "LE", "HE"]


def adjust(d: pd.DataFrame) -> pd.DataFrame:
    """Back-adjusted close: roll bars contribute zero return (the contract changed, price did not)."""
    r = np.log(d.close).diff()
    r[d.instrument_id != d.instrument_id.shift()] = 0.0
    d["ret_adj"] = r.fillna(0.0)
    d["close_adj"] = d.close.iloc[-1] * np.exp(d.ret_adj.cumsum() - d.ret_adj.sum())
    return d


def fetch(client, root: str) -> pd.DataFrame:
    parts = []
    edges = [START] + [f"{y}-01-01" for y in range(2011, int(END[:4]) + 1)] + [END]
    for a, b in zip(edges[:-1], edges[1:]):        # yearly chunks: one 16-year request stalls
        x = client.timeseries.get_range(dataset=DATASET, symbols=[f"{root}.v.0"], stype_in="continuous",
                                        schema="ohlcv-1h", start=a, end=b).to_df()
        if len(x):
            parts.append(x)
    d = pd.concat(parts)
    d.index = d.index.tz_convert("UTC").tz_localize(None)
    d.index.name = "time"
    d = d[["instrument_id", "open", "high", "low", "close", "volume"]].sort_index()
    return adjust(d[~d.index.duplicated(keep="last")])


def main() -> None:
    import databento as db  # noqa: PLC0415
    client = db.Historical(Path("~/.databento_key").expanduser().read_text().strip())
    OUT.mkdir(parents=True, exist_ok=True)
    for root in sys.argv[1:] or DEFAULT:
        f = OUT / f"{root}.csv.gz"
        if f.exists():
            print(f"{root}: exists, skip")
            continue
        try:
            d = fetch(client, root)
        except Exception as e:  # noqa: BLE001
            print(f"{root}: FAILED {str(e)[:160]}")
            continue
        d.to_csv(f, float_format="%.10g")
        rolls = int((d.instrument_id != d.instrument_id.shift()).sum()) - 1
        print(f"{root}: {len(d):,} bars {d.index[0]} -> {d.index[-1]}, {rolls} rolls, "
              f"last close {d.close.iloc[-1]:.4g}", flush=True)


if __name__ == "__main__":
    main()
