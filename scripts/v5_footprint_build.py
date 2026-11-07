#!/usr/bin/env python3
"""Hourly FOOTPRINT bars from Binance 1-minute klines (real aggressor volume).

A footprint chart splits each bar's volume by PRICE and by AGGRESSOR (market buys lifting the ask
vs market sells hitting the bid). Binance kline files carry `taker_buy_base_volume` per minute, so
every minute is one (price, buy-vol, sell-vol) cell — a 60-cell footprint per hour. Gold CFDs have
no traded volume (Switch tick_volume is 60% zeros), so gold's flow is proxied by PAXGUSDT
(1 token = 1 oz, 2020-08 ..).

Per UTC hour we keep only what the strategy tests need (disk is tight, minutes are not re-stored):
  o h l c        price
  v tb           volume, taker-BUY volume  -> delta = 2*tb - v
  vwap n         volume-weighted price, trades
  poc            point of control: position (0=low .. 1=high) of the max-volume price bucket (10)
  lo_v lo_tb     volume / taker-buy volume of minutes that traded in the bar's bottom 25%
  hi_v hi_tb     same for the top 25%   (absorption: heavy selling at the low that fails = bullish)

    python scripts/v5_footprint_build.py            # BTCUSDT ETHUSDT PAXGUSDT
Output: data/binance_1m/{SYM}_fp_H1.csv.gz (git-ignored directory)
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
ZIPS = ROOT / "data" / "binance_1m" / "zips"
COLS = ["o", "h", "l", "c", "v", "tb", "vwap", "n", "poc", "lo_v", "lo_tb", "hi_v", "hi_tb"]


@njit(cache=True)
def hours(hid, o, h, l, c, v, tb, n):
    m = len(hid)
    starts = [0]
    for i in range(1, m):
        if hid[i] != hid[i - 1]:
            starts.append(i)
    starts.append(m)
    k = len(starts) - 1
    out = np.full((k, 13), np.nan)
    key = np.empty(k, np.int64)
    for j in range(k):
        a, b = starts[j], starts[j + 1]
        key[j] = hid[a]
        H, L = h[a:b].max(), l[a:b].min()
        V, TB = v[a:b].sum(), tb[a:b].sum()
        rng = H - L
        typ = (h[a:b] + l[a:b] + c[a:b]) / 3.0
        out[j, 0], out[j, 1], out[j, 2], out[j, 3] = o[a], H, L, c[b - 1]
        out[j, 4], out[j, 5], out[j, 7] = V, TB, n[a:b].sum()
        out[j, 6] = (typ * v[a:b]).sum() / V if V > 0 else c[b - 1]
        if rng > 0 and V > 0:
            bk = np.zeros(10)
            for i in range(a, b):
                q = min(9, int((typ[i - a] - L) / rng * 10))
                bk[q] += v[i]
            out[j, 8] = (np.argmax(bk) + 0.5) / 10
            lo_v = lo_tb = hi_v = hi_tb = 0.0
            for i in range(a, b):
                if l[i] <= L + 0.25 * rng:
                    lo_v += v[i]
                    lo_tb += tb[i]
                if h[i] >= H - 0.25 * rng:
                    hi_v += v[i]
                    hi_tb += tb[i]
            out[j, 9], out[j, 10], out[j, 11], out[j, 12] = lo_v, lo_tb, hi_v, hi_tb
    return key, out


def month(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    a = np.loadtxt(io.BytesIO(raw), delimiter=",", usecols=(0, 1, 2, 3, 4, 5, 8, 9), ndmin=2,
                   comments=None, converters=None) if raw[:1].isdigit() else \
        np.loadtxt(io.BytesIO(raw), delimiter=",", usecols=(0, 1, 2, 3, 4, 5, 8, 9), skiprows=1, ndmin=2)
    t = a[:, 0].astype(np.int64)
    t = np.where(t > 10**14, t // 1000, t)          # 2025+ spot files are in MICROseconds
    hid = t // 3_600_000
    key, out = hours(hid, a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5], a[:, 7], a[:, 6])
    return pd.DataFrame(out, columns=COLS, index=pd.to_datetime(key * 3600, unit="s"))


def build(sym: str) -> pd.DataFrame:
    parts = [month(p) for p in sorted(ZIPS.glob(f"{sym}-1m-*.zip"))]
    d = pd.concat(parts)
    d = d[~d.index.duplicated(keep="last")].sort_index()
    d.index.name = "time"
    d.to_csv(ROOT / "data" / "binance_1m" / f"{sym}_fp_H1.csv.gz", float_format="%.8g")
    return d


if __name__ == "__main__":
    for s in sys.argv[1:] or ["BTCUSDT", "ETHUSDT", "PAXGUSDT"]:
        d = build(s)
        dl = (2 * d.tb - d.v) / d.v
        print(f"{s}: {len(d):,} hours {d.index[0]} -> {d.index[-1]}  zero-vol hours "
              f"{100 * (d.v <= 0).mean():.2f}%  median hourly $vol {np.median(d.v * d.vwap):,.0f}  "
              f"delta mean {dl.mean():+.3f} sd {dl.std():.3f}", flush=True)
