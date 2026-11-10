#!/usr/bin/env python3
"""FREE external data as stand-aside gates (user request 2026-11-10: "start from the free sources, don't buy").

A. FRED macro regimes -> ZigZag variant B on gold (CME GC 2012-26, Switch 2018-26). Block NEW entries
   while the regime is hawkish (pre-registered sign: bad for gold). Publication-safe lags:
     HIKE    effective fed funds (DFF) up > 25bp vs 126 business days earlier      (data to t-1)
     Y2UP    2-year Treasury yield (DGS2) higher than 63 business days earlier     (data to t-1)
     BEDOWN  10y breakeven inflation (T10YIE) lower than 20 business days earlier  (data to t-1)
     CPIDN   CPI YoY (CPIAUCNS, never revised) lower than 3 months earlier; month m used from the
             1st of m+2 (CPI is published mid-m+1)
B. Binance BTCUSDT perpetual FUNDING -> holding BTC (2020-01..now). Flat (0 return) while longs are
   crowded: 7-day mean funding above its trailing-365-day 80th / 90th percentile (data to t-1).
Null for every cell: the SAME gate mask shifted to random dates (200 draws); family max-stat per part.
GDELT news tone was also attempted but the API refuses this IP and the VPS (rate-limit wall).

    python scripts/v5_free_data_gates.py --fetch
    python scripts/v5_free_data_gates.py
"""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_zigzag_fall_gate import day_to_bars  # noqa: E402
from scripts.v5_zigzag_long_history import OUT as ZZ, SRC, THR, load, sharpe  # noqa: E402
from scripts.v5_zigzag_regime import bt  # noqa: E402

DATA = ROOT / "data" / "free_data"
OUT = ROOT / "data" / "v5_runs" / "free_data_gates"
FRED = ("DFF", "DGS2", "T10YIE", "CPIAUCNS")
N_NULL = 200
UA = {"User-Agent": "research-script/1.0"}


def get(url: str) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()


def fetch() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for s in FRED:
        (DATA / f"fred_{s}.csv").write_bytes(get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}"))
        print("FRED", s)
    rows, t = [], 1577836800000                                   # 2020-01-01
    while True:
        r = json.loads(get(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&startTime={t}&limit=1000"))
        if not r:
            break
        rows += r
        t = r[-1]["fundingTime"] + 1
        if len(r) < 1000:
            break
        time.sleep(0.3)
    f = pd.DataFrame(rows)
    f["time"] = pd.to_datetime(f.fundingTime, unit="ms")
    f[["time", "fundingRate"]].to_csv(DATA / "btc_funding.csv", index=False)
    print("funding", len(f), f.time.iloc[0], f.time.iloc[-1])
    kl, t = [], 1577836800000
    while True:
        r = json.loads(get(f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&startTime={t}&limit=1000"))
        if not r:
            break
        kl += r
        t = r[-1][0] + 86_400_000
        if len(r) < 1000:
            break
    k = pd.DataFrame([(x[0], float(x[4])) for x in kl], columns=["t", "close"])
    k["time"] = pd.to_datetime(k.t, unit="ms")
    k[["time", "close"]].to_csv(DATA / "btc_daily.csv", index=False)
    print("btc daily", len(k), k.time.iloc[0], k.time.iloc[-1])


def fred(s: str) -> pd.Series:
    d = pd.read_csv(DATA / f"fred_{s}.csv")
    d.columns = ["date", "v"]
    d["date"] = pd.to_datetime(d.date)
    return pd.to_numeric(d.set_index("date").v, errors="coerce").dropna()


def macro_flags() -> dict:
    dff, d2, be = fred("DFF"), fred("DGS2"), fred("T10YIE")
    f = {"HIKE": dff - dff.shift(126) > 0.25, "Y2UP": d2 > d2.shift(63), "BEDOWN": be < be.shift(20)}
    cpi = fred("CPIAUCNS")
    yoy = cpi / cpi.shift(12) - 1
    cd = yoy < yoy.shift(3)
    cd.index = cd.index + pd.offsets.MonthBegin(2)                # month m usable from the 1st of m+2
    f["CPIDN"] = cd
    out = {}
    for k, v in f.items():
        v = v.astype(float)
        v.index = v.index + (pd.Timedelta(days=1) if k != "CPIDN" else pd.Timedelta(0))   # daily data: from t+1
        out[k] = v.resample("D").ffill() > 0.5
    return out


def part_a() -> str:
    flags = macro_flags()
    lines = ["A. FRED macro regimes -> ZigZag variant B (block new entries while the regime is ON)"]
    rng = np.random.default_rng(11)
    for src in ("cme", "switch"):
        cfg = SRC[src]
        st = pd.Timestamp(cfg["start"])
        g = load(src, cfg["gold"])
        p = pd.read_csv(ZZ / src / "probs.csv", index_col=0, parse_dates=True).iloc[:, 0].reindex(g.index).to_numpy()
        fires = np.isfinite(p) & (p >= THR)
        z = np.zeros(len(g), bool)
        tr0, r0 = bt(g, fires, z, st, cfg["swap"], 0)
        s0 = int((g.index < st).sum())
        shifts = rng.integers(60 * 24, len(g) - s0 - 60 * 24, N_NULL)
        rows, nulls = [], []
        for k, fl in flags.items():
            blk = day_to_bars(fl, g.index, 0)                     # lags already applied in macro_flags
            tr, r = bt(g, fires & ~blk, z, st, cfg["swap"], 0)
            s = sharpe(r)
            nl = []
            for sh in shifts:
                b2 = blk.copy()
                b2[s0:] = np.roll(blk[s0:], int(sh))
                nl.append(sharpe(bt(g, fires & ~b2, z, st, cfg["swap"], 0)[1]))
            nulls.append(nl)
            _, ron = bt(g, fires & blk, z, st, cfg["swap"], 0)    # what the blocked regime would have earned
            rows.append(dict(gate=k, on_pct=round(100 * blk[s0:].mean(), 0), trades=len(tr), SR=round(s, 2),
                             dSR=round(s - sharpe(r0), 2), SR_inside_regime=round(sharpe(ron), 2),
                             null_med=round(float(np.median(nl)), 2), p=round(float((np.array(nl) >= s).mean()), 3)))
        A = pd.DataFrame(rows)
        mx = np.array(nulls).max(axis=0)
        lines += [f"\n{src.upper()} (V0 Sharpe {sharpe(r0):+.2f})", A.to_string(index=False),
                  f"family: best {A.gate[A.SR.idxmax()]} {A.SR.max():+.2f} vs shifted max median {np.median(mx):+.2f}, "
                  f"p {(mx >= A.SR.max()).mean():.3f}"]
    return "\n".join(lines)


def part_b() -> str:
    f = pd.read_csv(DATA / "btc_funding.csv", parse_dates=["time"]).set_index("time").fundingRate
    px = pd.read_csv(DATA / "btc_daily.csv", parse_dates=["time"]).set_index("time").close
    r = px.pct_change().dropna()
    fd = f.resample("D").mean()
    m7 = fd.rolling(7).mean()
    rng = np.random.default_rng(12)
    start = pd.Timestamp("2021-01-01")                            # one year of funding history for the percentile
    rr = r[r.index >= start]
    rows, nulls = [], []
    for q in (0.8, 0.9):
        thr = m7.rolling(365, min_periods=300).quantile(q)
        flag = (m7 > thr).shift(1).reindex(rr.index).fillna(False).to_numpy(bool)   # known at t-1
        g = rr.where(~flag, 0.0)
        s = sharpe(g)
        nl = [sharpe(rr.where(~np.roll(flag, int(k)), 0.0)) for k in rng.integers(30, len(rr) - 30, N_NULL)]
        nulls.append(nl)
        rows.append(dict(gate=f"crowded>{int(q * 100)}pct", flat_pct=round(100 * flag.mean(), 0),
                         SR=round(s, 2), hold_SR=round(sharpe(rr), 2),
                         ret_bp_when_flagged=round(1e4 * rr[flag].mean(), 1), ret_bp_otherwise=round(1e4 * rr[~flag].mean(), 1),
                         null_med=round(float(np.median(nl)), 2), p=round(float((np.array(nl) >= s).mean()), 3)))
    A = pd.DataFrame(rows)
    mx = np.array(nulls).max(axis=0)
    c = pd.concat([m7.shift(1), r], axis=1).dropna()
    return ("B. Binance BTC funding -> holding BTC, 2021-01..now (crypto days, sqrt 252 kept for comparability)\n"
            + A.to_string(index=False)
            + f"\nfamily: best {A.SR.max():+.2f} vs shifted max median {np.median(mx):+.2f}, p {(mx >= A.SR.max()).mean():.3f}"
            + f"\ncorr(7d funding at t-1, BTC return t) {c.iloc[:, 0].corr(c.iloc[:, 1]):+.3f}")


def main() -> None:
    if "--fetch" in sys.argv:
        fetch()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    txt = part_a() + "\n\n" + part_b() + "\n"
    print(txt)
    (OUT / "report.txt").write_text(txt)


if __name__ == "__main__":
    sys.exit(main())
