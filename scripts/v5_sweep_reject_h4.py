#!/usr/bin/env python3
"""Friend's H4 SWEEP-AND-REJECT short, exactly as stated (user 2026-11-12). PRE-REGISTERED.

  "the first candle can do whatever it wants but the second candle opens, sweeps above the first
   candle's HIGH and closes in the opposite, below the first candle. Stop loss at the high of the
   second candle, entry at the open of the next 4-hour candle. Expects the market to go down."
   Markets: XAUUSD and "BTCXAU" (not a tradeable symbol, §3bu) -> XAUUSD and BTCUSD.

RULE on H4 (candle 1 = i-1, candle 2 = i), bearish:
    H2 > H1  and  C2 below candle 1, three pre-declared readings (nested, a ⊂ b ⊂ c):
      a  C2 < L1                (strict: an outside reversal)
      b  C2 < min(O1, C1)       (below candle 1's body)
      c  C2 < C1                (below candle 1's close)
    SHORT at O3 (next H4 open), stop = H2. Exits: target 1R / 2R / 3R, time stop 6 bars (24h),
    stop-first when one bar touches both. Trades whose stop sits < 0.2 ATR from the
    entry are skipped in EVERY arm (R undefined). Cost once per round trip: max(bar spread, floor);
    floors XAU $0.30, BTC 4.5bp (Switch spread column is 0 before 2025).

DATA, each single-provenance, H4 resampled from H1 with label = bar OPEN:
    Switch XAUUSDs (2016-09..) / BTCUSDs (2019-03..)  server clock, H4 grid = MT5's
    CME GC (2010-06..) back-adjusted via close_adj/close  UTC
    Binance BTCUSDT (2017-08..)                          UTC

CONTROLS (scripts/v5_liquidity_sweep.py machinery: simulate / random_base / shifted_gross)
    BASE     same side, same stop distance in ATR, random bars (drift control: gold/BTC rise)
    PLACEBO  same close below candle 1 but NO sweep (H2 <= H1): does the sweep matter?
    MIRROR   L2 < L1 and C2 above candle 1 -> LONG, stop L2
    LATE     entry one bar later (O4): the lookahead probe; must not be better
    NULL     signal set circularly shifted, GROSS R, 400 draws, max-stat over all 36 cells
PASS: family p < 0.05 AND net excess over BASE > 0 in both halves on >= 3 of 4 asset/sources for
one reading, AND placebo clearly worse.

    python scripts/v5_sweep_reject_h4.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.v5_liquidity_sweep as ls  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "sweep_reject_h4"
HOLD = 6
MIN_STOP_ATR = 0.2      # hygiene, same for every arm and control: a ~0 stop makes R explode
RRS = (1.0, 2.0, 3.0)
READ = ("a_lowC1", "b_bodyC1", "c_closeC1")
N_NULL = 5 if "--quick" in sys.argv else 400
RNG = np.random.default_rng(20261112)


def to_h4(h1: pd.DataFrame) -> pd.DataFrame:
    g = h1.resample("4h", label="left", closed="left")
    d = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "spr": g["spr"].median(),
                      "n": g["open"].count()}).dropna(subset=["open"])
    return d[d.n >= 2].drop(columns="n")


def load_switch(sym: str, floor_abs: float | None, floor_bp: float | None) -> pd.DataFrame:
    d = ls.load(sym, "H1")
    h4 = to_h4(d)
    fl = floor_abs if floor_abs is not None else floor_bp * 1e-4 * h4.close
    h4["spr"] = np.maximum(h4.spr.fillna(0), fl)
    return h4


def load_cme_gc() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data/databento/ohlcv_1h/GC.csv.gz", parse_dates=["time"]).set_index("time")
    k = d.close_adj / d.close
    h1 = pd.DataFrame({"open": d.open * k, "high": d.high * k, "low": d.low * k, "close": d.close_adj})
    h1["spr"] = 0.30 * k
    roll = d.instrument_id.ne(d.instrument_id.shift()).astype(float)
    roll.iloc[0] = 0.0
    h4 = to_h4(h1)
    has_roll = roll.resample("4h", label="left", closed="left").max().reindex(h4.index).fillna(0) > 0
    return h4[~has_roll]                     # a bar spanning a roll mixes two adjustments


def load_binance_btc() -> pd.DataFrame:
    b = pd.read_parquet(ROOT / "data/binance_1m/BTCUSDT_H1.parquet")
    h1 = b.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})[["open", "high", "low", "close"]]
    h1["spr"] = 4.5e-4 * h1.close
    return to_h4(h1)


def events(d: pd.DataFrame, reading: str, kind: str) -> pd.DataFrame:
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    o1, h1, l1, c1 = (np.r_[np.nan, x[:-1]] for x in (o, h, l, c))
    if kind in ("signal", "placebo"):
        thr = {"a_lowC1": l1, "b_bodyC1": np.minimum(o1, c1), "c_closeC1": c1}[reading]
        sweep = (h > h1) if kind == "signal" else (h <= h1)
        m, side, ext = sweep & (c < thr), -1, h
    else:                                                         # mirror
        thr = {"a_lowC1": h1, "b_bodyC1": np.maximum(o1, c1), "c_closeC1": c1}[reading]
        m, side, ext = (l < l1) & (c > thr), +1, l
    i = np.flatnonzero(m)
    i = i[i < len(d) - 2]
    return pd.DataFrame({"i": i, "side": side, "extreme": ext[i]})


def trades(d: pd.DataFrame, a: np.ndarray, ev: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    out = []
    for r in ev.itertuples():
        e = r.i + lag
        if e >= len(d) or not np.isfinite(a[r.i]):
            continue
        if (r.extreme - d["open"].iat[e]) * -r.side < MIN_STOP_ATR * a[r.i]:
            continue                         # stop (nearly) at the entry: R is undefined
        res = ls.simulate(d, e, r.side, r.extreme, HOLD)
        if res:
            out.append((d.index[r.i], r.i, r.side, (r.extreme - d["open"].iat[e]) * -r.side / a[r.i], *res))
    return pd.DataFrame(out, columns=["t", "i", "side", "stop_atr", "gR", "nR", "gbp"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    srcs = {"XAU-switch": load_switch("XAUUSDs", 0.30, None), "XAU-cmeGC": load_cme_gc(),
            "BTC-switch": load_switch("BTCUSDs", None, 4.5), "BTC-binance": load_binance_btc()}
    lines = ["Friend's H4 sweep-and-reject (see docstring). R = multiples of the stop distance (O3 to H2).", ""]
    for k, d in srcs.items():
        bpd = d.groupby(d.index.normalize()).size()
        lines.append(f"{k}: {len(d)} H4 bars {d.index[0]:%Y-%m-%d}..{d.index[-1]:%Y-%m-%d}, "
                     f"median bars/day {bpd.median():.0f}, cost median {1e4 * (d.spr / d.close).median():.2f}bp")
        assert 5 <= bpd.median() <= 7, k
    rows, sig_sets = [], []
    for k, d in srcs.items():
        a = ls.atr(d, 20)
        mid = d.index[len(d) // 2]
        yrs = (d.index[-1] - d.index[0]).days / 365.25
        for rd in READ:
            ev = events(d, rd, "signal")
            for rr in RRS:
                ls.RR = rr
                tr = trades(d, a, ev)
                if tr.empty:
                    continue
                base = ls.random_base(d, a, tr, RNG, draws=5, hold=HOLD)
                pl = trades(d, a, events(d, rd, "placebo"))
                mi = trades(d, a, events(d, rd, "mirror"))
                mbase = ls.random_base(d, a, mi, RNG, draws=5, hold=HOLD) if len(mi) else np.nan
                late = trades(d, a, ev, lag=2)
                h1m, h2m = tr.t < mid, tr.t >= mid
                by = tr.groupby(pd.to_datetime(tr.t).dt.year).nR.mean()
                rows.append(dict(
                    src=k, read=rd, RR=rr, n=len(tr), per_yr=round(len(tr) / yrs, 1),
                    hit=round((tr.gR > 0).mean(), 3), be_hit=round(1 / (1 + rr), 3),
                    gR=round(tr.gR.mean(), 3), nR=round(tr.nR.mean(), 3), gbp=round(tr.gbp.mean(), 1),
                    base_nR=round(base, 3), excess=round(tr.nR.mean() - base, 3),
                    exc_h1=round(tr.nR[h1m].mean() - base, 3), exc_h2=round(tr.nR[h2m].mean() - base, 3),
                    yrs_pos=f"{int((by > 0).sum())}/{len(by)}",
                    placebo_nR=round(pl.nR.mean(), 3) if len(pl) else np.nan, placebo_n=len(pl),
                    mirror_nR=round(mi.nR.mean(), 3) if len(mi) else np.nan,
                    mirror_exc=round(mi.nR.mean() - mbase, 3) if len(mi) else np.nan,
                    late_nR=round(late.nR.mean(), 3) if len(late) else np.nan))
                sig_sets.append((d, a, tr, rr))
        print("done", k, flush=True)
    A = pd.DataFrame(rows)
    # family max-stat on GROSS R: whole signal set circularly shifted
    obs = A.gR.to_numpy()
    shifts = RNG.integers(50, 10**6, N_NULL)
    null = np.full((N_NULL, len(sig_sets)), np.nan)
    for j, (d, a, tr, rr) in enumerate(sig_sets):
        ls.RR = rr
        for s in range(N_NULL):
            null[s, j] = ls.shifted_gross(d, a, tr, int(shifts[s] % (len(d) - 2)), hold=HOLD)
    A["null_med"] = np.round(np.nanmedian(null, axis=0), 3)
    A["p"] = np.round((null >= obs).mean(axis=0), 3)
    mx = np.nanmax(null, axis=1)
    b = A.loc[A.gR.idxmax()]
    pd.set_option("display.width", 260)
    lines += ["", A.to_string(index=False), "",
              f"family max-stat ({len(A)} cells): best gross {b.src} {b.read} {b.RR}R gR {b.gR:+.3f} "
              f"vs null-max median {np.median(mx):+.3f}, p {(mx >= b.gR).mean():.3f}",
              f"cells with net excess > 0 in BOTH halves: {int(((A.exc_h1 > 0) & (A.exc_h2 > 0)).sum())} of {len(A)}",
              f"cells beating their placebo (net): {int((A.nR > A.placebo_nR).sum())} of {len(A)}",
              f"cells where LATE entry is better (lookahead flag): {int((A.late_nR > A.nR).sum())} of {len(A)}"]
    txt = "\n".join(lines)
    print(txt)
    (OUT / "report.txt").write_text(txt + "\n")
    A.to_csv(OUT / "cells.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())
