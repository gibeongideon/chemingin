#!/usr/bin/env python3
"""VWAP + 9/30 EMA (user request 2026-11-12: "Now lets try VWAP and 9 & 30 ema"). PRE-REGISTERED.

THE RETAIL RECIPE, as commonly taught:
    LONG  while EMA9 > EMA30 AND close > session VWAP
    SHORT while EMA9 < EMA30 AND close < session VWAP
    FLAT  when they disagree, and flat at the session end (VWAP resets each session)
Decided at the CLOSE of bar t (VWAP_t includes bar t, which is known then), held over bar t+1.
No same-bar use of VWAP: the §3bi circularity (t +61.83 on anchored VWAP) came from scoring a bar
against a VWAP that already contained it.

VOLUME MUST BE REAL. Switch tick_volume is 60% zeros, so it is not used. Sources:
  * Binance 1-minute klines (real traded volume): BTCUSDT, ETHUSDT 2017-08.., PAXGUSDT 2020-08..
    (PAXG = 1 oz gold token: the gold read with real volume). Bars M5 / M15 / H1, session = UTC day.
  * CME futures H1 (Databento, real volume) 2010-06..2026: GC SI HG CL NG ES NQ YM RTY 6E 6J ZN.
    Session = CME Globex day (17:00 Chicago). Sessions containing a contract roll are skipped.

ARMS (the question is whether VWAP ADDS anything):
    EMA     9/30 state alone (intraday-flat)        VWAP   close vs VWAP alone
    COMBO   both must agree  <- PRIMARY
  each LS (long/short) and LO (long-only).
COSTS per side: crypto 4.5bp (Binance futures taker + slip); PAXG and GC at the Switch gold spread
($0.30 round trip); other CME 1.5 ticks. Gross is also printed.
CONTROLS
  1. Random timing: circular shift of the position vector (session-flat mask re-applied), 100-200
     draws, on GROSS Sharpe (a cost offset inside a zero-mean null manufactures significance).
     Family max-stat over every cell.
  2. Does VWAP add: paired t of COMBO minus EMA daily net returns.
  3. Drift: LO arms beside buy-and-hold. Halves.
PASS (all): family p < 0.05; COMBO-LS net Sharpe > 0 in both halves on a majority of assets;
paired t (COMBO - EMA) > 2.

    python scripts/v5_vwap_ema.py --build     # 1m zips -> M5/M15/H1 caches (once)
    python scripts/v5_vwap_ema.py
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIPS = ROOT / "data" / "binance_1m" / "zips"
CACHE = ROOT / "data" / "binance_1m"
CME = ROOT / "data" / "databento" / "ohlcv_1h"
OUT = ROOT / "data" / "v5_runs" / "vwap_ema"
CRYPTO = ("BTCUSDT", "ETHUSDT", "PAXGUSDT")
TFS = {"M5": "5min", "M15": "15min", "H1": "1h"}
TICK = dict(GC=0.1, SI=0.005, HG=0.0005, CL=0.01, NG=0.001, ES=0.25, NQ=0.25, YM=1.0, RTY=0.1,
            **{"6E": 0.00005, "6J": 0.0000005}, ZN=0.015625)
RNG = np.random.default_rng(20261112)


# ------------------------------------------------------------------ data
def build() -> None:
    for sym in CRYPTO:
        parts = []
        for z in sorted(ZIPS.glob(f"{sym}-1m-*.zip")):
            with zipfile.ZipFile(z) as zf:
                raw = zf.read(zf.namelist()[0])
            d = pd.read_csv(io.BytesIO(raw), header=None, usecols=range(6))
            if not str(d.iat[0, 0]).isdigit():          # some files carry a header row
                d = d.iloc[1:].astype(float)
            t = d[0].astype("int64")
            t = np.where(t > 10**14, t // 1000, t)       # 2025+ files are in microseconds
            d.index = pd.to_datetime(t, unit="ms")
            parts.append(d.iloc[:, 1:6].set_axis(list("ohlcv"), axis=1).astype(float))
        m = pd.concat(parts).sort_index()
        m = m[~m.index.duplicated()]
        for tf, rule in TFS.items():
            b = m.resample(rule, label="left", closed="left").agg(
                {"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}).dropna()
            b.to_parquet(CACHE / f"{sym}_{tf}.parquet")
            print(sym, tf, len(b), b.index[0], b.index[-1])


def crypto(sym: str, tf: str) -> pd.DataFrame:
    b = pd.read_parquet(CACHE / f"{sym}_{tf}.parquet")
    b = b[b.v > 0]
    b["ret"] = b.c.pct_change().fillna(0.0)
    b["sess"] = b.index.normalize()
    b["skip"] = False
    return b


def cme(root: str) -> pd.DataFrame:
    d = pd.read_csv(CME / f"{root}.csv.gz", parse_dates=["time"]).set_index("time")
    d = d.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"})
    d = d[d.v > 0]
    chi = d.index.tz_localize("UTC").tz_convert("America/Chicago")
    d["sess"] = (chi + pd.Timedelta(hours=7)).normalize().tz_localize(None)   # 17:00 CT -> next day
    roll = d.instrument_id.ne(d.instrument_id.shift())
    roll.iloc[0] = False
    d["skip"] = d.sess.isin(set(d.sess[roll]))
    d["ret"] = d.ret_adj.fillna(0.0)
    d["c_cont"] = d.close_adj
    return d


# ------------------------------------------------------------------ signal
def signals(b: pd.DataFrame) -> dict:
    cc = b["c_cont"] if "c_cont" in b else b.c
    e9 = cc.ewm(span=9, adjust=False).mean()
    e30 = cc.ewm(span=30, adjust=False).mean()
    tp = (b.h + b.l + b.c) / 3
    g = b.sess
    vwap = (tp * b.v).groupby(g).cumsum() / b.v.groupby(g).cumsum()
    ema = np.sign(e9 - e30).to_numpy()
    vw = np.sign(b.c - vwap).to_numpy()
    combo = np.where(ema == vw, ema, 0.0)
    return {"EMA": ema, "VWAP": vw, "COMBO": combo}


def flat_mask(b: pd.DataFrame) -> np.ndarray:
    """1 where a position decided at bar t may be held over bar t+1 (same session, not skipped)."""
    s = b.sess.to_numpy()
    ok = np.r_[s[1:] == s[:-1], False] & ~b.skip.to_numpy()
    return ok.astype(float)


def daily(b: pd.DataFrame, pos: np.ndarray, cost: float) -> tuple[pd.Series, pd.Series, float]:
    """pos[t] earns ret[t+1]; cost charged per unit of turnover. Returns (net, gross) daily, trades/yr."""
    nxt = np.r_[b.ret.to_numpy()[1:], 0.0]
    gross = pos * nxt
    turn = np.abs(np.diff(np.r_[0.0, pos]))
    net = gross - turn * cost
    day = b.index.normalize()
    G = pd.Series(gross, index=day).groupby(level=0).sum()
    N = pd.Series(net, index=day).groupby(level=0).sum()
    yrs = max((b.index[-1] - b.index[0]).days / 365.25, 1e-9)
    return N, G, float(turn.sum() / 2 / yrs)


def sr(r: pd.Series, ann: float) -> float:
    return float(r.mean() / r.std() * np.sqrt(ann)) if len(r) > 2 and r.std() > 0 else 0.0


# ------------------------------------------------------------------ study
def study(name: str, b: pd.DataFrame, cost: float, ann: float, n_null: int, rows: list, nulls: list):
    sig = signals(b)
    fm = flat_mask(b)
    hold = pd.Series(np.r_[b.ret.to_numpy()[1:], 0.0], index=b.index.normalize()).groupby(level=0).sum()
    mid = hold.index[len(hold) // 2]
    shifts = RNG.integers(len(b) // 20, len(b) - len(b) // 20, n_null)
    daily_net = {}
    for arm, s in sig.items():
        for mode in ("LS", "LO"):
            raw = s if mode == "LS" else np.maximum(s, 0.0)
            pos = raw * fm
            N, G, tpy = daily(b, pos, cost)
            daily_net[(arm, mode)] = N
            sg = sr(G, ann)
            nl = [sr(daily(b, np.roll(raw, int(k)) * fm, 0.0)[1], ann) for k in shifts]
            nulls.append(nl)
            nb = np.r_[b.ret.to_numpy()[1:], 0.0]
            ntr = max(tpy * (b.index[-1] - b.index[0]).days / 365.25, 1)
            rows.append(dict(
                mkt=name, arm=arm, mode=mode, expo=round(float(np.abs(pos).mean()), 2),
                trades_yr=round(tpy), gross_bp_tr=round(1e4 * float((pos * nb).sum()) / ntr, 2),
                cost_bp_tr=round(2e4 * cost, 2), SR_gross=round(sg, 2), SR_net=round(sr(N, ann), 2),
                SR_net_h1=round(sr(N[N.index < mid], ann), 2), SR_net_h2=round(sr(N[N.index >= mid], ann), 2),
                hold_SR=round(sr(hold, ann), 2), null_med=round(float(np.median(nl)), 2),
                p=round(float((np.array(nl) >= sg).mean()), 3)))
    d = (daily_net[("COMBO", "LS")] - daily_net[("EMA", "LS")]).dropna()
    return float(d.mean() / d.std() * np.sqrt(len(d))) if d.std() > 0 else 0.0


def main() -> None:
    if "--build" in sys.argv:
        build()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    rows, nulls, paired = [], [], {}
    for sym in CRYPTO:
        for tf in TFS:
            b = crypto(sym, tf)
            cost = (0.15 / b.c.median()) if sym == "PAXGUSDT" else 4.5e-4
            n = 100 if tf == "M5" else 200
            paired[f"{sym[:-4]}-{tf}"] = study(f"{sym[:-4]}-{tf}", b, cost, 365, n, rows, nulls)
            print("done", sym, tf, flush=True)
    for root in TICK:
        b = cme(root)
        cost = (0.15 / b.c.median()) if root == "GC" else 1.5 * TICK[root] / b.c.median()
        paired[f"{root}-H1"] = study(f"{root}-H1", b, cost, 252, 200, rows, nulls)
        print("done", root, flush=True)
    A = pd.DataFrame(rows)
    mx = np.array([max(col) for col in zip(*[n[:100] for n in nulls])])
    best = A.loc[A.SR_gross.idxmax()]
    P = A[(A.arm == "COMBO") & (A["mode"] == "LS")].copy()
    P["paired_t_vs_EMA"] = P.mkt.map(paired).round(2)
    both = int(((P.SR_net_h1 > 0) & (P.SR_net_h2 > 0)).sum())
    pd.set_option("display.width", 250)
    txt = "\n".join([
        "VWAP + 9/30 EMA — pre-registered (see docstring). Positions decided at bar close, held next bar,",
        "flat at session end. Sharpe on DAILY sums (crypto sqrt 365, CME sqrt 252). p = random-timing null on GROSS.",
        "", "PRIMARY: COMBO long/short", P.drop(columns=["arm", "mode"]).to_string(index=False),
        "", f"COMBO-LS net Sharpe > 0 in BOTH halves: {both} of {len(P)} markets",
        f"paired t (COMBO - EMA) > 2: {int((P.paired_t_vs_EMA > 2).sum())} of {len(P)};  < -2: {int((P.paired_t_vs_EMA < -2).sum())}",
        f"family max-stat ({len(A)} cells): best gross {best.mkt} {best.arm}-{best["mode"]} SR {best.SR_gross:+.2f}"
        f" vs null-max median {np.median(mx):+.2f}, p {(mx >= best.SR_gross).mean():.3f}",
        f"cells net-positive: {int((A.SR_net > 0).sum())} of {len(A)};  beating hold (LO arms): "
        f"{int(((A['mode'] == 'LO') & (A.SR_net > A.hold_SR)).sum())} of {int((A['mode'] == 'LO').sum())}",
        "", "ALL CELLS", A.to_string(index=False)])
    print(txt)
    (OUT / "report.txt").write_text(txt + "\n")
    A.to_csv(OUT / "cells.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())
