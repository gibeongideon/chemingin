#!/usr/bin/env python3
"""Second look #2: the §3ac HIGH-VOL TAPER, out of sample on GOLD (pre-registered in
data/v5_runs/second_look/PREREG.md).

§3ac found the monotone taper (size x 1.0/0.8/0.6/0.4/0.2 by the trailing-2y percentile quintile of EWMA
vol) lifted the XAU H4 champion 1.084 -> 1.287 on 2018-26, t@matched +1.23, both halves. §3ad then closed
it because it failed on BTC/NDX/BRENT. But a gold-specific effect was never tested OUT OF SAMPLE ON GOLD.
This runs the SAME frozen schedule, engine and gate (imported from v5_xau_stack_proposal /
v5_xau_champion_lifts) on CME GC H4 2012-2017 — years that played no part in choosing it.

PRIMARY (pre-registered): CME GC (Databento, back-adjusted) H1 -> H4, eval 2012-01..2017-12.
SUPPLEMENTARY (NOT pre-registered, reported for context only): OANDA H1 -> H4, eval 2009-01..2017-12.
Pass: matched-vol paired t > 0, positive in both halves, prop arm not worse.

    python scripts/v5_second_look_2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import v5_xau_champion_lifts as lc  # noqa: E402
from v5_xau_stack_proposal import pct_rank  # noqa: E402
from v5_xau_turn_prob import paired, per_year  # noqa: E402
from v5_zigzag_panel_probs import load_cme  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "second_look" / "2"
SCHED = {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2}     # frozen: §3ac "monotone taper 1.0->0.2"


def to_h4(h1: pd.DataFrame) -> pd.DataFrame:
    g = h1.groupby(h1.index.floor("4h"))
    h4 = pd.DataFrame(dict(open=g.open.first(), high=g.high.max(), low=g.low.min(), close=g.close.last())).dropna()
    h4["spread_px"] = lc.COST_USD
    return h4


def cme() -> pd.DataFrame:
    return to_h4(load_cme("GC"))


def oanda() -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "data/oanda_hf/XAUUSD_H1.parquet")
    d.index = pd.to_datetime(d.time, unit="s")
    return to_h4(d[["open", "high", "low", "close"]].sort_index()[lambda x: x.index >= "2007-01-01"])


def prop(daily: pd.Series) -> tuple:
    from v5_gft_check import challenge  # noqa: PLC0415
    r = daily.to_numpy()
    return challenge(r, r.std() * np.sqrt(252), 0.09, None, 0.0, False, False)


def run(name: str, df: pd.DataFrame, start: str, end: str = "2017-12-31", do_prop: bool = True) -> dict:
    lc.EVAL_START = start
    close = df["close"]
    ret = close.pct_change()
    fc = champion_signal(close)
    vol_ewma = np.sqrt(ret.pow(2).ewm(halflife=lc.VOL_HL, min_periods=20).mean()) * np.sqrt(lc.ANN_H4)
    q = np.clip((pct_rank(vol_ewma) * 5).fillna(2).astype(int), 0, 4)
    mult = q.map(SCHED).astype(float)
    base = lc.engine_v(df, fc.clip(0, 2)).loc[:end]
    tap = lc.engine_v(df, (fc * mult).clip(0, 2)).loc[:end]
    tm = lc.vol_match(tap, base)
    _, t, _ = paired(tm, base)
    yp, yn = per_year(tm, base)
    mid = base.index[len(base) // 2]
    halves = []
    for a, b in ((base[:mid], tap[:mid]), (base[mid:], tap[mid:])):
        _, th, _ = paired(lc.vol_match(b, a), a)
        halves.append((lc.sharpe(a), lc.sharpe(b), th))
    # 2x cost
    df2 = df.copy(); df2["spread_px"] *= 2
    s0 = lc.SLIP_USD; lc.SLIP_USD = s0 * 2
    b2 = lc.engine_v(df2, fc.clip(0, 2)).loc[:end]; t2d = lc.engine_v(df2, (fc * mult).clip(0, 2)).loc[:end]
    lc.SLIP_USD = s0
    _, t2, _ = paired(lc.vol_match(t2d, b2), b2)
    print(f"\n=== {name}: eval {base.index[0]:%Y-%m-%d} -> {base.index[-1]:%Y-%m-%d} ({len(base)} days) ===")
    print(f"  BASE champion  SR {lc.sharpe(base):+.3f}  maxDD {lc.dd_of(base):+.1f}%")
    print(f"  TAPER          SR {lc.sharpe(tap):+.3f}  maxDD {lc.dd_of(tap):+.1f}%  exposure kept "
          f"{float(mult[base.index[0]:end].mean()):.2f}")
    print(f"  MATCHED-VOL paired t {t:+.2f}  yrs+ {yp}/{yn}   2x cost t {t2:+.2f}")
    print(f"  halves: 1st base {halves[0][0]:+.2f} taper {halves[0][1]:+.2f} t {halves[0][2]:+.2f} | "
          f"2nd base {halves[1][0]:+.2f} taper {halves[1][1]:+.2f} t {halves[1][2]:+.2f}")
    row = dict(source=name, base_sr=lc.sharpe(base), taper_sr=lc.sharpe(tap), base_dd=lc.dd_of(base),
               taper_dd=lc.dd_of(tap), t_matched=t, yrs_pos=yp, yrs=yn, t_2x=t2,
               h1_t=halves[0][2], h2_t=halves[1][2])
    if do_prop:
        pb, pt = prop(base), prop(tap)
        print(f"  prop (GFT 2-step static, 9% dial): base pass {pb[0]:.1f}% daily-fail {pb[2]:.1f}% | "
              f"taper pass {pt[0]:.1f}% daily-fail {pt[2]:.1f}%")
        row.update(prop_base=pb[0], prop_taper=pt[0], dayfail_base=pb[2], dayfail_taper=pt[2])
    return row


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # known-answer control: the original 2018-26 window on the original-style data must show the
    # taper ahead (§3ac: 1.084 -> 1.287); here on the single-feed M15->H4 rebuild.
    from v5_second_look_1 import s1_h4  # noqa: PLC0415
    ka = run("KNOWN-ANSWER: M15->H4 2018-26 (selection window)", s1_h4(), "2018-01-01", "2026-12-31", do_prop=False)
    rows = [ka]
    rows.append(run("PRIMARY: CME GC H4 2012-17 (unseen)", cme(), "2012-01-01"))
    rows.append(run("SUPPLEMENTARY (not prereg): OANDA H4 2009-17", oanda(), "2009-01-01"))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "cells.csv", index=False)
    p = R.iloc[1]
    ok = (p.t_matched > 0) and (p.h1_t > 0) and (p.h2_t > 0) and (p.prop_taper >= p.prop_base - 1e-9)
    print(f"\nPRIMARY VERDICT (CME 2012-17): {'PASS' if ok else 'FAIL'}  "
          f"(t {p.t_matched:+.2f}, halves {p.h1_t:+.2f}/{p.h2_t:+.2f}, prop {p.prop_base:.1f}% -> {p.prop_taper:.1f}%)")
    print(f"wrote {OUT / 'cells.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
