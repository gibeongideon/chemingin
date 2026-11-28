#!/usr/bin/env python3
"""Second look #7: fast intraday XAU trend, LONG-ONLY, at today's cost (pre-registered addendum in
data/v5_runs/second_look/PREREG.md). §3c/§3aa priced it at $0.34-0.44 with swap ignored and found it real
but cost-dead; §3aa's own reopen condition was "$0.12 + swap modelled". Here: 20 years of single-feed
OANDA H1, costs $0.25 / $0.12, swap 0 / -6%/yr, hold vs flat-overnight, judged (T1) against a matched
constant-long hold and (T2) as a 50/50 partner of the H4 champion at matched vol.

    python scripts/v5_second_look_7.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import v5_xau_turn_prob as tp  # noqa: E402
from v5_second_look_1 import matched_t, s2_h4, sharpe  # noqa: E402
from v5_xau_fast_ls_screen import ANN_H1, BUFFER, MAX_LEV, SLIP_USD, TARGET_VOL, VOL_HL_H1, ewmac_fast  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "second_look" / "7"
EVAL = "2008-01-01"


def h1() -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "data/oanda_hf/XAUUSD_H1.parquet")
    d.index = pd.to_datetime(d.time, unit="s")
    return d[["open", "high", "low", "close"]].sort_index()[lambda x: x.index >= "2007-01-01"]


def engine(df, fc, spread, swap=0.0, flat_night=False):
    close = df["close"]
    ret = close.pct_change()
    vol = ret.ewm(halflife=VOL_HL_H1, min_periods=20).std() * np.sqrt(ANN_H1)
    pos = (fc.clip(0.0, 2.0) * (TARGET_VOL / vol)).clip(0, MAX_LEV)
    band = (BUFFER * (TARGET_VOL / vol).clip(0, MAX_LEV)).values
    p, out, held = pos.values.copy(), np.zeros(len(pos)), 0.0
    for i in range(len(p)):
        if np.isfinite(p[i]):
            b = band[i] if np.isfinite(band[i]) else 0.0
            if abs(p[i] - held) > b:
                held = p[i] - np.sign(p[i] - held) * b
        out[i] = held
    pos = pd.Series(out, index=pos.index)
    et = df.index.tz_localize("UTC").tz_convert("America/New_York")
    if flat_night:
        pos[(et.hour >= 16) & (et.hour < 18)] = 0.0
    pos = pos.shift(1).fillna(0.0)
    cost = (spread / 2.0 + SLIP_USD) / close
    net = pos * ret - pos.diff().abs().fillna(0.0) * cost
    if swap:
        roll = (et.hour == 17)                                   # the bar starting at the 17:00 ET rollover
        net = net - np.where(roll, pos * abs(swap) / 252.0, 0.0)
    eq = (1.0 + net.fillna(0.0)).cumprod().loc[EVAL:]
    eq = eq / eq.iloc[0]
    d = eq.resample("D").last().pct_change(fill_method=None).dropna()
    return d[d.index.dayofweek < 5]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = h1()
    fc = ewmac_fast(df, ((12, 48), (15, 60))).clip(lower=0.0)
    one = pd.Series(1.0, index=df.index)
    h4 = s2_h4()
    rows = []
    for C, sw, fn in itertools.product((0.25, 0.12), (0.0, -0.06), (False, True)):
        fast = engine(df, fc, C, sw, fn)
        hold = engine(df, one, C, sw, fn)
        h4c = h4.copy(); h4c["spread_px"] = C
        _, champ = tp.engine(h4c, champion_signal(h4c["close"]), eval_start=EVAL)
        champ = champ[champ.index.dayofweek < 5]
        j = fast.index.intersection(champ.index).intersection(hold.index)
        f, hdl, c = fast.reindex(j).fillna(0), hold.reindex(j).fillna(0), champ.reindex(j).fillna(0)
        t1 = matched_t(f, hdl)[0]
        book = 0.5 * f * (c.std() / f.std()) + 0.5 * c
        t2, da = matched_t(book, c)
        mid = j[len(j) // 2]
        h_1, h_2 = matched_t(book[:mid], c[:mid])[0], matched_t(book[mid:], c[mid:])[0]
        prim = (C == 0.25 and sw == 0.0 and not fn)
        tag = f"C ${C:.2f} swap {sw:+.0%} {'flat-night' if fn else 'hold'}" + ("  <- PRIMARY" if prim else "")
        print(f"{tag:<44} fast SR {sharpe(f):+.3f}  hold SR {sharpe(hdl):+.3f}  T1 t {t1:+.2f} | champ SR {sharpe(c):+.3f}"
              f"  50/50 SR {sharpe(book):+.3f}  T2 t {t2:+.2f} halves {h_1:+.2f}/{h_2:+.2f}  corr {np.corrcoef(f, c)[0, 1]:+.2f}")
        rows.append(dict(cost=C, swap=sw, flat_night=fn, primary=prim, fast_sr=sharpe(f), hold_sr=sharpe(hdl), t1=t1,
                         champ_sr=sharpe(c), book_sr=sharpe(book), t2=t2, t2_h1=h_1, t2_h2=h_2,
                         corr=float(np.corrcoef(f, c)[0, 1])))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "cells.csv", index=False)
    p = R[R.primary].iloc[0]
    ok = p.t1 >= 2 and p.t2 >= 2 and p.t2_h1 > 0 and p.t2_h2 > 0
    print(f"\nPRIMARY VERDICT: {'REVIVED' if ok else 'not revived'} (T1 {p.t1:+.2f}, T2 {p.t2:+.2f}, "
          f"halves {p.t2_h1:+.2f}/{p.t2_h2:+.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
