#!/usr/bin/env python3
"""ZigZag RECOVERY ADD (§3ds, user request 2026-11-09): when a ZigZag buy first moves toward its stop,
place a second buy "to complement the first and recover if it reverses".

Same 539 live fires (Switch H1 2024-01..2026-11), event set fixed, no profit lock, 3.0 ATR trail,
swap included. Base position = 1 unit risking 1R (3 ATR) = 0.25% equity.
GRID (27 cells, pre-registered):
  trigger d  : add when price falls d x ATR(entry) below entry, d in {1.0, 1.5, 2.0} (stop is 3.0)
  size m     : add m x the first position, m in {0.5, 1.0, 2.0}
  exit mode  : SHARED  add rides with the original (same stop/trail, closes together)
               ADD_BE  add takes profit back at the ORIGINAL entry price (recovers the drawdown);
                       otherwise shares the stop; the original runs on untouched
               ALL_BE  BOTH close at the combined break-even (+ spread) once price recovers to it
Fills: the add is a resting buy limit at e - d*ATR; a gap below fills at the open (+ spread).
Same-bar ambiguity is resolved PESSIMISTICALLY: add fill and stop in one bar -> both happen.
CONTROLS (memory: scale-in = leverage, lot progressions = leverage):
  * MATCHED RISK: each variant's daily returns rescaled to the base's volatility; report the
    vol-matched total and paired t vs base (the gate: t >= 2 and better in both halves).
  * Worst single trade (R of the base unit) and max drawdown, raw and at matched vol.
  * Skipped-sample: mean base R of trades whose add FILLED vs NOT (adds only fill on trades going wrong).

    python scripts/v5_zigzag_recovery_add.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_zigzag_live_wf import MAX_HOLD, RISK, SPREAD, START, SWAP, TRAIL, load, sharpe  # noqa: E402
from scripts.v5_zigzag_lock_ladder import R_USD, entries  # noqa: E402
from scripts.v5_zigzag_signal import atr  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "zigzag_recovery_add"
DS, MS, MODES = (1.0, 1.5, 2.0), (0.5, 1.0, 2.0), ("SHARED", "ADD_BE", "ALL_BE")


def trade(o, h, l, a, s, d, m, mode):
    """Return (exit bar, pnl in price units summed over units, filled?) for one entry."""
    ae = a[s - 1]
    e = o[s] + SPREAD
    sl, best = e - TRAIL * ae, e
    lvl = e - d * ae if d else -np.inf
    add_px, add_open, fill_j = None, False, -1
    pnl = 0.0
    for j in range(s, len(o)):
        # 1) add fill (resting limit; fills before the stop when both are in this bar -> pessimistic)
        if d and add_px is None and l[j] <= lvl:
            add_px = min(lvl, o[j]) + SPREAD
            add_open, fill_j = True, j
        # 2) stop
        if l[j] <= sl:
            x = min(sl, o[j])
            pnl += x - e + (m * (x - add_px) if add_open else 0.0)
            return j, pnl, add_px is not None
        # 3) recovery exits (checked on the bar's high, after the stop: pessimistic)
        # NO recovery exit on the fill bar: H1 cannot tell whether the high came after the low
        if add_open and j > fill_j and mode == "ADD_BE" and h[j] >= e:
            pnl += m * (max(e, o[j]) - add_px)
            add_open = False
        if add_open and j > fill_j and mode == "ALL_BE":
            be = (e + m * add_px) / (1 + m) + SPREAD
            if h[j] >= be:
                x = max(be, o[j])
                return j, (x - e) + m * (x - add_px), True
        # 4) max hold
        if j - s >= MAX_HOLD:
            x = o[j]
            pnl += x - e + (m * (x - add_px) if add_open else 0.0)
            return j, pnl, add_px is not None
        best = max(best, h[j])
        if np.isfinite(a[j]):
            sl = max(sl, best - TRAIL * a[j])
    x = o[-1]
    return len(o) - 1, pnl + x - e + (m * (x - add_px) if add_open else 0.0), add_px is not None


def maxdd(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min() * 100)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    g = load("XAUUSDs")
    p = pd.read_csv(ROOT / "data/v5_runs/zigzag_live_wf/probs.csv", index_col=0, parse_dates=True).iloc[:, 0]
    o, h, l = (g[x].to_numpy() for x in ("open", "high", "low"))
    a, t = atr(g).to_numpy(), g.index
    ent = entries(g, p)
    days = pd.date_range(START.normalize(), t[-1].normalize(), freq="D")

    def run(d, m, mode):
        R, F, ret = [], [], pd.Series(0.0, index=days)
        for s in ent:
            j, pnl, filled = trade(o, h, l, a, s, d, m, mode)
            risk = TRAIL * a[s - 1]
            units = 1 + (m if filled else 0)
            pnl += units * o[s] * SWAP * (t[j] - t[s]).total_seconds() / 86400 / 365
            R.append(pnl / risk)
            F.append(filled)
            ret[t[j].normalize()] += RISK * pnl / risk
        return np.array(R), np.array(F), ret[ret.index.dayofweek < 5]

    bR, _, bret = run(0.0, 0.0, "SHARED")
    half = (bret.index < bret.index[len(bret) // 2])
    rows = []
    for d, m, mode in itertools.product(DS, MS, MODES):
        R, F, ret = run(d, m, mode)
        k = bret.std() / ret.std()
        rm = ret * k                                              # matched to the base's volatility
        diff = rm - bret
        tt = diff.mean() / diff.std() * np.sqrt(len(diff))
        rows.append(dict(trigger_atr=d, size=m, mode=mode, fill_pct=round(100 * F.mean(), 1),
                         usd_total=round(R.sum() * R_USD), sharpe=round(sharpe(ret), 2),
                         maxdd=round(maxdd(ret), 1), worst_R=round(R.min(), 2),
                         usd_matched=round(rm.sum() / RISK * R_USD), t_matched=round(tt, 2),
                         h1=round(diff[half].sum() / RISK * R_USD), h2=round(diff[~half].sum() / RISK * R_USD),
                         base_R_if_filled=round(bR[F].mean(), 3) if F.any() else np.nan,
                         base_R_if_not=round(bR[~F].mean(), 3)))
    df = pd.DataFrame(rows).sort_values("t_matched", ascending=False)
    df["PASS"] = (df.t_matched >= 2) & (df.h1 > 0) & (df.h2 > 0)
    txt = (f"ZigZag Switch H1 {START:%Y-%m-%d}..{t[-1]}, {len(ent)} entries, R = ${R_USD:.0f}\n"
           f"BASE (no add): ${bR.sum() * R_USD:,.0f}, Sharpe {sharpe(bret):.2f}, maxDD {maxdd(bret):.1f}%, "
           f"worst trade {bR.min():.2f}R\n\n" + df.to_string(index=False) +
           f"\n\nPASS cells: {int(df.PASS.sum())} / {len(df)}\n"
           "usd_matched = the variant scaled to the BASE's daily volatility (same risk); t_matched = paired t of\n"
           "that against the base; h1/h2 = its $ gain over the base in each half. base_R_if_filled / _if_not =\n"
           "mean R of the ORIGINAL trade on entries where the add filled / did not (selection check).\n")
    print(txt)
    (OUT / "report.txt").write_text(txt)


if __name__ == "__main__":
    sys.exit(main())
