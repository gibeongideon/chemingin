#!/usr/bin/env python3
"""ZigZag profit-lock LADDER and post-lock trail speed (§3dq, user request 2026-11-09).

User rule: +$100 -> lock +$50 (live), add +$200 -> lock +$80 and +$300 -> lock +$120, and "find a
sweet spot to trail the SL slowly after locking +50".
Live sizing risks ~0.25% x $141k = ~$352 per trade (3 ATR stop), so $X = X/352 R.
GRID: ladder {NONE, L1 = live [100->50], L3 = user [100->50, 200->80, 300->120]} x post-lock trail
k in {3.0 (live), 2.5, 2.0, 1.5, 1.0} x ATR (before the first lock the stop trails 3.0 ATR as live).
EVENT SET HELD FIXED (memory: breakeven-stop-refuted-and-event-set-rule): every variant exits the
SAME entries — the BASE (live) fires of §3dg, taken one-at-a-time as the live bot does under the
current rule — so differences are exit effects only. Reported: total R, mean R, win rate, Sharpe of
the daily stream, and paired t of per-trade R vs the LIVE rule (L1, k=3.0).
Resolution: H1 bars; a stop raised on bar j applies from bar j+1 (no same-bar look-ahead). The live
watcher acts each minute, so very tight post-lock trails are judged conservatively here.

    python scripts/v5_zigzag_lock_ladder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_zigzag_live_wf import MAX_HOLD, RISK, SPREAD, START, SWAP, THR, TRAIL, load, sharpe  # noqa: E402
from scripts.v5_zigzag_signal import atr  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "zigzag_lock_ladder"
R_USD = 0.0025 * 141_000
LADDERS = {"NONE": [], "L1 live": [(100, 50)], "L3 user": [(100, 50), (200, 80), (300, 120)]}
KS = (3.0, 2.5, 2.0, 1.5, 1.0)


def entries(g: pd.DataFrame, p: pd.Series) -> list[int]:
    """Entry bar indices under the LIVE rule (one position at a time, L1 lock, 3 ATR)."""
    o, h, l = (g[x].to_numpy() for x in ("open", "high", "low"))
    a, pr, t = atr(g).to_numpy(), p.reindex(g.index).to_numpy(), g.index
    out, i, n = [], 0, len(g)
    while i < n - 1:
        if t[i] >= START and np.isfinite(pr[i]) and pr[i] >= THR and np.isfinite(a[i]):
            out.append(i + 1)
            j, _ = trade(o, h, l, a, i + 1, LADDERS["L1 live"], 3.0)
            i = j
        i += 1
    return out


def trade(o, h, l, a, s, ladder, k_post):
    e = o[s] + SPREAD
    risk = TRAIL * a[s - 1]
    sl, best, locked = e - risk, e, False
    steps = [(tr / R_USD * risk, lk / R_USD * risk) for tr, lk in ladder]
    for j in range(s, len(o)):
        if l[j] <= sl:
            return j, (min(sl, o[j]) - e)
        if j - s >= MAX_HOLD:
            return j, (o[j] - e)
        best = max(best, h[j])
        for trig, lk in steps:
            if best - e >= trig:
                sl = max(sl, e + lk)
                locked = True
        k = k_post if locked else TRAIL
        if np.isfinite(a[j]):
            sl = max(sl, best - k * a[j])
    return len(o) - 1, (o[-1] - e)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    g = load("XAUUSDs")
    p = pd.read_csv(ROOT / "data/v5_runs/zigzag_live_wf/probs.csv", index_col=0, parse_dates=True).iloc[:, 0]
    o, h, l = (g[x].to_numpy() for x in ("open", "high", "low"))
    a, t = atr(g).to_numpy(), g.index
    ent = entries(g, p)
    days_idx = pd.date_range(START.normalize(), t[-1].normalize(), freq="D")
    res = {}
    for lname, lad in LADDERS.items():
        for k in KS:
            if lname == "NONE" and k != 3.0:
                continue                                   # no lock -> post-lock trail never applies
            R, ret = [], pd.Series(0.0, index=days_idx)
            for s in ent:
                j, pnl = trade(o, h, l, a, s, lad, k)
                risk = TRAIL * a[s - 1]
                pnl += o[s] * SWAP * (t[j] - t[s]).total_seconds() / 86400 / 365
                R.append(pnl / risk)
                ret[t[j].normalize()] += RISK * pnl / risk
            res[(lname, k)] = (np.array(R), ret[ret.index.dayofweek < 5])
    base = res[("L1 live", 3.0)][0]
    rows = []
    for (lname, k), (R, ret) in res.items():
        d = R - base
        tt = d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else 0.0
        rows.append(dict(ladder=lname, post_lock_trail=k, trades=len(R), win=round(100 * (R > 0).mean(), 1),
                         meanR=round(R.mean(), 4), totR=round(R.sum(), 2), usd_total=round(R.sum() * R_USD),
                         sharpe=round(sharpe(ret), 2), dR_vs_live=round(d.mean(), 4), t_vs_live=round(tt, 2)))
    df = pd.DataFrame(rows).sort_values(["ladder", "post_lock_trail"], ascending=[True, False])
    txt = (f"ZigZag Switch H1 {START:%Y-%m-%d}..{t[-1]}, {len(ent)} entries (live fires, fixed event set), "
           f"R = ${R_USD:.0f}\n" + df.to_string(index=False) +
           "\n\nt_vs_live = paired t of per-trade R against the LIVE rule (L1 +$100->+$50, 3.0 ATR trail).\n")
    print(txt)
    (OUT / "report.txt").write_text(txt)


if __name__ == "__main__":
    sys.exit(main())
