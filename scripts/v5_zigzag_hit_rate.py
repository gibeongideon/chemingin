#!/usr/bin/env python3
"""ZigZag variants ranked by HIT RATE, gain ignored (user 2026-11-12: "look for a variant with high
accuracy, ignore the gain, high percentage hit"). Research only.

GEOMETRY BUYS HIT RATE: a small target with a wide stop wins often for ANY entry on a rising asset.
So every cell is printed beside the SAME bracket on RANDOM bars (same period) — the lift
(zigzag hit − random hit) is the only part the detector contributes. EV is printed too, so the
price of each hit-rate point is visible.

Entries: ZigZag variant-B probabilities (OOS walk-forward, data/v5_runs/zigzag_long_history),
P >= thr, de-duplicated to one entry per 24h (a fixed event set, independent of the exit),
long at the next open + $0.25. Bracket: TP = tp x ATR, SL = sl x ATR (ATR of the signal bar),
time exit at `hold` bars. WIN = the trade closes with P&L > 0 (TP, or a positive time exit).
Grid: thr 0.60/0.70/0.80 x tp 0.25/0.5/1.0/1.5 x sl 1/2/3/5 x hold 24/120 = 96 cells per source.
Sources: CME GC 2012-26, Switch 2018-26 (swap −6%/yr on longs).

    python scripts/v5_zigzag_hit_rate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_zigzag_long_history import OUT as ZZ, SPREAD, SRC, load  # noqa: E402
from scripts.v5_zigzag_signal import atr  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "zigzag_hit_rate"
THRS, TPS, SLS, HOLDS = (0.60, 0.70, 0.80), (0.25, 0.5, 1.0, 1.5), (1.0, 2.0, 3.0, 5.0), (24, 120)
RNG = np.random.default_rng(20261112)


@njit(cache=True)
def run(o, h, l, a, idx, tp, sl, hold, spread):
    n = len(o)
    win = np.zeros(len(idx))
    pnl = np.zeros(len(idx))
    for k in range(len(idx)):
        i = idx[k]
        s = i + 1
        e = o[s] + spread
        up, dn = e + tp * a[i], e - sl * a[i]
        px = o[min(s + hold, n - 1)]
        for j in range(s, min(s + hold, n - 1)):
            if l[j] <= dn:
                px = min(dn, o[j])
                break
            if h[j] >= up:
                px = max(up, o[j])
                break
        pnl[k] = (px - e) / (sl * a[i])          # in R (risk = sl ATR)
        win[k] = 1.0 if px > e else 0.0
    return win, pnl


def dedupe(bars: np.ndarray, gap: int) -> np.ndarray:
    out, last = [], -10**9
    for b in bars:
        if b - last >= gap:
            out.append(b); last = b
    return np.array(out, dtype=np.int64)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for src, cfg in SRC.items():
        g = load(src, cfg["gold"])
        p = pd.read_csv(ZZ / src / "probs.csv", index_col=0, parse_dates=True).iloc[:, 0].reindex(g.index).to_numpy()
        o, h, l = (g[x].to_numpy() for x in ("open", "high", "low"))
        a = atr(g).to_numpy()
        ok = (g.index >= pd.Timestamp(cfg["start"])) & np.isfinite(a) & (np.arange(len(g)) < len(g) - 130)
        rnd = dedupe(np.sort(RNG.choice(np.flatnonzero(ok), 6000, replace=False)), 1)
        mid = g.index[np.flatnonzero(ok)[ok.sum() // 2]]
        for thr in THRS:
            ent = dedupe(np.flatnonzero(ok & np.isfinite(p) & (p >= thr)), 24)
            yrs = (g.index[ok][-1] - g.index[ok][0]).days / 365.25
            h1 = g.index[ent] < mid
            for tp in TPS:
                for sl in SLS:
                    for hold in HOLDS:
                        w, r = run(o, h, l, a, ent, tp, sl, hold, SPREAD)
                        wr, rr = run(o, h, l, a, rnd, tp, sl, hold, SPREAD)
                        rows.append(dict(src=src, thr=thr, tp_atr=tp, sl_atr=sl, hold_h=hold, n=len(ent),
                                         per_wk=round(len(ent) / yrs / 52, 2),
                                         hit=round(w.mean(), 3), hit_h1=round(w[h1].mean(), 3),
                                         hit_h2=round(w[~h1].mean(), 3), random_hit=round(wr.mean(), 3),
                                         lift_pp=round(100 * (w.mean() - wr.mean()), 1),
                                         EV_R=round(r.mean(), 3), random_EV_R=round(rr.mean(), 3)))
        print("done", src, flush=True)
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "cells.csv", index=False)
    pd.set_option("display.width", 220)
    top = A.sort_values("hit", ascending=False)
    both = A.pivot_table(index=["thr", "tp_atr", "sl_atr", "hold_h"], columns="src",
                         values=["hit", "lift_pp", "EV_R", "per_wk"]).dropna()
    both["min_hit"] = both["hit"].min(axis=1)
    both = both.sort_values("min_hit", ascending=False)
    txt = "\n".join([
        "ZigZag variants by HIT RATE (gain ignored). lift_pp = zigzag hit - same bracket on random bars.",
        "", "Top 20 cells (each source separately):", top.head(20).to_string(index=False),
        "", "Top 15 settings by the LOWER of the two sources' hit rate:", both.head(15).round(3).to_string(),
        "", "Largest LIFT over random (what the detector itself adds), min over sources:",
        both.assign(min_lift=both["lift_pp"].min(axis=1)).sort_values("min_lift", ascending=False)
            .head(10).round(3).to_string()])
    print(txt)
    (OUT / "report.txt").write_text(txt + "\n")


if __name__ == "__main__":
    sys.exit(main())
