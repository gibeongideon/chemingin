#!/usr/bin/env python3
"""ZigZag LATE lock + high take-profit (§3dr, user request 2026-11-09).

§3dq: no lock at all beat every early lock (+$100 -> +$50 etc.). User: "a lock after a long win and a
TP high enough to take the best profits". Same 539 live fires, EVENT SET FIXED, R = $352, swap included.
GRID (pre-registered, 40 cells):
  lock    NONE | +$400 -> +$200 | +$600 -> +$300 | +$800 -> +$400
  TP      none | +$700 | +$1000 | +$1500 | +$2000      (limit at entry + TP; gap above fills at open)
  post-lock trail 3.0 | 2.5 ATR
Same-bar ambiguity: if a bar touches both SL and TP, SL is assumed first (pessimistic).
SELECTION CONTROL: pick the best cell on the FIRST half of trades (by entry date), report its
second-half result against NO-lock/no-TP on the same trades (paired t). Also printed: the
distribution of each trade's peak open profit (MFE), which bounds what any lock/TP can capture.

    python scripts/v5_zigzag_late_lock.py
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

OUT = ROOT / "data" / "v5_runs" / "zigzag_late_lock"
LOCKS = {"NONE": [], "400->200": [(400, 200)], "600->300": [(600, 300)], "800->400": [(800, 400)]}
TPS = (None, 700, 1000, 1500, 2000)
KS = (3.0, 2.5)


def trade(o, h, l, a, s, ladder, k_post, tp_usd):
    e = o[s] + SPREAD
    risk = TRAIL * a[s - 1]
    usd = risk / R_USD                                  # price move per $1 of P&L
    sl, best, locked, mfe = e - risk, e, False, 0.0
    tp = e + tp_usd * usd if tp_usd else np.inf
    steps = [(tr * usd, lk * usd) for tr, lk in ladder]
    for j in range(s, len(o)):
        if l[j] <= sl:
            return j, min(sl, o[j]) - e, mfe
        if h[j] >= tp:
            return j, max(tp, o[j]) - e, max(mfe, tp - e)
        if j - s >= MAX_HOLD:
            return j, o[j] - e, mfe
        best = max(best, h[j])
        mfe = best - e
        for trig, lk in steps:
            if mfe >= trig:
                sl, locked = max(sl, e + lk), True
        if np.isfinite(a[j]):
            sl = max(sl, best - (k_post if locked else TRAIL) * a[j])
    return len(o) - 1, o[-1] - e, mfe


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    g = load("XAUUSDs")
    p = pd.read_csv(ROOT / "data/v5_runs/zigzag_live_wf/probs.csv", index_col=0, parse_dates=True).iloc[:, 0]
    o, h, l = (g[x].to_numpy() for x in ("open", "high", "low"))
    a, t = atr(g).to_numpy(), g.index
    ent = entries(g, p)
    half = np.array([t[s] for s in ent]) < t[ent[len(ent) // 2]]
    days = pd.date_range(START.normalize(), t[-1].normalize(), freq="D")
    res, mfe_usd = {}, None
    for (ln, lad), tp, k in itertools.product(LOCKS.items(), TPS, KS):
        if ln == "NONE" and k != 3.0:
            continue
        R, M, ret = [], [], pd.Series(0.0, index=days)
        for s in ent:
            j, pnl, mfe = trade(o, h, l, a, s, lad, k, tp)
            risk = TRAIL * a[s - 1]
            pnl += o[s] * SWAP * (t[j] - t[s]).total_seconds() / 86400 / 365
            R.append(pnl / risk)
            M.append(mfe / risk * R_USD)
            ret[t[j].normalize()] += RISK * pnl / risk
        res[(ln, tp, k)] = (np.array(R), ret[ret.index.dayofweek < 5])
        if ln == "NONE" and tp is None:
            mfe_usd = np.array(M)
    base = res[("NONE", None, 3.0)][0]

    def pt(d):
        return d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else 0.0

    rows = []
    for (ln, tp, k), (R, ret) in res.items():
        rows.append(dict(lock=ln, tp=tp or "none", post_trail=k, win=round(100 * (R > 0).mean(), 1),
                         usd_total=round(R.sum() * R_USD), sharpe=round(sharpe(ret), 2),
                         t_vs_nolock=round(pt(R - base), 2), h1_usd=round(R[half].sum() * R_USD),
                         h2_usd=round(R[~half].sum() * R_USD)))
    df = pd.DataFrame(rows).sort_values("usd_total", ascending=False)
    # selection control: best on half 1 (excluding the baseline itself), judged on half 2
    cand = df[~((df["lock"] == "NONE") & (df["tp"] == "none"))]
    pick = cand.sort_values("h1_usd", ascending=False).iloc[0]
    key = (pick["lock"], None if pick["tp"] == "none" else pick["tp"], pick["post_trail"])
    R = res[key][0]
    d2 = (R - base)[~half]
    q = np.percentile(mfe_usd, [50, 75, 90, 95, 99])
    live = ROOT / "data/v5_runs/zigzag_lock_ladder/report.txt"
    txt = (f"ZigZag Switch H1 {START:%Y-%m-%d}..{t[-1]}, {len(ent)} entries (fixed event set), R = ${R_USD:.0f}\n"
           f"peak open profit (MFE) per trade, $: median {q[0]:.0f}, 75% {q[1]:.0f}, 90% {q[2]:.0f}, "
           f"95% {q[3]:.0f}, 99% {q[4]:.0f}; share reaching $400 {100 * (mfe_usd >= 400).mean():.1f}%, "
           f"$800 {100 * (mfe_usd >= 800).mean():.1f}%, $1500 {100 * (mfe_usd >= 1500).mean():.1f}%\n\n"
           + df.to_string(index=False)
           + f"\n\nSELECTION CONTROL: best cell on half 1 = lock {pick['lock']} / TP {pick['tp']} / trail {pick['post_trail']} "
           f"(half-1 ${pick['h1_usd']:,} vs no-lock ${int(round(base[half].sum() * R_USD)):,}); on HALF 2 it makes "
           f"${int(round(R[~half].sum() * R_USD)):,} vs no-lock ${int(round(base[~half].sum() * R_USD)):,}, "
           f"paired t {pt(d2):+.2f}\n"
           f"For reference, the live rule (ladder +100/200/300, trail 2.5) is in {live.relative_to(ROOT)}\n")
    print(txt)
    (OUT / "report.txt").write_text(txt)


if __name__ == "__main__":
    sys.exit(main())
