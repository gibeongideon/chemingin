#!/usr/bin/env python3
"""OCTOBER A6 — re-price the intraday closures at the LIVE 0.504bp spread.

Pre-registered in data/v5_runs/october/PREREGISTRATION.md. Design in OCTOBER.MD.

WHY. §3bx measured the live Switch quote at **0.504bp** against the **1.5bp one-way floor** that
BEST_FINDINGS #7's cost-wall table is built on, taking M15's breakeven hit rate from **0.6363 to
0.5343**. §3bx then re-priced exactly TWO of roughly fifteen intraday closures (the ORB, which died
of a separate null bug, and the overnight decomposition). **Every other intraday verdict in this
file was decided at 1.5-2.34bp.** §3bx's own closing line says the revision *"widens what is worth
testing for non-trend structures"* -- and then it tested two.

TWO PARTS, because arithmetic alone would be dishonest:
  PART 1 re-nets the recorded figures. This is only valid where the file records BOTH a net number
  and the cost basis it assumed; where it does not, the row says so rather than guessing. A sign
  flip here REOPENS a question, it does not answer one.
  PART 2 re-runs, from the data, the one intraday structure that actually survived its own tests:
  §3bj's generic prior-day high/low FIRST BREAK, exit at close -- *"+2.06bp net over 2,430 trades at
  Sharpe +0.47"*, which §3bj explicitly called a measurement rather than a strategy. It is the
  natural beneficiary of a 3x cheaper round trip, and it is cheap to re-measure properly.

WHAT A FLIP DOES NOT BUY. §3bx-D showed how an intraday "survivor" can be a harness artefact: the
ORB's `p 0.0000` came from a deterministic cost offset sitting inside a zero-mean null while the
statistic was `abs(t)`. So any cell that turns positive here must still clear a null run on GROSS
pricing the SIGNED maximum before it counts as reopened.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "v5_runs" / "october"
LIVE_RT = 0.504 * 1.5          # 0.756bp — the live round trip
RNG = np.random.default_rng(20261029)

# (finding, structure, recorded NET bp, the cost basis it was decided at, note)
RECORDED = [
    ("3bi #6", "round-number stop runs",        -3.03, 2.34, "net bp recorded"),
    ("3bi #7", "Monday gap fill",               -1.75, 2.34, "net bp recorded"),
    ("3bj",    "prior-day first break (generic)", 2.06, 2.34, "THE survivor — re-run in part 2"),
    ("3be",    "session ORB (best cell)",         1.50, 2.34, "midpoint of the filed 1-2bp; "
                                                              "basis destroyed by 3bx-D's null bug"),
    ("3bi #5", "gold vs DXY+yields divergence", -12.45, 2.34, "net bp recorded"),
]
# Closures whose verdict is NOT a net-bp figure and therefore cannot be re-netted arithmetically.
NOT_RENETTABLE = [
    ("1",      "M15 next-bar fade", "breakeven spread $0.04-0.07 vs a $0.34 live spread — the gap "
                                    "is ~5x, not ~3x, so 0.504bp does not reach it"),
    ("3c/3f",  "intraday TREND runner", "already tested AT $0.12, tighter than the live quote: H4 "
                                        "still won 1.27 vs M30 0.93 / H1 0.98. Signal quality, not "
                                        "cost — re-pricing cannot touch it"),
    ("3bf",    "08:30 ET release trigger", "0/72 cells; verdict was a signed t of +1.39 against a "
                                           "null median of +2.32, i.e. a NULL comparison, not a "
                                           "cost comparison"),
    ("3bc",    "M15 entry timing on ICT zones", "killed by ADVERSE SELECTION (t -2.76 to -8.60) and "
                                                "by a flat 0.5-ATR limit beating every zone; costs "
                                                "cancel in the paired difference"),
    ("3bj",    "NR7 first break on M15", "+0.43 against a NON-NR7 control of +0.47 — a control "
                                         "comparison, so a cheaper spread lifts both equally"),
    ("3bx-B",  "overnight decomposition", "already re-priced at 0.76bp by 3bx: net +1.07%/yr "
                                          "against buy-and-hold's +11.96%"),
]


def load(tf: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / f"data/XAUUSD_{tf}_long.csv", parse_dates=["time"], index_col="time")
    return d[~d.index.duplicated(keep="last")].sort_index()


def prior_day_break(d: pd.DataFrame, cost_bp: float) -> dict:
    """§3bj's structure: first break of the prior day's high or low, exit at the session close.

    Prior-day levels are strictly causal — built from COMPLETED days only, shifted one day.
    """
    day = d.index.normalize()
    hi = d["high"].groupby(day).max().shift(1)
    lo = d["low"].groupby(day).min().shift(1)
    pdh = hi.reindex(day).to_numpy(float)
    pdl = lo.reindex(day).to_numpy(float)
    h, l, c = (d[k].to_numpy(float) for k in ("high", "low", "close"))
    nd = day.to_numpy()

    rows = []
    start = 0
    for i in range(1, len(d) + 1):
        if i == len(d) or nd[i] != nd[start]:
            sl = slice(start, i)
            if np.isfinite(pdh[start]) and np.isfinite(pdl[start]):
                up = np.flatnonzero(h[sl] > pdh[start])
                dn = np.flatnonzero(l[sl] < pdl[start])
                fu = up[0] if len(up) else 10**9
                fd = dn[0] if len(dn) else 10**9
                if min(fu, fd) < 10**9:
                    side = 1 if fu < fd else -1
                    entry = pdh[start] if side == 1 else pdl[start]
                    exit_ = c[i - 1]
                    rows.append(side * (exit_ / entry - 1.0) * 1e4 - cost_bp)
            start = i
    r = np.array(rows)
    if len(r) < 50:
        return dict(n=len(r))
    return dict(n=len(r), mean_bp=float(r.mean()), hit=float((r > 0).mean()),
                t=float(r.mean() / (r.std() / np.sqrt(len(r)))),
                sharpe=float(r.mean() / r.std() * np.sqrt(252)))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 104)
    print("OCTOBER A6 — RE-PRICING THE INTRADAY CLOSURES AT THE LIVE 0.504bp")
    print("=" * 104)
    print(f"  filed cost basis 2.34bp round trip  ->  live {LIVE_RT:.3f}bp  "
          f"= every net figure improves by {2.34 - LIVE_RT:+.3f}bp")

    print(f"\n  PART 1 — arithmetic re-netting (valid ONLY where a net bp AND its basis are filed)")
    print(f"  {'§':<9} {'structure':<34} {'filed net':>10} {'re-netted':>10}  verdict")
    print("  " + "-" * 92)
    rows = []
    for sec, name, net, basis, note in RECORDED:
        new = net + (basis - LIVE_RT)
        flip = (net < 0) and (new > 0)
        rows.append(dict(section=sec, structure=name, filed_net=net, basis=basis,
                         renetted=new, flips=flip))
        v = "*** FLIPS POSITIVE ***" if flip else ("already positive" if net > 0 else "still negative")
        print(f"  {sec:<9} {name:<34} {net:>+10.2f} {new:>+10.2f}  {v}")
    nflip = sum(r["flips"] for r in rows)
    print(f"\n  cells that flip sign on cost alone: {nflip} of {len(rows)}")

    print(f"\n  PART 1b — closures that CANNOT be re-netted arithmetically, and why")
    for sec, name, why in NOT_RENETTABLE:
        print(f"    §{sec:<8} {name:<32} {why}")

    print(f"\n  PART 2 — re-running §3bj's prior-day FIRST BREAK from the data at several costs")
    print(f"  (the one intraday structure that survived its own tests: filed +2.06bp over 2,430)")
    for tf in ("H1", "M15"):
        d = load(tf)
        print(f"\n    --- {tf}: {len(d):,} bars {d.index[0].date()} -> {d.index[-1].date()} ---")
        print(f"    {'cost bp':>9} {'trades':>8} {'mean bp':>9} {'hit':>8} {'t':>8} {'Sharpe':>9}")
        for cost in (2.34, 1.50, LIVE_RT, 0.0):
            r = prior_day_break(d, cost)
            if r.get("mean_bp") is None:
                print(f"    {cost:>9.3f} {r['n']:>8}  too few"); continue
            tag = "  <- LIVE" if abs(cost - LIVE_RT) < 1e-9 else ("  <- GROSS" if cost == 0 else "")
            print(f"    {cost:>9.3f} {r['n']:>8} {r['mean_bp']:>+9.3f} {r['hit']:>8.4f} "
                  f"{r['t']:>+8.2f} {r['sharpe']:>+9.3f}{tag}")
    pd.DataFrame(rows).to_csv(OUT / "a6_reprice.csv", index=False)
    print(f"""
  WHAT A FLIP BUYS, AND WHAT IT DOES NOT. A sign change on cost alone REOPENS a question; it does
  not answer one. §3bx-D is the cautionary case: the ORB's "SURVIVES p 0.0000" came from a
  deterministic cost offset inside a zero-mean null while the statistic was abs(t), and the honest
  re-run put it at p 0.8425 with the best cell BELOW its own null's median. Any cell above must
  clear a null run on GROSS pricing the SIGNED maximum before it is treated as live.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
