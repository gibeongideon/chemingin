#!/usr/bin/env python3
"""Martingale on the high-hit ZigZag variants (user 2026-11-12: "lets apply some best martingale on
this", after §3eo). Research only. PRE-REGISTERED.

TRADES: one position at a time (next entry only after the previous closes), ZigZag P >= thr, long at
the next open + $0.25, bracket TP/SL in ATR of the signal bar, 120h time exit. R = P&L / SL distance.
VARIANTS (§3eo):  V94  P>=0.60 TP 0.25 SL 5     V96  P>=0.80 TP 0.25 SL 5     V78  P>=0.80 TP 1 SL 3
SIZING (risk = fraction of CURRENT equity lost if the stop is hit; base 0.25%):
  FLAT        0.25% always
  DOUBLE Kk   x2 after each loss, back to base after a win, at most k doublings (k = 2, 3, 5)
  RECOVER     after losses, size so ONE win repays all losses since the last win plus a base win
              (the classic "recover everything"), risk capped at 25% of equity
  ANTI        x2 after each win (k <= 3), back to base after a loss (the reverse progression)
METRICS on the real sequence (CME GC 2012-26, Switch 2018-26) and on 2,000 bootstrap paths of 5
years (trades resampled iid — §3cz found win/loss signs serially independent, z +0.88):
  CAGR, maxDD, P(maxDD >= 10%) (a prop-firm wall), P(maxDD >= 50%) (ruin).

    python scripts/v5_zigzag_martingale.py
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

OUT = ROOT / "data" / "v5_runs" / "zigzag_martingale"
VARIANTS = {"V94 P.60 TP.25 SL5": (0.60, 0.25, 5.0), "V96 P.80 TP.25 SL5": (0.80, 0.25, 5.0),
            "V78 P.80 TP1 SL3": (0.80, 1.0, 3.0)}
SCHEMES = ["FLAT", "DOUBLE K2", "DOUBLE K3", "DOUBLE K5", "RECOVER", "ANTI K3"]
BASE, CAP, HOLD = 0.0025, 0.25, 120
N_BOOT = 2000
RNG = np.random.default_rng(20261112)


@njit(cache=True)
def sequential(o, h, l, a, fire, tp, sl, hold, spread):
    n = len(o)
    R = np.zeros(n)
    T = np.zeros(n, dtype=np.int64)
    m, i = 0, 0
    while i < n - hold - 2:
        if not fire[i]:
            i += 1
            continue
        s = i + 1
        e = o[s] + spread
        up, dn = e + tp * a[i], e - sl * a[i]
        j_end = s + hold
        px, j = o[j_end], j_end
        for j in range(s, j_end):
            if l[j] <= dn:
                px = min(dn, o[j])
                break
            if h[j] >= up:
                px = max(up, o[j])
                break
        R[m] = (px - e) / (sl * a[i])
        T[m] = i
        m += 1
        i = j + 1
    return R[:m], T[:m]


def equity(R: np.ndarray, scheme: str, win_R: float) -> np.ndarray:
    eq, out = 1.0, np.empty(len(R) + 1)
    out[0] = 1.0
    k, owed = 0, 0.0
    for t, r in enumerate(R):
        if scheme == "FLAT":
            f = BASE
        elif scheme.startswith("DOUBLE"):
            f = BASE * 2 ** min(k, int(scheme[-1]))
        elif scheme == "RECOVER":
            f = min(CAP, (owed + BASE * win_R) / win_R) if owed > 0 else BASE
        else:                                                   # ANTI K3
            f = BASE * 2 ** min(k, 3)
        f = min(f, CAP)
        pnl = f * r                                             # fraction of equity
        eq *= 1 + pnl
        out[t + 1] = eq
        if scheme == "ANTI K3":
            k = k + 1 if r > 0 else 0
        else:
            k = 0 if r > 0 else k + 1
        if scheme == "RECOVER":
            owed = 0.0 if r > 0 else owed - pnl
        if eq <= 0.01:
            out[t + 2:] = eq
            break
    return out


def stats(curve: np.ndarray, years: float) -> tuple[float, float]:
    dd = float((curve / np.maximum.accumulate(curve) - 1).min())
    cagr = float(curve[-1] ** (1 / years) - 1) if curve[-1] > 0 else -1.0
    return cagr, dd


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for src, cfg in SRC.items():
        g = load(src, cfg["gold"])
        p = pd.read_csv(ZZ / src / "probs.csv", index_col=0, parse_dates=True).iloc[:, 0].reindex(g.index).to_numpy()
        o, h, l = (g[x].to_numpy() for x in ("open", "high", "low"))
        a = atr(g).to_numpy()
        ok = (g.index >= pd.Timestamp(cfg["start"])) & np.isfinite(a)
        years = (g.index[-1] - pd.Timestamp(cfg["start"])).days / 365.25
        for vname, (thr, tp, sl) in VARIANTS.items():
            R, T = sequential(o, h, l, a, ok & np.isfinite(p) & (p >= thr), tp, sl, HOLD, SPREAD)
            win_R = tp / sl
            per_yr = len(R) / years
            n5 = max(int(round(per_yr * 5)), 5)
            boot = RNG.choice(R, size=(N_BOOT, n5), replace=True)
            # longest losing streak in the real sequence
            ls, cur = 0, 0
            for r in R:
                cur = cur + 1 if r <= 0 else 0
                ls = max(ls, cur)
            for sc in SCHEMES:
                cg, dd = stats(equity(R, sc, win_R), years)
                bdd = np.array([stats(equity(b, sc, win_R), 5.0) for b in boot])
                rows.append(dict(src=src, variant=vname, scheme=sc, trades=len(R), per_yr=round(per_yr, 1),
                                 hit=round((R > 0).mean(), 3), meanR=round(R.mean(), 4), max_loss_streak=ls,
                                 CAGR_pct=round(100 * cg, 2), maxDD_pct=round(100 * dd, 1),
                                 boot_med_CAGR=round(100 * float(np.median(bdd[:, 0])), 2),
                                 P_DD10=round(float((bdd[:, 1] <= -0.10).mean()), 3),
                                 P_DD50=round(float((bdd[:, 1] <= -0.50).mean()), 3)))
        print("done", src, flush=True)
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "cells.csv", index=False)
    pd.set_option("display.width", 220)
    txt = ("Martingale on high-hit ZigZag variants (see docstring). Real sequence + 2,000 bootstrap 5-yr paths.\n\n"
           + A.to_string(index=False) + "\n")
    print(txt)
    (OUT / "report.txt").write_text(txt)


if __name__ == "__main__":
    sys.exit(main())
