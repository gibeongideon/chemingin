"""ARM 1 stage 3 — Z5 (max-statistic null) and Z7 (PBO) for the ZigZag ATR bracket.

TWO METHOD RULES BIND HERE, BOTH LEARNED THE HARD WAY (SS3bf):

  * BOOTSTRAP **GROSS**, NOT NET. A deterministic cost offset inside a zero-mean null
    manufactures significance in proportion to sample size, always negative -- it produced
    "SURVIVES p 0.0000" twice on studies that were in fact dead. Costs are a hurdle applied to
    the point estimate, never a random quantity inside the bootstrap.
  * PRICE THE **SIGNED** MAXIMUM, not the absolute one. A deployable cell has to beat the best
    POSITIVE cell a no-edge dataset hands you; ranking on |t| lets a significantly negative cell
    pose as a survivor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from v5_xau_turning_ml import zigzag_swings, atr                          # noqa: E402
from v5_zigzag_atr_bracket import load, COST_USD, ZZ_ORDER, ZZ_THETA, MAX_HOLD_H1  # noqa: E402

OUT = ROOT / "data/v5_runs/zigzag_atr"
rng = np.random.default_rng(11)
DRAWS, BLOCK = 2000, 60      # block >= the 48-bar holding period, per SS3ak


def main() -> int:
    st = np.load(OUT / "_outcomes.npy", allow_pickle=True).item()
    h1 = load("H1"); close = h1["close"].values
    p = pd.read_csv(OUT / "oos_prob_h1.csv", parse_dates=["time"],
                    index_col="time")["p"].reindex(h1.index).values
    scored = np.isfinite(p)
    fires = np.flatnonzero(scored & (p >= 0.60))
    cells = [c for c in st if c.startswith("ATR")]

    # ---------- per-cell GROSS excess over what any bar earns with the same bracket
    series = {}
    for c in cells:
        r = st[c]
        f = fires[np.isfinite(r["exit_px"][fires])]
        allb = np.flatnonzero(scored & np.isfinite(r["exit_px"]))
        gross_f = (r["exit_px"][f] - close[f]) / close[f]          # GROSS: no cost inside
        gross_b = float(np.mean((r["exit_px"][allb] - close[allb]) / close[allb]))
        series[c] = gross_f - gross_b
    cost_bp = 1e4 * COST_USD / float(np.median(close))

    print("=" * 92)
    print("Z5 -- MAX-STATISTIC NULL on GROSS excess, signed maximum, best of "
          f"{len(cells)} declared cells")
    print(f"       stationary block bootstrap, block {BLOCK} >= the {MAX_HOLD_H1}-bar hold, "
          f"{DRAWS} draws")
    print("=" * 92)

    obs = {}
    for c, s in series.items():
        t = float(np.mean(s) / np.std(s) * np.sqrt(len(s))) if np.std(s) > 0 else 0.0
        obs[c] = t
    best_cell = max(obs, key=obs.get)
    obs_max = obs[best_cell]

    n = min(len(s) for s in series.values())
    nulls = np.empty(DRAWS)
    arr = {c: np.asarray(s, float) for c, s in series.items()}
    for d in range(DRAWS):
        starts = rng.integers(0, n - BLOCK, size=n // BLOCK + 1)
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
        ts = []
        for c in cells:
            x = arr[c][:n][idx]
            x = x - arr[c][:n].mean()             # impose the zero-mean null
            ts.append(x.mean() / x.std() * np.sqrt(len(x)) if x.std() > 0 else 0.0)
        nulls[d] = max(ts)                        # the SIGNED maximum
    pval = float(np.mean(nulls >= obs_max))

    for c in sorted(obs, key=obs.get, reverse=True):
        print(f"  {c:<20} gross excess {1e4*np.mean(series[c]):+6.2f}bp   t {obs[c]:+6.2f}"
              f"   net {1e4*np.mean(series[c]) - cost_bp:+6.2f}bp")
    print(f"\n  round trip                              {cost_bp:.2f}bp   "
          "(the hurdle, applied to the point estimate -- NOT inside the bootstrap)")
    print(f"\n  observed signed max t  {obs_max:+.3f}   ({best_cell})")
    print(f"  null of the max:  p50 {np.percentile(nulls,50):+.3f}   "
          f"p95 {np.percentile(nulls,95):+.3f}   p99 {np.percentile(nulls,99):+.3f}")
    print(f"  Reality-Check p = {pval:.4f}   ->  "
          f"{'SURVIVES' if pval <= 0.05 else 'DOES NOT SURVIVE'}")

    # ---------- Z7 PBO over the declared grid
    print("\n" + "=" * 92)
    print("Z7 -- PBO (combinatorially symmetric cross-validation) over the 6-cell grid")
    print("=" * 92)
    from src.evaluation.dsr_pbo import pbo_cscv
    m = min(len(s) for s in series.values())
    M = np.column_stack([np.asarray(series[c], float)[:m] for c in cells])
    res = pbo_cscv(M, n_partitions=10)
    print(f"  PBO {res.pbo:.3f} over {res.n_splits} splits   bar is < 0.30   "
          f"->  {'PASS' if res.pbo < 0.30 else 'FAIL'}")

    # ---------- the specificity peak: is thr 0.75 real?
    print("\n" + "=" * 92)
    print("The thr=0.75 peak in the Z5 sweep -- significance, before anyone gets excited")
    print("=" * 92)
    r = st["ATR tp1.0/sl1.0"]
    allb = np.flatnonzero(scored & np.isfinite(r["exit_px"]))
    base = float(np.mean((r["exit_px"][allb] - close[allb] - COST_USD) / close[allb]))
    for t in (0.70, 0.75, 0.80):
        f = np.flatnonzero(scored & (p >= t) & np.isfinite(r["exit_px"]))
        x = (r["exit_px"][f] - close[f] - COST_USD) / close[f] - base
        last, keep = -10**9, []
        for i in f:
            if i - last >= MAX_HOLD_H1:
                keep.append(np.where(f == i)[0][0]); last = i
        xn = x[keep]
        tt = float(np.mean(xn) / np.std(xn) * np.sqrt(len(xn))) if np.std(xn) > 0 else 0.0
        print(f"  thr {t:.2f}:  {len(f):>5} fires -> {len(xn):>4} NON-OVERLAPPING   "
              f"excess {1e4*np.mean(xn):+6.2f}bp   t {tt:+.2f}   "
              f"{'' if abs(tt) >= 2 else '(indistinguishable from zero)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
