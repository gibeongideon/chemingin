#!/usr/bin/env python3
"""ZigZag-MADL: the live ZigZag (360601) retrained on PAYOFF, borrowing the one transferable idea
of the MADL paper (Michańków et al. 2024, §3dj): optimise what the trade EARNS, weighted by how
much is at stake — not whether a label was right.

The live model learns P(bar is within 3 bars of a zigzag bottom): a payoff-blind label (control
#8: AUC is payoff-blind; §3dg: the live detector's timing equals random timing, p 0.48/0.20).
MADL = -sign(R * R_hat) * |R|. In classification form that is a log-loss on sign(R) with sample
weight |R|, where R is the REALISED R-multiple of the live trade (entry next open + spread, 3 ATR
trail, max hold 120) started at that bar. Trees train on this directly (no zero-gradient problem).

VARIANTS (pre-registered; same features, pooling, monthly walk-forward and trade rules as
`v5_zigzag_live_wf.py`):
  BASE        live: pivot label, unweighted, fire P >= 0.60          (must reproduce §3dg)
  PIVOT_W     pivot label, sample weight |R|, fire P >= 0.60
  MADL_50     label R > 0, weight |R|, fire P >= 0.50 (the paper's sign rule: E[R] > 0)
  MADL_60     same model, fire P >= 0.60
Labels at each refit are built on data truncated at the cut; the last MAX_HOLD+2 bars of every
series are purged (a trade label needs up to 120 bars of future).
CONTROLS: circular-shift timing null (200 draws, same fire count and trade rules), with and
without the +$100->+$50 lock; hold gold (same days, with swap).

    python scripts/v5_zigzag_madl.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_zigzag_live_wf import (  # noqa: E402
    MAX_HOLD, SPREAD, START, SYMS, TRAIL, load, sharpe, simulate)
from scripts.v5_zigzag_signal import LEAK_MAX, MIN_TRAIN, TOL, TRAIN_BARS, _frame, atr, intrabar_leak  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "zigzag_madl"
N_NULL, SEED = 200, 7
PURGE_R = MAX_HOLD + 2


@njit(cache=True)
def trade_r(o, h, l, a, spread):
    """Realised R of the live trade started at every bar (nan if unfinished/undefined)."""
    n = len(o)
    out = np.full(n, np.nan)
    for i in range(n - 1):
        if not np.isfinite(a[i]) or a[i] <= 0:
            continue
        e = o[i + 1] + spread
        risk = TRAIL * a[i]
        sl, best = e - risk, e
        for j in range(i + 1, n):
            if l[j] <= sl:
                out[i] = (min(sl, o[j]) - e) / risk
                break
            if j - (i + 1) >= MAX_HOLD:
                out[i] = (o[j] - e) / risk
                break
            best = max(best, h[j])
            if np.isfinite(a[j]):
                sl = max(sl, best - TRAIL * a[j])
    return out


def series_r(d: pd.DataFrame, spread: float) -> np.ndarray:
    return trade_r(d.open.to_numpy(float), d.high.to_numpy(float), d.low.to_numpy(float),
                   atr(d).to_numpy(float), spread)


def probs(gold, aux):
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415
    Fall, _, okall = _frame(gold)
    out = {k: pd.Series(np.nan, index=gold.index) for k in ("BASE", "PIVOT_W", "MADL")}
    months = pd.date_range(START, gold.index[-1] + pd.offsets.MonthBegin(1), freq="MS")
    for m0, m1 in zip(months[:-1], months[1:]):
        test = (gold.index >= m0) & (gold.index < m1)
        if not test.any():
            continue
        X, yp, yr = [], [], []
        for s, d in [("XAUUSDs", gold)] + sorted(aux.items()):
            d = d[d.index < m0]
            if len(d) < MIN_TRAIN + PURGE_R:
                continue
            F, y, ok = _frame(d)
            r = series_r(d, SPREAD if s == "XAUUSDs" else 0.0)
            hi = len(d) - max(TOL + 1, PURGE_R)          # one purge for all models: same rows
            lo = max(0, hi - TRAIN_BARS)
            msk = np.zeros(len(d), bool)
            msk[lo:hi] = True
            msk &= ok & np.isfinite(r)
            X.append(F.values[msk])
            yp.append(y[msk])
            yr.append(r[msk])
        X, yp, yr = np.vstack(X), np.concatenate(yp), np.concatenate(yr)
        w = np.abs(yr)
        rows = test & okall
        kw = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0)
        for name, (lab, wt) in {"BASE": (yp, None), "PIVOT_W": (yp, w),
                                "MADL": ((yr > 0).astype(int), w)}.items():
            clf = HistGradientBoostingClassifier(**kw).fit(X, lab, sample_weight=wt)
            out[name][rows] = clf.predict_proba(Fall.values[rows])[:, 1]
        print(f"{m0:%Y-%m} rows {len(yr):,}  R>0 {100*(yr > 0).mean():.0f}%  mean R {yr.mean():+.3f}  "
              f"fires B {int((out['BASE'][test] >= .6).sum())} PW {int((out['PIVOT_W'][test] >= .6).sum())} "
              f"M50 {int((out['MADL'][test] >= .5).sum())} M60 {int((out['MADL'][test] >= .6).sum())}",
              flush=True)
    return out


def shifted(p: pd.Series, k: int) -> pd.Series:
    """Circular shift of the OOS segment only: same values, random timing."""
    q = p.copy()
    seg = q.index >= START
    q[seg] = np.roll(q[seg].to_numpy(), k)
    return q


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gold = load("XAUUSDs")
    aux = {}
    for s in SYMS[1:]:
        d = load(s)
        if intrabar_leak(d) > LEAK_MAX:
            print("refuse", s)
            continue
        aux[s] = d
    P = probs(gold, aux)
    pd.DataFrame(P).to_csv(OUT / "probs.csv")
    live = ROOT / "data" / "v5_runs" / "zigzag_live_wf" / "probs.csv"
    lines = [f"Switch XAUUSDs H1 walk-forward {START:%Y-%m-%d}..{gold.index[-1]}, pooled {sorted(aux)}"]
    if live.exists():
        lp = pd.read_csv(live, index_col=0, parse_dates=True).iloc[:, 0]
        j = lp.dropna().index.intersection(P["BASE"].dropna().index)
        lines.append(f"BASE vs §3dg live probs: corr {np.corrcoef(lp[j], P['BASE'][j])[0, 1]:.4f}, "
                     f"max |diff| {np.abs(lp[j] - P['BASE'][j]).max():.4f}")
    rng = np.random.default_rng(SEED)
    n_oos = int((gold.index >= START).sum())
    shifts = rng.integers(24 * 20, n_oos - 24 * 20, N_NULL)
    rows = []
    for name, key, thr in (("BASE", "BASE", .6), ("PIVOT_W", "PIVOT_W", .6),
                           ("MADL_50", "MADL", .5), ("MADL_60", "MADL", .6)):
        p = (P[key] >= thr).astype(float)                              # fire mask 0/1 (NaN -> 0)
        for lock in (True, False):
            import scripts.v5_zigzag_live_wf as lw                        # simulate() reads THR
            lw.THR = 0.5
            tr, ret = simulate(gold, p, lock)
            null = np.array([sharpe(simulate(gold, shifted(p, int(k)), lock)[1]) for k in shifts])
            sh = sharpe(ret)
            eq = (1 + ret).cumprod()
            rows.append(dict(variant=name, lock=lock, trades=len(tr), win=round(100 * (tr.R > 0).mean(), 1),
                             avgR=round(tr.R.mean(), 3), totR=round(tr.R.sum(), 1),
                             ann=round(100 * (eq.iloc[-1] ** (252 / len(ret)) - 1), 2),
                             mdd=round(100 * (1 - eq / eq.cummax()).max(), 2), sharpe=round(sh, 2),
                             null_p50=round(float(np.median(null)), 2),
                             null_p95=round(float(np.percentile(null, 95)), 2),
                             p=round(float((null >= sh).mean()), 3),
                             by_year={y: round(v, 1) for y, v in tr.groupby(tr.t_out.dt.year).R.sum().items()}))
            print(rows[-1], flush=True)
    gd = gold[gold.index >= START].close.resample("D").last().dropna()
    gr = gd.pct_change().dropna() - 0.06 / 252
    lines.append(f"hold gold same days (with swap): SHARPE {sharpe(gr):+.2f}")
    txt = "\n".join(lines) + "\n" + pd.DataFrame(rows).to_string(index=False) + (
        f"\n\np = share of {N_NULL} circular-shift timing replays (same fires, same trade rules) "
        "with Sharpe >= the variant's.\n")
    print(txt)
    (OUT / "report.txt").write_text(txt)


if __name__ == "__main__":
    sys.exit(main())
