#!/usr/bin/env python3
"""GHOST-LIMIT REFUSAL — the adverse-selection law (§3bc) turned from a cost into a signal.

§3bc measured that a pullback limit FILLS on the trades that go on to lose (base|FILLED -0.20%)
and is SKIPPED on the ones that work (base|SKIPPED +0.22..+0.35%, t up to -8.60). Every strategy
treats that as a cost of limit orders. This asks whether the NON-FILL is itself information:

  PROCEDURE (long-only; drift assets; pre-registered, nothing tuned)
    at each server-day close D, rest a GHOST buy limit at close - K x D1 ATR14 (never sent)
    watch the next N H1 bars
    if the ghost never fills (the market REFUSES to discount) -> buy at market at bar N+1 open
    exit: hold H days at the close (H in {1, 5}); spread charged once.
  grid: K in {0.5, 1.0} x N in {4, 8} x H in {1, 5} = 8 cells per asset, 3 assets = 24 cells.

  THE CONTROL THAT KILLED "AT THE N-DAY HIGH" (window-extremes-are-tautology): a refusal mostly
  means the window went UP. So the signal must beat ALL day-close entries at the SAME bar N+1
  WITH THE SAME WINDOW RETURN: entries are bucketed by window return (deciles of ret/ATR) and the
  excess is refused - all, within bucket, averaged over the refused set's bucket mix.
  Also printed: unconditional drift (all entries) and the FILLED set.
  Null: max over 24 cells of the matched excess t vs 300 draws where refusal labels are
  permuted WITHIN return buckets (keeps the momentum, destroys the refusal information).

    python scripts/v5_ghost_limit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sweep_switch"
OUT = ROOT / "data" / "v5_runs" / "ghost_limit"
POINT = {"XAUUSDs": 0.01, "BTCUSDs": 0.01, "USTECs": 0.1}
KS, NS, HS = (0.5, 1.0), (4, 8), (1, 5)
N_NULL, SEED = 300, 11


def load(sym: str) -> pd.DataFrame:
    d = pd.read_csv(DATA / f"{sym}_H1.csv", parse_dates=["time"]).set_index("time").sort_index()
    d["spr"] = d["spread"] * POINT[sym]
    return d[~d.index.duplicated()]


def build(h1: pd.DataFrame) -> pd.DataFrame:
    """One row per server day: decision at that day's LAST H1 close."""
    day = h1.index.normalize()
    pos = np.arange(len(h1))
    last = pd.Series(pos, index=h1.index).groupby(day).max().to_numpy()
    d1 = h1.groupby(day).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    pc = d1.close.shift(1)
    tr = pd.concat([d1.high - d1.low, (d1.high - pc).abs(), (d1.low - pc).abs()],
                   axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
    return pd.DataFrame({"i": last, "close": d1.close.to_numpy(), "atr": atr},
                        index=d1.index).iloc[20:]


def cell(h1, days, K, N, H):
    o, l, c, s = (h1[x].to_numpy() for x in ("open", "low", "close", "spr"))
    idx = h1.index
    rows = []
    for r in days.itertuples():
        i = r.i
        if i + N + 1 >= len(h1) or not np.isfinite(r.atr):
            continue
        ghost = r.close - K * r.atr
        refused = l[i + 1:i + N + 1].min() > ghost
        e_i = i + N + 1
        entry = o[e_i]
        tgt = idx[e_i] + pd.Timedelta(days=H)
        x_i = min(np.searchsorted(idx, tgt) - 1, len(h1) - 1)
        if x_i <= e_i:
            continue
        ret = (c[x_i] - entry - s[e_i]) / entry * 1e4                    # net bp
        wret = (o[e_i] - r.close) / r.atr                                 # window return, ATR
        rows.append((idx[e_i], refused, wret, ret))
    t = pd.DataFrame(rows, columns=["t", "refused", "wret", "ret"])
    t["b"] = pd.qcut(t.wret, 10, labels=False, duplicates="drop")
    return t


def matched_excess(t: pd.DataFrame, lab: np.ndarray) -> tuple[float, float]:
    """mean over refused rows of (ret - mean ret of ALL rows in the same bucket); t-stat."""
    mb = t.groupby("b")["ret"].transform("mean").to_numpy()
    x = (t.ret.to_numpy() - mb)[lab]
    if len(x) < 10:
        return float("nan"), float("nan")
    return float(x.mean()), float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows, store = [], []
    for sym in POINT:
        h1 = load(sym)
        days = build(h1)
        for K in KS:
            for N in NS:
                for H in HS:
                    t = cell(h1, days, K, N, H)
                    lab = t.refused.to_numpy()
                    ex, tt = matched_excess(t, lab)
                    yrs = t[lab].assign(y=pd.to_datetime(t[lab].t).dt.year)
                    mb = t.groupby("b")["ret"].transform("mean")
                    yy = (t.ret - mb)[lab].groupby(yrs.y).mean()
                    half = lab.sum() // 2
                    xr = (t.ret - mb)[lab].to_numpy()
                    rows.append(dict(sym=sym, K=K, N=N, H=H, n_ref=int(lab.sum()),
                                     refuse_rate=round(lab.mean(), 3),
                                     all_bp=round(t.ret.mean(), 2),
                                     refused_bp=round(t.ret[lab].mean(), 2),
                                     filled_bp=round(t.ret[~lab].mean(), 2),
                                     matched_excess_bp=round(ex, 2), t=round(tt, 2),
                                     halves=f"{xr[:half].mean():+.1f}/{xr[half:].mean():+.1f}",
                                     yrs_pos=f"{int((yy > 0).sum())}/{len(yy)}"))
                    store.append((t, lab))
    res = pd.DataFrame(rows)
    obs = np.nanmax(res.t.to_numpy())
    null = []
    for _ in range(N_NULL):
        best = -np.inf
        for t, lab in store:
            perm = lab.copy()
            for b, g in t.groupby("b").groups.items():
                g = np.asarray(g)
                perm[g] = rng.permutation(lab[g])
            best = max(best, matched_excess(t, perm)[1])
        null.append(best)
    null = np.array(null)
    p = float((null >= obs).mean())
    pd.set_option("display.width", 200, "display.max_columns", 20)
    txt = res.to_string(index=False) + (
        f"\n\nbest matched t {obs:+.2f}  vs within-bucket permutation max-null "
        f"p50 {np.median(null):+.2f} p95 {np.percentile(null, 95):+.2f}  ->  p = {p:.3f}\n")
    print(txt)
    (OUT / "report.txt").write_text(txt)


if __name__ == "__main__":
    sys.exit(main())
