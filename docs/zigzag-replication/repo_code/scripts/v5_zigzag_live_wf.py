#!/usr/bin/env python3
"""Walk-forward Sharpe of the LIVE ZigZag harness (magic 100001) on broker, 2024-01 .. now.

Replays `configs/zigzag_variant_b.example.json` as deployed, reusing the live code's own
`_frame` (features + zigzag label) and the same HistGradientBoosting settings:
  * POOLED model: gold + EURUSDs/GBPUSDs/USDJPYs H1, trailing 20,000 bars per series,
    TOL+1 purge, aux series truncated at the cut. Labels are rebuilt on data TRUNCATED at each
    refit, as the live job does, so the zigzag never sees bars after the cut.
  * refit MONTHLY (live refits hourly: a small approximation, it can only make the model staler)
  * fire when P(bottom) >= 0.60 on a closed H1 bar -> BUY at the next bar's open + $0.25 spread
  * stop 3.0 x ATR14(H1) below entry, ratcheted 3.0 ATR behind the highest high (on bar close)
  * lock: once the peak is +0.29R, floor the stop at +0.146R (the live +$100 -> +$50 rule at
    0.25% risk on the account), toggle with --no-lock
  * max hold 120 bars, one position at a time, long swap -6%/yr
  Equity at the live 0.25% risk; Sharpe from daily returns, sqrt(252). Benchmark: holding gold
  (with the same swap) over the same days.

    python scripts/v5_zigzag_live_wf.py --fetch     # on the server, under a bridge lock
    python scripts/v5_zigzag_live_wf.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_zigzag_signal import (  # noqa: E402
    MIN_TRAIN, TOL, TRAIN_BARS, _frame, atr, intrabar_leak, LEAK_MAX)

OUT = ROOT / "data" / "v5_runs" / "zigzag_live_wf"
SYMS = ("XAUUSDs", "EURUSDs", "GBPUSDs", "USDJPYs")
START = pd.Timestamp("2024-01-01")
THR, TRAIL, MAX_HOLD, SPREAD, SWAP = 0.60, 3.0, 120, 0.25, -0.06
RISK = 0.0025
LOCK_AT, LOCK_FLOOR = 0.29, 0.146


def fetch(port: int) -> None:
    from mt5linux import MetaTrader5  # noqa: PLC0415
    m = MetaTrader5(host="localhost", port=port)
    if not m.initialize():
        sys.exit(f"init failed {m.last_error()}")
    OUT.mkdir(parents=True, exist_ok=True)
    for s in SYMS:
        m.symbol_select(s, True)
        r, n = None, 60000
        while n >= 5000 and (r is None or len(r) == 0):
            r = m.copy_rates_from_pos(s, 16385, 1, n)          # pos 1: closed bars only
            n //= 2
        a = np.array([tuple(x)[:5] for x in r], float)
        d = pd.DataFrame(a[:, 1:], columns=["open", "high", "low", "close"],
                         index=pd.to_datetime(a[:, 0], unit="s"))
        d.index.name = "time"
        d.to_csv(OUT / f"{s}_H1.csv")
        print(s, len(d), d.index[0], "->", d.index[-1])
    # NO m.shutdown(): bridge 18000 is shared.


def load(s: str) -> pd.DataFrame:
    d = pd.read_csv(OUT / f"{s}_H1.csv", parse_dates=["time"]).set_index("time")
    return d[~d.index.duplicated()].sort_index()


def probs(gold: pd.DataFrame, aux: dict) -> pd.Series:
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415
    Fall, _, okall = _frame(gold)
    out = pd.Series(np.nan, index=gold.index)
    months = pd.date_range(START, gold.index[-1] + pd.offsets.MonthBegin(1), freq="MS")
    purge = TOL + 1
    for m0, m1 in zip(months[:-1], months[1:]):
        test = (gold.index >= m0) & (gold.index < m1)
        if not test.any():
            continue
        Xs, ys = [], []
        for s, d in [("XAUUSDs", gold)] + sorted(aux.items()):
            d = d[d.index < m0]                                   # nothing from the test month
            if len(d) < MIN_TRAIN + purge:
                continue
            F, y, ok = _frame(d)                                  # labels built on truncated data
            hi = len(d) - purge
            lo = max(0, hi - TRAIN_BARS)
            msk = np.zeros(len(d), bool)
            msk[lo:hi] = True
            msk &= ok
            Xs.append(F.values[msk])
            ys.append(y[msk])
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                             max_leaf_nodes=31, random_state=0)
        clf.fit(np.vstack(Xs), np.concatenate(ys))
        rows = test & okall
        out[rows] = clf.predict_proba(Fall.values[rows])[:, 1]
        print(f"{m0:%Y-%m} fit on {sum(len(y) for y in ys):,} rows; fires "
              f"{int((out[test] >= THR).sum())}", flush=True)
    return out


def simulate(g: pd.DataFrame, p: pd.Series, lock: bool) -> tuple[pd.DataFrame, pd.Series]:
    o, h, l, c = (g[x].to_numpy() for x in ("open", "high", "low", "close"))
    a = atr(g).to_numpy()
    pr = p.to_numpy()
    t = g.index
    trades, pos = [], None
    for i in range(len(g) - 1):
        if pos is not None:
            j = i
            exit_px = None
            if l[j] <= pos["sl"]:
                exit_px = min(pos["sl"], o[j])                    # gap through the stop
            elif j - pos["i"] >= MAX_HOLD:
                exit_px = o[j]
            if exit_px is not None:
                days = (t[j] - pos["t"]).total_seconds() / 86400
                pnl = exit_px - pos["e"] + pos["e"] * SWAP * days / 365
                trades.append((pos["t"], t[j], pnl / pos["risk"]))
                pos = None
            else:
                pos["best"] = max(pos["best"], h[j])
                pos["sl"] = max(pos["sl"], pos["best"] - TRAIL * a[j])
                if lock and pos["best"] - pos["e"] >= LOCK_AT * pos["risk"]:
                    pos["sl"] = max(pos["sl"], pos["e"] + LOCK_FLOOR * pos["risk"])
        if pos is None and t[i] >= START and np.isfinite(pr[i]) and pr[i] >= THR \
                and np.isfinite(a[i]):
            e = o[i + 1] + SPREAD
            risk = TRAIL * a[i]
            pos = dict(i=i + 1, t=t[i + 1], e=e, risk=risk, sl=e - risk, best=e)
    tr = pd.DataFrame(trades, columns=["t_in", "t_out", "R"])
    days = pd.date_range(START.normalize(), t[-1].normalize(), freq="D")
    ret = pd.Series(0.0, index=days)
    for r in tr.itertuples():
        ret[r.t_out.normalize()] += RISK * r.R
    ret = ret[ret.index.dayofweek < 5]
    return tr, ret


def sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--port", type=int, default=18000)
    a = ap.parse_args()
    if a.fetch:
        fetch(a.port)
        return
    gold = load("XAUUSDs")
    aux = {}
    for s in SYMS[1:]:
        d = load(s)
        lk = intrabar_leak(d)
        if lk > LEAK_MAX:
            print(f"refuse {s}: leak {lk:.3f}")
            continue
        aux[s] = d
    p = probs(gold, aux)
    p.to_csv(OUT / "probs.csv")
    lines = [f"broker XAUUSDs H1, walk-forward {START:%Y-%m-%d} -> {gold.index[-1]}, "
             f"pooled with {sorted(aux)}"]
    for lock in (True, False):
        tr, ret = simulate(gold, p, lock)
        eq = (1 + ret).cumprod()
        dd = (1 - eq / eq.cummax()).max()
        yrs = len(ret) / 252
        by = tr.groupby(tr.t_out.dt.year).R.agg(["count", "sum"]).round(1).to_dict("index")
        lines.append(f"{'with' if lock else 'without'} +$100->+$50 lock: trades {len(tr)}, "
                     f"win {100*(tr.R > 0).mean():.0f}%, avg {tr.R.mean():+.3f}R, total "
                     f"{tr.R.sum():+.1f}R -> {100*(eq.iloc[-1]-1):+.2f}% at 0.25% risk "
                     f"({100*(eq.iloc[-1]**(1/yrs)-1):+.2f}%/yr), maxDD {100*dd:.2f}%, "
                     f"SHARPE {sharpe(ret):+.2f}   by year {by}")
    gd = gold[gold.index >= START].close.resample("D").last().dropna()
    gr = gd.pct_change().dropna() + SWAP / 252
    lines.append(f"hold gold same days (with swap): {100*((1+gr).prod()-1):+.1f}%, "
                 f"SHARPE {sharpe(gr):+.2f}")
    txt = "\n".join(lines)
    print(txt)
    (OUT / "report.txt").write_text(txt + "\n")


if __name__ == "__main__":
    sys.exit(main())
