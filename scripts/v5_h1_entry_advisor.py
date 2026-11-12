#!/usr/bin/env python3
"""H1 ENTRY ADVISOR — a short-horizon BUY/SELL call refreshed every hour (user 2026-11-12: "split
into general Buy and Sell and specific for 6 hours and less ... no it is 4 Hours, can we get a new
one for shorter time frame for entry"). MEASUREMENT FIRST; PRE-REGISTERED.

The live advisor decides on CLOSED H4 bars (6 readings a day). This fits the same pipeline on H1:
  features  `f_base` (the advisor's BASE block) computed on H1 bars
  walk-fwd  `walk_forward` from scripts/v5_advisor_measure.py (expanding yearly refits 2018..2026,
            three-way purged split: fit / select model class / calibrate)
  scoring   `score` (accuracy vs DRIFT and PERSISTENCE, AUC, Brier skill, block-bootstrap CIs)
CELLS (12, declared before running):
  TOUCH k/h  P(price reaches +k*ATR14(H1) BEFORE -k*ATR within h hours), k 0.75/1.0/1.5, h 4/6/8.
             Path = the H1 bars strictly after the decision bar's close; a bar touching both
             levels is ambiguous and dropped (first_touch_atr convention).
  DIR h      P(close[t+h] > close[t]), h 4/6/8.
SOURCES: Switch XAUUSDs H1 2016-09.. (the feed the advisor reads live) and CME GC H1 2010.. as an
independent replication (back-adjusted, bars spanning a roll dropped).
WHAT WOULD MAKE IT SHIPPABLE AS A MEASURED CALL (all, per cell, both sources):
  1 AUC CI > 0.5 and accuracy beats DRIFT (always-the-majority) with the CI excluding 0
  2 the top and bottom reliability buckets sit on the right side of the base rate in BOTH halves
  3 family alignment null: OOS probabilities circularly shifted against the labels (200 draws),
    max |AUC - 0.5| over all 12 cells; observed best must exceed its 95th percentile
  4 control #8 (payoff): for TOUCH cells the calls in the outer buckets must clear the spread in
    ATR (Switch $0.30 round trip / H1 ATR)
Whatever the outcome, the card can show the call — but labelled with these numbers.

    python scripts/v5_h1_entry_advisor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_advisor_measure import score, walk_forward  # noqa: E402
from scripts.v5_repr_ceiling import PURGE_EXTRA, f_base  # noqa: E402
from scripts.v5_h1_features import f_base_h1  # noqa: E402
from scripts.v5_sweep_reject_h4 import ls  # noqa: E402  (liquidity-sweep loader, Switch meta)
from src.v5.advisor_labels import first_touch_atr, fwd_dir  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "h1_entry_advisor"
KS, HS = (0.75, 1.0, 1.5), (4, 6, 8)
EDGES = (0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 1.0)
N_NULL = 200
RNG = np.random.default_rng(20261112)


def load_switch() -> pd.DataFrame:
    d = ls.load("XAUUSDs", "H1")
    return d[["open", "high", "low", "close", "spr"]]


def load_gc() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data/databento/ohlcv_1h/GC.csv.gz", parse_dates=["time"]).set_index("time")
    k = d.close_adj / d.close
    out = pd.DataFrame({"open": d.open * k, "high": d.high * k, "low": d.low * k, "close": d.close_adj})
    roll = d.instrument_id.ne(d.instrument_id.shift())
    roll.iloc[0] = False
    out["spr"] = 0.30 * k
    return out[~roll]


def atr14(d: pd.DataFrame) -> pd.Series:
    pc = d.close.shift()
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def auc(y, p) -> float:
    r = pd.Series(p).rank().values
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def run(src: str, d: pd.DataFrame) -> tuple[list, list, dict]:
    X = f_base_h1(d)
    A = atr14(d)
    cells = []
    for k in KS:
        for h in HS:
            ft = first_touch_atr(d.index, d.close, A, d.high, d.low, d.index, k_atr=k, horizon_bars=h)
            y = 1.0 - ft["y"].to_numpy(dtype=float)          # 1 = UP barrier first (BUY was right)
            cells.append((f"TOUCH k{k} h{h}", y, h, k))
    for h in HS:
        y = fwd_dir(d.close, h)["y"].to_numpy(dtype=float)
        cells.append((f"DIR h{h}", y, h, None))
    rows, oos, store = [], [], {}
    cost_atr = float((d.spr / A).median())
    for name, y, h, k in cells:
        wf = walk_forward(X, y, purge=h + PURGE_EXTRA, index=d.index)
        m = wf["mask"] & np.isfinite(y)
        idx = np.flatnonzero(m)
        yy, pp = y[m], wf["p_cal"][m]
        pers = (d.close.diff(h) > 0).astype(float).to_numpy()[m]
        s = score(yy, pp, pers, wf["drift"][m], wf["clim"][m], d.index[m], h)
        T = d.index[m]
        mid = T[len(T) // 2]
        base = yy.mean()
        lo, hi = pp <= EDGES[1], pp >= EDGES[-2]
        def _rate(mask, half):
            mm = mask & ((T < mid) if half == 0 else (T >= mid))
            return yy[mm].mean() if mm.sum() >= 30 else np.nan
        # payoff of the outer calls (TOUCH cells only): +k ATR when right, -k when wrong, resolved only
        ev_buy = (k * (2 * yy[hi].mean() - 1) - cost_atr) if (k and hi.sum()) else np.nan
        ev_sell = (k * (1 - 2 * yy[lo].mean()) - cost_atr) if (k and lo.sum()) else np.nan
        rows.append(dict(src=src, cell=name, n=s["n"], base=round(base, 3), acc=round(s["acc"], 3),
                         acc_drift=round(s["acc_drift"], 3), d_drift_pp=round(s["d_drift_pp"], 2),
                         d_drift_ci=f"[{s['d_drift_ci'][0]:+.2f},{s['d_drift_ci'][1]:+.2f}]",
                         auc=round(s["auc"], 4), bss=round(s["bss"], 4),
                         buy_share=round(hi.mean(), 3), buy_hit=round(yy[hi].mean(), 3) if hi.any() else np.nan,
                         buy_h1=round(_rate(hi, 0), 3), buy_h2=round(_rate(hi, 1), 3),
                         sell_share=round(lo.mean(), 3), sell_hit=round(1 - yy[lo].mean(), 3) if lo.any() else np.nan,
                         sell_h1=round(1 - _rate(lo, 0), 3), sell_h2=round(1 - _rate(lo, 1), 3),
                         ev_buy_atr=round(ev_buy, 3), ev_sell_atr=round(ev_sell, 3), cost_atr=round(cost_atr, 3),
                         yrs_drift=s["years_drift"]))
        oos.append((yy, pp))
        store[name] = dict(y=yy, p=pp, index=T)
    return rows, oos, store


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    allrows, lines = [], ["H1 ENTRY ADVISOR — measurement (see docstring). y=1 means price went UP first / closed higher.", ""]
    for src, d in (("switch", load_switch()), ("cmeGC", load_gc())):
        rows, oos, _ = run(src, d)
        obs = np.array([abs(auc(y, p) - 0.5) for y, p in oos])
        mx = []
        for _ in range(N_NULL):
            vals = []
            for y, p in oos:
                sh = int(RNG.integers(500, len(p) - 500))
                vals.append(abs(auc(y, np.roll(p, sh)) - 0.5))
            mx.append(max(vals))
        thr = float(np.quantile(mx, 0.95))
        for r, o in zip(rows, obs):
            r["auc_null95"] = round(0.5 + thr, 4)
        best = int(np.argmax(obs))
        lines.append(f"{src}: family alignment null — best |AUC-0.5| {obs[best]:.4f} ({rows[best]['cell']}) "
                     f"vs null-max 95th pct {thr:.4f}, p {(np.array(mx) >= obs[best]).mean():.3f}")
        allrows += rows
        print("done", src, flush=True)
    R = pd.DataFrame(allrows)
    pd.set_option("display.width", 280)
    lines += ["", R.to_string(index=False), "",
              "buy = outer-top bucket (p_up >= 0.60), sell = outer-bottom (p_up <= 0.40); *_h1/_h2 = first/second half;",
              "ev_*_atr = payoff of those calls in H1-ATR after the spread (TOUCH cells, resolved windows only)."]
    txt = "\n".join(lines)
    print(txt)
    (OUT / "report.txt").write_text(txt + "\n")
    R.to_csv(OUT / "cells.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())
