"""Does POOLED cross-instrument training improve the LIVE H1 bottom detector?

WHY THIS IS A REAL QUESTION AND NOT A COPY-PASTE. §3as measured a +0.0590 AUC lift from pooling
on **D1**, where gold has only 4,318 bars and ~250 decided non-overlapping events — the lift was
largely CURING A STARVED SAMPLE (§3ak's binding constraint). The live harness runs **H1**, where
gold has **67,747 bars**, roughly 15x the D1 history. A starvation cure should shrink or vanish
when the sample is no longer starved, so the D1 number must not be assumed to transfer.

DATA SAFETY. §3av found all 18 FX **D1** files are forward-stamped, leaking ret[t+1] through
`clpos`/`upwick`/`lowick` at corr 0.42-0.53. The **H1** files are CLEAN — XAUUSD 0.019,
EURUSD 0.028, GBPUSD 0.024, USDJPY 0.008 — because the defect is a daily-bar boundary artifact.
Every series is screened here anyway, and any series over 0.15 is refused rather than trusted.

PROTOCOL, identical to the D1 study so the numbers are comparable:
  GOLD-ONLY   train on gold H1, predict gold  (the incumbent, what the live harness does now)
  POOLED      train on all clean H1 series, predict gold
  TRANSFER    train on every series EXCEPT gold, predict gold
Walk-forward by calendar year, purge TOL+1 BARS at each boundary, AUC with Hanley-McNeil SE,
per-year sign count, and precision at the recall points the live threshold actually operates at.

DECISION RULE, pre-registered: upgrade the live detector ONLY if POOLED beats GOLD-ONLY by more
than 2 x SE **and** wins a majority of test years. Otherwise the harness stays gold-only and
this file records why. Detection quality is the only thing on trial here — §3x/§3as already
established the strategy has NEGATIVE EXPECTANCY, and this cannot and does not change that.

    python scripts/v5_zigzag_pooled_h1.py
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from scripts.v5_pooled_bottom_detector import auc, hm_se, intrabar_leak  # noqa: E402
from scripts.v5_xau_turning_ml import atr, features, label_near, zigzag_swings  # noqa: E402

TOL, ZZ_ORDER, ZZ_THETA = 3, 5, 1.5
LEAK_MAX = 0.15
GOLD = "XAUUSD"
# every H1 series in data/; the FX ones are clean at H1 (see the module docstring)
SERIES = ["XAUUSD_H1_long", "EURUSD_H1_long", "GBPUSD_H1_long", "USDJPY_H1_long"]


def load_h1(name: str) -> pd.DataFrame | None:
    f = ROOT / f"data/{name}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, parse_dates=["time"]).set_index("time")
    d = d.rename(columns=str.lower)
    d = d[~d.index.duplicated(keep="last")].sort_index()
    if not {"open", "high", "low", "close"} <= set(d.columns) or len(d) < 5000:
        return None
    return d


def build() -> dict:
    X, Y, S, T = [], [], [], []
    for nm in SERIES:
        d = load_h1(nm)
        sym = nm.split("_")[0]
        if d is None:
            print(f"  {sym:8s} SKIP (missing or too short)")
            continue
        lk = intrabar_leak(d)
        if lk > LEAK_MAX:
            print(f"  {sym:8s} REFUSED — intrabar leak {lk:.3f} > {LEAK_MAX}")
            continue
        F = features(d)
        if "hour" in F.columns:
            F = F.drop(columns=["hour"])          # §3p: within-day artifact on gold
        _, buys = zigzag_swings(d, ZZ_ORDER, ZZ_THETA * atr(d))
        y = label_near(np.asarray(buys, int), len(d), TOL)
        ok = F.notna().all(axis=1).values
        print(f"  {sym:8s} {int(ok.sum()):7d} bars  leak {lk:.3f}  "
              f"base rate {y[ok].mean():.3f}")
        X.append(F[ok]); Y.append(y[ok]); S.append(np.full(ok.sum(), sym))
        T.append(d.index[ok])
    return dict(X=pd.concat(X), Y=np.concatenate(Y), S=np.concatenate(S),
                T=pd.DatetimeIndex(np.concatenate([t.values for t in T])))


def prec_at_recall(p, y, rec):
    o = np.argsort(-np.asarray(p)); y = np.asarray(y)
    need = int(np.ceil(rec * y.sum()))
    hit = taken = 0
    for i in o:
        taken += 1; hit += y[i]
        if hit >= need:
            break
    return hit / max(taken, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.60)
    a = ap.parse_args()
    t0 = time.time()
    print("building the H1 panel (every series screened for intrabar leakage):")
    P = build()
    X, Y, S, T = P["X"], P["Y"], P["S"], P["T"]
    is_gold = S == GOLD
    print(f"\npooled {len(X):,} bars   gold {int(is_gold.sum()):,}   "
          f"({time.time()-t0:.0f}s)")
    print(f"NOTE: gold H1 alone has {int(is_gold.sum()):,} bars vs 4,318 on D1, so the D1 "
          f"lift\n      (+0.0590) was largely a starvation cure and may not reproduce here.\n")

    years = sorted(set(T.year))
    arms = ("GOLD-ONLY", "POOLED", "TRANSFER")
    res = {k: dict(p=[], y=[]) for k in arms}
    peryr = {}
    for yr in years:
        te = np.flatnonzero((T.year == yr) & is_gold)
        if len(te) < 500:
            continue
        past = np.flatnonzero(T.year < yr)
        if len(past) < 5000:
            continue
        # purge TOL+1 BARS off each instrument's training tail
        keep = np.ones(len(past), bool)
        for sym in np.unique(S[past]):
            idx = past[S[past] == sym]
            if len(idx):
                keep[np.isin(past, idx[-(TOL + 1):])] = False
        past = past[keep]
        sel = {"GOLD-ONLY": past[is_gold[past]], "POOLED": past,
               "TRANSFER": past[~is_gold[past]]}
        for k, tr in sel.items():
            if len(tr) < 3000 or len(np.unique(Y[tr])) < 2:
                continue
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                               max_leaf_nodes=31, min_samples_leaf=40,
                                               l2_regularization=1.0, random_state=0)
            m.fit(X.values[tr], Y[tr])
            p = m.predict_proba(X.values[te])[:, 1]
            res[k]["p"].append(p); res[k]["y"].append(Y[te])
            peryr.setdefault(yr, {})[k] = auc(p, Y[te])
        print(f"  {yr}: gold test {len(te):6d}  train gold-only {len(sel['GOLD-ONLY']):7d}"
              f"  pooled {len(sel['POOLED']):7d}   ({time.time()-t0:.0f}s)")

    print(f"\n{'arm':11s} {'n':>7s} {'AUC gold':>9s} {'SE':>7s} {'vs gold-only':>13s} "
          f"{'P@rec20':>8s} {'P@rec10':>8s} {'P@rec05':>8s}")
    base = None
    for k in arms:
        if not res[k]["p"]:
            print(f"{k:11s} no data"); continue
        p = np.concatenate(res[k]["p"]); y = np.concatenate(res[k]["y"])
        au = auc(p, y); se = hm_se(au, y.sum(), len(y) - y.sum())
        if k == "GOLD-ONLY":
            base = au
        d = "" if k == "GOLD-ONLY" else f"{au - base:+13.4f}"
        print(f"{k:11s} {len(y):7d} {au:9.4f} {se:7.4f} {d:>13s} "
              f"{prec_at_recall(p, y, 0.20):8.3f} {prec_at_recall(p, y, 0.10):8.3f} "
              f"{prec_at_recall(p, y, 0.05):8.3f}")

    print(f"\n{'year':>6s} {'GOLD':>7s} {'POOLED':>7s} {'TRANS':>7s} {'POOL-GOLD':>10s}")
    wins = n = 0
    for yr in sorted(peryr):
        d = peryr[yr]
        if "GOLD-ONLY" not in d or "POOLED" not in d:
            continue
        dd = d["POOLED"] - d["GOLD-ONLY"]
        wins += dd > 0; n += 1
        print(f"{yr:6d} {d['GOLD-ONLY']:7.3f} {d['POOLED']:7.3f} "
              f"{d.get('TRANSFER', float('nan')):7.3f} {dd:+10.3f}")
    if n:
        from math import comb
        pv = sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n
        print(f"\nPOOLED beats GOLD-ONLY in {wins}/{n} years, one-sided sign-test p = {pv:.4f}")

    p_g = np.concatenate(res["GOLD-ONLY"]["p"]); y_g = np.concatenate(res["GOLD-ONLY"]["y"])
    p_p = np.concatenate(res["POOLED"]["p"])
    se = hm_se(auc(p_g, y_g), y_g.sum(), len(y_g) - y_g.sum())
    lift = auc(p_p, y_g) - auc(p_g, y_g)
    ok = (lift > 2 * se) and (n and wins > n / 2)
    print(f"\nPRE-REGISTERED DECISION: lift {lift:+.4f} vs 2xSE {2*se:.4f}, "
          f"years {wins}/{n}")
    print(f"  -> {'UPGRADE the live detector to POOLED' if ok else 'KEEP the live detector GOLD-ONLY'}")
    print("\nThis is about DETECTION QUALITY only. §3x/§3as: the strategy has NEGATIVE")
    print("EXPECTANCY (walk-forward SR -0.76; EV/fire below the drift it displaces). A better")
    print("detector of a worthless target is still worthless — nothing here changes that.")
    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
