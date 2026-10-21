#!/usr/bin/env python3
"""The user's spec, built and measured: a continuous P(up) that HOLDS while confident and FLIPS.

THE ASK, verbatim in substance: "a prediction of the market going up or down with probability...
it keeps firing the probability as long as it is high enough, it keeps the trade on; the opposite
trigger should close the trade and create the opposite trade."

That is a well-defined object -- a calibrated probability plus a HYSTERESIS band -- and it has
never been tested here, because the live advisor (3az/3ba) is warn-only and the champion is
LONG-ONLY by construction (`champion_recipe` ends in `.clip(0, 2)`). So the flip-to-short half of
the spec is exactly what `kill-the-shorts` killed, and the open question is whether a probability
framing rescues it.

THE HYSTERESIS IS THE POINT, and it is not the same as a threshold. A plain threshold re-decides
every bar and churns; hysteresis decides only at the edges and HOLDS in between:
    p > hi        -> target +1 (long)
    p < 1 - hi    -> target -1 (short)   [long-only arm: 0 instead]
    otherwise     -> KEEP the previous position
That third line is what "keeps the trade on" means, and it is also what makes turnover tractable.

CONTROLS. Walk-forward with a purge of the full horizon (no fold sees its own future); costs at the
live Switch quote; benchmarked against buy-and-hold AND the deployed champion, never against zero;
and a shuffled-label control so "it trades a lot and makes money" cannot pass on its own.

WHY H1 AND A 24h HORIZON. Measured on this data: P(next H1 bar has the same sign as this one) is
0.4787 -- gold ANTI-persists bar to bar, which is why a chart at that resolution looks full of
tradeable swings while perfect-but-one-bar-late calls lose money. Persistence only appears further
out: 0.5060 at 1h, 0.5119 at 4h, 0.5253 at 24h. So the probability is asked at the horizon where
there is something to predict.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from v5_xau_champion_lifts import champion_recipe          # noqa: E402

SPREAD_BP = 0.504                 # live Switch XAUUSDs
RT = SPREAD_BP * 1.5              # round trip, repo convention
OUT = ROOT / "data" / "v5_runs" / "prob_hysteresis"
HORIZON = 24                      # hours
N_FOLDS = 8
RNG = np.random.default_rng(20261021)


def load_h1() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data/XAUUSD_H1_long.csv", parse_dates=["time"], index_col="time")
    return d[~d.index.duplicated(keep="last")].sort_index()


def features(d: pd.DataFrame) -> pd.DataFrame:
    """Strictly causal and deliberately SCALE-FREE (z-scores/quantiles), because 3bw showed an 8x
    regime change in absolute range is already normalised away by this kind of feature."""
    c, h, l = d["close"].astype(float), d["high"].astype(float), d["low"].astype(float)
    r = c.pct_change()
    X = pd.DataFrame(index=d.index)
    for n in (1, 2, 4, 8, 24, 48, 120):
        # NOTE n=1 is EXCLUDED from the z-score: rolling(1).std() is undefined, so z1 came back
        # all-NaN and `isfinite(...).all(axis=1)` then rejected EVERY row, leaving the first fold
        # with zero training samples. The momentum term is fine at n=1.
        if n > 1:
            X[f"z{n}"] = (c - c.rolling(n).mean()) / c.rolling(n).std()
        X[f"mom{n}"] = (c / c.shift(n) - 1.0) / r.rolling(120).std()
    X["vol_ratio"] = r.rolling(24).std() / r.rolling(120).std()
    X["rng_z"] = (((h - l) / c.shift(1)) - ((h - l) / c.shift(1)).rolling(48).mean()) \
        / ((h - l) / c.shift(1)).rolling(48).std()
    X["hour_sin"] = np.sin(2 * np.pi * d.index.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * d.index.hour / 24)
    X["streak"] = np.sign(r).groupby((np.sign(r) != np.sign(r).shift()).cumsum()).cumcount() + 1
    X["streak"] *= np.sign(r)
    return X.replace([np.inf, -np.inf], np.nan)


def walk_forward_proba(X: pd.DataFrame, y: np.ndarray, purge: int) -> np.ndarray:
    """Expanding-window walk-forward. Each fold trains ONLY on bars whose label resolved before the
    fold starts -- the purge is the full horizon, so no fold can see its own future."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    p = np.full(len(X), np.nan)
    n = len(X)
    edges = np.linspace(int(n * 0.3), n, N_FOLDS + 1).astype(int)
    Xv = X.to_numpy(np.float32)
    for i in range(N_FOLDS):
        te0, te1 = edges[i], edges[i + 1]
        tr_end = te0 - purge
        if tr_end < 500:
            continue
        tr = slice(0, tr_end)
        ok = ~np.isnan(y[tr]) & np.isfinite(Xv[tr]).all(axis=1)
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06,
                                           max_depth=4, random_state=7)
        m.fit(Xv[tr][ok], y[tr][ok])
        te = slice(te0, te1)
        good = np.isfinite(Xv[te]).all(axis=1)
        pr = np.full(te1 - te0, np.nan)
        if good.any():
            pr[good] = m.predict_proba(Xv[te][good])[:, 1]
        p[te] = pr
    return p


def hysteresis(p: np.ndarray, hi: float, long_only: bool) -> np.ndarray:
    """p > hi -> +1 ; p < 1-hi -> -1 (or 0) ; otherwise HOLD the previous position."""
    lo = 1.0 - hi
    pos = np.zeros(len(p))
    cur = 0.0
    for i, pi in enumerate(p):
        if np.isfinite(pi):
            if pi > hi:
                cur = 1.0
            elif pi < lo:
                cur = 0.0 if long_only else -1.0
        pos[i] = cur
    return pos


def score(pos: np.ndarray, ret: pd.Series, bpy: float) -> dict:
    pos = pd.Series(pos, index=ret.index).shift(1).fillna(0.0)   # act on the NEXT bar
    cost = pos.diff().abs().fillna(0.0) * (RT / 1e4)
    net = (pos * ret - cost).dropna()
    if net.std() == 0 or len(net) < 50:
        return dict(sharpe=np.nan, cagr=np.nan, maxdd=np.nan, turns=0.0, exposure=0.0)
    eq = (1 + net).cumprod()
    yrs = len(net) / bpy
    return dict(sharpe=float(net.mean() / net.std() * np.sqrt(bpy)),
                cagr=float(eq.iloc[-1] ** (1 / yrs) - 1),
                maxdd=float((eq / eq.cummax() - 1).min()),
                turns=float(pos.diff().abs().sum() / yrs),
                exposure=float(pos.abs().mean()))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_h1()
    c = d["close"].astype(float)
    ret = c.pct_change()
    bpy = len(d) / max((d.index[-1] - d.index[0]).days / 365.25, 1e-9)
    X = features(d)
    y = (c.shift(-HORIZON) > c).astype(float).where(c.shift(-HORIZON).notna()).to_numpy()

    print("=" * 104)
    print("CONTINUOUS P(up) + HYSTERESIS — the requested design, measured honestly")
    print("=" * 104)
    print(f"  XAUUSD H1  {d.index[0].date()} -> {d.index[-1].date()}  {len(d):,} bars ({bpy:.0f}/yr)")
    print(f"  horizon {HORIZON}h   walk-forward {N_FOLDS} folds, purge {HORIZON} bars   "
          f"cost {RT:.2f}bp round trip")

    p = walk_forward_proba(X, y, purge=HORIZON)
    ev = np.isfinite(p) & np.isfinite(y)
    base = float(np.nanmean(y[ev]))
    acc = float(((p[ev] > 0.5) == (y[ev] > 0.5)).mean())
    print(f"\n  OOS bars scored {ev.sum():,}   base rate P(up {HORIZON}h) = {base:.4f}   "
          f"raw accuracy @0.5 = {acc:.4f}")
    print(f"  --> accuracy MUST be read against the base rate, not 0.5 "
          f"(drift-baseline rule): lift = {acc - max(base, 1-base):+.4f}")

    # benchmarks
    bh = score(np.ones(len(ret)), ret, bpy)
    ch = champion_recipe(c, 1.0, 1.5)
    chs = score(ch.to_numpy(), ret, bpy)
    print(f"\n  {'arm':<34} {'Sharpe':>8} {'CAGR':>8} {'maxDD':>8} {'turns/yr':>9} {'exposure':>9}")
    print("  " + "-" * 82)
    print(f"  {'buy & hold':<34} {bh['sharpe']:+8.3f} {100*bh['cagr']:+7.1f}% "
          f"{100*bh['maxdd']:7.1f}% {bh['turns']:9.1f} {bh['exposure']:9.2f}")
    print(f"  {'CHAMPION (long-only, deployed)':<34} {chs['sharpe']:+8.3f} {100*chs['cagr']:+7.1f}% "
          f"{100*chs['maxdd']:7.1f}% {chs['turns']:9.1f} {chs['exposure']:9.2f}")

    rows = []
    for hi in (0.52, 0.55, 0.60, 0.65):
        for lo_only in (False, True):
            pos = hysteresis(p, hi, lo_only)
            s = score(pos, ret, bpy)
            tag = f"P>{hi:.2f} hysteresis {'LONG-ONLY' if lo_only else 'LONG/SHORT'}"
            rows.append(dict(hi=hi, long_only=lo_only, **s))
            print(f"  {tag:<34} {s['sharpe']:+8.3f} {100*s['cagr']:+7.1f}% "
                  f"{100*s['maxdd']:7.1f}% {s['turns']:9.1f} {s['exposure']:9.2f}")
    pd.DataFrame(rows).to_csv(OUT / "hysteresis_grid.csv", index=False)

    # shuffled-probability control: same turnover profile, no information
    print("\n  --- CONTROL: the identical rule on a SHUFFLED probability (no information) ---")
    best = max(rows, key=lambda r: (r["sharpe"] if np.isfinite(r["sharpe"]) else -9))
    ps = p.copy()
    fin = np.isfinite(ps)
    null = []
    for _ in range(200):
        q = ps.copy()
        v = q[fin].copy(); RNG.shuffle(v); q[fin] = v
        null.append(score(hysteresis(q, best["hi"], best["long_only"]), ret, bpy)["sharpe"])
    null = np.array([x for x in null if np.isfinite(x)])
    print(f"    best real arm  P>{best['hi']:.2f} "
          f"{'LONG-ONLY' if best['long_only'] else 'LONG/SHORT'}: Sharpe {best['sharpe']:+.3f}")
    print(f"    shuffled null: median {np.median(null):+.3f}  p95 {np.percentile(null,95):+.3f}  "
          f"-> {'BEATS the null' if best['sharpe'] > np.percentile(null,95) else 'INSIDE the null'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
