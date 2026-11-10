#!/usr/bin/env python3
"""ZigZag variant B — standalone replication of the walk-forward Sharpe.

Self-contained: needs only numpy, pandas, scipy, scikit-learn and the four CSVs in ./data.
Read README.md first; every step below is numbered to match it.

    python replicate.py                 # model + backtest + statistics (~5-10 min)
    python replicate.py --null 300      # also the random-timing null (adds ~2 min)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# ---- constants (README §1) --------------------------------------------------------------
GOLD, AUX = "XAUUSDs", ("EURUSDs", "GBPUSDs", "USDJPYs")
START = pd.Timestamp("2024-01-01")          # first walk-forward test month
ZZ_ORDER, ZZ_THETA = 5, 1.5                  # swing = local extreme over +/-5 bars, >= 1.5 ATR leg
TOL = 3                                      # label: within +/-3 bars of a swing low
PURGE = TOL + 1                              # drop the last 4 training bars (their label sees the future)
TRAIN_BARS, MIN_TRAIN = 20_000, 4_000        # trailing training window per series
LEAK_MAX = 0.15                              # refuse a series whose bar-shape features leak next-bar return
GBM = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0)
THR = 0.60                                   # fire when P(bottom) >= 0.60
TRAIL = 3.0                                  # stop distance and trail, in ATR(14) H1
TP_R = 4.3                                   # take-profit at entry + 4.3 x (3 ATR)
MAX_HOLD = 120                               # bars
SPREAD = 0.25                                # $ per oz, paid at entry
SWAP = -0.06                                 # long financing, per year of notional
RISK = 0.0025                                # 0.25% of equity risked per trade (= 1 R)


# ---- §2 data ----------------------------------------------------------------------------
def load(sym: str) -> pd.DataFrame:
    d = pd.read_csv(DATA / f"{sym}_H1.csv", parse_dates=["time"]).set_index("time")
    return d[~d.index.duplicated()].sort_index()


def intrabar_leak(d: pd.DataFrame) -> float:
    """|corr(close position in bar t, return of bar t+1)|. Clean feeds ~0.02; forward-stamped ~0.4."""
    nxt = d.close.pct_change().shift(-1)
    x = (d.close - d.low) / (d.high - d.low).replace(0, np.nan)
    ok = x.notna() & nxt.notna()
    return float(abs(x[ok].corr(nxt[ok])))


# ---- §3 indicators and features ---------------------------------------------------------
def atr(d: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([d.high - d.low, (d.high - d.close.shift()).abs(),
                    (d.low - d.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()          # Wilder smoothing


def rsi(c: pd.Series, n: int) -> pd.Series:
    dlt = c.diff()
    up = dlt.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-dlt.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def features(d: pd.DataFrame) -> pd.DataFrame:
    """20 causal features: every value at bar t uses bars <= t only."""
    c, h, l, o = d.close, d.high, d.low, d.open
    a = atr(d)
    rng = (h - l).replace(0, np.nan)
    f = pd.DataFrame(index=d.index)
    for w in (10, 20, 50):
        f[f"z{w}"] = (c - c.rolling(w).mean()) / c.rolling(w).std()
        f[f"pctl{w}"] = c.rolling(w).apply(lambda x: (x[-1] > x).mean(), raw=True)
    for n in (7, 14):
        f[f"rsi{n}"] = rsi(c, n)
    f["ema_dist"] = (c - c.ewm(span=50).mean()) / a
    f["mom"] = (c - c.shift(5)) / a
    f["accel"] = f["mom"] - (c.shift(5) - c.shift(10)) / a
    f["upwick"] = (h - np.maximum(c, o)) / rng
    f["lowick"] = (np.minimum(c, o) - l) / rng
    f["clpos"] = (c - l) / rng
    f["atr_z"] = (a - a.rolling(100).mean()) / a.rolling(100).std()
    up, dn = (c.diff() > 0).astype(int), (c.diff() < 0).astype(int)
    f["run_up"] = up * (up.groupby((up != up.shift()).cumsum()).cumcount() + 1)
    f["run_dn"] = dn * (dn.groupby((dn != dn.shift()).cumsum()).cumcount() + 1)
    r14 = f["rsi14"]
    f["bull_div"] = ((l <= l.rolling(20).min()) & (r14 > r14.rolling(20).min() + 3)).astype(int)
    f["bear_div"] = ((h >= h.rolling(20).max()) & (r14 < r14.rolling(20).max() - 3)).astype(int)
    return f


# ---- §4 the label: ZigZag swing lows ------------------------------------------------------
def swing_lows(d: pd.DataFrame) -> np.ndarray:
    hi, lo = d.high.values, d.low.values
    theta = (ZZ_THETA * atr(d)).values
    imax = set(argrelextrema(hi, np.greater_equal, order=ZZ_ORDER)[0])
    imin = set(argrelextrema(lo, np.less_equal, order=ZZ_ORDER)[0])
    cand = sorted([(i, "H") for i in imax] + [(i, "L") for i in imin])
    piv = []                                                 # (a) alternate H/L, keep the more extreme
    for i, t in cand:
        if piv and piv[-1][1] == t:
            j = piv[-1][0]
            if (hi[i] > hi[j]) if t == "H" else (lo[i] < lo[j]):
                piv[-1] = (i, t)
        else:
            piv.append((i, t))
    kept = []                                                # (b) a leg must be >= 1.5 ATR
    for i, t in piv:
        p = hi[i] if t == "H" else lo[i]
        if not kept:
            kept.append((i, t, p))
            continue
        j, tj, pj = kept[-1]
        if abs(p - pj) >= theta[i]:
            kept.append((i, t, p))
        elif t == tj and ((t == "H" and p > pj) or (t == "L" and p < pj)):
            kept[-1] = (i, t, p)
    return np.array([i for i, t, _ in kept if t == "L"], int)


def frame(d: pd.DataFrame):
    F = features(d)
    y = np.zeros(len(d), int)
    for i in swing_lows(d):
        y[max(0, i - TOL):i + TOL + 1] = 1
    return F, y, F.notna().all(axis=1).values


# ---- §5 walk-forward model ----------------------------------------------------------------
def walk_forward(gold: pd.DataFrame, aux: dict) -> pd.Series:
    Fg, _, okg = frame(gold)
    out = pd.Series(np.nan, index=gold.index)
    months = pd.date_range(START, gold.index[-1] + pd.offsets.MonthBegin(1), freq="MS")
    for m0, m1 in zip(months[:-1], months[1:]):
        test = (gold.index >= m0) & (gold.index < m1)
        if not test.any():
            continue
        X, Y = [], []
        for d in [gold] + [aux[k] for k in sorted(aux)]:
            d = d[d.index < m0]                              # truncate BEFORE labelling: no future swings
            if len(d) < MIN_TRAIN + PURGE:
                continue
            F, y, ok = frame(d)
            hi = len(d) - PURGE
            m = np.zeros(len(d), bool)
            m[max(0, hi - TRAIN_BARS):hi] = True
            m &= ok
            X.append(F.values[m])
            Y.append(y[m])
        clf = HistGradientBoostingClassifier(**GBM).fit(np.vstack(X), np.concatenate(Y))
        rows = test & okg
        out[rows] = clf.predict_proba(Fg.values[rows])[:, 1]
        print(f"  {m0:%Y-%m}: trained on {sum(len(y) for y in Y):,} rows, "
              f"{int((out[test] >= THR).sum())} fires", flush=True)
    return out


# ---- §6 trade simulation ------------------------------------------------------------------
def backtest(g: pd.DataFrame, fire: np.ndarray, spread: float = SPREAD, tp_r: float = TP_R):
    o, h, l = (g[x].to_numpy() for x in ("open", "high", "low"))
    a, t, n = atr(g).to_numpy(), g.index, len(g)
    fire = fire & (t >= START) & np.isfinite(a)
    trades, i = [], 0
    while i < n - 1:
        if not fire[i]:
            i += 1
            continue
        s = i + 1                                            # enter at the NEXT bar's open
        e = o[s] + spread
        risk = TRAIL * a[i]                                  # 1 R in $/oz
        sl, best = e - risk, e
        tp = e + tp_r * risk if tp_r else np.inf
        j, px, why = n - 1, o[-1], "open"
        for j in range(s, n):
            if l[j] <= sl:                                   # stop checked first (pessimistic)
                px, why = min(sl, o[j]), "stop"
                break
            if h[j] >= tp:
                px, why = max(tp, o[j]), "tp"
                break
            if j - s >= MAX_HOLD:
                px, why = o[j], "time"
                break
            best = max(best, h[j])
            sl = max(sl, best - TRAIL * a[j])                # ratchet: applies from the next bar
        days = (t[j] - t[s]).total_seconds() / 86400
        pnl = px - e + e * SWAP * days / 365
        trades.append((t[s], t[j], round(e, 2), round(px, 2), round(risk, 2), pnl / risk, j - s, why))
        i = j                                                # flat after bar j; bar j may fire again
    tr = pd.DataFrame(trades, columns=["t_in", "t_out", "entry", "exit", "risk", "R", "bars", "exit_reason"])
    days = pd.date_range(START.normalize(), t[-1].normalize(), freq="D")
    ret = pd.Series(0.0, index=days)
    for r in tr.itertuples():
        ret[r.t_out.normalize()] += RISK * r.R               # equity return booked on the exit day
    return tr, ret[ret.index.dayofweek < 5]


# ---- §7 statistics ------------------------------------------------------------------------
def sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0


def block_bootstrap(r: pd.Series, n: int = 2000, blk: int = 20, seed: int = 7) -> np.ndarray:
    x, rng, L, out = r.to_numpy(), np.random.default_rng(seed), len(r), []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + blk) % L for s in rng.integers(0, L, L // blk + 1)])[:L]
        y = x[idx]
        out.append(y.mean() / y.std() * np.sqrt(252) if y.std() > 0 else 0)
    return np.percentile(out, [2.5, 97.5])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", type=int, default=0, help="random-timing draws (0 = skip)")
    args = ap.parse_args()

    print("§2 data and leak screen")
    gold = load(GOLD)
    aux = {}
    for s in AUX:
        d, lk = load(s), None
        lk = intrabar_leak(d)
        print(f"  {s}: {len(d):,} bars, leak {lk:.3f} -> {'use' if lk <= LEAK_MAX else 'REFUSE'}")
        if lk <= LEAK_MAX:
            aux[s] = d
    print(f"  {GOLD}: {len(gold):,} bars {gold.index[0]} .. {gold.index[-1]}, leak {intrabar_leak(gold):.3f}")

    cache = HERE / "probs.csv"
    if cache.exists():
        print("§5 using cached probs.csv (delete it to retrain)")
        p = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        print("§5 walk-forward model (monthly refits)")
        p = walk_forward(gold, aux)
        p.rename("prob").to_csv(cache)
    pr = p.reindex(gold.index).to_numpy()
    fires = np.isfinite(pr) & (pr >= THR)

    print("§6 backtest")
    tr, ret = backtest(gold, fires)
    tr.to_csv(HERE / "trades.csv", index=False)

    print("§7 results")
    eq = (1 + ret).cumprod()
    weeks = (gold.index[-1] - START).days / 7
    w, ls = tr.R[tr.R > 0], tr.R[tr.R <= 0]
    lo, hi = block_bootstrap(ret)
    gd = gold[gold.index >= START].close.resample("D").last().dropna()
    gr = gd.pct_change().dropna() + SWAP / 252
    gr = gr[gr.index.dayofweek < 5]
    print(f"  trades {len(tr)} = {len(tr) / weeks:.2f} per week; fire rate {100 * fires[gold.index >= START].mean():.1f}% of bars")
    print(f"  win {100 * len(w) / len(tr):.1f}%  avg win {w.mean():+.2f}R  avg loss {ls.mean():+.2f}R  "
          f"expectancy {tr.R.mean():+.3f}R  total {tr.R.sum():+.1f}R")
    print(f"  exits {tr.exit_reason.value_counts().to_dict()}  median hold {tr.bars.median():.0f}h")
    print(f"  equity at 0.25% risk {100 * (eq.iloc[-1] - 1):+.2f}%  maxDD {100 * (1 - eq / eq.cummax()).max():.2f}%")
    print(f"  SHARPE {sharpe(ret):+.2f}   block-bootstrap 95% CI [{lo:+.2f}, {hi:+.2f}]")
    print(f"  per year {({y: round(sharpe(x), 2) for y, x in ret.groupby(ret.index.year)})}")
    print(f"  2x spread {sharpe(backtest(gold, fires, spread=2 * SPREAD)[1]):+.2f}   "
          f"no TP {sharpe(backtest(gold, fires, tp_r=0)[1]):+.2f}")
    print(f"  buy & hold gold, same days, with swap: Sharpe {sharpe(gr):+.2f}")
    if args.null:
        s0 = int((gold.index < START).sum())
        seg, rng, nl = fires[s0:], np.random.default_rng(13), []
        for k in rng.integers(500, len(seg) - 500, args.null):
            z = fires.copy()
            z[s0:] = np.roll(seg, int(k))                    # same fires, random times, test window only
            nl.append(sharpe(backtest(gold, z)[1]))
        nl = np.array(nl)
        print(f"  random timing ({args.null}): median {np.median(nl):+.2f}, 95th {np.percentile(nl, 95):+.2f}, "
              f"p {(nl >= sharpe(ret)).mean():.3f}")


if __name__ == "__main__":
    main()
