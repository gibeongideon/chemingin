"""PHASE 1 — INFORMATION CEILING: is data REPRESENTATION the bottleneck for XAU direction?

THE ARGUMENT. Every candidate representation (fracdiff, directional change, entropy, path
signatures, matrix profile, SAX, Renko...) is a measurable function R of the same trailing OHLCV
window W. Therefore I(Y; R(W)) <= I(Y; W) for all R. So there is no need to test eight
representations against a multiple-testing penalty of eight: **estimate the ceiling I(Y; W)
once.** If the raw window carries no information over the incumbent feature block, then no
function of it can, and the whole representation list closes at K=1.

This is the one phase of SHORT-VARIANT-PLAN.md whose value survives the Phase 0b stop
(V5_FINDINGS §3an), because it answers a question independent of the short thesis.

THREE ARMS:
  BASE     the incumbent block that produced this repo's ONE accuracy edge (§3r REGIME+PRICE):
           Hurst, ADX, Choppiness, ATR ratio, daily slope/z, plus z-scores, RSI, momentum,
           wick geometry.
  CEILING  the raw trailing window itself — 64 vol-normalised H4 log-returns + 64 normalised
           ranges — into a high-capacity model. The empirical stand-in for I(Y; W).
  SOURCE   BASE + M15 INTRABAR PATH. The only genuinely new input identified in scoping:
           information the H4 close series provably does not contain (realised vol inside the
           bar, max adverse/favourable excursion, path efficiency, direction changes, timing of
           the extreme). Answers a different question from CEILING — is there information
           OUTSIDE the H4 window?

STATISTIC: incremental information in **bits per independent event** = out-of-sample mean
log-loss reduction versus BASE, evaluated ONLY on non-overlapping events so the units are
honest. Not AUC: MANDATORY CONTROL #7 records SE(AUC) = 0.0367 at these event counts and a
best-of-8 pure-noise lift of +0.0526, so an AUC move of 0.02 is meaningless here.

TRAINING vs EVALUATION, deliberately different. Training uses ALL purged decided events
(~3,278 at K=12) because 128 features cannot be fitted on 274; evaluation uses only
non-overlapping events because overlapping forward windows are not independent observations.
Correlated training samples cost efficiency, not validity. Overlapping EVALUATION would
overstate significance by ~sqrt(K).

LABEL: vol-neutral first-touch — did -2% arrive before +2% within K bars. It is a trade, its
precision is its win rate, breakeven precision is 0.5097 at 1.94bp one-way. Ambiguous same-bar
double touches are dropped (46% of K=30 windows are undecided). Pre-registered in
`data/v5_runs/short_variant/PREREGISTRATION.md`; nothing here was re-parameterised after a
result was seen.

TWO MODEL CLASSES per arm (L2 logistic + regularised HistGB), best OOS log-loss taken, so that
"no information" cannot be confused with "the model could not fit it".

    python scripts/v5_repr_ceiling.py --k 12
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

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from scripts.v5_volregime_taper_crossasset import load_sleeve  # noqa: E402
from src.features.indicators import adx  # noqa: E402

BARRIER, PURGE_EXTRA = 0.02, 8
YEARS = list(range(2018, 2027))
WIN = 64                      # CEILING window length in H4 bars


# ---------------------------------------------------------------- label
def first_touch(df: pd.DataFrame, k: int) -> pd.Series:
    """1 if -BARRIER is touched before +BARRIER within k bars, 0 if the reverse, NaN if
    neither or both in the same bar (undecided -> dropped, never guessed)."""
    c, h, l = (df[x].astype(float).values for x in ("close", "high", "low"))
    out = np.full(len(c), np.nan)
    for t in range(len(c) - k):
        up, dn = c[t] * (1 + BARRIER), c[t] * (1 - BARRIER)
        for j in range(t + 1, t + k + 1):
            hu, hd = h[j] >= up, l[j] <= dn
            if hu and hd:
                break                       # ambiguous
            if hd:
                out[t] = 1.0; break
            if hu:
                out[t] = 0.0; break
    return pd.Series(out, index=df.index)


# ---------------------------------------------------------------- arms
def f_base(df: pd.DataFrame) -> pd.DataFrame:
    """The §3r REGIME+PRICE block — the incumbent, and this repo's only accuracy edge."""
    c, h, l = (df[x].astype(float) for x in ("close", "high", "low"))
    ret = c.pct_change()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(halflife=14, min_periods=10).mean()
    F = {}
    for n in (10, 20, 50):
        m, sd = c.rolling(n).mean(), c.rolling(n).std()
        F[f"z{n}"] = (c - m) / sd.replace(0, np.nan)
        F[f"pctl{n}"] = c.rolling(n).rank(pct=True)
    for n in (7, 14):
        d = c.diff()
        up = d.clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1 / n, min_periods=n).mean()
        F[f"rsi{n}"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    F["ema_dist"] = (c - c.ewm(span=50, min_periods=25).mean()) / atr14
    F["mom5"] = (c - c.shift(5)) / atr14
    F["accel"] = F["mom5"] - (c.shift(5) - c.shift(10)) / atr14
    F["upwick"] = (h - np.maximum(c, df["open"].astype(float))) / (h - l).replace(0, np.nan)
    F["lowick"] = (np.minimum(c, df["open"].astype(float)) - l) / (h - l).replace(0, np.nan)
    F["clpos"] = (c - l) / (h - l).replace(0, np.nan)
    F["atr_z"] = (atr14 - atr14.rolling(100).mean()) / atr14.rolling(100).std().replace(0, np.nan)
    # REGIME components
    lv = np.log(c)
    F["hurst100"] = _hurst(lv, 100)
    F["adx14"] = adx(df.rename(columns=str.lower), 14) if "high" in df else np.nan
    F["chop14"] = (100 * np.log10(tr.rolling(14).sum() /
                   (h.rolling(14).max() - l.rolling(14).min()).replace(0, np.nan)) / np.log10(14))
    F["vol_ratio"] = atr14 / tr.ewm(halflife=50, min_periods=30).mean()
    d1 = c.resample("D").last().dropna()
    F["d1_slope"] = np.sign(d1.ewm(span=20, min_periods=10).mean().diff()).reindex(
        c.index, method="ffill").shift(6)
    F["d1_z"] = ((d1 - d1.rolling(20).mean()) / d1.rolling(20).std().replace(0, np.nan)
                 ).reindex(c.index, method="ffill").shift(6)
    return pd.DataFrame(F).replace([np.inf, -np.inf], np.nan)


def _hurst(lv: pd.Series, w: int) -> pd.Series:
    lags = np.array([1, 2, 4, 8, 16]); ll = np.log(lags)
    def h(x):
        v = [np.var(x[lag:] - x[:-lag]) for lag in lags]
        v = np.maximum(v, 1e-18)
        return np.polyfit(ll, np.log(v), 1)[0] / 2.0
    return lv.rolling(w).apply(h, raw=True)


def f_ceiling(df: pd.DataFrame) -> pd.DataFrame:
    """The raw trailing window: WIN vol-normalised log-returns + WIN normalised ranges.
    The empirical stand-in for I(Y; W)."""
    c, h, l = (df[x].astype(float) for x in ("close", "high", "low"))
    r = np.log(c / c.shift())
    sd = r.ewm(halflife=42, min_periods=20).std().replace(0, np.nan)
    z = (r / sd)
    rng = ((h - l) / c) / ((h - l) / c).ewm(halflife=42, min_periods=20).mean().replace(0, np.nan)
    F = {f"z_l{i}": z.shift(i) for i in range(WIN)}
    F.update({f"rg_l{i}": rng.shift(i) for i in range(WIN)})
    return pd.DataFrame(F).replace([np.inf, -np.inf], np.nan).clip(-8, 8)


def f_source(df: pd.DataFrame, m15: pd.DataFrame) -> pd.DataFrame:
    """BASE + M15 INTRABAR PATH. Each H4 bar's interior, from completed M15 bars only:
    realised vol, max adverse/favourable excursion, path efficiency, direction changes and
    when in the bar the extreme happened. Not derivable from H4 closes."""
    base = f_base(df)
    o = m15["open"].astype(float); c15 = m15["close"].astype(float)
    h15 = m15["high"].astype(float); l15 = m15["low"].astype(float)
    r15 = np.log(c15 / c15.shift())
    grp = m15.index.floor("4h")
    agg = pd.DataFrame({
        "ib_rv": r15.groupby(grp).std(),
        "ib_n": r15.groupby(grp).count(),
        "ib_dirchg": (np.sign(r15).diff().abs() > 0).groupby(grp).sum(),
        "ib_pathlen": r15.abs().groupby(grp).sum(),
        "ib_net": r15.groupby(grp).sum(),
        "ib_hi_t": h15.groupby(grp).apply(lambda x: float(np.argmax(x.values)) / max(len(x), 1)),
        "ib_lo_t": l15.groupby(grp).apply(lambda x: float(np.argmin(x.values)) / max(len(x), 1)),
    })
    agg["ib_eff"] = agg["ib_net"].abs() / agg["ib_pathlen"].replace(0, np.nan)
    agg["ib_mae"] = (l15.groupby(grp).min() / o.groupby(grp).first() - 1)
    agg["ib_mfe"] = (h15.groupby(grp).max() / o.groupby(grp).first() - 1)
    agg = agg.drop(columns=["ib_pathlen", "ib_net"])
    sd = agg["ib_rv"].ewm(halflife=42, min_periods=20).mean().replace(0, np.nan)
    agg["ib_rv"] = agg["ib_rv"] / sd
    A = agg.reindex(df.index)
    return pd.concat([base, A.replace([np.inf, -np.inf], np.nan)], axis=1)


# ---------------------------------------------------------------- evaluation
def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def walk_forward_ll(X: pd.DataFrame, y: pd.Series, k: int, seed: int = 0):
    """Expanding yearly refits, purge k+PURGE_EXTRA. Returns per-year (ll, n, auc) on the
    NON-OVERLAPPING evaluation subset, best of two model classes."""
    ok = X.notna().all(axis=1) & y.notna()
    X, y = X[ok], y[ok]
    out = []
    for yr in YEARS:
        te_mask = (X.index >= f"{yr}-01-01") & (X.index <= f"{yr}-12-31")
        if te_mask.sum() < 30:
            continue
        t0 = X.index[te_mask][0]
        cut = X.index.get_indexer([t0])[0] - (k + PURGE_EXTRA)
        if cut < 300:
            continue
        tr = np.zeros(len(X), bool); tr[:cut] = True
        Xtr, ytr = X[tr].values, y[tr].values
        if len(np.unique(ytr)) < 2 or tr.sum() < 250:
            continue
        # non-overlapping evaluation subset within the test year
        te_idx = np.flatnonzero(te_mask)
        keep, last = [], -10**9
        for i in te_idx:
            if i - last >= k:
                keep.append(i); last = i
        if len(keep) < 8:
            continue
        Xte, yte = X.iloc[keep].values, y.iloc[keep].values
        if len(np.unique(yte)) < 2:
            continue
        sc = StandardScaler().fit(Xtr)
        best = None
        for mdl in (LogisticRegression(C=0.05, max_iter=2000),
                    HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=8,
                                                   l2_regularization=10.0, learning_rate=0.05,
                                                   min_samples_leaf=40, random_state=seed)):
            try:
                if isinstance(mdl, LogisticRegression):
                    mdl.fit(sc.transform(Xtr), ytr); p = mdl.predict_proba(sc.transform(Xte))[:, 1]
                else:
                    mdl.fit(Xtr, ytr); p = mdl.predict_proba(Xte)[:, 1]
            except Exception:
                continue
            ll = logloss(yte, p)
            if best is None or ll < best[0]:
                best = (ll, p)
        if best is None:
            continue
        ll, p = best
        base_rate = ytr.mean()
        ll0 = logloss(yte, np.full(len(yte), base_rate))
        out.append(dict(year=yr, n=len(yte), ll=ll, ll_const=ll0, auc=_auc(yte, p)))
    return pd.DataFrame(out)


def _auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(p).rank().values
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--synthetic", action="store_true", help="run the i.i.d. Gaussian control")
    args = ap.parse_args()
    t0 = time.time()

    df = load_sleeve(dict(path="data/XAUUSD_H4_long.csv"))
    m15 = load_sleeve(dict(path="data/XAUUSD_M15_long.csv"))
    if args.synthetic:
        rng = np.random.default_rng(3)
        r = rng.normal(0, df["close"].pct_change().std(), len(df))
        px = df["close"].iloc[0] * np.cumprod(1 + r)
        df = pd.DataFrame({"open": px, "high": px * 1.001, "low": px * 0.999,
                           "close": px}, index=df.index)
        print("*** SYNTHETIC i.i.d. GAUSSIAN CONTROL — any positive dI here is a LEAK ***\n")

    y = first_touch(df, args.k)
    arms = {"BASE": f_base(df), "CEILING": f_ceiling(df)}
    if not args.synthetic:
        arms["SOURCE (BASE+M15 path)"] = f_source(df, m15)

    print(f"XAU H4, K={args.k} bars ({args.k/6:.1f} days), barrier +/-{BARRIER:.0%}, "
          f"purge {args.k+PURGE_EXTRA}")
    print(f"decided labels {int(y.notna().sum()):,} of {len(y):,} "
          f"({y.notna().mean():.0%});  down-rate {y.mean():.3f}\n")
    print(f"{'arm':24s} {'feats':>6s} {'ev':>5s} {'logloss':>8s} {'vs const':>9s} "
          f"{'dI bits/ev':>11s} {'yrs+':>6s} {'AUC':>6s}")
    res = {}
    for name, X in arms.items():
        R = walk_forward_ll(X, y, args.k)
        if R.empty:
            print(f"{name:24s} — insufficient data"); continue
        w = R.n / R.n.sum()
        ll = float((R.ll * w).sum()); ll0 = float((R.ll_const * w).sum())
        auc = float((R.auc * w).sum())
        res[name] = R
        dI_const = (ll0 - ll) / np.log(2)
        print(f"{name:24s} {X.shape[1]:6d} {int(R.n.sum()):5d} {ll:8.4f} "
              f"{ll0 - ll:+9.4f} {dI_const:+11.4f} {(R.ll < R.ll_const).sum():3d}/{len(R):<2d} "
              f"{auc:6.3f}")
    if "BASE" in res:
        b = res["BASE"]
        print(f"\n{'incremental vs BASE':24s} {'dI bits/event':>14s} {'yrs better':>11s}")
        for name, R in res.items():
            if name == "BASE":
                continue
            j = R.merge(b, on="year", suffixes=("", "_b"))
            w = j.n / j.n.sum()
            dI = float(((j.ll_b - j.ll) * w).sum()) / np.log(2)
            print(f"{name:24s} {dI:+14.4f} {(j.ll < j.ll_b).sum():8d}/{len(j):<2d}")
        print(f"\nPRE-REGISTERED PASS needs dI >= +0.010 bits/event AND >= 7/9 years better")
    print(f"\nelapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
