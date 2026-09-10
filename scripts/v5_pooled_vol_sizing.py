"""QUEUE #34 — predict VOLATILITY, not direction, and size with it. Pooled across 51 series.

WHY THIS SURVIVES CONTROL #8. §3as retired ~20 queued approaches because their mechanism was
"rank the same 7-bar bottom label better", and a 5.9-sigma AUC lift there produced -0.702
Sharpe. #8 explicitly does NOT apply to programmes that change the TARGET. Volatility is a
different target, and it is the one where the prior is genuinely favourable: vol is an order of
magnitude more predictable than direction, and the repo's single strongest overlay ever
(range-based vol estimators, t +2.25) lived here before being killed as a 2022+ artifact.

WHAT IS ACTUALLY NEW. Two things, and only together:
  1. §3ai/§3ad tried vol sizing with TRAILING estimators (realised, Parkinson, Garman-Klass,
     Rogers-Satchell). None was a FORECAST — they are all backward-looking averages, and the
     live bot's gap-aware ATR beat all four.
  2. Tonight established the one durable capability of the whole run: pooled cross-instrument
     training works and TRANSFERS (+0.0570 on an arm that never saw gold). That has never been
     pointed at volatility.

THE HONEST FRAMING, which is where this kind of study usually cheats. Trailing vol is already a
very strong baseline because vol is highly persistent — a naive "tomorrow's vol = last 60 days'
vol" scores an R^2 most papers would report as a success. So the question is NOT "can we
predict vol" (yes, trivially) but **"is there incremental information over the trailing
estimator the book already uses, and does it convert at book level?"**

CONTROL #8's ANALOGUE FOR A REGRESSION TARGET. An R^2 gain is this study's AUC: necessary,
decorative, and not money. So the primary metric is the ECONOMIC one — champion book Sharpe
under each sizing rule, **at matched vol** (the method fix from §3ai: an overlay that mostly
de-risks must be levered to the base's vol before a paired-t, or it is rewarded for being
smaller rather than better). R^2 is reported second and decides nothing.

FOUR ARMS:
  INCUMBENT    size ~ 1 / trailing 60d realised vol. What the book does today.
  ML-POOLED    size ~ 1 / pooled HistGB forecast of forward 20d vol, walk-forward by year.
  ML-TRANSFER  same, but never trained on the target instrument. Tests universality again.
  ORACLE-VOL   size ~ 1 / REALISED forward 20d vol. The ceiling (control #5) — if this does
               not clear the bar, no forecast of it can.

    python scripts/v5_pooled_vol_sizing.py
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

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from scripts.v5_pooled_bottom_detector import GOLD, YEARS, load_d1  # noqa: E402
from scripts.v5_pooled_economics import bars_per_year, perf  # noqa: E402
from scripts.v5_xau_champion_lifts import champion_recipe  # noqa: E402

FWD_VOL = 20          # forecast horizon: forward 20-bar realised vol
MIN_BARS = 1200
PURGE_DAYS = FWD_VOL + 1   # the target itself looks 20 bars ahead — purge MUST cover it
ARMS = ("INCUMBENT", "ML-POOLED", "ML-TRANSFER", "ORACLE-VOL")


def vol_features(d: pd.DataFrame) -> pd.DataFrame:
    """Causal, scale-free volatility features. Every column is a ratio, a log-ratio or a
    percentile, so the matrix is comparable across a bond future and bitcoin — the property
    that made the pooled bottom detector transfer.

    Deliberately includes the four trailing estimators §3ai tested, so the ML arm cannot win
    merely by having access to a better backward-looking average: it must beat their span.
    """
    c, h, l, o = d["close"], d["high"], d["low"], d["open"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=d.index)
    base = r.rolling(60).std()
    for w in (5, 10, 20, 60, 120, 250):
        f[f"rv{w}"] = np.log(r.rolling(w).std() / base)       # vol term structure
    # the four estimators from 3ai, as log-ratios to close-to-close
    park = np.sqrt((np.log(h / l) ** 2).rolling(20).mean() / (4 * np.log(2)))
    gk = np.sqrt((0.5 * np.log(h / l) ** 2
                  - (2 * np.log(2) - 1) * np.log(c / o) ** 2).rolling(20).mean())
    rs = np.sqrt((np.log(h / c) * np.log(h / o)
                  + np.log(l / c) * np.log(l / o)).rolling(20).mean().clip(lower=0) + 1e-12)
    f["park"] = np.log(park / base)
    f["gk"] = np.log(gk / base)
    f["rs"] = np.log(rs / base)
    f["gap"] = np.log((np.log(o / c.shift(1)).abs().rolling(20).mean() + 1e-9) / base)
    # persistence / clustering
    f["vol_of_vol"] = np.log(r.rolling(20).std().rolling(60).std() / base + 1e-9)
    f["absr_ac1"] = r.abs().rolling(60).corr(r.abs().shift(1))
    f["rv_pctile"] = r.rolling(20).std().rolling(500, min_periods=250).rank(pct=True)
    # leverage effect: vol responds asymmetrically to down moves
    f["semi_ratio"] = np.log((r.clip(upper=0).rolling(20).std() + 1e-9)
                             / (r.clip(lower=0).rolling(20).std() + 1e-9))
    f["ret20"] = r.rolling(20).sum() / base
    f["dd"] = (c / c.rolling(120).max() - 1) / base
    f["range_pctile"] = ((h - l) / c).rolling(500, min_periods=250).rank(pct=True)
    return f


def build_panel() -> dict:
    syms = sorted(p.name.replace("_D1_long.csv", "")
                  for p in (ROOT / "data").glob("*_D1_long.csv"))
    X, Y, S, T, BASE = [], [], [], [], []
    kept = 0
    for s in syms:
        d = load_d1(s)
        if d is None:
            continue
        try:
            F = vol_features(d)
        except Exception:
            continue
        r = np.log(d["close"]).diff()
        base = r.rolling(60).std()
        # TARGET: log ratio of forward realised vol to the trailing estimator. Predicting the
        # RATIO rather than the level is what makes this a test of INCREMENTAL information —
        # a model that only learns "vol is persistent" scores zero here by construction.
        fv = r.shift(-1).rolling(FWD_VOL).std().shift(-(FWD_VOL - 1))
        y = np.log(fv / base)
        ok = F.notna().all(axis=1).values & y.notna().values & (base > 0).values
        if ok.sum() < MIN_BARS:
            continue
        kept += 1
        X.append(F[ok]); Y.append(y[ok]); S.append(np.full(ok.sum(), s))
        T.append(d.index[ok]); BASE.append(base[ok])
    print(f"panel: {kept} instruments, {sum(len(x) for x in X):,} usable bars")
    return dict(X=pd.concat(X), Y=pd.concat(Y), S=np.concatenate(S),
                T=pd.DatetimeIndex(np.concatenate([t.values for t in T])),
                base=pd.concat(BASE))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-bp", type=float, default=3.0)
    ap.add_argument("--target-vol", type=float, default=0.01)
    a = ap.parse_args()
    t0 = time.time()
    P = build_panel()
    X, Y, S, T = P["X"], P["Y"], P["S"], P["T"]
    print(f"target = log(fwd {FWD_VOL}d vol / trailing 60d vol): "
          f"mean {Y.mean():+.4f}  sd {Y.std():.4f}")
    print("(a model that only learns 'vol is persistent' scores R^2 = 0 on this target)\n")

    pred = {k: np.full(len(X), np.nan) for k in ("ML-POOLED", "ML-TRANSFER")}
    for yr in YEARS:
        te = np.flatnonzero(T.year == yr)
        cut = pd.Timestamp(f"{yr}-01-01") - pd.Timedelta(days=PURGE_DAYS)
        tr = np.flatnonzero((T.year < yr) & (T <= cut))
        if len(tr) < 2000 or len(te) < 50:
            continue
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                          max_leaf_nodes=31, min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=0)
        m.fit(X.values[tr], Y.values[tr])
        pred["ML-POOLED"][te] = m.predict(X.values[te])
        # TRANSFER: for the gold economics, a model that never saw gold
        trg = tr[S[tr] != GOLD]
        m2 = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                          max_leaf_nodes=31, min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=0)
        m2.fit(X.values[trg], Y.values[trg])
        pred["ML-TRANSFER"][te] = m2.predict(X.values[te])
        print(f"  {yr}: train {len(tr):7d}  test {len(te):6d}  ({time.time()-t0:.0f}s)")

    # ---------------- STEP 1: is there incremental information at all? ----------------
    print("\n" + "=" * 84)
    print("STEP 1 — INCREMENTAL INFORMATION over the trailing estimator (decorative, per #8)")
    print("=" * 84)
    ok = ~np.isnan(pred["ML-POOLED"])
    yv = Y.values[ok]
    print(f"{'arm':14s} {'R2 vs trailing':>15s} {'corr':>8s} {'n':>9s}")
    for k in ("ML-POOLED", "ML-TRANSFER"):
        pv = pred[k][ok]
        r2 = 1 - np.sum((yv - pv) ** 2) / np.sum(yv ** 2)   # vs the zero-ratio baseline
        print(f"{k:14s} {r2:+15.4f} {np.corrcoef(pv, yv)[0,1]:+8.4f} {ok.sum():9d}")
    print("R^2 is measured against predicting ZERO log-ratio, i.e. against the trailing")
    print("estimator itself. Positive => genuine incremental vol information.")

    # ---------------- STEP 2: does it convert at book level? ----------------
    print("\n" + "=" * 84)
    print("STEP 2 — THE ECONOMIC TEST. Champion book Sharpe under each sizing rule,")
    print("at MATCHED VOL (3ai's method fix). This is the metric that decides.")
    print("=" * 84)
    idx = pd.DataFrame(dict(sym=S, t=T)).assign(row=np.arange(len(S)))
    books = {k: {} for k in ARMS}
    for s in sorted(set(S)):
        d = load_d1(s)
        if d is None:
            continue
        # RESTRICT EVERY ARM TO THE OOS WINDOW. SPX carries history back to 1927 and
        # USDJPY to 1996; building INCUMBENT over a symbol's full history while the ML arms
        # sit flat wherever they have no prediction hands the baseline ~85 years of free
        # in-sample SPX and is not a comparison. All arms trade the same bars or none do.
        sub = idx[(idx["sym"] == s) & ~np.isnan(pred["ML-POOLED"][idx["row"].values])]
        rows = sub["row"].values
        if len(rows) < 400:
            continue
        tt = pd.DatetimeIndex(sub["t"].values)
        c = d["close"]
        ret = c.pct_change().reindex(tt)
        ret = ret.clip(ret.quantile(0.001), ret.quantile(0.999))
        fc = champion_recipe(c, 1 / 6, 1.5).reindex(tt).ffill().fillna(0.0)
        base = P["base"].values[rows]
        fwd_real = np.exp(Y.values[rows]) * base            # realised forward vol
        est = {
            "INCUMBENT":   base,
            "ML-POOLED":   np.exp(pred["ML-POOLED"][rows]) * base,
            "ML-TRANSFER": np.exp(pred["ML-TRANSFER"][rows]) * base,
            "ORACLE-VOL":  fwd_real,
        }
        for k, v in est.items():
            if np.isnan(v).all():
                continue
            lev = pd.Series(np.clip(a.target_vol / np.where(v > 0, v, np.nan), 0, 5),
                            index=tt).ffill().fillna(0.0)
            pos = fc * lev
            turn = pos.diff().abs().fillna(pos.abs())
            books[k][s] = (pos.shift(1).fillna(0.0) * ret
                           - turn.shift(1).fillna(0.0) * a.cost_bp / 1e4)

    def bk(k):
        return pd.concat(books[k], axis=1).mean(axis=1).dropna()

    print(f"\nbook legs: {len(books['INCUMBENT'])} instruments")
    ref = bk("INCUMBENT")
    bpy = bars_per_year(ref.index)
    print(f"\n{'sizing rule':14s} {'Sharpe':>8s} {'SE':>7s} {'CAGR':>8s} {'maxDD':>8s} "
          f"{'dSharpe':>8s} {'paired t':>9s} {'years':>7s}")
    for k in ARMS:
        if not books[k]:
            continue
        v = bk(k)
        both = pd.DataFrame(dict(x=v, y=ref)).dropna()
        # MATCHED VOL: lever the arm to the incumbent's vol before differencing
        sx = both["x"] * (both["y"].std() / both["x"].std())
        dl = sx - both["y"]
        t = dl.mean() / dl.std() * np.sqrt(len(dl)) if dl.std() > 0 else np.nan
        g = pd.DataFrame(dict(x=sx, y=both["y"])).groupby(both.index.year).mean()
        p1, p0 = perf(sx, bpy), perf(both["y"], bpy)
        print(f"{k:14s} {p1['sr']:+8.3f} {p1['se']:+7.3f} {p1['cagr']:+8.3f} "
              f"{p1['dd']:+8.3f} "
              f"{p1['sr']-p0['sr']:+8.3f} {t:+9.2f} "
              f"{int((g['x']>g['y']).sum()):3d}/{len(g):<3d}")

    # ---------------- gold alone, the deployable question ----------------
    print(f"\n--- GOLD sleeve alone (the book's largest weight) ---")
    if GOLD in books["INCUMBENT"]:
        rg = books["INCUMBENT"][GOLD].dropna()
        bg = bars_per_year(rg.index)
        print(f"{'sizing rule':14s} {'Sharpe':>8s} {'dSharpe':>8s} {'paired t':>9s} {'years':>7s}")
        for k in ARMS:
            if GOLD not in books[k]:
                continue
            both = pd.DataFrame(dict(x=books[k][GOLD], y=rg)).dropna()
            sx = both["x"] * (both["y"].std() / both["x"].std())
            dl = sx - both["y"]
            t = dl.mean() / dl.std() * np.sqrt(len(dl)) if dl.std() > 0 else np.nan
            g = pd.DataFrame(dict(x=sx, y=both["y"])).groupby(both.index.year).mean()
            print(f"{k:14s} {perf(sx, bg)['sr']:+8.3f} "
                  f"{perf(sx, bg)['sr']-perf(both['y'], bg)['sr']:+8.3f} {t:+9.2f} "
                  f"{int((g['x']>g['y']).sum()):3d}/{len(g):<3d}")

    print("\nDECISION (control #5 then #8): if ORACLE-VOL does not clear dSharpe +0.20 with")
    print("t >= 2.5 at matched vol, perfect vol foresight is not worth having and no forecast")
    print("of it can be. If ORACLE clears but the ML arms do not, the information exists and")
    print("is not extractable — report it that way, do not round it up.")
    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
