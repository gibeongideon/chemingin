"""QUEUE #1 VALIDATION — does the POOLED bottom detector's AUC lift MONETISE?

THE RESULT UNDER TEST. `v5_pooled_bottom_detector.py` found that training the ZigZag bottom
detector on a 51-instrument D1 panel lifts gold's out-of-sample AUC 0.6876 -> 0.7452 (+0.0576,
SE 0.0092), precision@recall20 0.523 -> 0.651, in **15 of 15** test years (sign-test
p = 0.00003). A shuffled-label control collapsed every arm to 0.50. The TRANSFER arm — trained
on every instrument EXCEPT gold — scored +0.0570, i.e. essentially the same, which says the
bottom-pivot relationship is **universal across assets**, not gold-specific.

WHY THAT IS NOT YET A RESULT. MANDATORY CONTROL #5: the oracle ceiling comes BEFORE the
detector. §3x already measured perfect foresight of gold's bottoms at dSharpe **+0.017** as a
champion overlay and **-0.675** standalone, and diagnosed why: *the champion is already long at
bottoms.* A trend follower does not need a bottom detector. If that verdict holds on D1, a
better detector of a worthless label is still worthless, and the AUC lift — however real — is a
measurement, not money.

WHAT IS GENUINELY NEW. §3x's oracle was gold-only at H4. The TRANSFER finding makes the panel
itself the object: 51 instruments, ~50x the events, and a detector that provably transfers. So
this script asks the economic question at BOTH levels:

  GOLD   the incumbent question, re-asked on D1.
  PANEL  equal-weight basket of all 51 instruments, entries timed by the pooled detector.
         Never tested. This is where the power is, and where the transfer result points.

For each level: ORACLE ceiling first, then the detector, against buy-and-hold AND the champion
at matched vol, with costs, standard errors and a per-year sign count.

DAY-COUNT DISCIPLINE (method rule 1.3.6 — an inverted book ranking was traced to this).
Crypto quotes weekends, so instruments have 259-365 bars/year. Every Sharpe here is annualised
by the series' OWN realised bars/year, and basket returns are formed on the union calendar and
annualised by ITS bars/year. Never a blanket sqrt(252).

    python scripts/v5_pooled_economics.py                 # full run, caches OOS probabilities
    python scripts/v5_pooled_economics.py --cost-bp 6     # cost sensitivity
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

from scripts.v5_pooled_bottom_detector import (  # noqa: E402
    GOLD, TOL, YEARS, auc, build_panel, load_d1, prec_at_recall,
)
from scripts.v5_xau_champion_lifts import champion_recipe  # noqa: E402

CACHE = ROOT / "data" / "v5_runs" / "pooled_oos_probs.csv"
PURGE_DAYS = TOL + 1


# ------------------------------------------------------------------ OOS probabilities
def oos_probs(force: bool = False) -> pd.DataFrame:
    """Walk-forward POOLED model, predicting EVERY instrument in each test year.

    Identical protocol to the detector script — train strictly before January of the test
    year across the whole panel, purge TOL+1 days — but the prediction set is the entire
    panel rather than gold alone, because the panel economics is the new question.
    """
    if CACHE.exists() and not force:
        d = pd.read_csv(CACHE, parse_dates=["time"])
        print(f"loaded cached OOS probabilities: {len(d):,} rows, "
              f"{d['sym'].nunique()} instruments")
        return d

    t0 = time.time()
    X, Y, S, T = build_panel()
    print(f"pooled rows {len(X):,}  base rate {Y.mean():.3f}")
    out = []
    for yr in YEARS:
        te = T.year == yr
        cut = pd.Timestamp(f"{yr}-01-01") - pd.Timedelta(days=PURGE_DAYS)
        tr = (T.year < yr) & (T <= cut)
        if tr.sum() < 2000 or te.sum() < 50:
            continue
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                           max_leaf_nodes=31, min_samples_leaf=40,
                                           l2_regularization=1.0, random_state=0)
        m.fit(X.values[tr], Y.values[tr])
        p = m.predict_proba(X.values[te])[:, 1]
        out.append(pd.DataFrame(dict(time=T[te], sym=S.values[te],
                                     prob=p, y=Y.values[te])))
        print(f"  {yr}: train {int(tr.sum()):7d}  predict {int(te.sum()):6d}  "
              f"({time.time()-t0:.0f}s)")
    d = pd.concat(out, ignore_index=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(CACHE, index=False)
    print(f"cached {len(d):,} OOS predictions -> {CACHE.relative_to(ROOT)}")
    return d


# ------------------------------------------------------------------ performance plumbing
def bars_per_year(idx: pd.DatetimeIndex) -> float:
    """Realised bars per year for THIS series. The day-count trap: gold quotes ~259/yr,
    bitcoin ~365, and annualising both with sqrt(252) once inverted a book ranking."""
    yrs = (idx[-1] - idx[0]).days / 365.25
    return len(idx) / max(yrs, 1e-9)


def perf(ret: pd.Series, bpy: float) -> dict:
    r = ret.dropna()
    if len(r) < 60 or r.std() == 0:
        return dict(sr=np.nan, se=np.nan, cagr=np.nan, dd=np.nan, n=len(r))
    sr = r.mean() / r.std() * np.sqrt(bpy)
    yrs = len(r) / bpy
    eq = (1 + r).cumprod()
    return dict(sr=float(sr),
                se=float(np.sqrt((1 + sr * sr / 2) / max(yrs, 1e-9))),  # Lo's SE
                cagr=float(eq.iloc[-1] ** (1 / max(yrs, 1e-9)) - 1),
                dd=float((eq / eq.cummax() - 1).min()), n=len(r))


def net(pos: pd.Series, ret: pd.Series, cost_bp: float) -> pd.Series:
    """Position applied to the NEXT bar's return, minus turnover cost. pos must already be
    shifted-safe (built from information available at the bar it is indexed on)."""
    p = pos.reindex(ret.index).fillna(0.0)
    turn = p.diff().abs().fillna(p.abs())
    return p.shift(1).fillna(0.0) * ret - turn.shift(1).fillna(0.0) * cost_bp / 1e4


def match_vol(a: pd.Series, b: pd.Series) -> pd.Series:
    """Lever `a` to `b`'s realised vol. Comparing an overlay that only de-risks against a
    base at different vol is the mistake `vol_match` was written to stop."""
    sa, sb = a.std(), b.std()
    return a * (sb / sa) if sa > 0 else a


def per_year_sign(a: pd.Series, b: pd.Series) -> tuple[int, int]:
    """Years in which a's mean return beats b's — the repo's robustness workhorse."""
    df = pd.DataFrame(dict(a=a, b=b)).dropna()
    g = df.groupby(df.index.year).mean()
    return int((g["a"] > g["b"]).sum()), len(g)


# ------------------------------------------------------------------ trade structures
def structures(close: pd.Series, sig: pd.Series, ret: pd.Series,
               cost_bp: float, holds=(3, 5, 10, 20)) -> dict:
    """Every way this repo knows to turn a bottom signal into a position.

    `sig` is in [0,1] and is indexed on the bar whose CLOSE the decision is made at, so
    `net()`'s shift(1) is what makes it causal.
    """
    champ = champion_recipe(close, 1 / 6, 1.5).reindex(ret.index).ffill().fillna(0.0)
    out = {"BUY-HOLD": net(pd.Series(1.0, index=ret.index), ret, cost_bp),
           "CHAMPION": net(champ, ret, cost_bp)}
    for h in holds:
        # STANDALONE: flat, go long for h bars on a fire. rolling max = "fired within h bars"
        st = sig.rolling(h, min_periods=1).max().reindex(ret.index).fillna(0.0)
        out[f"STANDALONE-h{h}"] = net(st, ret, cost_bp)
        # BOOST: champion, sized up while a bottom is live. The §3x overlay form.
        out[f"BOOST-h{h}"] = net((champ * (1 + st)).clip(0, 2), ret, cost_bp)
    return out


def report(name: str, res: dict, bpy: float, bench: str = "BUY-HOLD") -> pd.DataFrame:
    rows = []
    b = res[bench]
    for k, v in res.items():
        p = perf(v, bpy)
        w, n = per_year_sign(v, b) if k != bench else (0, 0)
        rows.append(dict(structure=k, sharpe=p["sr"], se=p["se"], cagr=p["cagr"],
                         maxdd=p["dd"], yrs_beat=f"{w}/{n}" if n else "-"))
    d = pd.DataFrame(rows).set_index("structure")
    print(f"\n--- {name} (annualised on {bpy:.0f} bars/yr, benchmark {bench}) ---")
    print(d.to_string(float_format=lambda x: f"{x:+.3f}"))
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-bp", type=float, default=3.0,
                    help="one-way cost in bp on turnover (repo convention 0.75 x spread_bp)")
    ap.add_argument("--thr", type=float, default=0.60)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--k", type=int, default=8, help="cross-sectional legs per side")
    a = ap.parse_args()
    print(f"one-way cost {a.cost_bp:.1f}bp   detector threshold {a.thr:.2f}\n")

    d = oos_probs(a.force)

    # ======================================================= LEVEL 1: GOLD
    print("\n" + "=" * 78)
    print("LEVEL 1 — GOLD.  Re-asking the question 3x answered at H4 (oracle +0.017).")
    print("=" * 78)
    g = d[d["sym"] == GOLD].set_index("time").sort_index()
    px = load_d1(GOLD)
    close = px["close"]
    ret = close.pct_change().reindex(g.index).dropna()
    bpy = bars_per_year(ret.index)

    oracle = pd.Series((g["y"].reindex(ret.index) > 0).astype(float), index=ret.index)
    det = pd.Series((g["prob"].reindex(ret.index) >= a.thr).astype(float), index=ret.index)
    print(f"gold OOS bars {len(ret):,}   label base rate {oracle.mean():.3f}   "
          f"detector fires {det.mean():.3f} of bars   "
          f"AUC {auc(g['prob'], g['y']):.4f}")
    p20, n20 = prec_at_recall(g["prob"], g["y"], 0.20)
    print(f"precision@recall20 {p20:.3f} on {n20} signals")

    ro = report("GOLD / ORACLE (perfect label knowledge)",
                structures(close, oracle, ret, a.cost_bp), bpy)
    rd = report("GOLD / POOLED DETECTOR",
                structures(close, det, ret, a.cost_bp), bpy)

    # ======================================================= LEVEL 2: PANEL
    print("\n" + "=" * 78)
    print("LEVEL 2 — 51-INSTRUMENT PANEL.  Never tested. Where the TRANSFER result points.")
    print("=" * 78)
    syms = sorted(d["sym"].unique())
    leg_o, leg_d, leg_b, leg_c = {}, {}, {}, {}
    for s in syms:
        pxs = load_d1(s)
        if pxs is None:
            continue
        ds = d[d["sym"] == s].set_index("time").sort_index()
        r = pxs["close"].pct_change().reindex(ds.index).dropna()
        if len(r) < 250:
            continue
        # winsorise: one instrument's 40% crypto day must not decide a 51-asset basket
        r = r.clip(r.quantile(0.001), r.quantile(0.999))
        c = pxs["close"]
        o = pd.Series((ds["y"].reindex(r.index) > 0).astype(float), index=r.index)
        t = pd.Series((ds["prob"].reindex(r.index) >= a.thr).astype(float), index=r.index)
        st_o = o.rolling(5, min_periods=1).max()
        st_d = t.rolling(5, min_periods=1).max()
        champ = champion_recipe(c, 1 / 6, 1.5).reindex(r.index).ffill().fillna(0.0)
        # vol-normalise each leg so the basket is risk- not price-weighted
        v = r.rolling(60, min_periods=30).std().shift(1)
        k = (0.01 / v).clip(0, 5).fillna(0.0)
        leg_o[s] = net(st_o * k, r, a.cost_bp)
        leg_d[s] = net(st_d * k, r, a.cost_bp)
        leg_b[s] = net(pd.Series(1.0, index=r.index) * k, r, a.cost_bp)
        leg_c[s] = net(champ * k, r, a.cost_bp)
    print(f"panel legs built: {len(leg_d)} instruments")

    def basket(legs: dict) -> pd.Series:
        return pd.concat(legs, axis=1).mean(axis=1).dropna()

    pan = {"BUY-HOLD": basket(leg_b), "CHAMPION": basket(leg_c),
           "ORACLE-h5": basket(leg_o), "DETECTOR-h5": basket(leg_d)}
    # the honest comparison: everything levered to buy-and-hold's vol
    ref = pan["BUY-HOLD"]
    pan = {k: (v if k == "BUY-HOLD" else match_vol(v, ref)) for k, v in pan.items()}
    bpy_p = bars_per_year(ref.index)
    rp = report("PANEL basket, vol-matched to buy-and-hold", pan, bpy_p)

    # ---- the decisive paired comparisons ----
    print("\n" + "=" * 78)
    print("PAIRED TESTS — the numbers that decide this")
    print("=" * 78)
    def paired(nm: str, x: pd.Series, y: pd.Series, bpy_: float) -> None:
        df = pd.DataFrame(dict(x=x, y=y)).dropna()
        dl = df["x"] - df["y"]
        if len(dl) < 60 or dl.std() == 0:
            print(f"{nm:48s}  insufficient overlap"); return
        t = dl.mean() / dl.std() * np.sqrt(len(dl))
        sx, sy = perf(df["x"], bpy_)["sr"], perf(df["y"], bpy_)["sr"]
        w, n = per_year_sign(df["x"], df["y"])
        print(f"{nm:48s}  dSharpe {sx-sy:+.3f}   paired t {t:+6.2f}   years {w}/{n}")

    gd = structures(close, det, ret, a.cost_bp)
    go = structures(close, oracle, ret, a.cost_bp)
    paired("GOLD  ORACLE-BOOST-h5   vs CHAMPION", go["BOOST-h5"], go["CHAMPION"], bpy)
    paired("GOLD  DETECT-BOOST-h5   vs CHAMPION", gd["BOOST-h5"], gd["CHAMPION"], bpy)
    paired("GOLD  ORACLE-STANDALONE vs BUY-HOLD", go["STANDALONE-h5"], go["BUY-HOLD"], bpy)
    paired("GOLD  DETECT-STANDALONE vs BUY-HOLD", gd["STANDALONE-h5"], gd["BUY-HOLD"], bpy)
    paired("PANEL ORACLE-h5         vs BUY-HOLD", pan["ORACLE-h5"], pan["BUY-HOLD"], bpy_p)
    paired("PANEL DETECTOR-h5       vs BUY-HOLD", pan["DETECTOR-h5"], pan["BUY-HOLD"], bpy_p)
    paired("PANEL DETECTOR-h5       vs CHAMPION", pan["DETECTOR-h5"], pan["CHAMPION"], bpy_p)

    # ================================================= LEVEL 3: cross-sectional
    print("\n" + "=" * 78)
    print("LEVEL 3 — CROSS-SECTIONAL.  Drift cancels; the detector is asked a question it")
    print("can actually answer, and only the TRANSFER result makes scores comparable.")
    print("=" * 78)
    xs = cross_sectional(d, a.cost_bp, k=a.k, hold=5)
    bpy_x = bars_per_year(xs["XS DETECTOR L/S"].dropna().index)
    rows = []
    for nm, v in xs.items():
        pp = perf(v.dropna(), bpy_x)
        rows.append(dict(structure=nm, sharpe=pp["sr"], se=pp["se"],
                         cagr=pp["cagr"], maxdd=pp["dd"], n=pp["n"]))
    print(f"\n--- top/bottom {a.k} of ~50, 5-day stagger, {a.cost_bp:.0f}bp one-way "
          f"(annualised on {bpy_x:.0f} bars/yr) ---")
    print(pd.DataFrame(rows).set_index("structure").to_string(
        float_format=lambda x: f"{x:+.3f}"))
    print()
    paired("XS DETECTOR L/S vs XS RANDOM L/S", xs["XS DETECTOR L/S"],
           xs["XS RANDOM  L/S"], bpy_x)
    paired("XS ORACLE   L/S vs XS RANDOM L/S", xs["XS ORACLE  L/S"],
           xs["XS RANDOM  L/S"], bpy_x)
    zz = xs["XS DETECTOR L/S"].dropna()
    gy = zz.groupby(zz.index.year).mean()
    print(f"XS DETECTOR positive years: {int((gy > 0).sum())}/{len(gy)}")

    print("\nDECISION RULE (MANDATORY CONTROL #5): if the ORACLE row does not clear")
    print("dSharpe +0.20 with t >= 2.5, the LABEL is not worth detecting and the AUC lift")
    print("is a measurement rather than money — regardless of how significant it is.")




# ==================================================================== CROSS-SECTIONAL
def cross_sectional(d: pd.DataFrame, cost_bp: float, k: int = 8,
                    hold: int = 5) -> dict:
    """QUEUE #38, and the structure the TRANSFER result actually implies.

    The long-only panel timer above must lose: a bottom LOOKS like a big decline, so the
    detector is long ~35% of the time and specifically after drops, forfeiting the panel's
    +9.8%/yr drift. Buy-and-hold in a 15-year bull panel is the one benchmark a market timer
    cannot beat. That is the repo's false-positive wall, not a verdict on the signal.

    Cross-sectionally the drift cancels. Each day, rank every instrument by P(bottom) and go
    long the top k / short the bottom k, dollar-neutral. The common factor is differenced away,
    so the question becomes the one the detector can actually answer: *given two assets, is the
    one that looks more like a bottom the better buy?* This is only legitimate because the
    TRANSFER arm proved the score is comparable ACROSS instruments (+0.0570 trained without
    gold at all) — a gold-only detector's probabilities would not be.

    ORACLE arm included: the same ranking on the true label, so control #5 still gates it.
    """
    pv = d.pivot_table(index="time", columns="sym", values="prob")
    yv = d.pivot_table(index="time", columns="sym", values="y")
    px = {}
    for s in pv.columns:
        f = load_d1(s)
        if f is not None:
            px[s] = f["close"]
    R = pd.DataFrame({s: px[s].pct_change() for s in px}).reindex(pv.index)
    R = R.clip(R.quantile(0.001), R.quantile(0.999), axis=1)
    # vol-normalise so the spread is risk- not price-weighted
    V = R.rolling(60, min_periods=30).std().shift(1)
    K = (0.01 / V).clip(0, 5)

    def spread(score: pd.DataFrame, long_only: bool) -> pd.Series:
        rk = score.rank(axis=1, ascending=False, na_option="keep")
        n = score.notna().sum(axis=1)
        lo = (rk.le(k)).astype(float)
        sh = (rk.gt(n.values[:, None] - k)).astype(float)
        w = lo if long_only else (lo - sh)
        w = w.where(score.notna(), 0.0) * K.reindex_like(score).fillna(0.0)
        # smooth over `hold` days = an equal-weight overlay of `hold` staggered books,
        # which is how the repo turns a daily signal into a holding period without
        # rebalancing the whole basket every bar
        w = w.rolling(hold, min_periods=1).mean()
        gross = w.abs().sum(axis=1).replace(0, np.nan)
        w = w.div(gross, axis=0).fillna(0.0)
        turn = w.diff().abs().sum(axis=1).fillna(0.0)
        return (w.shift(1) * R).sum(axis=1) - turn.shift(1).fillna(0.0) * cost_bp / 1e4

    return {
        "XS ORACLE  L/S":   spread(yv + np.random.default_rng(0).normal(0, 1e-9, yv.shape),
                                  False),
        "XS DETECTOR L/S":  spread(pv, False),
        "XS DETECTOR LONG": spread(pv, True),
        "XS RANDOM  L/S":   spread(pd.DataFrame(
            np.random.default_rng(7).random(pv.shape),
            index=pv.index, columns=pv.columns).where(pv.notna()), False),
    }


if __name__ == "__main__":
    main()
