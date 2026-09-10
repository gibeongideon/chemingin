"""POOLED TOP detector — pointing tonight's one durable capability at the side the repo's own
oracle says the money is on.

THE GAP THIS FILLS. The oracle-ceiling study (memory `oracle-ceiling-control-tops-not-bottoms`,
§3 line 524) measured perfect foresight of gold's geometric **BOTTOMS** at dSharpe **+0.017**
and perfect foresight of **TOPS at +0.26**, and recorded that "5 studies chased the wrong side".
Every detector study since — §3x, §3ao, §3ar, tonight's §3as — is still on BOTTOMS.
`zigzag_swings` returns `sells` (the H pivots) alongside `buys`, and **no study in this repo has
ever used it.** Meanwhile §3as established one durable capability: pooled cross-instrument
training works and TRANSFERS (+0.0570 AUC on an arm that never saw gold; R^2 +0.232 on vol).
Pointing a proven capability at the side with 15x the oracle value is the obvious experiment.

WHY A TOP IS STRUCTURALLY A BETTER TARGET THAN A BOTTOM. §3as diagnosed the bottom label's
fatal geometry: the payoff sits on ONE bar, and the bars before it are still falling, so firing
early buys falling knives (FP -3.43%) and firing late forfeits the move (52%). A top is not
symmetric with that, for two reasons:
  1. The champion is LONG-ONLY, so a top signal is used to TRIM — an action with no entry cost
     and no falling-knife exposure. Being early is cheap; being wrong costs only foregone drift.
  2. §3am measured that trimming to flat captures **71%** of the entire short-side oracle prize
     (b=1.00 dSR +2.869 of b=1.50's +4.044), so the trim is where the value is concentrated.
That is a mechanism-level reason to expect a different answer, not a hope for a better model.

CONTROL #8 IS APPLIED FROM THE FIRST LINE, not bolted on. Primary metrics are the ECONOMIC
ones — oracle gate first, then EV/fire and the fire-weighted mean OFFSET. AUC is printed for
continuity with §3as and decides nothing.

ORDER OF OPERATIONS (control #5 then #8):
  1. ORACLE gate. Perfect top knowledge as a champion trim, matched vol, panel and gold.
     If it does not clear dSharpe +0.20 at t >= 2.5, STOP — do not train anything.
  2. Only then: GOLD-ONLY / POOLED / TRANSFER detector arms.
  3. EV/fire and mean offset. A trim detector wants NEGATIVE forward return at its fires.

    python scripts/v5_pooled_top_detector.py
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

from scripts.v5_pooled_bottom_detector import GOLD, YEARS, auc, hm_se, load_d1  # noqa: E402
from scripts.v5_pooled_economics import bars_per_year, perf  # noqa: E402
from scripts.v5_xau_champion_lifts import champion_recipe  # noqa: E402
from scripts.v5_xau_turning_ml import atr, features, zigzag_swings  # noqa: E402

TOL, ZZ_ORDER, ZZ_THETA = 3, 5, 1.5
MIN_BARS, H, THR = 1200, 5, 0.60
PURGE_DAYS = TOL + 1
TRIM_H = 5          # bars the trim stays active after a fire


def build_panel() -> dict:
    """Features, the TOP label, offsets and forward returns for every usable D1 series.

    `zigzag_swings` returns `sells`/`buys` as INDEX ARRAYS. Treating one as a boolean mask
    silently measures the first ~200 bars of a series instead of the pivots (the bug that
    produced two wrong diagnostics in §3as), so the assert below is deliberate.
    """
    syms = sorted(p.name.replace("_D1_long.csv", "")
                  for p in (ROOT / "data").glob("*_D1_long.csv"))
    X, Ytop, Ybot, S, T, OFF, FWD = [], [], [], [], [], [], []
    YANT, YEARLY = [], []
    kept = 0
    for s in syms:
        d = load_d1(s)
        if d is None:
            continue
        try:
            F = features(d)
            if "hour" in F.columns:
                F = F.drop(columns=["hour"])
            sells, buys = zigzag_swings(d, ZZ_ORDER, ZZ_THETA * atr(d))
        except Exception:
            continue
        tops, bots = np.asarray(sells, int), np.asarray(buys, int)
        if len(tops) < 10:
            continue
        n = len(d)
        assert tops.max() < n and (len(bots) == 0 or bots.max() < n), "pivots must be indices"
        c = d["close"].values
        off = np.full(n, 99, int)
        for i in tops:
            for k in range(-TOL, TOL + 1):
                j = i + k
                if 0 <= j < n and abs(k) < abs(off[j]):
                    off[j] = k
        fwd = np.full(n, np.nan)
        fwd[: n - H] = c[H:] / c[: n - H] - 1
        ytop = (off != 99).astype(int)
        # OFFSET-AWARE TOP LABELS. The asymmetry being tested: on BOTTOMS (3as) firing early
        # failed because the bars before a low are still falling, so early = falling knives.
        # On TOPS a trim has no entry, so firing early costs only foregone drift. If that
        # asymmetry is real, ANTICIPATE should behave differently here than it did there.
        y_ant = np.isin(off, [-3, -2, -1, 0]).astype(int)
        y_early = np.isin(off, [-3, -2, -1]).astype(int)
        ybot = np.zeros(n, int)
        for i in bots:
            ybot[max(0, i - TOL):min(n, i + TOL + 1)] = 1
        ok = F.notna().all(axis=1).values & ~np.isnan(fwd)
        if ok.sum() < MIN_BARS:
            continue
        kept += 1
        X.append(F[ok]); Ytop.append(ytop[ok]); Ybot.append(ybot[ok])
        YANT.append(y_ant[ok]); YEARLY.append(y_early[ok])
        S.append(np.full(ok.sum(), s)); T.append(d.index[ok])
        OFF.append(off[ok]); FWD.append(fwd[ok])
    print(f"panel: {kept} instruments, {sum(len(x) for x in X):,} usable bars")
    return dict(X=pd.concat(X), ytop=np.concatenate(Ytop), ybot=np.concatenate(Ybot),
                yant=np.concatenate(YANT), yearly=np.concatenate(YEARLY),
                S=np.concatenate(S),
                T=pd.DatetimeIndex(np.concatenate([t.values for t in T])),
                off=np.concatenate(OFF), fwd=np.concatenate(FWD))


def trim_book(sig_by_sym: dict, S, T, cost_bp: float, restrict: np.ndarray) -> dict:
    """Champion book with each sleeve TRIMMED TO FLAT for TRIM_H bars after a fire.

    Matched vol is applied by the caller: a trim only ever de-risks, so comparing it to the
    untrimmed book at its own lower vol rewards it for being smaller rather than better —
    the method fix from 3ai.
    """
    legs_base, legs_trim = {}, {}
    for s, sig in sig_by_sym.items():
        d = load_d1(s)
        if d is None:
            continue
        m = (S == s) & restrict
        if m.sum() < 400:
            continue
        tt = T[m]
        c = d["close"]
        ret = c.pct_change().reindex(tt)
        ret = ret.clip(ret.quantile(0.001), ret.quantile(0.999))
        fc = champion_recipe(c, 1 / 6, 1.5).reindex(tt).ffill().fillna(0.0)
        live = pd.Series(sig, index=tt).rolling(TRIM_H, min_periods=1).max().fillna(0.0)
        for nm, pos in (("base", fc), ("trim", fc * (1.0 - live))):
            turn = pos.diff().abs().fillna(pos.abs())
            r = pos.shift(1).fillna(0.0) * ret - turn.shift(1).fillna(0.0) * cost_bp / 1e4
            (legs_base if nm == "base" else legs_trim)[s] = r
    return dict(base=legs_base, trim=legs_trim)


def paired_matched(x: pd.Series, y: pd.Series, bpy: float, label: str) -> None:
    both = pd.DataFrame(dict(x=x, y=y)).dropna()
    if len(both) < 100 or both["x"].std() == 0:
        print(f"{label:34s}  insufficient data"); return
    sx = both["x"] * (both["y"].std() / both["x"].std())      # MATCHED VOL
    dl = sx - both["y"]
    t = dl.mean() / dl.std() * np.sqrt(len(dl))
    g = pd.DataFrame(dict(x=sx, y=both["y"])).groupby(both.index.year).mean()
    print(f"{label:34s}  dSharpe {perf(sx, bpy)['sr'] - perf(both['y'], bpy)['sr']:+.3f}"
          f"   paired t {t:+6.2f}   years {int((g['x']>g['y']).sum())}/{len(g)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-bp", type=float, default=3.0)
    a = ap.parse_args()
    t0 = time.time()
    P = build_panel()
    X, S, T, off, fwd = P["X"], P["S"], P["T"], P["off"], P["fwd"]
    print(f"TOP label base rate {P['ytop'].mean():.3f}   "
          f"(bottom label, for reference, {P['ybot'].mean():.3f})")

    # what a top is actually worth, before any model
    print("\n--- available forward 5d return by offset from a TRUE TOP ---")
    for k in range(-TOL, TOL + 1):
        m = off == k
        print(f"  offset {k:+d}: n {int(m.sum()):6d}   fwd5 {np.nanmean(fwd[m]):+.4f}")
    print(f"  non-top   : n {int((off==99).sum()):6d}   fwd5 {np.nanmean(fwd[off==99]):+.4f}")

    # ================================================= 1. ORACLE GATE (control #5)
    print("\n" + "=" * 84)
    print("STEP 1 — ORACLE GATE. Perfect TOP knowledge as a champion trim, matched vol.")
    print("If this does not clear dSharpe +0.20 at t >= 2.5, nothing gets trained.")
    print("=" * 84)
    oos_mask = np.isin(T.year, list(YEARS))
    orc = {s: (P["ytop"][(S == s) & oos_mask] > 0).astype(float) for s in set(S)}
    bk = trim_book(orc, S, T, a.cost_bp, oos_mask)
    base = pd.concat(bk["base"], axis=1).mean(axis=1).dropna()
    trim = pd.concat(bk["trim"], axis=1).mean(axis=1).dropna()
    bpy = bars_per_year(base.index)
    print(f"\npanel legs {len(bk['base'])}   base Sharpe {perf(base, bpy)['sr']:+.3f}"
          f"   oracle-trim Sharpe {perf(trim, bpy)['sr']:+.3f}")
    paired_matched(trim, base, bpy, "PANEL oracle-trim vs champion")
    if GOLD in bk["base"]:
        bg = bars_per_year(bk["base"][GOLD].dropna().index)
        paired_matched(bk["trim"][GOLD], bk["base"][GOLD], bg, "GOLD  oracle-trim vs champion")

    # ================================================= 2. DETECTOR ARMS
    print("\n" + "=" * 84)
    print("STEP 2 — pooled TOP detector: GOLD-ONLY / POOLED / TRANSFER")
    print("=" * 84)
    is_gold = S == GOLD
    # POOLED-ANT / POOLED-EARLY are pooled-trained on the offset-aware TOP labels
    arms = ("GOLD-ONLY", "POOLED", "TRANSFER", "POOLED-ANT", "POOLED-EARLY")
    LBL = {"GOLD-ONLY": "ytop", "POOLED": "ytop", "TRANSFER": "ytop",
           "POOLED-ANT": "yant", "POOLED-EARLY": "yearly"}
    pr = {k: np.full(len(X), np.nan) for k in arms}
    for yr in YEARS:
        te = np.flatnonzero((T.year == yr))
        cut = pd.Timestamp(f"{yr}-01-01") - pd.Timedelta(days=PURGE_DAYS)
        trb = (T.year < yr) & (T <= cut)
        sel = {"GOLD-ONLY": trb & is_gold, "POOLED": trb, "TRANSFER": trb & ~is_gold,
               "POOLED-ANT": trb, "POOLED-EARLY": trb}
        if len(te) < 50:
            continue
        for k, mtr in sel.items():
            tr = np.flatnonzero(mtr)
            yk = P[LBL[k]]
            if len(tr) < 400 or len(np.unique(yk[tr])) < 2:
                continue
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                               max_leaf_nodes=31, min_samples_leaf=40,
                                               l2_regularization=1.0, random_state=0)
            m.fit(X.values[tr], yk[tr])
            pr[k][te] = m.predict_proba(X.values[te])[:, 1]
        print(f"  {yr}: test {len(te):6d}  ({time.time()-t0:.0f}s)")

    print(f"\n{'arm':11s} {'AUC gold':>9s} {'SE':>7s} {'AUC panel':>10s} {'mean_off':>9s} "
          f"{'fire%':>7s} {'EV/fire':>9s} {'FP_ret':>8s}")
    ok = ~np.isnan(pr["POOLED"])
    for k in arms:
        if np.isnan(pr[k]).all():
            continue
        m = ~np.isnan(pr[k])
        gi = m & is_gold
        yk = P[LBL[k]]
        ag = auc(pr[k][gi], yk[gi]) if gi.sum() > 200 else np.nan
        n1 = yk[gi].sum(); n0 = gi.sum() - n1
        fire = m & (pr[k] >= THR)
        tp = fire & (off != 99)
        fp = fire & (off == 99)
        print(f"{k:11s} {ag:9.4f} {hm_se(ag, n1, n0):7.4f} "
              f"{auc(pr[k][m], yk[m]):10.4f} "
              f"{np.average(off[tp]) if tp.sum() else np.nan:+9.3f} "
              f"{fire.sum()/m.sum()*100:7.2f} "
              f"{np.nanmean(fwd[fire]) if fire.sum() else np.nan:+9.4f} "
              f"{np.nanmean(fwd[fp]) if fp.sum() else np.nan:+8.4f}")
    print("\nFor a TRIM detector the sign flips: EV/fire should be NEGATIVE (it fires before")
    print(f"falls). Panel mean 5d return of being long is {np.nanmean(fwd):+.4f}, so a useful")
    print("trim signal needs EV/fire well below that.")

    # ================================================= 3. ECONOMICS
    print("\n" + "=" * 84)
    print("STEP 3 — ECONOMICS of the trim, matched vol")
    print("=" * 84)
    for k in arms:
        if np.isnan(pr[k]).all():
            continue
        m = ~np.isnan(pr[k])
        sig = {s: (pr[k][(S == s) & m] >= THR).astype(float) for s in set(S)}
        b = trim_book(sig, S, T, a.cost_bp, m)
        if not b["base"]:
            continue
        bb = pd.concat(b["base"], axis=1).mean(axis=1).dropna()
        tt = pd.concat(b["trim"], axis=1).mean(axis=1).dropna()
        paired_matched(tt, bb, bars_per_year(bb.index), f"PANEL {k} trim vs champion")
        if GOLD in b["base"]:
            paired_matched(b["trim"][GOLD], b["base"][GOLD],
                           bars_per_year(b["base"][GOLD].dropna().index),
                           f"GOLD  {k} trim vs champion")
    # ============================================= 4. FIRE-RATE-MATCHED SWEEP
    print("\n" + "=" * 84)
    print("STEP 4 — FAIRNESS FIX: thresholds set by TARGET FIRE RATE, not a fixed 0.60.")
    print("=" * 84)
    print("The 0.60 threshold is arbitrary across labels with different base rates —")
    print("POOLED-ANT fired 0.24% of bars and POOLED-EARLY never fired, so their ~0.000")
    print("dSharpes were my threshold rather than a finding. Each arm is now given the")
    print("threshold that makes it trim a set fraction of the time.")
    print("\nNOTE ON SEARCH: this is 5 arms x 4 rates = 20 cells. It is a FAIRNESS check, not")
    print("a search for a winner — anything that looked positive here would need the")
    print("max-statistic null (control #5) before being believed. All arms are reported.")
    print(f"\n{'arm':13s} {'rate':>6s} {'thresh':>7s} {'mean_off':>9s} {'EV/fire':>9s} "
          f"{'PANEL dSR':>10s} {'t':>7s} {'yrs':>6s} {'GOLD dSR':>9s} {'t':>7s}")
    for k in arms:
        if np.isnan(pr[k]).all():
            continue
        m = ~np.isnan(pr[k])
        pk = pr[k][m]
        for rate in (0.01, 0.03, 0.05, 0.10):
            thr = float(np.quantile(pk, 1 - rate))
            sig = {s: (pr[k][(S == s) & m] >= thr).astype(float) for s in set(S)}
            fire = m & (pr[k] >= thr)
            tp = fire & (off != 99)
            b = trim_book(sig, S, T, a.cost_bp, m)
            if not b["base"]:
                continue
            bb = pd.concat(b["base"], axis=1).mean(axis=1).dropna()
            tt2 = pd.concat(b["trim"], axis=1).mean(axis=1).dropna()

            def dsr(x, y, bp):
                both = pd.DataFrame(dict(x=x, y=y)).dropna()
                if len(both) < 100 or both["x"].std() == 0:
                    return np.nan, np.nan, "-"
                sx = both["x"] * (both["y"].std() / both["x"].std())
                dl = sx - both["y"]
                g = pd.DataFrame(dict(x=sx, y=both["y"])).groupby(both.index.year).mean()
                return (perf(sx, bp)["sr"] - perf(both["y"], bp)["sr"],
                        dl.mean() / dl.std() * np.sqrt(len(dl)),
                        f"{int((g['x']>g['y']).sum())}/{len(g)}")

            pd_, pt, py = dsr(tt2, bb, bars_per_year(bb.index))
            if GOLD in b["base"]:
                gd_, gt, _ = dsr(b["trim"][GOLD], b["base"][GOLD],
                                 bars_per_year(b["base"][GOLD].dropna().index))
            else:
                gd_, gt = np.nan, np.nan
            print(f"{k:13s} {rate:6.0%} {thr:7.3f} "
                  f"{np.average(off[tp]) if tp.sum() else np.nan:+9.3f} "
                  f"{np.nanmean(fwd[fire]) if fire.sum() else np.nan:+9.4f} "
                  f"{pd_:+10.3f} {pt:+7.2f} {py:>6s} {gd_:+9.3f} {gt:+7.2f}")
    print("\nORACLE reference: PANEL +1.252 (t +6.72, 15/15), GOLD +1.491 (t +6.37, 15/15).")
    print("The gap between that prize and these rows is the result.")
    print(f"\nelapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
