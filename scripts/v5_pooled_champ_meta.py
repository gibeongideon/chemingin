"""QUEUE #35 — POOLED CHAMP-META: predict the sign of the CHAMPION'S OWN forward P&L.

WHY THIS IS THE RIGHT FINAL EXPERIMENT, and why it is not a repeat.

Tonight produced four instances of one failure shape (MANDATORY CONTROL #8): a real,
transferable, statistically overwhelming prediction of a label, worth nothing economically.
  §3as  bottoms — AUC +0.0576 at 5.9 sigma, 15/15 years -> panel dSharpe -0.702
  §3at  volatility — R^2 +0.230, and PERFECT foresight is worth -0.146
  §3au  tops — oracle +1.252 (t +6.72, 15/15), best honest detector +0.090 (t +0.84)

Every one of those has a gap between "the label is valuable" and "the prediction pays",
because the label is a GEOMETRIC object (a pivot, a vol level) and the money is something
else. **This target closes that gap by construction: the label IS the economics.**
`y = sign(champion position x forward 5d return)` — "does the book's current position make
money from here". A model of that cannot be valuable-but-unmonetisable, because monetising it
is the definition of the label. The only question left is whether it is predictable.

WHAT IS NEW versus the two prior meta-labeling attempts (memory
`xau-per-trade-prob-sizing-disproven`, "meta-labeling fails twice"):
  1. Both were GOLD-ONLY. Tonight's one durable capability is that pooled cross-instrument
     training works and TRANSFERS (+0.0570 AUC on an arm that never saw gold; R^2 +0.232 on
     vol; and on tops it repaired a bad detector from -0.456 to -0.075). Never applied here.
  2. Neither was gated on control #8's economics-first protocol or an oracle ceiling.
  3. The prior best on this target was AUC 0.593 at H4 — described in the repo as the best ever
     achieved here on any label. This runs it at D1 across 51 instruments.

HONEST PRIOR. Meta-labeling has failed twice on gold, and ~130 pre-registered structures have
failed in this repo. The reason to run it anyway is the structural argument above, not optimism.

PROTOCOL. Oracle gate first (control #5), then GOLD-ONLY / POOLED / TRANSFER, economics at
MATCHED VOL with a paired-t and a per-year sign count (control #8). Two position rules:
  GATE   flat whenever P(profit) < threshold — the form that failed before.
  SIZE   position x P(profit) scaled continuously — never tried; §3r gated but never sized.

    python scripts/v5_pooled_champ_meta.py
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
from scripts.v5_pooled_vol_sizing import vol_features  # noqa: E402
from scripts.v5_xau_champion_lifts import champion_recipe  # noqa: E402
from scripts.v5_xau_turning_ml import features  # noqa: E402

H = 5                       # P&L horizon in bars
MIN_BARS = 1200
PURGE_DAYS = H + 1          # the label looks H bars ahead — the purge must cover it
COST_BP = 3.0


def build_panel() -> dict:
    """Reversal features + volatility features + the champion's own forecast, and the label
    `sign(champion position x forward H-bar return)`.

    The champion forecast is included as a FEATURE (queue #7, never done): the model must be
    able to condition on how strong the position it is judging actually is.
    """
    syms = sorted(p.name.replace("_D1_long.csv", "")
                  for p in (ROOT / "data").glob("*_D1_long.csv"))
    X, Y, S, T, PNL, FC = [], [], [], [], [], []
    kept = 0
    for s in syms:
        d = load_d1(s)
        if d is None:
            continue
        try:
            F1 = features(d)
            if "hour" in F1.columns:
                F1 = F1.drop(columns=["hour"])
            F2 = vol_features(d)
            fc = champion_recipe(d["close"], 1 / 6, 1.5)
        except Exception:
            continue
        F = pd.concat([F1, F2.add_prefix("v_")], axis=1)
        F["champ_fc"] = fc
        c = d["close"]
        ret = c.pct_change()
        ret = ret.clip(ret.quantile(0.001), ret.quantile(0.999))
        # forward H-bar return, then the P&L the CURRENT position would earn over it
        fwd = c.shift(-H) / c - 1
        pnl = fc * fwd
        y = (pnl > 0).astype(int)
        ok = (F.notna().all(axis=1).values & pnl.notna().values
              & (fc.abs() > 1e-6).values)          # only bars where a position exists
        if ok.sum() < MIN_BARS:
            continue
        kept += 1
        X.append(F[ok]); Y.append(y[ok].values); S.append(np.full(ok.sum(), s))
        T.append(d.index[ok]); PNL.append(pnl[ok].values); FC.append(fc[ok].values)
    print(f"panel: {kept} instruments, {sum(len(x) for x in X):,} bars with a live position")
    return dict(X=pd.concat(X), Y=np.concatenate(Y), S=np.concatenate(S),
                T=pd.DatetimeIndex(np.concatenate([t.values for t in T])),
                pnl=np.concatenate(PNL), fc=np.concatenate(FC))


def book(rule, S, T, restrict, cost_bp=COST_BP) -> dict:
    """Per-instrument champion book under a position rule. `rule(sym, index) -> multiplier`."""
    legs_b, legs_r = {}, {}
    for s in sorted(set(S)):
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
        mult = rule(s, m)
        for nm, pos in (("b", fc), ("r", fc * mult)):
            turn = pos.diff().abs().fillna(pos.abs())
            r = pos.shift(1).fillna(0.0) * ret - turn.shift(1).fillna(0.0) * cost_bp / 1e4
            (legs_b if nm == "b" else legs_r)[s] = r
    return dict(base=legs_b, rule=legs_r)


def paired_matched(x: pd.Series, y: pd.Series, bpy: float, label: str) -> tuple:
    both = pd.DataFrame(dict(x=x, y=y)).dropna()
    if len(both) < 100 or both["x"].std() == 0:
        print(f"{label:40s}  insufficient data"); return (np.nan, np.nan)
    sx = both["x"] * (both["y"].std() / both["x"].std())
    dl = sx - both["y"]
    t = dl.mean() / dl.std() * np.sqrt(len(dl))
    g = pd.DataFrame(dict(x=sx, y=both["y"])).groupby(both.index.year).mean()
    ds = perf(sx, bpy)["sr"] - perf(both["y"], bpy)["sr"]
    print(f"{label:40s}  dSharpe {ds:+.3f}   paired t {t:+6.2f}   "
          f"years {int((g['x']>g['y']).sum())}/{len(g)}")
    return (ds, t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-bp", type=float, default=COST_BP)
    a = ap.parse_args()
    t0 = time.time()
    P = build_panel()
    X, Y, S, T = P["X"], P["Y"], P["S"], P["T"]
    print(f"label = sign(champion position x forward {H}d return): "
          f"base rate {Y.mean():.4f}")
    print("(this label IS the economics — a valuable-but-unmonetisable gap cannot open here)")

    oos = np.isin(T.year, list(YEARS))

    # ================================================= 1. ORACLE GATE
    print("\n" + "=" * 84)
    print("STEP 1 — ORACLE GATE (control #5): flat whenever the position is about to lose.")
    print("=" * 84)
    ymap = {s: pd.Series(Y[(S == s) & oos].astype(float),
                         index=T[(S == s) & oos]) for s in set(S)}
    b = book(lambda s, m: ymap[s].reindex(T[m]).fillna(1.0), S, T, oos, a.cost_bp)
    bb = pd.concat(b["base"], axis=1).mean(axis=1).dropna()
    rr = pd.concat(b["rule"], axis=1).mean(axis=1).dropna()
    bpy = bars_per_year(bb.index)
    print(f"\npanel legs {len(b['base'])}   champion {perf(bb, bpy)['sr']:+.3f}   "
          f"oracle-gated {perf(rr, bpy)['sr']:+.3f}")
    paired_matched(rr, bb, bpy, "PANEL oracle-gate vs champion")
    if GOLD in b["base"]:
        paired_matched(b["rule"][GOLD], b["base"][GOLD],
                       bars_per_year(b["base"][GOLD].dropna().index),
                       "GOLD  oracle-gate vs champion")

    # ================================================= 2. DETECTOR ARMS
    print("\n" + "=" * 84)
    print("STEP 2 — pooled champ-meta: GOLD-ONLY / POOLED / TRANSFER")
    print("=" * 84)
    is_gold = S == GOLD
    arms = ("GOLD-ONLY", "POOLED", "TRANSFER")
    pr = {k: np.full(len(X), np.nan) for k in arms}
    for yr in YEARS:
        te = np.flatnonzero(T.year == yr)
        cut = pd.Timestamp(f"{yr}-01-01") - pd.Timedelta(days=PURGE_DAYS)
        trb = (T.year < yr) & (T <= cut)
        sel = {"GOLD-ONLY": trb & is_gold, "POOLED": trb, "TRANSFER": trb & ~is_gold}
        if len(te) < 50:
            continue
        for k, mtr in sel.items():
            tr = np.flatnonzero(mtr)
            if len(tr) < 400 or len(np.unique(Y[tr])) < 2:
                continue
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                               max_leaf_nodes=31, min_samples_leaf=40,
                                               l2_regularization=1.0, random_state=0)
            m.fit(X.values[tr], Y[tr])
            pr[k][te] = m.predict_proba(X.values[te])[:, 1]
        print(f"  {yr}: test {len(te):6d}  ({time.time()-t0:.0f}s)")

    print(f"\n{'arm':11s} {'AUC gold':>9s} {'SE':>7s} {'AUC panel':>10s} {'mean P':>8s}")
    for k in arms:
        if np.isnan(pr[k]).all():
            continue
        m = ~np.isnan(pr[k]); gi = m & is_gold
        ag = auc(pr[k][gi], Y[gi]) if gi.sum() > 200 else np.nan
        n1 = Y[gi].sum(); n0 = gi.sum() - n1
        print(f"{k:11s} {ag:9.4f} {hm_se(ag, n1, n0):7.4f} "
              f"{auc(pr[k][m], Y[m]):10.4f} {np.nanmean(pr[k][m]):8.4f}")
    print(f"\nprior best on this target anywhere in the repo: AUC 0.593 (H4, gold only)")

    # ================================================= 3. ECONOMICS
    print("\n" + "=" * 84)
    print("STEP 3 — ECONOMICS, matched vol. GATE (the form that failed twice) and SIZE.")
    print("=" * 84)
    for k in arms:
        if np.isnan(pr[k]).all():
            continue
        m = ~np.isnan(pr[k])
        pmap = {s: pd.Series(pr[k][(S == s) & m], index=T[(S == s) & m]) for s in set(S)}
        for nm, fn in (
            ("SIZE  (P/mean, continuous)",
             lambda s, mm: (pmap[s].reindex(T[mm]) / np.nanmean(pr[k][m]))
                            .clip(0, 2).ffill().fillna(1.0)),
            ("GATE  (flat if P<median)",
             lambda s, mm: (pmap[s].reindex(T[mm])
                            >= np.nanmedian(pr[k][m])).astype(float).ffill().fillna(1.0)),
        ):
            bk = book(fn, S, T, m, a.cost_bp)
            if not bk["base"]:
                continue
            b2 = pd.concat(bk["base"], axis=1).mean(axis=1).dropna()
            r2 = pd.concat(bk["rule"], axis=1).mean(axis=1).dropna()
            paired_matched(r2, b2, bars_per_year(b2.index), f"PANEL {k:11s} {nm}")
            if GOLD in bk["base"]:
                paired_matched(bk["rule"][GOLD], bk["base"][GOLD],
                               bars_per_year(bk["base"][GOLD].dropna().index),
                               f"GOLD  {k:11s} {nm}")
    print("\nDECISION: needs dSharpe >= +0.20 at t >= 2.0 with >= 11/15 years, at matched vol.")
    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
