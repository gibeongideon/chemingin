"""ARM 2 — bounded model-class sweep on the H1 ZigZag bottom label. CLOSURE, not a search.

K = 6 DECLARED CELLS, one fixed config per class, NO grids. Pre-registered.

WHY THIS IS BOUNDED. MANDATORY CONTROL #7: the expected AUC lift from the best of K pure-noise
candidates grows with K, so a wide sweep provably raises the bar faster than the statistic
(SS3ak: 21,300 triggers, observed max |z| 11.52 against a null MEDIAN of 9.34, Reality-Check
p 0.110 -- "just test more rules" is a strictly losing move). Small param grids are where K
explodes, so there are none.

WHY THE VERDICT STATISTIC IS NOT AUC. MANDATORY CONTROL #8: pooled training lifted this very
detector's AUC by 5.9 standard errors in 15 of 15 years and produced -0.702 Sharpe. AUC is
payoff-blind. The reported verdict is dEV/fire and dmean-offset; dAUC is printed and cannot
promote anything.

TRAPS GUARDED (each one a logged mistake or a known footgun):
  * identical walk-forward schedule for every class, so the comparison is apples to apples;
  * NO early stopping with an eval_set -- drawing one from the test side is silently fatal, and
    the sibling defect is already in the log (v5_repr_ceiling.py:219-232 selects the model class
    on the test fold, worth ~+0.006 bits and biased toward the higher-capacity arm);
  * CatBoost train_dir pinned to the run directory, or it writes catboost_info/ into the repo;
  * purge 8 = TOL + order, the label's true dependency reach.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from v5_xau_turning_ml import features, zigzag_swings, label_near, atr    # noqa: E402
from v5_pooled_bottom_detector import auc, hm_se                           # noqa: E402
from v5_zigzag_atr_bracket import load, COST_USD, ZZ_ORDER, ZZ_THETA, TOL  # noqa: E402

OUT = ROOT / "data/v5_runs/zigzag_atr"
TRAIN_BARS, PURGE, REFIT_EVERY, MIN_TRAIN = 20_000, TOL + ZZ_ORDER, 1000, 4_000
SEED = 7


def make(name):
    """ONE fixed configuration per class. No tuning, by design -- see the module docstring."""
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=0.05, max_iter=2000, random_state=SEED))
    if name == "histgb":        # the INCUMBENT, frozen at the deployed settings = the control
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                              max_leaf_nodes=31, random_state=0)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                             reg_lambda=1.0, tree_method="hist", n_jobs=4,
                             random_state=SEED, eval_metric="logloss")
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                              min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                              n_jobs=4, random_state=SEED, verbose=-1)
    if name == "catboost":      # NEVER fitted on XAU before this run
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05,
                                  l2_leaf_reg=3.0, random_seed=SEED, verbose=0,
                                  allow_writing_files=False,
                                  train_dir=str(OUT / "_catboost"))
    raise ValueError(name)


CLASSES = ["histgb", "logreg", "xgboost", "lightgbm", "catboost"]


def main() -> int:
    h1 = load("H1")
    A = atr(h1)
    _, buys = zigzag_swings(h1, ZZ_ORDER, ZZ_THETA * A)
    y = label_near(buys, len(h1), TOL)
    F = features(h1).drop(columns=["hour"])
    ok = F.notna().all(axis=1).values
    X = F.values
    close = h1["close"].values
    piv = np.zeros(len(h1), bool); piv[buys] = True

    st = np.load(OUT / "_outcomes.npy", allow_pickle=True).item()
    cell = "ATR tp1.0/sl1.0"          # the best DETECTOR cell from Z3, fixed before this run
    r = st[cell]

    starts = list(range(TRAIN_BARS, len(h1), REFIT_EVERY))
    print("=" * 96)
    print(f"ARM 2 -- bounded model sweep, K={len(CLASSES)+1} declared cells, no grids")
    print(f"  label: H1 ZigZag bottom (order {ZZ_ORDER}, theta {ZZ_THETA}xATR, tol +/-{TOL})")
    print(f"  walk-forward: {len(starts)} refits x {TRAIN_BARS:,} trailing bars, purge {PURGE}")
    print(f"  bracket for the economics: {cell}")
    print("=" * 96)

    rows, probs = [], {}
    for name in CLASSES:
        p = np.full(len(h1), np.nan)
        import time as _t
        t0 = _t.time()
        for s in starts:
            e = min(s + REFIT_EVERY, len(h1))
            hi = s - PURGE
            lo = max(0, hi - TRAIN_BARS)
            tr = np.arange(lo, hi)[ok[lo:hi]]
            if len(tr) < MIN_TRAIN or y[tr].sum() < 50:
                continue
            m = make(name)
            m.fit(X[tr], y[tr])
            te = np.arange(s, e)[ok[s:e]]
            if len(te):
                p[te] = m.predict_proba(X[te])[:, 1]
        probs[name] = p
        sc = np.isfinite(p)
        a = auc(p[sc], y[sc])
        n1, n0 = int(y[sc].sum()), int((1 - y[sc]).sum())
        se = hm_se(a, n1, n0)

        # ---- the verdict statistics: CONTROL #8, not AUC
        thr = float(np.nanquantile(p[sc], 1 - 0.078))   # fire rate matched to the incumbent @0.60
        fires = np.flatnonzero(sc & (p >= thr) & np.isfinite(r["exit_px"]))
        allb = np.flatnonzero(sc & np.isfinite(r["exit_px"]))
        ev = float(np.mean((r["exit_px"][fires] - close[fires] - COST_USD) / close[fires]))
        evb = float(np.mean((r["exit_px"][allb] - close[allb] - COST_USD) / close[allb]))
        dec = fires[r["outcome"][fires] != 0]
        decb = allb[r["outcome"][allb] != 0]
        hit = float(np.mean(r["outcome"][dec] > 0))
        base = float(np.mean(r["outcome"][decb] > 0))
        nxt = np.searchsorted(buys, fires)
        offs = [f - min([buys[j] for j in (k - 1, k) if 0 <= j < len(buys)],
                        key=lambda b: abs(f - b)) for f, k in zip(fires, nxt)]
        rows.append(dict(model=name, auc=round(a, 4), se=round(se, 4), fires=len(fires),
                         hit=round(hit, 4), base=round(base, 4), skill=round(hit - base, 4),
                         ev_fire_bp=round(1e4 * ev, 2),
                         ev_minus_anybar_bp=round(1e4 * (ev - evb), 2),
                         mean_offset=round(float(np.mean(offs)), 3),
                         secs=round(_t.time() - t0, 1)))
        print(f"  {name:<10} AUC {a:.4f} (SE {se:.4f})  skill {hit-base:+.4f}  "
              f"EV-anybar {1e4*(ev-evb):+.2f}bp  offset {np.mean(offs):+.3f}  "
              f"[{_t.time()-t0:.0f}s]")

    tb = pd.DataFrame(rows)
    inc = tb[tb.model == "histgb"].iloc[0]
    tb["dAUC"] = (tb.auc - inc.auc).round(4)
    tb["dEV_bp"] = (tb.ev_minus_anybar_bp - inc.ev_minus_anybar_bp).round(2)
    tb["dOffset"] = (tb.mean_offset - inc.mean_offset).round(3)
    tb.to_csv(OUT / "arm2_model_sweep.csv", index=False)

    # best-of-K pure-noise AUC floor at this SE (CONTROL #7) -- what a sweep this wide buys free
    se = float(inc.se)
    draws = rng = np.random.default_rng(3)
    floor = {k: float(np.mean(np.max(rng.normal(0, se, size=(20000, k)), axis=1)))
             for k in (2, 3, 6, 10, 20)}
    print("\n  best-of-K pure-noise AUC lift at SE %.4f (CONTROL #7):" % se)
    print("   " + "  ".join(f"K={k}: +{v:.4f}" for k, v in floor.items()))
    print("\n" + tb[["model", "auc", "dAUC", "skill", "ev_minus_anybar_bp", "dEV_bp",
                     "mean_offset", "dOffset"]].to_string(index=False))
    np.save(OUT / "_sweep_probs.npy", probs, allow_pickle=True)
    print(f"\n  wrote {OUT}/arm2_model_sweep.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
