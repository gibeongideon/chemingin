"""MEASUREMENT HARNESS for the XAUUSD direction advisor. Pre-registered in
`data/v5_runs/xau_advisor/PREREGISTRATION.md` (committed 6df4b12) — read it before changing
anything here, and amend it by appending rather than editing.

    python scripts/v5_advisor_measure.py --tier-c              # the gate; run this first
    python scripts/v5_advisor_measure.py --family adv          # 8 adverse cells
    python scripts/v5_advisor_measure.py --family dir          # 8 direction cells
    python scripts/v5_advisor_measure.py --family dir --null alignment --draws 200

TIER C RUNS FIRST AND BLOCKS EVERYTHING. `--family` refuses to run unless the replication
artifact exists and passed. Without it a null result is uninterpretable — it could be the
pipeline rather than the market, which is the trap §3av fell into from the opposite direction.

THE WALK-FORWARD, and the one piece of genuinely new machinery. Expanding yearly refits, purge
`h + PURGE_EXTRA` between train and test, and a THREE-WAY purged split inside train:

    train[0:75%]        fit the classifier
    gap (h + 8)
    train[75%:87%]      SELECT the model class by log-loss
    gap (h + 8)
    train[87%:]         fit the calibrator

The third slice exists because `oos_prob_all_bars` (`v5_m15_trim_overlay.py:57`) selects between
HistGB and logistic on the last 400 training rows — exactly where an isotonic calibrator would
otherwise be fitted. Calibrating on the slice that chose the model is a double-use that makes
the probability look better than it is. Calibrator by slice size: >=500 isotonic, >=200 Platt,
otherwise none and the cell is inadmissible for Tier A (PREREGISTRATION §5.3).

BASELINES. Three, always, on the identical OOS event set: PERSISTENCE (comparability to §3r),
DRIFT (train-prior "always up" — the co-primary, because the advisor's reader defaults to drift,
not persistence, and §3r's +2.12pp becomes -1.3pp against it), CLIMATOLOGY (the Brier reference).
"""
from __future__ import annotations

import argparse
import json
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
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from scripts.v5_range_multitf import atr_series  # noqa: E402
from scripts.v5_repr_ceiling import PURGE_EXTRA, f_base, f_source, first_touch  # noqa: E402
from src.v5.advisor_calibration import (  # noqa: E402
    block_bootstrap_ci, brier, brier_decomposition, brier_skill, ece, mce,
    operating_points, prob_edges, reliability_table,
)
from src.v5.advisor_labels import (  # noqa: E402
    climatology, drift_baseline, first_touch_atr, fwd_dir, resolution_rate,
)

OUT = ROOT / "data" / "v5_runs" / "xau_advisor"
YEARS = list(range(2018, 2027))
BLOCK = 60
MIN_DECIDED, MIN_NONOVERLAP = 1000, 400        # PREREGISTRATION §6

# PREREGISTRATION §2 — the declared grids. Do not extend.
DIR_CELLS = [(h, arm) for h in (1, 2, 3, 6) for arm in ("BASE", "SOURCE")]
ADV_CELLS = [(k, hz) for k, hz in
             ((0.5, 4), (0.5, 6), (0.5, 9), (0.75, 6), (0.75, 9), (1.0, 6), (1.0, 9), (1.0, 4))]
ADV_PRIMARY = (0.5, 6)


# ----------------------------------------------------------------------------- data
def load_frames() -> tuple:
    """H4 + M15 from the SAME feed family. Never mixed: `data/XAUUSD_*_long.csv` and
    `data/v5_runs/range_study/*` differ by a median $5.84, rising to $13.33 in 2026."""
    def rd(p):
        d = pd.read_csv(p, parse_dates=["time"], index_col="time").sort_index()
        d = d[~d.index.duplicated(keep="last")]
        d.columns = [c.lower() for c in d.columns]
        return d
    h4 = rd(ROOT / "data/XAUUSD_H4_long.csv")
    m15 = rd(ROOT / "data/XAUUSD_M15_long.csv")
    return h4, m15


# ----------------------------------------------------------------------------- model
def _pick_and_fit(Xf, yf, Xs, ys, seed=7):
    """Fit both candidate classes on the FIT slice, choose by log-loss on the SELECT slice."""
    out = []
    hg = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=8, l2_regularization=10.0,
                                        learning_rate=0.05, min_samples_leaf=40,
                                        random_state=seed).fit(Xf, yf)
    out.append(("histgb", hg, None, _ll(ys, hg.predict_proba(Xs)[:, 1])))
    sc = StandardScaler().fit(Xf)
    lr = LogisticRegression(C=0.05, max_iter=2000).fit(sc.transform(Xf), yf)
    out.append(("logit", lr, sc, _ll(ys, lr.predict_proba(sc.transform(Xs))[:, 1])))
    out.sort(key=lambda r: r[3])
    return out[0]


def _proba(kind, model, scaler, X):
    return model.predict_proba(scaler.transform(X) if scaler is not None else X)[:, 1]


def _ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def walk_forward(X: pd.DataFrame, y: np.ndarray, purge: int, index) -> dict:
    """Expanding yearly refits with the three-way purged split. Returns pooled OOS arrays."""
    T = pd.DatetimeIndex(index)
    ok = X.notna().all(axis=1).values & np.isfinite(y)
    p_raw = np.full(len(y), np.nan)
    p_cal = np.full(len(y), np.nan)
    drift = np.full(len(y), np.nan)
    clim = np.full(len(y), np.nan)
    folds = []
    Xv = X.values
    for yr in YEARS:
        te = np.flatnonzero((T.year == yr) & ok)
        tr = np.flatnonzero((T.year < yr) & ok)
        if len(tr) > purge:
            tr = tr[:-purge]                                  # purge the label window
        if len(tr) < 600 or len(te) < 60:
            continue
        n = len(tr)
        i1, i2 = int(n * 0.75), int(n * 0.87)
        fit = tr[:i1]
        sel = tr[i1 + purge:i2] if i2 > i1 + purge else tr[i1:i2]
        cal = tr[i2 + purge:] if len(tr) > i2 + purge else np.array([], int)
        if len(fit) < 300 or len(sel) < 50 or len(np.unique(y[fit])) < 2:
            continue
        kind, model, scaler, _ = _pick_and_fit(Xv[fit], y[fit], Xv[sel], y[sel])
        pr = _proba(kind, model, scaler, Xv[te])
        p_raw[te] = pr
        # calibrator by slice size (PREREGISTRATION §5.3)
        which = "none"
        if len(cal) >= 200 and 0 < y[cal].mean() < 1:
            pc = _proba(kind, model, scaler, Xv[cal])
            if len(cal) >= 500:
                p_cal[te] = IsotonicRegression(out_of_bounds="clip").fit(pc, y[cal]).predict(pr)
                which = "isotonic"
            else:
                lr = LogisticRegression(C=1.0).fit(pc.reshape(-1, 1), y[cal])
                p_cal[te] = lr.predict_proba(pr.reshape(-1, 1))[:, 1]
                which = "platt"
        else:
            p_cal[te] = pr
        drift[te] = drift_baseline(y[tr])
        clim[te] = climatology(y[tr])
        folds.append(dict(year=yr, n_train=len(tr), n_fit=len(fit), n_sel=len(sel),
                          n_cal=len(cal), n_test=len(te), model=kind, calibrator=which))
    return dict(p_raw=p_raw, p_cal=p_cal, drift=drift, clim=clim, folds=folds,
                mask=np.isfinite(p_raw))


# ----------------------------------------------------------------------------- scoring
def nonoverlap(idx: np.ndarray, h: int) -> np.ndarray:
    """Greedy subset at least `h` positions apart — the honest-units selector from
    `walk_forward_ll` (`v5_repr_ceiling.py:207-212`)."""
    keep, last = [], -10**9
    for i in idx:
        if i - last >= h:
            keep.append(i); last = i
    return np.array(keep, int)


def score(y, p_cal, persistence, drift, clim, index, h, kind="direction") -> dict:
    """Every number the gates in PREREGISTRATION §7 are stated in terms of."""
    acc = ((p_cal >= 0.5).astype(float) == y).astype(float)
    acc_p = (persistence == y).astype(float)
    acc_d = (drift == y).astype(float)
    d_pers, d_drift = acc - acc_p, acc - acc_d
    lo_p, hi_p, se_p = block_bootstrap_ci(d_pers, block=BLOCK, alpha=0.10)
    lo_d, hi_d, se_d = block_bootstrap_ci(d_drift, block=BLOCK, alpha=0.10)
    bs_diff = (clim - y) ** 2 - (p_cal - y) ** 2          # positive => model better
    lo_b, hi_b, _ = block_bootstrap_ci(bs_diff, block=BLOCK, alpha=0.10)
    ref = float(np.mean(clim))
    bss = brier_skill(y, p_cal, ref)
    T = pd.DatetimeIndex(index)
    per_yr = pd.DataFrame(dict(d_pers=d_pers, d_drift=d_drift), index=T).groupby(T.year).mean()
    dec = brier_decomposition(y, p_cal, prob_edges(kind))
    r = pd.Series(p_cal).rank().values
    n1, n0 = y.sum(), len(y) - y.sum()
    auc = float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan
    return dict(
        n=int(len(y)), base_rate=float(np.mean(y)), acc=float(np.mean(acc)),
        acc_persistence=float(np.mean(acc_p)), acc_drift=float(np.mean(acc_d)),
        d_pers_pp=float(np.mean(d_pers) * 100), d_pers_ci=[lo_p * 100, hi_p * 100],
        d_drift_pp=float(np.mean(d_drift) * 100), d_drift_ci=[lo_d * 100, hi_d * 100],
        years_pers=f"{int((per_yr.d_pers > 0).sum())}/{len(per_yr)}",
        years_drift=f"{int((per_yr.d_drift > 0).sum())}/{len(per_yr)}",
        brier=brier(y, p_cal), brier_ref=brier(y, clim), bss=bss,
        bss_ci_excludes_0=bool(lo_b > 0), bss_diff_ci=[lo_b, hi_b],
        auc=auc, ece=ece(y, p_cal, prob_edges(kind)), mce=mce(y, p_cal, prob_edges(kind)),
        reliability_term=dec["reliability"], resolution_term=dec["resolution"],
    )


# ----------------------------------------------------------------------------- features
def features_for(arm: str, h4: pd.DataFrame, m15: pd.DataFrame) -> pd.DataFrame:
    return f_base(h4) if arm == "BASE" else f_source(h4, m15)


# ----------------------------------------------------------------------------- TIER C
def _replicate_3r(X: pd.DataFrame, y: np.ndarray, persistence: np.ndarray,
                  index, h: int) -> dict:
    """§3r's protocol EXACTLY, for the replication gate only.

    One HistGB per fold on ALL purged strictly-past bars, **raw** `predict_proba`, 0.5 threshold.
    Deliberately NOT the advisor's three-way-split calibrated pipeline: comparing a calibrated
    probability at a 0.5 threshold against §3r's accuracy is a different quantity, and
    AMENDMENT 1b records what that mismatch did — the calibrated probability collapsed into the
    drift baseline (acc 53.56% vs drift 53.62%), because at fwd6 the base rate is 0.536 so an
    isotonic map puts nearly every bar above 0.5.
    """
    T = pd.DatetimeIndex(index)
    ok = X.notna().all(axis=1).values & np.isfinite(y)
    pr = np.full(len(y), np.nan)
    Xv = X.values
    for yr in YEARS:
        te = np.flatnonzero((T.year == yr) & ok)
        tr = np.flatnonzero((T.year < yr) & ok)
        if len(tr) > h + PURGE_EXTRA:
            tr = tr[:-(h + PURGE_EXTRA)]
        if len(tr) < 600 or len(te) < 60 or len(np.unique(y[tr])) < 2:
            continue
        m = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=8,
                                           l2_regularization=10.0, learning_rate=0.05,
                                           min_samples_leaf=40, random_state=0)
        m.fit(Xv[tr], y[tr])
        pr[te] = m.predict_proba(Xv[te])[:, 1]
    m = np.isfinite(pr) & np.isfinite(y)
    acc = ((pr[m] >= 0.5).astype(float) == y[m]).astype(float)
    acc_p = (persistence[m] == y[m]).astype(float)
    d = acc - acc_p
    lo, hi, _ = block_bootstrap_ci(d, block=BLOCK, alpha=0.10)
    yr_tbl = pd.Series(d, index=T[m]).groupby(T[m].year).mean()
    return dict(n=int(m.sum()), acc=float(np.mean(acc)), acc_persistence=float(np.mean(acc_p)),
                base_rate=float(np.mean(y[m])), d_pers_pp=float(np.mean(d) * 100),
                d_pers_ci=[lo * 100, hi * 100],
                years_pers=f"{int((yr_tbl > 0).sum())}/{len(yr_tbl)}")


def tier_c(h4, m15) -> dict:
    """The replication gate. Blocks every candidate cell until it passes.

    Targets per AMENDMENT 1: Family D replicates §3r's RAW-probability protocol; Family A uses
    §3ao's own `walk_forward_ll` at **K=30**, the headline cell (K=12 is only the CLI default).
    """
    print("=" * 78)
    print("TIER C — HARNESS REPLICATION. Nothing else may be read until this passes.")
    print("=" * 78)
    res = {}

    # --- Family D: 3r's protocol, raw proba, h=6 / BASE -> +2.12pp over persistence
    X = features_for("BASE", h4, m15)
    fd = fwd_dir(h4["close"], 6)
    y = fd["y"].values
    s = _replicate_3r(X, y, fd["persistence"].values, h4.index, 6)
    ok_d = 1.10 <= s["d_pers_pp"] <= 3.10
    res["direction"] = dict(**s, passed=bool(ok_d), window=[1.10, 3.10],
                            protocol="3r verbatim: single HistGB, raw proba, 0.5 threshold")
    print(f"\nFamily D  h=6 BASE, 3r protocol (raw proba)   n {s['n']}")
    print(f"  acc {s['acc']*100:.2f}%   persistence {s['acc_persistence']*100:.2f}%   "
          f"base rate {s['base_rate']*100:.2f}%")
    print(f"  d_persistence {s['d_pers_pp']:+.2f}pp  CI90 "
          f"[{s['d_pers_ci'][0]:+.2f},{s['d_pers_ci'][1]:+.2f}]  years {s['years_pers']}")
    print(f"  -> window [+1.10,+3.10]pp : "
          f"{'PASS' if ok_d else '*** FAIL ***'}   (3r recorded +2.12pp / 51.97% / 8-9 yrs)")

    # --- Family A: 3ao's own walk_forward_ll at K=30, the headline cell (AMENDMENT 1a)
    from scripts.v5_repr_ceiling import walk_forward_ll
    K = 30
    yft = pd.Series(first_touch(h4, K), index=h4.index)
    rb = walk_forward_ll(features_for("BASE", h4, m15), yft, K)
    rs = walk_forward_ll(features_for("SOURCE", h4, m15), yft, K)
    llb = float(np.average(rb.ll, weights=rb.n))
    lls = float(np.average(rs.ll, weights=rs.n))
    dI = (llb - lls) / np.log(2)
    auc_s = float(np.average(rs.auc, weights=rs.n))
    mrg = rb.merge(rs, on="year", suffixes=("_b", "_s"))
    yrs = int((mrg.ll_s < mrg.ll_b).sum())
    ok_a = abs(dI - 0.0142) <= 0.0100
    res["adverse"] = dict(K=K, dI_bits=dI, target=0.0142, tol=0.0100, auc_source=auc_s,
                          n_base=int(rb.n.sum()), n_source=int(rs.n.sum()),
                          years=f"{yrs}/{len(mrg)}", passed=bool(ok_a),
                          protocol="3ao walk_forward_ll verbatim (selects model on test set)")
    print(f"\nFamily A  3ao headline cell K=30, fixed 2%, SOURCE vs BASE")
    print(f"  n BASE {int(rb.n.sum())} / SOURCE {int(rs.n.sum())}   AUC_source {auc_s:.4f}   "
          f"years {yrs}/{len(mrg)}")
    print(f"  dI {dI:+.4f} bits/event   target +0.0142 +/- 0.0100 : "
          f"{'PASS' if ok_a else '*** FAIL ***'}")
    print(f"  (3ao recorded dI +0.0142, AUC 0.603, 7/9 years, n 251)")

    res["passed"] = bool(ok_d and ok_a)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tier_c.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\nTIER C: {'PASS — candidate cells may now be read' if res['passed'] else '*** FAIL — STOP AND DEBUG ***'}")
    print(f"wrote {(OUT / 'tier_c.json').relative_to(ROOT)}")
    return res


def require_tier_c():
    f = OUT / "tier_c.json"
    if not f.exists():
        raise SystemExit("TIER C has not been run. `--tier-c` first (PREREGISTRATION §4).")
    if not json.loads(f.read_text()).get("passed"):
        raise SystemExit("TIER C FAILED. Candidate cells are uninterpretable until it passes.")


# ----------------------------------------------------------------------------- families
def run_direction(h4, m15) -> pd.DataFrame:
    require_tier_c()
    rows, store = [], {}
    for h, arm in DIR_CELLS:
        t0 = time.time()
        X = features_for(arm, h4, m15)
        fd = fwd_dir(h4["close"], h)
        y = fd["y"].values
        wf = walk_forward(X, y, h + PURGE_EXTRA, h4.index)
        m = wf["mask"] & np.isfinite(y)
        if m.sum() < MIN_DECIDED:
            rows.append(dict(h=h, arm=arm, n=int(m.sum()), admissible=False))
            continue
        nov = nonoverlap(np.flatnonzero(m), h)
        s = score(y[m], wf["p_cal"][m], fd["persistence"].values[m], wf["drift"][m],
                  wf["clim"][m], h4.index[m], h)
        s.update(h=h, hours=h * 4, arm=arm, n_nonoverlap=int(len(nov)),
                 admissible=bool(len(nov) >= MIN_NONOVERLAP),
                 replication=bool(h == 6 and arm == "BASE"), secs=round(time.time() - t0, 1))
        rows.append(s)
        store[f"{h}_{arm}"] = dict(idx=h4.index[m], y=y[m], p=wf["p_cal"][m])
        print(f"  h={h} ({h*4}h) {arm:6s} n {s['n']:6d}  acc {s['acc']*100:5.2f}%  "
              f"pers {s['acc_persistence']*100:5.2f}%  drift {s['acc_drift']*100:5.2f}%  "
              f"dP {s['d_pers_pp']:+5.2f}  dD {s['d_drift_pp']:+5.2f}  "
              f"BSS {s['bss']:+.4f}  AUC {s['auc']:.4f}")
    return pd.DataFrame(rows), store


def run_adverse(h4, m15) -> tuple:
    require_tier_c()
    a = atr_series(h4, 20)
    ov = h4.index[(h4.index >= m15.index[0]) & (h4.index <= m15.index[-1])]
    rows, store = [], {}
    for k, hz in ADV_CELLS:
        nb = hz * 4                                  # M15 bars in `hz` hours
        # dec_bar_minutes=240: an H4 bar stamped t closes at t+4h, so the path must start
        # there. Without it the path walks the bars that formed the decision close and the
        # label scores AUC 0.90 (see first_touch_atr's docstring).
        ft = first_touch_atr(ov, h4.loc[ov, "close"].values, a.loc[ov].values,
                             m15["high"].values, m15["low"].values, m15.index, k, nb,
                             dec_bar_minutes=240)
        rr = resolution_rate(ft)
        for arm in ("BASE", "SOURCE"):
            X = features_for(arm, h4, m15).reindex(ov)
            y = ft["y"].values
            # nb is in M15 bars; one H4 bar is 16 of them, so the label spans nb/16 H4 bars
            wf = walk_forward(X, y, max(1, nb // 16) + PURGE_EXTRA, ov)
            m = wf["mask"] & np.isfinite(y)
            if m.sum() < MIN_DECIDED:
                rows.append(dict(k_atr=k, hours=hz, arm=arm, n=int(m.sum()),
                                 admissible=False)); continue
            hbars = max(1, nb // 16)
            nov = nonoverlap(np.flatnonzero(m), hbars)
            # A first-touch label has no "repeat the last move" analogue, so the comparability
            # slot is filled with the BASE-RATE rule. That makes d_pers == d_drift here by
            # construction, and the honest reading of the adverse gate is therefore "does it
            # beat the base rate", which is the gate that matters anyway (§3ay).
            per = np.full(m.sum(), 1.0 if rr["base_rate"] > 0.5 else 0.0)
            s = score(y[m], wf["p_cal"][m], per, wf["drift"][m], wf["clim"][m],
                      ov[m], hbars, kind="adverse")
            s.update(k_atr=k, hours=hz, arm=arm, n_nonoverlap=int(len(nov)),
                     resolved_frac=rr["resolved_frac"], ambiguous=rr["ambiguous"],
                     admissible=bool(len(nov) >= MIN_NONOVERLAP and rr["resolved_frac"] >= 0.20),
                     primary=bool((k, hz) == ADV_PRIMARY))
            rows.append(s)
            store[f"{k}_{hz}_{arm}"] = dict(idx=ov[m], y=y[m], p=wf["p_cal"][m])
            print(f"  k={k:<4} {hz}h {arm:6s} resolved {rr['resolved_frac']:.3f}  "
                  f"n {s['n']:6d}  base {s['base_rate']:.4f}  acc {s['acc']*100:5.2f}%  "
                  f"AUC {s['auc']:.4f}  BSS {s['bss']:+.4f}")
    return pd.DataFrame(rows), store


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier-c", action="store_true")
    ap.add_argument("--family", choices=["dir", "adv"])
    ap.add_argument("--null", choices=["none", "alignment", "synthetic", "maxstat"],
                    default="none")
    ap.add_argument("--draws", type=int, default=200)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    h4, m15 = load_frames()
    print(f"H4 {len(h4)} bars {h4.index[0].date()}..{h4.index[-1].date()}   "
          f"M15 {len(m15)} bars {m15.index[0].date()}..{m15.index[-1].date()}")

    if a.tier_c:
        tier_c(h4, m15)
        return
    if not a.family:
        raise SystemExit("pass --tier-c or --family {dir,adv}")

    if a.family == "dir":
        print("\nFamily D — DIRECTION, 8 declared cells")
        df, store = run_direction(h4, m15)
        df.to_csv(OUT / "horizon_sweep.csv", index=False)
    else:
        print("\nFamily A — ADVERSE MOVE, 8 declared cells")
        df, store = run_adverse(h4, m15)
        df.to_csv(OUT / "adverse_grid.csv", index=False)

    # reliability + operating points for every admissible cell
    kind = "direction" if a.family == "dir" else "adverse"
    rel_all, op_all = [], []
    for key, d in store.items():
        t = reliability_table(d["y"], d["p"], edges=prob_edges(kind), index=d["idx"],
                              n_boot=1000)
        t.insert(0, "cell", key)
        rel_all.append(t)
        o = operating_points(d["p"], d["y"], index=d["idx"], hysteresis=0.02, n_boot=500)
        o.insert(0, "cell", key)
        op_all.append(o)
    if rel_all:
        pd.concat(rel_all).to_csv(OUT / f"reliability_{a.family}.csv", index=False)
        pd.concat(op_all).to_csv(OUT / f"operating_points_{a.family}.csv", index=False)
    # Persist the per-cell OOS arrays so the max-statistic null can re-pair fitted scores
    # against a shuffled label without refitting 16 cells x 200 draws (which would be hours).
    # That prices the SELECTION, which is what a max-statistic null is for; the fitting bias is
    # priced separately by the synthetic control.
    flat = []
    for k, v in store.items():
        flat.append(pd.DataFrame(dict(cell=k, time=pd.DatetimeIndex(v["idx"]),
                                      y=v["y"], p=v["p"])))
    if flat:
        pd.concat(flat, ignore_index=True).to_parquet(OUT / f"oos_{a.family}.parquet")
        print(f"  persisted OOS scores -> oos_{a.family}.parquet "
              f"({sum(len(f) for f in flat):,} rows, {len(flat)} cells)")
    print(f"\nwrote {(OUT).relative_to(ROOT)}/  "
          f"{'horizon_sweep' if a.family=='dir' else 'adverse_grid'}.csv, "
          f"reliability_{a.family}.csv, operating_points_{a.family}.csv")


if __name__ == "__main__":
    main()
