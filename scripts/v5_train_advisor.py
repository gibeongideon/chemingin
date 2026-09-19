"""Freeze the XAUUSD advisor artifact. The MEASURED verdict is baked in, not re-derived live.

WHY THE ARTIFACT CARRIES THE VERDICT. `v5_advisor_verdict.py` decided that the direction panel is
TIER B (no measured edge) and the adverse panel is TIER B-plus TOP-TAIL-ONLY. Those are facts
about the measurement, not about the model weights, so a service that recomputed them at serve
time could disagree with V5_FINDINGS §3az/§3ba. Everything the card displays — the tier, the
`usable` flag, the reliability table, the operating points, the base rate, the resolution rate
and the literal advice strings — is written here, read-only, and the runtime prints it verbatim.

WHAT IS AND IS NOT FITTED HERE.
  * ADVERSE (shipped, warn-only): cell `k=1.0 ATR / 9h / BASE`, on the CLEAN single-feed dataset.
    §3ba withdrew the cell this file originally shipped: `1.0/4/SOURCE` was chosen for the only
    positive Brier skill of 24 cells (+0.0058) and an operating point of 0.646, both of which
    were artifacts of pairing FTMO H4 bars with another broker's M15. On one feed its Brier skill
    is -0.0065 and its operating point 0.548, and SOURCE loses to BASE in 8 of 8 cells.
    `1.0/9/BASE` is shipped because it produced the family-max reliability gap (+16.52pp, the
    statistic that cleared the best-of-16 null at p 0.010) AND its top bucket sits above the base
    rate in BOTH halves (0.596 / 0.536 against 0.490). `1.0/6/BASE` is closer to the 4-6h horizon
    the user asked for but FAILS that gate outright: its top bucket has no first-half
    observations at all, meaning the model never produced a high reading before 2022 — §3ap's
    killer exactly.
  * BASE, not SOURCE, is also strictly better operationally: no M15 at serve time, so the
    feed-mismatch and M15-staleness failure modes leave the live path entirely.
  * DIRECTION (shipped `usable: false`): fitted anyway, at h=2 (8h) / BASE, so the card can
    print a number ALONGSIDE the statement that it has no measured edge. Omitting it entirely
    would be less honest than showing it with its own refutation attached: the user asked for a
    direction call and is owed the measurement, not silence.

WHY THE FINAL FIT USES THE SAME THREE-WAY SPLIT AS THE WALK-FORWARD. The walk-forward's numbers
are only a valid description of THIS artifact if this artifact is built by the same procedure.
`_pick_and_fit` on a FIT slice, model class chosen on a purged SELECT slice, calibrator fitted on
a purged CALIBRATE tail — identical to `walk_forward`, just with the test slice being the
unknown future instead of a held-out year.

    python scripts/v5_train_advisor.py                     # writes data/models/xau_advisor.*
    python scripts/v5_train_advisor.py --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from scripts.v5_advisor_measure import (  # noqa: E402
    BLOCK, _pick_and_fit, _proba, features_for, load_frames,
)
from scripts.v5_range_multitf import atr_series  # noqa: E402
from scripts.v5_repr_ceiling import PURGE_EXTRA  # noqa: E402
from src.v5.advisor_calibration import (  # noqa: E402
    operating_points, prob_edges, reliability_table,
)
from src.v5.advisor_labels import climatology, first_touch_atr, fwd_dir, resolution_rate  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "xau_advisor"
MODEL = ROOT / "data" / "models" / "xau_advisor"

# ---- the shipped cells, fixed by the §3ba verdict. Changing these changes the product. ----
ADV = dict(k_atr=1.0, hours=9, arm="BASE")
DIR = dict(h=2, arm="BASE")
# 0.55, not 0.60: on the clean feed this cell's best measured lift is at 0.55 (observed 0.570,
# CI [0.530,0.596] against a 0.490 base, 10.6% coverage). 0.60 gives 0.563 on less coverage --
# no better, and on fewer events. Chosen from the operating curve, which is the statistic the
# product consumes (§3az's own lesson about picking within a qualified family).
ADV_WARN = 0.55
ADV_CLEAR = 0.50         # release below this: a 5pp dead-band, and still above the 0.490 base
CLEAN_FEED = True        # rebuild H4 from M15 so the label and the features are one broker


def _split(n: int, purge: int) -> tuple:
    """FIT / SELECT / CALIBRATE, purged, in the walk-forward's proportions (75% / 12% / tail)."""
    i1, i2 = int(n * 0.75), int(n * 0.87)
    fit = np.arange(0, i1)
    sel = np.arange(i1 + purge, i2) if i2 > i1 + purge else np.arange(i1, i2)
    cal = np.arange(i2 + purge, n) if n > i2 + purge else np.array([], int)
    return fit, sel, cal


def _fit_final(X: pd.DataFrame, y: np.ndarray, purge: int) -> dict:
    """The frozen estimator + calibrator, by the walk-forward's own procedure."""
    ok = X.notna().all(axis=1).values & np.isfinite(y)
    idx = np.flatnonzero(ok)
    Xv, yv = X.values[idx], y[idx]
    fit, sel, cal = _split(len(idx), purge)
    kind, model, scaler, ll = _pick_and_fit(Xv[fit], yv[fit], Xv[sel], yv[sel])
    calib, which = None, "none"
    if len(cal) >= 200 and 0 < yv[cal].mean() < 1:
        pc = _proba(kind, model, scaler, Xv[cal])
        if len(cal) >= 500:
            calib, which = IsotonicRegression(out_of_bounds="clip").fit(pc, yv[cal]), "isotonic"
        else:
            calib = LogisticRegression(C=1.0).fit(pc.reshape(-1, 1), yv[cal])
            which = "platt"
    return dict(model_class=kind, model=model, scaler=scaler, calibrator=calib,
                calibrator_kind=which, select_logloss=ll, features=list(X.columns),
                n_fit=int(len(fit)), n_select=int(len(sel)), n_calibrate=int(len(cal)),
                train_prior=climatology(yv[np.concatenate([fit, sel, cal])]
                                       if len(cal) else yv[np.concatenate([fit, sel])]))


def _measured(family: str, cell: str) -> dict:
    """Pull this cell's MEASURED numbers out of the Phase-1 artifacts. Nothing is recomputed:
    if the grid on disk disagrees with §3az, the mismatch must surface here, not be papered over
    by a fresh fit that happens to land somewhere else."""
    grid = pd.read_csv(OUT / ("adverse_grid_cleanfeed.csv" if family == "adv"
                              else "horizon_sweep.csv"))
    if family == "adv":
        r = grid[(grid.k_atr == ADV["k_atr"]) & (grid.hours == ADV["hours"])
                 & (grid.arm == ADV["arm"])]
    else:
        r = grid[(grid.h == DIR["h"]) & (grid.arm == DIR["arm"])]
    if not len(r):
        raise SystemExit(f"cell {cell} not in the Phase-1 grid — re-run v5_advisor_measure.py")
    r = r.iloc[0]
    suffix = "_cleanfeed" if family == "adv" else ""
    rel = pd.read_csv(OUT / f"reliability_{family}{suffix}.csv")
    rel = rel[rel.cell == cell]
    ops = pd.read_csv(OUT / f"operating_points_{family}{suffix}.csv")
    ops = ops[ops.cell == cell] if "cell" in ops.columns else ops
    keep = ["n", "base_rate", "acc", "acc_drift", "d_drift_pp", "d_drift_ci", "bss",
            "bss_ci_excludes_0", "auc", "ece", "years_drift", "resolved_frac"]
    out = {k: (json.loads(r[k]) if isinstance(r.get(k), str) and r[k].startswith("[")
               else (float(r[k]) if isinstance(r.get(k), (int, float, np.floating, np.integer))
                     else r.get(k)))
           for k in keep if k in r.index}
    out["reliability"] = rel.drop(columns=["cell"], errors="ignore").to_dict("records")
    out["operating_points"] = ops.drop(columns=["cell"], errors="ignore").to_dict("records")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(MODEL))
    a = ap.parse_args()

    verdict = json.loads((OUT / "verdict.json").read_text())
    if not verdict.get("tier_c"):
        raise SystemExit("TIER C did not pass — the harness is not replicating §3r/§3ao. Stop.")

    h4_filed, m15 = load_frames()
    if CLEAN_FEED:
        from scripts.v5_advisor_feedcheck import feed_divergence, rebuild_h4_from_m15
        h4 = rebuild_h4_from_m15(m15)
        ix = h4.index.intersection(h4_filed.index)
        d = (h4.loc[ix, "close"] - h4_filed.loc[ix, "close"]).abs()
        pre = d[ix.year <= 2022]
        if pre.median() > 0.01 or (pre <= 0.10).mean() < 0.99:
            raise SystemExit("rebuilt H4 does not reproduce the filed H4 pre-2023 — stop (§3ba)")
        print(f"CLEAN FEED: H4 rebuilt from M15; reconstruction verified on {len(pre):,} "
              f"pre-2023 bars (median {pre.median():.4f}, "
              f"{(pre <= 0.10).mean()*100:.1f}% within $0.10)")
        print(feed_divergence(h4_filed, m15).to_string(float_format=lambda x: f"{x:.4f}"))
    else:
        h4 = h4_filed
    atr = atr_series(h4, 20)
    ov = h4.index[(h4.index >= m15.index[0]) & (h4.index <= m15.index[-1])]

    # ---------------------------------------------------------------- adverse (the shipped one)
    nb = ADV["hours"] * 4
    ft = first_touch_atr(ov, h4.loc[ov, "close"].values, atr.loc[ov].values,
                         m15["high"].values, m15["low"].values, m15.index,
                         ADV["k_atr"], nb, dec_bar_minutes=240)
    rr = resolution_rate(ft)
    Xa = features_for(ADV["arm"], h4, m15).reindex(ov)
    adv = _fit_final(Xa, ft["y"].values, max(1, nb // 16) + PURGE_EXTRA)
    adv_cell = f"{ADV['k_atr']}_{ADV['hours']}_{ADV['arm']}"

    # ---------------------------------------------------------------- direction (usable: false)
    fd = fwd_dir(h4["close"], DIR["h"])
    Xd = features_for(DIR["arm"], h4, m15)
    dr = _fit_final(Xd, fd["y"].values, DIR["h"] + PURGE_EXTRA)
    dir_cell = f"{DIR['h']}_{DIR['arm']}"

    m_adv, m_dir = _measured("adv", adv_cell), _measured("dir", dir_cell)

    import sklearn
    meta = {
        "trained_on": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Recorded because a pickled estimator is only guaranteed to behave under the version
        # that wrote it. Training on a desktop at sklearn 1.6.1 and serving on a VPS at 1.7.2
        # makes sklearn itself warn that results may be invalid, and a silently wrong
        # probability is the one failure this whole product is built to avoid.
        "sklearn": sklearn.__version__,
        "python": sys.version.split()[0],
        "findings": "V5_FINDINGS.md §3az",
        "preregistration": "data/v5_runs/xau_advisor/PREREGISTRATION.md",
        "overall_verdict": verdict["overall"],
        "feed": ("CLEAN: H4 rebuilt from M15 so label and features are one broker (§3ba). "
                 "The filed XAUUSD_H4_long.csv splices FTMO bars from 2023-06-28 onward."
                 if CLEAN_FEED else "FILED (spliced) — see V5_FINDINGS §3ba"),
        "serve_needs_m15": ADV["arm"] == "SOURCE",
        "data": {"h4_last_bar": str(h4.index[-1]), "m15_last_bar": str(m15.index[-1]),
                 "overlap_bars": int(len(ov)),
                 "feed": "data/XAUUSD_{H4,M15}_long.csv — one feed family, never mixed"},

        "adverse": {
            "cell": adv_cell, **ADV,
            "tier": "B_plus_top_tail",
            "usable": "tails_only",
            "direction_of_label": "y=1 means the DOWN (-k*ATR) barrier is touched first",
            "warn_at": ADV_WARN, "clear_at": ADV_CLEAR,
            "resolved_frac": rr["resolved_frac"],
            "base_rate_measured": m_adv.get("base_rate"),
            "measured": m_adv,
            "advice_warn": ("elevated risk of an adverse move in the next ~4h — "
                            "consider trimming or exiting longs"),
            "advice_clear": "no elevated adverse-move risk measured",
            "must_never_say": ["low risk", "go short", "sell", "safe"],
            "limits": [
                f"speaks for only the {rr['resolved_frac']*100:.0f}% of 4h windows that resolve "
                f"+/-{ADV['k_atr']} ATR; the rest end inside the band and are not forecast",
                f"at p>={ADV_WARN} the measured adverse rate is "
                f"{[o for o in m_adv['operating_points'] if abs(o['threshold']-ADV_WARN)<1e-9][0]['observed']:.3f} "
                f"against a base rate of {m_adv.get('base_rate'):.3f} — roughly 1 in 3 of these "
                f"warnings still resolves the other way",
                "the LOW end is NOT trustworthy: the bottom bucket's 2018-21 frequency sits at "
                "~0.500 against a ~0.487 base, so a low reading means 'no signal', not 'safe'",
                f"Brier skill {m_adv.get('bss'):+.4f} with a CI spanning 0 — the RANKING is "
                "informative, the per-bar number is the measured bucket frequency",
                f"only {m_adv.get('years_drift')} years beat the base-rate rule",
            ],
        },

        "direction": {
            "cell": dir_cell, **DIR, "horizon_hours": DIR["h"] * 4,
            "tier": "B",
            "usable": False,
            "measured": m_dir,
            "edge_note": (
                "NO MEASURED EDGE. The 8-cell direction family has negative Brier skill in every "
                "cell and its best cell does not clear its own best-of-8 max-statistic null "
                "(reliability gap +8.04pp vs null p95 +10.76, p=0.175; AUC 0.5164 vs p95 0.5184, "
                "p=0.100). The number below is shown so it can be judged, not acted on."),
            "print_instead": "base_rate",
            "drift_note": (
                "Against 'assume gold rises' this model is +0.29pp at 4h, -0.42pp at 8h, "
                "-0.93pp at 12h and -1.33pp at 24h. Its published +2.12pp is against PERSISTENCE, "
                "a weak baseline on a drifting asset."),
        },

        "thresholds_are_margins_around_the_base_rate": (
            "Never threshold a calibrated probability at 0.5. At fwd6 gold's base rate is 0.536, "
            "so isotonic puts nearly every bar above 0.5 and a 0.5 cut IS the drift forecast "
            "(measured: acc 53.56% vs drift 53.62%)."),
        "no_order_path": ("This artifact is advisory. Nothing that imports it may construct an "
                          "order request; tests/test_v5_xau_advisor.py asserts the import closure "
                          "contains no order_send / TRADE_ACTION."),
    }

    print(f"ADVERSE  {adv_cell}: {adv['model_class']}/{adv['calibrator_kind']}  "
          f"fit {adv['n_fit']} sel {adv['n_select']} cal {adv['n_calibrate']}  "
          f"prior {adv['train_prior']:.4f}")
    print(f"         measured AUC {m_adv.get('auc'):.4f}  BSS {m_adv.get('bss'):+.4f}  "
          f"base {m_adv.get('base_rate'):.4f}  resolved {rr['resolved_frac']:.3f}")
    for o in m_adv["operating_points"]:
        star = "  <== shipped" if abs(o["threshold"] - ADV_WARN) < 1e-9 else ""
        print(f"           p>={o['threshold']:.2f}  cov {o['coverage']*100:5.2f}%  "
              f"n {int(o['n']):5d}  observed {o['observed']:.4f}{star}")
    print(f"DIRECTION {dir_cell}: {dr['model_class']}/{dr['calibrator_kind']}  "
          f"prior {dr['train_prior']:.4f}  -> usable: false (tier B)")
    print(f"h4 last bar {h4.index[-1]}   m15 last bar {m15.index[-1]}")

    if a.dry_run:
        print("\ndry-run: nothing written")
        return

    import joblib
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"adverse": adv, "direction": dr}, p.with_suffix(".joblib"))
    blob = json.dumps(meta, indent=2, default=float)
    meta["model_id"] = hashlib.sha256(
        p.with_suffix(".joblib").read_bytes()).hexdigest()[:12]
    p.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=float))
    print(f"\nwrote {p.with_suffix('.joblib')}  (model_id {meta['model_id']})")
    print(f"      {p.with_suffix('.json')}")


if __name__ == "__main__":
    main()
