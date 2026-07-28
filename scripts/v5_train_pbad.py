"""Train and save the p_bad artifact used by the champ-meta risk variants.

p_bad = calibrated P(the champion's own forward k-bar P&L < 0), from causal H4
features. The live bot multiplies champion exposure by (1 - beta * p_bad).

Protocol mirrors the research run (V5_FINDINGS §3p) exactly, so the live model is
the same object the walk-forward measured — only fitted on ALL history up to now
instead of per-fold:

  * label threshold taken from the TRAINING data only (no expanding-quantile peek)
  * rows whose label window (t..t+k) would run past the data end are dropped
  * isotonic calibration fitted on a PURGED tail slice of train (an 8-bar gap),
    not sklearn's unpurged internal CV, which is over-confident on overlapping labels

    python scripts/v5_train_pbad.py                 # train + save
    python scripts/v5_train_pbad.py --dry-run       # report only, write nothing
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_xau_turn_prob import (  # noqa: E402
    PURGE, badness, build_features, champion_signal, load_h4)

OUT = ROOT / "data" / "models" / "xau_pbad_h4"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=24, help="forward P&L window in H4 bars")
    ap.add_argument("--q", type=float, default=0.30, help="bad-window quantile")
    ap.add_argument("--cost", type=float, default=0.448, help="live spread floor USD/oz")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score

    df = load_h4(args.cost)
    champ = champion_signal(df["close"])
    F = build_features(df)
    tgt = badness(df, champ, "champmeta", args.k)

    X = F.values
    ok = np.isfinite(X).all(axis=1) & np.isfinite(tgt.values)
    # the last k bars have an incomplete forward window -> unusable as labels
    ok[-(args.k + PURGE):] = False
    Xo, to = X[ok], tgt.values[ok]
    thr = float(np.quantile(to, args.q))
    y = (to <= thr).astype(int)

    cut = int(len(Xo) * 0.80)
    Xf, yf = Xo[:cut], y[:cut]
    Xc, yc = Xo[cut + PURGE:], y[cut + PURGE:]      # purged calibration tail

    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                         l2_regularization=1.0, random_state=7)
    clf.fit(Xf, yf)
    raw = clf.predict_proba(Xc)[:, 1]
    iso = None
    if len(Xc) > 50 and 0 < yc.mean() < 1:
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw, yc)

    auc = float(roc_auc_score(yc, raw)) if 0 < yc.mean() < 1 else float("nan")
    meta = {
        "trained_on": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "features": list(F.columns),
        "k": args.k, "q": args.q, "threshold": thr,
        "cost_floor_usd": args.cost,
        "n_fit": int(len(Xf)), "n_calib": int(len(Xc)),
        "base_rate": round(float(y.mean()), 4),
        "holdout_auc": round(auc, 4),
        "data_last_bar": str(df.index[-1]),
        "purge_bars": PURGE,
        "warning": ("champ-meta FAILED its research gates (paired t -2.11 vs champion, "
                    "3/9 years positive). It trades drawdown for return; it is NOT alpha. "
                    "See V5_FINDINGS §3p."),
    }
    print(f"features {len(meta['features'])}  fit rows {meta['n_fit']}  "
          f"calib rows {meta['n_calib']}  base rate {meta['base_rate']}")
    print(f"threshold (train-only) {thr:+.6f}   purged-holdout AUC {auc:.4f}")
    print(f"last usable bar {df.index[ok][-1]}   data ends {df.index[-1]}")

    if args.dry_run:
        print("dry-run: nothing written")
        return
    import joblib
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "iso": iso}, p.with_suffix(".joblib"))
    p.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {p.with_suffix('.joblib')}\n      {p.with_suffix('.json')}")


if __name__ == "__main__":
    main()
