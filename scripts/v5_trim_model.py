#!/usr/bin/env python3
"""Live model for the gold-sleeve META TRIM A/B (§3fa #1 / Phase D; pre-registered in
data/v5_runs/trim_ab/PREREG.md). Reproduces ONE yearly fold of scripts/v5_xau_turn_prob.walk_forward_prob:
label = the champion's own forward 24-bar P&L ('champmeta'), bad = train-slice 30% quantile, features =
build_features, HistGB + isotonic calibration on a purged tail. Fit on every H4 bar whose label window closes
PURGE bars before 1 January of --year; used for that whole year (retrain each January).

    python scripts/v5_trim_model.py train --csv data/XAUUSDs_H4_switch.csv --year 2026
    python scripts/v5_trim_model.py score --csv data/XAUUSDs_H4_switch.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

MODEL = ROOT / "data" / "v5_runs" / "trim_ab" / "model.joblib"
K, Q = 24, 0.30


def load_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"], index_col="time").sort_index()
    return df[~df.index.duplicated(keep="last")][["open", "high", "low", "close"]].astype(float)


def train(csv, year: int, out=MODEL) -> dict:
    import joblib  # noqa: PLC0415
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415
    from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415
    import v5_xau_turn_prob as tp  # noqa: PLC0415
    from src.v5.xau_dual_signals import champion_signal  # noqa: PLC0415
    df = load_csv(csv)
    F = tp.build_features(df)
    tgt = tp.badness(df, champion_signal(df["close"]), "champmeta", K)
    X = F.values
    ok = np.isfinite(X).all(axis=1) & np.isfinite(tgt.values)
    ts = pd.Timestamp(f"{year}-01-01")
    pos = np.arange(len(df))
    tr = ok & (df.index < ts) & (pos < np.searchsorted(df.index, ts) - K - tp.PURGE)
    thr = float(np.quantile(tgt.values[tr], Q))
    y = (tgt.values <= thr).astype(int)
    Xtr, ytr = X[tr], y[tr]
    cut = int(len(Xtr) * 0.80)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                         l2_regularization=1.0, random_state=7).fit(Xtr[:cut], ytr[:cut])
    Xc, yc = Xtr[cut + tp.PURGE:], ytr[cut + tp.PURGE:]
    iso = IsotonicRegression(out_of_bounds="clip").fit(clf.predict_proba(Xc)[:, 1], yc)
    meta = dict(year=year, n_train=int(tr.sum()), thr=thr, base_rate=float(ytr.mean()),
                first=str(df.index[tr][0]), last=str(df.index[tr][-1]), features=list(F.columns), csv=str(csv))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(clf=clf, iso=iso, meta=meta), out)
    return meta


def p_bad(csv, model=MODEL) -> tuple[float, str]:
    """Calibrated P(bad forward window) for the LAST bar in the file (the newest closed H4 bar)."""
    import joblib  # noqa: PLC0415
    import v5_xau_turn_prob as tp  # noqa: PLC0415
    m = joblib.load(model)
    df = load_csv(csv)
    F = tp.build_features(df)
    if list(F.columns) != m["meta"]["features"]:
        raise RuntimeError("feature set changed since training")
    x = F.values[-1:]
    if not np.isfinite(x).all():
        raise RuntimeError("non-finite features on the last bar")
    p = float(m["iso"].predict(m["clf"].predict_proba(x)[:, 1])[0])
    return p, str(df.index[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["train", "score"])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--year", type=int, default=pd.Timestamp.utcnow().year)
    ap.add_argument("--model", default=str(MODEL))
    a = ap.parse_args()
    if a.what == "train":
        print(train(a.csv, a.year, a.model))
    else:
        p, t = p_bad(a.csv, a.model)
        print(f"p_bad {p:.3f} at bar {t}  -> trim multiplier at b=0.5: {1 - 0.5 * p:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
