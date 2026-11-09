#!/usr/bin/env python3
"""ZigZag PANEL walk-forward probabilities (§3dt stage 1).

The live ZigZag model (features/label `_frame`, HistGB 300/0.05/31, trailing 20,000 bars per series,
TOL+1 purge, every series truncated at the cut) trained POOLED on 26 CME futures (Databento GLBX
hourly, single provenance, 2010-06..2026-11, back-adjusted) and refit QUARTERLY from 2018-01.
Each series is screened with `intrabar_leak` like the live job. Output: per-root probability series
for 2018-01.. (strictly out of sample at every point).

    python scripts/v5_zigzag_panel_probs.py
Output: data/v5_runs/zigzag_panel/probs.csv (wide: one column per root)
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_zigzag_signal import LEAK_MAX, MIN_TRAIN, TOL, TRAIN_BARS, _frame, intrabar_leak  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "zigzag_panel"
START = pd.Timestamp("2018-01-01")
ROOTS = ["GC", "SI", "HG", "PL", "CL", "NG", "ES", "NQ", "YM", "RTY", "ZN", "ZB", "ZF",
         "6E", "6J", "6B", "6A", "6C", "6S", "BTC", "ETH", "ZC", "ZS", "ZW", "LE", "HE"]


def load_cme(root: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / f"data/databento/ohlcv_1h/{root}.csv.gz", parse_dates=["time"], index_col="time")
    k = d.close_adj / d.close
    g = pd.DataFrame(dict(open=d.open * k, high=d.high * k, low=d.low * k, close=d.close_adj), index=d.index)
    return g[~g.index.duplicated(keep="last")].sort_index()


CAL = "--calendar" in sys.argv     # user 2026-11-09: add hour-of-day and day-of-week features


def calendar(idx: pd.DatetimeIndex) -> np.ndarray:
    """New York local hour as sin/cos (no midnight wrap: 23h sits next to 0h; session-aligned across
    DST) + weekday one-hot. Known at the bar's open, so no look-ahead."""
    ny = idx.tz_localize("UTC").tz_convert("America/New_York")
    hr = ny.hour + ny.minute / 60.0
    dow = np.eye(7)[ny.dayofweek][:, :5]
    return np.column_stack([np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24), dow])


def frame(d: pd.DataFrame):
    F, y, ok = _frame(d)
    if CAL:
        F = pd.concat([F, pd.DataFrame(calendar(d.index), index=F.index,
                                       columns=["h_sin", "h_cos", "mon", "tue", "wed", "thu", "fri"])], axis=1)
    return F, y, ok


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415
    OUT.mkdir(parents=True, exist_ok=True)
    data, frames = {}, {}
    for r in ROOTS:
        d = load_cme(r)
        lk = intrabar_leak(d)
        if lk > LEAK_MAX:
            print(f"refuse {r}: leak {lk:.3f}")
            continue
        data[r] = d
        frames[r] = frame(d)
        print(f"{r}: {len(d):,} bars, leak {lk:.3f}", flush=True)
    out = {r: pd.Series(np.nan, index=d.index) for r, d in data.items()}
    end = max(d.index[-1] for d in data.values())
    cuts = pd.date_range(START, end + pd.offsets.QuarterBegin(1), freq="QS")
    kw = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0)
    for c0, c1 in zip(cuts[:-1], cuts[1:]):
        t0 = time.time()
        X, Y = [], []
        for r, d in data.items():
            n = int((d.index < c0).sum())
            if n < MIN_TRAIN:
                continue
            # labels at bar i need pivots to i+TOL: rebuild the frame on data TRUNCATED at the cut
            F, y, ok = frame(d.iloc[:n])
            hi = n - (TOL + 1)
            lo = max(0, hi - TRAIN_BARS)
            m = np.zeros(n, bool)
            m[lo:hi] = True
            m &= ok
            X.append(F.values[m])
            Y.append(y[m])
        clf = HistGradientBoostingClassifier(**kw).fit(np.vstack(X), np.concatenate(Y))
        for r, d in data.items():
            Fa, _, oka = frames[r]
            rows = (d.index >= c0) & (d.index < c1) & oka
            if rows.any():
                out[r][rows] = clf.predict_proba(Fa.values[rows])[:, 1]
        print(f"{c0:%Y-%m}: trained on {sum(len(y) for y in Y):,} rows in {time.time() - t0:.0f}s", flush=True)
    P = pd.DataFrame(out)
    P = P[P.index >= START]
    P.to_csv(OUT / ("probs_cal.csv.gz" if CAL else "probs.csv.gz"), float_format="%.4f")
    print("saved", P.shape)


if __name__ == "__main__":
    sys.exit(main())
