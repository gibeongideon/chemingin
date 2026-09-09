"""LIVE signal for the "zigzag" bottom detector — shared by the executor and the notifier.

WHAT THIS IS. The detector from V5_FINDINGS §3 / §3x: a walk-forward HistGradientBoosting
classifier on causal reversal/exhaustion features, predicting whether the current H1 bar sits
within +/-`TOL` bars of a confirmed ZigZag swing LOW. Its classification numbers are the best
this project has produced: **71% precision @ 22% recall, 75% @ 10%, 80% @ 5%, PR-AUC 0.60,
against a 33.5% random base rate (2.1-2.4x lift)**.

WHAT THIS IS NOT — READ THIS BEFORE DEPLOYING ANYTHING THAT IMPORTS IT.
Traded as a scalp (enter on the flag, exit at first TP/SL/max-hold) the same detector is
**DISPROVEN** (§3x): all 48 grid cells negative full-sample (best SR -0.44, DD -41.2%),
walk-forward **SR -0.76 / DD -35.7%** against buy-and-hold's +0.79, paired t **-3.72**,
**0 of 8 years better**, DSR **0.000**, PBO 0.024. It has NEGATIVE EXPECTANCY. It is deployed
here purely as an OBSERVATION HARNESS at the user's explicit request ("I just want to see it in
action, no profit needed"), on a demo account. Nothing about this file is a recommendation.

The gap between the two paragraphs above is this project's central finding, restated: detection
accuracy and tradeable edge are different quantities. A 0.15-0.6% scalp target gets eaten by a
$0.448 round-trip cost far more than a wider structure would.

DESIGN NOTES
  * Features, ZigZag labelling and the model are IMPORTED VERBATIM from
    `scripts/v5_xau_turning_ml.py` rather than reimplemented, so the live signal is
    byte-identical to the backtested one. Same discipline as `champion_signal`.
  * `hour` is dropped from the live feature set. §3p found that on gold it picks up a
    within-day artifact off ragged Sunday bars.
  * The model refits on a trailing window of strictly-past bars every run, with a PURGE gap of
    TOL+1 bars so the label window cannot leak into training.
  * Returns a probability AND the trade decision, but never places an order.
  * PARAMETERS come from the WALK-FORWARD SELECTION, not the full-sample best. Re-running the
    §3x selection (trailing 3y, re-picked each January) gives 2023/2024/2025 = (0.6, 0.6%,
    0.5%, 24) and 2026 = (0.6, 0.6%, 0.5%, 48). Threshold 0.6 / TP 0.6% / SL 0.5% is stable
    across four consecutive years; only max-hold flips. The full-sample best was (0.5, 0.6%,
    0.5%, 48) at SR -0.441 and is deliberately NOT used.

    python scripts/v5_zigzag_signal.py --tf H1        # print the current state
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

# verbatim from the tested detector — do not reimplement
from scripts.v5_xau_turning_ml import features, zigzag_swings, label_near, atr  # noqa: E402

TOL = 3                      # a bar counts as "at a bottom" within +/-3 bars of the pivot
ZZ_ORDER, ZZ_THETA = 5, 1.5  # fractal order and ATR multiple, as tested
TRAIN_BARS = 20_000          # trailing training window (~3.2y of H1)
MIN_TRAIN = 4_000


@dataclass
class ZigzagState:
    """Everything the executor or notifier needs, and nothing it does not."""
    asof: str                # timestamp of the last CLOSED bar used
    close: float
    prob: float              # P(this bar is within TOL of a swing low)
    threshold: float
    fires: bool              # prob >= threshold
    n_train: int
    stale_hours: float
    tp_pct: float
    sl_pct: float
    max_hold: int

    def as_dict(self) -> dict:
        return asdict(self)


def load_tf(tf: str = "H1") -> pd.DataFrame:
    f = ROOT / f"data/XAUUSD_{tf}_long.csv"
    d = pd.read_csv(f, parse_dates=["time"]).set_index("time")
    d = d[~d.index.duplicated(keep="last")].sort_index()
    return d.rename(columns=str.lower)


def bottom_probability(df: pd.DataFrame, threshold: float, tp_pct: float, sl_pct: float,
                       max_hold: int, train_bars: int = TRAIN_BARS) -> ZigzagState:
    """Fit on a trailing window of strictly-past bars, score the LAST CLOSED bar.

    Purge: the label at bar i depends on pivots up to i+TOL, so the final TOL+1 bars of the
    training window are dropped. Without that the most recent training labels would peek at
    bars the model is about to be asked to predict."""
    F = features(df)
    if "hour" in F.columns:
        F = F.drop(columns=["hour"])          # §3p: within-day artifact on gold
    # theta is an ATR-SCALED SERIES, not a scalar — zigzag_swings indexes it per bar.
    # Matches both tested callers: `theta = THETA_MULT * atr(df)`.
    _, buys = zigzag_swings(df, ZZ_ORDER, ZZ_THETA * atr(df))
    y = label_near(buys, len(df), TOL)

    ok = F.notna().all(axis=1).values
    n = len(df)
    score_i = n - 1                            # last CLOSED bar (caller must pass closed bars)
    purge = TOL + 1
    hi = score_i - purge                       # training may not include the purge gap
    lo = max(0, hi - train_bars)
    tr = np.zeros(n, bool)
    tr[lo:hi] = True
    tr &= ok
    if tr.sum() < MIN_TRAIN or len(np.unique(y[tr])) < 2 or not ok[score_i]:
        prob = float("nan")
        n_train = int(tr.sum())
    else:
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                             max_leaf_nodes=31, random_state=0)
        clf.fit(F.values[tr], y[tr])
        prob = float(clf.predict_proba(F.values[[score_i]])[0, 1])
        n_train = int(tr.sum())

    last = df.index[score_i]
    stale = (pd.Timestamp.now(tz="UTC").tz_localize(None) - last).total_seconds() / 3600
    return ZigzagState(asof=str(last), close=float(df["close"].iloc[score_i]),
                       prob=prob, threshold=threshold,
                       fires=bool(np.isfinite(prob) and prob >= threshold),
                       n_train=n_train, stale_hours=round(stale, 2),
                       tp_pct=tp_pct, sl_pct=sl_pct, max_hold=max_hold)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="H1")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--tp", type=float, default=0.006)
    ap.add_argument("--sl", type=float, default=0.005)
    ap.add_argument("--max-hold", type=int, default=48)
    args = ap.parse_args()
    df = load_tf(args.tf)
    s = bottom_probability(df, args.threshold, args.tp, args.sl, args.max_hold)
    print(f"ZIGZAG bottom detector — {args.tf}, {len(df):,} bars")
    print(f"  last closed bar   {s.asof}   close ${s.close:,.2f}   ({s.stale_hours:.1f}h old)")
    print(f"  trained on        {s.n_train:,} bars (trailing, purged by {TOL+1})")
    print(f"  P(at a bottom)    {s.prob:.4f}   threshold {s.threshold:.2f}")
    print(f"  FIRES             {'YES -> long' if s.fires else 'no'}")
    print(f"  if it fires       TP +{s.tp_pct:.2%}  SL -{s.sl_pct:.2%}  "
          f"max hold {s.max_hold} bars")
    if s.stale_hours > 6:
        print(f"  ** data {s.stale_hours:.0f}h stale — refresh before acting **")
    print("\n  NEGATIVE EXPECTANCY: walk-forward SR -0.76, 0/8 years, DSR 0.000 (§3x).")
    print("  Observation harness only. This module places no orders.")


if __name__ == "__main__":
    main()
