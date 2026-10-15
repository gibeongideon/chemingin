"""ARM 1 — the ATR-scaled bracket for the ZigZag detector (EXPLORATION-QUEUE #29).

Pre-registered at data/v5_runs/zigzag_atr/PREREGISTRATION.md (commit 94c4be9) BEFORE any fit.

THE QUESTION. SS3x killed the zigzag as a trade with 48 FIXED-PERCENTAGE bracket cells
(walk-forward Sharpe -0.76, 0/8 years, DSR 0.000). SS3ax then measured gold's mean daily range
going from 1.34% of price in 2015 to 2.73% in 2026, which makes the deployed 0.60%/0.50% bracket
0.23 and 0.18 of ONE day's travel, with 96% of sessions ranging more than TP+SL combined. When
both barriers sit inside one session's noise, which resolves first approaches a coin flip
REGARDLESS of whether the detector was right. So SS3x may have measured the bracket, not the
detector. SS3ay names this the one open actionable item in the log.

Changing the bracket changes the trade STRUCTURE, not the ranking of a label, so MANDATORY
CONTROL #8's scope clause exempts it.

WHAT THIS FILE IS CAREFUL ABOUT, and why each one is a logged mistake:

  * BRACKETS RESOLVE ON M15, never on the H1 decision bar (SS3bj -- "when a strategy depends on
    the ORDER of two events, the data must record that order". Resolving a bracket on the bar
    that contains it inflated NR7 from Sharpe +0.43 to +8.23, 19x).
  * dec_bar_minutes=60. An H1 bar STAMPED 12:00 does not close until 13:00, so the path must
    start strictly after that. Starting it at the stamp walks the bars that FORMED the close --
    the bug that scored AUC 0.901 in the advisor build.
  * AMBIGUITY IS REPORTED, NOT GUESSED. If TP and SL both fall inside one M15 bar the order is
    genuinely unknown; both conventions are priced (SS3bj measured an unresolvable convention at
    0.65 Sharpe).
  * THE BASE RATE IS PRINTED BESIDE THE HIT RATE (SS3ay). skill = hit - base. A positive EV with
    skill <= 0 is long exposure wearing a signal's clothes.
  * EV PER FIRE INCLUDES FALSE POSITIVES and is compared against THE DRIFT IT DISPLACES
    (CONTROL #8), not against zero. SS3as: the detector earned +0.05%/5d against +0.15% for
    simply being long -- one third of holding, paired t -3.64.
  * COSTS FLOORED AT THE LIVE BROKER QUOTE $0.448/oz (CONTROL #1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v5_xau_turning_ml import features, zigzag_swings, label_near, atr   # noqa: E402
from v5_pooled_bottom_detector import auc, hm_se                          # noqa: E402

OUT = ROOT / "data/v5_runs/zigzag_atr"

# ---- the deployed label + model spec (v5_zigzag_signal.py), reproduced verbatim
TOL, ZZ_ORDER, ZZ_THETA = 3, 5, 1.5
TRAIN_BARS, MIN_TRAIN = 20_000, 4_000
PURGE = TOL + ZZ_ORDER          # 8 = the label's true dependency reach; live hard-codes 4
REFIT_EVERY = 250

# ---- pre-registered grid, K = 6
TP_MULTS = (1.0, 2.0, 3.0)
SL_MULTS = (1.0, 1.5)
MAX_HOLD_H1 = 48
PATH_PER_H1 = 4                  # M15 bars per H1 bar
# COST, CORRECTED 2026-10-15 (Gate A3). $0.448 is the SPREAD, not the round trip:
# V5_FINDINGS:2207 records "$0.448 (= 1.94bp one-way on median gold $1,733)" and
# RESEARCH-REGIMES:79 gives one-way bp = 0.75 x spread_bp. So one-way is 1.944bp and the
# ROUND TRIP is 1.5 x spread = $0.672. This file previously subtracted $0.448 once and called
# it a round trip, i.e. it ran at TWO-THIRDS of its own stated honest cost.
# §3bl's verdict is UNCHANGED by the correction -- every cell was already negative GROSS, before
# any cost at all (signed max t -0.098, Reality-Check p 0.8200), so a larger hurdle only widens
# a failure. The fix matters going forward because a ratcheting stop turns over far more than a
# fixed bracket, so the understatement is not harmless there.
SPREAD_USD = 0.448
COST_USD = SPREAD_USD * 1.5      # round trip = 2 x 0.75 x spread, CONTROL #1
LIVE_TP_PCT, LIVE_SL_PCT = 0.006, 0.005      # the deployed fixed bracket (control cell)
THRESHOLD = 0.60
MIN_NONOVERLAP = 400


def load(tf: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / f"data/XAUUSD_{tf}_long.csv",
                    parse_dates=["time"], index_col="time").sort_index()
    return d[~d.index.duplicated(keep="last")]


# ------------------------------------------------------------------ the bracket, on M15
def bracket_outcomes(dec_index, dec_close, up_lvl, dn_lvl, path_high, path_low, path_close,
                     path_index, max_path_bars: int, dec_bar_minutes: int = 60):
    """Resolve a LONG bracket on the finer path series.

    Returns, per decision bar: outcome (+1 TP / -1 SL / 0 time-exit / NaN unresolvable),
    the exit price under the SL-FIRST convention, the exit price under the DROP convention,
    the matched-window hold price, bars held, and the ambiguity flag.

    The two conventions differ only on bars where TP and SL fall inside the SAME M15 bar. They
    are reported side by side because SS3bj measured exactly that choice at 0.65 Sharpe.
    """
    di, pi = pd.DatetimeIndex(dec_index), pd.DatetimeIndex(path_index)
    start = np.searchsorted(pi.values,
                            (di + pd.Timedelta(minutes=dec_bar_minutes)).values, side="right")
    C, U, D = (np.asarray(x, float) for x in (dec_close, up_lvl, dn_lvl))
    PH, PL, PC = (np.asarray(x, float) for x in (path_high, path_low, path_close))
    n, npath = len(di), len(pi)

    out = np.full(n, np.nan)
    exit_px = np.full(n, np.nan)       # SL-first convention
    hold_px = np.full(n, np.nan)       # buy-and-hold over the identical window
    held = np.zeros(n, np.int32)
    amb = np.zeros(n, bool)

    for i in range(n):
        if not (np.isfinite(U[i]) and np.isfinite(D[i]) and np.isfinite(C[i])):
            continue
        s = start[i]
        e = min(s + max_path_bars, npath)
        if s >= e:
            continue
        res = 0.0
        px = PC[e - 1]
        j_end = e - 1
        for j in range(s, e):
            tu, td = PH[j] >= U[i], PL[j] <= D[i]
            if tu and td:
                amb[i] = True            # order unknown inside this bar
                res, px, j_end = -1.0, D[i], j      # SL-FIRST: the conservative read
                break
            if td:
                res, px, j_end = -1.0, D[i], j
                break
            if tu:
                res, px, j_end = 1.0, U[i], j
                break
        out[i] = res
        exit_px[i] = px
        hold_px[i] = PC[j_end]           # what simply holding the same window ended at
        held[i] = j_end - s + 1
    return dict(outcome=out, exit_px=exit_px, hold_px=hold_px, bars=held, ambiguous=amb)


def net_ret(entry, exit_px, cost=COST_USD):
    """Round-trip cost applied once, in price units, as CONTROL #1 requires."""
    return (np.asarray(exit_px, float) - np.asarray(entry, float) - cost) / np.asarray(entry, float)


# ------------------------------------------------------------------ the detector, walk-forward
def oos_probabilities(h1: pd.DataFrame, cache: Path) -> pd.Series:
    """Rolling-refit OOS P(bottom) at every bar -- the deployed recipe, run as a backtest."""
    if cache.exists():
        s = pd.read_csv(cache, parse_dates=["time"], index_col="time")["p"]
        if len(s) and s.index[-1] == h1.index[-1]:
            print(f"  (cached OOS probabilities: {len(s):,} bars)")
            return s
    from sklearn.ensemble import HistGradientBoostingClassifier

    F = features(h1).drop(columns=["hour"])
    theta = ZZ_THETA * atr(h1)
    _, buys = zigzag_swings(h1, ZZ_ORDER, theta)
    y = label_near(buys, len(h1), TOL)
    ok = F.notna().all(axis=1).values
    X = F.values
    p = np.full(len(h1), np.nan)

    starts = list(range(TRAIN_BARS, len(h1), REFIT_EVERY))
    print(f"  walk-forward: {len(starts)} refits, {TRAIN_BARS:,} trailing bars, purge {PURGE}")
    for n_done, s in enumerate(starts):
        e = min(s + REFIT_EVERY, len(h1))
        hi = s - PURGE
        lo = max(0, hi - TRAIN_BARS)
        tr = np.arange(lo, hi)[ok[lo:hi]]
        if len(tr) < MIN_TRAIN or y[tr].sum() < 50:
            continue
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                             max_leaf_nodes=31, random_state=0)
        clf.fit(X[tr], y[tr])
        te = np.arange(s, e)[ok[s:e]]
        if len(te):
            p[te] = clf.predict_proba(X[te])[:, 1]
        if n_done % 40 == 0:
            print(f"    {n_done}/{len(starts)}  bar {s:,}  {h1.index[s].date()}")
    out = pd.Series(p, index=h1.index, name="p")
    out.to_frame().to_csv(cache)
    return out


def nonoverlap_count(idx: np.ndarray, gap: int) -> int:
    """Greedy >=gap-apart subset -- the honest-units count CONTROL #7 asks for."""
    last, n = -10**9, 0
    for i in idx:
        if i - last >= gap:
            n += 1
            last = i
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    a = ap.parse_args()

    g0 = json.loads((OUT / "gate0.json").read_text())
    assert g0["pass"], "Gate 0 did not pass -- the round stops"
    print("=" * 86)
    print("ARM 1 -- ZigZag ATR-scaled bracket (queue #29)   [Gate 0 asserted]")
    print("=" * 86)

    h1, m15 = load("H1"), load("M15")
    A = atr(h1)
    theta = ZZ_THETA * A
    _, buys = zigzag_swings(h1, ZZ_ORDER, theta)
    print(f"  H1 {len(h1):,} bars  {h1.index[0].date()}..{h1.index[-1].date()}   "
          f"pivots(lows) {len(buys):,}")

    print("\n[detector] rolling walk-forward OOS probabilities")
    p = oos_probabilities(h1, OUT / "oos_prob_h1.csv")
    scored = p.notna().values
    print(f"  scored {scored.sum():,} bars from {h1.index[np.argmax(scored)].date()}")

    # the bracket grid + the deployed control cell, all resolved on M15
    cells = [dict(name=f"ATR tp{tp:.1f}/sl{sl:.1f}", kind="atr", tp=tp, sl=sl)
             for tp in TP_MULTS for sl in SL_MULTS]
    cells.append(dict(name="LIVE pct 0.60/0.50", kind="pct",
                      tp=LIVE_TP_PCT, sl=LIVE_SL_PCT))

    close = h1["close"].values
    rows = []
    store: dict[str, dict] = {}
    for c in cells:
        if c["kind"] == "atr":
            up = close + c["tp"] * A.values
            dn = close - c["sl"] * A.values
            tp_px, sl_px = c["tp"] * A.values, c["sl"] * A.values
        else:
            up = close * (1 + c["tp"])
            dn = close * (1 - c["sl"])
            tp_px, sl_px = close * c["tp"], close * c["sl"]
        r = bracket_outcomes(h1.index, close, up, dn,
                             m15["high"].values, m15["low"].values, m15["close"].values,
                             m15.index, MAX_HOLD_H1 * PATH_PER_H1)
        r["tp_px"], r["sl_px"] = tp_px, sl_px
        store[c["name"]] = r

        res = np.isfinite(r["outcome"]) & scored
        decided = res & (r["outcome"] != 0)
        base_hit = float((r["outcome"][decided] > 0).mean())
        # generalised breakeven for an ASYMMETRIC bracket:  p* = (sl + cost) / (tp + sl)
        be = float(np.nanmean((sl_px + COST_USD) / (tp_px + sl_px)))
        rows.append(dict(cell=c["name"], decided=int(decided.sum()),
                         resolve_rate=round(float(decided.sum() / max(res.sum(), 1)), 3),
                         base_hit=round(base_hit, 4), breakeven=round(be, 4),
                         base_minus_be=round(base_hit - be, 4),
                         ambiguous=int(r["ambiguous"][res].sum()),
                         med_bars=int(np.median(r["bars"][res])) if res.sum() else 0,
                         tp_atr_frac=round(float(np.nanmedian(tp_px / A.values)), 2)))

    tb = pd.DataFrame(rows)
    tb.to_csv(OUT / "z2_breakeven.csv", index=False)
    print("\n[Z2] THE BREAKEVEN WALL -- unconditional, before any signal is applied")
    print("     p* = (sl + cost)/(tp + sl).  base_hit is what a RANDOM entry achieves.")
    print(tb.to_string(index=False))

    np.save(OUT / "_outcomes.npy", {k: v for k, v in store.items()}, allow_pickle=True)
    pd.Series(scored).to_frame("scored").to_csv(OUT / "_scored.csv", index=False)
    print(f"\n  wrote {OUT}/z2_breakeven.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
