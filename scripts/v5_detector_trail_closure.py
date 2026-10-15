"""GATE D — attach a TRAILING stop to the bottom detector. Pre-registered as a CLOSURE.

THE CEILING IS KNOWN BEFORE THE RUN (Gate A2). From §3bl's own oracle table, `ev_minus_hold` is
-0.94 to -1.43bp against a 33.29bp oracle-to-honest gap: **the most any exit change can recover
on this family is 4.3%**. And §3bl's Z5 found all six gross excesses NEGATIVE before any cost
(signed max t -0.098, Reality-Check p 0.8200) -- there is no gross edge to tax, and no exit taxes
a zero less than another exit does.

So this is run to CLOSE EXPLORATION-QUEUE #28 (structural exit) and #33 (trailing stop) with
numbers rather than by inference -- the service §3bl performed for Tier 5 -- not as a candidate.

THE ONE THING IT CAN LEGITIMATELY SETTLE: §3bl found that with PERFECT hindsight of the pivot a
fixed bracket earns LESS than simply holding ("a bracket on a long-only signal in a drifting
asset truncates the winners that pay for it"). A trailing stop is the one exit that does not
truncate. So: does ORACLE-TRAIL beat ORACLE-HOLD where ORACLE-BRACKET did not?

Cost uses the CORRECTED round trip ($0.672 = 1.5 x spread, Gate A3), which matters here because
a ratchet touches far more often than a fixed stop.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from v5_xau_turning_ml import zigzag_swings, atr                            # noqa: E402
from v5_zigzag_atr_bracket import load, COST_USD, ZZ_ORDER, ZZ_THETA, MAX_HOLD_H1  # noqa: E402

OUT = ROOT / "data/v5_runs/trail_decomp"
PATH_PER_H1 = 4
TRAIL_ATRS = (1.0, 2.0, 3.0)
HARD_SL_ATR = 3.0


def trail_outcomes(dec_idx, dec_close, atr_entry, trail_k, hard_k,
                   PH, PL, PC, path_index, dec_index, max_path_bars, dec_bar_minutes=60):
    """Walk M15 with a RATCHETING stop. Long only.

    Discipline copied from src/v5/xau_trend.py:192-220, which is the engine that runs live:
      * the stop is checked against the level set by PRIOR bars, THEN the peak is updated from
        this bar's extreme. Ratcheting first would let the stop chase the same bar's high and
        manufacture an exit at the top.
      * the peak is seeded at the ENTRY price, never from a pre-entry bar -- that bug produced
        phantom instant trail activation (src/v5/xau_m15_exec.py:227-229 and its regression test).
      * ATR is frozen at entry (xau_trend.py:186), not recomputed live.
    Ambiguity (stop and a new peak inside one M15 bar) is COUNTED and resolved stop-first, the
    conservative read -- §3bj measured that convention at 0.65 Sharpe on daily bars.
    """
    di, pi = pd.DatetimeIndex(dec_index), pd.DatetimeIndex(path_index)
    start = np.searchsorted(pi.values,
                            (di + pd.Timedelta(minutes=dec_bar_minutes)).values, side="right")
    n, npath = len(dec_idx), len(pi)
    exit_px = np.full(n, np.nan); hold_px = np.full(n, np.nan)
    bars = np.zeros(n, np.int32); amb = np.zeros(n, bool); stopped = np.zeros(n, bool)
    for q in range(n):
        i = dec_idx[q]
        a = atr_entry[i]
        if not np.isfinite(a) or a <= 0:
            continue
        entry = dec_close[i]
        # `start` is computed for EVERY decision bar, so it must be indexed by the BAR index i,
        # never by the selection position q. Indexing it by q paired a 2018 entry price with a
        # 2015 path and produced ev_hold of -2620bp with a 100% stop rate on bar 1 -- the tell
        # was that the "hold" comparator, which cannot be influenced by the stop, was absurd.
        s, e = start[i], min(start[i] + max_path_bars, npath)
        if s >= e:
            continue
        peak = entry                       # seeded AT ENTRY, never from a pre-entry bar
        stop = entry - hard_k * a          # the hard protective stop until the trail overtakes it
        px, j_end, hit = PC[e - 1], e - 1, False
        for j in range(s, e):
            if PL[j] <= stop:              # 1. exit check FIRST, against the PRIOR stop
                if PH[j] >= peak:
                    amb[q] = True          # a new peak and the stop in one bar: order unknown
                px, j_end, hit = stop, j, True
                break
            if PH[j] > peak:               # 2. only then ratchet
                peak = PH[j]
                stop = max(stop, peak - trail_k * a)
        exit_px[q] = px; hold_px[q] = PC[j_end]; bars[q] = j_end - s + 1; stopped[q] = hit
    return dict(exit_px=exit_px, hold_px=hold_px, bars=bars, ambiguous=amb, stopped=stopped)


def net(entry, px):
    return 1e4 * (px - entry - COST_USD) / entry


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    h1, m15 = load("H1"), load("M15")
    A = atr(h1).values
    close = h1["close"].values
    _, buys = zigzag_swings(h1, ZZ_ORDER, ZZ_THETA * atr(h1))
    p = pd.read_csv(ROOT / "data/v5_runs/zigzag_atr/oos_prob_h1.csv",
                    parse_dates=["time"], index_col="time")["p"].reindex(h1.index).values
    scored = np.isfinite(p)
    fires = np.flatnonzero(scored & (p >= 0.60))
    orc = buys[(buys > 0) & (buys < len(h1) - MAX_HOLD_H1 * PATH_PER_H1 // 4 - 2)]

    g = json.loads((ROOT / "data/v5_runs/trail_decomp/gate_a.json").read_text())
    print("=" * 104)
    print("GATE D — trailing stop on the bottom detector. CLOSURE, not a candidate.")
    print(f"  PRE-REGISTERED CEILING (A2): any exit change can recover at most "
          f"{g['A2']['ceiling_pct']:.1f}% of the {g['A2']['gap_bp']:.1f}bp oracle-to-honest gap.")
    print(f"  cost: corrected round trip ${COST_USD:.3f} (was ${COST_USD/1.5:.3f}, understated 1.5x)")
    print("=" * 104)

    rows = []
    for k in TRAIL_ATRS:
        for name, sel in (("ORACLE (true pivot)", orc), ("DETECTOR p>=0.60", fires)):
            r = trail_outcomes(sel, close, A, k, HARD_SL_ATR,
                               m15["high"].values, m15["low"].values, m15["close"].values,
                               m15.index, h1.index, MAX_HOLD_H1 * PATH_PER_H1)
            ok = np.isfinite(r["exit_px"])
            ev = float(np.mean(net(close[sel][ok], r["exit_px"][ok])))
            hold = float(np.mean(net(close[sel][ok], r["hold_px"][ok])))
            rows.append(dict(arm=name, trail_atr=k, n=int(ok.sum()),
                             ev_trail_bp=round(ev, 2), ev_hold_bp=round(hold, 2),
                             ev_minus_hold=round(ev - hold, 2),
                             stop_rate=round(float(r["stopped"][ok].mean()), 3),
                             ambiguous=round(float(r["ambiguous"][ok].mean()), 4),
                             med_bars=int(np.median(r["bars"][ok]))))
    tb = pd.DataFrame(rows)
    print("\n  ev_minus_hold > 0 would mean the TRAIL beats simply holding to the same exit —")
    print("  the one thing §3bl's fixed bracket could NOT do (it was -0.94 to -1.43bp).")
    print(tb.to_string(index=False))

    orc_t = tb[tb.arm.str.startswith("ORACLE")]
    det_t = tb[tb.arm.str.startswith("DETECTOR")]
    print(f"\n  ORACLE   best ev_minus_hold {orc_t.ev_minus_hold.max():+.2f}bp")
    print(f"  DETECTOR best ev_minus_hold {det_t.ev_minus_hold.max():+.2f}bp   "
          f"best EV/fire {det_t.ev_trail_bp.max():+.2f}bp")
    print(f"\n  AMBIGUITY re-measured for the TRAIL (NOT inherited from the bracket's 0.1-0.4%): "
          f"{100*tb.ambiguous.max():.2f}% worst cell")
    tb.to_csv(OUT / "gate_d_trail_closure.csv", index=False)
    print(f"\n  wrote {OUT}/gate_d_trail_closure.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
