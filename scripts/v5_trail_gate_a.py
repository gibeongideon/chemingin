"""GATE A — arithmetic pre-mortem for the trailing-stop round. No fitting, no new models.

Three numbers, each able to kill an arm before it is written. They are produced FIRST and
committed, because the point of a pre-mortem is that it cannot be revised after seeing a result.

  A1 ANSWERABILITY (CONTROL #7). Count DECIDED, NON-OVERLAPPING events -- not bars, not windows.
     SS3am: at a 30-bar horizon XAU H4 2018+ has 13,660 bars, 455 non-overlapping windows and
     only 247 decided events; counting windows overstated the sample 1.9x. Then invert the
     economics to a required AUC lift and print Hanley-McNeil SE beside it.

  A2 EXIT CEILING. The most any exit change can recover on the bottom-detector family, read off
     SS3bl's own oracle table. Writing it down BEFORE the run is what stops a 1bp improvement
     being reported as a finding.

  A3 COST, CORRECTED. `v5_zigzag_atr_bracket.net_ret` subtracts $0.448 once and calls it a round
     trip. It is not: V5_FINDINGS:2207 states "$0.448 (= 1.94bp one-way on median gold $1,733)",
     and RESEARCH-REGIMES:79 gives one-way bp = 0.75 x spread_bp. So $0.448 is the SPREAD, the
     one-way cost is 1.94bp and the ROUND TRIP is 1.5 x $0.448 = $0.672. SS3bl therefore ran at
     two-thirds of its own stated honest cost. It changed nothing there -- everything failed --
     but a ratcheting stop has materially higher turnover than a fixed bracket, so the
     understatement must be removed before any trail cell is believed.

     A3 also re-derives the cost in BASIS POINTS per year, because gold tripled while the dollar
     spread did not. That RELAXES the breakeven wall over time. It is reported to remove an
     excuse, never as a licence: delivered skill on this family is 0.000, so cost was never the
     binding constraint.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from v5_pooled_bottom_detector import hm_se                      # noqa: E402

OUT = ROOT / "data/v5_runs/trail_decomp"
SPREAD_USD = 0.448                 # the SPREAD, per V5_FINDINGS:2207
ONE_WAY_MULT = 0.75                # RESEARCH-REGIMES:79
ROUND_TRIP_USD = SPREAD_USD * 2 * ONE_WAY_MULT     # = 1.5 x spread = $0.672


def nonoverlap_count(n_bars: int, horizon: int) -> int:
    return int(np.floor(n_bars / max(horizon, 1)))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rep: dict = {}
    h1 = pd.read_csv(ROOT / "data/XAUUSD_H1_long.csv", parse_dates=["time"],
                     index_col="time").sort_index()
    h1 = h1[~h1.index.duplicated(keep="last")]
    close = h1["close"]

    print("=" * 92)
    print("GATE A — arithmetic pre-mortem (trailing-stop round)")
    print("=" * 92)

    # ---------------------------------------------------------------- A3 first: cost
    print("\n[A3] COST — the convention, corrected")
    med = float(close.median())
    spread_bp = 1e4 * SPREAD_USD / med
    print(f"  spread            ${SPREAD_USD:.3f}/oz  = {spread_bp:.3f} bp at median gold ${med:,.0f}")
    print(f"  one-way  (0.75x)  {ONE_WAY_MULT*spread_bp:.3f} bp   "
          f"<- V5_FINDINGS:2207 states 1.94bp one-way at median $1,733")
    print(f"  ROUND TRIP        {2*ONE_WAY_MULT*spread_bp:.3f} bp  = ${ROUND_TRIP_USD:.3f}/oz")
    print(f"  *** v5_zigzag_atr_bracket.net_ret subtracts ${SPREAD_USD:.3f} and calls it a round")
    print(f"      trip -> it understates cost by {ROUND_TRIP_USD/SPREAD_USD:.1f}x. FIX BEFORE ANY TRAIL CELL.")
    yr = close.groupby(close.index.year).median()
    print("\n  the same $0.448 in basis points, by year (gold tripled; the spread did not):")
    rows = []
    for y, px in yr.items():
        sb = 1e4 * SPREAD_USD / float(px)
        rows.append(dict(year=int(y), median_px=round(float(px), 1),
                         spread_bp=round(sb, 3), round_trip_bp=round(2 * ONE_WAY_MULT * sb, 3)))
    tb = pd.DataFrame(rows)
    print(tb.to_string(index=False))
    print("  -> the wall RELAXED over time. This removes an excuse; it is not a result.")
    rep["A3"] = dict(spread_usd=SPREAD_USD, round_trip_usd=ROUND_TRIP_USD,
                     understatement=ROUND_TRIP_USD / SPREAD_USD,
                     by_year=rows)

    # ---------------------------------------------------------------- A1 answerability
    print("\n[A1] ANSWERABILITY (CONTROL #7) — decided, NON-OVERLAPPING events")
    scored = len(h1[h1.index >= "2018-05-15"])          # the detector's live window
    a1 = []
    for label, H in (("same-day (24 H1 bars)", 24), ("2 days (48)", 48), ("5 days (120)", 120),
                     ("10 days (240)", 240), ("20 days (480)", 480), ("§3ay horizon (16)", 16)):
        n = nonoverlap_count(scored, H)
        se = hm_se(0.55, n // 2, n - n // 2) if n > 20 else float("nan")
        a1.append(dict(window=label, horizon_bars=H, nonoverlap=n, se_auc=round(se, 4),
                       floor_K5=round(se * 1.163, 4), floor_K12=round(se * 1.629, 4)))
        print(f"  {label:<22} H={H:>3}  non-overlapping {n:>6,}  SE(AUC) {se:.4f}  "
              f"best-of-5 floor +{se*1.163:.4f}  best-of-12 +{se*1.629:.4f}")
    rep["A1"] = a1
    print(f"  NOTE: §3ay filed SE 0.0035 was computed on ALL labelled bars at 16-fold overlap —")
    print(f"        a CONTROL #7 violation. The honest SE at H=16 is "
          f"{a1[-1]['se_auc']:.4f}, {a1[-1]['se_auc']/0.0035:.1f}x the filed value.")

    # ---------------------------------------------------------------- A2 exit ceiling
    print("\n[A2] EXIT CEILING on the bottom-detector family (from §3bl's own oracle table)")
    f = ROOT / "data/v5_runs/zigzag_atr/z1_z3_cells.csv"
    if f.exists():
        z = pd.read_csv(f)
        orc = z[z.arm == "ORACLE"]
        det = z[z.arm == "DETECTOR"]
        worst_drag = float(orc.ev_minus_hold.abs().max())
        gap = float(orc.ev_minus_anybar.max()) - float(det.ev_minus_anybar.max())
        print(f"  oracle EV/fire            {orc.ev_fire.min():+.2f} .. {orc.ev_fire.max():+.2f} bp")
        print(f"  oracle EV - hold          {orc.ev_minus_hold.min():+.2f} .. "
              f"{orc.ev_minus_hold.max():+.2f} bp   <- what ANY exit change can recover")
        print(f"  oracle-to-honest gap      {gap:.2f} bp")
        print(f"  *** CEILING = {worst_drag:.2f} / {gap:.2f} = "
              f"{100*worst_drag/gap:.1f}% of the gap. PRE-REGISTERED. ***")
        rep["A2"] = dict(max_recoverable_bp=worst_drag, gap_bp=gap,
                         ceiling_pct=100 * worst_drag / gap)
    print("  So: a trail may be worth testing on the CHAMPION, but it cannot revive the detector.")

    (OUT / "gate_a.json").write_text(json.dumps(rep, indent=2))
    print(f"\n  wrote {OUT}/gate_a.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
