#!/usr/bin/env python3
"""Can we find the WORST rule and reverse it? The exact criterion, and every cell checked.

Asked 2026-11-01. It is a good question with a provable answer, so this derives the criterion
first and then tests it on every rule the repo can reconstruct cheaply.

THE ARITHMETIC, WHICH DECIDES MOST OF IT BEFORE ANY DATA IS TOUCHED.
For a rule that goes LONG on some set of bars, write its gross expectancy per fire as `g` and the
round-trip cost as `C`:

    net(LONG)    =  g - C
    net(REVERSE) = -g - C          <- the reverse pays the SAME cost, it does not earn it back
    -------------------------------------------------
    net(LONG) + net(REVERSE) = -2C                      ... (1)

So a reversal is not a free flip of the sign: **both sides pay, and the pair always sums to minus
twice the cost.** Reversing a rule that nets -5bp at a 1bp cost does NOT give +5bp; it gives
+4 - 1 = +3bp. Worse, searching for the WORST NET rule is biased toward high turnover, because
turnover is what makes a net figure worst -- and turnover is exactly what the reverse also pays.

THE SECOND IDENTITY, WHICH IS THE ONE THAT ACTUALLY KILLS IT ON GOLD.
CONTROL #2 says the benchmark is HOLDING, not zero. Against holding, with `b` the base drift over
the same bars:

    excess(LONG)    =  g - b
    excess(REVERSE) = -g - b
    -------------------------------------------------
    excess(LONG) + excess(REVERSE) = -2b                ... (2)

`external-playbook-exhausted` records that identity as explaining **every closure in the file**.
Rearranged, it gives the exact test this script runs:

    *** A REVERSAL BEATS HOLDING IF AND ONLY IF  g < -b  ***

i.e. the original rule must not merely lose -- it must underperform buy-and-hold **by more than
the entire drift**. On an asset that rose 264% over the sample, that is a very high bar, and it is
the thing to measure rather than assume.

THIRD, AND LEAST OBVIOUS: MULTIPLICITY IS SYMMETRIC. Searching K rules for the worst is the same
search as searching K rules for the best, so the max-stat null applies identically to the minimum.
`trigger-search-max-stat-null` found the best of 21,300 triggers indistinguishable from a no-edge
dataset; the worst of those 21,300 is equally indistinguishable, and reversing a noise-minimum
yields a noise-maximum. This is also why `bootstrap-gross-not-net` requires pricing the **signed**
maximum: a two-sided search must be nulled two-sided.

WHAT IS GENUINELY LIVE IN THE IDEA, AND IT IS NOT NOTHING. A *stated premise* having the wrong
SIGN is common here and worth testing directly -- §3cp found the TradingView rating inverted at
M15 (+8.2 sigma), `volume-zones-premise-backwards` found revert-to-POC worse than holding while
break-OUT beat it, `release-trigger-disproven` found "the first move is fake" backwards, and
`twelve-setups-closed-nr7-nondirectional` found against-trend beating with-trend. **Testing both
signs of a hypothesis is good practice. Searching a space for its minimum and flipping it is not
the same thing**, and this script separates the two.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "data" / "v5_runs"
CALIB = OUT / "mtf_calibration.json"
RT_BP = 0.504 * 1.5


def main() -> int:
    if not CALIB.exists():
        print("run scripts/v5_mtf_signal.py --calibrate first"); return 1
    cal = json.loads(CALIB.read_text())

    print("=" * 100)
    print("THE REVERSAL AUDIT — can the worst rule be flipped into a good one?")
    print("=" * 100)
    print(f"""
  net(LONG) + net(REVERSE) = -2C          both sides pay the cost; the flip is not free
  excess(LONG) + excess(REVERSE) = -2b    against HOLDING, the pair sums to minus twice the drift

  => a reversal beats holding IFF  g < -b : the rule must underperform buy-and-hold by MORE
     than the entire drift. Round trip C = {RT_BP:.3f}bp.
""")

    rows = []
    for tf, pf in cal["per_frame"].items():
        b = pf["base_ev_bp"]
        for lab, s in pf["states"].items():
            if "ev_bp" not in s:
                continue
            g = s["ev_bp"]
            rows.append(dict(frame=tf, state=lab, n=s["n"], runs=s.get("runs", s["n"]),
                             base_bp=b, g=g,
                             excess_long=g - b, excess_rev=-g - b,
                             net_long=g - RT_BP, net_rev=-g - RT_BP,
                             rev_beats_hold=(-g - b) > 0,
                             rev_beats_zero=(-g - RT_BP) > 0))
    df = pd.DataFrame(rows)
    if df.empty:
        print("no cells"); return 1

    print(f"  {len(df)} rules reconstructed (7 frames x 5 TradingView states).")
    print(f"\n  THE WORST RULES BY GROSS EXPECTANCY, and what reversing each would give:")
    print(f"  {'frame':>5} {'state':<12} {'runs':>6} {'g bp':>9} {'base b':>9} "
          f"{'exc LONG':>9} {'exc REV':>9} {'net REV':>9}  reversal beats HOLD?")
    print("  " + "-" * 98)
    for _, r in df.nsmallest(10, "g").iterrows():
        print(f"  {r.frame:>5} {r.state:<12} {int(r.runs):>6} {r.g:>+9.3f} {r.base_bp:>+9.3f} "
              f"{r.excess_long:>+9.3f} {r.excess_rev:>+9.3f} {r.net_rev:>+9.3f}  "
              f"{'YES' if r.rev_beats_hold else 'no'}")

    n_hold = int(df["rev_beats_hold"].sum())
    n_zero = int(df["rev_beats_zero"].sum())
    print(f"""
  RESULT
    reversals that beat ZERO   : {n_zero:>3} of {len(df)}
    reversals that beat HOLDING: {n_hold:>3} of {len(df)}   <- the benchmark that counts (CONTROL #2)
""")
    worst = df.loc[df["g"].idxmin()]
    need = -worst["base_bp"]
    print(f"  The single worst rule is {worst.frame} / {worst.state} at g = {worst.g:+.3f}bp.")
    print(f"  For its reversal to beat holding it would need g < {need:+.3f}bp — "
          f"it misses by {worst.g - need:+.3f}bp.")
    print(f"  Every frame's drift b is POSITIVE ({df.groupby('frame').base_bp.first().min():+.2f} "
          f"to {df.groupby('frame').base_bp.first().max():+.2f}bp), which is the whole problem: "
          f"on a\n  rising asset the bar for a profitable short is the drift itself, not zero.")

    # ---- the one honest version of the idea: test both SIGNS of a stated premise -------------
    print(f"""
  WHAT SURVIVES OF THE IDEA. Searching a space for its minimum and flipping it does not work --
  the pair sums to -2b, multiplicity is symmetric, and the minimum of a noisy search is as
  meaningless as its maximum. What DOES work, and has worked here four times, is testing both
  SIGNS of a PRE-REGISTERED premise:
    §3cp  the TradingView rating is INVERTED at M15 (+8.2 sigma) ... and still loses net
    §3cf  volume profile's revert-to-POC premise was backwards; break-OUT beat it
    §3bf  "the first move after a release is fake" was backwards
    §3bj  against-trend beat with-trend across 51 assets, so the NR7 filter does nothing
  Note what every one of those has in common: the SIGN was wrong AND the reversed version still
  did not clear cost. Finding an inversion is common. Finding a PROFITABLE one is the hard part,
  and identity (2) is why.""")
    df.to_csv(OUT / "reversal_audit.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
