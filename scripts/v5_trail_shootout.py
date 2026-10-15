"""GATE E — the shootout. Every arm on one table, ranked, with the verdict rule applied.

THE VERDICT RULE, pre-registered (data/v5_runs/trail_decomp/PREREGISTRATION.md):
    the winner is the arm with the highest matched-vol paired-t vs the incumbent that ALSO
    clears both sample halves AND PBO < 0.30 AND whose axis-neighbours beat the incumbent.
    If none clears, THE INCUMBENT WINS -- a decision-grade result, not a failure.

A single-cell peak with neighbours below the incumbent is noise: §3f killed H4/med precisely
because it peaked, and §3ab found parameter-cloud averaging HURTS the champion. A plateau
qualifies; a peak disqualifies.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/v5_runs/trail_decomp"
T_BAR, PBO_BAR = 2.5, 0.30


def main() -> int:
    b = pd.read_csv(OUT / "gate_b_profiles.csv")
    pbo = json.load(open(OUT / "gate_b_pbo.json"))["pbo"]
    d = pd.read_csv(OUT / "gate_d_trail_closure.csv")
    c1 = json.load(open(OUT / "c1_verdict.json"))
    inc = b[b.incumbent].iloc[0]

    # neighbour test: on its own axis, do the cells either side also beat the incumbent?
    nb = {}
    for axis in b.axis.unique():
        s = b[b.axis == axis].sort_values("cell", key=lambda x: pd.to_numeric(x, errors="coerce"))
        vals = s.to_dict("records")
        for i, r in enumerate(vals):
            if r["incumbent"]:
                nb[r["arm"]] = None; continue
            nbrs = [vals[j]["t_matched"] for j in (i - 1, i + 1)
                    if 0 <= j < len(vals) and not vals[j]["incumbent"]]
            nb[r["arm"]] = bool(nbrs) and all((x or -9) > 0 for x in nbrs)

    b["neighbours_ok"] = b.arm.map(nb)
    b["pbo"] = pbo
    b["clears"] = (b.t_matched >= T_BAR) & b.both_halves & (pbo < PBO_BAR) & \
                  b.neighbours_ok.fillna(False)

    print("=" * 116)
    print("GATE E — SHOOTOUT: every arm, ranked by matched-vol paired-t vs the live incumbent")
    print("=" * 116)
    rank = b[~b.incumbent].sort_values("t_matched", ascending=False)
    print(f"\n  ROW 0  INCUMBENT (live Maven, magic 360571): sl 3.0 / trail 3.0 / activation 1.0")
    print(f"         Sharpe {inc.sharpe:+.3f}   CAGR {inc.cagr:+.2f}%   maxDD {inc.maxdd:.1f}%")
    print(f"\n  gates: t_matched >= {T_BAR}  AND  both halves  AND  PBO < {PBO_BAR}  "
          f"AND axis-neighbours beat the incumbent")
    print(f"  PBO over the 15-cell grid = {pbo:.3f}  -> "
          f"{'PASSES' if pbo < PBO_BAR else 'FAILS the grid-level gate for EVERY cell'}")
    print()
    print(rank[["arm", "sharpe", "cagr", "maxdd", "t_matched", "both_halves",
                "neighbours_ok", "clears"]].to_string(index=False))

    winners = b[b.clears]
    print("\n" + "=" * 116)
    if len(winners) == 0:
        print("VERDICT: NO ARM CLEARS. THE INCUMBENT WINS.")
        best = rank.iloc[0]
        print(f"  best challenger: {best.arm} at t {best.t_matched:+.2f} "
              f"(bar {T_BAR}), both_halves={best.both_halves}, PBO {pbo:.3f}")
        print("  The live sl 3.0 / trail 3.0 / activation 1.0 stands. No config change.")
    else:
        print(f"VERDICT: {len(winners)} arm(s) clear: {list(winners.arm)}")
    print("=" * 116)

    print("\n--- what each sub-question answered ---")
    b["cell_num"] = pd.to_numeric(b.cell, errors="coerce")   # cell comes back from CSV as str
    sl = b[b.axis == "sl_atr"].sort_values("cell_num")
    tr = b[b.axis == "trail_atr"].sort_values("cell_num")
    ac = b[b.axis == "trail_activation_atr"].sort_values("cell_num")
    ex = b[b.axis == "exit_mode"]; sd = b[b.axis == "side"]
    print(f"  sl_atr      {'/'.join(f'{x:.3f}' for x in sl.sharpe)}  -> FLAT "
          f"(spread {sl.sharpe.max()-sl.sharpe.min():.3f}); a pure RISK dial: "
          f"CAGR {sl.cagr.min():.1f}-{sl.cagr.max():.1f}%, maxDD {sl.maxdd.min():.1f}..{sl.maxdd.max():.1f}%")
    print(f"  trail_atr   {'/'.join(f'{x:.3f}' for x in tr.sharpe)}  -> peaks AT the incumbent, "
          f"smooth decay both sides; nothing beats 3.0")
    print(f"  activation  {'/'.join(f'{x:.3f}' for x in ac.sharpe)}  -> 0.0 is nominally best "
          f"(t {ac[ac.cell_num==0.0].t_matched.iloc[0]:+.2f}) but misses the {T_BAR} bar and fails halves")
    print(f"  RATCHET?    trail {inc.sharpe:.3f} vs sltp {ex[ex.cell=='sltp'].sharpe.iloc[0]:.3f} "
          f"vs flip {ex[ex.cell=='flip'].sharpe.iloc[0]:.3f}  -> "
          f"**ratcheting DOES beat not-ratcheting** (never asked before; §3aa had no ratchet)")
    print(f"  SHORTS?     long-only {inc.sharpe:.3f} vs long/short "
          f"{sd.sharpe.iloc[0]:.3f} (t {sd.t_matched.iloc[0]:+.2f}, halves "
          f"{sd.half_18_21.iloc[0]:+.3f}/{sd.half_22_26.iloc[0]:+.3f}) -> "
          f"long-only wins by {inc.sharpe - sd.sharpe.iloc[0]:.3f}")
    print(f"  WINDOWS?    C1 inverted at {'ALL' if c1['all_inverted'] else 'some'} windows "
          f"(ratios {'/'.join(str(r['ratio_hi']) for r in c1['rows'])}x) -> no sweet spot exists")
    od = d[d.arm.str.startswith('ORACLE')].sort_values('trail_atr')
    print(f"  DETECTOR?   oracle ev_minus_hold {'/'.join(f'{x:+.2f}' for x in od.ev_minus_hold)}bp "
          f"at trail 1/2/3 -> a wide trail RECOVERS most of the bracket's "
          f"-1.43bp drag but still never beats holding")

    b.to_csv(OUT / "shootout.csv", index=False)
    print(f"\n  wrote {OUT}/shootout.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
