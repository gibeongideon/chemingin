"""Apply the pre-registered gates mechanically and write `verdict.json`.

No eyeballing. Every gate in `data/v5_runs/xau_advisor/PREREGISTRATION.md` §7 is evaluated from
the artifacts written by `v5_advisor_measure.py`, and the tier follows from the booleans rather
than from a judgement call. A cell reaches Tier A only if all seven gates pass; Tier B-plus only
if the tails are individually informative in BOTH halves; otherwise Tier B.

Tier B is a shippable verdict, not a failure to report — the pre-registration says so in advance
(§11), and the live artifact carries `usable: false` so the service prints the base rate and says
there is no measured edge rather than dressing one up.

    python scripts/v5_advisor_verdict.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "v5_runs" / "xau_advisor"

GATES = ("g1_pers_ci", "g2_drift", "g3_years", "g4_bss", "g5_reliability")


def _cell_key(row, family: str) -> str:
    return (f"{int(row['h'])}_{row['arm']}" if family == "dir"
            else f"{row['k_atr']}_{int(row['hours'])}_{row['arm']}")


def eval_gates(row: pd.Series, rel: pd.DataFrame, family: str) -> dict:
    """Gates 1-5 from PREREGISTRATION §7. Gates 6-7 (nulls) are evaluated only if 1-5 pass,
    because running 200-draw nulls on a cell that has already failed is wasted compute."""
    g = {}
    ci = row.get("d_pers_ci")
    ci = json.loads(ci) if isinstance(ci, str) else ci
    dci = row.get("d_drift_ci")
    dci = json.loads(dci) if isinstance(dci, str) else dci

    g["g1_pers_ci"] = bool(ci is not None and ci[0] > 0)
    g["g2_drift"] = bool(row["d_drift_pp"] > 0 and dci is not None and dci[0] > -0.50)
    yp = str(row.get("years_pers", "0/0")).split("/")
    yd = str(row.get("years_drift", "0/0")).split("/")
    g["g3_years"] = bool(int(yp[0]) >= 7 and int(yd[0]) >= 6)
    g["g4_bss"] = bool(row["bss"] > 0 and row.get("bss_ci_excludes_0", False))

    # gate 5: monotone, >=4pp top-minus-bottom, every bucket n>=200, top above base in BOTH halves
    g5 = False
    tails = False
    if rel is not None and len(rel) >= 2:
        r = rel.sort_values("lo")
        obs = r["observed"].values
        gap = float(obs[-1] - obs[0]) * 100
        mono = bool(np.all(np.diff(obs) >= -0.01))
        nmin = bool((r["n"] >= 200).all())
        base = float(row["base_rate"])
        h1, h2 = r["obs_first_half"].values, r["obs_second_half"].values
        top_both = bool(np.isfinite(h1[-1]) and np.isfinite(h2[-1])
                        and h1[-1] > base and h2[-1] > base)
        g5 = bool(gap >= 4.0 and mono and nmin and top_both)
        # Tier B-plus: the tails, each judged SEPARATELY and each required to hold in BOTH
        # halves. They are reported separately because the data splits them: across the adverse
        # family the TOP bucket sits above the base rate in both halves while the BOTTOM bucket
        # does not (its 2018-21 frequency lands at ~0.500 against a ~0.487 base). An advisor may
        # therefore warn on elevated risk and must stay silent on "low risk".
        top_ok = bool(np.isfinite(h1[-1]) and np.isfinite(h2[-1])
                      and obs[-1] >= base + 0.04 and h1[-1] > base and h2[-1] > base
                      and r["n"].values[-1] >= 200)
        bot_ok = bool(np.isfinite(h1[0]) and np.isfinite(h2[0])
                      and obs[0] <= base - 0.04 and h1[0] < base and h2[0] < base
                      and r["n"].values[0] >= 200)
        g["_top_tail_ok"] = top_ok
        g["_bot_tail_ok"] = bot_ok
        tails = bool(top_ok or bot_ok)
        g["_rel_gap_pp"] = gap
        g["_rel_monotone"] = mono
        g["_rel_top_both_halves"] = top_both
    g["g5_reliability"] = g5
    g["_tails_only"] = tails
    return g


def tier_for(g: dict, admissible: bool) -> str:
    if not admissible:
        return "INADMISSIBLE"
    if all(g[k] for k in GATES):
        return "A_pending_nulls"          # gates 6-7 still to run
    if g.get("_top_tail_ok") and g.get("_bot_tail_ok"):
        return "B_plus_both_tails"
    if g.get("_top_tail_ok"):
        return "B_plus_top_tail"      # may warn on elevated risk, silent on "low risk"
    if g.get("_bot_tail_ok"):
        return "B_plus_bot_tail"
    return "B"


def main() -> None:
    verdict = {"preregistration": "data/v5_runs/xau_advisor/PREREGISTRATION.md",
               "tier_c": json.loads((OUT / "tier_c.json").read_text()).get("passed"),
               "families": {}}
    for family, grid_f, rel_f in (("dir", "horizon_sweep.csv", "reliability_dir.csv"),
                                  ("adv", "adverse_grid.csv", "reliability_adv.csv")):
        gf, rf = OUT / grid_f, OUT / rel_f
        if not gf.exists():
            continue
        grid = pd.read_csv(gf)
        rel = pd.read_csv(rf) if rf.exists() else pd.DataFrame()
        rows = []
        for _, row in grid.iterrows():
            if not bool(row.get("admissible", False)):
                rows.append(dict(cell=_cell_key(row, family), tier="INADMISSIBLE",
                                 n=int(row.get("n", 0))))
                continue
            key = _cell_key(row, family)
            rc = rel[rel["cell"] == key] if len(rel) else None
            g = eval_gates(row, rc, family)
            rows.append(dict(cell=key, tier=tier_for(g, True),
                             n=int(row["n"]), base_rate=float(row["base_rate"]),
                             acc=float(row["acc"]), acc_drift=float(row["acc_drift"]),
                             d_pers_pp=float(row["d_pers_pp"]),
                             d_drift_pp=float(row["d_drift_pp"]),
                             bss=float(row["bss"]), auc=float(row["auc"]),
                             years_pers=row.get("years_pers"),
                             years_drift=row.get("years_drift"),
                             **{k: v for k, v in g.items()}))
        verdict["families"][family] = rows

        print("=" * 100)
        print(f"FAMILY {family.upper()} — pre-registered gates, evaluated mechanically")
        print("=" * 100)
        print(f"{'cell':18s} {'n':>6s} {'base':>6s} {'acc':>6s} {'drift':>6s} {'dD':>6s} "
              f"{'BSS':>8s} {'AUC':>7s} {'g1':>3s} {'g2':>3s} {'g3':>3s} {'g4':>3s} {'g5':>3s} "
              f"{'top':>3s}{'bot':>3s} {'gap':>6s} {'TIER':>18s}")
        for r in rows:
            if r["tier"] == "INADMISSIBLE":
                print(f"{r['cell']:18s} {r['n']:6d} {'':>6s} {'':>6s} {'':>6s} {'':>6s} "
                      f"{'':>8s} {'':>7s} {'':>3s} {'':>3s} {'':>3s} {'':>3s} {'':>3s} "
                      f"{'':>3s}{'':>3s} {'':>6s} {'INADMISSIBLE':>18s}")
                continue
            tick = lambda b: " Y" if b else " ."
            print(f"{r['cell']:18s} {r['n']:6d} {r['base_rate']:6.3f} {r['acc']:6.3f} "
                  f"{r['acc_drift']:6.3f} {r['d_drift_pp']:+6.2f} {r['bss']:+8.4f} "
                  f"{r['auc']:7.4f} {tick(r['g1_pers_ci']):>3s} {tick(r['g2_drift']):>3s} "
                  f"{tick(r['g3_years']):>3s} {tick(r['g4_bss']):>3s} "
                  f"{tick(r['g5_reliability']):>3s} {tick(r.get('_top_tail_ok')):>3s}"
                  f"{tick(r.get('_bot_tail_ok')):>3s} "
                  f"{r.get('_rel_gap_pp', float('nan')):6.2f} {r['tier']:>18s}")
        print()

    # headline
    tiers = [r["tier"] for f in verdict["families"].values() for r in f]
    verdict["any_tier_A"] = any(t.startswith("A_") for t in tiers)
    verdict["any_tails_only"] = any(t.startswith("B_plus") for t in tiers)
    verdict["overall"] = ("A_pending_nulls" if verdict["any_tier_A"]
                          else "B_plus" if verdict["any_tails_only"] else "B")
    # gate 6 results, if the nulls have been run
    for fam in ("dir", "adv"):
        f = OUT / f"null_maxstat_{fam}.json"
        if f.exists():
            n = json.loads(f.read_text())
            verdict.setdefault("nulls", {})[fam] = dict(
                observed_best_gap_pp=n["observed_best_gap_pp"],
                null_gap_p95=n["null_gap"]["p95"], null_gap_p99=n["null_gap"]["p99"],
                observed_best_auc=n["observed_best_auc"],
                null_auc_p95=n["null_auc"]["p95"], null_auc_p99=n["null_auc"]["p99"],
                gap_cleared_p95=n["gap_cleared_p95"], auc_cleared_p95=n["auc_cleared_p95"],
                p_value_gap=n["p_value_gap"], p_value_auc=n["p_value_auc"])
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, default=float))
    print("=" * 100)
    print(f"OVERALL VERDICT: {verdict['overall']}")
    print("  g1 d_persistence CI>0 | g2 beats DRIFT | g3 years | g4 Brier skill>0 | "
          "g5 reliability")
    if verdict["overall"] == "B":
        print("\n  TIER B — no measured edge that clears the pre-registered bar. This was the")
        print("  declared expected outcome (PREREGISTRATION §11). The live artifact ships")
        print("  usable:false and the advisor prints the BASE RATE, saying so in plain words.")
        print("  Nulls (gates 6-7) are NOT run: they price a search, and nothing survived to")
        print("  price.")
    print(f"\nwrote {(OUT / 'verdict.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
