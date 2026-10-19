"""ONE-SHOT RE-VERIFICATION of every number this round reported, from scratch.

Re-derives each headline independently and checks it against the figure filed in V5_FINDINGS,
so a transcription error or a stale cache cannot survive. Any mismatch beyond tolerance is
printed as FAIL -- the point is to be able to fail, not to reprint what was already said.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = "/home/rock/anaconda3/envs/envmt5/bin/python"

# (script, needle, filed value, tolerance) -- the filed values come from V5_FINDINGS 3bp/3bq/3br
CHECKS = [
    ("v5_btc_strategy_scan.py", "champion SLOW (1/12)", 1.719, 0.02, "3bp BTC 1/12 Sharpe"),
    ("v5_btc_strategy_scan.py", "BUY & HOLD (vol-targeted)", 1.055, 0.02, "3bp BTC B&H Sharpe"),
    ("v5_btc_vs_xau.py", "A  XAU champion", 0.919, 0.02, "3bq XAU champion Sharpe"),
    ("v5_btc_vs_xau.py", "B  BTC trend", 1.363, 0.02, "3bq BTC Sharpe"),
    ("v5_btc_vs_xau.py", "A+B  50/50", 1.511, 0.02, "3bq A+B Sharpe"),
    ("v5_book_width_compare.py", "XAU+BTC 50/50", 1.511, 0.02, "3br XAU+BTC Sharpe"),
    ("v5_book_width_compare.py", "XAU+BTC+NDX", 1.696, 0.02, "3br XAU+BTC+NDX Sharpe"),
    ("v5_book_width_compare.py", "core-4 gold-tilted", 1.395, 0.02, "3br core-4 tilted Sharpe"),
]


def run(script: str) -> str:
    r = subprocess.run([PY, str(ROOT / "scripts" / script)], capture_output=True, text=True,
                       timeout=1800, cwd=str(ROOT))
    if r.returncode != 0:
        print(f"  *** {script} EXITED {r.returncode} ***\n{r.stderr[-800:]}")
    return r.stdout


def first_float_after(line: str) -> float | None:
    for tok in line.split():
        try:
            return float(tok.replace("%", "").replace("+", ""))
        except ValueError:
            continue
    return None


def main() -> int:
    outs, fails = {}, 0
    for script in sorted({c[0] for c in CHECKS}):
        print(f"re-running {script} ...", flush=True)
        outs[script] = run(script)
    print("\n" + "=" * 92)
    print("RE-VERIFICATION — every filed headline re-derived from scratch")
    print("=" * 92)
    print(f"  {'check':34s} {'filed':>9s} {'re-run':>9s} {'delta':>9s}  verdict")
    for script, needle, filed, tol, label in CHECKS:
        line = next((l for l in outs[script].splitlines() if needle in l), None)
        if line is None:
            print(f"  {label:34s} {filed:9.3f} {'—':>9s} {'—':>9s}  FAIL: line not found")
            fails += 1
            continue
        got = first_float_after(line.replace(needle, ""))
        if got is None:
            print(f"  {label:34s} {filed:9.3f} {'—':>9s} {'—':>9s}  FAIL: no number parsed")
            fails += 1
            continue
        ok = abs(got - filed) <= tol
        fails += 0 if ok else 1
        print(f"  {label:34s} {filed:9.3f} {got:9.3f} {got-filed:+9.3f}  "
              f"{'ok' if ok else 'FAIL'}")
    print("=" * 92)
    print(f"  {'ALL FILED NUMBERS REPRODUCE' if fails == 0 else str(fails) + ' MISMATCH(ES)'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
