#!/usr/bin/env python3
"""LABS001 "Daily rewards" lab plan (user screenshot 2026-11-27): can our book pass it?

    100K  $99   1 phase, target 5%   daily loss 1%   max loss 3% (EOD)   min days 0   split 85%

"Max Loss (EOD)" is ambiguous, so both readings are simulated:
  STATIC   floor fixed at 97,000 (3% below the start), checked on end-of-day equity
  TRAILING floor = highest end-of-day equity so far minus 3% of the start (futures-prop style), so
           a profit raises the floor. Strictly harder than STATIC.
Daily loss 1%: a day fails if the close-to-close loss x 1.5 (intraday-tail safety, as in
v5_account_chooser) exceeds 1% of the start balance.
Engine: block bootstrap (block 21 days) of the DEPLOYED XAU+BTC book's daily returns
(v5_account_chooser_2026_10.xau_btc_book), 3 seeds x 1200 paths, horizon 36 months.

FUNDED: the same walls apply after passing, so the funded dial must also survive a $3,000 wall.
Reported: expected 12-month payout at 85% split, net of breach (a breach ends the account).

    python scripts/v5_lab001_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from v5_account_chooser_2026_10 import xau_btc_book  # noqa: E402

TARGET, DAYLOSS, MAXLOSS, SAFETY, SPLIT, FEE = 0.05, 0.01, 0.03, 1.5, 0.85, 99.0
NSIM, BLOCK, MAXD = 1200, 21, 756
DIALS = (0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07)


def paths(r, seed, n_days):
    rng = np.random.default_rng(seed)
    out = np.empty((NSIM, n_days))
    for i in range(NSIM):
        idx = []
        while len(idx) < n_days:
            s, L = rng.integers(0, len(r)), rng.geometric(1 / BLOCK)
            idx.extend([(s + j) % len(r) for j in range(L)])
        out[i] = r[np.array(idx[:n_days])]
    return out


def run(x, trailing, target=TARGET):
    """x: daily returns (already dialled). Returns outcome code and day: 'pass','day','dd','time'."""
    eq, peak = 1.0, 1.0
    for t, d in enumerate(x):
        if -d * eq * SAFETY > DAYLOSS:          # daily loss is $ of the START balance
            return "day", t
        eq *= 1 + d
        peak = max(peak, eq)
        floor = (peak if trailing else 1.0) - MAXLOSS
        if eq < floor:
            return "dd", t
        if target is not None and eq >= 1 + target:
            return "pass", t
    return ("time", len(x)) if target is not None else ("alive", eq)


def main() -> int:
    r = xau_btc_book().to_numpy()
    sd = r.std() * np.sqrt(252)
    eqc = np.cumprod(1 + r)
    print(f"XAU+BTC book: Sharpe {r.mean() / r.std() * np.sqrt(252):+.3f}  vol {100 * sd:.1f}%  "
          f"maxDD {100 * (eqc / np.maximum.accumulate(eqc) - 1).min():.1f}%  worst day {100 * r.min():.2f}%  "
          f"({len(r)} days)")
    for trailing in (False, True):
        print(f"\n=== max loss read as {'TRAILING (EOD high-water)' if trailing else 'STATIC'} ===")
        print(f"  {'dial':>5} {'pass%':>6} {'med mo':>7} {'fail_day':>9} {'fail_dd':>8} {'timeout':>8}"
              f"   funded 12mo: {'survive':>8} {'payout $':>9}")
        for dial in DIALS:
            k = dial / sd
            res, fund = [], []
            for seed in range(3):
                P = paths(r, seed, MAXD) * k
                res += [run(p, trailing) for p in P]
                F = paths(r, 100 + seed, 252) * k       # funded year at the same dial and walls
                for p in F:
                    eq, peak, alive = 1.0, 1.0, True
                    for d in p:
                        if -d * eq * SAFETY > DAYLOSS:
                            alive = False; break
                        eq *= 1 + d; peak = max(peak, eq)
                        if eq < (peak if trailing else 1.0) - MAXLOSS:
                            alive = False; break
                    fund.append((alive, max(eq - 1, 0.0) if alive else 0.0))
            codes = np.array([c for c, _ in res])
            days = np.array([t for c, t in res if c == "pass"])
            n = len(codes)
            surv = np.mean([a for a, _ in fund])
            pay = np.mean([g for _, g in fund]) * 100_000 * SPLIT
            print(f"  {100 * dial:>4.1f}% {100 * (codes == 'pass').mean():>6.1f} "
                  f"{(np.median(days) / 21 if len(days) else float('nan')):>7.1f} "
                  f"{100 * (codes == 'day').mean():>9.1f} {100 * (codes == 'dd').mean():>8.1f} "
                  f"{100 * (codes == 'time').mean():>8.1f}   {100 * surv:>16.1f}% {pay:>9,.0f}")
    print(f"\nfee ${FEE:.0f}; payout = mean 12-month profit x {SPLIT:.0%} on 100K, zero if breached "
          "(simplification: no intra-year withdrawals, which under a TRAILING floor would help)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
