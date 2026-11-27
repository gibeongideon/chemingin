#!/usr/bin/env python3
"""LABS001 with a STATIC 3% floor (user 2026-11-27: "the dd is static, think of leaving a balance to give a
buffer"). Two levers v5_lab001_check.py did not model:

1. DAILY GUARD. The book's executor already runs an intraday daily-loss guard (challenge_guards,
   daily_guard_frac) that FLATTENS the book when the day's loss reaches g. A bad day then costs ~g + slip
   instead of failing the account. Modelled on daily closes: if the day's close loss x 1.5 (intraday-low
   proxy, as before) reaches g, the day is realised at -(g + SLIP) and the account survives; it is a FAIL
   only if that realised loss exceeds the 1% firm limit (it cannot with g + SLIP < 1%), so gap risk is
   stressed separately with a larger slip.
2. BUFFER. The floor is fixed at 97,000. Profit LEFT in the account is cushion. Funded stage: each day,
   anything above 1 + B is withdrawn (paid at 85%), so B is the buffer you keep. The challenge has no
   withdrawals, so profit there is automatically buffer. Optional CUSHION SIZING: dial x (cushion / 3%),
   capped at 2x, i.e. risk grows only as the buffer grows (CPPI on a static floor).

Daily loss limit = 1% of the START balance ($1,000), unaffected by the buffer.

    python scripts/v5_lab001_buffer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from v5_account_chooser_2026_10 import xau_btc_book  # noqa: E402
from v5_lab001_check import BLOCK, MAXD, paths  # noqa: E402

TARGET, FLOOR, SAFETY, SPLIT = 0.05, 0.97, 1.5, 0.85
GUARD, SLIP = 0.008, 0.001
N = 1200


def step(d, k, eq, guard, slip):
    """One day at size k (fraction of the book). Returns realised return on START balance terms."""
    x = k * d
    if guard is not None and -x * SAFETY * eq >= guard:
        return -(guard + slip) / eq                 # flattened intraday at the guard
    return x


def challenge(r, sd, dial, guard, slip, cushion_size):
    res = []
    for seed in range(3):
        for p in paths(r, seed, MAXD):
            eq, out = 1.0, ("time", MAXD)
            for t, d in enumerate(p):
                k = dial / sd * (min(2.0, (eq - FLOOR) / (1 - FLOOR)) if cushion_size else 1.0)
                x = step(d, k, eq, guard, slip)
                if -x * eq > 0.01 or (guard is None and -k * d * SAFETY * eq > 0.01):
                    out = ("day", t); break
                eq *= 1 + x
                if eq < FLOOR:
                    out = ("dd", t); break
                if eq >= 1 + TARGET:
                    out = ("pass", t); break
            res.append(out)
    c = np.array([o for o, _ in res])
    days = np.array([t for o, t in res if o == "pass"])
    return 100 * (c == "pass").mean(), (np.median(days) / 21 if len(days) else np.nan), \
        100 * (c == "day").mean(), 100 * (c == "dd").mean(), 100 * (c == "time").mean()


def funded(r, sd, dial, buf, guard, slip, cushion_size, start_eq=1.0):
    pays, surv = [], []
    for seed in range(3):
        for p in paths(r, 100 + seed, 252):
            eq, paid, alive = start_eq, 0.0, True
            for d in p:
                k = dial / sd * (min(2.0, (eq - FLOOR) / (1 - FLOOR)) if cushion_size else 1.0)
                x = step(d, k, eq, guard, slip)
                if -x * eq > 0.01 or (guard is None and -k * d * SAFETY * eq > 0.01):
                    alive = False; break
                eq *= 1 + x
                if eq < FLOOR:
                    alive = False; break
                if eq > 1 + buf:
                    paid += eq - (1 + buf); eq = 1 + buf
            pays.append(paid * 100_000 * SPLIT)
            surv.append(alive)
    return 100 * np.mean(surv), np.mean(pays), np.median(pays)


def main() -> int:
    r = xau_btc_book().to_numpy()
    sd = r.std() * np.sqrt(252)
    print(f"XAU+BTC book vol {100 * sd:.1f}%; static floor 97,000; daily limit $1,000; guard {100 * GUARD:.1f}% "
          f"+ slip {100 * SLIP:.1f}%")

    print("\nCHALLENGE (target 5%)   [profit you make is the buffer; no withdrawals]")
    print(f"  {'setup':<34}{'dial':>6}{'pass%':>7}{'med mo':>8}{'day%':>7}{'dd%':>7}{'time%':>7}")
    for name, g, s, cs in (("no guard (previous run)", None, 0, False),
                           ("guard 0.8%", GUARD, SLIP, False),
                           ("guard 0.8% + cushion sizing", GUARD, SLIP, True),
                           ("guard 0.8%, GAP stress slip 0.5%", GUARD, 0.005, False)):
        for dial in (0.015, 0.02, 0.03, 0.04, 0.05):
            pr, mo, dy, dd, tm = challenge(r, sd, dial, g, s, cs)
            print(f"  {name if dial == 0.015 else '':<34}{100 * dial:>5.1f}%{pr:>7.1f}{mo:>8.1f}{dy:>7.1f}{dd:>7.1f}{tm:>7.1f}")

    print("\nFUNDED 12 months (85% split; withdraw everything above 1 + buffer each day)")
    print(f"  {'setup':<30}{'dial':>6}" + "".join(f"{'buf ' + str(int(100 * b)) + '%':>16}" for b in (0, .01, .02, .03, .05)))
    print(f"  {'':<36}" + "".join(f"{'surv%  mean $':>16}" for _ in range(5)))
    for name, cs in (("guard 0.8%, fixed dial", False), ("guard 0.8% + cushion sizing", True)):
        for dial in (0.02, 0.03, 0.04, 0.05):
            cells = []
            for buf in (0.0, 0.01, 0.02, 0.03, 0.05):
                sv, mean, _ = funded(r, sd, dial, buf, GUARD, SLIP, cs, start_eq=1.0)
                cells.append(f"{sv:>6.1f} {mean:>8,.0f}")
            print(f"  {name if dial == 0.02 else '':<30}{100 * dial:>5.1f}%" + "".join(f"{c:>16}" for c in cells))
    print("\n(a buffer above 0% must first be EARNED: those cells start at 1.0 and pay nothing until equity passes 1+B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
