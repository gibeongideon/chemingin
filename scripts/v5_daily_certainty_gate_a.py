#!/usr/bin/env python3
"""GATE A for the 'daily pattern that almost never fails' question -- ARITHMETIC ONLY, NO FITTING.

The user asked for a daily signal that, when it appears, makes a given outcome "almost certain".
Before searching for one, this script answers whether such a claim is ESTABLISHABLE at all on the
data we have, because three separate arithmetic facts bound it and all three are cheap to compute:

  A1  THE BASE RATE.  Gold rises on most days. A signal that is "80% reliable" is worthless if the
      unconditional rate is 78%, and the repo has already been burnt by thresholds anchored at 0.5
      instead of the base rate (xau-drift-baseline-inverts-3r). Every precision below is reported
      BESIDE its own base rate, never alone, and the benchmark is HOLDING the asset -- not zero
      (cot-positioning-disproven: benchmark XAU vs holding gold, t +6.28 at 4w).

  A2  COST AT D1.  Unlike the intraday work in this repo, cost is NOT the binding constraint at a
      daily horizon: at Switch Markets gold's round trip is 0.75bp against a ~100bp daily sigma.
      Printed here to REMOVE it as an excuse, not to license anything.

  A3  THE BEST-OF-K NOISE FLOOR -- the one that actually decides this question.  "Almost never
      fails" means high precision, and high precision is exactly what a large search manufactures
      on small event counts. If we test K candidate patterns and each fires N times, the MAXIMUM
      precision observed under a pure no-edge null is already far above the base rate. This script
      simulates that floor so the later search has a bar set before it starts, per
      trigger-search-max-stat-null: searching harder provably raises the bar faster than the
      statistic.

Outputs a table of (K, N) -> noise-floor precision, and inverts it: given a claimed precision,
how many non-overlapping events are needed to clear the floor at all.

No model is fitted here and no pattern is proposed. This file exists to make the later search
falsifiable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(20261021)

# Measured live at Switch Markets 2026-10-19 (V5_FINDINGS 3bs). spread_bp -> round trip is
# 1.5 x spread_bp, the repo's convention (one-way = 0.75 x spread): $0.448 is the SPREAD, which
# this project got wrong once already.
COSTS_BP = {
    "GOLD": 0.50 * 1.5,          # XAUUSDs $0.210 on $4,164
    "BTC": 4.46 * 1.5,           # BTCUSDs $36.82 on $82,504
}
COSTS_BP["BTC/XAU ratio (2 legs)"] = COSTS_BP["GOLD"] + COSTS_BP["BTC"]


def load(alias: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / f"{alias}_D1_long.csv", parse_dates=["time"])
    return d.set_index("time").sort_index()


def describe(name: str, close: pd.Series) -> dict:
    r = close.pct_change().dropna()
    return dict(
        asset=name, n_days=len(r),
        start=str(r.index.min().date()), end=str(r.index.max().date()),
        base_up=float((r > 0).mean()),
        daily_sigma_bp=float(r.std() * 1e4),
        mean_abs_bp=float(r.abs().mean() * 1e4),
        drift_bp_day=float(r.mean() * 1e4),
    )


def noise_floor(base: float, n_events: int, k: int, trials: int = 4000,
                q: float = 0.95) -> float:
    """The q-quantile of MAX precision over k no-edge patterns, each firing n_events times.

    Under the null every pattern is just n_events draws of Bernoulli(base), and the best of k such
    patterns is what a search REPORTS. The bar is the q-QUANTILE of that maximum, not its mean.

    *** WHY NOT THE MEAN -- A CORRECTION TO THIS SCRIPT'S FIRST VERSION. *** The first run returned
    `.mean()`, so "beating the floor" meant beating the AVERAGE best-of-K noise result, which under
    the null happens half the time: a 50% false-positive rate dressed up as a significance test.
    It made a 0.90 claim look establishable on 15 events at K=500, where the true requirement is
    several times that. q=0.95 makes clearing the bar a 1-in-20 event under the null, which is what
    the bar was supposed to mean.
    """
    draws = RNG.binomial(n_events, base, size=(trials, k)) / n_events
    return float(np.quantile(draws.max(axis=1), q))


def events_needed(base: float, claim: float, k: int, cap: int = 4000) -> int | None:
    """Smallest N at which `claim` precision clears the best-of-k p95 noise floor.

    Monotone in N only in expectation, so this walks N upward and takes the first N for which the
    claim clears AND stays clear at the next step -- a lone crossing is sampling noise in the
    simulation, not an answer.
    """
    grid = list(range(10, 200, 5)) + list(range(200, cap, 50))
    for i, n in enumerate(grid[:-1]):
        if claim > noise_floor(base, n, k, trials=3000) and \
                claim > noise_floor(base, grid[i + 1], k, trials=3000):
            return n
    return None


def main() -> None:
    print("=" * 100)
    print("GATE A -- is 'a daily pattern that almost never fails' ESTABLISHABLE? (arithmetic only)")
    print("=" * 100)

    gold = load("GOLD")["close"]
    btc = load("BTC")["close"]
    # The BTC/XAU cross does not exist as a tradeable symbol (checked: 2,067 Switch symbols, only
    # XAUUSDs/XAUEURs/BTCUSDs/BTCEURs). It is SYNTHESISED here and would need two legs to trade.
    common = gold.index.intersection(btc.index)
    ratio = (btc.reindex(common) / gold.reindex(common)).dropna()

    print("\n--- A1  BASE RATES AND SCALE (the benchmark is HOLDING, never zero) ---")
    rows = [describe("GOLD", gold), describe("BTC", btc), describe("BTC/XAU ratio", ratio)]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    print("\n--- A2  COST AT A DAILY HORIZON (printed to remove it as an excuse) ---")
    print(f"  {'instrument':<24} {'round trip':>11} {'daily sigma':>12} {'cost/sigma':>11}")
    sig = {"GOLD": rows[0]["daily_sigma_bp"], "BTC": rows[1]["daily_sigma_bp"],
           "BTC/XAU ratio (2 legs)": rows[2]["daily_sigma_bp"]}
    for k, c in COSTS_BP.items():
        print(f"  {k:<24} {c:>9.2f}bp {sig[k]:>10.1f}bp {c / sig[k]:>10.2%}")
    print("  => At D1 the spread is a rounding error against the daily move. Cost is NOT the")
    print("     binding constraint here, unlike every intraday study in this repo. What binds is")
    print("     statistical, and A3 is the reason.")

    print("\n--- A3  THE BEST-OF-K NOISE FLOOR: precision a NO-EDGE search already reports ---")
    print("    (p95 of the best-of-K maximum: clearing it is a 1-in-20 event under the null)")
    for name, base in (("GOLD", rows[0]["base_up"]), ("BTC", rows[1]["base_up"])):
        print(f"\n  {name}  (base rate {base:.3f} -- this is what 'reliable' must beat)")
        print(f"    {'events':>7} | " + " | ".join(f"K={k:<6}" for k in (1, 50, 500, 5000)))
        print("    " + "-" * 56)
        for n in (15, 25, 50, 100, 250, 500):
            cells = " | ".join(f"{noise_floor(base, n, k):<8.3f}" for k in (1, 50, 500, 5000))
            print(f"    {n:>7} | {cells}")

    print("\n  HOW TO READ THIS. Find the row for the event count a pattern actually fires, and")
    print("  the column for how many patterns were searched. That cell is the precision you would")
    print("  see from PURE NOISE. A '90% reliable' pattern that fires 25 times, found by screening")
    print("  500 candidates, is INDISTINGUISHABLE FROM NOTHING -- the noise floor is already there.")

    print("\n--- A3b  INVERTED: events needed for a precision claim to clear the floor ---")
    base_g = rows[0]["base_up"]
    print(f"  GOLD, base rate {base_g:.3f}. Minimum NON-OVERLAPPING events required:")
    print(f"    {'claim':>7} | " + " | ".join(f"K={k:<7}" for k in (50, 500, 5000)))
    print("    " + "-" * 44)
    for claim in (0.70, 0.80, 0.90, 0.95):
        cells = []
        for k in (50, 500, 5000):
            n = events_needed(base_g, claim, k)
            cells.append(f"{n:<9}" if n else "never    ")
        print(f"    {claim:>7.2f} | " + " | ".join(cells))

    yrs = rows[0]["n_days"] / 252
    print(f"\n  Gold has {rows[0]['n_days']:,} daily bars = {yrs:.1f} years. A pattern firing")
    print(f"  25 times needs ~{yrs / 25 * 12:.1f} months between fires; one firing 250 times fires")
    print("  roughly monthly. BOTH are affordable -- the constraint is not history length, it is")
    print("  that precision and frequency trade off against each other and the search penalty.")

    print("\n" + "=" * 100)
    print("GATE A VERDICT")
    print("=" * 100)
    floor_25_500 = noise_floor(base_g, 25, 500)
    floor_250_500 = noise_floor(base_g, 250, 500)
    floor_15_500 = noise_floor(base_g, 15, 500)
    print(f"""
  The question IS answerable, but only in one corner of the space, and it is not the corner the
  phrase 'almost never fails' points at.

  * A rare, near-perfect pattern is NOT establishable. At 25 fires and a 500-candidate screen the
    no-edge floor is already {floor_25_500:.3f} precision on gold. Any 'this almost always works'
    claim at that frequency is unfalsifiable on this data, however good it looks.

  * A FREQUENT, MODEST pattern IS establishable. At 250 fires the same screen's floor drops to
    {floor_250_500:.3f}, so a genuine edge only has to clear that to be real.

  So the search must be pointed at high-FREQUENCY, low-excess patterns and judged against the
  base rate -- the opposite of the intuition that a rarer signal is a safer one. Rarity is what
  makes a pattern unprovable, and it is also what makes it look certain.
""")


if __name__ == "__main__":
    main()
