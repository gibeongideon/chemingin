# Pre-registration — "a daily signal that almost never fails" (XAUUSD, BTCUSD, BTC/XAU)

Committed BEFORE the first fit. Filed 2026-10-21.

## The question
A daily pattern whose appearance makes a given action near-certain. Explored on XAUUSD and
BTCUSD, plus the BTC/XAU ratio. `BTCXAU` DOES NOT EXIST as a tradeable symbol (checked: all
2,067 Switch Markets symbols; only XAUUSDs/XAUEURs/BTCUSDs/BTCEURs), so the ratio is synthesised
and would need two legs, costing 7.44bp round trip against gold's 0.75bp.

## What Gate A already settled (`scripts/v5_daily_certainty_gate_a.py`)
Base rates: GOLD 0.525 up-days, BTC 0.523. Cost at D1 is a rounding error (gold 0.75bp round trip
against a ~100bp daily sigma = 0.76% of one sigma), so **cost is not the binding constraint** and
cannot be the excuse. What binds is the **best-of-K noise floor**, p95 of the maximum precision
over K no-edge candidates:

| fires | K=1 | K=50 | K=500 | K=5000 |
|---|---|---|---|---|
| 15 | — | 0.867 | 0.933 | 0.967 |
| 25 | — | 0.800 | 0.880 | 0.920 |
| 50 | — | 0.740 | 0.780 | 0.820 |
| 250 | 0.576 | 0.620 | 0.640 | 0.656 |
| 500 | 0.560 | 0.592 | 0.606 | 0.618 |

**Therefore: a rare near-perfect pattern is UNFALSIFIABLE on this data and a frequent modest one
is testable.** Rarity is what makes a pattern look certain and also what makes it unprovable. The
search is pointed accordingly.

## Pre-registered decision rules
1. **K is counted and declared.** Every candidate evaluated counts, including ones abandoned.
2. **Precision is reported BESIDE its own base rate, never alone** (xau-drift-baseline-inverts-3r:
   thresholds must sit at the base rate, never 0.5).
3. **The benchmark is HOLDING the asset, not zero** (cot-positioning-disproven, t +6.28 at 4w).
   A signal must beat buy-and-hold over the same days, not merely be positive.
4. **CONTROL #8: EV per fire in bp and mean offset are reported with every precision.** A 5.9-sigma
   AUC lift once delivered -0.702 Sharpe.
5. **The null is a CIRCULAR SHIFT of the outcome series, not a permutation.** Shifting by a random
   lag >= 20 days preserves the autocorrelation of BOTH the signal and the outcome and destroys
   only their alignment; permutation would destroy the serial structure that makes these patterns
   fire in clusters and would therefore understate the floor.
6. **The whole search is re-run inside each null draw** and the MAXIMUM precision is recorded, so
   the p-value prices the search, not the winner (trigger-search-max-stat-null).
7. **A pattern is a result only if it clears ALL of:** p95 of the max-stat null; the base rate by
   more than one SE; a positive EV net of the measured live spread; AND the same sign in both
   halves of the sample (split-sample control, rule 1.3.8 — a pooled t of +2.25 with 7/9 positive
   years was still an artifact).
8. **Overlapping fires are counted honestly.** Streak and regime patterns fire in clusters, so the
   effective sample is reported alongside the raw count.

## Excluded as already closed — not re-tested
Window extremes / "at the N-day high" (tautology, 4 kills) · NR7 directional (non-directional
across 51 assets) · session/ORB and any intraday execution (sub-cost) · extreme-timing (drift) ·
COT / implied vol / real yields / DXY (0/60 beat holding gold) · pivot & turning-point detection
(closed on mechanism) · regime-gating the champion (SR 1.08->1.01).

## Expected outcome, recorded so it cannot be rounded up
Most likely: nothing clears rule 7, and the durable deliverable is the noise-floor table plus a
direct answer that the highest-reliability daily pattern in gold is the unconditional long. The
honest prize here is a bound, not a signal.
