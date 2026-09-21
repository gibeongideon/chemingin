# Pre-registration: CFTC positioning as a XAUUSD signal

Declared 2026-09-21, BEFORE any cell was scored. This is a NEW DATA TYPE for this repo — nothing
in V5_FINDINGS has ever used positioning data. Every prior study here has been price-on-price,
and the recurring conclusion (§3ao, §3be, §3bf) has been "you need a different data type, not a
different rule". This is that attempt.

## Why this, and why it is not another intraday trigger

The binding constraint on every recent study was cost, not signal: the directional content at a
session open or a release is 0.3-3bp gross against a 2.34bp round trip. A weekly signal does not
have that problem — gold's weekly move is ~150bp, so the same round trip is **1.5% of the
signal** rather than ~100% of it. Cost is structurally not the wall here, which is the first time
that has been true in this line of work.

It also matches the stated cadence: the report publishes once a week.

## THE LAG IS THE WHOLE GAME

CFTC Commitments of Traders is surveyed **Tuesday at close** and published **Friday 15:30 ET**.
A backtest that uses Tuesday's positioning to trade Tuesday's close is reading a number that did
not exist for three more days. This is the single most common way COT studies lie, and it is worth
roughly the whole reported effect in published critiques.

**Declared rule: a report dated Tuesday T is usable only from Friday T+3 at 20:30 UTC.** All
signals are therefore aligned to the FOLLOWING MONDAY's open, which is strictly later than the
release and needs no intraday precision. A secondary "cheating" arm that trades Tuesday's close
is run DELIBERATELY, reported alongside, and labelled — as the size of the lookahead, not as a
result.

## The series

Disaggregated futures-only report, COMEX gold. The measured quantity is **Managed Money** net
position (long minus short), normalised three ways because the raw contract count is not
comparable across a decade of changing open interest:

1. `net_oi`  — net / open interest
2. `z156`    — z-score of `net_oi` over the trailing 156 weeks (3 years), expanding, causal
3. `cotidx`  — the retail-standard COT index: percentile rank of `net_oi` in its trailing 156
               weeks, i.e. stochastic position in [0,1]

All three use PRIOR weeks only. No full-sample standardisation.

## Declared grid — 36 cells, and nothing else

| axis | values | n |
|---|---|---|
| measure | `net_oi`, `z156`, `cotidx` | 3 |
| rule | CONTRARIAN (fade crowded), MOMENTUM (follow) | 2 |
| threshold | extreme 20% / 30% of the trailing distribution | 2 |
| horizon | 1 week, 2 weeks, 4 weeks | 3 |
| | | **36** |

One entry convention (Monday open after release), one exit (horizon end), no tuning.

## Gates, applied mechanically

1. **Null on GROSS, signed maximum, over exactly these 36 cells.** §3bf: bootstrapping a
   net-of-cost series under a zero-mean null manufactures significance in proportion to sample
   size and always negatively, because the cost offset is deterministic. Costs are a hurdle on
   the point estimate.
2. **The mirror must oppose.** CONTRARIAN and MOMENTUM are exact opposites, so a consistent sign
   with a negative mirror is evidence; both positive is a bug.
3. **Per-year stability**, reported not optimised.
4. **EV per fire and the base rate beside any hit rate** (control #8).
5. **The skipped sample**: what did the baseline earn on weeks the trigger did NOT fire
   (`measure-the-skipped-sample`) — a trigger that only fires on weeks that were going to work
   anyway is an adverse-selection artifact.
6. The lagged arm is primary. The unlagged arm is a diagnostic of the lookahead's size.

## Expected outcome, stated in advance

Published evidence on COT positioning as a commodity timing signal is **mixed to weak**; the
effect that survives in the literature is at crowded extremes and is small. The honest prior is
that this fails, and the value of running it is that it fails (or not) on a data type this repo
has never touched, with the lag handled correctly.
