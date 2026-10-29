# Pre-registration — OCTOBER round, Tier A

Committed BEFORE the first fit. Filed 2026-10-29. Full design: `OCTOBER.MD`, `OCTOBER-PLAN.md`.

## The bar every row must clear before it is even worth fitting

MANDATORY CONTROL #7: XAU H4 2018+ = 13,660 bars -> 455 windows -> **247 decided events**; required
AUC lift +0.024 against SE(AUC) **0.0367**; best-of-K pure-noise lift **+0.029 (K=3) / +0.053 (K=8)**.
*"Any approach that only nudges AUC is unfalsifiable here."*

**So each arm must RAISE POWER (P) or CHANGE THE TARGET (T). Declared per arm below. An arm that
does neither is a repeat and will not be run.**

## Arms, declared with K

| arm | route | K (cells) | primary statistic | pre-registered gate |
|---|---|---|---|---|
| **A1 bar-language** | P+T | 3 orders x 2 horizons = 6 | held-out perplexity vs baselines, then EV/fire | beats order-0 freq AND order-1 Markov AND a shuffled-token control; then EV/fire > 0 net of live cost |
| **A2 champ-meta H1** | T | 4 | matched-vol paired-t vs champion | t >= +2.0, positive in BOTH halves, EV/fire > 0 |
| **A3 strategy stack** | T | 5 | matched-vol paired-t vs champion AND vs the Ridge baseline | must beat both; features residualised against the champion first |
| **A4 volume zones** | T | 8 | EV/fire vs a flat ATR-offset control | oracle ceiling first (#5); must beat the flat-offset control that beat every ICT zone in 3bc |
| **A5 MAE regression** | T | 4 | oracle-MAE book dSharpe, then R² | **if ORACLE-MAE does not improve the book, close immediately** (the 3at pattern) |
| **A6 intraday re-price** | — | ~15 | sign flip at 0.504bp | a flip only reopens a cell; it must then clear the fixed null harness |

## Controls binding every arm
1. Control cell with a KNOWN answer in every script: champion ~+1.02, XAU+BTC book ~+1.284 at
   Switch costs. A wrong control means the harness is wrong, not the finding.
2. `intrabar_leak(d) > 0.15` screened before any fit. FX D1 excluded via `--drop-forward-stamped`
   -- note the bar alphabet's **WICK axis is exactly the leaked field** on those 18 files.
3. **Never `XAUUSD_H4_long.csv`** (spliced 2023-06-28; pre-2023 correlates 0.18-0.40 with real
   broker quotes). Rebuild from M15 or use the single-provenance Switch/FundingPips files.
4. Nulls on **GROSS**, pricing the **SIGNED maximum**; a test asserts a deliberately negative cell
   cannot pass.
5. EV/fire **and** mean offset beside every classifier claim (#8). Positive mean offset = a
   confirmation detector = untradeable regardless of precision.
6. Benchmark vs **buy-and-hold** (#2), never zero.
7. Do NOT reuse `v5_repr_ceiling.py::walk_forward_ll` -- it selects the model class on the TEST
   fold (`:219-232`), worth ~+0.006 bits, biased toward higher-capacity arms.

## Scoring: both bars, decided per candidate
- **Bar 1 (CONTROL #6):** DSR >= 0.90 AND PBO < 0.30 AND survives walk-forward re-selection.
- **Bar 2 (staged):** matched-vol t >= +2.0 vs the deployed book, both halves positive, then demo.
(The deployed XAU+BTC book itself FAILS bar 1 at DSR 0.000 / PBO 0.381.)

## Expected outcome, recorded so it cannot be rounded up
**Tier A closes 5 of 6.** A2 is the likeliest survivor because its label IS the economics; A4 is the
genuine unknown with no prior in this repo; A1 is the most interesting and the most likely to yield
a real information gain that no trade structure can collect. If that is the result, the round's
value is the BOUND -- as Gate A's best-of-K table was worth more than the 761 patterns it bounded.
