# PRE-REGISTRATION — XAUUSD conditional-short variant

Written **before** any Phase 0b / Phase 1 experiment runs. Nothing below may be
re-parameterised after a result is seen. Commit this file, record its hash, then run.

Git hash at registration: see the commit that adds this file.
Date: 2026-09-09. Plan: `SHORT-VARIANT-PLAN.md`. Evidence: `V5_FINDINGS.md` §3am.

## 1. Fixed design parameters

| parameter | value | why fixed now |
|---|---|---|
| primary horizon K | **12 H4 bars (2 trading days)** | skew most negative here (-0.654) and oracle peaks here (+3.694) |
| horizon sweep | 6, 12, 24, 30, 42 H4 bars | all counted in any max-statistic null |
| label | **vol-neutral first-touch**: did -2% arrive before +2% within K bars | it is a trade; precision = win rate |
| barrier | +/-2.0%, symmetric | chosen from the down-move census (median in-band move -3.24%) |
| ambiguous same-bar double-touch | **dropped as undecided** | 46% of K=30 windows are undecided; M15 path may resolve them in Phase 1b |
| cost | 1.94bp one-way ($0.448 on median gold $1,733); 3.88bp round trip | live-measured, MANDATORY CONTROL #1 |
| breakeven precision | **0.5097** | derived, not chosen |
| eval window | 2018+ for H4; 2008+ for GOLD_D1 | matches `engine_bp`'s `EVAL_START` |
| bear regime | 2011-09-30 .. 2015-12-31 | defined from GOLD_D1 segmentation, not tuned |

## 2. The numbers to beat (all verified with `engine_bp`)

- Champion (benchmark): SR **+1.098** at 0.75bp.
- Short leg (`ls_signal` negative): **-0.216**; diversification hurdle `rho*c` = **-0.2143**.
- Required standalone short SR for book dSharpe +0.20: **+0.465**; for +0.10: **+0.256**.
- Bear-regime ceiling (unconditional short 2011-2015): **+0.475**. Margin: **+0.015**.
- Perfect regime switch: standalone +0.228, book dSharpe **+0.225**.

## 3. Decision rules

### Phase 0b gate — SPECIFICITY, not recall
A causal regime switch, fitted on strictly-past data only, must achieve
**false-positive rate <= 5% on non-bear periods at recall >= 50%**.
- FP <= 5% -> PASS, proceed to Phase 1.
- FP > ~10% -> **STOP the programme.** At 10% FP the prize is +0.059; no representation,
  feature, or COT series fixes a specificity problem.
Rationale (§3am): recall 100%->50% costs 0.015 book dSharpe; FP 0%->10% costs 0.166.

### Phase 1 PASS — all of the following
1. incremental information **>= 0.010 bits per independent event**;
2. above the **alignment null** 99th percentile (`block_shuffle`, block >= 2K, 200 draws);
3. above the **max-statistic null** 95th percentile if more than one arm is compared;
4. positive in **>= 7 of 9** OOS years;
5. implied precision at the chosen operating point **>= 0.607**;
6. **synthetic i.i.d. Gaussian control returns <= 0** information and AUC <= ~0.52.

"AUC went up by 0.02" is a **FAIL by construction**: SE(AUC) at these event counts is **0.0367**
and the best-of-8 pure-noise lift is **+0.0526**.

### Phase 4 acceptance — diversification-adjusted
Paired dSharpe **>= +0.20** on the combined book, paired **t >= 2.0**, **>= 7/9 years**, against
the block-shuffled null; plus **DSR >= 0.90**, **PBO < 0.30**, and survival of walk-forward
**re-selection** (CONTROL #6). A standalone Sharpe alone is not acceptance.

## 4. Admissibility (MANDATORY CONTROL #7)

A label or operating point is **inadmissible** unless it has **>= 45 decided, non-overlapping
events** and meets the false-positive bound. Verified counts: K=12 -> **274** decided events
(121 down / 153 up); K=30 -> **247** (104 / 143). Count decided events, never bars or windows —
counting windows overstated the K=30 sample by 1.9x.

## 5. Stopping conditions, declared in advance

The programme stops if any of these fire:
1. Phase 0b: no causal switch reaches FP <= ~10%.
2. Phase 1: CEILING arm fails its alignment null -> representation is not the bottleneck and
   the whole representation list is closed.
3. Phase 1c: no downside information across the 51-instrument panel.
4. Synthetic-noise control ever returns positive information -> stop and fix the pipeline
   before trusting any result.

## 6. Known hazards

- `GOLD_D1_long.csv` vs `XAUUSD_H4_long.csv`: **$177.07** max close difference, return
  correlation **0.8873**. Diagnose before any P&L splice.
- `XAUUSD_M15_spliced.csv` is a different feed from `XAUUSD_M15_long.csv` ($405 max difference).
  Do not mix.
- `probe_lookahead` cannot catch: state fitted outside the feature function, hyperparameters
  chosen on the full sample, or missing purge. Mitigations in `SHORT-VARIANT-PLAN.md`.
