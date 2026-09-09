# XAUUSD conditional-short variant — measurability-first plan

**Status: STOPPED 2026-09-09 at the Phase 0b gate.** Phase D complete, Phase 0b run and
FAILED — all 15 causal regime switches give a negative short Sharpe (best -0.106 vs a +0.465
requirement); best book contribution **+0.046** against a perfect-switch ceiling of +0.225.
Phase 1 then RAN and closed the entire representation list (§3ao: raw-window CEILING dI -0.0099,
inside its null). Its one positive — the M15 intrabar path (dI +0.0142, above the alignment
null's p99) — was tested as a champion trim overlay and **DISPROVEN 0/16** (§3ap: 12 of 16 cells
clear t@matched +1.50 but every cell is negative in 2018-21 and positive in 2022-26; PBO 0.687).
**PROGRAMME COMPLETE — nothing here remains to run.**
**Scoping evidence:** `V5_FINDINGS.md` §3am. **Pre-registration:**
`data/v5_runs/short_variant/PREREGISTRATION.md`.

## Goal and form

A second XAUUSD variant that can **SELL**, targeting moves that resolve in ~4 hours to 1 week.
Form fixed with the user: **conditional short only** — flat by default, net short only on a
strong down-signal, **never symmetric** (the symmetric short is closed; the H4 damping curve is
monotone with no interior optimum). XAUUSD only; portfolios out of scope until this is
exhausted.

## The three numbers that define the problem

1. **Prize:** a *perfect* regime switch is worth **+0.225 book dSharpe**. Required standalone
   short Sharpe for +0.20 is **+0.465**.
2. **Ceiling:** in the only bear regime available (GOLD_D1 2011-09..2015-12, gold -9.4%/yr) an
   **unconditional** short scores **+0.475**. Margin over the requirement: **+0.015**. Razor thin.
3. **The wall:** a **10% false-positive rate destroys 74% of the prize** (+0.225 -> +0.059),
   while cutting recall from 100% to 50% costs only 0.015. **Specificity is the binding
   constraint, not recall.**

Also settled during scoping: cost is *not* the obstacle (short-only H4 gross -0.211 vs -0.225 at
$0.448 — cost is 6.6% of the gap); the short leg sits essentially **on** the diversification
hurdle (-0.216 vs -0.2143); trimming to flat already captures **71%** of the oracle prize;
downside-specific features are worth **+0.007 AUC**.

## Horizon

Primary **12 H4 bars (2 days)**, swept 6-42. Two independent measurements agree: forward-return
skew is most negative at 12 bars (-0.654, decaying to -0.278 by 5 days), and the oracle decays
on the same axis (+3.694 -> +2.740 -> +2.239). The left tail is *shallower* than the right at
every horizon (|p5|/p95 ~0.89), so a short must earn everything from timing abrupt drops.

## Phases

### Phase D — documentation (COMPLETE)
`V5_FINDINGS.md` §3am; MANDATORY CONTROL #7 (measurability + specificity gate); this file; the
pre-registration; memory entry `xau-short-variant-scoping`.

### Phase 0 — the gate (1.5 days, no new features)
**0a.** Pre-registration committed with its git hash. Done as part of Phase D.
**0b. THE GATE — bear-regime holdout, re-specified around specificity.** Diagnose the
`GOLD_D1` <-> `XAUUSD_H4` feed offset first ($177.07 max close difference, return corr 0.8873 —
fine for an information test, never splice into a P&L series undiagnosed). Then, on
2008-2026 with 2011-09..2015-12 as the target regime, measure **what false-positive rate a
causal regime switch actually achieves** using only strictly-past data.
- **PASS** if a causal switch achieves **FP <= 5%** on non-bear periods at recall >= 50%.
- **STOP** if no causal switch gets under ~10% FP. At 10% the prize is +0.059, which is not
  worth a deployment, and no representation, feature or COT series changes a specificity
  problem.
**0c. COT as a regime-tilt input only** (1 day, folded in). Honest Friday-release timestamps.
Not a timing signal — 50 obs/yr against 9-22 events/yr.

### Phase 1 — information ceiling, not a representation ladder (2-3 days, only if 0b passes)
Every candidate representation is a measurable function of the same trailing window, so
`I(Y;R(W)) <= I(Y;W)`. **Estimate the ceiling once.** Three arms: BASE (the §3r REGIME+PRICE
block), **CEILING** (the raw trailing window into a heavily regularised high-capacity ensemble),
SOURCE (BASE + M15 intrabar path + GOLD_D1 2008+ pooled). If CEILING's incremental information
is ~0, no representation of the H4 window can help — including all eight originally listed — and
Phase 1 closes in one run.
Statistic: **incremental information in bits per independent event** (OOS log-loss reduction on
non-overlapping events), not AUC. Label: **vol-neutral first-touch** (did -2% arrive before +2%
within K bars) — it is a trade, its precision is its win rate, breakeven precision 0.5097.
Three required nulls: alignment (`block_shuffle`, block >= 2K, 200 draws), max-statistic
(`max_stat`/`trigger_null_p`) if more than one arm is compared, and a **synthetic i.i.d. Gaussian
control** — any positive information there is a pipeline leak, full stop.

### Phase 1c — power escape (2 days, parallel)
The same information test as a **51-instrument D1 panel**, 5-day horizon, 2008+: ~47,000
independent windows, SE(AUC) 0.037 -> ~0.003. **Falsification only** — no downside information
across 51 instruments means none on gold; the converse does not follow.

### Phase 1b — at most TWO representations (2-3 days, only if Phase 1 passes)
(i) **M15/M30 intrabar path** — genuinely outside the H4 window, probe-clean, and it resolves
the first-touch label's 46% undecided windows. (ii) **Matrix profile**, per-fold causal window,
abandoned the moment the probe or the noise control fires (`stumpy.stump` over a whole series
leaks by construction). Nothing else earns a run.

### Phase 2 — exit asymmetry only (2 days)
Drop the feature work (+0.007, measured). Keep **short-side take-profit**: `V5_FINDINGS.md:323`
("do not add profit-taking") was measured on the *long* book, and a short fighting +14%/yr drift
genuinely should take profit — an asymmetry, not a repeat. Wire
`staged_position(..., side="short")` at H4: it exists at `scripts/v5_xau_fast_ls_exits.py:132`
and has **never been called**.

### Phase 4 — build and gate (1 week, only if 0b and 1 pass)
Acceptance is **diversification-adjusted**: paired dSharpe >= +0.20 on the combined book,
paired t >= 2.0, >= 7/9 years, against the block-shuffled null. Plus DSR >= 0.90, PBO < 0.30,
and survival of walk-forward *re-selection* (CONTROL #6).

## Leakage rules (non-negotiable)

- `probe_lookahead` at **`n=3500, warmup_buffer=2200`** plus a 400-bar offset — the champion's
  slowest span is 1536 H4 bars, and a shorter panel "passes" by comparing nothing.
- Add a **future-replacement probe**: replace all bars after *t* with a different random path,
  require bit-identical features. Truncation catches deletion; this catches insertion.
- **Every representation must be a pure `fn(df) -> DataFrame` with no external fitted state.**
  An offline-fitted encoder is perfectly truncation-invariant and completely leaked — this is
  why `src/features/latent_encoder.py` is excluded rather than leak-audited.
- **Every scalar hyperparameter re-derived per fold** from strictly-past data (fracdiff *d*, DC
  threshold, matrix-profile window, SAX breakpoints).
- Purge copied verbatim from `walk_forward_prob`, not re-derived.

## Reuse, do not rewrite

| need | use |
|---|---|
| walk-forward block ablation with implied precision/recall | `scripts/v5_xau_intermarket_accuracy.py` (`score_pivot` already emits `p_at_r20`/`rec_at_p70`) |
| engine / paired-t / per-year / attainability frontier / block shuffle | `scripts/v5_xau_turn_prob.py` |
| bp-cost engine, sleeve loader | `scripts/v5_volregime_taper_crossasset.py` |
| max-statistic bootstrap | `scripts/v5_trigger_search.py::max_stat`, `trigger_null_p` |
| short-side exits | `scripts/v5_xau_fast_ls_exits.py::staged_position(side="short")` |
| full mandatory pipeline | `src/evaluation/walk_forward_grid.py::run_concept_pipeline` (zero callers, written for this) |
| leak probe | `src/evaluation/lookahead_probe.py` |
| DSR / PBO | `src/evaluation/dsr_pbo.py` |

## Closed — do not re-run

Symmetric long/short at any horizon; carry-shorts (monotone harm despite +2.06%/yr short carry);
fast-horizon long/short (0/64 exit cells clear +0.50); tops/sell classification (**six**
failures, all "supervised classification of a swing-high label"); the representations in §3am
list C; crash-hedge shorts (0.97 vs 1.26); session/hour filters; SL/TP grids on a 52%-accuracy
classifier; "search more triggers".
