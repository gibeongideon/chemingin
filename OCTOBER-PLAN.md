<!-- Promoted from the Claude Code plan file `i-wanttt-you-to-playful-widget.md` on
     2026-10-29 at the user's request, so the executed plan lives in the repo next to its
     deliverable (`OCTOBER.MD`) rather than only in ~/.claude/plans/. Verbatim apart from
     this header. -->

# OCTOBER.MD — 20 approaches, and the 6 that run first

## Context

After ~130 failed pre-registered structures the repo declares the signal side exhausted. The user
wants a institutional-grade exploration round anyway, built around: the ZigZag revisited with
simple AI, **strategies-as-features in an ensemble**, **candle structure as a language**, and
**time-bucketing as a possible edge-killer**. Deliverable is `OCTOBER.MD` with 20 approaches, then
immediate execution of the best tier.

### The three facts that shape everything below

**1. THE USER'S CENTRAL IDEA IS BUILT AND UNRUN.** A complete bar-language stack exists:
- `src/features/bar_tokenizer.py` (334 lines) — a **45-symbol candle alphabet**, `TOKEN = DIR_SIZE_WICK`
  (`UP|DN|DJ` × `XS|SM|MD|LG|XL` × `BW|NW|TW`), with `encode_ids()` and `encode_sequence()`.
- `src/models/bar_lm_model.py` (440 lines) — a 4-layer Transformer over that vocabulary, **registered
  in `model_registry.py`, configured in `config.yaml`, and NEVER TRAINED.** No checkpoint on disk,
  zero mentions in 6,232 lines of findings.
- `src/core/suffix_ae_sizer.py` — a **suffix automaton over a 3-letter price-DNA alphabet**. Zero
  mentions anywhere. Never measured.
- `src/features/meta_labels.py` — triple-barrier labels, **zero callers in the entire repo**.

**2. §3ao DOES NOT CLOSE IT**, for three reasons found in the repo's own text:
- §3ao's window `W` is only `[z-return, normalised range] × 64`. **No open, no body/range ratio, no
  wick asymmetry.** The `DIR_SIZE_WICK` alphabet is *not* a measurable function of it, so the
  `I(Y;R(W)) ≤ I(Y;W)` argument does not reach it.
- That argument fixes the label `Y`. **CONTROL #8's own scope note exempts "programmes that change
  the TARGET"**, and next-symbol prediction changes it.
- No n-gram, Markov transition matrix or next-token objective exists in any form.

**3. NO STACKED ENSEMBLE OVER REPO SIGNALS HAS EVER BEEN BUILT.** Verified three ways. The nearest
object, `src/cta/ml_combine.py` (purged walk-forward Ridge over 4 EWMAC speeds + xsmom + rev5 +
vol_z), **has no recorded result anywhere** — and is the most reusable stacking plumbing here.

### What the repo says will kill these, and must therefore be designed against

- **CONTROL #7 is the binding constraint**: XAU H4 yields ~250 decided non-overlapping events,
  SE(AUC) 0.0367, best-of-8 noise floor **+0.053**. *"Any approach that only nudges AUC is
  unfalsifiable here."* **So only two routes exist past it: RAISE POWER or CHANGE THE TARGET.**
- **CONTROL #8**: a 5.9σ AUC gain produced dSharpe **−0.702**. Report EV/fire and mean offset.
- **The conversion step, not the model, is named as the real constraint** (BEST_FINDINGS #5).
- **Collinearity will sink a naive stack**: SuperTrend↔champion **0.77–0.88**, champion↔buy-and-hold
  **0.96**, champion↔trim variants 0.96. *"A genuinely different return stream needs a genuinely
  different EDGE SOURCE."*
- **Combination has failed five times**: ICT confluence 0/4, 6-component stack 0/29, feature-block
  stacking, two books combined (1.41→1.26), book-level M15 trim 0/32.
- **Three leak classes a stack is uniquely exposed to**: feature-side lookahead (shuffled-label
  controls cannot see it), data misalignment (`intrabar_leak > 0.15`), and **provenance** (§3ba — a
  label resolved on one series with features from that same series holds privileged information).
  **A stack mixing H4- and M15-derived signals hits the third directly.**
- **Time-bucketing is already half-closed**: §3bw recency weighting `corr(IS,OOS)` +0.364 → **+0.140**
  and top-10 OOS precision down to the base rate; §3ax independently negative. **But calendar-period
  exclusion, train-window-length tuning, sliding-vs-expanding, and HMM-state-conditioned *training*
  have never been done.**

### User decisions taken

Document all 20 **and execute the top tier now** · build the **token sequence model**, not LLM API
calls · **score every survivor against both bars** (CONTROL #6 *and* staged demo) and decide per
candidate.

---

## Deliverable 1 — `OCTOBER.MD`

Format follows `EXPLORATION-QUEUE.md` (numbered tiers, one table row per approach). Every row
carries: **hypothesis · why it is not already closed (with § citation) · data & power · controls ·
pre-registered gate · honest prior · kill rule**. Ranked, not listed.

### Tier A — runs immediately (6)

| # | approach | why it is open |
|---|---|---|
| **1** | **Candle-language sequence model.** `bar_tokenizer` 45-token alphabet on **270,458 M15 gold bars**; order-1/2/3 **Markov transition matrix and n-gram first**, `bar_lm_model` transformer only if the cheap version shows signal. Target = next-token distribution, then the economically-relevant marginal (P(next body is UP·LG)). | The alphabet carries **wick asymmetry and body/range, which §3ao's `W` lacks**; next-token **changes the target**, which #8 exempts; and 270k tokens is the one place this repo has abundant sample against its ~250-event wall. |
| **2** | **Champ-meta at H1** (queue #35 verbatim). Target `y = sign(champion position × forward return)`. | *"The one target where control #8's gap CANNOT open, because the label IS the economics."* Oracle **+2.534, t +17.16, 15/15**. Done at H4 (AUC 0.593, best ever here) and at D1 (**retracted on FX forward-stamp leakage**). **Never at H1, never on clean non-FX data.** |
| **3** | **Strategies-as-features stack.** ~25 repo signals → meta-model. Target is **champion P&L sign**, not direction. Baseline = `ml_combine.ml_forecast` Ridge (which has no recorded result). | **Never built.** Designed against the known trap: residualise every feature against the champion first, so the stack can only be credited for what is *orthogonal* to it. |
| **4** | **Volume-derived supply/demand** — volume profile, POC, value area, HVN/LVN. | **Zero hits for volume profile / POC / value area / footprint in 6,232 lines or any `src/` file.** Every zone the ICT sweep tested was **swing-derived**. This is the one untouched branch, and the closest available proxy to the order flow §3r names as the only data type that could change the answer. |
| **5** | **Regression on forward MAE (max adverse excursion).** | Only **one** regression target has ever existed here (§3at volatility) and it closed on a **negative oracle**, not model failure. MAE **changes the target** (#8-exempt) and feeds **stop placement** — the conversion step BEST_FINDINGS names as the actual binding constraint. |
| **6** | **Re-price every intraday closure at the live 0.504bp.** | §3bx revised the cost wall (M15 breakeven **0.6363 → 0.5343**) and then re-priced only **two** of ~15 intraday closures. Every other one was decided at 1.5–2.34bp. Mechanical, cheap, and §3bx itself says the revision *"widens what is worth testing for non-trend structures."* |

### Tier B — queued, designed (9)

7 Triple-barrier via `meta_labels.py` (**zero callers**) + uniqueness weighting (queue #3, never
implemented) · 8 Pool the 4 book sleeves with an instrument dummy (queue #4) · 9 BTC **2-cell**
pre-registered speed test — §3bp's own stated promotion path, never executed (DSR 0.743 vs 0.90) ·
10 Regime-gate **BTC** — §3u's explicit untaken escape clause, *"gating an asset that lacks the
champion's own adaptive risk control"* · 11 Suffix-automaton price-DNA sizer (built, never measured)
· 12 Train-window length & sliding-vs-expanding (never compared; distinct from recency weighting) ·
13 **HMM-state-conditioned training** (the HMM was only ever a *feature*, never a training
partition) · 14 Tops via a **non-classification** framing (*"all 6 attempts were variations on
supervised classification of a swing-high label"*) · 15 ORB re-measured on the **fixed** harness at
live cost (§3bx-D destroyed §3be's basis, so it is now genuinely unmeasured rather than closed).

### Tier C — designed, low prior, not run (5)

16 Conformal / selective prediction (queue #26) · 17 Cross-pair **token** transfer — train the
bar-LM on 51 series, predict gold (pooling gave the biggest AUC lift in repo history) · 18
Quantile/pinball regression of the return distribution · 19 Monotonic constraints (queue #25) · 20
**Calendar-period exclusion** (COVID/2020) — the user's "trim out some period", run with the N_eff
bar moved in the same step per §3bw's rule.

---

## Deliverable 2 — execute Tier A

**Pre-registration first**, `data/v5_runs/october/PREREGISTRATION.md`, committed before the first
fit, git hash recorded in results (the `data/v5_runs/short_variant/` pattern).

**Build** (reuse, do not rewrite):
- `src/features/bar_tokenizer.py::encode_ids` · `src/models/bar_lm_model.py` ·
  `src/features/meta_labels.py::triple_barrier_labels`
- `src/cta/ml_combine.py::build_features/ml_forecast` (purged Ridge, `purge_mult=1.6`)
- `src/evaluation/dsr_pbo.py::{deflated_sharpe_ratio, pbo_cscv}` · `src/cta/bootstrap.py::block_bootstrap_sharpe`
- `src/v5/advisor_calibration.py` (`brier_skill`, `reliability_table`, `operating_points`, `ece`)
- `scripts/v5_daily_certainty_scan.py::maxstat_null` — **the strongest null design here** (circular
  shift of the outcome ≥20 days, whole search re-run per draw). **Promote it to `src/evaluation/`;
  there are currently nine near-duplicate local copies.**
- `scripts/v5_pooled_bottom_detector.py::intrabar_leak` (threshold 0.15) ·
  `src/evaluation/lookahead_probe.py` (size the panel to the slowest window — 1536 H4 bars)
- `src/v5/mtf_frames.py::aggregate` for any H4 rebuild — **never** `XAUUSD_H4_long.csv` (spliced
  from 2023-06-28; pre-2023 correlates **0.18–0.40** with real broker quotes)

**New scripts:** `v5_bar_language.py` · `v5_champ_meta_h1.py` · `v5_strategy_stack.py` ·
`v5_volume_zones.py` · `v5_mae_regression.py` · `v5_intraday_reprice.py`

**Do NOT reuse `v5_repr_ceiling.py::walk_forward_ll`** — it **selects the model class on the test
fold** (`:219-232`), worth ≈ +0.006 bits and biased toward higher-capacity arms.

### Gates, pre-registered per approach
Both bars reported for every survivor, per the user's choice:
- **Bar 1 (CONTROL #6):** DSR ≥ 0.90 **and** PBO < 0.30 **and** survives walk-forward re-selection.
- **Bar 2 (staged):** matched-vol paired-t ≥ +2.0 vs the deployed book, positive in **both** sample
  halves, then demo.
- **Always:** EV/fire **and** mean offset (#8) · benchmark vs **buy-and-hold** (#2) · oracle ceiling
  **before** any detector (#5) · event count + best-of-K floor **before** fitting (#7) · costs at the
  live quote (#1).

### Honest priors, recorded so nothing is rounded up
#1 the most likely to show *something* and the most likely to be unmonetisable — expect a real
next-token lift that fails EV/fire · #2 the best risk-adjusted bet, because its label is the
economics · #3 expect collinearity to eat it; the orthogonalisation is what makes it a test rather
than a repeat · #4 genuinely unknown, no prior in this repo · #5 expect real R² and negative
economics, the §3at shape · #6 expect 1–2 of ~15 to flip sign and none to clear a gate.
**Most likely overall outcome: Tier A closes 5 of 6 and the document's value is the bound.**

## Verification

1. Pre-registration committed before the first fit; git hash in every results file.
2. **Control cell with a known answer in every script** — the champion must reproduce ~+1.02 and the
   XAU+BTC book ~+1.284 at Switch costs. This is how the `scale=1/6` bug, the ZigZag deadlock and
   the Gate-D path-index bug were all caught.
3. **Leak screen before any fit**: `intrabar_leak > 0.15` on every series; `--drop-forward-stamped`
   for the 18 FX D1 files; single-feed provenance asserted per §3ba.
4. **Nulls on GROSS with the signed maximum** — a test must assert a deliberately negative cell
   cannot pass (§3bx-D: a cost offset inside a zero-mean null manufactured `p 0.0000` on a dead study).
5. Tokenizer sanity: round-trip `encode_ids` → token counts; assert the 45-symbol vocabulary is
   actually populated and not collapsed onto 3 symbols by a bad ATR scale.
6. **No live config changes.** Switch 360595, FundingPips 360598/360597 untouched; 0 failed units.
7. Write-up: a new `V5_FINDINGS.md` section per approach carrying the verdict **including nulls**,
   `OCTOBER.MD` row statuses updated, and a memory entry for each durable fact.
