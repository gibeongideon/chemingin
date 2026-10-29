# OCTOBER — 20 approaches, ranked, gated, with honest priors

**Filed 2026-10-29.** Successor to `EXPLORATION-QUEUE.md` (38 items, ~32 closed). Scope set by the
user: revisit the ZigZag with simple AI, combine repo strategies as **features in an ensemble**,
treat **candle structure as a language**, test whether **time-bucketing is the edge-killer**, add
supply/demand and anything else — mathematical or not — and take `BTCUSD`/`XAUUSD` as the base.

---

## 0. The honest prior, and the only two ways past it

**~130 pre-registered trade structures have failed here.** `RESEARCH-REGIMES.md` §1.1 states *"the
signal side is exhausted for this strategy family"*, and `BEST_FINDINGS.MD` #5 is sharper: *"this
project's binding constraint is not model capacity or feature representation. It is the conversion
step — forecast to net-of-cost position surviving a regime split."*

**MANDATORY CONTROL #7 is the operative wall.** XAU H4 2018+ yields 13,660 bars → 455 windows →
**247 decided events**. Required AUC lift +0.024 against SE(AUC) **0.0367** = a 0.65σ question. The
best-of-K pure-noise lift is **+0.029 at K=3 and +0.053 at K=8**. Hence: *"any approach that only
nudges AUC is unfalsifiable here."*

> **Therefore every approach below must do one of exactly two things, and each row says which:**
> **(P) RAISE POWER** — more independent events than gold's ~250.
> **(T) CHANGE THE TARGET** — CONTROL #8's own scope note exempts *"programmes that change the
> TARGET"*; it only retires those whose mechanism is "rank the same label better."
>
> An approach that does neither is a repeat, however new it sounds.

**Five prior refutations of "combination"** are on record and constrain Tier A #3 specifically: ICT
confluence 0/4 · the 6-component stack 0/29 · feature-block stacking adding nothing over one block ·
combining two books *hurting* (1.41 → 1.26) · book-level M15 trim 0/32. *"Stacking raises precision;
on this asset precision has never been the failing variable."*

---

## 1. What the reconnaissance found — three facts that reshaped this document

### 1.1 The user's central idea is already BUILT and has NEVER been run

| file | what it is | status |
|---|---|---|
| `src/features/bar_tokenizer.py` (334 ln) | A **45-symbol candle alphabet**. `TOKEN = DIR_SIZE_WICK`: `DIR ∈ {UP,DN,DJ}` × `SIZE ∈ {XS,SM,MD,LG,XL}` (body in ATR multiples) × `WICK ∈ {BW,NW,TW}` (bottom/neutral/top dominant). 45 + PAD + UNK = 47. Exposes `encode_sequence()` and `encode_ids()`. | **Code only.** Zero mentions in 6,232 lines of findings. |
| `src/models/bar_lm_model.py` (440 ln) | *"Bar Language Model — a small Transformer encoder (~200k params) trained on discretised OHLCV token sequences... the model learns sequential patterns in this 'bar language'."* 4 × TransformerEncoderLayer, 4 heads, d_model 32. | **NEVER TRAINED.** Registered in `model_registry.py:245`, configured in `config.yaml:208`, handled by `walk_forward.py:359`. **No checkpoint on disk.** |
| `src/core/suffix_ae_sizer.py` | A **suffix automaton** over a 3-letter price-DNA alphabet `{U,D,F}`, window 16, history 500 — "how familiar is the current sequence against the full historical pattern index". Ported from MQL `CMoneySuffixAE`, has unit tests. | **Zero mentions in any document. Never measured.** |
| `src/features/candle_tokenizer.py` | KMeans k=32 candle archetypes ("hammer/pin", "doji/indecision", ...). | Wired into `pipeline.py`, **`candle_tokenizer_enabled: false` everywhere.** |
| `src/features/meta_labels.py` | `triple_barrier_labels()`, `side_barrier_meta_label()`. | **Implemented, ZERO callers in the entire repo.** |
| `src/models/e2e_lstm.py` | 2-layer LSTM → 3-class. | *"exists, never on XAU"* (queue #24). |

### 1.2 §3ao does NOT close "candle structure as a language"

The representation ceiling (dI **−0.0099**, inside its synthetic null) is real and closes fracdiff,
directional change, entropy/complexity, recurrence graphs, matrix profile, path signatures, SAX and
Renko at one multiple-testing penalty. **It does not reach the bar alphabet, for three reasons
visible in the repo's own text:**

1. **The tested window is not the OHLC window.** `v5_repr_ceiling.py:141` builds only
   `z = log(c/c₋₁)/σ` and `rng = (h−l)/c` — **no open, no body/range ratio, no wick asymmetry.**
   `DIR_SIZE_WICK` is therefore **not a measurable function of `W`**, so `I(Y;R(W)) ≤ I(Y;W)` does
   not bind it.
2. **The DPI argument fixes `Y`.** Next-symbol prediction changes it, and #8 exempts that.
3. **It was a feature test, not a sequence test.** `f_ceiling` emits 128 *independent lagged
   columns* into i.i.d.-row classifiers; nothing consumes order. Corpus grep: **`n-gram` 0 hits ·
   `markov` 0 · `next-token` 0 · `LSTM` 0 · `transformer` 0 · `candlestick` 0 · `doji` 0.**

### 1.3 No stacked ensemble over repo signals has ever existed

Verified by import graph, filename scan and document scan. The nearest object,
`src/cta/ml_combine.py::ml_forecast` (purged walk-forward Ridge over 4 EWMAC speeds + xsmom + rev5 +
vol_z, `purge_mult=1.6`), **has no recorded result anywhere** and is the most reusable stacking
plumbing in the repo.

**The trap it must be designed against**, measured: SuperTrend↔champion **0.77** on XAU and
**0.86–0.88** on the second basket; champion↔buy-and-hold **0.96**; champion↔trim variants 0.96.
*"A genuinely different return stream needs a genuinely different EDGE SOURCE."*

---

## 2. TIER A — runs immediately

| # | approach | route | hypothesis & why it is open | gate | honest prior |
|---|---|---|---|---|---|
| **A1** ✅**CLOSED §3cc** | **Candle-language sequence model.** 45-token alphabet on **270,458 M15 gold bars**. Order-1/2/3 **Markov transition matrix and n-grams FIRST**; the transformer only if the cheap version shows signal. | **P + T** | 270k tokens against the ~250-event wall is the largest power gain available without new data, and next-token changes the target. §1.2 above establishes it is unclosed. | Must beat (a) an **order-0 token-frequency** baseline, (b) an **order-1 Markov** baseline, and (c) a **shuffled-token** control, on held-out perplexity AND on EV/fire of the one economically-relevant marginal. | **Most likely to show something and most likely to be unmonetisable** — expect a real perplexity lift that dies at EV/fire. |
| **A2** ✅**CLOSED §3cd** | **Champ-meta at H1** (queue #35 verbatim). Target `y = sign(champion position × forward return)`. | **T** | *"The one target where control #8's gap CANNOT open, because the label IS the economics."* Oracle **+2.534, t +17.16, 15/15**. Run at H4 (AUC 0.593, best ever here) and at D1 (**retracted — FX forward-stamp leak**). **Never at H1 on clean non-FX data.** | Matched-vol paired-t ≥ +2.0 vs the champion, both halves, EV/fire positive, and the FX panel excluded via `--drop-forward-stamped`. | **The best risk-adjusted bet in the document**, precisely because monetising the label is the definition of the label. |
| **A3** ✅**CLOSED §3ce** | **Strategies-as-features stack.** ~25 repo signals → meta-model, target = **champion P&L sign**. Baseline = `ml_combine.ml_forecast`. | **T** | Never built. **Designed against the known trap: every feature is residualised against the champion first**, so the stack can only be credited for what is orthogonal to it. | Must beat the Ridge baseline AND the champion at matched vol, with the orthogonalisation in place. Collinear features must not be able to carry it. | Expect collinearity to eat it. The residualisation is what makes this a test rather than a sixth repeat of "combination". |
| **A4** | **Volume-derived supply/demand** — volume profile, POC, value area, HVN/LVN. | **T** | **Zero hits for volume profile / POC / value area / footprint in 6,232 lines or any `src/` file.** Every zone the 12-concept ICT sweep tested was **swing-derived**. This is the one untouched branch of the supply-demand family, and the cheapest proxy for the order flow §3r names as *the* data type that could change the answer. | Oracle ceiling first (#5). Then EV/fire vs a flat ATR-offset control — the control that beat every ICT zone in §3bc. | **Genuinely unknown. No prior in this repo**, which is itself the reason to run it. |
| **A5** | **Regression on forward MAE** (max adverse excursion). | **T** | Only **one** regression target has ever existed here (§3at volatility) and it closed on a **negative oracle**, not model failure. MAE feeds **stop placement** — the conversion step `BEST_FINDINGS` #5 names as the actual binding constraint. | Oracle-MAE first: if perfect MAE foresight does not improve the book, close immediately (the §3at pattern). Then R² *and* matched-vol economics. | Expect real R² and negative economics. Running the oracle first makes that cheap to establish. |
| **A6** | **Re-price every intraday closure at the live 0.504bp.** | — | §3bx revised the cost wall (M15 breakeven **0.6363 → 0.5343**) and then re-priced only **two** of ~15 intraday closures. **Every other one was decided at 1.5–2.34bp.** §3bx itself says the revision *"widens what is worth testing for non-trend structures."* | Any cell that flips sign must then clear the fixed null harness (GROSS, signed maximum) before it counts as reopened. | Expect 1–2 of ~15 to flip sign and none to clear a gate. Cheap and mechanical. |

---

## 3. TIER B — designed, queued

| # | approach | route | why it is open |
|---|---|---|---|
| **B7** | **Triple-barrier labels + uniqueness weighting.** | P | `meta_labels.py` has **zero callers**; queue #3 (López de Prado ch.4 average-uniqueness weighting) was **never implemented** — *"the repo drops overlapping events entirely; weighting keeps them at honest weight."* Directly attacks the 247-event wall. |
| **B8** | **Pool the 4 book sleeves** (XAU/BTC/NDX/BRENT) with an instrument dummy. | P | Queue #4, never run. *"§3ae only ever pooled RETURNS, never LABELS."* 4× the events on the one label family the file calls economically live. |
| **B9** | **BTC 2-cell speed test.** Incumbent 1/6 vs one declared alternative. | — | §3bp's **own stated promotion path**, never executed: *"a pre-registered 2-cell test, not a wider sweep."* BTC fails only DSR (0.743 vs 0.90) and its walk-forward speeds **wander the whole grid**, which is exactly what a 2-cell test is for. |
| **B10** | **Regime-gate BTC.** | — | §3u's explicit, untaken escape clause: *"untested: gating a DIFFERENT signal, or gating on an asset that lacks the champion's own adaptive risk control."* BTC is precisely such an asset. |
| **B11** | **Suffix-automaton price-DNA sizer.** | T | `src/core/suffix_ae_sizer.py` is implemented and unit-tested with **zero mentions in any document**. A sizing multiplier is a different target from a direction label. |
| **B12** | **Train-window length; sliding vs expanding.** | — | Never compared as a finding (`config.yaml` just says `expanding`). **Distinct from recency weighting** (§3bw), which weighted *within* a window rather than choosing the window. |
| **B13** | **HMM-state-conditioned TRAINING.** | — | The causal HMM (`DiagGaussianHMM`, `filter_causal`) was only ever a **feature** and added nothing (§3r). Using its state to *partition training* is a different operation and has never been done. |
| **B14** | **Tops via a non-classification framing.** | T | `BEST_FINDINGS` Honest Gaps: *"all 6 attempts were variations on SUPERVISED classification of a swing-high label... whether every attempt so far has shared some blind spot the framing itself creates is not resolved."* Largest oracle prize in the repo (+1.252). |
| **B15** | **ORB re-measured on the fixed harness.** | — | §3bx-D destroyed §3be's statistical basis (the null was recentred while the observed statistic carried −cost). **The ORB question is now genuinely unmeasured rather than closed**, and the live cost is 3× cheaper than the figure it died at. |

---

## 4. TIER C — designed, low prior, not run

| # | approach | why it is last |
|---|---|---|
| **C16** | Conformal / selective prediction (queue #26). | Abstention cannot create edge where precision was never the failing variable. |
| **C17** | **Cross-pair token transfer** — train the bar-LM on 51 series, predict gold. | Pooling gave the largest AUC lift in repo history (+0.0576, 5.9σ) **and lost money** (dSharpe −0.702). Only worth running if A1 survives. |
| **C18** | Quantile / pinball regression of the return distribution. | Changes the target, but the repo has no mechanism story for why distribution shape would be more forecastable than its sign. |
| **C19** | Monotonic feature constraints (queue #25). | A regularisation, i.e. the weakest possible lever against a power problem. |
| **C20** | **Calendar-period exclusion** (COVID / 2020 / pre-2015). | The user's "trim out some period". §3bw already showed that shrinking the sample manufactures results: at a 250-day half-life the search reported a **0.974-precision pattern and 54 survivors vs 3**, entirely from N_eff falling 226 → 33. **Runnable only with the significance bar moved in the same step.** |

---

## 5. Non-negotiable controls for every row

1. **Pre-register before the first fit**, git hash recorded in the results file.
2. **A control cell with a known answer in every script** — the champion must reproduce ~+1.02, the
   XAU+BTC book ~+1.284 at Switch costs. This caught the `scale=1/6` bug, the ZigZag deadlock and
   the Gate-D path-index bug.
3. **Leak screen before any fit**: `intrabar_leak(d) > 0.15`; `--drop-forward-stamped` for the 18 FX
   D1 files (`clpos`/`upwick`/`lowick` carry `ret[t+1]` at corr 0.42–0.53 — **note the WICK axis of
   the bar alphabet is exactly that field**); single-feed provenance asserted per §3ba.
4. **Never use `XAUUSD_H4_long.csv`** — spliced from 2023-06-28; its pre-2023 half correlates
   **0.18–0.40** with real broker quotes. Rebuild from M15 via `mtf_frames.aggregate`, or use the
   single-provenance `XAUUSDs_H4_switch.csv` / `XAUUSD_H4_fp.csv`.
5. **Nulls on GROSS, pricing the SIGNED maximum.** A deterministic cost offset inside a zero-mean
   null manufactures significance with sample size (§3bx-D reported `p 0.0000` on a dead study).
6. **Report EV/fire and the mean offset** beside any classifier claim (#8). A positive mean offset
   means a confirmation detector, which is untradeable regardless of precision.
7. **Benchmark against buy-and-hold** (#2), never zero. Gold's drift is the thing to beat:
   `excess(LONG) + excess(SHORT) = −2·base`.
8. **Do not reuse `v5_repr_ceiling.py::walk_forward_ll`** — it selects the model class on the test
   fold (`:219-232`), worth ≈ +0.006 bits, biased toward higher-capacity arms.

## 6. Scoring — both bars, per the user's choice

| bar | criterion |
|---|---|
| **1 — CONTROL #6** | DSR ≥ 0.90 **and** PBO < 0.30 **and** survives walk-forward re-selection. |
| **2 — staged** | matched-vol paired-t ≥ +2.0 vs the deployed book, positive in **both** halves, then demo for 1–3 months. |

Every survivor is scored against both and the decision is taken per candidate. For reference the
**deployed XAU+BTC book itself fails bar 1** (DSR 0.000 / PBO 0.381) and sits on demo for that reason.

## 7. Expected outcome, recorded so nothing is rounded up

**Tier A closes 5 of 6.** A2 is the likeliest survivor because its label is the economics; A4 is the
genuine unknown; A1 is the most interesting and the most likely to produce a real information gain
that no trade structure can collect. If that is the result, **the document's value is the bound** —
the same way Gate A's best-of-K table was worth more than the 761 patterns it bounded.

_Last updated 2026-10-29._
