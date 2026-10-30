# OCTOBER — 21 approaches, ranked, gated, with honest priors

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
| **A4** ✅**CLOSED §3cf** | **Volume-derived supply/demand** — volume profile, POC, value area, HVN/LVN. | **T** | **Zero hits for volume profile / POC / value area / footprint in 6,232 lines or any `src/` file.** Every zone the 12-concept ICT sweep tested was **swing-derived**. This is the one untouched branch of the supply-demand family, and the cheapest proxy for the order flow §3r names as *the* data type that could change the answer. | Oracle ceiling first (#5). Then EV/fire vs a flat ATR-offset control — the control that beat every ICT zone in §3bc. | **Genuinely unknown. No prior in this repo**, which is itself the reason to run it. |
| **A5** ✅**CLOSED §3cg** | **Regression on forward MAE** (max adverse excursion). | **T** | Only **one** regression target has ever existed here (§3at volatility) and it closed on a **negative oracle**, not model failure. MAE feeds **stop placement** — the conversion step `BEST_FINDINGS` #5 names as the actual binding constraint. | Oracle-MAE first: if perfect MAE foresight does not improve the book, close immediately (the §3at pattern). Then R² *and* matched-vol economics. | Expect real R² and negative economics. Running the oracle first makes that cheap to establish. |
| **A6** 🟡**LEAD §3ch–§3cl** | **Re-price every intraday closure at the live 0.504bp.** | — | §3bx revised the cost wall (M15 breakeven **0.6363 → 0.5343**) and then re-priced only **two** of ~15 intraday closures. **Every other one was decided at 1.5–2.34bp.** §3bx itself says the revision *"widens what is worth testing for non-trend structures."* | Any cell that flips sign must then clear the fixed null harness (GROSS, signed maximum) before it counts as reopened. | Expect 1–2 of ~15 to flip sign and none to clear a gate. Cheap and mechanical. |
| **A7** | **TradingView Technical Ratings, replicated — and its EFFECTIVE number of opinions measured.** Build all 26 components exactly (11 oscillators: RSI14, Stoch %K, CCI20, ADX14, Awesome, Momentum10, MACD, StochRSI, Williams %R, Bull-Bear Power, Ultimate; 15 MAs: EMA/SMA at 10/20/30/50/100/200, Ichimoku base, VWMA20, HMA9), reproduce the Buy/Sell/Neutral vote, then sweep the consensus threshold for the "sweet spot". | **T (strength, not direction)** | Requested by the user from the live XAUUSD page. The repo has tested most components **individually** (RSI, Stochastic, Bollinger, ADX, and the champion *is* an EWMAC+breakout blend) and A3 built 23 signals, but **the vote-counting AGGREGATOR itself has never been tested**, and nor has consensus STRENGTH as distinct from consensus direction. ⚠️ Pure direction is #8-retired, so the primary deliverable is the **multiplicity bound**, not an edge. | Report (a) the **effective rank** of the 26-component correlation matrix, (b) correlation of the vote count with a single price-vs-MA indicator and with the champion forecast, (c) EV/fire by consensus strength vs **holding** and vs the champion. | **The vote is not 26 opinions.** The user's own screenshot shows EMA/SMA 10/20/30/50/100/200 all "Sell" because price (4,023) sits below every one (4,068→4,502) — that is **one fact counted twelve times**. Expect effective rank 2-4, vote-vs-champion \|corr\| > 0.8, and no EV past the champion. |

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

## 4b. TIER D — the FILL-CAPTURE family, opened by §3ci (2026-10-30)

§3ci split A6's +10.161bp into **+2.807bp of free drift, +4.729bp of level-fill capture and ~+2.6bp
of genuine conditional excess**. The middle term is a new *mechanism*, not a new signal: a resting
order at a price known in advance owns the distance price travels past that level before any bar
closes. That is execution geometry, and it is the only thing in this round that is not a forecast.
Everything below is that same mechanism aimed somewhere else — **and D21 must run before any of the
others, because it can kill the whole family in one afternoon.**

| # | approach | route | why it is open — and what would kill it |
|---|---|---|---|
| **D21** ✅**RUN §3cj** | **The PLACEBO LEVEL control.** Re-run the prior-high break against a resting buy-stop at an *arbitrary* level the same distance above the day's open, same bars-left, same drift adjustment. | — | **The +4.729bp fill advantage is measured CONDITIONAL ON THE LEVEL BEING CROSSED, and any level that gets crossed has price travelling past it by construction.** §3ci controlled for *unconditional* drift, not for *conditional-on-crossing* drift. This is the shape that has already killed five findings here — `window-extremes-are-tautology`, `extreme-timing-patterns-are-drift`, `measure-the-skipped-sample`, `waiting-for-a-dip-selects-losers`, `volume-zones-premise-backwards`. If a placebo level captures the same +4.729bp, then "prior-day high" is decoration and A6 is intraday momentum with a path selection on it. **Cost: one script. Prize: it decides whether D22–D26 are worth writing.** |
| **D22** | **Resting BUY LIMIT at the prior-day LOW** (not the sell-stop, which is the dead leg at −1.545bp). | — | The short *stop* leg is closed; a long *limit* at the same level has never been tested. §3bc's killer control was **a flat 0.5-ATR limit that beat every ICT zone on gain-per-fill AND filled 2.4× as often** — so the honest form of this is "daily level vs flat offset", and the flat offset is the favourite. ⚠️ `waiting-for-a-dip-selects-losers` (t −5.23) says limit fills are adversely selected; this is the test of whether a *pre-known* level escapes that. |
| **D23** | **Prior-WEEK high, resting buy-stop, exit Friday close.** | P→weaker | Same geometry on a slower clock: the gap past a weekly level should be larger, because it takes more momentum to reach one. **But the sample collapses from ~1,446 days to ~290 weeks**, which runs straight into CONTROL #7 — so the pre-registered question is whether the per-event fill advantage scales *faster* than √n shrinks the t-statistic. Also must respect `xau-weekly-cycle-disproven` (exit Thursday, not Friday). |
| **D24** | **Stop-entry overlay on the CHAMPION.** Convert each position *increase* from a market order at the H4 close into a resting buy-stop a fixed ATR fraction above it; unfilled increases are simply not taken. | **T** (execution, not signal) | The champion rebalances at market. If level-fill capture is a general property rather than a property of prior-day highs, it should show up here — and this is the one place in the book where it would be worth real money rather than 1,446 measurement trades. It is also a **confirmation filter** in §3as's sense ("a detector with a positive mean offset is a CONFIRMATION detector"), and the pullback mirror of it is already known to lose. ⚠️ Gate at matched vol per `xau-champion-five-lifts-fail`, and expect turnover to fall, which must not be credited as edge. |
| **D25** | **The opening-range high as the resting level** — B15's ORB, re-asked as a fill question. | — | §3bx-D destroyed §3be's statistical basis, so the ORB is **unmeasured rather than closed**. It was always tested as a *signal*; the level-fill vs close-fill split was never applied to it, and the ORB high is the one other level in this repo that is known in advance and sits above price. Merge with B15 — do not run them separately. |
| **D26** | **Round numbers and the session-open price as resting levels.** | — | Lowest prior in the family. §3bi #6 filed round-number stop runs at −3.03bp, still −1.45bp re-netted at the live quote — but that tested them as a *signal*. Only worth touching if D21 survives and D22–D24 show the mechanism generalises past a single level type. |

**D21 RAN, AND IT RE-DESCRIBES THE FAMILY RATHER THAN KILLING IT (§3cj).** An arbitrary level at a
matched ATR distance above the open earns **~60% of the excess over drift and ~71% of the net**;
the prior-day high's advantage over it sits at **p 0.05–0.09**, outside this file's bar. The fill
advantage, though, IS specific to it at **p 0.0000 in both frames** — breaks of yesterday's high
really are more violent than crossings of an arbitrary level. So D22–D26 are no longer "is this
level special?" They are **"where should a resting order sit?"** — a one-parameter family whose
multiplicity has to be priced before any cell in it is believed. The fixed-ATR grid in §3cj is the
first row of that pricing: five cells, best at 0.25 ATR, t +1.87.

**The one operational fact all six depend on** is now being measured live: `scripts/v5_fill_probe.py`
(magic 360599, FundingPips-Trial demo, bridge 18816, one minimum-lot resting buy-stop per day at the
prior-day high) records the requested level against the actual fill price. **§3ch's slippage ladder
says the strategy survives ~4bp and dies at ~8bp, and 4bp is almost exactly the fill advantage** —
so the probe is not a nice-to-have, it is the measurement that sets the whole family's ceiling.

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

**Tier A closes 6 of 7.** A2 is the likeliest survivor because its label is the economics; A4 is the
genuine unknown; A1 is the most interesting and the most likely to produce a real information gain
that no trade structure can collect. If that is the result, **the document's value is the bound** —
the same way Gate A's best-of-K table was worth more than the 761 patterns it bounded.

_Last updated 2026-10-29._
