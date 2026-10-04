# BEST FINDINGS — the strongest results this project has produced

The top five, ranked by **strength of evidence**, not by whether they made money. Three of the
five are not tradeable and are here because the measurement itself is solid — that distinction
is the single most important thing this project has learned, so it is recorded rather than
hidden.

Every figure below is traceable to a `V5_FINDINGS.md` section and to a script. Where a number
was later corrected, the corrected value is shown and the correction is named.

---

## 1. Swap-free gold — the only change that ever made the deployed book better

**TRADEABLE. The largest real improvement in the project's history.**

Maven carries `GoldEternal`: `swap_mode = DISABLED`, i.e. **zero overnight financing**, spread
$0.55 (1.23bp), and it tracks XAUUSD at **0.998 daily-return correlation**.

| | XAUUSD | GoldEternal |
|---|---|---|
| overnight financing | **−3.70%/yr** | **0.00%** |
| spread | $0.42 (0.94bp) | $0.55 (1.23bp) |
| **XAU sleeve net Sharpe** | **+0.862** | **+1.065** |

**+0.203 Sharpe for +0.33bp of extra spread.** The free-lunch objection was tested, not waved
away: the log basis GoldEternal/XAUUSD regressed on time drifts **+0.17%/yr** (H1, 2,711 bars)
inside a ~1% band — nothing resembling the −3.70%/yr it avoids.

**Why it ranks first:** the gain is *contractual*, not estimated. A saving of 3.70%/yr is
3.70%/yr whether the strategy works or not, so it needs no out-of-sample validation. Set
against it: **every signal-side improvement ever attempted here has been worth 0.000.**

Caveats: 167 days of instrument history (launched ~2026-03-13) — enough to exclude a large
embedded drift, not a small one, so re-measure monthly. Only **5 of Maven's 68 symbols** are
swap-free and none of them is an index, crypto or energy, so the win is confined to gold.
*§3ag, §3ah · `scripts/v5_maven_carry_book.py`*

---

## 2. Diversification, and matching the book to the binding constraint

**TRADEABLE. The second real win, and the only place advanced mathematics lost cleanly.**

Net of measured broker carry, business-day aligned:

| book | net Sharpe | maxDD |
|---|---|---|
| single-sleeve gold (XAUUSD) | +0.862 | −22.8% |
| single-sleeve gold (GoldEternal) | +1.065 | −19.7% |
| core-4 equal weight | +1.096 | −10.5% |
| **core-4, gold at 50%** | **+1.212** | −11.5% |

Sleeve correlations sit between **+0.04 and +0.14** — the diversification is real. And the
**binding-constraint rule**: the best book differs by *which rule fails you*. FTMO at a 9% dial
is killed by the 5% **daily** limit, so widening 4→9 sleeves takes pass rate **88.6% → 94.1%**
and daily breaches **9.3% → 0.0%**. Where no daily limit binds, the narrower higher-Sharpe book
wins instead.

**What lost:** HRP, RMT/Marchenko-Pastur covariance cleaning, min-CVaR and inverse-vol — **0 of
20 configurations beat equal weight**. The classic 1/N result: estimation error swamps the
optimisation gain.

Honest caveat: the 50% gold tilt has a paired t of only **+0.75**, so it is "real but unproven";
what supports it is that the driver is contractual carry and the response is a **plateau**
(40% → +1.200, 50% → +1.212, 60% → +1.196), not a spike.
*§3ae, §3af, §3ah · `scripts/v5_portfolio_construction.py`*

---

## 3. The "zigzag" bottom detector — 80% precision, and untradeable

**NOT TRADEABLE. The highest raw accuracy number the project has produced.**

A walk-forward HistGradientBoosting detector on causal ZigZag swing lows (±3 bars, amplitude
≥ 1.5×ATR):

| operating point | precision | random base | lift |
|---|---|---|---|
| recall 22% | **71%** | 33.5% | **2.1×** |
| recall 10% | **75%** | 33.5% | 2.2× |
| recall 5% | **80%** | 33.5% | 2.4× |

PR-AUC **0.60**. Top features: 5-bar ATR-momentum, rank-in-window, z-score, RSI7. Hand-written
rules cap out around 62%, so the model is genuinely adding something.

**Why it is not tradeable — three independent reasons, and the third closes the family.**

1. **As a trade.** "Enter on the flag, exit at the first reasonable profit": **all 48 grid cells
   negative full-sample** (best −0.44), walk-forward **−0.76** against buy-and-hold's +0.79,
   paired t −3.72, **0 of 8 years better**, DSR 0.000. *(§3x)*
2. **The oracle ceiling says the target was worthless before any detector existed.** Perfect
   foresight of gold's bottoms is worth **dSharpe +0.017**, and **−0.675 standalone**. The
   champion is long-only and near-always-in, so it is *already long at bottoms*. The value was in
   TOPS (+0.26) and in trimming before bad windows (+0.70 to +4.04) — the opposite side from the
   one five studies chased. *(§3p)*
3. **The mechanism, measured.** The fire-weighted mean true-positive offset is **+0.967**: the
   detector fires 28% of the time one bar *after* the low and 5% one bar before. It is a
   **CONFIRMATION detector, not an anticipation detector**. So **EV per fire is +0.05% per 5 days
   against +0.15% for simply being long** — one third of what holding earns, paired t **−3.64**.
   *(§3as, which became MANDATORY CONTROL #8)*

**Label engineering does not rescue it, and that is what closes the family.** Offset-aware labels
worked *mechanically* (mean offset +0.967 → −0.489, capture 51% → 62%, AUC up to 0.8839) and got
**monotonically worse economically**. The same held on the top side: oracle trim **+1.252** panel
against a best honest detector of **+0.090** — a **14× oracle-to-achievable gap**, the largest in
this repo. Fire early and you buy falling knives (FP −3.43%); fire late and the move is gone.
*"A pivot is only identifiable once it has resolved, and the bars where it has not yet resolved
are, by construction, the bars that look like continuation."* *(§3as, §3au)*

**It was traded live, and it lost.** Five orders placed 2026-09-10..16 took the FTMO demo from
**$100,000.00 to $99,564.82**. §3ax gives an independent mechanical explanation: gold's mean
daily range went from **1.34% of price in 2015 to 2.73% in 2026**, which makes the harness's
fixed +0.60%/−0.50% bracket **0.23 and 0.18 of a single day's travel**, with 96% of sessions
ranging more than TP+SL combined. Both barriers sit inside one session's noise. Trading was
disabled 2026-09-16; the harness still runs signal-only.

**Read the lift, not the headline.** Widening the tolerance window inflates absolute precision
by raising the random base rate; the 2.1–2.4× lift is the honest statistic.
*§3, §3p, §3x, §3as, §3au, §3aw, §3ax · `scripts/v5_xau_turning_ml.py`, `v5_xau_turning_scalp.py`,
`v5_pooled_bottom_detector.py`*

---

## 4. The REGIME accuracy edge — statistically solid, monetisable five different ways, and none of them worked

**NOT TRADEABLE. The most statistically careful positive in the project.**

Predicting the sign of the next 6 H4 bars (~1 trading day) from a regime block — Hurst(100),
ADX(14), Choppiness(14), ATR ratio, and two daily features:

| | value |
|---|---|
| accuracy | **51.97%** |
| **persistence baseline** | **49.85%** (not 50%) |
| edge | **+2.12pp** |
| block-bootstrap 90% CI | **[+0.51pp, +3.69pp]** |
| P(edge ≤ 0) | **1.5%** |
| years positive | **8 of 9** |

Beating *persistence* rather than a coin flip is the part most published work skips.

**Five trade structures, all failed:** standalone (SR −0.44, t −3.3); champion overlay (t +0.4 /
−1.7); confidence gating (t −2.5 to −3.7 at every threshold, even where accuracy reached 55.6%);
five forecast horizons from 1 to 16 days; and an 80-cell SL/TP bracket grid where **79 of 80
cells were Sharpe-negative and DSR was 0.0000**.

Also tested and adding nothing over this block: multi-scale Hurst, a causal 3-state HMM, Haar
wavelet energy shares, causal zigzag state, and a 12-symbol intermarket panel with a synthetic
DXY. Its own recorded conclusion: *"the only thing that could change the answer is a materially
different data type."*
*§3r · `scripts/v5_xau_intermarket_accuracy.py`*

---

## 5. The representation CEILING — the result that closed a whole family of ideas

**NOT TRADEABLE. Recorded 2026-09-09; its headline was RETRACTED 2026-09-19 and this entry is
the corrected version.**

> **CORRECTION.** This entry previously led with "M15 intrabar path — the first input to beat a
> proper information null (+0.0142 bits, above its alignment null's p99)". **§3ba withdrew that
> on 2026-09-19.** `data/XAUUSD_H4_long.csv` carries FTMO's quotes from 2023-06-28 while
> `XAUUSD_M15_long.csv` never did — median |Δ| **$13.61** by 2026 — so the SOURCE arm held
> privileged information about its own label's resolution path. On a provenance-clean single
> feed SOURCE **loses to BASE 8 times out of 8**. The M15-intrabar finding is withdrawn.
> What follows is what survived the feed rebuild.

Testing whether *representation* was the bottleneck: give a model the **raw 64-bar window**
(128 features — vol-normalised log returns and normalised ranges) and see whether it extracts
more than the incumbent 21-feature regime block, measured in **bits per independent event**.

| arm | features | incremental information | years better | AUC |
|---|---|---|---|---|
| BASE (regime + price block) | 21 | — | — | 0.592 |
| **CEILING (raw 64-bar window)** | **128** | **−0.0099 bits/event** | **3/9** | **0.557** |

**The raw window carries nothing over the incumbent block** — and it landed *below its synthetic
null's median* (+0.0090), with dI known to be positively biased toward the higher-capacity arm.

**Why that one measurement is worth more than most positives.** Every candidate representation is
a function of the same trailing window, so `I(Y; R(W)) ≤ I(Y; W)`. One ceiling therefore closes
**fracdiff, directional change / intrinsic time, entropy and complexity measures, recurrence and
visibility graphs, matrix profile, path signatures, SAX, and range/Renko bars — at a single
multiple-testing penalty**, without testing one of them individually. It also bounds LSTM,
transformer and autoencoder latents on H4 price alone.

**The lesson that generalises:** this project's binding constraint is not model capacity or
feature representation. It is the conversion step — forecast to net-of-cost position surviving a
regime split.

*Two methodological corrections also live here and are load-bearing:* `walk_forward_ll`
(`v5_repr_ceiling.py:219-232`) **selects the model class on the test fold**, worth ≈ +0.006 bits
and larger for the higher-capacity arm; and the single-draw synthetic control **fired a false
positive** at +0.0319 bits with 7/9 years on pure Gaussian noise — a result that would have
passed its own pre-registered bar.

*§3ao, §3ap, §3aq, corrected by §3ba · `scripts/v5_repr_ceiling.py`, `v5_m15_trim_overlay.py`,
`src/v5/mtf_frames.py`*

---

## Honourable mentions

- **SuperTrend on XAU — DSR 0.9948** on a 20-cell grid. The cleanest statistical validation of
  any signal here; *"this is what confidently real looks like."* Not deployed because it
  correlates 0.77–0.88 with the champion recipe — interchangeable, not additive. *(§3s)*
- **The oracle-ceiling control (#5)** — trade the label with perfect hindsight *before* building
  a detector. It retroactively explained why five bottom-detection studies were doomed
  (ceiling +0.017) and cost about five minutes. *(§3p)*
- **The false-positive wall** — a perfect bear-regime switch is worth +0.225 book dSharpe, but a
  **10% false-positive rate destroys 74% of it** while cutting recall from 100% to 50% costs
  0.015. Specificity, not coverage, is what conditional signals must buy. *(§3am)*
- **The maximum-statistic null** — searching 21,300 conditional triggers, the null distribution
  of the *maximum* |z| has a median of **9.34**. A z of 9 is unremarkable in a search that size,
  which is why "just test more rules" is provably a losing move. *(§3ak)*
- **The down-move census** — 175 peak-to-trough moves >2% on XAU H4 2015-2026, median −3.9% over
  5.3 trading days; the 84 that resolve within a week are worth **27.5%/yr gross** if perfectly
  captured. The opportunity is real; the detection is what fails. *(§3am)*

---

## What all five have in common

**Two of the five made money, and neither came from predicting the market.** Both came from
measuring what a position *costs* — a swap table and a correlation matrix. Costs are published,
contractual and deterministic; they need no out-of-sample validation.

**Three of the five are genuine, statistically careful positives that produce no P&L.** 80%
precision, a +2.12pp accuracy edge with a bootstrap CI clear of zero, and incremental
information above a p99 null — and between them, roughly 130 pre-registered trade structures
that all failed.

One more thing worth stating: **combination is not the missing ingredient.** Five independent
refutations now exist — ICT confluence 0/4, the 6-component stack 0/29, feature-block stacking
adding nothing over a single block, combining two books *hurting* (1.41 → 1.26), and the
book-level M15 trim 0/32. Stacking raises precision; on this asset precision has never been the
failing variable.

The lesson, stated once: **information is necessary and not sufficient.** On this instrument the
conversion step — from a correct forecast to a net-of-cost position that survives a regime split
— is where the money goes. Anyone repeating this work should budget most of their effort there,
and should measure what they pay before optimising what they predict.

---

*Companion documents: `V5_FINDINGS.md` (the full ledger, §-numbered), `RESEARCH-REGIMES.md`
(prop-firm-constrained vs unconstrained findings), `README.MD` (current state),
`BlogPOST-ZeroCarry.html` and `BlogPOST-SMC.html` (public write-ups).
Last updated 2026-10-04. Entries #3 and #5 were corrected on that date: #5's headline was
retracted by §3ba (the two-broker splice) and #3 gained its mechanism closure (§3as/§3au) and
its live result (§3ax).*
