# Exploration queue — 38 untried approaches, autonomous run 2026-09-10

Built by enumerating dimensions this repo has never CROSSED, each checked against the tried
list in `V5_FINDINGS.md` / `BEST_FINDINGS.MD`. Web research first (conformal prediction,
selective classification, 2026 XAUUSD ML surveys) confirmed the public field offers little the
repo has not already refuted — its two main offerings, confidence gating and intermarket
correlations, are both in the failed pile (§3r). So the value is in the crossings, not the
literature.

**Honest prior.** ~130 pre-registered trade structures have failed here. The signal side is
documented as exhausted. This queue exists because the user asked for a long list of genuinely
untried ideas, tested properly — not because I expect a winner. **Results are reported
whichever way they come out; nothing gets rounded up.**

**The binding constraint, stated up front.** MANDATORY CONTROL #7: at these horizons XAU H4
yields ~250 decided non-overlapping events, SE(AUC) = 0.0367, and the best-of-8 pure-noise AUC
lift is +0.053. **Any approach that only nudges AUC is unfalsifiable here.** So the queue is
ordered to put POWER-INCREASING and MECHANISM-CHANGING ideas first, and pure feature additions
last.

---

## Tier 1 — attack the power problem (the identified binding constraint)

| # | approach | why untried / why it might matter |
|---|---|---|
| 1 | **Pooled cross-instrument panel training.** Train the bottom detector on all 51 D1 series 2008+, predict gold. | ~47,000 independent windows against gold's ~250. SE(AUC) 0.037 → ~0.003. §3ao designed this as "Phase 1c" and never ran it. The single highest-value item in the queue. |
| 2 | **Transfer then fine-tune**: pre-train on the panel, fine-tune on gold with a small learning rate. | Standard in every other ML field, never tried here. |
| 3 | **Sample weighting by average uniqueness** (López de Prado ch.4) to correct for overlapping labels. | The repo drops overlapping events entirely; weighting keeps them at honest weight. Never implemented. |
| 4 | **Pool the four book sleeves** (XAU/BTC/NDX/BRENT) into one detector with an instrument dummy. | 4x the events; §3ae only ever pooled *returns*, never *labels*. |

## Tier 2 — combine two known-real information sources (never crossed)

| # | approach | why |
|---|---|---|
| 5 | **M15 intrabar path features added to the ZIGZAG bottom label.** | §3ao proved the intrabar path carries real information (+0.0142 bits, above p99) on a *first-touch* label. It has never been given to the *bottom-pivot* label, which is a different target with far better classification numbers. Two proven-real inputs, never combined. |
| 6 | **REGIME block added to the zigzag detector.** | The regime block is the repo's one accuracy edge (+2.12pp); the zigzag detector does not use it. |
| 7 | **Champion forecast as a feature** in the bottom detector. | Alignment information the detector has never seen. |
| 8 | **Prior-day / prior-week levels** (`smc_signals.prev_day_levels`, implemented, never used on this label). | Distance-to-level is the one ICT primitive that is a continuous feature rather than a discrete pattern. |

## Tier 3 — time and session structure (user's explicit suggestion)

| # | approach | why this is NOT the failed `hour` feature |
|---|---|---|
| 9 | **London–NY overlap flag** (12:00–16:00 UTC), the highest-liquidity window. | §3p dropped raw `hour` because it found a within-day artifact off ragged Sunday bars. A *session* flag is a different, economically-motivated variable. §3ac's session test was a filter on the champion, not a feature in a detector. |
| 10 | **Cyclical time encoding** — sin/cos of hour-of-day and day-of-week. | Removes the discontinuity that made raw `hour` an artifact magnet. |
| 11 | **Time since session open / to session close**, continuous. | |
| 12 | **Day-of-month and month-end flags** (rebalancing and fixing flows). | Never tested at any horizon. |
| 13 | **Bars since the last confirmed pivot** (structural clock rather than wall clock). | |

## Tier 4 — label engineering (the target, not the features)

| # | approach | why |
|---|---|---|
| 14 | **ZigZag order/theta sweep** (order 3/5/8 x theta 1.0/1.5/2.0). | Only order 5 / theta 1.5 was ever used. The label's own definition has never been varied. |
| 15 | **Tolerance sweep** (±1, ±3, ±5 bars). | ±3 only. Note the lift caveat: widening tolerance inflates absolute precision by raising the base rate. |
| 16 | **Economically-filtered pivot**: a bottom only counts if the next N bars rise ≥ X. | Ties the label to a tradeable move — the repo's labels are geometric, not economic. |
| 17 | **Triple-barrier label** on the pivot (`meta_labels.py`, implemented, unused). | |
| 18 | **Quantile label**: bottom decile of forward 24h return. | |

## Tier 5 — model and ensembling (user's explicit suggestion)

| # | approach | why |
|---|---|---|
| 19 | **Seed-bagged ensemble** — average 15 seeds. | Variance reduction. RL taught us seed variance here is enormous (−0.415 to +0.068). Never applied to the detector. |
| 20 | **Label-ensemble** — average predictions from the differently-labelled models of Tier 4. | Averaging over label *definition* uncertainty, not parameter uncertainty. Genuinely novel here. |
| 21 | **Timeframe-ensemble** — M30 + H1 + H4 detectors averaged. | |
| 22 | **Probability calibration** (isotonic on a purged tail). | `turn_prob` calibrates; the zigzag detector does not. |
| 23 | **Fold-local autoencoder latents** (`latent_encoder.py`, six modes, never applied to XAU). | The user's "encoder" idea. Must be fold-local — an offline-fitted encoder is the known leak. |
| 24 | **Small sequence model** (LSTM/transformer) on the raw H1 window for the bottom label. | `e2e_lstm.py` exists, never on XAU. |
| 25 | **Monotonic feature constraints** where the sign is known a priori. | Regularisation by economics; never tried. |
| 26 | **Conformal / selective prediction with an explicit coverage-risk curve.** | Distinct from §3r's raw-probability confidence gate: an adaptive, coverage-guaranteed threshold. Low prior (gating failed) but a different mechanism. |
| 27 | **Two-level uncertainty** — separate model uncertainty from data uncertainty, abstain on the former. | |

## Tier 6 — trade structure (where everything has always died)

| # | approach | why |
|---|---|---|
| 28 | **Structural exit** — hold to the next confirmed ZigZag pivot instead of a fixed TP. | Every prior exit grid used fixed %, R-multiples or time caps. A structural exit matches the label's own logic and has never been tested. |
| 29 | **ATR-scaled TP/SL** instead of fixed percentages. | Fixed % ignores the vol regime; the grid was 0.15–0.6% flat. |
| 30 | **Limit entry on a pullback** rather than market entry. | Improves the entry price on a mean-reverting label; never tested. |
| 31 | **Champion-alignment gate** — trade the bottom only when the champion is already long. | |
| 32 | **Probability-proportional sizing** (continuous), not a gate. | §3r gated; it never *sized*. |
| 33 | **Trailing stop after +0.3%.** | |

## Tier 7 — reframe the target

| # | approach | why |
|---|---|---|
| 34 | **Predict volatility, not direction**, and use it for sizing. | Vol is an order of magnitude more predictable than direction. The repo's one strong overlay (range-based vol estimators, t +2.25) lived here before being killed as a 2022+ artifact. |
| 35 | **Champ-meta at H1** — predict the sign of the champion's next-k-bar P&L. | Done at H4 (AUC 0.593, the best ever here); never at H1. |
| 36 | **Predict time-to-target** rather than direction. | |
| 37 | **Predict the drawdown-first probability** (already the first-touch label) on the *pivot* subset only. | Conditioning one real signal on the other. |
| 38 | **Cross-sectional bottom rank** across the metals complex rather than absolute. | Removes the common factor; §3af showed cross-sectional signals need synchronous closes, which metals satisfy. |

---

## Execution protocol

Non-negotiable, from `V5_FINDINGS.md` MANDATORY CONTROLS 1–7:

1. **Lookahead probe** every new feature function at `n=3500, warmup_buffer=2200`.
2. **Walk-forward from day one** — never a full-sample pick.
3. **Report against the right benchmark** — buy-and-hold AND the champion, matched vol.
4. **Standard errors on everything.** An AUC move below 0.037 is not a result.
5. **Max-statistic null** whenever more than a couple of variants are compared.
6. **Synthetic-noise control** on anything that looks like a winner. This caught a false positive
   tonight that would have passed my own pre-registered bar.
7. **Split-sample halves.** Three things died on this tonight alone.

**Disqualification is the expected outcome for each item.** The log below records every result,
including the boring ones, so a future session does not re-run them.

---

## Results log

| # | approach | result | verdict |
|---|---|---|---|
| **1** | Pooled cross-instrument panel training | gold OOS AUC **0.6876 -> 0.7452** (+0.0576, SE 0.0092 = 5.9 sigma), **15/15** years, sign-test p 0.00003, shuffled-label control clean at 0.50, split halves flat (+0.045 early / +0.047 late) | **REAL AND ROBUST — the largest detector lift in repo history.** But see below: it loses money. |
| **2** | Transfer (train without gold at all) | **+0.0570** — indistinguishable from POOLED | **REAL. The bottom-pivot relationship is UNIVERSAL across assets**, not gold-specific. New result. |
| **1-2 economics** | 3 structures, ~14 variants | panel dSharpe **-0.702** (t -3.17, 3/15); XS gross 1.2 SE from zero; gold overlay +0.066 (t 1.81). Panel ORACLE +0.564 (t +3.10) — the label IS valuable | **DISQUALIFIED on money.** A 5.9-sigma AUC gain -> negative P&L. |
| **mechanism** | why the lift does not convert | detector mean true-positive offset **+0.967**; fires 28% one bar AFTER the low, 5% one bar before; captures +1.24% of the +2.60% available; 4,484 false positives at -2.01% erase it; **EV/fire +0.05% vs +0.15% for holding, t -3.64** | **CONFIRMATION detector, not anticipation.** Became MANDATORY CONTROL #8. |
| **14/16** | offset-aware labels (ANTICIPATE / PIVOT-ONLY / RET-WEIGHT) | mechanically successful — mean offset **-0.489**, capture 51%->62%, AUC 0.7794 (PIVOT-ONLY 0.8839) — and **economically worse** (EV/fire -0.0034 / -0.0060 / -0.0021) | **DISQUALIFIED, and it closes the family:** payoff sits on one unknowable bar; late = move gone, early = falling knives (FP -3.43% vs -2.37%). |
| **38** | cross-sectional bottom rank | gross Sharpe -0.312 / inverted +0.312 vs SE 0.269; every net variant negative; rank corr with 60d XS momentum only -0.260 (so not merely inverted momentum); XS ORACLE gross +5.076 | **DISQUALIFIED.** |
| **5-8, 9-13, 19-27** | Tier 2 features, Tier 3 time/session features, Tier 5 models & ensembling | **retired without runs by MANDATORY CONTROL #8** — every one of them improves the *ranking* of the same 7-bar bottom label, the axis just proven not to convert (+0.0576 AUC -> -0.702 Sharpe) | **CLOSED (~20 of 38 items).** |
| **34** | pooled ML **volatility** forecast for sizing | **R^2 +0.2299** incremental over trailing vol (TRANSFER +0.2320) — real, large, transferable. Economics at matched vol: ML-POOLED **-0.011** (t -0.20), and **ORACLE-VOL -0.146** | **DISQUALIFIED AT THE CEILING.** Perfect vol foresight makes the book worse; no forecast can beat knowing it exactly. §3at |
| **TOPS** (not queued) | pooled **TOP** detector — the side the repo's own oracle says the money is on, never once used | **ORACLE +1.252 panel / +1.491 gold, t +6.7, 15/15** — the largest oracle prize in repo history (6x the gate). Best honest detector **+0.090 at t +0.84** across 20 fire-rate-matched cells | **DISQUALIFIED.** 14x oracle-to-achievable gap. §3au |
| **TOPS-early** | offset-aware TOP labels (ANT / EARLY) | offset flipped +1.323 -> -0.300 / -1.26, so label engineering WORKS. Economics got monotonically WORSE: POOLED-EARLY **-0.751 at t -3.04** | **REFUTED MY OWN MECHANISM PREDICTION.** Early tops = strong uptrends; trimming them forfeits the drift that IS the edge. |
| **35** | champ-meta at D1, pooled | ORACLE **+2.534** (t +17.16, 15/15). Appeared to give panel **+0.261, t +8.54, 14/15, block-p 0.0000**, all TIMING (XS-ONLY +0.000), 38/50 legs positive p 0.000153, shuffle/vol-tilt/trailing-Sharpe controls all clean. **RETRACTED** — the real 4-sleeve book gives **-0.009 (t -0.24)** and all 18 t>+2 legs are FX pairs leaking the next bar | **DISQUALIFIED + NEW DATA HAZARD.** 18 FX D1 files are forward-stamped; `clpos`/`upwick`/`lowick` carry ret[t+1] at corr 0.42-0.53. Screen with `intrabar_leak()` > 0.15. §3av |
| 36-37 | time-to-target, drawdown-first on pivots | not run | remain open for a future session |

### Errors made and corrected during this run

1. **`buys` is an INDEX ARRAY, not a boolean mask.** `np.flatnonzero(np.asarray(buys).astype(bool))`
   maps every nonzero index to True and returns `0,1,2,...,n_pivots`, silently measuring the
   first ~200 bars of each series instead of the pivots. It produced a confident, wrong claim
   that a ZigZag bottom is a SELL signal (h=5 mean -0.130, t -10.98). The true event path:
   **+2.77% mean 5-day return from a pivot, 92.6% positive.** Assert `piv.max() < len(series)`.
2. **Conditioning on |forward move| selects on the dependent variable**, which inflated a
   label-1 forward return to +1.19 vol units. The top magnitude quintile is *defined* by the
   outcome being large.
3. **A shuffled-LABEL control cannot detect FEATURE lookahead.** It came back clean at 0.50 in
   §3as and I treated that as evidence of no leakage. Shuffling the label destroys every
   relationship, leaky or not — it tests label leakage ONLY.
4. **Truncation-based `probe_lookahead` cannot detect DATA misalignment.** The feature code is
   causal; the FX *bars* are forward-stamped. Truncate-and-recompute reproduces the same values,
   so the probe passes. The direct test — correlate every feature column against the NEXT bar's
   return, per instrument — takes seconds and is the only thing that caught it.
5. **Panel comparisons must assert a SHARED evaluation window.** SPX has history to 1927 and
   USDJPY to 1996; letting the baseline trade years the other arms could not moved ORACLE-VOL
   from +0.367 to -0.146 and flipped a verdict.
6. **A ratio proxy is not the harm.** `forward_stamp_ratio` misses USDJPY (0.632 < 0.649) which
   still leaks at 0.317. Screen on `intrabar_leak()` = |corr(clpos[t], ret[t+1])| instead.
