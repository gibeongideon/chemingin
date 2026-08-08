# Intraday high-R:R research plan — few wins, positive expectancy

**Status:** proposed, 2026-08-07. Supersedes nothing; this is a *new framing* of a
question this repo has answered "no" to six times (§1 fade, §3 turning points, §3c,
§3l martingale, §3n LLM, §3o SMC-H1). Read "Why this is different" before assuming
it is a seventh repeat.

---

## 1. Why this framing is genuinely different (measured, not asserted)

Every prior intraday attempt traded a **near-1:1 payoff** and needed to beat ~50%.
The spread was ~10x the per-trade edge, so it died. This proposal inverts that.

For a stop of `k × ATR` and a reward:risk of `RR`:

```
breakeven win rate   p_be  = (1 + cost/R) / (1 + RR)
random-walk baseline p_rand = 1 / (1 + RR)
REQUIRED EDGE = p_be - p_rand = (cost/R) / (1 + RR)
```

**The required edge is governed by `cost/R`, and a wider stop shrinks it linearly.**
Measured on real data (3y median ATR, live FTMO round-trip cost incl. slippage):

| instrument / TF | cost / 1×ATR | cost/R @2×ATR | cost/R @4×ATR |
|---|---|---|---|
| EURUSD M15 | 21.1% | 10.6% | 5.3% |
| GBPUSD M15 | 19.3% | 9.7% | 4.8% |
| **XAUUSD M15** | **15.6%** | **7.8%** | **3.9%** |
| USDJPY M15 | 13.3% | 6.6% | 3.3% |
| **XAUUSD M30** | **10.7%** | **5.3%** | **2.7%** |
| XAUUSD H1 | 7.9% | 4.0% | 2.0% |

Required edge above a random entry:

| setup | random win% | needed win% | **edge needed** |
|---|---|---|---|
| XAU M15, 2×ATR stop, RR 3 | 25.0% | 26.9% | **1.9 pp** |
| XAU M30, 2×ATR stop, RR 3 | 25.0% | 26.3% | **1.3 pp** |
| XAU M30, 4×ATR stop, RR 4 | 20.0% | 20.5% | **0.5 pp** |

Compare: the M15 fade needed to overcome a spread **10x its edge**. Here we need
**1-2 percentage points**. That is a 20-40x easier cost hurdle, and it is the honest
reason to spend time on this.

## 2. The trap on the other side — measurability

A small required edge is also a **hard-to-prove** edge. Win-rate standard error is
`sqrt(p(1-p)/n)`; to resolve the edge at 2 SE:

| setup | edge needed | trades to prove it |
|---|---|---|
| RR 2, 2×ATR | ~3.3 pp | ~1,000 |
| **RR 3, 2×ATR** | **~1.9 pp** | **~1,900** |
| RR 4, 4×ATR | ~0.5 pp | **~25,600 (infeasible)** |

**Therefore the target zone is RR 2-3 with a 2×ATR stop.** RR 4+ with wide stops is
mathematically attractive and statistically unprovable — we would be trusting a number
we cannot distinguish from zero. Do not go there. Also note a wide stop + big target
means long holds: a 4×ATR/RR4 gold trade is a multi-day move, i.e. it stops being
intraday and converges on the H4 trend book we already run.

## 3. Non-negotiable controls (V5_FINDINGS MANDATORY CONTROLS, incl. #5)

1. **Oracle ceiling FIRST** (control #5, added 2026-07-29). Before building any entry
   model, trade the setup with perfect hindsight and quote the ceiling + the
   (precision, recall) contour. If a cheating entry cannot clear the bar, stop.
2. **EMPIRICAL random baseline, not `1/(1+RR)`.** The theoretical figure ignores
   intrabar path: a wick can hit the stop before the target on a path that ends
   favourably, and entry pays spread immediately. Measure the true random win rate by
   simulating 10,000 random entries per instrument/TF/stop/RR. **This is the real
   benchmark** and it will be below the theoretical one.
3. **Costs floored at live broker quotes** ($0.45 gold FTMO, +$0.10 slip/side) with a
   stress pass at 1.5x. Report cost/R alongside every result.
4. **Swap/financing included** — newly discovered 2026-08-07: gold longs pay ~6.6%/yr
   of notional. Negligible for a 4-hour trade, material if holds stretch to days.
   Charge it per calendar night held.
5. **Walk-forward, per-year** (the user's explicit ask), never a single split. Report
   win rate, expectancy in R, and net P&L **per calendar year** so a regime that
   carries the whole result is visible.
6. **Quote SE on the win rate** next to every win rate, and the trade count needed.
7. **No trailing-stop self-deception.** The prior candle run booked a 77.8% win rate
   with only +4.9% total return because `trail_activation=15p` converted 3R targets
   into small trailed exits. If we claim RR 3, we must MEASURE the realised R
   distribution, not the nominal one.

---

## 4. The plan — 10 approaches, cheapest and most decisive first

### Phase 0 — Feasibility (no model, ~1 day)

**A1. Oracle ceiling + empirical random baseline.**
Grid over instrument × TF (M5/M15/M30) × stop (1,1.5,2,3 ×ATR) × RR (1.5,2,3,4).
For each cell compute: (a) empirical random-entry win rate and expectancy,
(b) perfect-hindsight entry (enter only when the target *will* be hit first) — the
ceiling, (c) the win-rate contour that clears net-positive expectancy after live cost.
**Gate:** proceed only for cells where `needed_edge ≤ 3pp` AND `trades/yr ≥ 250`
(so ~1,900 trades accrue within a walk-forward-able window).
*Deliverable: the shortlist of (instrument, TF, stop, RR) cells worth modelling.*

**A2. Instrument screen for volatility efficiency.**
The user wants volatile instruments. The right metric is not volatility but
**ATR / spread** (already computed above for what we have locally). Extend to the
live FTMO universe — BTCUSD, ETHUSD, US100.cash, UKOIL.cash, XAUUSD — by pulling
M5/M15/M30 history over the bridge and computing cost/1×ATR. Crypto trades 24/7
(no weekend gap risk) and has high ATR/spread; indices have session structure.
*Gate: rank by cost/R; carry the top 3 forward. Single instrument first, per the ask.*

### Phase 1 — Entry (the core question)

**A3. Empirical random-entry map → where does the market itself help?**
Before any ML: measure win rate at fixed (stop, RR) conditioned on simple states —
session (Tokyo/London/NY/overlap), day of week, ATR percentile, distance from the
day's open/VWAP, and whether the H4 champion is long. Any state where the *random*
win rate is already 2-4pp above baseline is free edge with no model.
*This is the highest-value/lowest-cost step and it directly answers "know the entry".*

**A4. Volatility-breakout entries.** The classic intraday structure and the one most
compatible with high RR: opening-range breakout (session-anchored), ATR-expansion
breakout, prior-day high/low breaks. Long targets suit breakouts; mean-reversion
does not (it caps upside, which is why the fade needed 1:1 and died).

**A5. Higher-timeframe alignment filter.** Take intraday entries ONLY in the
direction of the H4 champion signal (a *proven* positive-edge object). This is the
one linkage to something already known to work, and only needs to add ~2pp.

**A6. Revisit the candle green/red model — honestly.**
`scripts/train_candle_model.py` already predicts a 1-bar M15 direction at **SL 10p /
TP 30p (1:3)** with CatBoost + session/MTF features. Recorded runs
(`data/v5_runs/*-candle-trail-v5*`) show Sharpe 2.15-4.47 and 57-78% win rates — but
they are **not trustworthy as-is**: (i) `trail_activation=15p` destroys the 3R payoff,
(ii) total returns are tiny (+0.8% to +4.9%), (iii) the `enc30` variants use the
encoder with a documented leakage incident (`pair_meta.json`: `"leaky": true,
wf_sharpe -37.351`), (iv) the `costx` variant collapses win rate 57%→46%.
**Refit fold-local, no encoder, no trail, fixed 3R exit, per-year reporting** — then
we learn whether the "high candle prediction accuracy" was real or an artifact.
Apply to XAU/volatile instruments, not just EURUSD/USDJPY.

**A7. Meta-labelling on a base entry (the §2-compliant ML use).**
Do NOT ask ML for direction. Take A4's breakout entries (a base with measurable
expectancy) and train a classifier on `side_barrier_meta_label` to answer "should I
take THIS one?". Sizing/gating on a base with positive edge is the only ML framing
this repo has not falsified. Purged walk-forward, isotonic calibration on a purged tail.

### Phase 2 — Dynamic management (the "be smarter" ask)

**A8. Early take-profit from the MFE distribution.**
Record max-favourable-excursion per trade. Test: if price reaches xR (0.5/1/1.5/2) and
then stalls (momentum decay, N bars without a new extreme, or an opposing candle
signal), does banking beat holding for the full 3R? Report the realised-R histogram
so the true payoff is visible, not the nominal one.

**A9. Early stop-loss from the MAE distribution.**
Symmetric question: given max-adverse-excursion, does cutting at 0.5R on a momentum
flip beat waiting for the full 1R stop? Cutting losers early **raises effective RR**
and is the single most direct lever on expectancy — but it also lowers win rate, so
it must be judged on expectancy in R, never on win rate.
Also test a **time stop** (exit after N bars unresolved) — intraday edges decay, and
an unresolved trade is consuming risk budget for nothing.

**A10. Breakeven / partial-scale rules.** Move stop to breakeven at +1R; scale out
half at 2R and run the rest. Popular and *usually* value-destroying — measure it
honestly against a fixed-3R control rather than assuming.

### Phase 3 — Risk management (the user's stated key)

Applied to whichever entry survives, and reported per year:
- **Fixed-fractional risk** per trade (0.25-1% of equity), sized from the actual stop
  distance — reuse `v5_xau_trend.run_trades` sizing and `size_lots`.
- **Daily loss limit** and **max concurrent positions**, aligned to prop-firm rules.
- **Vol-scaled sizing** — reuse `risk_scalar()` (vol-target × drawdown scaler).
- **Consecutive-loss behaviour**: at RR 3 a 25-30% win rate means **runs of 8-12
  losses are normal**. Quote the loss-streak distribution up front — this is the
  psychological and drawdown reality of the "few wins" design, and the reason
  martingale-style recovery must NOT be bolted on ([[martingale-predictor-bot]]).

### Phase 4 — Basket (only after a single instrument passes)

Combine the surviving per-instrument strategies with the proven equal-class-risk +
vol-target machinery (`v5_basket_challenge.py`). Intraday sleeves should be
*uncorrelated* to the H4 trend book — measure that correlation explicitly, since a
diversifier is worth more than a standalone (see the 1.67 four-asset result).

---

## 5. Verification / definition of success

- Per-year table: trades, win%, ±SE, expectancy in R, net P&L, max loss streak, maxDD.
- Beats the **empirical random-entry** baseline by ≥2 SE, in ≥7 of 9 years.
- Survives cost at 1.0x and 1.5x live spread.
- Realised-R histogram matches the claimed RR (no trail self-deception).
- Block-bootstrap CI on Sharpe; DSR across all grid cells tested.
- **Kill criterion:** if the oracle ceiling in A1 shows a perfect entry cannot clear
  net-positive expectancy at live cost, stop and record it as the 7th disproof.

## 6. Honest prior

The cost arithmetic is genuinely favourable for the first time — 1-2pp of required
edge instead of a 10x spread wall — and A3 (state-conditioned random-entry map) may
find some of that for free. But six prior disproofs in this exact domain, plus the
measurability wall at RR≥4, mean the base rate is low. The most likely honest outcome
is a small, real, *fragile* edge at RR 2-3 on the lowest-cost/R instrument, whose
value is as an **uncorrelated diversifier** to the H4 book rather than as a
standalone. That would still be worth having.
