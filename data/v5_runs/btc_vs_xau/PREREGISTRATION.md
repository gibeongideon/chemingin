# PRE-REGISTRATION — BTC trend vs the XAU champion, head to head

Written before the comparison is run. §3bp found BTC trend to be a real but unproven lead
(walk-forward 1.401, DSR 0.743 against a 0.90 bar) and recorded that the walk-forward speed picks
wander the entire grid — 1/3, 1/48, 1/48, 1/12, 1/6, 1/24, 1/24, 1/24 — so **no speed is stably
identifiable**. §3bp's own stated next step was a two-cell test, not a wider sweep, because §3ak
proved searching harder raises the bar faster than the statistic.

## The declared cells — TWO. No parameter search.

| cell | spec |
|---|---|
| **A — XAU champion (incumbent)** | `champion_signal` on H4 rebuilt from M15, trail 3.0 / sl 3.0 / activation 1.0, risk 0.5%. Live on FTMO magic 360592. |
| **B — BTC trend** | `champion_recipe(close, 1/6, 1.5)` on BTC D1 — **the book's existing sleeve speed**, chosen BEFORE seeing this comparison and *not* the full-sample argmax (1/12). Using 1/12 would import the +0.318 selection bias §3bp measured. |

Combination (A+B) is reported as a third row for information, but the deploy decision is A vs B.

## Method rules that bind

- **Common business-day index before annualising** (rule 1.3.6). BTC quotes ~365 bars/yr against
  gold's 252; a common `sqrt(252)` across mismatched day counts inverted a book ranking once.
- **Matched vol before any paired-t** (rule 1.3.1).
- **Benchmark each against its OWN buy-and-hold** (CONTROL #2), never against zero and never
  against the other asset's benchmark.
- **Split halves** (rule 1.3.8) — a pooled t of +2.25 with 7/9 positive years was still an artifact.
- **DSR ≥ 0.90 and PBO < 0.30** (CONTROL #6), reported for both.
- Costs floored at live quotes: XAU $0.672 round trip (the corrected figure, Gate A3), BTC $1
  expressed as a **per-bar bp of the live price** (it was 22bp in 2014, 0.16bp now).

## The deploy gate — declared now

> **B replaces A only if, on the common index at matched vol, B beats A on paired-t at ≥ +2.0,
> AND B beats its own buy-and-hold in BOTH halves, AND B's FTMO pass rate exceeds A's.**
> Raw Sharpe alone does not decide it: `book-shootout` established that **Sharpe does not predict
> pass rate — the dial-scaled LEFT TAIL against the daily limit does.**

If B wins on Sharpe but not on pass rate, the honest action is to keep A and record why.

## Honest prior

A (gold) has never beaten its own buy-and-hold; it sells drawdown control. B does beat its own
buy-and-hold in both halves, which is the stronger property. So B is expected to win on Sharpe.
The open question is the **left tail against FTMO's 5% daily limit**, where bitcoin's volatility
is the obvious hazard and where Sharpe is known not to be predictive.
