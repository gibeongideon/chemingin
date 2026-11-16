# RANK.MD — everything that survived, in order

**What this is.** `V5_FINDINGS.md` is ~8,700 lines and roughly 130 pre-registered structures, the
large majority of which failed. This file is the short answer to "what do we actually know?",
ranked. If you read one file in this repo before starting work, read this one.

**Ranking criterion.** Not interestingness — *load-bearing-ness*. A finding ranks high if removing
it would change what is deployed, or would let a dead idea back through the door. Three tiers:

| tier | meaning |
|---|---|
| **Live** | it is running on a real account right now and earns or protects money |
| **Law** | a control that kills bad ideas fast; ignoring it has already produced a false positive here |
| **Bound** | a measured limit that closes a whole family of ideas, so they are never re-run |

**How research is run here: [`RULES.MD`](RULES.MD).** Last updated **2026-11-16**. `BEST_FINDINGS.MD` predates §3cj and is kept only for its longer
write-ups of items 8, 16 and 17.

---

## The table

| # | Finding | Measured | Tier | Where |
|---|---|---|---|---|
| 1 | **Financing is the real lever, not signals** — swap costs more than any signal earns | −3.79%/yr, −0.29 Sharpe; swap-free gold took the XAU sleeve 0.886 → **1.090**; BTC carry −40%/yr, so swap-free = **+0.477** Sharpe | **Live** | §3q, §3cr |
| 2 | **Diversify; do not improve the single asset** | single-XAU ceiling ~1.06; XAU+BTC **1.69**; XAU+BTC+NDX **1.71** | **Live** | §3bo, book 360595 |
| 3 | **Kill the shorts** | long-only 1.091 vs long/short 0.954; the short leg lost on **10 of 10** signals; hysteresis long-only +0.932 vs long/short +0.641 | **Live** | §3ab, §3bn, §3bz |
| 4 | **A ratcheting 3-ATR trail beats every alternative exit** | trail **1.091** vs fixed SL/TP 1.014 vs signal-flip 0.828 | **Live** | §3bn |
| 5 | **A high take-profit helps; a low one hurts** | TP at 4.3R t **+2.78**, better in both halves; TP at +$700 t −2.4 | **Live** | §3dr |
| 6 | **Intraday circuit breaker** — no new buys once the day is already down > 1 ATR | CME 0.21 → **0.56** (t +4.12, p 0.000), Switch 0.42 → 0.53 (p 0.005); passes in both sources | **Live** | §3dz |
| 7 | **Downside-vol SIZE DIAL — half size, do not skip** | **16/16** cells ≥ 95th pctile, improving both halves in both sources; maxDD −5.81% → **−2.80%** at better CAGR | **Live** | §3eu |
| 8 | **Measure the ORACLE CEILING before building a detector** (control #5) | top oracle **+1.252** vs best detector **+0.090** — closed pivot detection outright | **Law** | §3as, §3ao |
| 9 | **Accuracy is payoff-blind** (control #8) | a **5.9σ** AUC improvement produced **−0.702** Sharpe | **Law** | §3au |
| 10 | **Benchmark against buy-and-hold, never against zero** (control #2) | COT 0/36, external playbook 0/60 — every "win" was a loss against simply holding gold | **Law** | §3bg, §3bk |
| 11 | **`excess(LONG) + excess(SHORT) = −2 × base`** | an identity, not a result: on a drifting asset both sides of a mirror pair lose unless the rule is genuinely selective. Explains most closures in the file | **Law** | §3bh |
| 12 | **Enter at CONFIRMATION, never on the pivot bar** | 0.97 → 0.19–0.67; on the same events a 1.000 win rate becomes **0.48** | **Law** | §3cf, §3eq |
| 13 | **Waiting for a dip selects losers** | FILLED **−22.8bp** vs SKIPPED **+17.9bp** (t −5.23); independently confirmed twice | **Law** | §3bc, §3bz |
| 14 | **An exit-rule test must hold the EVENT SET fixed** | the breakeven stop flipped +0.20 → **−3.55** purely from exposure; §3eu: sizing ≠ skipping, because skipping frees the harness's single slot | **Law** | §3co, §3eu |
| 15 | **Score a FILTER against a random same-size drop** | an SMA50 filter looked like +0.244 Sharpe and was a coin flip in the other source. Fewer trades moves Sharpe and drawdown by itself | **Law** | §3er |
| 16 | **The intraday cost wall** | breakeven hit rate M15 **0.587**, H1 0.549, D1 0.510; the best honest AUC ever reached here is ~0.59 | **Bound** | §3ay |
| 17 | **Representation is not the bottleneck** | the raw-window ceiling closed fracdiff, directional change, entropy, signatures, SAX and Renko in a single result | **Bound** | §3ao |
| 18 | **The giveback family is closed — all five levers** | locking 50% of peak gives **86% wins and −85% return**; to shrink swings cut LOT SIZE, never scale out | **Bound** | §3cx, §3cy, §3dq |
| 19 | **risk_frac is a pure dial** | Sharpe invariant at **+1.02** across 0.25–4%: sizing moves P&L and drawdown, never edge | **Bound** | §3ap |
| 20 | **Single-source data only** | FTMO spliced into the H4 files from 2023-06-28 (25 of 38 pairs); all 18 FX D1 files forward-stamp `ret[t+1]` | **Bound** | §3ba, §3bd |
| 21 | **Intraday on gold: momentum is real everywhere, and LATENCY is the cost** — only a resting stop at the prior server-day high clears it | 7/7 M1 cells t +3.0..+5.0 vs matched hold (2007-26, p 0.0000); six net −0.23..+0.36; **prior-day-high buy-stop +0.76** at a level fill, +0.49 five seconds late, +0.26 an hour late; placebo level p 0.000. Not deployed | **Bound** (+ one lead) | §3ex |
| 22 | **Pretrained candle/time-series models forecast RANGE, not direction** | Kronos + Chronos, two protocols, two sources: direction family p 0.94; H1 IC **−0.030** vs the paper's forex 0.040; ZigZag dial wrong-signed; embedding probe p 0.185; headline ORACLE next-day t −0.35 | **Bound** | §3ew |
| 22 | **A pretrained candle model is not the missing ingredient either** | Kronos + Chronos zero-shot on gold: direction p 0.94 / 0.58, paper-protocol IC −0.030 vs the paper's pooled +0.040, ZigZag dial p 0.68 / 0.33, embedding probe p 0.19; human-labelled headlines: same day t +25.9, next day t −0.35 | **Bound** | §3ew |

---

## The one line that matters

Items 1–7 are everything that ever made money here — and **four of the seven are cost or risk
decisions, not signals**. Items 8–20 are what the ~130 failures bought: the laws and bounds that
now kill a bad idea in an hour instead of a week.

## How to use it

- **Before designing anything**, check 8, 10 and 16. Most new ideas die on one of those three.
- **Before believing a backtest**, check 11, 12, 13 and 14. Most false positives here came from
  one of those four.
- **Before deploying**, check 15 and 19 — and remember that the ZigZag's own Sharpe is
  [−0.30, +0.70] on CME and [−0.26, +1.08] on Switch, i.e. not distinguishable from zero. Several
  "improvements" above improve the *shape* of something whose edge is not established.
- **Before adding a feed**, check 20.

## Maintenance

Add a row only when a result is *load-bearing* — it changed a deployment or closed a family.
Everything else belongs in `V5_FINDINGS.md`. When a row is refuted, do not delete it: move it down
and say what killed it, the way §3er refuted §3es's recommended action.
