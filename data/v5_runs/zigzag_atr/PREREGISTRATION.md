# PRE-REGISTRATION — ZigZag ATR-scaled bracket (EXPLORATION-QUEUE #29)

Written **before the first fit**. Commit this file and record its hash in the results.

## The question

§3x disproved the zigzag as a trade using a 48-cell grid of **fixed-percentage** brackets
(0.15–0.6% flat): walk-forward Sharpe **−0.76**, 0 of 8 years better than buy-and-hold,
DSR 0.000. §3ax then found gold's mean daily range went from **1.34% of price in 2015 to 2.73%
in 2026**, which makes the deployed 0.60%/0.50% bracket **0.23 and 0.18 of a single day's
travel**, with **96.0% of recent sessions ranging more than TP+SL combined**. When both barriers
sit inside one session's noise, which one resolves first approaches a coin flip *regardless of
whether the detector was right*.

**So §3x may have measured the bracket rather than the detector.** §3ay names this the one open,
actionable item in the log. This round settles it.

Changing the bracket changes the **trade structure**, not the ranking of a label, so MANDATORY
CONTROL #8's scope clause exempts it ("#8 retires any programme whose mechanism is 'rank the
same label better'. It does not touch programmes that change the TARGET.").

## Honest prior, recorded so the result cannot be rounded up afterwards

**This is expected to fail, most likely at Z3.** The bracket changes which barrier resolves; it
does not change the detector's fire-weighted mean offset of **+0.967** (§3as), and that offset is
why EV/fire is **+0.05% per 5 days against +0.15% for simply being long** (paired t −3.64).

The one reason it is worth a day: §3as measured EV on a **fixed 5-day forward return**. A 2-ATR
target against a 1-ATR stop is **asymmetric** and can pay at a sub-50% hit rate. Those are
different statistics and only the first has ever been measured on this detector.

## Declared grid — FIXED. K = 6. No additions.

| | |
|---|---|
| signal | the **deployed** H1 detector: HistGB(300, lr 0.05, 31 leaves) on `v5_xau_turning_ml.features` minus `hour`, ZigZag `order=5, theta=1.5·ATR`, `label_near(tol=3)` |
| training | rolling 20,000 trailing H1 bars, refit every 250 bars, **purge 8** (= TOL + order, the label's true dependency reach; Gate 0 measured purge 4 vs 8 at ΔAUC +0.0000) |
| **cells** | `tp ∈ {1.0, 2.0, 3.0}·ATR(14,H1)` × `sl ∈ {1.0, 1.5}·ATR` = **6** |
| max hold | **48 H1 bars** (the deployed value). `24` is reported as a robustness ROW, not a 7th cell |
| control cell | the deployed fixed **+0.60% / −0.50%**, frozen, **not a candidate** |
| threshold | swept for the Z5 specificity gate only; the headline cell is quoted at the deployed **0.60** |
| cost | round trip floored at the live FTMO quote **$0.448/oz** (CONTROL #1) |
| resolution | brackets resolved on **M15**, `dec_bar_minutes=60` (§3bj: order of events needs intraday; that bug inflated NR7 19×) |
| ambiguity | TP and SL inside the same M15 bar → counted separately and reported under **both** conventions (SL-first, and dropped). §3bj measured an unresolvable convention at 0.65 Sharpe |
| window | 2015-01-01 … **2026-06-15** (the clean single-feed frames end there; the spliced files are not used) |

## Gates — all pre-registered, evaluated in this order

Controls run **#7 before #5**: answerability is the more restrictive and kills cheaper.

| # | criterion | threshold |
|---|---|---|
| **Z0** | non-overlapping decided fires (≥48 H1 bars apart) | **≥ 400** (`v5_advisor_measure.MIN_NONOVERLAP`) |
| **Z2** | measured bracket hit rate vs its **own** breakeven | `p* = (sl + cost)/(tp + sl)`; must exceed it |
| **Z1** | **oracle** (fire exactly at true pivots) dSharpe vs **buy-and-hold gold**, best cell | **≥ +0.20 and t ≥ 2.5** (CONTROL #5; CONTROL #2 fixes the benchmark as holding gold, never zero) |
| **Z3** | **EV/fire net > the drift it displaces** → `skill = hit − base > 0` | **the gate §3as failed: +0.05% vs +0.15%** |
| **Z4** | walk-forward Sharpe, paired vs buy-and-hold | **> 0, t ≥ +1.5, ≥ 6/11 years** |
| **Z5** | best-of-6 **signed** max-stat null on **GROSS** returns | **p ≤ 0.05** |
| **Z6** | both halves positive | 2015–2020 **and** 2021–2026 (rule 1.3.8) |
| **Z7** | PBO over the 6-cell grid (DSR reported, **not** gated — rule 1.3.2) | **< 0.30** |

**KILL: failure of Z1 or Z3 closes the arm.** A kill means: write the verdict into
`V5_FINDINGS.md`, mark queue #29 done, leave `zigzag-obs` running signal-only, and hold no
further zigzag research sessions. (Decided with the user before the run.)

## Method rules that bind, each one a logged mistake

- **Bootstrap GROSS, not net** (§3bf). A deterministic cost offset inside a zero-mean null
  manufactures significance in proportion to sample size — it produced "SURVIVES p 0.0000" twice
  on dead studies. Costs are a hurdle on the point estimate, never random inside the bootstrap.
- **Price the SIGNED maximum**, not |t| — ranking on absolute value lets a significantly negative
  cell pose as a survivor.
- **Print the base rate beside the hit rate** (§3ay). Positive EV with `skill ≤ 0` is long
  exposure, not skill.
- **Report the fire-weighted mean offset** (#8). A detector with a positive mean offset is a
  CONFIRMATION detector and is not tradeable regardless of its precision.
- **Gate on specificity, not recall** (#7). A 10% false-positive rate cost 74% of the prize in
  §3am, while cutting recall 100%→50% cost 0.015.

## Gate 0 result (prerequisite, already run)

`data/v5_runs/zigzag_atr/gate0.json` — PASSED. H1 vs M15 agree at **1.000 in all 12 years**
(the zigzag, unlike the advisor, lives on a genuine single-broker feed); `intrabar_leak` H1
0.0191 / M15 0.0166 against a 0.15 bar; lookahead probe clean on 96,330 comparisons;
**purge 4 vs 8 = ΔAUC +0.0000**, so the live hard-coded purge is harmless. Clean frames end
2026-06-15 (110 days stale) and the backtest window ends there by choice.
