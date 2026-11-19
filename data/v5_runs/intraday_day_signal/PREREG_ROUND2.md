# PREREGISTRATION, ROUND 2 — "can you try other techniques?" (§3ez, §3fa, §3fb)

Written 2026-11-16 evening, before any outcome below was computed. Repo HEAD `72b2bdc`; side
branch `intraday-day-signal`. Nothing here may be edited after the first run; deviations go in
the results under "DEVIATIONS".

## The ask

User, 2026-11-16, after the PDH day-trade executor went live on Switch (§3ey):
> "can you try other techniques?"

## What makes this round legitimate rather than a re-run (RULES.MD §1)

**A new data type, again.** The public OANDA library used in §3ex (HF `bunqi/OANDA_*`) covers
~127 instruments at 5-second / 1-minute / 5-minute bars from 2002-2006 (BTC 2016) to 2026-11-13,
one provenance. The repo has never had intraday history for anything but gold beyond a few
months, so three things that were unmeasurable are now measurable:

| technique | status before | why it can be asked now |
|---|---|---|
| **A. the passing rule on other instruments** | §3ex: "the same rule failing on NQ/BTC under its own pre-registration (not run — it would be a new study)" | exact intraday path for 24 Switch-tradeable instruments, 20 years |
| **B. level salience** | OCTOBER D23 open; §3cj/§3ex say the level is special, not which levels are MORE special | a panel gives ~60,000 breaks instead of gold's 2,500 |
| **C. meta-labeling the breaks (ML)** | only ever tried on the trend bot's fires (failed twice) and on ZigZag (a lead); `src/features/meta_labels.py` has zero callers; never on a base rule with a real edge | CONTROL #7's power problem: pooled panel training is the one durable capability here (§3as) |
| **D. gold in other currencies** | never: no XAUEUR intraday history existed | XAUEUR M5 2006-2026 |

Closed and NOT re-run: every exit variant (§3ex: not a lever), tight stops (§3ex ladder), the
short leg on gold as a book component (RANK #3), the vol dial on this rule (§3ex: 86th pctile),
level types already scored as signals (round numbers, opening range: §3bi, §3ex O1), forecasting
models (§3ew dominates ARIMA/GARCH/LSTM-style forecasters: a pretrained model found nothing).

## Data, clock, the frozen rule

OANDA **M5 mid** bars, UTC → New York; trading day = 17:00 NY → 17:00 NY (the MT5 server day).
Days with fewer than half the instrument's median bar count are dropped. Results from
2007-01-01 (BTC from its start + warm-up). The rule is §3ey's deployed variant, UNCHANGED:

* LONG leg: resting buy stop at the prior day's high; skip a day whose first bar opens at/above
  it; stop 1.0 × ATR14(D1) below the entry; exit at the close of the last bar before 16:40 NY;
  one trade per day. Fill = max(level, the trigger bar's open); the full round trip is charged.
* SHORT leg: the exact mirror at the prior day's low.
* Known answer: gold's LONG leg on M5 must reproduce the M1 figure (net +0.79) within 0.05.

**Panel and round-trip cost** (Switch census 2026-11, live spread rounded UP to the next 0.5bp,
floor 1.0bp; gold keeps §3ex's fixed $0.30; every cost also at 2×):

| class | instruments (cost bp) |
|---|---|
| indices | NAS100 1.0 · US30 1.0 · DE30 1.0 · SPX500 2.0 · JP225 2.5 · AU200 3.5 · US2000 5.0 · UK100 6.0 |
| FX | EURUSD 1.5 · GBPUSD 1.5 · USDJPY 1.5 · USDCAD 1.5 · GBPJPY 1.5 · EURJPY 2.0 · AUDJPY 2.0 · EURGBP 2.0 · USDCHF 2.0 · AUDUSD 2.5 · NZDUSD 3.0 |
| metals | XAUUSD $0.30 · XAGUSD 5.5 |
| energy | WTICO 4.5 · BCO 5.0 |
| crypto | BTCUSD 4.5 |

24 instruments × 2 legs = **48 cells**.

---

## STUDY A (§3ez) — is the stop-cascade break a property of gold, or of levels?

**Hypothesis.** A resting stop at the prior-day extreme earns more than holding at matched
exposure on instruments other than gold, net of their own costs, and an equal-weight book of the
legs beats the single gold leg already deployed.

**Per cell:** trades/wk, gross bp, gross / net / 2× Sharpe (1/vol weighting as §3ex), halves,
2007-14 vs 2015-26, breakeven cost, and `t_a` = t of `gross − n̄ × HOLD` (signed exposure, so a
short leg is benchmarked against being short the drift). Family: stationary block bootstrap
(block 20, 5,000 draws) of the recentred `a_d` matrix, signed max over the 48 cells, gross.

**Books (fixed now):**
* `BOOK_LONG` — equal-weight mean of the 1/vol LONG legs of every instrument available that day.
* `BOOK_LS` — the same over all 48 legs.
* `BOOK_WF` — walk-forward selection, the only honest way to "trade what works": in calendar year
  Y (from 2012) hold the legs whose NET Sharpe over all days before Y is > +0.30 with ≥ 3 years
  of history; equal weight; flat if none.

**PASS (a book):** net Sharpe ≥ +0.80 at 1× and ≥ +0.40 at 2×; both halves positive; positive in
2007-14 and in 2015-26 (`BOOK_WF`: 2012-18 and 2019-26); AND paired t ≥ +2.0 against the gold
LONG leg alone at matched vol — a book that does not beat what is already live is not an upgrade.
**KILL:** fewer than 4 of the 23 non-gold LONG legs net-positive → the rule is gold-specific;
closed, and no instrument is added to the executor.
**If a book passes:** the 5-second fill audit of §3ex is repeated on its three largest
contributors before anything is said about deployment.

## STUDY B (§3fa) — level salience

**Hypothesis.** The more days the level has stood, the more stops rest beyond it, the stronger
the continuation. Freshness `N` = number of consecutive days before yesterday whose high was
below yesterday's high (capped at 20), so the level is an (N+1)-day high. Mirror for lows.

**Statistic.** Pooled over the panel's LONG trades: Spearman(N, gross return in ATR units);
null = permutation of N across trades WITHIN instrument-year blocks, 2,000 draws. Printed: mean
return by bucket N = 0 · 1-2 · 3-5 · 6-10 · >10, per asset class, and the same for shorts.
**PASS:** rho > 0 with p < 0.01, same sign in ≥ 3 of the 5 classes, and in both halves.
A pass makes `N` a pre-specified feature of Study C and nothing more.

## STUDY C (§3fb) — meta-labeling: a learned break-quality score

**Hypothesis.** Features known at or before the trigger rank the breaks out of sample well
enough to lift the live gold rule, and the book, as a SIZE dial.

**Events:** every LONG and SHORT trade of Study A (side is a feature). **Label:** the trade's net
return in ATR units. **Features, all scale-free, all strictly before the fill:**
* known at the open — freshness N; distance open→level / ATR; prior-day range / ATR; prior close
  location in its range; 5- and 20-day return / (ATR·√n); downside-vol state (§3eu definition on
  daily bars: 20-day downside std vs its 250-day median); weekday; overnight gap / ATR;
* known at the trigger — minutes since the day's first bar; New York clock hour; day's adverse
  excursion before the trigger / ATR; run from the day's extreme to the level / ATR; price change
  over the previous 30 and 120 minutes / ATR; the last 30 minutes' range vs the 20-day mean range
  at that clock time; count of earlier bars today within 0.05 ATR of the level;
* identity — side, asset class (one-hot). No instrument identity.

**Model (fixed, not tuned):** `HistGradientBoostingRegressor(max_depth=3, max_iter=200,
learning_rate=0.05, min_samples_leaf=200, l2_regularization=1.0)`; a ridge on standardised
features is printed beside it as the capacity check. **Walk-forward:** fit on trades that exited
before 1 January of year Y, predict year Y, Y = 2012..2026. **Arms:** POOLED · TRANSFER (gold
removed from training) · GOLD-ONLY.

**Scored on gold's LONG trades (the live rule), then on `BOOK_LONG`:**
1. Spearman(score, R) out of sample, permutation p;
2. SIZE dial `m = 0.5 + expanding percentile rank of the score among earlier OOS trades` (mean 1),
   event set fixed, against the same multipliers dealt at random (2,000) — §3eu protocol — with
   the inverse dial as the sign control;
3. SKIP the bottom 30% (expanding threshold) against a random same-size drop (500) — control #10.
Controls: a shuffled-LABEL fit must collapse to no lift; ridge vs GBM printed (model class is not
supposed to be the lever, §3bl).

**PASS:** dial shuffle p < 0.05 on gold (POOLED arm) with both halves positive and the inverse
negative; AND the same on `BOOK_LONG`. **KILL:** gold OOS Spearman not above zero at p < 0.10 →
closed; nothing is added to the executor.

## STUDY D — gold in euros (gold LONG trades only)

At each gold trigger: (D1) is XAUEUR at/above ITS OWN prior-day high at that bar? (D2) is the
day-so-far move mostly the dollar — EURUSD's change from the day's open more than half of
XAUUSD's? Mean gross return in ATR units, group vs complement, permutation p (2,000).
**PASS:** p < 0.01 and the same sign in both halves. Otherwise a description, not a filter.

## Honest priors

A: a book clearing every bar incl. beating gold alone ~15%; equity indices are the textbook
mean-reverters at this horizon and FX costs 2-4× gold's. B ~30%. C ~15% — the repo's own record
on overlays is 1 lead and many failures, and CONTROL #8 says a ranking gain need not be money.
D ~10%. The base rate here is that the next idea fails.

## What is NOT being done

No parameter of the rule is varied by instrument (stop, exit time, day boundary, gap rule). No
instrument is selected with hindsight outside `BOOK_WF`. No model setting is searched. No live
order is placed and the executor is not touched in this round.
