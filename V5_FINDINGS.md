# V5 FINDINGS — settled experiments (do not repeat)

> **Read `RESEARCH-REGIMES.md` first.** It partitions everything below into prop-firm-constrained
> findings (regime P), unconstrained findings (regime U, current since 2026-09-04) and
> invariant findings — plus an explicit resume point for each regime. Several conclusions here
> are only correct under a given rule set and flip when the rules come off.


Running ledger of XAUUSD research that has been **run, measured, and closed**. Each
entry: what was tried, the honest result, and the verdict. If an idea here is
marked DISPROVEN / DEAD, do not re-run it without a materially new angle.

Conventions: net Sharpe from daily-resampled equity × √252; eval window = 2017+;
the live HFM cent account's true gold cost is **$0.34 spread** (the `XAUUSD_*_long.csv`
spread column understates by 10× — use `--fixed-spread-usd 0.34`, not the raw column).

---

## MANDATORY CONTROLS (2026-07-24, #5 added 2026-07-29) — five ways this project has fooled itself

Each of these silently manufactured a "result" that evaporated when the control
was added. Run all five before believing any new backtest.

**1. Floor every cost at the LIVE broker quote.** The 10× gold understatement in
the Conventions above is not special — it is universal, and far worse on illiquid
names. Measured against live FTMO quotes 2026-07-24:

| asset | CSV models | live | error |
|---|---|---|---|
| NATGAS | 4.0 bp | 199.4 bp | **50×** |
| HEATOIL | 4.0 bp | 80.1 bp | **20×** |
| PALL | 5.0 bp | 56.4 bp | 11× |
| BRENT / WTI / SILVER | ~3 bp | 8–9 bp | ~3× |
| gold H4 | $0.30/oz | $0.448/oz | 1.5× |
| NDX / SPX / DJI / BTC / SOL / NIKKEI / DAX | — | — | already conservative |

A first pass of §3i crowned `XAU+NDX+HEATOIL` the best book in the study on a 20×
cost subsidy; at honest cost HEATOIL collapses **+0.47 → +0.09 SR** and the whole
agriculture complex goes negative. Use `np.maximum(csv_spread, close*live_bp/1e4)`.

**2. Benchmark against BUY & HOLD over the same window.** A long-only trend
follower in a bull market posts a fine Sharpe with zero timing skill:

| window | buy & hold gold | always-in champion | weekly cycle |
|---|---|---|---|
| 2008+ (18.4y) | **+0.57** | +0.64 | +0.44 |
| 2022+ (4.4y) | **+1.15** | +1.11 | +1.07 |

The strategy never beats owning the metal — its Sharpe just tracks gold's own
Sharpe in whichever window you pick. It is selling **drawdown control** (−10.0% vs
−23.1% in 2022+), not alpha. `v5_xau_weekly_cycle.py` prints this as STEP 0.

**3. Walk-forward the UNIVERSE, not just the parameters.** Choosing which assets
to trade using the full sample roughly doubles apparent Sharpe:
quality-17 always-in **+1.03** → weekend-flat **+0.61** → walk-forward top-8
(re-picked each January on trailing 3y) **+0.26**.

**4. Quote a standard error.** Lo (2002): SE ≈ √((1+SR²/2)/years). At 4.4 years
SE ≈ **0.6**, so a post-2022 "+1.07" has a 95% CI of roughly [−0.11, +2.25] — not
distinguishable from the full-sample +0.44, nor from zero. Short windows cannot
resolve a Sharpe. Watch also for the **fractal regime tell**: post-2022 looked
strong, but within it 1st-half +0.11 vs 2nd-half +1.75, i.e. really 2024-2026.

**5. QUOTE THE ORACLE CEILING before building a detector** (added 2026-07-29, §3p).
Trade the label with PERFECT HINDSIGHT first and report (a) standalone Sharpe,
(b) paired dSharpe + t vs the incumbent, (c) the (precision, recall) contour that
reaches dSharpe +0.20. **If oracle dSharpe < +0.20 or oracle t < 2.5, do not build the
detector** — no calibration, encoder, or walk-forward can lift a ceiling. This control
retroactively explains §3: five studies chased gold BOTTOMS, whose perfect-hindsight
value is **+0.017** (and **-0.675** standalone), while the value was in TOPS (+0.26).
It costs ~5 minutes and would have saved months. `scripts/v5_xau_turn_prob.py --approach 1`.

**6. A "CONFIDENTLY REAL" claim needs DSR>=0.90 AND PBO<0.30 AND a walk-forward re-check
— never a raw Sharpe alone** (added 2026-08-19, §3s/§3t). Borrowed from Forven's Deflated
Sharpe Ratio + Probability of Backtest Overfitting (both already implemented in
`src/evaluation/dsr_pbo.py`, rarely actually invoked before this date) after this project
kept re-learning the same lesson on its own leads within a single session:
- **SuperTrend on XAU** (§3s): DSR 0.9948 on a 20-cell grid — genuinely real, survives
  every check. This is what "confidently real" looks like.
- **The SL/TP bracket search** (§3r round 5): DSR 0.0000 on an 80-cell grid, despite a
  cell that looked fine on raw Sharpe (+0.015) and even had a low PBO (0.000) — **PBO can
  be misleadingly low when a grid's variation is a monotonic COST gradient rather than a
  genuine edge gradient; check DSR too, the two are not redundant.**
- **Cross-asset divergence on SPX/NDX** (§3t): the full-sample-picked cell's Sharpe
  (+0.547) was cut MORE THAN HALF (+0.238) by a proper walk-forward re-selection (trailing
  3y, re-picked each January — this repo's own precedent from control #3). **DSR/PBO on a
  full-sample grid is necessary but not sufficient — a walk-forward re-check of the
  SELECTION PROCESS itself, not just the winning cell's own robustness tests, is what
  actually caught this.**
- **The bar, stated plainly: standalone DSR>=0.90, PBO<0.30, AND the winning
  configuration must survive being re-selected walk-forward (not just tested walk-forward
  after being picked on the full sample).** Below that bar, a result is "a real, unproven
  lead" (cross-asset divergence's residual correlation/DD effect) or "noise" (the bracket
  search) — never quietly rounded up to "found an edge."

**Guiding principle behind all six controls — evidence may only DOWNGRADE, never let a
good story UPGRADE.** A verified failure (dead cost-adjusted Sharpe, a walk-forward
collapse, a lookahead-probe fail) always overrides an exciting-looking number; an
exciting-looking number never overrides a verified failure, no matter how good the
narrative for why "this time is different." This session's own arc is the case study:
GOLD/SILVER divergence, SuperTrend-basket blends, and the cross-asset-divergence flagship
blend all *looked* like discoveries before the actual checks (honest cost, decomposition,
walk-forward re-selection) downgraded them — never the reverse.

---

## HARD RULES — consolidated risk limits by account/model

One glance-able table instead of re-deriving these from `v5_basket_challenge.py`'s
`MODELS` dict and `CHALLENGEBOT.MD`'s prose each time. `guard_frac`/`halt_frac` are this
repo's own proactive de-risk/flatten thresholds, set BELOW the broker's hard limit so the
bot always has a buffer before a real breach.

| Account / model | vol dial | Phase 1 target | Phase 2 target | Daily loss (hard) | Max loss (hard, static) | Guard (de-risk) | Halt (flatten) |
|---|---|---|---|---|---|---|---|
| FundingPips Standard (default) | 7% | 8% | 5% | 5% | 10% | 3.5% | 8% |
| FundingPips Flex | 7% | 10% | 6% | 4% | 12% | 3.0% | 10% |
| FundingPips Pro | 5% | 6% | 6% | 3% | 6% | 2.2% | 5% |
| FundingPips OneStep | 7% | 10% | — | 5% | 6% | 3.5% | 5% |
| FTMO 2-Step | 7% | 10% | 5% | 5% | 10% | 3.5% | 8% |
| Cent account (live, HFM Live2) | n/a — single XAU champion, ~1% risk/0.01 lot | — | — | — | — | — | — |

Daily loss is measured on floating P&L (equity, not just closed balance), resets at the
broker's platform-time midnight. Max loss is static against the INITIAL account size, not
a trailing high-water mark. Source of truth for FundingPips mechanics: `CHALLENGEBOT.MD`
§1; for the numeric dials: `scripts/v5_basket_challenge.py::MODELS`.

---

## Champions currently trusted (positive net edge)

| Strategy | Net edge | Status | Notes |
|---|---|---|---|
| Sharpe-1.6 drift portfolio | eval SR **1.59**, DSR 0.9998 | backtested, **not deployed** | champion recipe across BTC/indices/XAU/silver + LS diversifiers; the real upside |
| Long-only XAU champion | eval **0.99–1.04**, live ~0.97 | **LIVE** (acct 360542) | H4, vol-targeted EWMAC+breakout, conc^1.5, long-only. `data/v5_runs/xau-longonly-champion/` |
| LS ensemble | SR **0.81** | **LIVE** (acct 360541) | long-short trend diversifier |
| 2nd independent basket: NIKKEI+COFFEE+ETH+DAX | eval SR **0.81**, CI [+0.05,+1.58] | backtested, **not deployed**, weaker than flagship | same recipe, DIFFERENT asset set, ~0.3 corr to BTC/NDX. Detail below. |

Single-XAU ceiling ≈ 1.06; the jump to 1.6 is multi-asset diversification only.

### FundingPips challenge — DIVERSIFY the book (2026-07-15)
Single-XAU passes the FP 2-Step only ~61% once the daily-loss rule is measured on *floating* P&L (day_safety=1.5 proxy) — all risk in one asset breaches the 5% intraday line ~32% of the time. Diversifying the same champion recipe across the tradeable drift classes {eq_us SPX/NDX/DJI, eq_eu DAX/FTSE/STOXX, eq_ap NIKKEI/ASX, crypto BTC/ETH, xau H4-champ, metal SILVER}, equal-class-risk at 9% vol, lifts realistic pass to **~80% (daily-loss fails 32%→7%), median ~9.4mo**. (Full basket incl. rates/energy = 93% but those aren't offered.) Engine `scripts/v5_basket_challenge.py` (`--backtest` validates eval SR 1.27 / 80% pass; `--targets` emits live per-symbol leverage). Live MT5 order-wiring deferred until the FP account is purchased (needs real symbol names + terminal). Single-XAU `v5_xau_challenge.py` remains the fallback. Detail in CHALLENGEBOT.MD / memory.

### Basket improvement experiments (2026-07-16, `data/v5_runs/basket-ls-experiment/`)
- **Long/short per sleeve (like the cent `ls` bot) — DISPROVEN, badly:** eval SR 1.26→0.23, pass 92%→48%. Drift assets trend UP; shorting them bleeds. Crash-hedge (shorts only in deep downtrends) also worse (0.97). Do NOT add shorts.
- **Sharpe-weighting sleeves = lookahead illusion:** 1.46/96% in-sample, but walk-forward (past-data weights) = 1.25/88.7% ≈ equal-class. Equal-class weighting is already near-optimal; keep it.
- **GENUINE robust win: portfolio-level VOLATILITY TARGETING** (scale book to constant trailing vol, causal). Eval SR **1.26→1.39**, +dd-scaler **→1.43**; pass 91.9%→94.3% @7% (96.8% @6%), and FASTER. Recommended upgrade — scale per-symbol target leverages by a trailing-vol scalar on the book's own returns. Report: `basket-ls-experiment/REPORT.md`.

### FundingPips — FOCUSED book beats the 12-instrument basket (2026-07-18, `scripts/v5_xau_focus_challenge.py`)
Prompted by a live basket drawdown scare (a −1% wobble at 3 days = pure noise at 7% vol). Tested XAU-focused subsets vs the deployed 6-class basket on the exact FP 2-Step sim (realistic day_safety=1.5, vol-target ON, eval 2017+):

| Book | #a | SR17 | SR21 | pass% | fail-DD | median |
|---|---|---|---|---|---|---|
| 6-class basket (was deployed) | 12 | 1.43 | 1.22 | 94.3 | 5.7 | 12.3mo |
| XAU only | 1 | 1.12 | 1.20 | 94.2 | 5.8 | 17.0mo |
| **XAU+BTC+NDX equal ⅓** | **3** | **1.71** | **1.28** | **98.7** | **1.4** | **11.4mo** |
| XAU-tilt 50%+BTC/NDX 25% | 3 | 1.69 | 1.33 | 98.8 | 1.2 | 11.6mo |
| +SILVER (eq ¼) | 4 | 1.56 | 1.19 | 98.2 | 1.8 | 12.2mo |

**XAU+BTC+NDX (equal ⅓) dominates the basket on every FP metric with ¼ the symbols.** Correlations 2017+: XAU/BTC 0.08, XAU/NDX 0.04, BTC/NDX 0.12 (truly independent); the old basket diluted into correlated index clones (SPX≈NDX≈DJI 0.86) and gold-correlated SILVER (0.58). **DEPLOYED: engine `CLASSES` + `configs/v5_basket_challenge.json` switched to the 3-asset focused book (old 6-class kept as `BASKET_FULL` for revert).** Verified: `--backtest` SR 1.706 / pass 98.7%; `--targets` emits XAU/BTC/NDX only.

### A second, independent basket exists — NIKKEI+COFFEE+ETH+DAX, weaker than the flagship (2026-08-11)

User asked whether the SAME champion recipe works on a genuinely different instrument
set, independent of the deployed XAU+BTC+NDX+BRENT book. Screened every non-deployed
`data/*_D1_long.csv` instrument (~24 candidates: metals, energies, softs/ags, equity
indices, crypto alts, bonds) with the identical recipe, reusing
`v5_basket_challenge.py`'s `_load_asset`/`build()` directly rather than a new engine.

**Two real bugs caught during the search, both worth remembering:**
1. Blindly trusting `v5_instrument_search.py`'s STEP-1 screen would have kept NATGAS as a
   "CANDIDATE" (standalone SR +0.30 on the raw CSV spread) — but MANDATORY CONTROL #1's own
   documented live cost for NATGAS is 199.4bp (50× the CSV). Applying it flips NATGAS to SR
   **-0.36** — dead. PALL similarly weakens +0.40→+0.17 at its documented 56.4bp live cost.
   Any candidate without a documented live quote (softs/ags, alt-crypto) was instead
   stress-tested at 1×/3×/5×/10× the CSV spread before being trusted; SUGAR/WHEAT/SOY/FTSE/
   PLAT failed even mild stress and were dropped.
2. `vbc.build()`'s class-stream math does `al[m].fillna(0.0)` for missing history, which
   silently ZERO-FILLS (not excludes) a newer asset's pre-listing period — this
   under-weights the book for however long a new member (SOL: data from 2020-04; ETH: from
   2017-11) didn't exist yet, AND makes a `.dropna()`-based buy&hold comparison cover a
   different, shorter window than the strategy's own blended statistic. Caught via a
   buy&hold Sharpe (+1.26) that looked disconnected from the strategy window it was meant to
   benchmark. Any new basket construction with mixed asset ages MUST restrict eval start to
   strictly after the youngest member's data begins, or discard/replace that member.

**The survivor: NIKKEI + COFFEE + ETH + DAX**, one per class, equal-class-risk, same
vol-target+DD-scaler as the deployed book. Max pairwise correlation among the four is 0.20
(NIKKEI-DAX); NIKKEI and DAX are both already on MANDATORY CONTROL #1's "cost verified
conservative" list. Clean window (2018-01-01+, strictly after ETH's start):

| metric | this new basket | deployed XAU+BTC+NDX (same window/sim) |
|---|---|---|
| eval SR | **+0.81** (95% CI [+0.05,+1.58]) | +1.41 |
| paired-t vs buy&hold of same assets | +1.31 (favourable, not significant) | (already established) |
| FTMO realistic pass% / median | **84.9% / 21.1mo** | 97.0% / 15.5mo |
| corr to deployed book | XAU 0.08, BRENT 0.07, BTC 0.32, NDX 0.30 | — |
| years positive (of 10) | 7-8 | — |

**Verdict: real, but clearly a second-tier book, not a replacement.** The CI barely
excludes zero, and unlike the XAU champion's "sells drawdown control" story, this book's
maxDD is roughly a wash vs its own buy&hold (-13.1% vs -13.6%) — most of the honest claim
here is "the recipe generalizes to a new asset set with a plausible edge," not "found
another 1.4+ Sharpe book." Survives a 2x cost-stress test (SR 0.92→0.84). NOT deployed;
backtested only, same status as the Sharpe-1.6 drift portfolio above. Backup: swap ETH for
LTC (data from 2014, no history-length caveat) for a fully clean, slightly weaker (SR
~0.76-0.78) alternative. Full methodology + correlation matrix: Claude memory
`new-independent-basket-nikkei-coffee-eth-dax`.

**Do not re-open naively:** the 70-candidate diversifier search (§3q) already found breadth
mostly fails as an ADDITION to the existing book — that is a different question from "is
there a second standalone book," which is what this entry answers, but the same caution on
over-fitting to a full-sample instrument pick (MANDATORY CONTROL #3) applies if this is
pushed further.

**Tested combining it with the flagship anyway (same day) — confirms §3q, don't merge.**
Equal-class merge (7 classes total) makes the flagship WORSE: SR 1.41->1.26, FTMO pass
97.0%->93.9%, median 15.5->16.2mo, paired t -0.88, only 3/9 years better — the new basket's
lower standalone Sharpe (0.81) and non-trivial correlation to BTC/NDX (0.30-0.32) means
equal-class-weighting (~4/7 risk budget) dilutes more than it diversifies. A hand-tuned
SMALL tilt (15-25% allocation, not equal-class) is closer to a wash: Sharpe ticks up
~2% (1.426->1.453) but that is a pure volatility-reduction effect — CAGR actually falls
(14.3%->12.9%), the paired-t of the tilt is **-1.81** (mean return is lower, not higher),
and FTMO pass-rate does not improve (96.7%->96.0%). **No weight tested helps. Run it as
its own separate account/sleeve, never merged into the flagship.**

---

## DISPROVEN / DEAD (do not re-run)

### 1. M15 next-bar fade (mean-reversion) — DEAD net of spread
- **Signal:** fade extreme closes (close near bar LOW → long next bar; near HIGH → short). `scripts/v5_xau_fade_backtest.py`, live paper `scripts/v5_xau_fade_paper.py`.
- **Gross edge is real:** zero-cost M15 hours{8,20,22} FLAT ≈ Sharpe +1.9–2.3, win 54.8%, consistent every year.
- **Dies net of the true $0.34 spread:** win 36%, Sharpe **−6.8**, total −90% over 11.4 yrs. Deployed hours-restricted config included. The per-trade edge is a few cents; the spread is ~10× it.
- **Break-even spread ≈ $0.04–0.07** (need institutional/raw). HFM Zero/Raw ~$0.10–0.15 all-in is still above it.
- **Live paper +$10 over 2 weeks was a lucky window** — 11-yr net is −18%/yr.
- **Verdict:** not deployable as a spread-crossing taker at any timeframe (M15/M30/H1/H4 all tested). Only theoretical path = maker/limit fills that *earn* the spread (untested, likely infeasible retail).

### 2. Martingale / anti-martingale overlays on the fade — DEAD (ruin machine)
- `scripts/v5_xau_fade_martingale.py` — engines flat / double4 (classic capped-4 doubling) / recover4 (deficit-targeted, ladder capped at 4 trades, reset on recovery).
- **On the NET signal (36% win): recover4 BUSTS 100% of 500 block-bootstrap paths, ruin by ~trade 1,266.** double4 −61%, flat −23%.
- **On a GROSS positive-edge signal (54.8% win): recover4 = +152% vs flat +6% / double4 +13%, 0% bust, DD 11%** — proves the martingale is a *lever*, not an edge-creator.
- **Verdict:** decision variable is solely sign(net edge). Base fade is net-negative → no martingale rescues it. Do not deploy any progressive sizing on a negative-edge base.

### 3. Turning-point DETECTION (buy/sell reversal accuracy) — real skill, NOT tradeable
- `scripts/v5_xau_turning_points.py` (accuracy), `scripts/v5_xau_turning_ml.py` (precision push), `scripts/v5_xau_turning_trade.py` (trade sim), `scripts/v5_xau_champion_plus_detector.py` (combine). Ground truth = ZigZag swings; all detectors causal; scored precision/recall/F1 vs a random-coverage baseline.
- **Detection works above chance, robustly (no decay to 2024+):** best rule = Bollinger z-score, buy precision ~50–58% vs 33.5% random (lift 1.5–1.9×). **Bottoms detect materially better than tops** (V-panics vs rounded tops) — every config. Fade candle-shape = **zero** turning skill (lift 1.00×).
- **70% precision IS reachable OOS** with a HistGradientBoosting model — **BUY/bottoms tol=±3 → 71% precision @ 22% recall (2.1× random), 75%@10%, 80%@5%, PR-AUC 0.60.** SELL/tops cannot (70% only @ ~9% recall). Rules alone cap ~62%. Top features: 5-bar ATR-momentum, rank-in-window, z-score, RSI7. CAVEAT: judge by LIFT — widening tol inflates absolute precision by raising the random base.
- **But precision ≠ P&L.** Standalone bottom-long (~27 flags/mo, ~6–16 trades/mo): only profitable exit was hold-48-bars (+35%, SR 0.81) which merely harvests up-drift and **LOST to buy&hold gold (+113%)** in the same 2024–26 bull window. Short holds flat-to-negative.
- **Combining with the champion FAILS.** Overlaying the OOS detector on the H4 champion at equal exposure (pure timing test): champion alone eval SR **0.99** → every overlay worse (mildest bottom-boost 0.88, strong 0.66; trim-tops 0.88/0.65; both 0.40; detector-only-long −0.02). Turnover explodes 9.8→64→427 (prob wobble churns spread). Reason: champion is TREND-following (more exposure after breakouts), detector is MEAN-REVERSION (buy dips) — antagonistic; reversion-timing dilutes the trend edge and pays cost.
- **Anti-martingale ('untimartingale') sizing on the detector trades DOESN'T help** (`scripts/v5_xau_detector_antimartingale.py`). Best base (hold-48) flat Sharpe 0.84 → anti 0.30–0.53 at 2–3× the drawdown; hold-12 anti clearly negative (SR −0.5 to −0.8). Root cause: **win/loss lag-1 autocorrelation ≈ 0 (+0.05 to −0.03) — wins do not cluster.** Progressive sizing (anti *or* martingale) only adds value when outcomes are serially correlated; here trades are ~independent, so pressing winners just adds variance with no expected-return gain → Sharpe falls. Sizing cannot create edge on a streak-less, buy&hold-losing base.
- **Verdict:** detector is real science but not a tradeable edge and does not improve the trend bot. Leave the champion alone (it handles entry via vol-targeting + buffer). Turning-point timing and trend-following are orthogonal-to-antagonistic.

### 3b. Fast (non-trend) strategy exploration (2026-07-16, `data/v5_runs/fast-strategies/`)
Wide sweep for faster, consistent, non-trend edges, all net of spread, OOS=2021+.
- **Cross-sectional reversal on GLOBAL indices = FAKE** (Sharpe 3.5): non-synchronous close times (NIKKEI/ASX close hours before US) leak future info. US-only synchronous = −0.37. Classic artifact — reject any cross-sectional signal on assets with different close times.
- Standalone fast edges are WEAK net of cost: 1-day reversal best NDX 0.65; turn-of-month = rest-of-month (no distinct edge); day-of-week tiny.
- **THE FINDING — diversified FAST ENSEMBLE (overnight + short-term reversal, ~26 daily signals, avg corr 0.02):** select on 2017-2020, deploy 2021+ OOS → **OOS Sharpe 1.30** (IS 1.32, held), consistent (2022 only −0.4). Decompose OOS: **overnight-only 1.19** (the driver, but needs LIVE fill check — close→open gap may not be tradeable on 24h CFDs), **reversal-only 0.57** (cleanly tradeable close→close, ~0 corr to trend). **TREND+FAST 50/50 = 1.44** vs trend alone 1.17 → real diversifying lift. No get-rich-quick fast scheme; the ensemble is a genuine COMPLEMENT to trend, not a replacement.
- **VERIFIED DEAD on HFM's real tradeable instruments (2026-07-16, `scripts/v5_overnight_verify.py` + `v5_fast_verify_all.py` vs demo 57482374):** overnight ensemble −4.65 (backtest used CASH-index close 16:00→open 09:30 window; HFM's futures-CFDs break at a different time → no premium, slightly negative); intraday +1.22 but 0.97 corr to buy-hold = JUST DRIFT; reversal R1/R2/R5 = +0.13/+0.02/−0.35 = no edge (backtest 0.57 doesn't hold on real closes/spreads). **NOTHING in the fast family is deployable.** The backtest edges died on: bar-boundary shifts, wider real spreads, and drift-in-disguise. **RULE: always verify a fast edge on the BROKER's own symbols before building — cash/downloaded data lies about overnight windows and spreads.** Durable edge stays the TREND/DRIFT book. Report: `data/v5_runs/fast-strategies/REPORT.md`.

### 3c. Fast / intraday TREND runner (2026-07-17, `data/v5_runs/fast-trend/`)
Hunt for a *short-term trend* bot to complement the slow H4 champion (more
trades/day). Distinct from the dead fade/reversal work — this is trend, not
mean-reversion. Engines: `scripts/v5_xau_fast_trend_lab.py` (vectorized
vol-target sweep, buffered, combined-book vs champion) + `v5_xau_fast_trend_discrete.py`
(real lot/stop engine, monkeypatched fast champion signal).
- **Long-only beats LS at every speed** (kill-the-shorts again). Trend edge is
  gross-positive at ALL speeds (breakout gross SR ≤1.3) but **net Sharpe falls
  with turnover — spread is the tax.** No intraday-specific alpha: **session-ORB
  and intraday-momentum are gross-positive, net-DEAD** (same failure as fade).
- **Carver no-trade buffer cuts turnover 3–4× at ~no net-Sharpe loss** — key
  lever for spread-bound fast books; but slowing it down raises corr-to-champion
  to 0.73–0.80.
- **A fast sleeve does NOT improve the combined book:** champ-alone 1.28 →
  best 50/50 combo 1.23. On one asset a faster book is a correlated weaker clone.
- **Discrete reality kills the vectorized illusion:** net SR ~1.0 (continuous)
  → **0.4–0.5** (real 3×ATR stops whipsaw at 38% win, $0.34 spread, quantized
  lots); **negative 2017/2018/2021, edge only in the 2024–26 bull** = leveraged
  bull-beta, not robust.
- **The killer is the cent spread — account-type-fixable.** M30 fast champion,
  ~19 trades/mo: **$0.34 → 0.50 (dead); $0.12 raw/ECN → 0.85 (deployable);
  $0.02 → 1.07.** Halving spread ~doubles net Sharpe. M30 fast = best config.
- **LIVE VPS-VERIFIED (2026-07-17, read-only bridge probe):** live cent XAUUSDc
  spread measured **$0.36** (not $0.34 — slightly worse) → discrete net SR 0.49,
  DEAD confirmed. Raw-tier gold IS real at HFM (**XAUUSDb $0.10 FULL-tradable on
  demo**, discrete 0.89; XAUUSDr $0.16 disabled) but **NOT visible on the live
  cent account group** → no tight-spread path on the current live account. Gold
  swap −$0.72/oz/night long (unmodeled). Probes: `scripts/vps_spread_probe.py`,
  `vps_symbol_tradability.py`.
- **Verdict:** NOT deployable on the cent account (verified $0.36). Viable as an
  *activity* play only on a raw-tier HFM gold account (~$0.10–0.16, e.g. XAUUSDb),
  which the live cent group can't reach; and even there it doesn't lift a
  champion+fast portfolio (correlated). Real lever for more trades = cross-ASSET
  trend (BTC/NDX), not cross-speed on XAU. Report: `data/v5_runs/fast-trend/REPORT.md`.

### 3d. Holding-constraint stress: weekend / overnight bans (2026-07-20, `scripts/v5_holding_constraints.py`)
Funded (Master) accounts restrict holding — FundingPips banned weekend holds on 2-Step Flex Masters (29-Jan-2026). Measured the damage on GOLD+ETH+DJI (D1 proxies), Flex rules, eval 2017+:

| Scenario | Sharpe | CAGR | maxDD | pass% |
|---|---|---|---|---|
| hold through (Evaluation) | **+1.14** | 7.7% | −9.4% | **98.0** |
| NO weekend holding | +0.71 | 4.5% | −11.1% | 68.2 |
| NO overnight holding | **−1.03** | −4.1% | −33.6% | **0.4** |

- **Overnight ban is FATAL — strategy goes NEGATIVE.** For indices/gold essentially all long-run drift happens overnight (close→open); being flat every night hands away the edge while paying ~500 crossings/yr vs ~12. No dial fixes it, and an intraday replacement is already ruled out (§3c). **Do not run this book anywhere overnight holding is banned.**
- **Weekend ban is survivable AND adaptable: DROP THE CRYPTO SLEEVE.** GOLD+ETH+DJI 0.71/68.2% → **GOLD+DJI 1.08/79.1%** (GOLD+DJI+SPX 1.05/77.9%; NDX as third is worse on pass, 67%, too volatile for the daily line). Reason: crypto trades **24/7**, so a forced Friday-flat misses real moves, whereas gold/indices are closed anyway and only lose the Monday gap.
- **Action deferred to funding:** full runbook in `FUNDED-STAGE-PLAN.md` (config `classes` swap + a weekend auto-flat feature the executor does not yet have + ±5min news window). Nothing to change during Evaluation — weekend AND overnight holding are explicitly allowed there.

### 3e. Profit-take + dip re-entry overlay — DISPROVEN (2026-07-20, `scripts/v5_profit_take_overlay.py`)
Tested "bank the gains and buy back lower": when the book is up >= X% since entry, close everything; re-enter after price dips Y% from the exit level (or a max-wait timeout). Rationale was that floating profit counts against the firm's daily line, so realising it should protect the account. FTMO book (XAU+BTC+NDX), eval 2017+:

| Variant | Sharpe | CAGR | maxDD | pass% | median | %in-mkt |
|---|---|---|---|---|---|---|
| **baseline hold-through** | **+1.71** | **12.1** | −12.3 | **98.6** | **13.4** | 100 |
| take1.5/dip0.5/10d | 1.55 | 9.5 | −12.3 | 97.0 | 14.6 | 90.9 |
| take1.5/dip1.0/10d | 1.64 | 9.7 | −12.3 | 97.4 | 13.9 | 87.1 |
| take1.0/dip0.5/10d | 1.35 | 8.0 | −12.3 | 94.8 | 16.3 | 88.3 |
| take1.5/dip2.0/20d | 1.27 | 6.7 | −10.4 | 95.1 | 17.2 | 76.8 |

- **Every variant loses**, and performance tracks **%in-market** almost linearly — time out of the market is pure cost. maxDD does NOT improve (−12.3% throughout) except in the variant that gives up 45% of the CAGR.
- **The protective rationale is void at the live dial**: fail-daily is already **0.0%** at 7% vol, so there is no floating-profit risk to insure against — you truncate the right tail for nothing.
- **It does not rescue high-vol sprinting either** (FLEX rules, where 43–71% of failures ARE daily breaches): 10% dial 56.1% → 50.6/50.9/57.6; 14% 41.0 → 33.7/37.0/43.1; 18% 26.8 → 24.7/25.4/24.8. Tighter takes make **fail-daily WORSE** (43.0→47.9%) because a slower equity curve means more days exposed, i.e. more chances to catch a bad one. Only very loose settings (take 3%, 15d wait) are within noise of baseline — and those barely leave the market.
- **Verdict: do not add profit-taking or dip re-entry.** Trend books earn from a few large sustained winners; capping them removes the tail that funds all the small losses. Caveat: modelled at daily resolution, so a real intraday +1.5% trigger is approximated by exiting at that day's close — but the effect is large and monotone, so the conclusion holds.

### 3f. Multi-timeframe sweep M30/H1/H4 x speed, at REAL tight spreads (2026-07-20, `scripts/v5_multi_tf_trend.py`)
Re-opened the fast-trend question now that tight-spread gold is reachable (**FundingPips XAUUSDmicro $0.12** measured — the tier where §3c said fast trend *becomes* viable; FTMO XAUUSD $0.45; cent $0.36). Discrete engine, long-only champion recipe, eval 2017+.

Net Sharpe @ $0.12 (best case for fast configs):

| TF | ultra | vfast | fast | med | slow | trades/mo | maxDD |
|---|---|---|---|---|---|---|---|
| M30 | 0.68 | 0.88 | 0.93 | 0.87 | 0.87 | ~20 | −27 to −45% |
| H1 | 0.59 | 0.65 | 0.82 | 0.94 | 0.98 | ~11 | −21 to −33% |
| **H4** | 0.59 | 1.03 | 1.09 | **1.27** | 1.05 | ~3.3 | **−9 to −15%** |

- **"Sensitive to trend changes" is WORSE, universally.** `ultra` is the worst config at EVERY timeframe and EVERY spread (0.59–0.68). Faster reaction buys noise, not earlier trend detection.
- **H4 beats M30/H1 even at $0.12** (1.27 vs 0.93/0.98) with 1/3 the drawdown. So intraday is not merely cost-limited — the signal itself is worse. Tight spreads do NOT rescue intraday trend; §3c's "deployable at $0.12" was about a *single* fast config, and H4 still dominates it.
- **Spread sensitivity tracks turnover**: M30 (20 tr/mo) 0.93→0.47 as spread goes $0.12→$0.45; H4 (3.3 tr/mo) 1.09→1.01 (near-immune, consistent with the champion's known spread-invariance).
- **APPARENT WIN THAT FAILED VALIDATION — do not adopt.** `H4/med` scored 1.27 vs champion `H4/slow` 1.05, with better DD (−9.5%) and worst-year (−0.46). But (a) **split-sample: med 0.89 vs slow 0.96 in 2017-2020** — the entire edge is 2021+ (1.53 vs 1.13), i.e. regime-specific; (b) **sharp parameter peak**: trail 2.0/2.5/**3.0**/3.5/4.0 → 0.82/1.02/**1.27**/1.14/1.02, a robust edge has a flat surface; (c) selected from 45 swept configs (multiple testing). **Verdict: keep the champion speed set. The H4 slow champion remains the right choice.**

### 3g. FTMO vol-dial + 4th-sleeve instrument search (2026-07-20, `scripts/v5_instrument_search.py`, `v5_book_speed_sweep.py`)
**Book-level SPEED is exhausted — the curve is FLAT.** Running the whole 3-asset book 3x faster changes the median finish by 0.2mo (FTMO 13.2 vs 13.4mo, pass 98.2 vs 98.6). Flat (not peaked) = the live speed choice is robust, and no speed tuning will make the challenge finish sooner.

**FTMO VOL DIAL — 9% is the efficient point, NOT 10%** (FTMO rules, XAU+BTC+NDX):

| dial | pass% | fail-daily | median | note |
|---|---|---|---|---|
| 7% | 98.8 | 0.0 | 13.5mo | LIVE |
| 8% | 97.9 | 0.0 | 11.6mo | |
| **9%** | **96.4** | **0.0** | **10.3mo** | **−3.2mo for −2.4pp — best trade** |
| 10% | 87.4 | **8.1** | 8.6mo | daily limit starts binding — cliff |
| 12% | 65.6 | 29.9 | 6.2mo | |

The 5%-daily line is never approached until 10%; that is where fail-daily jumps 0→8.1% and pass drops 11pp for only 1.7 more months. **Do not go past 9%.**

**INSTRUMENT SEARCH — no 4th sleeve clears the bar.** Screened all 24 FTMO-tradeable instruments we hold D1 history for (FX excluded, dead post-2016) on standalone SR + correlation, then tested each as an equal-risk 4th sleeve. Screen correctly rejected the redundant ones: SPX (r=0.86 to NDX), ETH (0.64 to BTC), SILVER (0.58 to XAU), DJI (0.60 to NDX). Six candidates passed to stage 2:

| book | Sharpe | maxDD | pass% | median | 17-20 | 21-26 |
|---|---|---|---|---|---|---|
| **BASE XAU+BTC+NDX** | **1.71** | −12.3% | **98.6** | **13.4mo** | +2.28 | +1.28 |
| + BRENT (UKOIL) | 1.68 | **−9.7%** | **99.1** | 13.8mo | +2.14 | +1.34 |
| + NIKKEI (JP225) | **1.80** | −13.6% | 97.9 | **12.8mo** | +2.27 | +1.45 |
| + SOL (SOLUSD) | 1.74 | −12.9% | 98.2 | 13.1mo | +2.26 | +1.34 |
| + PALL / + DAX / + LTC | 1.62/1.52/1.41 | | 98.8/97.1/84.8 | slower | | |

- **Nothing satisfies "raise pass% AND cut median AND hold in both half-samples."** BRENT raises pass (+0.5pp) and cuts drawdown materially (−12.3%→−9.7%, and it is the LEAST correlated asset found: max r=0.05) but is 0.4mo slower. NIKKEI is fastest (−0.6mo) and highest Sharpe but costs 0.7pp pass and deepens DD. All differences are small/within noise.
- **Verdict: keep the 3-asset book.** It is already near-optimal; a 4th sleeve adds execution surface for no reliable gain. BRENT is the only one worth revisiting *if* the goal shifts from speed to drawdown reduction.
- **The ONLY reliable speed lever left is the vol dial (7%→9%, −3.2mo).**
- ⚠ **CORRECTION (2026-07-24):** this script uses raw `vbc._load_asset`, so its
  **commodity and agriculture rows are priced 3–50× too cheap** (see MANDATORY
  CONTROLS §1) — PALL in particular is not a real candidate. The index/crypto/metal
  rows are unaffected (already conservative), so the **BRENT and NIKKEI conclusions
  survive** and were independently reproduced at honest cost in §3i. Back-port the
  live-cost floor before re-running this file.

### 3h. FundingPips 10K — XAU sleeve went CLOSE-ONLY, and no replacement exists (2026-07-24, `scripts/v5_fp_xau_replacement.py`)
FundingPips flipped **XAUUSDmicro to `trade_mode=3` (CLOSEONLY)** between 2026-07-20 and 07-23 17:00 UTC, without notice. After a manual close the sleeve could never reopen: every hourly pass logged one line `ORDER REJECTED: retcode=10044` (TRADE_RETCODE_CLOSE_ONLY) and otherwise looked healthy, so the book silently ran **2 of 3 sleeves for ~14h**. Searched the entire 43-symbol FP universe for a substitute — **nothing qualifies**, and the reason is instructive:

- **The universe splits exactly along the size line, with no overlap.** Everything that FITS min-lot at $10k is FX or a redundant equity index (FX dead as always; STX50 +0.18 SR and FTSE100 +0.08, both correlated 0.40/0.32 to the DJI already held). Everything with a real diversifying edge is **structurally oversized**: SPX 1.58×, BTC 2.52×, BRENT 3.92×, NIKKEI 4.30×, WTI 5.05×, NDX 7.11×, DAX 9.88×, **GOLD (XAUUSD) 17.69×**, SILVER 21.81×.
- SPX500 is the only near-miss on size but is **0.838 correlated with DJI30** — same factor, fails diversification anyway.
- **Cost of losing the sleeve:** SR 1.41→1.04, pass 99.0%→96.2%, median 17.4→22.2mo, and the 2021-26 half thins to +0.49.
- **Account size is the lever and it is a big step.** Target leverage is scale-invariant while target notional scales with equity, so full-size XAUUSD needs **~$118k** to size correctly in this book (silver ~$145k, NDX ~$47k, BTC ~$17k). This is exactly why the 100K FTMO book trades plain XAUUSD and the 10K one cannot.
- **Verdict: the 10K book is SIZE-constrained, not idea-constrained. Do not re-run this search.** Wait for the broker to restore micro (the executor now detects `trade_mode` before sending and the daily report has a BROKER RESTRICTIONS section), or accept the 2-sleeve book.

### 3i. FTMO 100K — XAU alone, and a diversifier rebuild at HONEST cost (2026-07-24, `scripts/v5_ftmo_xau_diversifiers.py`)
Grew the book one sleeve at a time from XAU alone over the **real 167-symbol FTMO universe** (verified live). Sizing does not bind here — index CFDs have contract size 1.0, so min-lots are $62–$519 (~0.1–0.5% of equity) — so the search is purely about edge and correlation. **All costs floored at live quotes**, which changed the answer (see MANDATORY CONTROLS §1).

| book | Sharpe | maxDD | pass% | median | 17-20 | 21-26 |
|---|---|---|---|---|---|---|
| XAU alone | +1.12 | −10.2% | 95.4 | 19.2mo | +1.02 | +1.20 |
| XAU+NDX | +1.51 | −9.3% | 99.1 | 15.2mo | +1.76 | +1.34 |
| **XAU+NDX+BRENT** | +1.44 | **−7.7%** | **99.2** | 16.0mo | +1.54 | +1.36 |
| XAU+BTC+NDX (champion) | **+1.71** | −12.3% | 98.6 | **13.4mo** | +2.28 | +1.28 |

- **XAU alone IS deployable** on FTMO 100K (target notional $9,846 vs $4,048 min-lot = 0.41×, fits easily) — 95.4% pass on its own.
- **NDX, not BTC, is the best single partner for XAU** (99.1% vs 96.6% pass, −9.3% vs −13.7% DD).
- **BRENT is the only genuine non-equity/non-crypto diversifier** that survives honest pricing (corr +0.05 to XAU, SR +0.42 at real 7.58bp, min-lot $95). Independently reproduces §3g's finding with correct costs.
- **Sanity check that the sim is working:** stacking SPX or DJI *on top of* NDX craters pass% to ~85% despite fine Sharpe — all one US-equity factor, so the book concentrates and daily-loss failures spike.
- **Champion caveat for the FUNDED stage, not the challenge:** its −12.3% maxDD is measured over the full 9-year path and exceeds FTMO's 10% STATIC max loss (guard halts at 8%). Challenges are short so pass% stays 98.6, but for sustained funded trading XAU+NDX+BRENT at −7.7% is materially safer, and its half-samples are far more even (+1.54/+1.36 vs +2.28/+1.28 — the champion leans on BTC's 2017-20 era).

### 3j. Weekly Mon→Fri XAU cycle + profit targets — DISPROVEN (2026-07-24, `scripts/v5_xau_weekly_cycle.py`)
Extends §3d from "what does a weekend ban cost the basket" to "can a purpose-built Monday-entry / ≤5-day XAU cycle work instead". D1 gold 2008-2026, ~950 cycles, entry at Monday's OPEN from Friday's signal (strictly causal), costs floored at live FTMO quotes and charged both ways (~52 round trips/yr).

| variant (10% vol) | netSR | maxDD | 1st half | 2nd half |
|---|---|---|---|---|
| ALWAYS-IN D1 (benchmark) | **+0.64** | −22.4% | +0.36 | +0.83 |
| Mon→Thu (hold ≤4d) | +0.44 | −26.5% | +0.01 | +0.73 |
| Mon→Fri (hold ≤5d) | +0.37 | −33.5% | +0.17 | +0.51 |
| weekend-flat, daily resize | +0.35 | −28.7% | +0.07 | +0.55 |
| hold ≤1d | **−0.17** | −34.6% | −0.16 | −0.19 |

- **The Monday restriction is NOT the cost — refusing weekend risk is.** Weekend-flat with full daily resizing (+0.35) ≈ Monday-frozen weekly size (+0.37), both ~45% below always-in. The gold trend edge genuinely lives in weekend/gap exposure; no intraweek cleverness recovers it. (Portfolio-level confirmation in §3k: +1.03 → +0.61.)
- **Exit Thursday, not Friday** (+0.44 vs +0.37, −26.5% vs −33.5% DD). **1-day holds lose money** — you need 3–4 days to clear the round trip.
- **PROFIT TARGETS: only wide ones help, and there is a trap.** At hold ≤5d: TP 0.50% gives a **74.7% win rate and LOSES money** (−0.25 SR, −43.6% DD) — it clips every winner while losers run to Friday. Anything optimising win% on this book optimises the wrong number. Best is **TP 1.5% move**: SR +0.37→+0.50, maxDD −33.5%→−23.4%, FTMO pass 43.5%→51.0%. Percent targets beat ATR targets at the good end (1.5% = +0.50 vs 1.5×ATR = +0.42); the 1.5–3% plateau is what a robust parameter looks like.
- **Challenge viability: no.** Best variant passes **FTMO 51.0% / Flex 31.5%, median 25.6mo** — a coin flip, against 98.6%/99.2% for the diversified books.
- **Post-2022 looks great and is a mirage** — see MANDATORY CONTROLS §2 and §4. Do not retry XAU-alone here.

### 3k. Cushion engine — a hard 5% DD mandate is INFEASIBLE (2026-07-24, `scripts/v5_cushion_engine.py`)
Attacked a **different objective**: maximise return subject to a hard 5% floor, weekend-flat, 100K — not a Sharpe problem. Used all three levers a floor-constrained objective allows: **breadth** (28 markets), **positive skew** (discrete breakout + ATR stop), and **path-dependent leverage** (CPPI sizing off the CUSHION = equity − floor, not off equity). Weekend-flat is load-bearing here rather than a tax: it removes the gap risk that is CPPI's classic failure mode.

**The frontier — what return is actually buyable at each cap** (walk-forward universe, honest costs):

| DD cap | best CAGR | via | Calmar |
|---|---|---|---|
| 3.0% | 0.26% | vol-target 1% | 0.11 |
| **5.0%** | **0.50%** | vol-target 2% | 0.11 |
| 7.5% | 0.74% | vol-target 3% | 0.11 |
| 15.0% | 1.18% | vol-target 5% | 0.11 |

- **The binding constraint is CALMAR and it is structural.** Nothing in this repo exceeds Calmar ~1.3 even with full hindsight (XAU+NDX+BRENT: SR 1.44 at −7.7%). "High return" (~20%) at 5% DD needs **Calmar 4** — 3× to 30× beyond anything measured. **Quote the frontier; do not chase it.**
- **CPPI DOES enforce the floor:** maxDD −1.99% → −4.97% monotonically in the multiplier, never once breaching 5% over 13.2y. The architecture works.
- **But CPPI LOSES to plain low vol-targeting at matched DD** (0.21% vs 0.50% CAGR at the 5% cap). Classic **cash-lock** — the ratcheting floor de-risks after losses and cannot re-lever on recovery. Do not reach for CPPI again.
- **Discrete breakout + hard stop + Friday exit loses money** (−0.3% CAGR, 50.2% win rate; −11% sized off equity). Breakouts need weeks; a 2×ATR trail plus a weekly flat cuts every winner. Independently confirms §3j.
- **Breadth is not free.** Equal-risk over all 22 names = Sharpe 0.37; the ags and weak indices have ~zero IC and add noise, not breadth. The Fundamental Law only pays for POSITIVE-IC bets.
- **Verdict: if a 5% DD mandate is real, the honest options are ~1–2% CAGR, or relax the cap to ~10–12% where the existing books already live. No configuration delivers both.**

### 3l. Martingale "gambling" bot on the direction predictor — the spec RUINS; only a gentle progression survives (2026-07-24, `scripts/v5_martingale_predictor.py`)
User spec: martingale from the smallest lot, recover losses, reset on win, steered by the turning-point ML bottom detector (§3), on 100K with a **hard 10% wall** ($90k floor). Objective = return, not Sharpe. Reuses `v5_xau_fade_martingale.simulate` (flat/double4/recover4 + $-ruin model) and `v5_xau_turning_ml` features; costs floored at the live gold quote; block-bootstrap MC preserves loss clustering. Engine wiring sanity-checked (synthetic p=0.55 → survive, p=0.40 → 100% bust).

**Reconfirms and refines §2's law: a martingale is an edge LEVER, and AGGRESSIVENESS is the ruin dial.** The whole outcome reduces to Stage A — does the predictor's directional bet clear breakeven `p* = 1/(1+b)` after real spread?

- **All flags (proba≥0.5):** p = 43.5% vs p* = 43.7% = **dead breakeven**, flat P&L −3.5%. Every martingale ruins (double4 27%, recover4 95–99%). Matches the "precision ≠ P&L, lost to buy&hold" finding in §3.
- **High-conviction only (proba≥0.7, ~378 bets/2.5y, stop 1.5×ATR, target 1.5×stop):** p = 44.2% vs p* = 41.7% = **+2.5 pts, half-sample STABLE** (+2.7/+2.4), zero ≥8 loss-streaks — but only **~1 SE** (SE 2.6) and flat P&L +11.3% still loses ~12× to **buy&hold gold +130.8%**. (Rich-target rr=3.0 cells looked better but DECAY across halves +6.8→+0.6 — overfit, control caught them.)

Ruin frontier ON the +edge cell (500 block-bootstrap paths):

| engine | base$ | K | P(ruin) | medRet | maxDD |
|---|---|---|---|---|---|
| flat | 40k | any | 0.0% | +4.6% | −4.2% |
| double4 | 20k | 3 | 0.2% | +6.9% | −4.3% |
| double4 | 40k | 3 | 7.0% | +13.8% | −8.0% |
| double4 | 20k | 5 | 44.4% | −2.7% | −12.5% |
| **recover4 (the spec)** | 20k | 3 | **71.0%** | **−10.0%** | −20% |
| recover4 | 40k | 4 | 91.0% | −10.0% (p95 +253%) | −19% |

- **The user's exact mechanic — deficit-targeted "recover ALL losses" (recover4) — RUINS 71–97% even WITH the positive edge.** Recovering the whole deficit in one bet needs stakes a 5–6-loss cluster (which occurs) drives through the floor; return profile is a lottery (median −10% dead, p95 +250–580%). **Do not ship.**
- **Only a GENTLE progression survives:** classic doubling capped at **K=3** with a **small base** ($20–40k notional) = ~0–7% ruin, +7–14% median, maxDD <8%. K≥5 or base $80k → 44–89% ruin regardless of engine. Escalation depth and base ARE the ruin dial.
- **All of it rests on a ~1-SE edge that loses to buy&hold.** If the true edge is the thr=0.5 breakeven, the gentle cells revert to ruin too. **Verdict: not deployable.** The detector's honest use remains a long-entry FILTER, never a standalone or a martingale base.
- **INTRADAY 1:2R variant (2026-07-27, same tool `--tf M15/M30/H1 --rr 2.0`) — FAILS harder, do not build.** The gate degrades monotonically as the bar shrinks: M15 edge **−3 to −4.4 pts**, flat P&L **−40% to −104%** (every config); M30 +0.7 (±1.3 SE = noise, +9.6% but loses to buy&hold +114%); H1 ~+1.4 (~1 SE). Cause: the fixed ~$0.85 round-trip cost is a huge fraction of the shrinking intraday target (M15 target ≈ a few $ → spread eats 15–25%/trade; H4 ≈ $30 → ~3%) — same mechanism as §1's fade dying net of spread. 1:2R needs p*≈37%; detector nets ~34–37% intraday = at/below breakeven, and loss streaks cluster harder (≥8 losses 13× at M30 vs 0× at H1). recover4 on the least-bad cell ruins 89–99%. **Third confirmation the intraday XAU edge dies net of spread; the predictor's edge lives at swing/H4 scale only. Do not retry sub-H1 martingales.**

### 3m. Prop-firm account selection + two-stage weekend-flat plan (2026-07-27, `configs/v5_fundingpips_funded.json`, `scripts/v5_cushion_engine.asset_stream`)
Chose between three $100K prop plans for the deployed book. **All three restrict weekend holding** (evaluation AND master), so every arm is measured weekend-flat (live-cost floored, `asset_stream(weekend_flat=True)`).

| Plan | P1/P2 | Max loss | Daily | Split | Price | Verdict |
|---|---|---|---|---|---|---|
| A (FP 2-Step Std 8/5) | 8/5 | 10% | 5% | 80% | $522 | **reject** — *striking system* on master (warning at 1.2% floating loss/idea → split halves → 20% → breach) shreds a floating-position book; news restricted; lowest split, highest price |
| **B (10/6)** | 10/6 | **12%** | 4% | **95%** | **$499** | **BUY** — most DD room, best split, cheapest, no striking system |
| C (6/6) | 6/6 | 6% | 3% | 80% | $422 | **reject** — 6% max < the book's natural DD (the [3k] infeasibility); forces ~4% vol → slowest, lowest pay |

**Two-stage book split (the key finding — the stages want DIFFERENT books):**
- **CHALLENGE = XAU+BTC+NDX+BRENT weekend-flat @ ~7%.** You must HIT +10%/+6%, so BTC's return is required: with BTC pass 99.9%@6% / 87%@7%; without BTC (SR 1.09) it maxes at **80%** pass and grinds 25mo (too weak to reach the target, and the extra days rack up 4%-daily breaches). KEEP BTC.
- **FUNDED = XAU+NDX+BRENT weekend-flat @ 6%.** No target, just don't breach — so BTC's 24/7 weekend-gap risk is pure downside; drop it (§3d). @6%: −6.3% path DD (vs 12% limit), worst-day×1.5 −3.94% (0 breaches of the 4% daily line in 9.5y), ~$550/mo gross → **~$523/mo net @95%**. Config `configs/v5_fundingpips_funded.json` (magic 360563, vol 0.06).

**Honest caveats:** (1) B's **4% daily line is punishing weekend-flat** — Monday gaps mean pushing vol for speed spikes fail-daily fast, so the challenge takes **~15–20mo** to pass safely (vs ~10mo if weekends were allowed). (2) Two build gaps before live: the executor has **no Friday-flatten** feature yet, and the FP 100K symbol names (`XAUUSD`/`NDX100`/`UKOIL`) need live verification. Both flagged in the config; deferred until the account is bought & passed.

### 3n. Claude (LLM) as an H4 XAU analyst — NO tradeable edge (2026-07-27, `scripts/v5_claude_h4_backtest.py`)
Measure-first backtest of the Lumibot-style "AI Agent Strategy" the repo already scaffolds (`src/bots/ai_bot.py`, `src/v5/agents.py`) before any live 100K wiring. Feeds each closed H4 gold bar's metrics (price, RSI, SMA20/50, ATR, range, 10 recent candles) to **claude-opus-5** (adaptive thinking, structured output, prompt+response cached), gets `{action, confidence, reason}`, simulates net of the real $0.45 FTMO spread. 150 bars ≈ 1 month.

**Result: no edge, don't deploy.** 97 hold / 34 sell / 19 buy; confidence 97 medium / **0 high** (Claude never had high conviction on choppy gold — honest). Gated (medium+) traded bars 12, **hit rate 50.0%** (coin flip). **Net −2.07% vs buy&hold −1.25%** — lost AND underperformed inaction. All 53 raw directional calls hit 54.7% but that's within 1 SE (±6.9%) of 50% and dies to the spread; the confidence signal was useless (medium calls did *worse* than low). An LLM has no structural edge reading gold direction and pays the same spread wall as §1/§3l — **4th independent confirmation** the intraday-to-H4 XAU direction edge is uneconomic, now with a frontier LLM. Coherent narration ≠ predictive edge. The measure-first harness cost ~$2-4 on paper, zero capital risked — the correct way to vet any "let the LLM trade" idea. Do not re-run expecting a different sign.

**Overlay/veto framing also fails** (`scripts/v5_claude_overlay.py`, free — reuses the cached decisions). Claude as a VETO/downsize filter on the trend book (never opens its own trade): on the ls long/short base (active all 150 bars) it took base **+0.54% → VETO −1.96%**, downsize −0.71%. Claude opposed the trend on only 4 med+ bars and the base MADE +2.46% there (75% winning bars) — the vetoes **clipped winners**. A 50%-hit signal used as a veto just removes good trades. **All three LLM-directional framings (standalone, veto, downsize) add no value — close the thread.** Remaining honest LLM uses are non-trading: trade review/journaling, or macro/news synthesis for human decisions.

### 3o. SMC / supply-demand (break-of-structure) on XAU H4 — looked promising, FAILED walk-forward (2026-07-28, `scripts/v5_smc_xau.py`)
Smart-Money-Concepts engine (causal fractal swings; modes = liquidity **sweep**, **BOS** break-of-structure, **sr_bounce** zone reaction) with an OOS-honest grid. A textbook case of why the MANDATORY CONTROLS exist.

- **Single 70/30 split looked great:** BOS-long test Sharpe **1.44** at 38% exposure, 0.61 corr to the champion, grid train-test corr +0.67. But the test half (2023-2026) was a pure gold bull, so the number was **bull-beta, not skill**.
- **Walk-forward (re-select best config each year on past-only data, 2018-2026): Sharpe +0.54, net +62%, maxDD −14% — vs buy&hold +209%.** Underperforms passive gold ~3.4×. Made money only in the strong bull years (2020 +28%, 2023 +9%, 2025 +36%); lost/flat in every other year and **missed the +27% 2024 bull entirely (−1%)**.
- **Regime split (smoking gun):** BOS-long Sharpe by regime — 2015-18 chop +0.44, 2019-20 bull +1.34, **2021-22 FLAT −0.39 (loses)**, 2023-26 bull +1.51. A breakout/trend strategy that harvests uptrends and **bleeds when gold is flat** = bull-beta, same family as the champion (which is itself **0.96 correlated with buy&hold**), but worse than holding gold.
- **H1/intraday version already loses** (median test Sharpe −0.49 — turnover×spread).
- **Verdict: not deployable, not a new sleeve.** Real trend/breakout STRUCTURE (train-test corr +0.67 is genuine) but no regime-robust EDGE. Same trap as §3j (gold beta in a bull window). See [[smc-bos-h4-promising]]. Grids: `data/v5_runs/smc_grid_{H4,H1}.csv`.

### 3p. Top/bottom PROBABILITY -> sizing, re-asked with an ORACLE CEILING first (2026-07-29, `scripts/v5_xau_turn_prob.py`)

Re-ran the "detect tops/bottoms with probability, size by it" program with a new
mandatory pre-flight — **Stage 0: trade the label with PERFECT HINDSIGHT and measure
what it is worth before building any detector.** This reversed the program's premise.

**THE ORACLE CEILING (perfect hindsight; champion eval 1.041, buy&hold-vt 0.927):**

| perfect knowledge of | dSharpe vs champ | paired t | yrs+ |
|---|---|---|---|
| geometric **BOTTOMS** boost | +0.017 | 2.01 | 7/10 |
| geometric **BOTTOMS** standalone long | **-0.675** | -3.31 | - |
| geometric **TOPS** trim | +0.259 | 3.83 | 9/10 |
| downside-quantile k24 trim | +0.697 | 10.8 | 10/10 |
| champ-meta fwd-12bar P&L<0 trim | +0.824 | 11.7 | 10/10 |

**§3 optimized the worthless side.** The champion is long-only and near-always-in, so
it is ALREADY LONG at bottoms — perfect bottom knowledge is worth +0.017 and loses
0.675 standalone. All value is in **trimming before bad forward windows**. Plugging
§3's measured top spec (prec 0.70, recall 0.10) into the overlay gives ~0.99, i.e.
**worse than the champion — reproducing §3's overlay failure from the spec alone.**
Also: the "turnover explosion" §3 blamed was a SYMPTOM — the oracle runs 123
turnover/yr and still scores 1.86. **Recall was the disease.** Attainability frontiers
survive live $0.448 cost unchanged; downside-quantile clears +0.20 even at
precision 0.60 / recall 0.20.

**HONEST MODELS (walk-forward, expanding yearly refits from 2018, purge 8 bars + label
window, isotonic calibration on a PURGED train tail, `hour` feature dropped):**
all three label families **FAIL both pre-registered gates.**

| label | mean AUC | full IC (gate) | dSR b=1.0 | paired t | yrs+ | vs shuffled-p p95 |
|---|---|---|---|---|---|---|
| downside-quantile k24 | 0.537 | -0.037 (0.052) | +0.024 | **-2.93** | 2/9 | fails |
| economic fwd-24 | 0.531 | -0.009 (0.062) | +0.023 | **-2.94** | 2/9 | fails |
| **champ-meta k24** | **0.593** | -0.026 (0.052) | **+0.250** | **-2.11** | **3/9** | **1.338 > 1.155 PASSES** |

**champ-meta carries REAL information but it is a RISK CONTROL, not alpha.** AUC
0.59-0.62 (0.73 in 2021) is genuinely above chance, and it beats BOTH nulls — the
block-shuffled-p null (1.338 vs p95 1.155) and, decisively, the **matched-constant
control**: at the same average exposure, `const x0.62` gives SR 1.079 / CAGR 11.4% /
DD -10.6% / Calmar 1.08 while the timed trim gives **SR 1.338 / CAGR 12.5% / DD -9.4%
/ Calmar 1.33**. So the *timing of where to de-risk* is skillful.

But it **buys safety with return**: CAGR 18.7% -> 12.5% (-6.2pp) while maxDD 16.9% ->
9.4% (-44%), paired t **-2.11** (significantly negative), and it works in only **3 of 9
years**. Identical at live $0.448. That is the §4 law again — *probability sizing cuts
drawdown, adds no return* — but this time with the mechanism measured: the information
is real and is spent entirely on risk reduction.

**VERDICT: NOT DEPLOYED.** Fails Stage 1 (information) and Stage 2 (paired t >= 2.0,
>= 7/9 years) as pre-registered. Do not re-run bottom detectors of any kind. The one
legitimate follow-up is narrow: **as a DRAWDOWN control** (not an alpha overlay) on a
book where DD headroom is the binding prop-firm constraint — but 3/9-year robustness
must be explained first.

**Reusable deliverable: `--approach 1` is now the mandatory Stage-0 oracle screen.**
Before building a detector for any new label L, trade L with hindsight and report
(a) standalone SR, (b) paired dSR + t vs the incumbent, (c) the (precision, recall)
contour reaching dSR +0.20. If the oracle dSR < +0.20 or oracle t < 2.5, **do not build
the detector.** This is MANDATORY CONTROL #5.

### 3q. Deep book-improvement study — 8 approaches, ALL DEAD; and the FINANCING correction (2026-08-08, workflow wf_dce6ad0a-25b, 18 agents)

Attacked the deployed 4-asset FTMO book (XAU+BTC+NDX+BRENT @9%) from eight angles. Every
agent reproduced the baseline to 7 decimals (Sharpe 1.4640761, CAGR 13.68%/14.20%, maxDD
-12.69%, n=2229) before testing. **Nothing survived. All 9 adversarial verifications refuted.**

| approach | dSharpe | paired t | yrs better | beat shuffled null | trials | median trial |
|---|---|---|---|---|---|---|
| defensive regime overlay | +0.1047 | +2.02 | 9/9 | yes | 225 | -0.0913 |
| sizing / signal shape | +0.0766 | +0.77 | 6/9 | no | 339 | +0.0070 |
| correlation-aware sizing | +0.0461 | +1.45 | 7/9 | no | 103 | +0.0179 |
| carry (swap-based) | +0.0246 | +0.23 | 5/9 | no | 93 | -0.0543 |
| trend-speed ladder | +0.0215 | +0.63 | 6/9 | no | 94 | -0.1074 |
| cross-sectional momentum | -0.0283 | -1.51 | 3/9 | no | 960 | -0.0963 |
| profit-capture / re-entry | -0.0497 | -1.78 | 2/9 | no | 104 | +0.0117 |
| new diversifiers (70 cands) | -0.1467 | -0.36 | 1/9 | no | 6615 | -0.1688 |

**The defensive overlay was the only one to clear the gates — and it was the cleanest
selection-bias trap in this project.** It cut the book to zero when cross-sectional trend
agreement was in its bottom quintile: dSharpe +0.1047, t 2.02, 9/9 years. Refuted because
the brief PRE-ANNOUNCED that 2018 and 2022 were the losing years, and a 225-way search
duly found the overlay that fixes exactly those two. Independent re-tests: **-0.096, -0.010,
-0.095**. Median of its own 225 trials was -0.0913 and ZERO trials passed all four gates.
*Lesson: naming the target years in the prompt manufactured the result. Do not brief a
search with the answer you hope to find.*

Other notable disproofs:
- **Carry is NOT unexploited.** BRENT's +9.6%/yr is real but harvesting it always-long
  gives Sharpe 0.42 at 0.457 correlation to the book — it dilutes. And BTC's brutal
  -30.4%/yr is already neutralised: **vol-targeting has shrunk BTC's notional to 0.055 of
  equity**, so its total bill is only -1.68%/yr. A permutation null over all 4! carry-to-asset
  assignments put the TRUE mapping at the 12.5th percentile — worse than random relabelling.
- **XS-momentum did NOT replicate** (-0.028, t -1.51, 3/9 yrs) across 960 trials, despite the
  earlier standalone Sharpe 0.74 / corr 0.15 result. It does not survive as a book sleeve.
- **Profit-capture/re-entry fails** (-0.050, t -1.78, 2/9) — dies on turnover, as the prior said.
- **70 candidate diversifiers, 6615 trials, median -0.169.** Breadth still fails. Bonds
  (UST 2/5/10/30Y) did not rescue it.

**THE REAL FINDING — FINANCING IS -3.79%/yr, NOT -2.59%.** Measured per sleeve on live FTMO
swap rates against each sleeve's actual vol-targeted notional:

| sleeve | mean notional | rate | drag |
|---|---|---|---|
| XAUCHAMP | 0.245 | -6.6%/yr | -1.615%/yr |
| BTC | 0.055 | -30.4%/yr | -1.677%/yr |
| NDX | 0.215 | -6.9%/yr | -1.485%/yr |
| BRENT | 0.103 | **+9.6%/yr** | **+0.991%/yr** |
| **book** | | | **-3.785%/yr** |

**Impact: Sharpe 1.4641 -> 1.1763, CAGR +14.20% -> +11.16%, maxDD -12.69% -> -13.36%.**
FTMO 2-step pass net of financing: 95.0%/17.4mo @7%, **90.4%/12.9mo @9%**, 65.9%/8.8mo @11%
(previously quoted 94.6%/11.0mo @9% GROSS). Monthly on $100k net: mean $916, median $587,
59% of months positive, worst -$5,212.

**VERDICT: keep the book byte-identical. It is at its frontier for this signal family.**
Financing is now the single largest lever in the book (-0.29 Sharpe) and the only untested
way to move it is EXECUTION (a venue with cheaper overnight rates), not signal research.

### 3r. XAU H4 fwd-direction classifier (regime features) — ONE real accuracy edge found, FIVE trade structures ALL fail to monetise it (2026-08-10/11, `scripts/v5_xau_intermarket_accuracy.py`, `v5_xau_fwd6_regime_pnl.py`, `v5_xau_multihorizon_pnl.py`, `v5_xau_bracket_gridsearch.py`)

Re-opened §3's turning-point question — "predict XAUUSD short trends, accuracy only,
explore new feature families" — because the user's own H4 zigzag trend follower motivated
another pass. **Read this whole entry before re-trying any of it; it is the fifth+ time
this exact class of question has been asked in this repo.**

**THE ONE POSITIVE RESULT (genuine, robustness-checked, the only thing worth reusing):**
price+regime features (Hurst exponent, ADX, Choppiness Index, short/long ATR ratio, a
daily-timeframe EMA-slope+z filter) fed to HistGBoost, walk-forward expanding yearly
2018-2026, predict XAUUSD H4's next ~1-trading-day direction (sign of the 6-bar-forward
return) at **52.0% accuracy vs a 49.8% PERSISTENCE baseline (not a coinflip) — +2.12
percentage points, positive in 8 of 9 years, block-bootstrap (block=60, 2000 draws) 90%
CI [+0.51pp, +3.69pp], P(edge<=0)=1.5%.** This is real, causal (no lookahead), and the
first result in this project's turning-point research to survive that level of scrutiny.
**It is a statement about predictability, not about profitability — see below.**

What did NOT add to it (so don't re-build these as features for this question):
- **Zigzag-in-progress state** (causal leg direction/age/extension/streak) — redundant
  with the z-score/momentum already used; adds noise.
- **Intermarket panel** (DXY proxy from the FX basket, UST10Y/2Y curve, SPX/NDX/DJI, BTC,
  silver/platinum/palladium, copper, oil, all correctly lagged 1 day) — no incremental
  signal despite real effort and a real alignment bug fixed along the way (holiday-
  calendar mismatches were silently corrupting ~27% of rows before the fix).
- **Multi-window Hurst, a hand-rolled causal HMM regime state, and rolling Haar wavelet
  spectral-energy features** — all layered on top of the winning regime recipe, tested
  across 5 model implementations (logreg/RandomForest/HistGBoost/XGBoost/LightGBM); none
  reliably beat the plain regime block. A naive Hurst-window sweep (30-500) was noisy and
  non-monotonic — picking a "best" window off it would be a look-elsewhere artifact.
- **SELL/tops detection stayed at ~0% recall @ 70% precision under every one of the above** —
  a 6th confirmation of §3's bottoms>tops asymmetry; none of the new families touched it.

**THE P&L VERDICT — accuracy != trading edge, proven FIVE separate ways:**
1. **Standalone** (probability-sized continuous position, live $0.448 cost): Sharpe
   **-0.44**, paired t vs buy&hold **-3.3**, vs champion **-3.6**. Turnover (217-623/yr, a
   ~1-day-horizon signal) burns through the thin edge faster than it pays.
2. **As a tilt/overlay on the incumbent long-only champion**: light blend is
   statistically indistinguishable from the champion alone (t +0.4); a stronger blend is
   worse (t -1.7). No value as a filter either.
3. **Confidence-gated (trade only the top-conviction subset, flat otherwise)**: accuracy
   DOES rise with confidence (up to 55.6% at the sparsest, most-extreme threshold), but the
   paired t-stat vs both benchmarks stays significantly NEGATIVE at every threshold tested
   (-2.5 to -3.7) — sitting flat ~98% of the time forfeits gold's own structural drift
   faster than the improved-but-still-small edge can repay. Not a sample-size artifact of
   turnover; a structural opportunity-cost problem.
4. **Longer horizons (2/4/8/16 trading days)**: the diagnosed mechanism (position size
   grows, drawdown improves, paired-t creeps toward zero as horizon lengthens) is real and
   directionally confirmed, but NONE of 1/2/4/8/16-day horizons cross into beating either
   benchmark. Pushing further converges on the champion's own EWMAC/breakout speed range —
   any eventual "win" there is more likely a worse reimplementation of the champion already
   deployed than a new edge.
5. **Discrete SL/TP bracket trades** (enter on signal direction, exit only via an
   open-ended take-profit or a small stop, no fixed time horizon — 80-cell grid, SL
   0.05-0.75%, TP 0.10-2.00% of price): **79 of 80 cells Sharpe-negative, 0 of 80 clear
   +0.3.** The single best cell (SL 0.50%/TP 2.00%, a 1:4 R:R needing only 20% win rate)
   scored Sharpe +0.015 (indistinguishable from zero) with a **-33.7% maxDD** and a
   textbook noise signature: -0.9/+0.2/-8.2/-7.6/-13.9% then +8.8/+11.5/+12.0/-3.1% —
   three bad years cancelled by three good ones, not a stable effect.

**VERDICT: do not build anything tradeable on this classifier.** Five structurally
different ways of turning a genuinely-measured 52%-accuracy signal into a position all
fail the same way, which is strong evidence the ceiling is the signal itself (barely above
persistence), not the wrapper around it. **Do not re-try:** more feature engineering on
this exact question (price/regime/zigzag-state/intermarket/HMM/wavelet all covered), a
different confidence threshold, a different horizon, or a different SL/TP grid — all
five of those knobs were turned already and none worked. The only thing that could change
the answer is a **materially different data type** (real order flow / tick-level
microstructure, or an actual historical news/sentiment feed — this repo's
`src/v5/news_filter.py` only caches the LIVE current-week ForexFactory calendar for
trade-blocking, not a backtestable archive). Full numeric detail, per-round: Claude memory
`xau-regime-features-fwd-accuracy`.

### 3s. Borrowed from Forven comparison: a lookahead probe (new) + this repo's own DSR/PBO (existing, finally used) + SuperTrend (2026-08-19, `src/evaluation/lookahead_probe.py`, `scripts/v5_xau_supertrend.py`)

Following `FORVEN-COMPARISON.md`, borrowed two validation techniques and one strategy idea,
implemented fresh in this repo's own style (not copied — Forven is AGPL).

**New: `src/evaluation/lookahead_probe.py`** — right-truncation-invariance probe (build a
synthetic OHLC panel, compute a feature/signal function on it and on right-truncated
copies, assert interior bars match). Ran against every feature function from §3r plus the
LIVE champion/ls signals — **all PASS**; `zigzag_swings` (the forward-looking TARGET
generator) correctly **FAILS**, confirming the probe can actually detect a real violation
and that this repo's causal/target boundary has held throughout. Gotcha: the champion's
slowest EWMA span is 1536 H4 bars (256 trading days x 6) — a synthetic panel shorter than
that "passes" by comparing nothing; always size the probe panel to the function's slowest
window.

**Existing but never applied to this session's grids: `src/evaluation/dsr_pbo.py`**
(Deflated Sharpe Ratio + PBO, already in this repo per `AGENT_INSTRUCTIONS.MD` Phase 6).
Applied properly for the first time this session:
- §3r's 80-cell SL/TP bracket search: **DSR = 0.0000** — formal confirmation it was noise.
  **PBO = 0.000 on the SAME grid** looks contradictory but isn't: PBO measures whether the
  IS-best config's RANK holds up OOS, and the ranking here is dominated by a monotonic cost
  gradient (tighter stop -> mechanically more stop-outs -> always worse, every sub-period)
  — trivially rank-stable even with zero real edge. **DSR and PBO answer different
  questions; a low PBO must not be read as "not overfit" when a grid's variation is driven
  by a structural cost gradient rather than a genuine edge gradient.**
- The new SuperTrend grid below: **DSR = 0.9948** — the contrast against 0.0000 on the same
  session's other grid is itself a useful demonstration that the tool discriminates.

**New strategy borrowed: SuperTrend** (re-implemented from the public ATR-band
trend-state-machine formula, not copied). Structurally different from the champion's
EWMA-crossover + smoothed-Donchian blend. Lookahead-probe: PASS. Standalone, long-only, H4,
live cost: **all 20 of 20 parameter cells tested had POSITIVE Sharpe** (ATR period
7-21 x multiplier 2.0-4.0) — remarkably consistent, unlike almost every other grid this
session. Best cell (period=21, mult=2.5): **SR +1.067, CAGR +10.1%, maxDD -11.1%** — Sharpe
comparable to and drawdown clearly better than the deployed champion (SR 1.084, DD -17.0%).
**Does not beat the champion in paired comparison** (t -2.21, 2/9 years better); as a
champion overlay it's a statistical wash (t -0.10 to -0.39). Correlation to the champion's
own XAU return stream: **0.77** — both ride the same structural gold trend via different
mechanics, so there is little diversification benefit combining them on the SAME asset.
Consistent with [[xau-longonly-champion]]'s "single-XAU ceiling ~1.06": a different
well-built long-only trend mechanism on the same asset lands near the same ceiling, not
above it.

**Verdict:** the validation tooling is a durable win (found nothing broken, and cleanly
told a real signal from a fake one in the same session). SuperTrend itself is genuinely
validated, real, NOT noise — unusual for this session — but not deployable as an XAU
replacement or overlay.

**Follow-up, same day (`scripts/v5_supertrend_basket.py`): ran the "diversify elsewhere"
idea — closes too, more decisively.** Applied SuperTrend to the second independent basket
(NIKKEI/COFFEE/ETH/DAX). All 4 standalone-positive, DSR 0.71-0.93 (the family is real),
but **PBO is high for 3 of 4** (NIKKEI 0.77, ETH 0.71, DAX 0.74) — the specific "best"
parameter cell is not stable across resampled sub-periods even though SuperTrend-in-general
beats the null (only COFFEE has both good DSR and low PBO, 0.056 — a genuinely robust
pick). **Correlation to the champ-recipe version of the SAME asset: 0.86-0.87 across all
four** — even higher than XAU's 0.77 — and the SuperTrend-basket vs champ-recipe-basket
built from the same 4 assets score nearly identical Sharpe (0.811 vs 0.814) at 0.880
correlation to each other. Blended with the flagship at the "sweet spot" weight, raw
Sharpe ticks up (1.426->1.457) exactly like the champ-recipe basket's own tilt test did —
but checked properly (paired-t + FTMO pass-sim, not just the raw number): **t=-1.60
(negative), pass-rate 96.7%->96.2% (flat to worse), CAGR falls.** Same
volatility-reduction-not-return mirage as before, just wearing a different signal.

**Follow-up #2, same day (`scripts/v5_supertrend_fx.py`): does SuperTrend revive FX,
which the champ-recipe already found "comprehensively dead post-2016" (`v5_instrument_
search.py`)?** FX has no long-run drift (no kill-the-shorts prior applies), so tested both
long-only and long/short across all 18 pairs (36 combos), 9-cell ATR-period x multiplier
grid each. **0 of 36 combos cleared SR>0.30 + DSR>0.90 + PBO<0.30 together.** The standout,
AUDJPY long-only (SR +0.55, 9/9 cells positive, PBO 0.08 — the pick IS rank-stable),
reported an encouraging within-grid DSR of 0.78 — but that number only deflated against
AUDJPY's OWN 9-cell grid, when the search that actually produced it scanned 36 pair x
direction combinations. **Recomputed DSR using the cross-sectional dispersion of the
best-per-combo Sharpe across all 36 as the trial variance (the correct, screen-level trial
count — the exact "swarm-level selection" correction Forven's own DSR implementation uses,
see `FORVEN-COMPARISON.md`): AUDJPY's properly-deflated DSR is 0.4278 — a coin flip.**
Also cost-stress-tested directly (D1 signals trade rarely, so turnover was never the
risk): survives 10x the CSV spread (SR barely moves, 0.554->0.454) — so this fails on pure
statistics, not cost, a cleaner kill than most of this session's closures.

**Lesson: DSR's trial count must reflect the ENTIRE search that produced the candidate,
not just the innermost grid.** A within-family DSR can look excellent and still be a coin
flip once the outer screen that selected that family is accounted for — any future
per-instrument grid nested inside a wider instrument screen needs both numbers reported,
and the screen-level one trusted.

**Closed, not open, on both fronts.** A different long-only trend-following MECHANISM on
a strongly-trending asset converges to the same net exposure the champ-recipe already
extracts (XAU, the second basket, the flagship blend); on assets where the champ-recipe
already found nothing (FX), SuperTrend finds nothing either, once honestly deflated. Don't
re-try SuperTrend, or by the same logic any other single-mechanism trend filter, on either
front. A genuinely different return stream needs a genuinely different EDGE SOURCE
(mean-reversion, cross-sectional, carry, a different risk premium entirely) — not another
way to detect the same trend.

### 3t. Cross-asset divergence mean-reversion — a genuine different edge source, not proven yet but NOT closed (2026-08-19, `scripts/v5_cross_asset_divergence.py`)

Took §3s's own conclusion seriously and tried the actual different-edge-source candidate:
Forven's cross-asset-divergence idea — fade the z-scored divergence between two
correlated assets' RETURNS, expecting reversion. 7 pairs, `src/evaluation/fitness.py`
(new: composite Sharpe/DD/DSR/correlation ranking score, since this repo's continuous
engines have no natural win-rate/PF/trade-count to reuse from Forven's version).

**Thin-CSV-cost screen was misleading — commodities pairs were cost mirages, again.**
GOLD/SILVER topped the ranking (SR +0.667, DSR 0.843) despite a prior disproof
([[gold-silver-spread-disproven]]) — but at the ALREADY-DOCUMENTED live cost floor
(8.5bp, not a stress test, MANDATORY CONTROL #1's own number): **SR +0.667 -> +0.382, DSR
0.843 -> 0.183, paired-t vs buy&hold-gold -2.16, only 2/9 years better.** PLAT/PALL: same
pattern, worse (PALL's 56.4bp floor drops SR +0.374 -> +0.020, DSR 0.636 -> **0.000**).
The prior disproof holds — reached again, via a different (simpler, unweighted) signal
construction, killed by a different mechanism (cost, not lack of cointegration) — good
confirmation the earlier verdict wasn't an artifact of that specific methodology.

**FX and equity pairs don't have the cost problem, but aren't confidently real either.**
EURUSD/GBPUSD survives a 2x cost stress (genuinely liquid, no CSV-understatement trap):
SR +0.365, DSR 0.589, PBO 0.313 — "more likely real than not," short of the >=0.90 bar.
SPX/NDX and DAX/STOXX need no cost correction (both on the "already conservative" list):
SPX/NDX SR +0.500, DSR 0.681, PBO 0.274 (best-balanced survivor); DAX/STOXX SR +0.390,
DSR 0.549, PBO **0.758** (parameter pick unstable, same pattern as §3s's SuperTrend PBO
problem on NIKKEI/ETH/DAX).

**The one genuinely new, encouraging number this whole line has produced: correlation to
the champion.** GOLD/SILVER's divergence signal correlates to the LIVE XAU champion at
just **+0.088** — the lowest correlation to the champion found all session (SuperTrend was
0.77 on the SAME asset). Confirms mean-reversion signals occupy real, different territory
from every trend-following variant tried in §3s — the commodities pairs' failure is a DATA
problem (understated CSV cost, the same recurring trap), not evidence mean-reversion can't
work here.

**Verdict: not proven, but genuinely not closed either — unlike SuperTrend.** No pair
clears DSR>=0.90 + PBO<0.30. Two honest next steps flagged; both run same day (see below).

**Follow-up, same day (`scripts/v5_hedge_ratio_divergence.py`): ran both flagged next
steps.**

**(a) Rolling hedge-ratio spread** (rolling-OLS beta shifted 1 bar, spread =
logA-beta*logB, z-scored, discrete entry/exit/stop state machine — closer to the ORIGINAL
gold-silver disproof's construction) on GOLD/SILVER, PLAT/PALL, SPX/NDX, tested straight at
honest/documented cost (no thin-CSV screening this time). **GOLD/SILVER meaningfully
improved**: SR +0.382 (simple, honest cost) -> **+0.719** (hedge-ratio, same honest cost);
paired-t vs buy&hold-gold improved from **-2.16 to -0.85** (no longer significantly worse
than holding gold, though not better either). Still not confident: properly deflated
against the FULL 81-cell search space (3 pairs x 27 cells — caught and fixed an
under-counted first attempt that used only the 3 pair-level bests, the exact mistake
flagged as a lesson in §3s), **DSR = 0.376**. Correlation to the champion held at +0.080.
**PLAT/PALL stayed dead** (DSR 0.048). **SPX/NDX got WORSE with the more complex
construction** (DSR 0.127 vs the simple version's 0.681, PBO 0.726 vs 0.274) — the
hedge-ratio machinery suits precious metals' cointegration-adjacent behaviour specifically,
not a generic upgrade.

**(b) Finer SPX/NDX grid** (simple divergence, no cost correction needed, 81 cells: window
10-100, threshold 1.0-3.0). Best cell (window=10, thr=2.0): SR +0.547, **60 of 81 cells
positive** (the family works broadly), DD only -9.4%. Properly deflated for all 81 cells:
DSR 0.523, PBO 0.452 — moderate, still short of confident. Paired-t vs buy&hold-SPX -1.39,
2/9 years better. **Correlation to the champion: -0.014 — the single best (most
orthogonal) diversification number found in this entire session.**

**Final verdict on this line:** every variant — simple and hedge-ratio, commodities and
equities — lands in the same place: genuinely, consistently low-to-negative correlation to
the champion (relative-value mean-reversion IS a different edge source), but DSR that tops
out around 0.5-0.6 even after real methodological improvement, never reaching this
session's >=0.90 bar for "confidently real." **A genuine, small, currently-unproven
signal — not "nothing," not "found alpha."** Chose to size the diversification allocation
rather than stop — result below.

**Follow-up #2, same day: the flagship-blend test — the FIRST blend result this session
that survives full scrutiny, with an important twist.** SPX/NDX-divergence and
GOLD/SILVER-hedge are themselves near-zero correlated (-0.019); a naive 50/50 "mean-
reversion sleeve" blended into the flagship looked spectacular (SR 1.43->1.66, DD -16%->
-6%, correlation to flagship -0.000) — but **decomposing it caught a real problem**:
GOLD/SILVER-hedge ALONE actually **drops the FTMO pass-rate from 96.7% to 84.5%**, despite
looking fine on annual Sharpe and even helping 2022. The daily-granularity pass-sim (real
5%-daily-loss / 10%-max-loss rules) catches something the annual Sharpe view completely
misses.

**SPX/NDX-divergence ALONE is the clean, real result.** Blended with the flagship at
w=0.20-0.35: SR 1.42->1.53, DD -16.2%->-7 to -10%, **FTMO pass-rate 97.0%->99.1% (w=0.25),
median 15.4->14.7mo** — better on every metric that was actually checked, concentrated
exactly in the flagship's two worst years (2022 delta +0.94, 2018 +0.49), costing a little
in the flagship's best years (2019/2024/2025). **A sharp cliff past ~35-40% weight**:
pass-rate collapses to 71.9% (w=0.40), 59.8% (w=0.50), 32.7% (w=0.75) while annualized
Sharpe only degrades gradually (1.48 at w=0.40 — would look fine on Sharpe alone). **Sizing
this by Sharpe/DD would be a real mistake — the FTMO constraint breaks well before the
Sharpe curve warns you; always run the pass-sim across the full weight range before
choosing one.**

**Caveats before treating this as a find:** the winning cell (window=10, threshold=2.0)
was chosen via a full-sample grid search, not walk-forward year-by-year re-selection —
MANDATORY CONTROL #3's exact warning, untested here. DSR for this cell is 0.523 (moderate,
short of the >=0.90 bar this repo's confirmed findings clear) — real uncertainty remains
about the underlying edge itself, even though the near-zero correlation and multi-metric
portfolio improvement are measured facts.

**Follow-up #3, same day: ran the walk-forward re-validation — it substantially deflates
the finding, exactly as MANDATORY CONTROL #3 predicts.** Re-selected the best
(window, threshold) cell each year on a trailing 3y Sharpe only (this repo's own
precedent: "re-picked each January on trailing 3y"), applied strictly OOS, 2018-2026.

- **Walk-forward honest Sharpe: +0.238 — less than HALF the full-sample-picked cell's
  +0.547.** Selected cell jumps around erratically year to year (window
  30->15->15->15->100->10->10->30->40) — itself a fragility signal. 2024 and 2025 are
  both clearly negative (SR -1.48, -1.21). PSR = 0.767 ("more likely positive than not,"
  not confident).
- **Re-ran the flagship blend with this honest series**: correlation held at +0.001 (the
  orthogonality is real, survives honest selection) — but the pass-rate improvement
  **collapsed from 97.0%->99.1% down to just 97.0%->97.5-97.8%**, turning negative past
  w=0.40. Drawdown still improves monotonically with weight (a real, separate effect).
  2024/2025 now show the sleeve actively HURTING the blend.

**Final verdict: SURVIVES — near-zero correlation to the champion (+0.001) and a real,
modest drawdown-reduction effect at small weights. DOES NOT SURVIVE — the dramatic
pass-rate story and the "saves the worst years" narrative, both substantially inflated by
full-sample parameter selection.** Closed as a return-source lead; the only thing left
worth considering is a small allocation sized purely as a drawdown-hedge with this
weaker, honest number — not as found alpha. This is MANDATORY CONTROL #3 demonstrated
end-to-end on a lead this session generated itself. Detail: Claude memory
`cross-asset-divergence-explored`.

### 3u. Regime-GATING the champion (not another ML feature) — DISPROVEN, cleanly (2026-08-19, `scripts/v5_xau_regime_gate.py`)

Every prior regime study fed Hurst/ADX/Choppiness into a NEW ML classifier
([[xau-regime-features-fwd-accuracy]]: real accuracy, zero tradeable P&L across 5 trade
structures). This tested the one genuinely different application, borrowed from
Forven's `regime_gate.py` concept: don't predict anything — GATE the CHAMPION'S OWN
already-proven forecast directly. Cut its size when a causal, rolling ADX-percentile
rank says "not trending," full size otherwise. Lookahead-probed clean (needed a
4200-bar synthetic panel + 1700-bar warmup to clear both the champion's slowest
1536-bar EWMA and the 252-trading-day percentile lookback).

**Walk-forward selection built in from the start this time** — direct lesson from §3t:
every number below re-selects the best (threshold, reduction) cell each year on only a
trailing 3y Sharpe, never the full-sample-best cell (which, for the record, was thr=60/
red=0.75 at SR +1.058 — doesn't even beat the champion's own 1.084 at its rosiest, an
early warning the full grid confirmed).

**Decisively worse on every single metric:**
- Walk-forward gated Sharpe **+1.007 vs champion alone +1.084**.
- **Drawdown got WORSE, not better: -19.9% vs -17.0%** — the entire premise of a regime
  gate is risk reduction in bad regimes, and it did the opposite.
- Only **1 of 9 years better** (a trivial +0.07); every other year flat-to-clearly-worse,
  no clean "helps in chop" pattern.
- **Paired-t = -2.88** — one of the most statistically decisive negative results this
  session produced.
- **FTMO pass-rate collapsed 95.2% -> 80.3%.**

**Why:** the champion's own continuous vol-target + drawdown-scaler is already a
REACTIVE risk control; ADX is a smoothed, LAGGING trend-strength read. Stacking a second,
lagging gate on an already-adaptive signal doesn't add information — it mistimes the
de-risking (cutting exposure just as a real trend is confirming, not before the chop that
already hurt).

**Verdict: closes the SECOND of two plausible regime applications on XAU** (ML feature:
real-but-untradeable; discrete gate: actively harmful) — both now disproven for
well-understood, different reasons. Not a blanket statement that regime-gating never
works anywhere — untested: gating a DIFFERENT signal, or gating on an asset that lacks
the champion's own adaptive risk control. Detail: Claude memory
`regime-gate-champion-disproven`.

### 3v. ICT/SMC concept sweep, Phase 1 — BOS/Displacement/OTE-BOS/Breaker/PO3 all DISPROVEN on XAU (2026-08-20, `src/features/ict_primitives.py`, `scripts/v5_ict_structure_xau.py`, `v5_ict_blocks_xau.py`, `v5_ict_sessions_xau.py`)

Deep, planned sweep of ICT/Smart-Money-Concepts (plan:
`~/.claude/plans/i-wanttt-you-to-playful-widget.md`) — the user's full concept list (BOS,
Displacement, Premium/Discount, OTE, Breaker Block, Mitigation Block, IFVG, Silver
Bullet, PO3/AMD, Judas Swing, SMT Divergence) plus Equal Highs/Lows, Killzones, Turtle
Soup, Unicorn, decomposed into 10 reusable causal primitives (`ict_primitives.py`, all
individually lookahead-probe-verified) rather than 11+ one-off implementations. Phase 0
(infra) also fixed a real bug found by code review, never by a live discrepancy:
`smc_signals.py::fair_value_gaps()`'s gap-size was hardcoded to EURUSD's 0.0001 pip
convention, silently meaningless on every other instrument including gold — fixed to
`gap/close*1e4` (universal bp).

Phase 1 (Bucket A — trend/breakout-continuation concepts, expected to replicate §3o's
BOS bull-beta failure) ran the FULL mandatory pipeline (lookahead-probe -> walk-forward
selection from day one, trailing-3y re-pick each January -> paired-t vs buy&hold AND
champion -> DSR/PBO on the full grid searched -> regime split) on XAU H4/H1, five
concepts:

| concept | walk-forward SR | DD | t vs B&H | t vs champ | DSR | PBO |
|---|---|---|---|---|---|---|
| BOS (baseline, `v5_smc_xau.py`'s continuous-engine re-run) | +0.04 | -21.9% | -2.59 | -3.40 | 0.42 | 0.29 |
| Displacement (P9, large-ATR-body candle -> directional state) | +0.41 | -6.9% | -2.22 | -2.83 | 0.50 | 0.46 |
| OTE-filtered BOS (P1+P2+P8, full size only if recently in the 62-79% retracement zone) | -0.38 | -8.1% | -3.00 | -3.44 | 0.01 | 0.73 |
| Breaker Block (P4+P6, trade the post-flip role of a broken order block) | -0.67 | -42.2% | -3.62 | -4.08 | 0.00 | 0.00 |
| PO3/AMD (P7, Asian-accumulation/London-manipulation-fade/NY-distribution, H1) | -0.40 | -11.6% | -3.13 | -3.44 | 0.00 | 0.10 |

**All five DISPROVEN — none clear MANDATORY CONTROL #6 (DSR>=0.90 AND PBO<0.30 AND
walk-forward-survival).** buy&hold-vt SR=+0.94, champion SR=+1.08 over the same window;
every concept above is decisively below both, statistically (paired-t -2.2 to -4.1,
1-2 of 9 years better against either benchmark).

**Breaker Block is not just noise, it's actively harmful** (SR -0.67, DD -42.2%, DSR
0.00) — sanity-checked by mirroring the sign (`-fc`): mirrored is SR -1.0 to -1.1, i.e.
even worse, so this is not an inverted edge waiting to be flipped, it's genuinely
information-free at best, anti-correlated with realized moves at worst.

**Doesn't cleanly replicate BOS's specific "bull-beta" signature** — the plan expected a
uniform "wins in bull, bleeds in 2021-22 flat" pattern; only BOS itself and (loosely) the
OTE-filtered variant show that shape. Displacement is actually the OPPOSITE: best in
2021-22 flat (+0.82) and 2015-18 chop (+0.56), weakest in 2019-20 bull (+0.12) — a
different, still-losing failure mode, not the same one. PO3/AMD is negative in every
single regime segment (-0.15 to -0.63), no regime-dependence at all. Read: Bucket A's
common thread is "no real edge, cost-and-noise-dominated," not one single mechanism —
worth stating precisely rather than forcing a uniform bull-beta narrative onto results
that don't actually share it.

**PO3/AMD had to move to H1**: an H4-bar first attempt found the London killzone window
(5h wide) contains at most one H4 bar/day, so almost the entire (tol, hold) grid had zero
signal fires and zero variance (SR=-inf) — a resolution mismatch, not a result, per the
plan's own anticipation that session concepts need M15/M30/H1. Also caught mid-build: a
real bug in the asian-range helper — `Series.cummax()/cummin()` do NOT forward-fill
through NaN in this pandas version (verified directly), so a `.where(asian_mask)`-based
running high/low silently reverted to NaN outside the asian window itself, zeroing the
signal everywhere it needed to fire; fixed with an explicit per-day `.ffill()`.

**Verdict: Bucket A (trend/breakout-continuation reading of ICT concepts) closes on
XAU, cleanly, per plan.**

**Phase 2 (Bucket B — contrarian/fade concepts, genuinely uncertain, top plan priority)
also closes, just as cleanly, same day.** Same full pipeline, four more concepts:

| concept | walk-forward SR | DD | t vs B&H | t vs champ | DSR | PBO |
|---|---|---|---|---|---|---|
| Mitigation Block (P4+P6, fade AT a still-active/unflipped zone) | -1.06 | -59.1% | -4.13 | -4.36 | 0.00 | 0.07 |
| Inversion FVG (P5-fixed+P6, flip-and-continue, FVG-sourced) | -0.60 | -48.1% | -3.50 | -4.03 | 0.00 | 0.02 |
| PO3/AMD (moved here from Phase 1 table — H1, Asian-acc/London-manip-fade) | -0.40 | -11.6% | -3.13 | -3.44 | 0.00 | 0.10 |
| Judas Swing (P7+daily-open, displacement-then-reversal-through-open, H1) | -0.76 | -28.8% | -3.53 | -3.79 | 0.00 | 0.75 |
| EQH/EQL liquidity-pool fade (P1+equal_levels+P3, sweep filtered to equal-level pools) | -0.44 | -31.4% | -2.97 | -3.26 | 0.01 | 0.66 |

**Mitigation Block is the worst single result of the entire ICT sweep** (SR -1.06, DD
-59.1%) — mirror-checked (`-fc`) same as Breaker Block, and here the mirror actually
matters: the OPPOSITE reading (fade AWAY from a still-active zone rather than bounce
toward it) comes back to a mildly-positive-but-still-nowhere-near-viable +0.13, meaning
the classic "bounce at an unflipped order block" read is backwards on XAU H4, not merely
noisy — worth stating precisely since a future attempt might otherwise re-derive the same
(wrong) bounce intuition. Inversion FVG's mirror stayed decisively negative either way
(-0.74 to -0.79), i.e. genuinely information-free like its OB-sourced sibling, not a
sign-flip case.

**EQH/EQL-fade's regime split has one loud outlier** (+1.54 in 2021-22 flat vs -0.71 to
-1.12 everywhere else) that could look like a real regime-specific edge on a quick read —
it isn't: overall walk-forward SR is -0.44, DSR 0.012, PBO 0.659 (high overfitting risk),
i.e. exactly what one four-segment split throwing up one lucky segment by chance looks
like when the concept has no real edge underneath. Flagged explicitly so it isn't
mistaken for a lead later.

**Verdict: Phase 1 + Phase 2 together close NINE ICT/SMC concepts on XAU** (BOS,
Displacement, OTE-filtered BOS, Breaker Block, Mitigation Block, Inversion FVG, PO3/AMD,
Judas Swing, EQH/EQL-fade) — none clear DSR>=0.90/PBO<0.30/walk-forward-survival, most
are decisively negative in absolute terms (not merely "not better than champion"), several
carry WORSE drawdown than a flat book. Per plan, since nothing survived to the "genuine
uncertainty" bar, the EURUSD generalization check and FTMO pass-sim are skipped (nothing
to generalize or size).

### 3w. ICT/SMC sweep, Phases 3-6 — SMT Divergence/PD-filter/confluence: SWEEP CLOSED, zero survivors (2026-08-20, `scripts/v5_ict_smt_divergence.py`, `v5_ict_premium_discount_filter.py`, `v5_ict_confluence.py`)

**Phase 3, SMT Divergence** (P1+P10) — mechanistically different from the existing
`v5_cross_asset_divergence.py` family (Z-scored relative RETURNS vs SMT's discrete
STRUCTURAL disagreement: asset A confirms a new swing extreme, asset B doesn't), tested
walk-forward (trailing-3y re-pick, same discipline as everything else this sweep):

| pair | walk-forward SR | DD | DSR | PBO |
|---|---|---|---|---|
| SPX/NDX (existing infra, prior line's best pair) | -0.40 | -30.8% | 0.43 | 0.30 |
| GOLD/SILVER (checked against `gold-silver-spread-disproven`) | -0.51 | -27.8% | 0.08 | 0.78 |

Textbook MANDATORY CONTROL #6 catch: SPX/NDX's full-sample-best cell looked mildly
interesting (SR +0.27) — walk-forward collapses it to -0.40. GOLD/SILVER stays dead under
this DIFFERENT mechanism too, consistent with (not just analogous to) the z-score
precedent. **SMT Divergence disproven on both pairs tested.**

**Phase 4, the mandatory Premium/Discount-as-champion-filter test** (runs regardless of
Phase 1-3 outcomes, explicitly compared to `regime-gate-champion-disproven`'s ADX gate):
reduce the LIVE CHAMPION's forecast when its own direction disagrees with premium/
discount (long forecast while price sits in premium = "buying high," and the mirror).
Walk-forward selection consistently picked the MILDEST gate in the sweep (reduction=0.75,
i.e. only a 25% cut, selected in 8/9 years — not an erratic/fragile selection, a stable
one) — itself a tell: **walk-forward SR +1.035 vs champion alone's own +1.084, DD -18.4%
vs -17.0%, paired-t -3.30 (only 2/9 years better)**. DSR=0.855 (short of the >=0.90 bar).
**Disproven as an improvement** — but notably much MILDER damage than the ADX gate (SR
delta -0.05 vs ADX's -0.08; DD delta +1.4pp vs ADX's +2.9pp worse) — worth the comparison
since the plan flagged this test as "priced-in skepticism" going in, and the result
confirms the skepticism without being as decisively bad as its nearest precedent.

**Phase 5, confluence** — of the plan's 6 bounded combos, #2 (BOS+OTE) and #6
(Premium/Discount-as-filter) were already run as their own standalone tests above. The
other four had every ingredient already individually disproven, so per the plan's own
rule each got ONE quick full-sample sanity check, not a re-investment:

| combo | SR (full-sample, single config) |
|---|---|
| Unicorn (Breaker Block ∩ IFVG agreement) | -0.34 |
| Judas-proxy(sweep) confirmed by BOS | -0.48 |
| SMT Divergence confirmed by liquidity sweep (SPX) | -0.03 |
| PO3/AMD-style: BOS filtered to NY-open window only | -0.49 |

**No rescue effect anywhere — confluence closes clean, zero exceptions.**

**SWEEP VERDICT: every concept on the user's list, plus every additional concept the plan
added (Equal Highs/Lows, Killzones-as-session-filter, Turtle-Soup-adjacent EQHL-fade,
Unicorn), individually AND combined, on XAU (H4/H1) plus SPX/NDX for the one concept that
needed a second correlated instrument — DISPROVEN.** Twelve standalone tests, four
confluence checks, zero survivors of MANDATORY CONTROL #6 (DSR>=0.90 AND PBO<0.30 AND
walk-forward-survival). This is not an implementation-quality verdict — every primitive
passed its lookahead probe, two real bugs were caught and fixed during the build (the FVG
pip-size bug pre-dating this sweep, and the `cummax`/`cummin`-doesn't-ffill-through-NaN bug
found live in the PO3 build), and mirror-checks confirmed the negative results are
genuine, not sign-flip artifacts (except Mitigation Block's bounce read, which is
backwards, not merely noisy — flagged so a future attempt doesn't re-derive the same wrong
intuition). It is a verdict about XAU H4's actual information content relative to what
ICT/SMC price-action patterns claim to capture: this repo's already-proven champion
(continuous EWMAC/breakout trend-following, walk-forward SR ~1.08) already captures
whatever real directional information is in XAU's price action at this timeframe: none of
these more elaborate, more specific structural/session/liquidity readings add anything
the champion doesn't already have, and several (Breaker Block, Mitigation Block, IFVG)
are actively worse than noise. EURUSD-generalization and true-M5-Silver-Bullet were never
reached because nothing survived Phase 1-3 to warrant generalizing. Detail: Claude memory
`ict-smc-sweep-disproven`.

### 3x. Turning-point bottom-detector as a "cash out on first reasonable profit" scalp — DISPROVEN, user asked to deploy live (2026-08-20, `scripts/v5_xau_turning_scalp.py`)

User asked to deploy "the 72% accuracy zigzag" live on the cent account, parallel to the
existing dual bots, cashing out as soon as a trade turns positive rather than holding for
a bigger target. No script or memory in this repo documents a 72%-accuracy zigzag
predictor by that name — clarified with the user via AskUserQuestion, resolved to
`v5_xau_turning_ml.py`'s BOTTOM/BUY detector (71% precision @ 22% recall / up to 80% @ 5%
recall, the closest number on record; SELL/tops never gets close, so LONG-ONLY by
construction). That number is pure classification precision on a static split — never
before backtested as an actual trade structure, and the specific structure requested
("enter on flag, exit at first reasonable positive, no waiting") is different from every
prior P&L test of a related detector (`xau-turning-point-detectability`: overlay-on-
champion fails; hold-48-bars merely harvests drift and loses to buy&hold), so it earned
its own honest test rather than being waved through by precedent or deployed on the
user's say-so alone.

**Pipeline**: walk-forward expanding-window HistGBoost (refit each calendar year on
strictly-past data, purge=3 bars, same pattern as `v5_xau_intermarket_accuracy.py`) scores
an OOS bottom-probability at every XAU H1 bar 2018+. Grid of 48 cells (probability
threshold x take-profit % x stop-loss % x max-hold): enter long next bar's open when
prob>=threshold and flat, exit at first TP/SL touch or max-hold-bars-at-close, live
cent-account cost ($0.34 spread + $0.10 slip, round trip). Walk-forward SELECTED the grid
cell too (trailing 3y, re-picked each year), not a full-sample best cell.

**Result: decisively negative, at every level of scrutiny.**
- **Every one of the 48 grid cells has a negative FULL-SAMPLE Sharpe** (best cell: thr=0.5
  tp=0.6% sl=0.5% hold=48, SR **-0.44**, DD -41.2%) — this isn't a walk-forward-selection
  artifact catching an inflated in-sample number; the concept loses money at its very best
  configuration before any honest selection discipline is even applied.
- **Walk-forward (the trusted number): SR -0.76, DD -35.7%**, vs buy&hold's own +0.79 over
  the same window — **paired-t -3.72, 0 of 8 years better than buy&hold.**
- **DSR=0.000, PBO=0.024** — nowhere near MANDATORY CONTROL #6's DSR>=0.90/PBO<0.30 bar
  (the low PBO here reflects that the whole grid is consistently bad, not that any cell is
  a lucky outlier worth trusting).

**Verdict: DO NOT DEPLOY.** Same mechanism as every other "detection accuracy != trading
edge" result in this repo (7 prior confirmations, `xau-regime-features-fwd-accuracy`
lists them) — the 71%-precision headline describes how often a FLAG is right when it
fires, not what happens once live spread/slippage tax the resulting trades; a
scalp-sized 0.15-0.6%-of-price target gets eaten by the $0.44 round-trip cost far more
than a wider structure would. Reported to the user before any live wiring was built —
no executor, no config, no systemd timer for this exists or should be built from this
result. Detail: Claude memory `xau-turning-scalp-disproven`.

### 3y. Champion + adverse-excursion SCALE-IN (average down into the pullback) — it is LEVERAGE, not edge (2026-08-31, `src/v5/xau_trend_scalein.py`, `scripts/v5_xau_scalein_champion.py`)

User ask: take the live cent-account champion and have it "open position if the trade
goes negative enough of what was expected, so that when the trade starts going positive
it complements the existing" — i.e. add legs on an adverse excursion. Built as a separate
engine (`xau_trend.py` is live-deployed on magic 360542 and was NOT touched), mirroring
the base leg-for-leg. **STAGE 0 regression gate: variant with `max_adds=0` reproduces the
base exactly — 443 trades both, max |equity diff| 0.000000** — so everything below
measures the idea, not an engine difference.

54 cells: `add_trigger_frac` 0.20/0.35/0.50 (fraction of the trade's own stop distance,
adverse) x `max_adds` 1/2/3 x `add_size_mult` 0.5/1.0 x `stop_policy`:
- **original** — stop stays put, so total risk MULTIPLIES with each add (the naive reading).
- **risk_constant** — stop tightens after each add to hold loss-if-stopped at the original
  one-leg budget.
- **preallocated** — same tightening stop, but the FIRST leg is sized DOWN by
  1/(1+max_adds) so a fully scaled-in stack carries the ORIGINAL risk. This is the only
  cell type that isolates "better average entry" from "more exposure", i.e. the actual
  hypothesis.

Adds are filled conservatively: adverse excursion must be confirmed by a BAR CLOSE, then
filled at the NEXT bar's open + half-spread + slippage (strictly worse than the resting
buy-limit an optimistic sim would use).

**Results (base: SR +1.024, DD -11.2%, CAGR +8.96%, Calmar 0.797, worst trade -1.0R):**

| cut | median SR | median DD | worst DD | worst trade | median Calmar |
|---|---|---|---|---|---|
| base (one leg) | +1.024 | -11.2% | — | -1.0R | 0.797 |
| risk_constant | +1.026 | -13.2% | -21.1% | -1.1R | 0.768 |
| original | +1.011 | -13.8% | -17.8% | **-2.5R** | 0.759 |
| preallocated | +1.004 | -6.3% | -9.5% | -1.0R | **0.730** |

- **24/54 beat base Sharpe (a coin flip), but only 12/54 beat base CALMAR, and the median
  variant is WORSE on both** (SR +1.014, Calmar 0.757). Drawdown deteriorates almost
  everywhere.
- **THE DECISIVE TEST — leverage-matched control.** Highest-CAGR variant: +14.28% CAGR at
  -17.8% DD (Calmar 0.803, SR +1.012, worst trade -2.4R). The plain base champion with
  ONE dial turned up (`risk_frac` 1.0% -> 1.5%): **+13.41% CAGR at -16.5% DD, Calmar
  0.814, SR +1.024, worst trade -1.0R.** At matched return the unmodified champion is
  better on Sharpe, drawdown, Calmar AND tail — the scale-in's apparent return boost is
  simply higher average exposure, reachable more simply and with far better control by
  turning risk_frac up, with no averaging-down machinery to go wrong.
- **The hypothesis fails on its own terms.** `preallocated` — the risk-neutral form where
  the only possible source of gain is a better average entry — lands at median SR +1.004
  vs base +1.024 and Calmar 0.730 vs 0.797. Strip out the leverage and averaging down is
  slightly WORSE than entering once at full size.
- **Walk-forward selection over the grid: SR +1.061 vs base +1.024, but DD -14.7% vs
  -11.2%, paired-t +0.95 (not significant), 4 of 8 years better.** PBO **0.845** — picking
  a variant is noise-chasing.
- Regime split: no regime where it clearly helps. Worse in 2021-22 flat (-0.31 vs -0.04) —
  exactly where averaging down should hurt, since chop produces repeated adverse
  excursions that never recover — and worse in 2023-26 bull (+1.24 vs +1.48).
- **`original` stop policy triples tail risk** (worst single trade -2.5R vs the base's
  -1.0R) for no Sharpe gain. If any version of this were ever deployed it must not be
  that one.
- Deep triggers SELF-LIMIT: at trigger 0.35/0.50 the second add would sit at/through the
  stop, so `max_adds` >= 2 is inert — "add more legs" is only reachable at the shallow
  0.20 trigger.
- **Survives the cost control the wrong way**: at live-floored $0.448 the case gets weaker
  still (17/54 beat base Sharpe, only **7/54** beat base Calmar, preallocated median SR
  +0.974 vs base +1.014) — unsurprising, since adds mean more fills.

**METHODOLOGICAL LESSON worth more than the result — DSR is near-vacuous for an OVERLAY
question.** This run scored **DSR 0.998 with PBO 0.845**, which reads like a MANDATORY
CONTROL #6 pass and is nothing of the kind. All 54 cells are near-copies of an already-good
base and inherit its real edge, so the cross-trial variance DSR deflates against is tiny;
DSR is answering "is this series' Sharpe real?" (yes — the champion's) not "does the
overlay contribute?" For overlays the gate is the **paired test against the base** (+0.95,
insignificant) and **PBO** (0.845, selection is noise). Do not accept a high DSR on a grid
of variations-of-a-good-thing as evidence the variation helps.

**Verdict: DISPROVEN as an edge; available only as an obfuscated leverage dial.** Keep the
champion at one leg. If more return is wanted, raise `risk_frac` deliberately (and note the
book is already at its frontier per §3q). Detail: Claude memory `xau-scalein-is-leverage`.

### 3z. risk_frac — the sizing dial, quantified (2026-08-31, `scripts/v5_xau_riskfrac_sweep.py`)

§3y concluded "if you want more return, raise `risk_frac` deliberately." This quantifies
that, so the choice is a lookup rather than a guess. **Reference table, keep it.**

**Where the dial is**: `src/v5/xau_trend.PARAMS["risk_frac"]` (default 0.01 = 1% of CURRENT
equity risked over the 3xATR stop), overridden per-bot in config JSON, read at
`scripts/v5_xau_dual.py:114` as `risk_frac x conf_risk_scale[conf]` (0.5/1.0/1.5) and
converted to lots via `order_calc_profit`. One-line config edit; the signal and trade
timing do not change at all — which is precisely why it is clean.

XAUUSD H4 champion, eval 2018+, $100k, CSV-median cost (live-floored $0.448 in brackets
where it differs materially):

| risk_frac | Sharpe | CAGR | maxDD | worst day $ | peak margin | Maven pass (4%/10%) | -10% in 12mo (static / trailing) |
|---|---|---|---|---|---|---|---|
| 0.25% | +0.985 | +2.2% | -2.9% | -824 | 0.7% | 99.5% | 0.0% / 0.0% |
| **0.50%** (deployed) | +1.015 | +4.4% | -5.8% | -1,656 | 1.4% | **99.8%** | **0.0% / 0.0%** |
| 0.75% | +1.020 | +6.7% | -8.5% | -2,471 | 2.3% | 99.0% | 0.0% / 0.1% |
| **1.00%** (cent book) | +1.024 | +9.0% | -11.2% | -3,286 | 3.3% | 97.4% | 0.4% / 1.9% |
| 1.25% | +1.025 | +11.2% | -13.9% | -4,116 | 4.5% | 94.7% | 1.8% / 7.4% |
| 1.50% | +1.024 | +13.4% | -16.5% | -4,902 | 5.9% | 91.7% | 4.4% / 17.9% |
| 2.00% | +1.024 | +17.8% | -21.5% | -6,517 | 10.4% | 86.7% | 11.2% / 45.9% |
| 3.00% | +1.023 | +26.5% | -30.7% | -9,646 | 27.1% | 78.7% | 25.7% / 87.3% |
| 4.00% | +1.023 | +34.9% | -39.0% | -12,724 | **61.8%** | 74.5% | 37.1% / 97.5% |

- **Sharpe is INVARIANT (+1.02 across the whole range) and worst trade is -1.0R at every
  level.** Same signal, same trades, size scales linearly. Return and drawdown move
  together in exact proportion: this is a risk dial, not an edge lever. (Calmar drifts
  0.74->0.89 purely from compounding, not skill — don't read it as "leverage improves
  quality".)
- **Return scales ~linearly, breach probability scales VIOLENTLY.** 1% -> 2% doubles CAGR
  (9.0%->17.8%) but takes trailing-DD breach risk from 1.9% to 45.9% — a ~24x increase.
  The asymmetry is the whole point: there is no free return above ~1%.
- **Two drawdown conventions, very different answers** — conflating them is how a book
  gets sized too big. Static (Maven's stated "10% of STARTING balance", and FTMO's
  overall wall) is far more forgiving than trailing/high-water, because a -11% dip after
  the account is already +15% up never touches the static floor. Both reported.
- **Margin only becomes a real constraint past ~2.5%** (17% of equity) and is dangerous at
  4% (62%). Below 2% it is a non-issue at 1:75.
- **On Maven's daily rule specifically**: at 0.5% the worst historical single day is
  -$1,656 (-1.66%), and the 4%-daily sim shows 0.0% daily failures — so IF Maven's daily
  limit is really their public 4%, then 0.5-1.0% is comfortable. Against a literal 0%
  daily rule nothing works, at any risk_frac (§3x-adjacent finding, `v5_maven_zero_dd_champion.py`).
- Conclusion at live-floored $0.448 is unchanged (0.5%: SR +1.006, DD -5.8%; 1%: SR
  +1.014, DD -11.4%).

**Recommendation: 0.5% for prop accounts under a 10% wall (99.8% sim pass, zero breaches
in 12 months), 1.0% as the aggressive ceiling. Do not exceed 1.5%** — that is where
trailing breach risk crosses ~18% for a linear return gain. Matches the existing
`configs/v5_xau_champ_fundingpips.json` instruction ("risk_frac 0.005 ... do NOT raise it")
by an independent route. Never raise mid-challenge (`configs/v5_xau_challenge.json`).

### 3aa. 4-15h LONG/SHORT "trend zigzag" follower with signal-based TP — DISPROVEN at real spread; every gradient points back to the existing champion (2026-08-31, `scripts/v5_xau_fast_ls_screen.py`, `v5_xau_fast_ls_exits.py`)

User ask: a 6-12h-cadence "Trend Zigzag long term follower" that "behaves like our
champion but both sells and buys", follows 4-15 hour trends, and "takes profit using
discovered signal". Extensive research requested. Run as two staged phases — cheap signal
screen first, expensive exit-rule machinery only if the screen survived.

**FIRST, THE MEASURED COST (this reframes everything).** Sampled the LIVE Maven account
(12 ticks, 2026-08-31): **XAUUSD spread $0.44, rock stable**; `stops_level` 0;
**swap_long -45.33 / swap_short +25.25 USD per lot per night** — i.e. Maven is an
FTMO-class spread, NOT raw/ECN, and shorts EARN carry (a tailwind the long-only champion
structurally cannot use). Prior art being tested against:
`fast-trend-runner-spread-gated` (fast XAU trend is real but SR 0.50 @ $0.34, needs $0.12)
and `xau-longonly-champion` ("kill-the-shorts is THE lever").

**PHASE 1 — 10 signals x 3 cost levels, all lookahead-probed (all PASS).** Families:
`ewmac_fast` (champion math at H1-fast spans), `breakout_fast`, `zigzag_structure` (CAUSAL
confirmed fractal swings: long on HH+HL, short on LH+LL — the literal "zigzag"),
`supertrend`.

| signal | turn/yr | GROSS | $0.12 | $0.34 | **$0.44 (real)** |
|---|---|---|---|---|---|
| ewmac(12-60h) | 297 | +0.68 | +0.51 | +0.39 | **+0.34** |
| ewmac(6-48h) | 445 | +0.69 | +0.44 | +0.26 | +0.18 |
| ewmac(4-32h) | 658 | +0.69 | +0.31 | +0.04 | -0.07 |
| breakout(12-48h) | 732 | **+0.84** | +0.36 | +0.03 | -0.11 |
| breakout(6-24h) | 1389 | +0.71 | -0.27 | -0.92 | **-1.22** |
| zigzag(order5) | 295 | +0.58 | +0.27 | +0.07 | -0.03 |
| zigzag(order3) | 467 | +0.78 | +0.30 | -0.03 | -0.18 |
| supertrend(10,3) | 498 | -0.79 | -0.86 | -0.91 | -0.93 |

**Only 2 of 10 signals are net-positive at the real $0.44, best +0.34.** The gross edge is
genuinely there (+0.58..+0.84) and the spread eats all of it. Turnover is the
discriminator, monotonically: 297 turns/yr survives, 1389 turns/yr loses -1.22.

**THE SHORT SIDE LOSES ON 10 OF 10 SIGNALS** (at $0.44), and long-only roughly doubles
every long/short Sharpe:

| signal | both | long-only | short-only |
|---|---|---|---|
| ewmac(12-60h) | +0.34 | **+0.86** | -0.58 |
| ewmac(6-48h) | +0.18 | +0.76 | -0.65 |
| zigzag(order5) | -0.03 | +0.61 | -0.81 |
| breakout(6-24h) | -1.22 | -0.19 | -1.48 |

This independently reproduces `xau-longonly-champion`'s "kill-the-shorts" finding at a
completely different timeframe (H1 vs H4) with different signals — **the answer to "it
should both sell and buy" is that on XAU the sell side does not pay for itself**, at any
horizon tested so far. Phase-1 walk-forward long/short: SR +0.367, **DD -39.6%**, DSR 0.119.

**PHASE 2 — can a signal-based take-profit rescue it? 64 cells** (4 signals x 6 exit
rules x params): `fc_follow` (baseline), `sig_decay` (exit when |forecast| decays to a
fraction of its entry value — the literal "take profit using discovered signal"),
`sig_flip`, `time_cap` (6/12/15/24h), `r_target` (1/2/3R), `decay_or_cap`.

- **The mechanism is REAL: corr(turnover, net Sharpe @$0.44) = -0.58.** Lower turnover
  genuinely does buy net Sharpe, so the instinct behind the request was directionally right.
- **But the signal-decay TP BACKFIRES**, which is the counterintuitive finding worth
  keeping: median turnover 202/yr vs the do-nothing baseline's 126/yr, and median net
  Sharpe +0.03 vs +0.22. Exiting on decay means re-entering when the signal firms up
  again, so a "smart" exit churns MORE than simply holding the continuous forecast.
- `sig_flip` is exactly identical to `fc_follow` (turnover 126, net +0.22) — the state
  machine already closes on a hard forecast flip, so the explicit rule adds nothing.
- The exits that help are the ones that do LESS: `fc_follow` (+0.22 median) and
  `r_target` (+0.13). Simplicity wins on a cost-bound problem.
- **0 of 64 long/short cells clear net Sharpe +0.50.** Best long/short +0.36
  (ewmac(12-60h)/r_target 1R). Best LONG-ONLY +0.79 (ewmac(6-48h)/time_cap 6h) — 2.2x
  better, again.
- Walk-forward long/short over the full grid: **SR +0.421, DD -27.7%, DSR 0.551,
  PBO 0.591** — fails both gates. Regime split: 2021-22 flat **-0.84** (a fast trend
  follower bleeds in chop, as it must).

**VERDICT: DISPROVEN as specified.** Every axis of improvement found in 74 configurations
points the same direction — SLOWER and LONG-ONLY:
`best long/short fast (+0.36) < best long-only fast (+0.79) < deployed champion (+1.02,
slower still, long-only, H4)`. The champion already sits at the end of that gradient. A
4-15h long/short book on a $0.44 spread is not a variant of it, it is a strictly worse
point on the same curve.

**Loose end, stated rather than hidden**: swap is NOT modelled in these screens. Modelling
it would help the short leg (+25.25/lot/night) and hurt the long leg (-45.33), narrowing
the long-vs-short gap. It cannot flip the conclusion: at ~0.2-0.5 lots and 10% target vol,
short carry is worth at most ~+2.5%/yr of notional, against a short leg running -0.58 to
-1.48 Sharpe (≈ -6% to -15%/yr). If a long/short fast book is ever revisited it should be
on a **raw/ECN account ($0.12, where ewmac(12-60h) reaches +0.51)** and with swap modelled
explicitly — not on Maven's $0.44. Detail: Claude memory `xau-fast-longshort-disproven`.

### 3ab. Five UNTRIED champion lifts — 4 fail outright, the 5th is a 2022+ artifact that does NOT reach the live bot (2026-09-03, `scripts/v5_xau_champion_lifts.py`, `v5_xau_volest_validate.py`)

User asked for five ways to lift the long-only champion's Sharpe, explicitly NOT rehashing
anything already tried, researched independently. Checked each against §3a-§3aa + memory
first; none of the five mechanisms appears anywhere prior.

1. **Risk-normalisation quality** — swap the engine's close-to-close EWMA vol denominator
   for a range-based estimator (Parkinson / Garman-Klass / Rogers-Satchell / Yang-Zhang),
   which are 5-8x more statistically efficient on the same data.
2. **Endogenous conviction** — scale exposure by AGREEMENT among the champion's own 6
   constituents (3 EWMAC pairs + 3 breakout windows). Unlike §3u's ADX gate the
   conditioning variable is the signal's own internal structure, and nothing is fitted.
3. **Momentum life-cycle** — the champion is trend-AGE-blind; taper by how long the trend
   has already run.
4. **Parameter-cloud averaging** — average the champion's forecast over a dense
   neighbourhood of nearby parameterisations instead of using the swept point estimate.
5. **Third-moment conditioning** — trailing realised SKEW / downside-vs-upside semi-vol
   (every prior regime study used first/second moment only).

**A METHOD FIX THAT CHANGED THE ANSWER — read this before trusting any overlay test.**
Every one of these overlays multiplies the forecast by ≤1, i.e. DE-RISKS. A paired t-test
on raw daily returns therefore marks a genuine risk-adjusted improvement as a loss: the
COMBINED variant showed SR +1.155 vs base +1.084 and DD **-13.4% vs -19.8%** while its raw
paired-t read **-2.52**. The correct gate levers each candidate to the base's realised vol
FIRST, then compares returns — the same matched-risk correction that settled §3y. Under the
raw gate 0/29 cells passed; under the matched gate 3/29 did. **Use `t@matched-vol` for any
overlay that changes exposure.**

**Results under the corrected gate** (base SR +1.084, DD -19.8%, cost $0.448):

| idea | cells | best SR | best t@matched | verdict |
|---|---|---|---|---|
| 1 range-based vol | 5 | **+1.149** | **+2.25** | passes pooled — see below |
| 2 ensemble agreement | 3 | +1.075 | -0.63 | FAIL |
| 3 trend age | 12 | +1.114 | +0.76 | FAIL |
| 4 parameter cloud | 3 | +1.078 | -0.94 | FAIL |
| 5 skew / semi-vol | 6 | +1.113 | +0.52 | FAIL |

Idea 1 looked genuinely strong: all four range estimators beat the base at **7/9 years
each**, ordered exactly by their known statistical efficiency (Parkinson +1.63 → GK +1.75
→ RS +2.11 → YZ +2.25) — a family effect needing no cell selection, robust across every
spread ($0.12-$0.60, t +2.25 to +2.27) and every vol halflife (21-126, t +1.96 to +2.64).

**Then the split-sample control killed it.** Yang-Zhang vs base:
- **first half 2018-2021: SR +0.67 vs +0.68, t -0.25, 2 of 4 years** — no effect, slightly worse.
- second half 2022-2026: SR +1.53 vs +1.41, t +3.14, 5 of 5 years.

The entire effect is confined to 2022+. This is the same artifact signature that killed
§3o's BOS (a "2023-26 bull artifact") and §3f's H4/med config ("the entire edge is 2021+").
Pooled significance plus 7/9 years was NOT enough to catch it; only the split was.
External validation agrees it is weak: gross on BTC/NDX/SPX gives +0.02 Sharpe, t +0.6-0.7.

**AND IT DOES NOT REACH THE LIVE BOT — the actionable half of the question.** The lift was
measured in the CONTINUOUS research engine, which divides by close-to-close EWMA vol. The
DEPLOYED bot (`xau_trend.run_trades`, cent magic 360542 / Maven 360571) never does that: it
sizes off `wilder_atr`, i.e. TRUE range including |H-prevC|/|L-prevC| — **already
range-based and already gap-aware.** Testing the swap where it would actually matter, in the
discrete engine, level-calibrated to ATR's mean so only shape/timing is tested:

| stop/size vol | SR | DD | trades | t@matched vs ATR |
|---|---|---|---|---|
| **wilder_atr (DEPLOYED)** | **+1.024** | -11.2% | 443 | — |
| parkinson | +0.972 | -9.8% | 524 | -0.51 |
| garman_klass | +0.959 | -11.2% | 528 | -0.62 |
| rogers_satchell | +0.927 | -11.5% | 529 | -0.92 |
| yang_zhang | +0.957 | -11.0% | 534 | -0.64 |

**All four lose to Wilder ATR**, each generating ~85 more trades (noisier stop distances →
more stop-outs). Mechanistically coherent: in the continuous engine vol is a smooth position
scaler, where estimator efficiency helps; in the discrete engine it sets a STOP DISTANCE,
where what matters is avoiding noise stop-outs, and true range with gap terms is better
suited. **Do not change the live bot's ATR.**

**VERDICT: no improvement to the champion from any of the five.** Walk-forward selection
across all 30 cells actively destroys value (SR +0.883 vs base +1.084, t@matched -2.13,
PBO 0.603) — a noisy overlay grid is worse than leaving the champion alone.

**Three by-products worth keeping:**
- **The champion's point parameterisation is NOT overfit in parameter space.** Cloud
  averaging slightly HURTS (t -0.80 to -0.94), which is the reassuring direction — a
  fragile point estimate would have been improved by neighbourhood averaging.
- **The momentum life-cycle curve** (descriptive, full-sample, forward-looking BY DESIGN,
  so a lead and not a result): forward 6-bar return by trend age — age 1-12 bars
  **-1.8bp / 52.6% hit**, age 13-30 +11.8bp / 55.1%, age 31-60 **+15.4bp / 56.2%**, age
  61-120 +3.0bp / 51.2%. The WEAK phase is the first ~12 bars after the signal crosses
  (whipsaw), not the late phase. The ignition-delay variant built on it reached SR +1.114 /
  DD -15.4% but only t +0.76 — in-sample-informed and insignificant, so it stays a lead.
- **The research harness's close-to-close vol is mildly suboptimal**, but only post-2022,
  and the deployed engine doesn't use it — so this is a note about interpreting continuous-
  engine numbers, not a fix.

Detail: Claude memory `xau-champion-five-lifts-fail`.

### 3ac. The proposed 6-component stack (vol regime / trend strength / vol sizing / HTF / hours / cost threshold) — 0/29 pass, but ONE survives the split and stays a live lead (2026-09-03, `scripts/v5_xau_stack_proposal.py`)

An external proposal recommended stacking six components onto the champion. Tested
component-by-component then stacked, gated at matched vol (§3ab), split-sampled.
**Two of the six turned out to be already implemented, and the proposal's own warning
against "highly correlated indicators" correctly predicts the failure of its own #2.**

| # | component | status | best t@matched | verdict |
|---|---|---|---|---|
| 3 | volatility-scaled sizing | **ALREADY IN PLACE** | — | `pos = fc*(TARGET_VOL/vol)` (v5_xau_turn_prob.py:94); discrete: `lots = risk*eq/(sl_atr*ATR)` |
| 6 | cost-aware min edge | **ALREADY IN PLACE** | +0.20 | `BUFFER=0.1` no-trade band; sweep confirms 0.1 is optimal |
| 1 | volatility regime filter | tested | **+1.23** | best of the six — see below |
| 2 | trend-strength ratio | tested | -0.71 | FAIL, and collinear by construction |
| 4 | higher-timeframe regime | tested | -0.20 | FAIL, category error (see below) |
| 5 | time-of-day filter | measured | -1.73 | FAIL, structurally inapplicable |

**#3 and #6 already exist.** Volatility-scaled sizing IS the engine's position line, and the
discrete deployed bot sizes risk over 3xATR — both are exactly `target_vol/current_vol`.
The cost-aware rule is the causal no-trade band; sweeping its width confirms the deployed
0.1 is already right (buffer 0.0 → SR +1.087, **0.1 → +1.084**, 0.2 → +1.073, 0.4 → +1.017;
all t@matched ≤ +0.20). Explicit min-edge thresholds on the forecast (trade only fc>0.3/
0.5/0.8) are all NEGATIVE (t -0.31 to -1.10).

**#2 fails for the exact reason the proposal warns about.** `TrendStrength =
EWMA(r)/EWMA(|r|)` correlates with the champion's own forecast at **+0.48 (hl 12) → +0.67
(hl 24) → +0.81 (hl 48) → +0.85 (hl 120)** — because the champion's EWMAC forecast is
*itself* a signal-to-noise ratio, `(fast-slow EMA)/price_vol`. At the horizons that matter
it is 67-85% the same variable. Every gate variant loses (t -0.71 to -2.65). The
proposal's "don't add a highly correlated indicator" principle is right, and its own #2
is the highly correlated indicator.

**#4 is a category error against THIS base.** The proposal assumes an M15-style entry
signal needing H1/H4 confirmation. The champion's EWMAC speeds are already **16-256 DAYS**
(96-1536 H4 bars), so a "higher timeframe" D1 ewmac(16,64) sits *inside* the horizon range
the champion already blends. All six variants lose (t -0.20 to -1.70). §3ab idea 2
(agreement across the champion's own 6 horizons) failed for the same reason.

**#5 measured, not assumed** (per the proposal's instruction). P&L attribution by H4
bar-of-day: **every hour is positive** — 20:00 +1.74bp, 00:00 +1.22, 04:00 +1.06, 08:00
+0.93, 12:00 +0.42, 16:00 +0.34 (hit rates 45-49% throughout, normal for a positive-skew
trend follower). There is no negative session to remove. And structurally there cannot be
a useful hours filter: the champion changes position ~24x/YEAR and holds for weeks, so
"hold only during London" forces intraday exit/re-entry against a $0.448 spread. Applied
anyway: SR collapses +1.084 → **+0.632**, t -1.73.

**#1 IS THE ONE WORTH KEEPING ON THE LIST.** Unlike §3ab's vol-estimator finding, it
survived the split-sample:

| schedule | pooled SR | DD | t | 2018-21 (base +0.679) | 2022-26 (base +1.411) |
|---|---|---|---|---|---|
| as proposed (lo.5/norm1/hi.5/ext0) | +1.266 | -11.9% | +0.85 | +0.785, t+0.34 | +1.627, t+0.74 |
| cut extreme only | +1.278 | -16.4% | +0.99 | +0.872, t+0.68 | +1.614, t+0.76 |
| **monotone taper 1.0→0.2** | **+1.287** | **-12.7%** | **+1.23** | **+0.887, t+0.85** | **+1.647, t+1.06** |

Every schedule improves BOTH halves by a similar margin (+0.15 to +0.24 Sharpe), and the
mechanism is sound: **the redundancy check shows vol-targeting alone does NOT neutralise
the high-vol regime** — mean position in the highest-vol quintile is only 1.5x smaller
than in the lowest, and corr(vol percentile, existing position) is just **-0.19**, because
trends and volatility co-occur so a strong forecast partly offsets the 1/vol scaling. So
there is real room, and cutting the HIGH tail is what works (cutting the LOW tail is
clearly negative, t -1.82 — this is not symmetric noise). Practical too: 61% exposure
retained, only **1.62x** leverage to match base risk.

**But it does NOT clear the bar: t@matched +1.23, 5 of 9 years.** Per §3ab's hard lesson —
a pooled t of +2.25 with 7/9 years was still an artifact — +1.23 with 5/9 is well short.
Status: **real but unproven**, the same category as §3t's cross-asset divergence.

**THE STACK is a trap worth naming.** All components multiplied: SR +1.236, DD **-7.3%**
vs base -19.8%, and 6/9 years — superficially the best result on the page. It is achieved
by **retaining 17% of mean exposure and sitting fully flat 66% of bars**. t@matched +0.67.
Stacking six ≤1 multipliers is mostly a decision not to trade; the apparent Sharpe/DD gain
is de-risking, and at matched risk it is insignificant.

**VERDICT: 0/29 components clear the gate; keep the champion byte-identical.** The
proposal's closing target — 1.02 → 1.15-1.30 robust rather than a manufactured 2.5 — is
exactly the right bar, and #1 lands in that range (1.29) while failing significance,
which is precisely the honest distinction the proposal was arguing for.

**NEXT STEP if #1 is pursued** (recorded so it isn't re-derived): do NOT add more XAU
schedule variants — that is the multiple-testing trap on a single asset. Test the SAME
high-vol taper on the other book sleeves (BTC / NDX / BRENT). A general trend-following
property should appear there too; a gold-specific fluke will not. That is a genuinely
stronger test than anything more on XAU. Detail: memory `xau-volregime-taper-lead`.

### 3ad. High-vol taper CLOSED — it is gold-specific and it HURTS the portfolio (2026-09-03, `scripts/v5_volregime_taper_crossasset.py`)

§3ac left the volatility-regime taper as the one live lead ("real but unproven") and
pre-registered the test that would settle it: same schedule, other book sleeves, portfolio
as the primary claim, no per-instrument re-optimisation. **It fails on every pre-registered
criterion.**

Costs are one-way bp from live-verified spreads (BTC 1.5bp, BRENT 7.58bp verified
2026-07-24, NDX 0.5bp) + slippage at 50% of half-spread — anchored by reproducing XAU's
0.75bp. Champion speeds rescaled by 1/6 on D1 sleeves, because `champion_signal` hardcodes
`D=6` and would otherwise silently become a 96-1536 DAY trend follower on daily bars.

| sleeve | base SR | tapered SR | DD base→taper | t@matched | years |
|---|---|---|---|---|---|
| XAU (H4, deployed) | +1.098 | **+1.311** | -19.5% → -12.4% | +1.29 | 5/9 |
| BTC (D1) | +0.970 | +0.823 | -19.1% → -16.6% | **-1.43** | 3/9 |
| NDX (D1) | +0.523 | +0.439 | -20.7% → -18.7% | -0.75 | 4/9 |
| BRENT (D1) | +0.685 | +0.561 | -18.2% → -17.7% | -0.83 | 3/9 |

**1 of 4 sleeves improves. 0 of 4 are significant. 1 of 4 improves in both halves** (only
XAU: +0.922 vs +0.701 and +1.661 vs +1.418). BTC is worse in both halves, BRENT worse in
both, NDX better in the first and worse in the second.

**THE PRIMARY CLAIM — the portfolio — is where it fails hardest.** Equal-weight the four
sleeves and apply the taper to every one:

| portfolio | SR | DD | t@matched | years |
|---|---|---|---|---|
| **base (no taper)** | **+1.480** | **-8.0%** | — | — |
| tapered | +1.310 | -6.9% | **-1.18** | 2/9 |

Worse in both halves (2018-21: +1.584 vs +1.722; 2022-26: +1.041 vs +1.246; raw t -3.21).

**WHY it fails at book level, which is the transferable lesson.** The taper's whole value on
XAU was cutting exposure in high-vol whipsaw. But (a) the portfolio's base already runs
DD -8.0% at SR +1.480 — cross-sleeve diversification is ALREADY doing that smoothing, so the
taper's risk reduction is redundant and only removes return; and (b) high-vol regimes are not
synchronised across gold/crypto/equity/oil, so tapering each sleeve independently de-risks at
different times, which subtracts from the book's diversification rather than adding to it.
**A single-asset risk overlay can be genuinely useful standalone and still be value-destroying
inside a diversified book.**

**Pre-registration earned its keep here.** Cherry-picking a different schedule per sleeve
WOULD have manufactured a positive story — BRENT's "cut extreme only" reads +0.746 vs base
+0.685 (t +0.41) and BTC's "as proposed" -0.22 instead of -1.43. Fixing the schedule in
advance is the only reason that trap was avoided.

**VERDICT: taper CLOSED. Champion stays byte-identical, and do not apply a vol-regime taper
at book level.** The XAU-only effect (+0.21 Sharpe, both halves, t +1.29) is real enough to
have been worth testing and is now explained as gold-specific — most likely the same
crisis-vol behaviour that makes gold's high-vol episodes unusually whipsaw-prone.

**The genuinely useful number from this run**: the 4-sleeve equal-weight book at properly
per-instrument bp costs scores **SR +1.480 / DD -8.0%**, versus single-XAU's +1.098 / -19.5%.
That corroborates `sharpe16-drift-portfolio` from an independent cost model, and it is the
real answer to "how do I beat 1.02" — diversification across drift classes, not overlays on
one asset. Every overlay tried on the champion in §3u/§3y/§3ab/§3ac/§3ad has failed; the
book-level result has never needed one.

### 3ae. BOOK SHOOTOUT — FundingPips (XAUmicro+ETH+DJI30) vs FTMO (XAU+BTC+NDX+BRENT): statistically equivalent; the RULE SET and the LEFT TAIL decide, not Sharpe (2026-09-03, `scripts/v5_book_shootout.py`)

Both books had documented numbers from different harnesses, dials and windows, so they were
not comparable as recorded. Re-run through ONE harness: same champion recipe, same window
(1,658 common days, 2018-01..2026-06), equal weights, per-instrument one-way bp costs from
live-verified spreads, champion speeds rescaled 1/6 on D1 sleeves.

**Per-sleeve — the two books are largely SUBSTITUTES, not alternatives:**

| sleeve | cost bp | Sharpe | | sleeve | cost bp | Sharpe |
|---|---|---|---|---|---|---|
| XAUmicro | 0.23 | +1.105 | | XAU | 0.75 | +1.098 |
| ETH | 2.03 | +0.980 | | BTC | 1.13 | +0.970 |
| DJI30 | 0.15 | +0.572 | | NDX | 0.38 | +0.523 |
| | | | | BRENT | 5.69 | +0.685 |

ETH≈BTC, DJI30≈NDX, XAUmicro≈XAU (the micro sleeve wins only on cost). The books
correlate **+0.748** with each other. The only genuinely distinct sleeve is BRENT.

**Book level:**

| book | Sharpe | maxDD | CAGR | vol | Calmar | mean abs corr |
|---|---|---|---|---|---|---|
| FundingPips (3) | +1.368 | **-6.3%** | **+8.91%** | 8.3% | **1.419** | 0.114 |
| FTMO (4) | **+1.480** | -8.0% | +8.49% | 7.2% | 1.057 | **0.081** |

**Matched-vol paired t = -0.40 — statistically indistinguishable.** FTMO edges Sharpe; FP
edges drawdown, CAGR and Calmar. Both degrade similarly across halves (FP 1.627->1.145,
FTMO 1.722->1.246).

**Pass rates: the RULE SET dominates the book choice.**

| book | under FundingPips Flex (10/6, 4% daily, 12% static, 7% dial) | under FTMO 2-Step (10/5, 5% daily, 10% static, 9% dial) |
|---|---|---|
| FundingPips | 99.6%, 18.0mo | 96.9%, 12.3mo |
| FTMO | 99.8%, 16.6mo | 96.8%, 11.6mo |

Within a rule set the two books are identical to within noise (0.2pp / 0.1pp). Flex's looser
limits give ~99.7% either way; FTMO's tighter ones ~96.9% either way, but pass FASTER
(11.6-12.3mo vs 16.6-18.0mo) because of the higher dial.

**THE REAL FINDING — Sharpe does not predict pass rate; the SCALED LEFT TAIL vs the daily
limit does.** Normalising a book to a fixed vol dial scales up books with lower realised
vol, which amplifies their tails into the daily-loss rule:

| book | dial | worst scaled day | x1.5 safety | vs 5% daily | fail_day | pass |
|---|---|---|---|---|---|---|
| FTMO 3-sleeve (no BRENT) | 7% | -2.64% | -3.96% | ok | 0.0% | **98.7%** |
| FTMO 3-sleeve (no BRENT) | **9%** | -3.39% | **-5.09%** | **BREACH** | **13.9%** | 83.3% |
| FTMO 4-sleeve | 9% | -3.20% | -4.80% | ok | 0.0% | 96.8% |
| FundingPips 3-sleeve | 9% | -2.87% | -4.30% | ok | 0.0% | 96.9% |

**This independently reconstructs the deployed design decision.** The 98.7% at 7% dial
reproduces `ftmo-challenge-bot`'s documented ~98.8%, and it explains why BRENT was added
when the book moved to 9%: without it the 9% dial breaches the daily limit 13.9% of the time.
**BRENT earns its place on TAIL PROTECTION, not on return** — on returns it is
insignificant (matched-vol t = +0.47, Sharpe 1.480 vs 1.401), but it takes daily-breach
failures from 13.9% to 0.0% and pass from 83.3% to 96.8%.

**Do NOT combine the books or add sleeves.** 50/50 FP+FTMO looks best on Sharpe (+1.519)
but the matched-vol t vs FTMO alone is only **+0.26** (insignificant). All six unique
sleeves is actively WORSE despite better Sharpe (+1.426) and equal drawdown (-6.3%):
**pass collapses to 72.5% with 26.4% daily breaches**, because its lower vol (7.1%) forces a
higher dial multiplier which pushes the scaled tail through the limit. Same "breadth
dilutes" conclusion as §3-breadth, with the mechanism now identified.

**VERDICT — the question is account-size conditional, not quality conditional.** The books
are equivalent performers; they are not interchangeable in practice.
`configs/v5_fp_flex_10k.json` records why: *"The 100K book (XAU+BTC+NDX) CANNOT be sized
here — on FundingPips specs BTC ($642) and NDX100 ($5,746) min-lots round target lots to
ZERO."* Verified FP sizing at $10,000: XAUmicro 0.01 lot (1.18x target), ETH 0.13 (0.98x),
DJI30 0.01 (0.89x), gross 0.92x. So: **run the FP book on small accounts because it is the
only one that sizes; run the FTMO book on 100K because BRENT buys tail room at the 9% dial.
Keep both as deployed — there is no upgrade available here.** Detail: memory
`book-shootout-fp-vs-ftmo`.

### 3af. Advanced portfolio construction DISPROVEN; but the BINDING-CONSTRAINT rule is a real, deployable finding (2026-09-03, `scripts/v5_portfolio_construction.py`)

Every prior attempt on this book was a SIGNAL overlay and all 40+ failed. This round
attacked the only lever that ever worked — diversification — from the two angles never
tested: HOW sleeves are weighted, and adding §3b's never-deployed fast sleeve. Weights are
estimated, so all methods are re-fit walk-forward on a trailing 3y window only.

**A. ADVANCED WEIGHTING LOSES TO EQUAL WEIGHT — 0 of 20 configurations passed.**

| method | core-4 Sharpe | t@matched vs equal |
|---|---|---|
| **equal weight (incumbent)** | **+1.428** | — |
| inverse-vol | +1.359 | -1.16 |
| HRP (Lopez de Prado) | +1.293 | -1.28 |
| RMT min-var (Marchenko-Pastur cleaned) | +1.235 | -1.66 |
| min-CVaR 5% | +1.388 | -0.57 |

Same ordering on the wide 8-sleeve universe. This reproduces the classic 1/N result
(DeMiguel-Garlappi-Uppal): estimation error swamps the optimisation gain. **Hierarchical
risk parity, random-matrix covariance cleaning and CVaR optimisation are now CLOSED for
this book — do not re-attempt.** Reproduction caveat: my narrow 3-index rebuild of §3b's
reversal sleeve scored Sharpe **-0.027**, not the 0.57 recorded there (theirs was a ~26-signal
ensemble). That is a failed reproduction on my part, NOT a refutation of §3b.

**B. THE REAL FINDING — match the book to the BINDING CONSTRAINT.** Pass rate is decided by
which rule actually fails you, and the optimal book is *different* for each:

| rule set | binding constraint | best book | measured |
|---|---|---|---|
| FTMO 2-Step @9% dial | **5% DAILY** | **widen to 8-9 sleeves** | pass 88.6% -> **94.1%**, fail_day 9.3% -> **0.0%** (5 seeds) |
| FundingPips Flex @7% | neither binds | core 4 | 99.7% vs 99.1% wide — widening only dilutes |
| Maven (+4%, 10% static) | **10% MAX LOSS** | **core 4 (highest Sharpe)** | 99.9% / 6.3mo / fail_dd 0.1% vs single-XAU 99.5% / 7.1mo / 0.5% |

Mechanism, measured not assumed: at a 9% dial the core-4 book's worst day scaled x1.5
(the floating-P&L proxy) is **-5.07%**, which breaches a 5% daily limit; widening to 9
sleeves thins it to **-4.72%**, inside the limit, and daily failures go to zero. Where no
daily limit binds, that tail-thinning buys nothing and the return dilution dominates —
so the narrower, higher-Sharpe book wins instead.

**Stated plainly: this is RISK CONFIGURATION, not alpha.** The wide book does NOT earn more
(matched-vol t -0.34, 5 of 8 years, and the 2022-26 half is worse: Sharpe 1.04 vs 1.20). No
new edge was found in this round. What was found is that the same sleeves, re-weighted to
the rule that actually fails you, move an FTMO pass rate by **+5.5 points** and a Maven
median time-to-target by **~0.8 months** with no new signal at all.

**Deliverables:** `configs/v5_maven_book.json` (Maven upgraded from single-XAU to the
4-sleeve book at a 5% dial — same pass, faster, 5x lower fail_dd; blocked only by Maven's
server-side algo permission) and `configs/v5_ftmo_wide.json` (8-sleeve book, deploy ONLY at
the 9% dial, 4 of 8 symbols still need FTMO name verification). Maven's tradeable universe
was verified live: 68 symbols, all of XAUUSD/BTCUSD/ETHUSD/US100/US30/US500/GER30/BRENT/
WTI/XAGUSD/COFFEE at trade_mode=4, so the wide book is deployable there too if ever needed.

### 3ag. Five untested research directions ALL fail — and then the real win: a SWAP-FREE gold instrument found by measuring Maven's live carry (2026-09-03, `scripts/v5_factor_sleeves.py`, `v5_volume_signals_xau.py`, `v5_book_overlays.py`, `v5_spread_trend_sleeves.py`, `v5_financing_aware_book.py`, `v5_maven_carry_book.py`)

Internet-informed sweep of everything canonical this repo had never implemented, then the
execution lever §3q named and never pulled. **64 signal/portfolio trials, 0 keepers. One
execution finding worth more than every signal result in this repo's history.**

**A. THE FIVE NEGATIVE ROUNDS (all pre-registered, all gated at matched vol).**

| round | what was new | trials | result |
|---|---|---|---|
| canonical factors | VALUE (5y reversal, TS+XS), BAB/low-vol, cross-sectional SKEW, month SEASONALITY — none had ever been coded (only tsmom/xsmom/ewmac/fx_carry existed) | 7 | 0. TS value **-0.62**, BAB -0.52, skew -0.11, seasonality -0.14. Long-horizon reversal is *negatively* paid on this universe 2018-26 |
| volume (a NEW DATA TYPE) | volume & dollar bars (Lopez de Prado activity clock), volume-weighted price path, volume-expansion gate, price-volume divergence, Amihud illiquidity | 17 | 0. Best was the VW price path (SR 1.159 vs 1.098) at t@matched +0.55. Activity-clock sampling does **not** de-noise XAU trend: tick bars at a matched horizon score t +0.13 |
| book-level risk overlays (§3ac's own open question) | Moreira-Muir vol management, covariance-based portfolio vol targeting, drawdown-responsive sizing, crisis-correlation gate | 11 | 0, and every single t is negative (best -0.19). Per-sleeve vol targeting + diversification already control book risk; a second control layer only adds lag |
| relative-value trend | trend-follow 15 same-session SPREADS (curve, crack, quality, tech-vs-broad) instead of outright prices | 16 | 0. Correlation to the core book came out at +0.03 mean **exactly as the theory predicts** — but there is no edge to diversify into. Only NDX/SPX (+0.55) and NDX/DJI (+0.52) are positive and both halve in 2022-26 |
| financing-aware weights | re-weight sleeves on NET-of-financing returns with mu = common alpha MINUS the KNOWN drag (no expected return ever estimated — the one place optimisation should beat 1/N) | 13 | 0. The optimiser goes BRENT 0.50 / BTC 0.00 and *loses* (SR 0.90 vs equal 1.04): the diversification given up exceeds the carry saved. Best simple tilt "double BRENT" t +0.62, both halves + — real but under the bar |

Two clean by-products worth keeping. **Does the +swap on shorts flip "kill the shorts"?** No —
that verdict was reached GROSS, and Maven pays +2.06%/yr to be short gold, so it was a fair
question. Net of swap, long-only still wins (net SR +0.726 vs symmetric +0.610) and adding
carry shorts hurts monotonically (k=0.25 -> 0.723, k=1.0 -> 0.677). **And HRP/RMT/CVaR's loss
in §3af is now doubly confirmed:** even given a deterministic, estimation-error-free input,
optimisation still loses to equal weight on this book.

**B. THE REAL FINDING — MAVEN CARRIES A SWAP-FREE GOLD INSTRUMENT.** §3q closed with
"financing is the single largest lever (-0.29 Sharpe) and the only untested way to move it is
EXECUTION, not signal research." Measuring it, read-only, on the live Maven terminal:

| symbol | swap_mode | spread | long %/yr | short %/yr |
|---|---|---|---|---|
| XAUUSD | POINTS | $0.42 (0.94bp) | **-3.70** | +2.06 |
| **GoldEternal** | **DISABLED (no swap)** | $0.55 (1.23bp) | **0.00** | **0.00** |
| XAGUSD | DISABLED | 6.88bp | 0.00 | 0.00 |
| EURUSD | DISABLED | 0.77bp | 0.00 | 0.00 |
| US100 / US500 / US30 / GER30 | POINTS | 0.30 / 1.68 / 1.86 / 7.19bp | -3.96 / -4.42 / -4.82 / -3.71 | +1.00 / +1.12 / +1.22 / -0.18 |
| BRENT / WTI | POINTS | 8.49 / 11.12bp | -4.79 / -3.09 | **-5.40** / -3.72 |
| BTCUSD / ETHUSD | INT_OPEN | 8.99 / 6.38bp | **-30.00** | **-30.00** |

**The free-lunch objection was tested, not waved away.** A perpetual with no swap must charge
its funding somewhere, so the log basis GoldEternal/XAUUSD was regressed on time: slope
**+0.17%/yr** (H1, 2,711 bars) and **+0.21%/yr** (D1, 143 bars), inside a ~1% band, daily-return
correlation **0.998**. Nothing resembling a -3.7%/yr embedded drift. The instrument launched
~2026-03-13 so there are only 167 calendar days of history — enough to exclude a 3.7%/yr
drift, NOT enough to exclude a small one. Re-measure monthly
(`scripts/v5_maven_carry_book.py` docstring carries the probe).

**Impact, same signal, same sizing, only the instrument changes:**

| | net Sharpe | net CAGR | maxDD | Maven pass @5% | blow-up risk |
|---|---|---|---|---|---|
| XAU sleeve on XAUUSD | +0.886 | +11.85% | -22.8% | | |
| XAU sleeve on GoldEternal | **+1.090** | **+15.04%** | -19.7% | | |
| core-4 book with XAUUSD | +0.866 | | -10.88% | 96.84% | 2.98% |
| core-4 book with GoldEternal | **+0.951** | | -10.51% | **98.02%** | **1.96%** |

Cost of the switch: **+0.33bp** of extra one-way spread. The t@matched of +49.9 (9/9 years) is
*not* evidence of edge — it confirms the change is deterministic, which is exactly why it is
worth more than the 64 signal trials above.

**Carry is broker-specific and a book tuned on one venue is mis-specified on another.**
BRENT's +9.6%/yr at FTMO — the reason it was FTMO's best net sleeve — becomes -4.79%/yr long
AND -5.40% short at Maven. Never carry a financing assumption across venues.

**Also corrected here:** `configs/v5_maven_book.json` previously quoted 99.9% pass / 6.3mo at a
5% dial. Those were GROSS of financing. Net, the same book at the same dial is 98.0% / **16.7
months** — impractical for a 4% target — so the config now runs a 7% dial (93.8% / 11.1mo /
6.1% blow-up), with the full dial table in the file.

**Open actions:** (1) confirm with Maven that synthetic instruments are permitted, and verify
fills with one 0.01-lot order; (2) the same census on FTMO and HFM is BLOCKED — both bridges
return IPC timeout on the build-5836 terminals — and FTMO's gold carry is worse (-6.6%/yr), so
a swap-free equivalent there would be worth even more than it is at Maven.

### 3ah. Full 68-symbol carry census; commodity carry NOT backtestable; and a DAY-COUNT ARTIFACT corrected — the gold-tilted book is the best net book (2026-09-04, `scripts/v5_maven_carry_book.py`, `scripts/v5_basket_challenge.py`)

Follow-on to §3ag under a changed objective: the user removed prop-firm limits from scope, so
the metric is now net Sharpe and CAGR with drawdown reported but not constraining.

**A. ONLY 5 OF 68 MAVEN SYMBOLS ARE SWAP-FREE.** EURUSD (0.86bp), USDINR (1.12bp),
**GoldEternal (1.23bp)**, XAGUSD (7.05bp), USDNGN (8.01bp). There is **no swap-free index,
crypto or energy**, so the book's gross Sharpe 1.313 is NOT recoverable at this venue — the
GoldEternal win applies to the gold sleeve only. XAGUSD is swap-free but its 7bp spread leaves
a net sleeve Sharpe of +0.073.

**B. POSITIVE-CARRY INSTRUMENTS EXIST BUT ARE NOT BACKTESTABLE HERE.** The census found large
positive long carry: SUGAR **+21.56%/yr**, COCOA +14.75%, COFFEE +3.10%, and FX at +1..+3%
(USDCHF +3.05, GBPCHF +3.04, GBPJPY +2.98, USDJPY +2.68). The obvious objection — that the roll
is already in the price, so adding swap double-counts — was tested on the extreme cases and
REJECTED: NGAS carries **-165.9%/yr** while its price drifts only **-0.99%/yr** over 14.8 years,
so Maven's price series are spot-like and swap carries the whole roll, i.e. swap IS additive.
**But the rates are a single snapshot of a quantity that varies enormously through time.**
Applying today's +21.5% sugar carry across a 2008-26 backtest would manufacture a ~22%/yr sleeve
out of one day's term structure. This repo has no historical commodity term structure, so
**commodity carry is not testable here and no number should be quoted for it.** (FX carry *is*
testable — `data/rates_3m.csv` has historical 3m rates — and remains open.)

**C. CORRECTION — A DAY-COUNT ARTIFACT INVALIDATED §3ag's BOOK COMPARISON.** BTCUSD quotes
weekends, so the 4-sleeve book's daily series ran at ~365 days/yr while the gold sleeve ran at
~259, and both were annualised with sqrt(252). That is not apples-to-apples. Compounding every
sleeve onto a common BUSINESS-DAY index changes the ranking:

| book, net of Maven's live carry | net Sharpe | maxDD | 2018-21 | 2022-26 |
|---|---|---|---|---|
| gold only, XAUUSD (the old config) | +0.862 | -22.8% | +0.42 | +1.22 |
| gold only, GoldEternal | +1.065 | -19.7% | +0.64 | +1.40 |
| core-4 equal weight | +1.096 | -10.5% | +1.27 | +0.93 |
| **core-4, gold 50% / others 16.7%** | **+1.212** | -11.5% | +1.11 | +1.30 |
| wide-7 equal weight | +0.797 | -9.1% | +0.77 | +0.82 |

So **§3ag's "diversification is net-negative at Maven, gold alone wins" was wrong.** The book
beats gold alone and a gold-overweight book beats both. Sleeve net Sharpes (aligned):
GoldEternal +1.065, BTCUSD +0.570, BRENT +0.437, US100 +0.266, US500 +0.221, US30 +0.152,
XAGUSD +0.073, GER30 -0.003; all pairwise correlations +0.04..+0.14. Widening past four sleeves
still hurts, exactly as §3ae found — the added sleeves are net-negative after carry.

**Why a tilt is defensible here when §3af's optimisers all lost:** gold's advantage over the
other sleeves is driven substantially by a CONTRACTUAL zero carry, not by an estimated alpha,
and the response is a PLATEAU (40% gold +1.200, 50% +1.212, 60% +1.196, 70% +1.167) rather than
a spike. **Stated honestly, it is still under this repo's own bar:** paired-t at matched vol is
+0.93 vs gold alone (4/9 years) and +0.75 vs equal weight (5/9), against a +1.50 threshold.
"Real but unproven."

**D. ENGINE CHANGE.** `v5_basket_challenge.py` hard-coded equal class risk (`/Nc`), so no tilt
was expressible in config. Added an optional module-level `CLASS_W` plus a `class_weights`
argument to `target_leverage()`, wired through `v5_basket_challenge_exec.py` from
`cfg["class_weights"]`. Weights are normalised inside `build()`; `CLASS_W=None` reproduces the
equal-risk path, verified by a byte-identical before/after diff of `--backtest --model ftmo`.
Configs: `v5_gold_max_sharpe.json` (DD-unconstrained, 20% dial -> CAGR +24.9% / maxDD -26.1%;
40% -> +49.8% / -46.1%) and `v5_maven_book.json` (same tilt at its 7% dial).

**The honest bottom line on "massive improvement":** removing the prop-firm constraint buys
CAGR through leverage (11% -> 25-50%), not Sharpe. Sharpe moved +1.096 -> +1.212 from the tilt,
and +0.862 -> +1.065 on the gold sleeve from the swap-free instrument. Those are the real gains.

### 4. Earlier disproven overlays (see memory for detail)
- **Per-trade probability sizing / meta-labeling** — fails twice; vol-targeting only cuts drawdown, adds no return.
- **Gold-silver spread** — corr 0.79 but z-spread edge is pre-2015-only, dead OOS 2017+.
- **Session / regime / carry / RL single-XAU overlays** — none beat the plain trend champion.

---

## Operational notes

- ~~Live dual bots reconcile hourly via `xau-dual` user timer; the systemd service intermittently marks `failed` on a hung `winedevice.exe` at teardown — teardown fix pending.~~
  **FIXED 2026-07-24 — and it was NOT cosmetic.** A oneshot unit whose wrapper calls `start_mt5.sh` spawns wine helpers into *its own* cgroup; with no `KillMode` the teardown SIGTERMs them, times out after 90s, then SIGKILLs `winedevice.exe`, which takes down the **shared wineserver for the prefix and therefore `terminal64.exe`**. Because `mt5-terminal.service` is `RemainAfterExit=yes`, systemd still reports it `active (exited)` and never restarts it — MT5 ended up alive only ~90s out of every 30min while looking healthy. Fix: **`KillMode=none`** on any unit that launches wine (added to `deploy/xau-fast.service`). `xau-dual.service` and `xau-challenge-dry.service` still lack it — harmless while disabled, same failure if ever enabled.
- **`np.float64` must never cross the rpyc bridge.** rpyc ships numpy scalars to the Wine-side interpreter as the literal text `np.float64(...)` and `eval()`s it there **without numpy imported** → `NameError: name 'np' is not defined`. This silently disabled the **trailing stop** on the live cent account: every hourly pass failed, so the position kept its original stop while the engine believed it had been trailed (1.44% of equity at risk against a 1% target). Fixed by coercing to plain `float` at every boundary in `v5_xau_dual.py` (3 action sites + `size_lots`). Scope was cent-only — the basket executor reconciles by lot size and never sends a stop.
- **A prop firm can restrict a symbol mid-challenge without notice** (§3h). `v5_basket_challenge_exec.py` now checks `trade_mode` BEFORE sending, prints `!! BROKER-BLOCKED`, records a `blocked` CSV column, and keeps retcode 10044 as a backstop; `challenge_daily_report.py` queries every mapped symbol live and adds a BROKER RESTRICTIONS section plus a subject-line flag. A blocked sleeve must never again be a one-line log entry.

---

## Open threads (next research)

1. **Weekend-flat DIVERSIFIED book** — the one funded-stage question still unanswered. §3j killed XAU-alone and §3d says drop crypto under a weekend ban; the untested combination is Friday-flat applied to **XAU+NDX+BRENT**, where the portfolio vol-target and DD scaler do the work a single asset cannot. Harness exists (`v5_cushion_engine.py::build_book(weekend_flat=True)`).
2. **Back-port the live-cost floor to `v5_instrument_search.py`** so its commodity/ag rows become usable (see §3g correction).
3. ~~Cross-asset divergence mean-reversion~~ **RESOLVED (§3t) — walk-forward validation
   run, the exciting version was substantially a full-sample-selection artifact.** Honest
   SR 0.24 (not 0.55), pass-rate benefit 97.0%->97.7% (not 99.1%). Correlation to the
   champion (+0.001) and a modest drawdown effect survive; do not re-open expecting the
   original numbers. Remaining option, untaken: a small allocation sized purely as a
   drawdown hedge on the honest, weaker figures — not as a return source.
4. **FundingPips micro** — ticket the broker: temporary restriction or is the micro contract being retired? Determines whether the 10K book stays 2-sleeve permanently (§3h).
5. Do **not** re-open: XAU-alone weekly cycles, profit-take overlays, CPPI, breadth-for-its-own-sake, or a 4th sleeve on the FTMO book.
6. Do **not** re-open (§3r): the fwd-direction/regime classifier as a tradeable signal —
   standalone, champion overlay, confidence-gating, 1-16 day horizons, and an 80-cell SL/TP
   bracket grid all failed. Its ONE reusable result is the accuracy finding itself (52.0%
   vs 49.8% persistence, block-bootstrap confirmed) — a fact about predictability, not a
   strategy. Only a materially different data type (order flow, real historical news/
   sentiment) would justify reopening this.
7. Do **not** re-open (§3s): SuperTrend, or any other single-mechanism trend filter, as a
   diversification source OR as an FX revival attempt. Validated real on XAU (DSR 0.995)
   but interchangeable with the champ-recipe there (corr 0.77), on the second independent
   basket (corr 0.86-0.88), and blended with the flagship (paired-t -1.60, pass-rate
   flat-to-worse). On FX (18 pairs x 2 directions, 36 combos): 0 pass; the best-looking
   pair (AUDJPY) drops from DSR 0.78 to 0.43 (coinflip) once properly deflated for the
   36-combo screen that produced it, not just its own 9-cell grid. A different edge SOURCE
   is needed, not another trend-detection algorithm — and any future nested grid-inside-a-
   screen must deflate DSR against the OUTER screen's trial count, not the inner grid's.
8. Do **not** re-open (§3u): regime-gating the champion's own forecast (ADX-percentile or
   any similar trend-strength cut). Walk-forward-honest from the start: SR 1.08->1.01, DD
   WORSE not better (-17.0%->-19.9%), paired-t -2.88, FTMO pass 95.2%->80.3%. Both ways of
   using regime info on XAU (ML feature, discrete gate) are now disproven.

## Honest Gaps — what we still don't know, where this is weakest

Not a to-do list (that's Open Threads above) — a standing, refreshed statement of the
blind spots, so a reader doesn't mistake "extensively researched" for "fully understood."

- **We don't know WHY the single-XAU ceiling sits at ~1.06, only that it does.** Every
  well-implemented long-only trend mechanism tried (the champion's own EWMAC+breakout
  blend, SuperTrend §3s) converges to roughly the same Sharpe on gold. Whether that's a
  property of gold's own return-generating process, of this repo's specific cost/data
  window, or of long-only trend-following as a category, is not established.
- **Tops/sell-side detection has failed 6+ times across every feature family tried**
  (price, regime, HMM, wavelet, intermarket) — but all 6 attempts were variations on
  SUPERVISED classification of a swing-high label. Whether tops are fundamentally less
  detectable than bottoms in gold, or whether every attempt so far has shared some blind
  spot the framing itself creates, is not resolved.
- **No historical news/sentiment data exists locally.** `src/v5/news_filter.py` only
  caches the live current-week ForexFactory calendar for real-time trade-blocking. An
  entire class of potential edge (event-driven, sentiment-driven) is untested, not
  disproven — absence of evidence, not evidence of absence.
- **No tick-level order flow or COT (Commitment of Traders) positioning data.** Forven's
  crowding/liquidation-cascade concepts need data this repo doesn't have; COT reports are
  free, public, and cover exactly this repo's asset universe (gold, silver, FX, index
  futures) but have never been fetched or tested.
- **FX trend-following is "comprehensively dead" on the strength of TWO mechanisms**
  (the champ-recipe, and SuperTrend across 18 pairs/36 combos) — a real, repeated result,
  but two mechanisms is not an exhaustive search of "does any trend-following approach
  work on FX," only "these two don't."
- **Every walk-forward validation in this repo re-selects a discrete grid** (a
  finite list of parameter combinations) rather than continuously re-optimizing — the
  walk-forward machinery is proven to catch full-sample-selection inflation (§3t is the
  clean demonstration) but has not been stress-tested against a MUCH larger or continuous
  parameter space, where the same trap could reappear in a subtler form.
- **The cross-asset divergence residual finding (correlation +0.001 to the champion, a
  mild drawdown effect) is real but small, and has not been tried on any pair beyond the
  7 originally screened** — SPX/NDX was the best of a small, ad hoc set, not a systematic
  search of the full correlated-pair universe this repo's data could support.

_Last updated 2026-09-03._
