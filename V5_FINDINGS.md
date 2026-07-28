# V5 FINDINGS — settled experiments (do not repeat)

Running ledger of XAUUSD research that has been **run, measured, and closed**. Each
entry: what was tried, the honest result, and the verdict. If an idea here is
marked DISPROVEN / DEAD, do not re-run it without a materially new angle.

Conventions: net Sharpe from daily-resampled equity × √252; eval window = 2017+;
the live HFM cent account's true gold cost is **$0.34 spread** (the `XAUUSD_*_long.csv`
spread column understates by 10× — use `--fixed-spread-usd 0.34`, not the raw column).

---

## MANDATORY CONTROLS (2026-07-24) — four ways this project has fooled itself

Each of these silently manufactured a "result" that evaporated when the control
was added. Run all four before believing any new backtest.

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

---

## Champions currently trusted (positive net edge)

| Strategy | Net edge | Status | Notes |
|---|---|---|---|
| Sharpe-1.6 drift portfolio | eval SR **1.59**, DSR 0.9998 | backtested, **not deployed** | champion recipe across BTC/indices/XAU/silver + LS diversifiers; the real upside |
| Long-only XAU champion | eval **0.99–1.04**, live ~0.97 | **LIVE** (acct 360542) | H4, vol-targeted EWMAC+breakout, conc^1.5, long-only. `data/v5_runs/xau-longonly-champion/` |
| LS ensemble | SR **0.81** | **LIVE** (acct 360541) | long-short trend diversifier |

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
3. **FundingPips micro** — ticket the broker: temporary restriction or is the micro contract being retired? Determines whether the 10K book stays 2-sleeve permanently (§3h).
4. Do **not** re-open: XAU-alone weekly cycles, profit-take overlays, CPPI, breadth-for-its-own-sake, or a 4th sleeve on the FTMO book.

_Last updated 2026-07-24._
