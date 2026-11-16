# PREREGISTRATION — the intraday "signal for the day" (§3ew, §3ex)

Written 2026-11-16, before any outcome in this study was computed. Repo HEAD at writing:
`a72a21d`. Nothing below may be changed after the first run; deviations are listed in the
results under "DEVIATIONS", never edited in here.

## The ask (user, 2026-11-16)

> "Examine our best findings and in RANK.MD 1 to seven, apart from the diversified book, i need a
> intraday version that is the best, you can explore other methods not tried in this project, we
> need something like Zigzag but even one implemented in a different techniques, our aim is to
> have the best signal for the day, risk management to consider. Dont narrow yourself to what
> this repos has done ... try to avoid repetition, i still cannot believe simple LLM model cannot
> assist, as a feature, maybe we need better data manipulation"

## What was checked first (RULES.MD §1)

Closed and NOT re-run: fade (§1), fast trend (§3c/§3f), ORB by session (§3be/§3bx-D), release
trigger (§3bf), twelve setups incl. round numbers / LBMA fix / COMEX pinning / anchored VWAP
(§3bi), NR7 (§3bj), candle alphabet + n-gram (§3cc), own BERT (§3dl), Claude as H4 analyst
(§3n), H1 entry advisor (§3em), MTF panel (§3cp), session hand-overs (§3cv/§3cw), VWAP+EMA
(§3ej), representation ladder (§3ao), ZigZag gates/exits/frequency (§3dq-§3ev).

Genuinely NEW here, and the reason anything is run at all:

1. **A new data type.** OANDA XAUUSD **M1 mid bars 2006-03 → 2026-11-13** (7,181,205 bars, HF
   `bunqi/OANDA_XAUUSD_Gold` @ `6e269652`), single provenance. The repo's own M1 history caps at
   100,000 bars (73 days), which is why §3di filed gold's opening breakout as *unmeasurable on
   M15* (+1.17 / −1.26 by bar-order rule) and why the §3ch prior-high lead is still waiting on a
   fill assumption. Alignment verified before this file was written (no strategy outcome
   involved): return corr at lag 0 = 0.995 vs `XAUUSD_M15_long.csv`, 0.982 vs Switch M1, 0.992 vs
   CME GC H1; lags ±1 all |corr| < 0.013; 0 malformed bars. 2007-2014 has never been seen by any
   study in this repo.
2. **A new model class with new information in it.** Pretrained time-series foundation models:
   **Kronos-small** (24.7M params, a candle-native tokenizer + decoder pretrained on ~12B K-lines
   from 45 exchanges; arXiv 2508.02739) and **Chronos-Bolt-small** (Amazon, generic). RULES.MD
   says "I have a better model is not a reason" — the reason here is not capacity, it is that the
   weights carry OTHER MARKETS' history, i.e. §3as's pooled-panel idea (the one AUC lift that was
   ever real here) taken to its limit. §3ao's ceiling is an *estimate* from ~9,900 gold events;
   this tests whether it holds against a model that never needed gold events.
3. **Text, where an LLM actually has a structural advantage.** 10,570 human-annotated gold
   headlines 2000-2021 (Sinha & Khandait; HF `SaguaroCapital/...-gold`, CC BY-NC-ND). The human
   label is the ORACLE for any sentiment model, so control #5 can be run before any model.
4. **RANK #2 applied to the one real intraday effect.** §3ef's noise band was only ever run on
   gold. The effect was published on equity indices.

## Executability (RULES.MD §1)

Everything is market/stop orders on symbols Switch already quotes (XAUUSDs 0.65bp, USTECs
0.64bp, US500s 1.94bp, BTCUSDs 4.08bp; census 2026-11). No L2, no maker fills. Switch filled
42/42 stops at the level (§3eb). Flat before the 17:00 ET rollover = no swap (RANK #1).

---

# STUDY 1 (§3ew) — can a pretrained "LLM for candles" assist?

## Hypothesis

A pretrained candle/time-series foundation model's zero-shot forecast of the next candle carries
directional information about gold that (a) beats the base rate in money terms and (b) survives
cost, either standalone as the day's signal or as a size dial on existing ZigZag fires.

## Leak control specific to pretrained models (new control, stated before the run)

A pretrained model may have SEEN the test period. Kronos weights: created 2025-06-30, last
modified **2025-09-09**; its corpus is exchange K-lines and may include gold. Therefore:
`SEEN` = decisions ≤ 2025-09-09 (an UPPER BOUND, possibly contaminated), `CLEAN` = decisions
> 2025-09-09. **Kronos's confirmatory verdict is the CLEAN window only.** Chronos-Bolt's corpus
(public non-market datasets + synthetic, released 2024-11) does not contain these series; all
windows count. A claude-* API backtest is NOT run: its training memory covers the whole test
period, and §3n already measured it on post-cutoff bars.

## Frames (decision → outcome), all one-step-ahead

| frame | bar | decision | outcome (gross, mid) |
|---|---|---|---|
| D1 | 17:00 ET → 17:00 ET day | the day's open | that day's open → close return |
| S8 | 8h ET bars [00-08) [08-16) [16-24) | 08:00 ET | the [08-16) ET bar's open → close = NY session |
| H1 | 60-min UTC | every H1 close | next H1 bar open → close |

Sources: OANDA (D1/S8 built from M1, 2007-01 → 2026-11; H1 native) = primary; CME GC H1
2010-2026 (D1/S8 built from H1) = second source. Kronos H1 is computed on OANDA 2024-07-01 →
2026-11 only (compute), i.e. a SEEN-recent block and the CLEAN block of similar size.

## Models and settings (fixed; none tuned)

* **Kronos-small + Kronos-Tokenizer-base**, vendored code @ `67b630e`. Context = last 256 bars,
  OHLC only (volume = amount = 0, the repo's own no-volume path — Switch volume is 60% zeros),
  per-window z-score, clip ±5, timestamps = bar open (UTC). README sampling defaults T = 1.0,
  top_p = 0.9, top_k = 0. **S = 32 sampled next candles**, seed 0. One autoregressive step.
  Signal `k_mu` = mean over samples of `close/last_close − 1`. Also stored: `k_pup` (share of
  samples closing up), `k_rng` (mean high−low), `k_skew` (mean up-excursion − down-excursion).
  Efficiency only: a KV-cached decode of the last position, asserted equal to the reference
  `tokenizer.decode` (max |diff| < 1e-4) before use.
* **Chronos-Bolt-small**, context = last 512 closes, `prediction_length = 1`. Signal `c_mu` =
  `q50/last − 1`. Also stored: `c_rng = (q90 − q10)/last`, `c_skew = (q90 + q10 − 2·q50)/last`.

## T1 — direction as money (confirmatory)

Position: `pos = sign(mu − median of the previous 250 decisions' mu)` (causal; the threshold sits
at the signal's own base, never at 0 — §3r/[[xau-drift-baseline-inverts-3r]]). Long/short,
balanced by construction, so drift-neutral. Sign is FIXED: follow the model. No inversion.

* statistic: signed t of mean(`pos × outcome`), GROSS, clustered by day for H1;
* family: {Kronos, Chronos} × {D1, S8, H1} = **6 cells** per source;
* null: circular shift of the OUTCOME by ≥ 20 trading days, 2,000 draws, max signed t over the
  6 cells (max-stat, gross);
* also printed for every cell (control #8, #17): n, hit rate BESIDE base rate, mean bp BESIDE
  the cost, IC (Spearman), both halves, CLEAN vs SEEN for Kronos;
* long-only variant (long when `mu` above its trailing median, else flat) is scored against a
  **random same-size drop** of days (control #10, 500 resamples), and vs holding;
* net: one round trip per decision at $0.30 gold; H1 is an information test only (a per-bar
  round trip is 0.76bp against a ~0.08bp drift; no claim of tradeability can come from H1).

**PASS (a model "assists as the day's signal")** — all of: (i) D1 or S8 cell gross t ≥ +2.0 and
family p < 0.05 on OANDA; (ii) same cell t ≥ +1.0 on CME; (iii) both halves positive;
(iv) Kronos: CLEAN-window mean positive; (v) net Sharpe > 0 at $0.30 and the long-only variant
≥ 95th percentile of the random drop.
**KILL:** no cell with family p < 0.10 on OANDA → closed. No context/sample-count/temperature/
threshold changes afterwards.

## T2 — range / asymmetry information (descriptive, no pass bar)

Spearman(predicted range, realised range) and its PARTIAL after the trailing-20 mean realised
range; same for skew vs realised (close location in range). Reported because §3at already showed
perfect vol foresight does not help the long book — this is "what the model is good at", not a
money claim.

## T3 — as a FEATURE: size dial on ZigZag fires (fixed event set, §3eu protocol)

Base = variant B + circuit breaker, exactly `scripts/v5_zigzag_prob_sizing.py` (CME 2012-26,
Switch 2018-26). At each fire, on bars of 24 consecutive H1 bars ending at the fire bar:
`k_mu24` (Kronos, 256 blocks) and `c_mu24` (Chronos, 512 blocks). Dial: `m = 0.5 + r`, `r` =
expanding percentile rank of the feature among PREVIOUS fires (first 50 fires m = 1), rescaled to
mean 1. `INV = 1.5 − r` is the sign control. Null: the same multipliers dealt to the trades at
random, 2,000 times. Family = 2 features.
**PASS:** shuffle p < 0.05 in BOTH sources, INV negative in both, lift positive in both halves of
both sources. Otherwise closed. (Kronos: most fires are SEEN — a pass here would still need the
CLEAN fires to agree in sign before anything is said.)

## T5 — text: the human-label ORACLE first (control #5)

Per calendar date d: `s_d = (#positive − #negative)/(#positive + #negative)` over that date's
headlines (dates with none are skipped). Outcomes from OANDA M1:
* SAME (known-answer control): the 17:00 ET (d−1) → 17:00 ET (d) return. Must be strongly
  positive — 97% of headlines describe what gold already did.
* NEXT (confirmatory): 00:00 ET (d+1) → 17:00 ET (d+1), i.e. strictly after date d has ended in
  New York, so no headline can post-date the entry.
Statistic: mean NEXT return of `sign(s_d)` long/short in bp and its t; null = circular shift
≥ 20 days, 2,000 draws.
**PASS:** t ≥ +2.0 and null p < 0.05 and both halves positive → then, and only then, run FinBERT
(a 2019-vintage model, so 2020+ is post-cutoff) to see whether a model reaches the oracle.
**KILL:** oracle NEXT t < +2.0 → no sentiment model can exceed its own label; text-sentiment on
this corpus closes without a model being run.

## Honest priors (written before the run)

T1 pass ~5%. T3 pass ~5%. T5 oracle pass ~3% (SAME will be large; NEXT ≈ 0). T2: real, mostly
restating trailing range. The literature agrees with the repo (Tan et al. 2024: LLM components
add nothing to forecasters; no TSFM beats a random walk on liquid-asset returns).

---

# STUDY 2 (§3ex) — the intraday family on gold at M1, and RANK #2 applied to it

## Hypothesis

Gold's intraday momentum (§3ef: family p 0.013 / 0.007, gross 0.68-0.81) is real; what killed it
was cost and a close-based entry. With an exact M1 path, at least one single-trade-per-session
structure — entered by RESTING STOP at a level known in advance — is net-positive at the live
spread, beats holding at matched exposure, and holds in 2007-2014 (never tested here).

## Data, clock, cost

OANDA M1 mid, 2007-01-01 → 2026-11-13 (2006 dropped: 3,396 intraday gaps). UTC → New York.
Trading day D = [17:00 ET D−1, 17:00 ET D). Round-trip cost **$0.30** (Switch live $0.21-0.27,
HFM $0.34), stress **$0.60**. Stops: a buy-stop at L triggers when the MID high ≥ L (later and
rarer than a true ask trigger — conservative) and fills at `max(L, that bar's open)`; the full
round trip is then charged. Sell-stops mirror. Same-minute double touches are counted and
resolved AGAINST the trade.

## Cells — 7, fixed

| id | rule |
|---|---|
| N1 | noise band, NY 08:00-16:00 ET, 30-min CLOSE checks, long/short — §3ef verbatim (LOOKBACK 14) |
| N2 | N1 long-only |
| N3 | noise band with RESTING STOPS: during (t_k, t_k+1] a buy-stop rests at UB(t_k+1) and a sell-stop at LB(t_k+1) (both known in advance); exits at the 30-min checks when the close is back inside the band, flat at 16:00; re-entry allowed |
| N4 | N3 long-only |
| P1 | prior-day-high buy-stop, long-only, one per day, exit at the day's last bar (§3ch rule, exact path) |
| P2 | prior-day high/low, first break wins, other cancelled, exit at the day's last bar (§3bj rule) |
| O1 | 09:30 ET 5-minute opening range, both stops, first wins, protective stop 0.10 × ATR14(D1), flat 16:00 (§3di rule, the one gold could not be measured on) |

Sizing: N-cells 1/vol as §3ef (1% target on session vol, cap 4); P-cells same on day vol; O1
0.25% risk per trade on the stop distance. Sharpe is scale-free; the scaling only makes daily
series comparable.

## Statistic and null

`a_d = gross P&L_d − β × HOLD_d`, β = the cell's own mean signed exposure, HOLD = always long over
the cell's window (NY session or day). This is "versus holding at matched exposure" (controls #2,
#11) and is ≈ raw gross for the long/short cells. Family test: stationary block bootstrap
(mean block 20 days, 5,000 draws) of the recentred `a_d` matrix, SIGNED max t over the 7 cells
(White reality check, gross — control #14). Printed per cell: trades/wk, hit rate, mean bp per
trade beside the cost, gross / net / 2× Sharpe, halves, years positive, 2007-14 vs 2015-26,
double-touch count, and for long-only cells paired t vs matched HOLD.

**PASS (an intraday candidate):** (a) family p < 0.05; (b) net Sharpe ≥ +0.50 at $0.30 and > 0
at $0.60; (c) both halves net positive; (d) net > 0 in 2007-2014 AND in 2015-2026; (e) long-only
cells: paired t vs matched HOLD ≥ +2.0.
**KILL:** no cell passes (a) → closed, no parameter is touched. (a) without (b) = "real,
sub-cost": report the breakeven spread and stop.

## 2b — cross-instrument (RANK #2)

N1/N2 rule unchanged at 60-min checks on CME H1 2010-2026: NQ, ES, BTC (GC is §3ef's known
answer and must reproduce +0.25 / +0.19 net). Costs round trip: NQ 1.0bp (live 0.64), ES 2.0bp
(live 1.94), BTC 5.0bp (live 4.08), GC $0.30; all at 2× as stress. BOOK = equal-weight mean of
the 1/vol-scaled sleeves GC + NQ + BTC (ES is a replication cell, not a sleeve: 0.9 corr to NQ).
Family = {NQ, ES, BTC} × {LS, LO} + BOOK_LS + BOOK_LO = 8 cells; day-shift null as §3ef (one
joint shift for the book), 1,000 draws, max-stat.
Second source: Switch USTECs M15 2024-04 → 2026-11 and Binance BTCUSDT M15 (30-min checks).
**PASS (an intraday book):** BOOK_LS or BOOK_LO net ≥ +0.50 at 1× and > 0 at 2×, family
p < 0.05, both halves positive, second-source sleeves same sign.
**KILL:** NQ and ES gross < +0.30 on CME → the published effect does not survive H1 checks and
CFD cost; closed without changing session, lookback or anchor.

## 2c — risk management (applies to whatever survives; otherwise to the best net cell, labelled
"shape of something whose edge is not established")

Reported, not searched: the risk-per-day dial against P(day ≤ −3% / −4% / −5%), max drawdown and
worst week (RANK #19: the dial moves P&L and drawdown, never edge). Two overlays, each scored
against a RANDOM same-count control (controls #9, #10, 500 resamples), event set fixed:
(i) §3eu verbatim — half size when downside_vol(500 H1) ≥ its rolling median(2000);
(ii) §3dz verbatim for long-only cells — no new long once the day is down > 1 × ATR14(D1) from
the day's open.
**PASS:** ≥ 95th percentile in both halves. Otherwise not recommended.

## Honest priors

Any gold M1 cell passing all of (a)-(e): ~10%. N1 should reproduce ≈ +0.05..+0.25 net. N3: the
stop entry buys the overshoot AND the failed wicks; a martingale says the two cancel exactly, so
any gain is the momentum itself. Intraday book passing: ~10-15%. The base rate in this repo is
that the next idea fails.

## What is NOT being done, on purpose

No parameter of the band (lookback 14, session, check interval), no exit variant, no threshold,
no model setting is searched. No live order is placed. No paid data is bought.
