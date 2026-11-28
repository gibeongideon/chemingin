# Second-look round: pre-registration

Written 2026-11-28, **before any run of this round**. The plan is in
`/home/rock/.claude/plans/once-again-we-need-ethereal-clover.md`.
- **User decision:** success = improves matched-vol Sharpe **or** prop pass rate / daily-fail rate vs the deployed book, under the controls below.
- **Rule for later studies:** Phase B/C study cells are appended here, with a timestamp, BEFORE each of those studies runs.
- **Rule against tuning:** no cell may be added after its study has produced a number.

## Binding controls (all studies)

1. **One price feed per study.** `XAUUSD_H4_long.csv` is never used (FTMO splice from 2023-06-28).
   - Gold H4 S1 is rebuilt from `data/XAUUSD_M15_long.csv` with `src/v5/mtf_frames.aggregate`.
   - Gold H4 S2 is resampled from `data/oanda_hf/XAUUSD_H1.parquet` (OANDA mid, 2007+).
2. **Leak screen.** `intrabar_leak` ≤ 0.15 on every series used for features. Zero Switch spreads before 2025 are floored.
3. **Known-answer control cell in every script.** The script must reproduce a recorded number before its result counts.
4. **Max-stat nulls:** on GROSS returns, signed. Costs are deterministic and are applied after.
5. **Overlays:** compared at MATCHED VOLATILITY with the event set fixed. A dial is compared with a random dial of the same count.
6. **Revival bar (the Sharpe arm):**
   - matched-vol paired t ≥ 2.0 in the primary source, and > 0 in the second source where one exists
   - positive in both halves
   - survives 2× cost
   - beats its timing/shuffle null p95
7. **Revival bar (the prop arm):**
   - at the SAME vol dial, GFT 2-step static pass rate +5pp or more vs the base, AND daily-fail % not higher
   - in both sources
   - same bootstrap seeds as `scripts/v5_gft_check.py` (3 × 1200 paths, block 21, 36 months, no intraday guard, so the raw left tail is compared)
8. **Holdouts:** OANDA 2007-2014 gold and CME GC 2011-2017 are opened ONCE, only by #2 and #8.
9. **Trial array:** every cell run is listed in the study's output CSV. No DSR is quoted without it.

## Phase A cells

### #1 Champion meta-label trim (orig. §3p, `scripts/v5_xau_turn_prob.py` approach 4, label `champmeta`)
- **Unchanged from §3p:**
  - features `build_features`
  - label = the champion's own forward 24-bar P&L, "bad" = train-slice quantile 0.30
  - expanding yearly HistGB refits, purge 8 + k, isotonic calibration on a purged tail
  - overlay `fc = champ_fc × (1 − b × p_bad)`
  - engine `engine()` (vol target 10%, buffer 0.1, delay 1)
  - cost = $0.448 floor + $0.10 slip per side, as in §3p "live"
- **Cells:** b ∈ {1.0 (PRIMARY), 0.5}.
- **Sources:**
  - **S1 = M15→H4.** OOS 2018+, identical to §3p.
  - **S2 = OANDA H1→H4.** OOS 2012+. Its 2012-2017 slice is unseen by §3p.
- **Known-answer control:** S1 champion SR must land near §3p's 1.041 (± 0.15, since the feed differs).
- **Reported:**
  - SR, dSR, raw paired t (the old statistic), **matched-vol paired t (the new statistic)**
  - yrs+, halves, maxDD, CAGR at matched vol
  - 60-draw block-shuffled-p null p95
  - 2× cost
  - prop arm: GFT 2-step static at a 9% dial

### #2 High-vol taper, out of sample on gold (orig. §3ac/§3ad, `scripts/v5_xau_stack_proposal.py`)
- **Rule:** the single frozen schedule exactly as §3ac recorded it. It is read from that script and not re-tuned.
- **Data:** CME GC (Databento, `load_cme("GC")`), resampled to H4, from 2011-06 through 2017-12 ONLY. That window was never used to choose the taper.
- **Base:** the champion on the same bars.
- **Pass:** matched-vol paired t > 0, positive in both halves, and the prop arm is not worse.

### #3 Old de-risking overlays on the prop yardstick (orig. §3bz hysteresis, §3q corr-aware sizing)
- **Rules:** frozen as recorded in their scripts (located before the run and appended here).
- **Comparison:** each overlay's daily return series vs the base book at the SAME dial, through the prop simulators.
  - GFT 2-step static: `v5_gft_check.challenge`, no guard
  - FTMO 2-step: 10% / 5% targets, 5% daily, 10% static
- **Pass:** the prop arm (control 7) plus matched-vol t > −1, i.e. it costs no significant Sharpe.

### #4 Vetoes re-scored as size dials (on the ZigZag variant B + circuit breaker, CME and Switch cached probs)
- **Family (6 gates):**
  1. §3em G_SHORT
  2. §3er SMA50
  3. §3er "broke prior low"
  4. §3dy 50-day EMA rising
  5. §3dt champion ≥ 0.5
  6. §3ea 2-year-yield up
- **Dial:** size 0.5 when the gate says "bad", 1.0 otherwise. The event set is fixed (asserted).
- **Control:** a random dial halving the same number of trades, 200 resamples per gate.
- **Pass:**
  - ≥ 95th percentile vs the random dial in BOTH sources
  - positive in both halves in both sources
  - family max-stat p < 0.05 across the 6 gates
- **Known-answer control:** reproduce §3eu's vol taper (+0.109 CME / +0.215 Switch at size 0.5, ≥ 98th percentile).

## Phase B/C (primary bars; exact cells appended before each run)
- **#5 PDH × noise-band dial (OANDA M1 2007-26).**
  - Known-answer control: PDH net +0.75.
  - Pass: dial ≥ 95th percentile vs a random same-count dial, both halves, 2× cost.
- **#6 PDH panel (PREREG_ROUND2 Study A).**
  - Cells exactly as in `data/v5_runs/intraday_day_signal/PREREG_ROUND2.md`.
  - Family p < 0.05 and BOOK_WF paired t ≥ 2.
- **#7 Fast intraday trend, long-only (OANDA M1).**
  - Cost cells $0.25 and $0.12, swap modelled, plus flat-every-night.
  - Pass: beats matched hold AND adds to the H4 champion at matched vol (t ≥ 2).
- **#8 §3ak frozen best trigger** on OANDA H4 2007-2014 (holdout, opened once).
  - Pass: t ≥ 2 with the sign recorded in §3ak.
- **#9 Index reversal / overnight sleeve** on OANDA M5 indices with synchronised timestamps.
  - Pass: as a book sleeve, matched-vol t ≥ 2, both halves.
- **#10 FX carry, and the crypto trend panel vs each coin's own hold.**
  - Pass: as a book sleeve, matched-vol t ≥ 2, both halves. The crypto part uses pooled paired t vs hold.

## Addendum 2026-11-28 (before #3 ran)
- **#3 scope change.** §3q "correlation-aware sizing" has NO single frozen rule. It was the best of 103 workflow trials, median trial +0.018. Re-testing its best cell would be selection, so it is DROPPED.
- **§3bz hysteresis frozen cell:** P(up 24h) > 0.65 → long, < 0.35 → flat, otherwise hold.
  - Model, features and walk-forward imported unchanged from `scripts/v5_prob_hysteresis.py`, on `XAUUSD_H1_long.csv` (single-broker, 2015-2026).
  - Compared with the vol-targeted H4 champion built from the same broker's M15 (`v5_second_look_1.s1_h4`), over a common window.
  - Both are scaled to the same 9% dial in the prop simulator.
- **#3 pass:** control 7 (prop arm) AND matched-vol paired t > −1 vs the champion.
- **#1 and #2 results are already in. Neither changes any cell above.**

## Addendum 2026-11-28 (before #5 ran): PDH × noise-band dial
- **Trades.** PDH trades from `data/v5_runs/intraday_day_signal/pdh_trades.parquet` (§3ex P1, OANDA M1 2007-26, resting buy stop at the prior-day high, flat by the day's close). Day = 17:00 ET → 17:00 ET. `t_in` = minutes after 17:00 ET.
- **Band.** Noise band exactly as in `scripts/v5_noise_band_momentum.positions`: NY 08:00-16:00 ET, 30-minute checks, LOOKBACK 14, upper band = max(open, prev close) × (1 + σ).
- **D1 (PRIMARY).** State of the PREVIOUS NY session at its final check (15:30 ET). If above the upper band, size 1.0; otherwise 0.5.
- **D2.** Same-day NY state at the last 30-minute check strictly before the entry minute. If above the upper band, 1.0; if inside or below, 0.5. Entries before 08:30 ET → 1.0 (no information yet).
- **Known answer.** The PDH net Sharpe of the vol-weighted daily series must reproduce §3ex ≈ +0.75 (±0.1).
- **Control.** A random dial halving the same number of trades, 500 draws → percentile.
- **Pass:**
  - ≥ 95th percentile
  - dSR > 0 in both halves
  - still ≥ 95th percentile at 2× cost
  - for 2 cells, the primary must pass on its own, or both at the Bonferroni 97.5th percentile

## Addendum 2026-11-28 (before #6 ran): PDH panel = PREREG_ROUND2 Study A, run as written
- **Script.** `scripts/v5_pdh_panel.py` (`build`, then `report`), unchanged. Its own pass bars, kill rule and books (BOOK_LONG / BOOK_LS / BOOK_WF) apply verbatim. Studies B, C and D are NOT part of this round.
- **DEVIATION 1.** `data/oanda_hf/5/XAUUSD.parquet` did not exist, because the raw M5 download was incomplete. It was built by aggregating `XAUUSD_M1.parquet` (same bunqi/OANDA provenance) into 5-minute bars on epoch boundaries. Gold's known-answer check (LONG leg ≈ M1 net +0.79 ± 0.05) verifies it.
- **DEVIATION 2.** EURJPY, GBPJPY, AUDJPY and EURGBP were never downloaded. They are fetched now with the repo's own `scripts/fetch_oanda_hf.py`. If any fetch fails, the panel runs without it, and the kill rule stays at "fewer than 4 non-gold LONG legs net-positive".

## Addendum 2026-11-28 (before #7 ran): fast intraday trend, long-only
- **Signal.** `ewmac_fast(12-60h)` = `ewmac_fast(d, ((12,48),(15,60)))` from `scripts/v5_xau_fast_ls_screen.py`, long-only (clip ≥ 0). It was chosen in §3aa as the best at real cost; it is not re-chosen here.
- **Engine.** The `engine_h1` formula (10% vol target, buffer 0.1, delay 1, $0.10 slip per side), copied locally only to add swap and a flat-overnight option.
- **Data.** OANDA H1 mid 2007-26, eval 2008-01-01+.
- **Cells:**
  - cost C ∈ {0.25 (PRIMARY), 0.12} round-trip spread (+ slip)
  - swap ∈ {0 (PRIMARY, swap-free venue), −6%/yr on long notional charged at the 17:00 ET rollover}
  - overnight ∈ {hold (PRIMARY), flat 16:00-18:00 ET}
- **T1.** The fast sleeve vs a constant-long hold in the same engine and costs, matched-vol paired t.
- **T2.** A 50/50 book (fast sleeve + the H4 champion on OANDA H4, same C, each at 10% vol) vs the champion alone, matched-vol paired t and both halves.
- **Pass (primary cell).** T1 t ≥ 2 AND T2 t ≥ 2 AND T2 positive in both halves.

## Addendum 2026-11-28 (before #9 ran): index overnight and reversal sleeves, synchronised
- **Flaw being fixed.** §3b's "verification" priced the overnight leg on HFM futures-CFDs at their D1 bar boundary (17:00→18:00 ET). That is not the cash-session overnight window (16:00 → 09:30 ET) where the edge was found. On a 24-hour CFD that window IS tradeable.
- **Universe.** OANDA M5 mid NAS100USD, SPX500USD, US30USD, US2000USD. All share the 16:00 ET cash close, so the legs are synchronised by construction. Eval 2008-01-01+.
- **Costs.** Round trip per trade, from PREREG_ROUND2: NAS 1.0, US30 1.0, SPX 2.0, US2000 5.0 bp. Financing −4%/yr on long notional for every night held across the 17:00 ET rollover (PRIMARY includes it).
- **O1 (PRIMARY): overnight.** Buy at the close of the 15:55 M5 bar (16:00 ET), sell at the open of the 09:30 bar on the next trading day. Equal weight across the available instruments, each leg 1/vol-weighted with a trailing-60-day std of its own overnight returns (lagged one day).
- **R1: 1-day reversal.** Position = −sign(previous 16:00→16:00 return), held 16:00→16:00, same weighting and costs. Financing is charged on longs only.
- **Known answer.** O1 must show a positive gross overnight drift on SPX (literature / §3b's backtest).
- **Tests.**
  - T1: O1 vs holding the same index basket 24 hours at matched vol, i.e. not just drift.
  - T2: a sleeve test vs the deployed XAU+BTC book (`v5_account_chooser_2026_10.xau_btc_book`). The 50/50 book (sleeve scaled to the book's vol) vs the book alone, matched-vol t, both halves.
- **Pass (O1, primary).** T2 t ≥ 2, both halves positive, still positive at 2× cost. R1 is reported against the same bar.

## Addendum 2026-11-28 (before #10 ran)

### 10a: G10 FX carry
- **Universe.** USD vs EUR, GBP, AUD, NZD, JPY, CAD, CHF, via the 7 USD pairs in `data/*_D1_long.csv`. Close-to-close only: the §3av leak affects high/low, not closes.
- **Rates.** `data/rates_3m.csv`, monthly, forward-filled, used from the month AFTER the stamp (publication lag).
- **Signal.** At each month-end, rank the 8 currencies by 3m rate. Long the top 3, short the bottom 3, equal weight, each leg expressed vs USD (USD can be in either leg). Hold one month.
- **Return.** Spot return + carry accrual (rate differential / 252 per business day) − a 1.0%/yr broker swap markup on each leg's notional − 2bp per leg per rebalance.
- **Scaling.** Vol-targeted to 10% with a trailing-60-day std, lagged. Eval 2008-02+.
- **Pass.** Standalone net SR > 0 AND, as a sleeve on the XAU+BTC book (50/50 at matched vol), T2 t ≥ 2, both halves positive.

### 10b: crypto trend panel vs each coin's own hold
- **Coins.** BTC, ETH, SOL, LTC, AVA (`data/*_D1_long.csv`, calendar days).
- **Signal.** `champion_recipe(close, 1/6, 1.5)` long-only, the same engine as `xau_btc_book` (10% vol target, ewm hl 42, cost 4.5bp per turnover, financing −15%/yr on long notional). Hold = the same engine with forecast 1.
- **Test (T_a).** Per coin, the matched-vol daily difference (trend − hold). Pooled = cross-coin mean per calendar day; t on that series.
- **Pass.** Pooled t ≥ 2, ≥ 4 of 5 coins positive, both halves positive.
- **Sleeve test (T_b).** An equal-weight 5-coin trend sleeve vs the BTC-only trend sleeve inside the XAU+BTC book at matched vol (does breadth help?). Reported; not a pass condition.
- DEVIATION 3. `v5_pdh_panel.build` crashed on a redundant masking loop (never run before). It was removed; the next line already applied the same mask, and the loop only fed a Study C feature. Study A trades are unaffected. All four FX crosses downloaded, so the panel is the full 24 instruments.

## Addendum 2026-11-28 (before Phase D ran): combine the survivors
- **Survivors by the bars above.**
  - #1 champion meta-trim b = 0.5 (qualified: S1 t +2.73, S2 +0.89).
  - #10b crypto trend vs hold: a validation of the existing BTC sleeve, with no rule change.
- **Not carried.** #4 leads (Y2UP, CHAMP dials) and #6 legs (BTC / USDJPY PDH) did not pass their bars, so they are not carried.
- **Test.** The deployed XAU+BTC book (`v5_account_chooser_2026_10.xau_btc_book` construction, gold H4 from M15, BTC D1, 50/50, Switch costs) vs the same book with the gold sleeve's forecast × (1 − 0.5 × p_bad).
  - p_bad comes from #1's S1 walk-forward, unchanged.
  - Common window is 2018+.
- **Reported:**
  - Sharpe, matched-vol paired t, both halves
  - prop arm: GFT 2-step static (pass, months, daily-fail) at a 9% dial, and the same with the 2% intraday guard + cushion sizing actually deployed on the GFT account
- **Pass.** Matched-vol t ≥ 2 OR the prop arm (+5pp pass, daily-fail not higher), with both halves positive.
