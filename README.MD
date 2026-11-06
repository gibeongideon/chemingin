# MT5 V5 — systematic gold & crypto trend books, and the research ledger behind them

Long-only, vol-targeted **trend-following on assets that drift up** (gold first, then bitcoin), traded
through MetaTrader 5 on a headless Linux/Wine VPS — plus a research ledger of **several thousand
logged trials**, most of them honest failures. The project is optimised for **net Sharpe after real
broker costs and financing** (research regime U since 2026-09-04; prop-firm-specific results are
partitioned in `RESEARCH-REGIMES.md`).

**Read first:**

- `V5_FINDINGS.md` — the settled ledger (§1 … §3dk). Read its **MANDATORY CONTROLS** before trusting any number.
- `RESEARCH-REGIMES.md` — which conclusions hold under a drawdown wall (P) vs pure Sharpe (U).
- `researchpepers/*-FINDINGS.md` — external papers we replicated, one file per paper.
- `BlogPOST.md` — the plain-English story.

*Last updated 2026-11-06.*

---

## 1. Best findings at a glance

| # | Finding | Number | Where |
|---|---|---|---|
| 1 | **Kill the shorts.** On drift assets the short side loses in every family tested (10/10 signal families, 5 more times since) | long-only roughly doubles long/short Sharpe | §3, `xau-longonly-champion` |
| 2 | **The champion** — long-only EWMAC ∪ breakout, `conc^1.5`, fast vol-target, XAU H4 | Sharpe **≈1.02–1.10**, net of a $0.448 spread | `src/v5/xau_dual_signals.py::champion_signal` |
| 3 | **Diversification is the only thing that lifts it.** Same recipe, verbatim, on uncorrelated drift assets | XAU+BTC book **1.26** (2018–26, DD −13.4%); 4-sleeve XAU+BTC+NDX+BRENT **1.48** | `scripts/v5_book_overlays.py` |
| 4 | **The recipe is a template, not a gold fact.** Frozen params across 8 drift classes | portfolio **1.60** (DSR 0.9998; realistic live 1.2–1.4) | `data/v5_runs/SHARPE16_QUEST.md` |
| 5 | **Financing is the largest live lever.** Swap costs −3.79%/yr | −0.29 Sharpe; swap-free gold lifts the XAU sleeve **0.886 → 1.090** | §3q, §3ag |
| 6 | **Carry beats spread on slow sleeves.** BTC turns ~5.8×/yr, so its swap matters more than a 37× worse spread | venue choice moves BTC Sharpe by 0.2–0.5 | §3bq–§3br |
| 7 | **`risk_frac` is a pure dial.** Sharpe is invariant at ≈1.02 from 0.25% to 4% risk; tail risk is not | 1%→2% doubles CAGR, multiplies P(−10% DD) ~24× | `xau-riskfrac-sizing-table` |
| 8 | **BTC trend beats holding bitcoin** in both halves and at 10× spread | WF Sharpe 1.40; converts −83% DD into −13% | §3bp (DSR 0.743 — a *lead*) |

**The single-asset ceiling is real:** XAU alone tops out at ~1.05–1.10. Every attempt to push one
asset further — 40+ overlay configs, regime gates, ML detectors, intraday timing, new data types —
failed (§4). Sharpe comes from *more independent drift*, cheaper *financing*, and *honest sizing*.

---

## 2. The champion recipe

```text
forecast = 0.5·conc( max(EWMAC_mid⁺, Breakout_fast⁺) ) + 0.5·conc( Breakout_fast⁺ ) + 0.15
  ⁺          = clip at 0 (LONG-ONLY)
  conc(s)    = past-only-normalised s^1.5
  EWMAC_mid  = (16,64)/(32,128)/(64,256) days      Breakout_fast = 10/20/40 days
position = forecast × 10% vol-target / EWM vol (half-life 42 H4 bars), buffer 0.1, next-bar fill
```

- Survives 3× cost and a 2-bar delay.
- About half its Sharpe is gold's secular drift. It sells *drawdown control* relative to holding the metal: it does not beat buy-and-hold on gold, it beats it on BTC.
- The discrete live port (3×ATR trail, confidence-scaled risk) gives Sharpe ≈0.97, DD −10%.

---

## 3. What is live (Switch Markets demo + FundingPips trial, VPS `68.183.91.240`)

| Magic | System | Venue / bridge | Status |
|---|---|---|---|
| **360595** | XAU+BTC 50/50 champion book (`configs/v5_xau_btc_switch.json`) | Switch demo, 18815 | **live**, every 4h at :50 |
| **360598** | XAU+BTC book | FundingPips-Trial, 18816 | live, every 4h at :20 |
| **360601** | ZigZag bottom detector + 3×ATR trail | Switch, 18815 | live hourly (:19) — **earns gold's drift, not timing** (§3dg/§3dk) |
| 360597 | ZigZag + trail | FundingPips-Trial | live hourly (:24) |
| **360602** | Advisor executor — trades STRONG UP/DOWN of the adverse-move advisor | Switch | live every minute; in-sample edge only (§3de) |
| 360600 | Intraday trail-follower (`INTRADAY-TRAIL-SPEC.md`) | Switch | live, 5-min |
| 360599 | Fill probe — measures buy-stop slippage at the prior-day high | demo | measurement only |
| — | `xau-advisor` (5-min) + `advisor-notify` / `zigzag-notify` | Telegram + email | notifications |

**Broken, needs a decision:** `zigzag-obs` and `trend-signal` fail every run. `zigzag-obs` is on the dead FTMO bridge 18814; `trend-signal` fails the same way, and its logs haven't been checked yet.

---

## 4. What does NOT work (closed — do not re-run without a new data type)

The negative results are most of the value. Each is in `V5_FINDINGS.md` with its controls. Most
failures come down to five recurring reasons:

- **(a) drift in disguise:** looked good only because gold/BTC go up;
- **(b) cost wall:** a real effect smaller than the spread;
- **(c) exposure/leverage in disguise:** less risk, not more skill;
- **(d) timing ≈ random:** fails a random-timing null;
- **(e) measurement artifact:** leak, bar-order or a null bug.

### Overlays on the champion (§3w–3at)

| Tried | Result | Why it failed |
|---|---|---|
| ADX / regime **gates** | 1.084 → 1.007, DD worse, t −2.88 | (c) cuts exposure in trends' early phase; regimes are known only after the fact |
| **Scale-in / averaging down** (54 variants) | lose at matched CAGR | (c) it is leverage; plain higher `risk_frac` does it better |
| Five lifts, 6-component stack, vol taper | 0/29, 0/29; taper failed replication (1/4 sleeves) | collinear with the champion (+0.85) or a post-2022 artifact |
| **Risk oracles** (perfect foresight of future vol / MAE) | +0.085 Sharpe, t +0.68 | even perfect risk knowledge is worth little to a vol-targeted trend book |
| **Profit locks, breakeven stops, scaling out** | lock 50% of peak: 86% wins but −85% return; BE stop −6.4bp/trade | cut the right tail that pays for trend following |

### Entry timing (§3v, 3bi, 3db, 3dg, 3dk)

| Tried | Result | Why it failed |
|---|---|---|
| **ICT/SMC** (12 concepts + 4 confluences), **liquidity sweeps** | 0 survivors; "mitigation" works backwards | (d)/(a) every zone loses to a flat ATR offset at the same distance |
| **Turning-point / pivot detectors** | top oracle +1.25, best detector +0.09 | (d) a pivot is only knowable after confirmation, when the move is gone |
| **ZigZag** (HistGB, XGB/CatBoost, longer windows, payoff label) | live Sharpe = gold's; random timing p 0.48 | (a)/(d) the model learns gold's drift, not bottoms |
| **Wait for a pullback / limit entries** | filled −22.8bp vs skipped +17.9bp | adverse selection: the dips that fill are the ones that keep falling |

### Intraday / short horizon (§3be, 3bx, 3cc, 3dh, 3di)

| Tried | Result | Why it failed |
|---|---|---|
| **M15–H1 trend, next-bar fade, overnight drift** | gross edges real, net negative | (b) M15 needs a 0.587 hit rate to break even vs 0.510 at D1 |
| **Session ORB, release trigger (08:30 ET)** | p 0.84 after fixing the null | (e) the first "significance" was a cost offset inside a zero-mean null |
| **Candle language** (n-gram over candle tokens) | +0.0095 nats/token, hit 52.3% | (b) real information, too small to pay the spread |
| **BERT-style candle transformer** (pretrained pre-2024, frozen) | OOS masked-token loss *worse than token frequencies*; in ZigZag fails the timing null | no H1 sequence information exists to learn; the training gain was memorisation (§3dl) |
| **5-min ORB Stocks in Play** (share CFDs) | direction beats random by ~1 Sharpe gross; net −10 | (b) CFD spread = 0.66R per side vs the paper's 0.01R |

### ML / AI (§3dj and memory)

| Tried | Result | Why it failed |
|---|---|---|
| **Meta-labeling, DQN/RL, LLM H4 trader** | ≤ always-long; DQN = random | (a) every learner rediscovers "be long" |
| **LSTM + MADL loss** (paper) | no loss beats buy & hold | (e) sign() has zero gradient, so it never trains; (a) others collapse to always long |
| **Regime / direction classifiers** | 79/80 negative as P&L; 5.9σ AUC lift gave −0.70 Sharpe | AUC is payoff-blind: accuracy on small moves, losses on big ones |

### Factors, data, sizing and book construction

| Tried | Result | Why it failed |
|---|---|---|
| **Factors/data:**<br>• value, BAB, skew, seasonality<br>• COT<br>• IV/VRP<br>• real yields, DXY<br>• volume zones<br>• TV ratings<br>• XS-momentum | 0 keepers in ~1,100 trials | (a) contrarian readings have the wrong sign on a drift asset; the identity excess(L)+excess(S) = −2×drift |
| **Martingale, lot progressions, grids** (*Dynamic Grid Trading*) | ruin 71–97%; DGT loses to a matched constant-mix | (c) win/loss signs are serially independent (z +0.88), so a progression is leverage |
| **Shorts / long-short on gold** | lose on every signal tested | (a) shorting a drifting asset pays the drift |
| **Wider books** (weak sleeves) | 1.67 → 1.04 | weak sleeves dilute more than they diversify |
| **Optimised weights** (HRP / RMT / CVaR / ERC) | 0/20 vs equal weight | estimation error > the gain |

---

## 5. Live leads (real effect, not yet proven enough to deploy at size)

| Lead | Evidence | What's missing |
|---|---|---|
| **BTC trend sleeve** | WF 1.40, beats holding BTC in both halves | DSR 0.743 < 0.90; kept at 1/6 speed |
| **Long-only prior-high break** (stop order at yesterday's high) | net +10.2bp/trade, t +4.71, both halves, blend t ≥2 | entry slippage — fill probe 360599 is measuring it live |
| **Swap-free gold instruments** | +0.20 Sharpe on the XAU sleeve | availability per venue; rollable futures cost more (§3cr) |
| **BTC opening-range breakout** (9:30 ET, both stop orders, 10%-ATR stop, flat by 16:00) | pays in both directions; ETH replicates; book 1.26 → 1.32 | dies at 2× cost (0.10%/side); ETH spread at Switch 0.20% (§3di) |
| **Book + BTC breakout + XAU prior-high break** (10% + 10% risk) | Sharpe 1.286 → **1.391**, t +2.37, 7/9 years, DD −13.4% → −10.7% | falls below the book at 2× cost; stop-order slippage decides it (§3dm) |
| **Drift-class portfolio** (finding #4) | 1.60 historical | re-cost D1 legs with real CFD spreads; rates are not tradeable on retail CFDs |

---

## 6. Research papers replicated

| Paper | File | Verdict |
|---|---|---|
| Zarattini, Barbon & Aziz — 5-min ORB on *Stocks in Play* (SSRN 4729284) | `researchpepers/ssrn-4729284-FINDINGS.md` | Direction effect **replicates** (beats random by ~1 Sharpe gross); untradeable on share CFDs (spread 0.66R/side); moved to crypto → the cost-fragile lead above |
| Michańków et al. — MADL loss (J. Comp. Sci. 2024) | `researchpepers/MADL-S1877750324001686-FINDINGS.md` | **Does not replicate.** sign() has zero gradient, so "MADL-trained" = an untrained network; every loss → "always long" |
| *Dynamic Grid Trading* (arXiv 2506.11921) | §3df | Lower drawdown is lower exposure; loses to a matched constant-mix |
| SSRN 7445798 (prop-firm barrier maths) | `researchpepers/ssrn-7445798.pdf` | Read, not yet written up |

---

## 7. Potential research directions

Ranked by how well the evidence supports them. The pattern is consistent: **signal research on
gold price bars is exhausted; gains came from new independent drift, cheaper carry and better
execution.**

| # | Direction | Why it is promising | First concrete step |
|---|---|---|---|
| 1 | **More independent drift sleeves at honest cost** | The only lever that ever lifted Sharpe (gold 1.1 → book 1.26–1.6); class correlations ≈0–0.2 | Re-cost the drift-class portfolio's D1 legs with real Switch/FundingPips CFD spreads and swaps; add the sleeves that survive |
| 2 | **Financing / venue optimisation** | Swap is −0.29 Sharpe, larger than any signal gain; swap-free gold is worth +0.20 on its sleeve | A monthly carry census across venues; route each sleeve to its cheapest-carry venue |
| 3 | **Forward-test the cost-fragile leads** (BTC opening breakout, prior-high break) | Both are real effects whose verdict depends on fill quality, which only live fills can measure | Demo executors with their own magic numbers; compare realised slippage with the backtest's assumption |
| 4 | **Sizing layer: stream-level risk equalisation** | Worth about +0.3 Sharpe in the drift portfolio; implementable live; not a signal | Port the equaliser into the book executor; paper-run beside 360595 |
| 5 | **Genuinely new data types** (not derived from price bars) | 80/80 price-derived regime features failed: the information is not in OHLC | Crypto funding rates and on-chain flows for BTC; gold ETF holdings/flows; central-bank purchases (all retail-observable) |
| 6 | **Prospective intraday data capture** | Switch keeps only 100k M1 bars and ~1 month of ticks; every tight-stop study needs finer bars | A daily M1 + tick archiver on the VPS (like the FF-calendar archiver), feeding tick-imbalance-bar studies later |
| 7 | **Different venues for proven-but-blocked effects** | ORB "Stocks in Play" replicates in direction but needs per-share commissions; the ETH breakout needs ≤0.05% cost | Price a direct-market-access stock broker and a low-fee crypto venue; only proceed if executable retail |
| 8 | **Forward-test the best combination** (done as a backtest in §3dm: 1.391, t 2.37, fails 2× cost) | The only mix to clear t ≥ 2; its fate is one unmeasured number, live stop slippage | Read the fill probe 360599's slippage log; build a BTC-ORB demo executor; re-run `v5_combo_survivors.py` with the measured costs |

**Not worth more time** (closed on mechanism):
- entry-timing detectors;
- ICT/SMC;
- intraday trend on gold;
- loss-function or model-class swaps;
- overlays on the champion;
- grids and martingales;
- shorts on drift assets.

---

## 8. Methodology — the actual product

Thousands of failures are only informative if the measurement is trustworthy. `V5_FINDINGS.md` holds
the full MANDATORY CONTROLS. The laws that changed answers:

**Costs and benchmarks**
- **Floor costs at live quotes.** CSV spread columns understate venues by up to 10×. Measure by sampling ticks, and run a **2× cost stress before any book test** (§3di).
- **Benchmark against the incumbent and against holding the asset, not zero.** Since excess(LONG) + excess(SHORT) = −2 × drift, a long/short rule on a drift asset must beat *always long* to show skill.
- **Gate overlays at MATCHED VOL** (paired t, years better). Raw paired returns punish de-risking; DSR is vacuous for overlays because they all inherit the base's edge.

**Nulls**
- **Random timing / random direction nulls.** A detector that matches circular-shifted timing has no timing skill (ZigZag p 0.48).
- **Max-stat nulls on GROSS, signed.** A cost offset inside a zero-mean null manufactures significance (the ORB p 0.0000 → 0.84 episode). If flipping a parameter that shouldn't matter flips the verdict, run that flip first.
- **Clearing a max-stat null qualifies the family**, not the arg-max cell; deploy the most robust cell.

**Sampling traps**
- **Measure the SKIPPED sample.** Limit orders fill on losers; the skipped trades are the winners.
- **Window extremes are tautologies.** "At the N-day high" is determined by the window's return sign; split by outcome first.

**Labels and oracles**
- **Oracle ceiling first.** Hindsight-trade a label before building a detector for it; bottoms were worthless.
- **AUC is payoff-blind.** Report EV per fire beside hit rate, and print the base rate next to every hit rate.

**Leaks**
- **Lookahead probes:**
  - truncation invariance (`src/evaluation/lookahead_probe.py`);
  - an intrabar leak screen (`intrabar_leak() > 0.15` refuses a series);
  - cross-timeframe resampling is lookahead (a bar stamped t is only known at t + duration);
  - multi-broker splices are leaks.

**Measurement mechanics**
- **Tight stops need fine bars.** With a stop of ~one bar's range, the bar order decides the trade. Use M1, or calibrate the coarse result against M1, and report the pessimistic bound.
- **Watch the units.** Dollar constants don't transfer across instruments; bar-count lookbacks don't transfer across timeframes.
- **Record the trial array.** An unrecorded grid makes DSR unreproducible.

---

## 9. Layout

| Path | What |
|---|---|
| `src/v5/xau_dual_signals.py` | champion & long/short signals |
| `scripts/v5_basket_challenge.py`, `scripts/v5_basket_challenge_exec.py` | book engine + guard-first executor (`--config`) |
| `scripts/v5_xau_champion_lifts.py` | `engine_v`, `vol_match`, `report()` — the matched-vol overlay gate |
| `scripts/v5_book_overlays.py` | `sleeve_returns()` — per-sleeve champion streams for book tests |
| `scripts/v5_zigzag_signal.py`, `scripts/v5_zigzag_live_wf.py` | live ZigZag + its walk-forward replay |
| `scripts/v5_xau_advisor*.py`, `src/v5/xau_advisor_model.py` | adverse-move advisor, notifier, executor |
| `configs/` | one JSON per deployed book / venue |
| `deploy/` | systemd units, wrappers, VPS provisioning (`vps_provision.sh`) |
| `researchpepers/` | papers + per-paper FINDINGS |
| `data/` | market data (**git-ignored**: data never goes in git) and `data/v5_runs/` reports |

Recent research scripts:

| Script | § |
|---|---|
| `v5_orb_stocks_in_play.py` | §3dh |
| `v5_orb_xau_btc.py`, `v5_orb_book_integration.py` | §3di |
| `v5_madl_lstm.py` | §3dj |
| `v5_zigzag_madl.py` | §3dk |
| `v5_zigzag_bert.py` | §3dl |
| `v5_combo_survivors.py` | §3dm |
| `v5_dgt_grid.py` | §3df |
| `v5_advisor_pnl.py` | §3de |

---

## 10. Ops gotchas

**The shared bridge**
- **One MT5 session per bridge, shared over rpyc.** Never call `mt5.shutdown()` from a job: it drops the session for every other job on that bridge (2026-11-04 incident). High-frequency jobs must yield to peers' `/tmp/*.lock`.
- **`np.float64` must never cross the rpyc bridge.** Cast to `float` at the boundary.

**Time and history**
- **Clocks:** the Switch server is New York + 7h all year (16:30 server = 9:30 ET). Three gates once died because host time was used instead of server time.
- **Switch caps history:** M1 holds only 100,000 bars (~3 months). Plan intraday studies prospectively, since broker tick retention is ~1 month.

**Account state**
- **Sticky guards:** prop-style `phase_complete` latches flat forever; re-derive guards per venue, never copy them.
- **Manual trading trips the bot guard** on a shared account; never mix hand trades and a bot.
- **Account switching:** a cached `common.ini` beats the startup ini, and `term_running()` makes a restart a no-op.

**Wine and systemd**
- Any systemd unit launching Wine needs `KillMode=none`.
- Never `systemctl stop mt5-terminal`: it kills every instance.

**Repo hygiene**
- This repo can have **concurrent Claude sessions**; check `git reflog` before assuming data loss.
