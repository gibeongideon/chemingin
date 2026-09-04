# Research regimes: prop-firm-constrained vs unconstrained

**Why this file exists.** On 2026-09-04 the objective changed: *"you should not do research in
terms of prop-firm limitations, you just need to find a working plan no matter the drawdown."*
Almost every result in `V5_FINDINGS.md` before that date was optimised against a prop-firm rule
set — pass rate, daily-loss limit, static max loss. Some of those conclusions **flip** when the
rules come off, and some are **unaffected**. Mixing the two is how you end up deploying a
9%-dial tail-insurance book on an account that has no daily limit, or a 40%-dial book on a
challenge with a 10% wall.

This file partitions the work into three buckets and gives an explicit resume point for each
regime. **Read this before re-opening any line of research.**

- **Regime P** — prop-firm-constrained. Objective: maximise probability of passing / not
  breaching. The binding rule decides everything.
- **Regime U** — unconstrained. Objective: maximise net Sharpe, then dial leverage to taste.
  Drawdown is reported, not constrained.
- **Regime I** — invariant. True in both; never needs re-testing when the regime changes.

Current regime: **U** (since 2026-09-04).

---

## 1. Regime I — INVARIANT findings (do not re-test when switching)

### 1.1 The signal side is exhausted for this strategy family
Every one of these was disproven under a matched-vol gate and none of the verdicts depend on
drawdown rules. Full detail in `V5_FINDINGS.md` and the linked memories.

| line | verdict | §  |
|---|---|---|
| ICT/SMC — 12 concepts + 4 confluence combos | all fail walk-forward on XAU H4/H1, SPX, NDX | 3v-3x |
| Regime gating the champion (ADX percentile) | SR 1.08 -> 1.01, DD worse, t -2.88 | 3aa |
| Cross-asset divergence / mean reversion | walk-forward re-selection cut SR 0.55 -> 0.24 | 3z |
| Canonical factors: VALUE (5y reversal TS+XS), BAB, XS SKEW, SEASONALITY | 0/7. TS value **-0.62** | 3ag |
| VOLUME as a new data type: volume/dollar bars, VW price path, expansion gate, PV divergence, Amihud | 0/17. Activity-clock sampling does not de-noise trend (t +0.13) | 3ag |
| Book-level risk overlays: Moreira-Muir, covariance vol targeting, DD sizing, corr gate | 0/11, **every t negative** | 3ag |
| Relative-value trend on 15 same-session spreads | 0/16. Corr +0.03 as predicted, but no edge | 3ag |
| Advanced portfolio construction: HRP, RMT/Marchenko-Pastur, min-CVaR, inverse-vol | 0/20, equal weight wins | 3af |
| Financing-aware optimised weights (estimation-error-free mu) | still loses to equal weight | 3ag |
| Five champion lifts: range-vol, agreement, trend-age, param cloud, skew | 0/5 | 3ad |
| 6-component external stack | 0/29; high-vol taper is gold-specific | 3ac |
| ML/LLM predictors, turning-point detectors, meta-labelling | 5+ independent proofs of failure | 3r, 3o, oracle |
| Scale-in / averaging-down (54 variants) | it is leverage, not edge | 3ab |
| Fast/short-horizon XAU trend | real but dies on spread above ~$0.12 | 3f |

**Consequence:** do not spend regime-U budget re-running signal research hoping the missing
constraint changes the answer. It does not — none of these failed *because of* a drawdown rule.

### 1.2 Execution economics is the live lever (and the only one that has paid)
- **Financing is the largest single drag ever measured on this book**: -3.79%/yr, -0.29 Sharpe
  (§3q, FTMO rates).
- **GoldEternal (Maven) is swap-free** — `swap_mode=DISABLED`, spread $0.55 (1.23bp), tracks
  XAUUSD at **0.998** daily-return correlation. Log-basis drift +0.17%/yr (H1, 2,711 bars),
  +0.21%/yr (D1) inside a ~1% band: no embedded funding. Gold sleeve net Sharpe
  **+0.862 -> +1.065**, CAGR **+11.85% -> +15.04%**, for +0.33bp of extra spread. (§3ag)
- **Only 5 of Maven's 68 symbols are swap-free**: EURUSD, USDINR, GoldEternal, XAGUSD, USDNGN.
  **No swap-free index, crypto or energy** — so the book's gross Sharpe 1.313 is not
  recoverable at this venue. (§3ah)
- **Carry is broker-specific.** BRENT earns +9.6%/yr at FTMO and pays -4.79% long / -5.40%
  short at Maven. Never carry a financing assumption across venues.
- **Positive carry exists and is additive but is NOT backtestable here.** SUGAR +21.56%/yr,
  COCOA +14.75%. Additivity proven via NGAS (-165.9%/yr swap vs -0.99%/yr price drift over 14.8
  years => prices are spot-like). But these are a single snapshot of a time-varying rate and
  the repo has no historical commodity term structure. **Quote no number for commodity carry.**
  FX carry *is* testable — `data/rates_3m.csv` has historical 3m rates — and is still OPEN.
- Sleeve correlations, net of carry, business-day aligned: all pairs **+0.04 to +0.14**. The
  diversification is real in both regimes.

### 1.3 Method rules (hard-won; apply in both regimes)
1. **Matched-vol gating.** Any overlay that multiplies exposure by <=1 de-risks, so a raw
   paired-t marks a genuine risk-adjusted gain as a loss. Lever the candidate to the base's
   vol first (`vol_match`). Changed a verdict from 0/29 to 3/29 once.
2. **DSR is near-vacuous for overlays** — candidates inherit the base's edge, so trial variance
   collapses (logged DSR 0.998 with PBO 0.845). Gate on paired-t vs base plus PBO.
3. **Pre-register** schedule, assets and criterion before running. Prevented a cherry-picked
   positive in the taper replication.
4. **Costs in basis points, not dollars.** $0.10 is 0.002bp on BTC and 12bp on Brent; on EURUSD
   it produced an SR of -7.7. One-way bp = 0.75 x spread_bp. Floor CSV spreads — they understate
   live by 20-50x.
5. **The `D=6` trap.** `champion_signal` hardcodes 6 bars/day; on D1 data it silently becomes a
   96-1536 **day** trend follower. Use `champion_recipe(close, scale=1/6, conc_p=1.5)`.
6. **Align day counts before annualising.** BTCUSD quotes weekends. Comparing a 259-day/yr
   series to a 365-day/yr one with a common sqrt(252) inverted a book ranking on 2026-09-04.
   Compound every sleeve onto a common business-day index first.
7. **Synchronous closes** for any cross-sectional signal — a non-synchronous pair faked a
   Sharpe of 3.5 once.
8. **Split-sample control.** A pooled t of +2.25 with 7/9 positive years was still a post-2022
   artifact.

### 1.4 Ops facts
- Never `systemctl stop mt5-terminal.service` — its `ExecStop` has a bare
  `pkill -f terminal64.exe` + `wineserver -k` that kills every instance. The `-cent` and
  `-ftmo` units are prefix-scoped and safe to stop.
- Kill terminals only by verified `/proc/<pid>/environ` `WINEPREFIX`.
- `mt5linux` caches a module-level connection: **one port per process**, or a multi-port loop
  silently reuses the first connection and reports false results.
- MT5 **build** decides IPC health: `.mt5` runs the working Aug-28 build; `.mt5b` (cent, 18813)
  and `.mt5c` (FTMO, 18814) still run build 5836 from May-14 and return IPC timeout. Only
  `terminal64.exe` differs between the prefixes.
- `retcode 10026` = `TRADE_RETCODE_SERVER_DISABLES_AT` = autotrading disabled server-side.

---

## 2. Regime P — PROP-FIRM-CONSTRAINED findings

**These are parked, not deleted. Everything here is correct *given* a rule set.**

### 2.1 The core principle: match the book to the BINDING constraint
Pass rate is decided by *which rule fails you*, and the optimal book differs per rule set
(§3af):

| rule set | binding constraint | best book | measured |
|---|---|---|---|
| FTMO 2-Step @9% dial | **5% DAILY** | **widen to 8-9 sleeves** | pass 88.6% -> **94.1%**, fail_day 9.3% -> **0.0%** |
| FundingPips Flex @7% | neither binds | core 4 | 99.7% vs 99.1% wide — widening only dilutes |
| Maven (+4%, 10% static) | **10% MAX LOSS** | highest-Sharpe narrow book | see 2.3 |

Mechanism, measured: at a 9% dial the core-4 book's worst day scaled x1.5 (the floating-P&L
proxy) is **-5.07%**, which breaches a 5% daily limit; widening to 9 sleeves thins it to
**-4.72%**, inside the limit. Where no daily limit binds, that tail-thinning buys nothing and
return dilution dominates.

### 2.2 Regime-P-only concepts (meaningless in regime U)
- **Sharpe does not predict pass rate.** The dial-scaled LEFT TAIL versus the daily limit does.
  (§3ae book shootout.)
- **Tail insurance.** BRENT took FTMO `fail_day` from 13.9% to 0.0% — a reason to hold a sleeve
  that has nothing to do with its Sharpe.
- **`day_safety=1.5`** — the intraday floating-P&L proxy used in every pass sim.
- **risk_frac ceiling:** Sharpe is invariant +1.02 across 0.25-4%, but under a 10% wall use
  0.5% and never exceed 1.5%. 1% -> 2% doubles CAGR and 24x's breach risk.
- **Static vs trailing drawdown conventions** matter enormously: at 1.5% risk, 4.4% vs 17.9%
  breach probability.
- **The 5% DD mandate is infeasible**: the frontier is ~0.5-2.5% CAGR at 5% DD; Calmar is the
  wall; CPPI holds the floor but cash-locks.
- **Martingale/"recover all losses"** ruins 71-97% of the time even with a positive edge; only
  a gentle double at K3 survives a 10% wall, on a ~1-SE edge that loses to buy & hold.
- **Maven's stated 0% daily loss rule is infeasible** for any strategy — 41.7% of days lose.
  Exits must be managed manually.
- Weekend-flatten, Friday-flatten and news-window rules; FundingPips B-account restructure.

### 2.3 Regime-P dial tables (net of measured carry)
Maven, +4% target / 10% static max loss / no daily limit, gold-tilted book, 5 seeds x 1500 sims:

| dial | pass % | median months | blow-up % |
|---|---|---|---|
| 5% | 98.0 | 16.7 | 2.0 |
| 6% | 96.2 | 13.6 | 3.7 |
| **7%** | **93.8** | **11.1** | **6.1** |
| 8% | 91.5 | 9.2 | 8.5 |
| 9% | 89.1 | 7.8 | 10.9 |
| 11% | 84.0 | 5.7 | 16.0 |

### 2.4 Regime-P configs
- `configs/v5_maven_book.json` — 7% dial, gold-tilted, GoldEternal. **The regime-P deployable.**
- `configs/v5_ftmo_wide.json` — 8-sleeve wide book, **9% dial ONLY**. 4 of 8 symbols still need
  FTMO name verification. Do not use at 7%.
- `configs/v5_ftmo_challenge.json` — the original core-4 FTMO book (use at 7%).
- `configs/v5_xau_maven.json` — single-XAU fallback; switch its symbol to GoldEternal if used.

### 2.5 RESUME POINT for regime P
When you go back to a prop firm, pick up exactly here:
1. **Identify the binding rule first** by simulating which limit breaches first at your intended
   dial. Then set book WIDTH from 2.1: daily-limit-binding -> widen; max-loss-binding -> narrow
   and high-Sharpe. Do not default to the highest-Sharpe book.
2. **Set the dial from 2.3**, not from a Sharpe number.
3. **Route gold to a swap-free instrument** if the venue has one (regime-I finding, applies in
   both regimes) — worth +0.20 Sharpe on the gold sleeve.
4. **Re-measure the venue's carry map before sizing anything.** Do not reuse Maven's or FTMO's.
5. Unfinished regime-P work: verify the 4 unconfirmed symbols in `v5_ftmo_wide.json`; build the
   weekend/Friday-flatten feature the executor still lacks (needed for the FundingPips B
   account).

---

## 3. Regime U — UNCONSTRAINED findings (current)

**Objective:** maximise net Sharpe, then set leverage to the loss you are willing to take.
Drawdown is reported, never a constraint.

### 3.1 What changes the moment the rules come off
| question | regime P answer | regime U answer |
|---|---|---|
| book width | **widen** when a daily limit binds (4 -> 8/9 sleeves) | **narrow**: wide-7 scores +0.797 vs core-4's +1.096 — widening HURTS |
| dial | 5-9% | 20-40%+; the only limit is your tolerance |
| risk_frac | <=1.5% under a 10% wall | unbounded; Sharpe is leverage-invariant |
| what to optimise | dial-scaled left tail vs the daily limit | net Sharpe, full stop |
| why hold BRENT | tail insurance (fail_day 13.9% -> 0%) | its own net Sharpe of +0.437 |
| 5% DD mandate work | a real constraint study | **moot** |
| martingale/scale-in ruin analysis | binding | **moot** (but they still had no edge) |

### 3.2 The regime-U ranking (net of Maven's live carry, business-day aligned)
| book | net Sharpe | maxDD | 2018-21 | 2022-26 |
|---|---|---|---|---|
| gold only, XAUUSD | +0.862 | -22.8% | +0.42 | +1.22 |
| gold only, GoldEternal | +1.065 | -19.7% | +0.64 | +1.40 |
| core-4 equal weight | +1.096 | -10.5% | +1.27 | +0.93 |
| **core-4, gold 50% / others 16.7%** | **+1.212** | -11.5% | +1.11 | +1.30 |
| wide-7 equal weight | +0.797 | -9.1% | +0.77 | +0.82 |

Sleeve net Sharpes: GoldEternal +1.065, BTCUSD +0.570, BRENT +0.437, US100 +0.266, US500
+0.221, US30 +0.152, XAGUSD +0.073, GER30 -0.003.

**The tilt is a plateau, not a knife edge:** 40% gold +1.200, 50% +1.212, 60% +1.196, 70%
+1.167. It is defensible because gold's advantage is substantially CONTRACTUAL (zero carry) not
estimated. **But it is under this repo's own bar** — paired-t at matched vol is +0.93 vs gold
alone (4/9 years) and +0.75 vs equal weight (5/9), against a +1.50 threshold. Status: *real but
unproven.*

### 3.3 Leverage table (gold-tilted core-4)
| dial | CAGR | maxDD |
|---|---|---|
| 7% | ~8.5% | ~-4% |
| 20% | +24.9% | -26.1% |
| 40% | +49.8% | -46.1% |

**Sharpe is leverage-invariant, so this table is the whole of the "massive improvement".**
Removing prop-firm rules bought CAGR (~8.5% -> ~50%), not edge. The edge gain was
+0.862 -> +1.212 Sharpe, and that came from the swap-free instrument plus the tilt.

Backtest DD is close-to-close; real intraday excursions are worse.

### 3.4 Regime-U config
- `configs/v5_gold_max_sharpe.json` — gold-tilted core-4 on GoldEternal, 20% dial placeholder.
  **Breaches a 10% max-loss account by construction** — needs own capital, the HFM cent
  account, or a funded account with looser terms.

### 3.5 RESUME POINT for regime U
1. **Open:** FX carry using historical 3m rates (`data/rates_3m.csv`) — the one carry family
   that IS backtestable. Never tested with real broker swap numbers.
2. **Open:** the carry census on FTMO and HFM. Blocked on the build-5836 terminal fix
   (one-file `terminal64.exe` copy into `.mt5b`/`.mt5c` + restart those two prefix-scoped
   units). FTMO's gold carry is -6.6%/yr, so a swap-free equivalent there is worth MORE than at
   Maven.
3. **Open:** whether any venue offers a swap-free index or crypto. Maven does not; that is what
   caps the book at +1.212 instead of its gross +1.313.
4. **Closed, do not re-open:** everything in section 1.1.

---

## 4. Engine support for the two regimes

`scripts/v5_basket_challenge.py` hard-coded equal class risk (`/Nc`). Added 2026-09-04:
module-level `CLASS_W` plus a `class_weights` argument to `target_leverage()`, wired through
`scripts/v5_basket_challenge_exec.py` from `cfg["class_weights"]`. Weights are normalised inside
`build()`; `CLASS_W=None` reproduces the equal-risk path — verified by a byte-identical
before/after diff of `--backtest --model ftmo`.

**Switching regimes is a config change, not a code change:**

| field | regime P | regime U |
|---|---|---|
| `vol` | 0.05-0.09 (from the table in 2.3) | 0.20-0.40 |
| `classes` | width set by the binding rule | core 4 |
| `class_weights` | gold-tilted (same tilt is fine) | gold-tilted |
| `fp_symbol` for XAUCHAMP | swap-free instrument if the venue has one | same |

---

## 5. Corrections log

Errors made and fixed, recorded so they are not re-inherited:

1. **2026-09-03, `v5_maven_book.json` quoted 99.9% pass / 6.3 months at a 5% dial.** Those were
   GROSS of financing. Net of measured carry the same book at the same dial is 98.0% / 16.7
   months. Fixed; the table in 2.3 is authoritative.
2. **2026-09-04, "diversification is net-negative at Maven; gold alone wins."** Wrong — a
   day-count artifact. BTCUSD quotes weekends, so the book ran at ~365 days/yr and gold at ~259
   while both were annualised with sqrt(252). Business-day aligned, the book beats gold alone
   and the gold-tilted book beats both. This is now method rule 1.3.6.
3. **2026-09-03, a raw paired-t was used to gate de-risking overlays.** Self-caught; fixed by
   adding `vol_match`. Changed a round's result from 0/29 to 3/29. Now method rule 1.3.1.

---

_Last updated 2026-09-04. Companion documents: `V5_FINDINGS.md` (chronological detail, §-numbered),
`README.MD` (current state), `BlogPOST.md` (narrative)._
