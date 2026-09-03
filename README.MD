# MT5 V5 — XAU-centric prop-firm trend books

Systematic **trend-following books built to pass (and hold) prop-firm challenges** —
FTMO, FundingPips, Maven — trading gold plus a few uncorrelated diversifiers via MT5 on a
headless Linux/Wine VPS. The question that drives everything: *what book passes a 2-step
challenge at the highest probability and fastest median time, net of real broker costs?*

**Two documents matter most.** `V5_FINDINGS.md` is the settled research ledger — read its
**MANDATORY CONTROLS** before trusting any number in this repo. `BlogPOST.md` is the
plain-English write-up of what was found and what it cost to find out.

> The old EURUSD/USDJPY candle-champion path (`v5_validate_champion.py`, candle features)
> is **superseded**. The project pivoted to XAU trend books.

---

## Current state — 2026-09-03

**Nothing is trading right now.** Every trading timer on the VPS is disabled. This is a
deliberate, known state, not a fault:

| Account | Bridge | Book | Status |
|---|---|---|---|
| **FTMO-Demo 100K** (1514222432) | 18814 | XAU+BTC+NDX+BRENT @9% | **stopped 2026-08-28** by decision. All 8 positions closed for **+$3,215 realised**, final balance $102,944.17 (+2.9% into a +10% P1). Timers disabled. |
| **FundingPips 10K** (20414926) | 18812 | XAU(micro)+ETH+DJI30 | **dead** — broker returns *"Invalid account"* on login. Its bots had already been failing before this; its terminal prefix was repurposed for Maven. |
| **HFM cent live** (54939391) | 18813 | dual `ls` + `champ` | **drained to $0.12 USC** (~$0.001). Timer paused. Was a plumbing-validation account, never a returns account. |
| **Maven 100K** (10398574, *BNPL P1*) | 18812 | champion (long-only) | **built, logged in, verified — BLOCKED.** Orders rejected `retcode 10026` = *autotrading disabled by server*. Needs enabling in Maven's dashboard; nothing else is outstanding. |

**Maven, measured live:** spread **$0.44** (12 tick samples, stable — FTMO-class, *not*
raw/ECN), leverage 1:75, swap **−$45.33 long / +$25.25 short** per lot per night,
`stops_level` 0. Executor `scripts/v5_maven_champ.py` (magic 360571, config
`configs/v5_xau_maven.json`, `risk_frac` 0.5%) with a **once-per-day guard** so a manual
profit-take is never re-opened by the next hourly reconcile.

### Known infrastructure faults

- **Two of three MT5 terminals cannot serve the Python API.** After the 2026-08-28 reboot
  (kernel 6.8.0-124 → 6.8.0-138), the two **build 5836** instances (`.mt5b` cent, `.mt5c`
  FTMO) never reach "MCP started" and every `initialize()` returns `-10005 IPC timeout`.
  The **build 6140** instance (`.mt5`, now Maven) works. Terminal GUIs are fine; only the
  API channel is affected. Reviving them needs a build upgrade, not a restart.
- `mt5-terminal.service`'s `ExecStop` uses a bare `pkill -f terminal64.exe` + `wineserver -k`,
  which would kill **all** instances. Never `systemctl stop` it — kill by verified
  `/proc/<pid>/environ` `WINEPREFIX` instead.
- Maven's ini auto-login does **not** survive a terminal restart (the broker's server
  discovery is slower than the ini path's patience). A restart currently needs a one-time
  VNC login. Do not restart that terminal casually.

---

## The breakthrough: diversification, and only diversification

**The champion** (`src/v5/xau_dual_signals.py::champion_signal`) is a long-only blend of
EWMAC (16/32/64 vs 64/128/256 **days**) and 10/20/40-day breakout, concentrated at
`conc^1.5` with a small resting long tilt, run through a causal vol-target. On XAU H4, net
of a live $0.448 spread, it does **Sharpe ≈ 1.02–1.10, maxDD ≈ −11%** at 1% risk/trade.

Two levers produced that, and nothing else has since:

**1. Killing the shorts.** Re-confirmed independently in 2026-09 at a completely different
horizon: across **ten** signal families, the short side lost money on **all ten**, and
long-only roughly *doubled* every long/short Sharpe (e.g. EWMAC 12–60h: both +0.34,
long-only **+0.86**, short-only −0.58).

**2. Uncorrelated sleeves.** Re-measured cleanly with per-instrument basis-point costs:

| | Sharpe | maxDD |
|---|---|---|
| Gold alone | +1.098 | −19.5% |
| **4-sleeve book (XAU+BTC+NDX+BRENT)** | **+1.480** | **−8.0%** |

Higher Sharpe than any component and less than half the drawdown of the best one. NDX
standalone is a mediocre +0.52 and still earns its place — it contributes decorrelation,
not return.

### Two findings that changed how we size

**`risk_frac` is a pure risk dial with a violent asymmetry.** Sharpe is *invariant* at
+1.02 across 0.25%→4%; return scales linearly, breach probability does not:

| risk/trade | Sharpe | CAGR | maxDD | P(−10% in 12mo, trailing) |
|---|---|---|---|---|
| **0.50%** | +1.015 | +4.4% | −5.8% | **0.0%** |
| 1.00% | +1.024 | +9.0% | −11.2% | 1.9% |
| 2.00% | +1.024 | +17.8% | −21.5% | **45.9%** |

1%→2% doubles CAGR and multiplies 10%-drawdown risk by ~24×. **Use 0.5% under a 10% wall;
never exceed 1.5%.** Also: *static* drawdown (from starting balance) is far more forgiving
than *trailing* — 4.4% vs 17.9% breach at 1.5% risk. Sizing off the forgiving one by
accident is how books get too big.

**Sharpe does not predict prop-firm pass rate — the dial-scaled left tail does.** The two
deployed books are statistically *equivalent* (matched-vol paired t = −0.40) and 0.748
correlated, with ETH≈BTC and DJI30≈NDX as near-substitutes. What separates pass rates is
whether the vol dial pushes the worst day through the daily-loss limit:

| book | dial | worst scaled day ×1.5 | vs 5% daily | fail_day | pass |
|---|---|---|---|---|---|
| XAU+BTC+NDX | 7% | −3.96% | ok | 0.0% | **98.7%** |
| XAU+BTC+NDX | **9%** | **−5.09%** | **BREACH** | **13.9%** | 83.3% |
| +BRENT | 9% | −4.80% | ok | 0.0% | 96.8% |

**BRENT is tail insurance, not return** (on returns it is insignificant, t=+0.47; it takes
daily breaches 13.9%→0.0%). This independently reconstructed the deployed design decision.

---

## What did NOT work

The negative results are the bulk of the value here. Everything below was built, measured
honestly, and rejected — each is recorded in `V5_FINDINGS.md` with a "do not re-run" marker.

**Five consecutive overlay families on the champion, ~40+ configurations, all failed:**

| attempt | outcome |
|---|---|
| ADX-percentile regime **gate** | Sharpe 1.084→1.007, drawdown *worse*, paired t −2.88, pass 95%→80% |
| **scale-in / averaging down** (54 variants) | leverage in disguise — plain champion at higher `risk_frac` beat it on Sharpe, drawdown, Calmar *and* worst trade |
| **five untried lifts** (range-vol denominator, ensemble agreement, trend age, parameter cloud, skew) | 0/29 passed; the one that looked real was a post-2022 artifact and the live bot's ATR beat it anyway |
| **6-component stack** (vol regime, trend strength, HTF confirm, hours, cost filter) | 0/29 passed; 2 of 6 were *already implemented*, 1 was collinear with the champion at +0.85, 1 a category error, 1 had no bad session to remove |
| **high-vol taper** | best single result found (gold 1.098→1.311, both halves) — then failed pre-registered replication: 1 of 4 sleeves, and it *hurt* the portfolio (1.480→1.310) |

**Other closed lines:** 12 ICT/SMC concepts + 4 confluence combos (zero survivors);
turning-point bottom-detector as a scalp (all 48 cells negative); 4–15h long/short book
(shorts lose on 10/10 signals, cost wall); intraday/multi-timeframe (H4 beats M30/H1 *even
at $0.12 raw spread* — the intraday signal is genuinely worse, not merely cost-limited);
LLM-as-analyst (no edge); martingale (ruins 71–97% of runs); meta-labeling (failed twice).

**Maven's 0% daily-loss mandate is infeasible.** Backtested honestly: with tight stops
**41.7% of days** realise a loss; loosening stops trades frequency for severity (worst day
−5.0%). No parameterisation gets near "never lose on a day" — that constraint is
incompatible with taking directional risk at all. The account is therefore run with manual
exit discipline plus a 0.5% risk dial, not an automated zero-loss guarantee.

**A commercial EA ("Happy Gold – OxS") was reverse-engineered and proved un-replicable.**
The log is genuine (balance reconciles to the cent). Its direction rule was fully recovered
(momentum, t≈30) but carries **no standalone edge** (+$0.22 gross vs a $4.40 spread). The
edge is in *entry timing*, which is provably absent from 15-minute bars (every feature
t≤2.6 vs matched controls) and was **not** reproduced by tick-level burst triggers on 2.95M
real ticks. Four candidate mechanisms eliminated; see `strategy/REVERSE_ENGINEERING_REPORT.md`.

---

## Methodology — the actual product

Forty-plus failures are only informative if the measurement is trustworthy.
`V5_FINDINGS.md` holds the full MANDATORY CONTROLS list; the ones that changed answers:

- **Floor every cost at live broker rates.** The CSV spread column understates the venue by
  up to 10×. Measure the real spread by sampling ticks.
- **Benchmark against the incumbent, not zero.** Most overlays are positive-Sharpe and
  still worthless.
- **Gate de-risking overlays at MATCHED VOL.** Anything that multiplies exposure by ≤1 will
  show a *negative* raw paired-return test even when Sharpe and drawdown clearly improve.
  One variant read Sharpe 1.155 vs 1.084 and DD −13.4% vs −19.8% with a raw t of −2.52.
- **Split-sample everything, including things that pass.** A pooled t of +2.25 with 7-of-9
  years positive still turned out to be a post-2022 artifact.
- **DSR is near-vacuous for overlays.** When every candidate is a variation of an
  already-good base they all inherit its edge; we logged DSR 0.998 alongside PBO 0.845. For
  overlays the gate is the paired test against the base.
- **Probe for lookahead mechanically** (`src/evaluation/lookahead_probe.py`) — truncation
  invariance on synthetic data, run *before* trusting a backtest.
- **Pre-register the decisive test.** The taper replication would have produced a positive
  story if a per-asset schedule had been cherry-picked after the fact.
- **Watch the units.** Dollar constants don't transfer across instruments ($0.10 slippage is
  0.002bp on BTC and 12bp on Brent); bar-count lookbacks don't transfer across timeframes
  (`D=6` silently makes the champion a 96–1536 *day* system on daily bars). Both produced
  spectacular nonsense before being caught.

---

## Layout

**Engines** — `scripts/v5_basket_challenge.py` (champion recipe + portfolio vol-target),
`scripts/v5_basket_challenge_exec.py` (generic guard-first executor, `--config <firm>.json`),
`src/v5/xau_trend.py` (discrete trade-ticket engine) via `scripts/v5_xau_dual.py` /
`scripts/v5_maven_champ.py`.

**Configs** — `v5_ftmo_challenge.json` (4-asset @9%), `v5_fp_flex_10k.json` (10K book),
`v5_xau_maven.json` (Maven, 0.5% risk), `v5_xau_dual.json` (cent dual),
`v5_fundingpips_funded.json` (*planned* weekend-flat @6%),
`v5_xau_champ_fundingpips.json` (*planned* champ-only half-risk).

**Research** — `V5_FINDINGS.md` (ledger), `BlogPOST.md` (narrative),
`strategy/REVERSE_ENGINEERING_REPORT.md` (EA forensics), `FUNDED-STAGE-PLAN.md`,
`SPRINT-OPTIONS.md`, `CHALLENGEBOT.MD`. Session memory in
`~/.claude/projects/.../memory/` (`MEMORY.md` index) is the fast-recall layer for what is
deployed, disproven, and why.

---

## Immediate next actions

1. **Enable algorithmic trading for the Maven account** in Maven's dashboard — the only
   thing blocking a live champion there. Everything else is built and dry-run verified.
2. **Decide the FundingPips 10K account's fate** — the login is invalid; it needs renewing
   or writing off.
3. **Fund or retire the HFM cent account** — at $0.001 it cannot size a single trade.
4. **Upgrade the two build-5836 terminals** if their accounts are ever needed again; a
   restart alone will not fix the IPC fault.
5. If a Friday-flatten is wanted for FundingPips funded stage, that executor feature is
   still unbuilt (`fundingpips-b-account-plan` in memory).

## Ops gotchas

- **`np.float64` must never cross the rpyc bridge** — it ships to the Wine interpreter as
  literal `np.float64(...)` and is eval'd without numpy → `NameError`. Cast to plain `float`
  at the boundary (bit the cent bot's trailing stop and sizing twice).
- **Vendor trade logs may not share your feed's clock.** The reverse-engineered EA log ran
  **+2h** ahead; before correcting it, 43% of its trades looked "physically impossible."
  Always solve the offset by maximising in-bar fill consistency first.
- **Broker tick retention is ~1 month** — verified: ticks exist at 1 day/1 week/1 month and
  are *zero* at 3, 6, 9 and 11 months. Any tick-level study must be planned prospectively.
- Local dev MT5 on Wayland can die at launch (`nodrv_CreateWindow`); fix is `wineserver -k`
  then relaunch with the mutter Xwayland cookie, not `~/.Xauthority`.
- Any systemd unit launching Wine needs `KillMode=none`, or teardown kills the shared
  wineserver and takes the persistent terminal with it.
- **Prop firms can restrict a symbol mid-challenge without notice** (FundingPips set
  XAUUSDmicro CLOSE-ONLY once). The executor checks `trade_mode` before sending.
- This repo can have **concurrent Claude sessions** committing and switching branches; check
  `git reflog` before assuming a missing file is data loss.
