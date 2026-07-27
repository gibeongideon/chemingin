# MT5 V5 — XAU-centric prop-firm trend bots

Systematic **trend-following books built to pass (and hold) prop-firm challenges**
— FTMO, FundingPips — trading gold and a few uncorrelated diversifiers via MT5 on
a headless Linux/Wine VPS. The research question that drives everything: *what
book passes a 2-step challenge at the highest probability and fastest median time,
net of real broker costs?*

The single most important document is **`V5_FINDINGS.md`** — the settled research
ledger. Read its **MANDATORY CONTROLS** section before trusting any backtest
number in this repo; four different experiments here have fooled themselves
(optimistic CSV spreads, gold-beta masquerading as skill, universe-selection bias,
tiny-sample Sharpes), and the controls exist so it doesn't happen again.

> Note: the old EURUSD/USDJPY candle-champion path (`v5_validate_champion.py`,
> candle features) is **superseded**. The project pivoted to XAU trend books.

## Live deployments (VPS 68.183.91.240, user `trader`)

Three independent MT5 instances, one per account, isolated by Xvfb display +
rpyc bridge + magic number. Each runs a guard-first hourly reconcile.

| Account | Bridge | Book | Vol | Magic | Status |
|---|---|---|---|---|---|
| **FTMO-Demo 100K** | 18814 | XAU + BTC + NDX + BRENT (equal ¼) | 9% | 360561 | **live**, all 4 sleeves held, ~97% pass / ~10.5mo median |
| **FundingPips 10K** (FP-SIM1) | 18812 | XAU(micro) + ETH + DJI30 | vol-target | 360562 | live, **XAU sleeve blocked** (broker set XAUUSDmicro CLOSE-ONLY) → running 2 sleeves |
| **HFM cent live** (~$63) | 18813 | dual: `ls` (L/S) + `champ` (long-only) | 1% risk/trade | 360541 / 360542 | live validation account (plumbing, not returns) |

Core engines: `scripts/v5_basket_challenge.py` (`target_leverage`, the champion
recipe + portfolio vol-target) and `scripts/v5_basket_challenge_exec.py` (generic
guard-first executor, `--config <firm>.json`). The cent bots use the discrete
trade-ticket engine `src/v5/xau_trend.py` via `scripts/v5_xau_dual.py`.

## The strategy

**Champion recipe** (`champ_recipe_lo`): long-only, `conc^1.5` blend of
`max(EWMAC-mid, breakout-fast)` and breakout, run per-asset then combined
equal-class-risk and scaled to a target vol with a causal vol-target × drawdown
scaler. Long-only and diversification are the two big levers:

- Single-XAU Sharpe ceiling ≈ **1.06** (killing the shorts is what gets it there).
- Adding uncorrelated sleeves is the real win: **XAU+BTC+NDX ≈ 1.71**; the
  drawdown-safest honest book is **XAU+NDX+BRENT** (−7.7% DD, ~99% pass @7%).
- BRENT is the one genuine non-equity/non-crypto diversifier that survives real
  spread; BTC is needed for *return* (challenge) but dropped under a weekend-hold
  ban (funded).

## Configs

- `configs/v5_ftmo_challenge.json` — deployed FTMO 4-asset book @9% (a `vol`
  override field sets the dial; default lives in `MODELS`).
- `configs/v5_fp_flex_10k.json` — FundingPips 10K book.
- `configs/v5_xau_dual.json` — cent dual bots.
- `configs/v5_fundingpips_funded.json` — **planned** FundingPips-B funded book
  (XAU+NDX+BRENT weekend-flat @6%); needs a Friday-flatten executor feature.
- `configs/v5_xau_champ_fundingpips.json` — **planned** champ-only half-risk
  single-XAU (survives 5%/10% at 95.5%, but ~33mo median — kept for record).

## Research ledger

- **`V5_FINDINGS.md`** — every experiment run/measured/closed, with verdicts and
  "do not re-run" markers. Sections: champions trusted, disproven/dead, prop-firm
  account selection, mandatory controls, open threads.
- **`FUNDED-STAGE-PLAN.md`**, `SPRINT-OPTIONS.md`, `CHALLENGEBOT.MD` — plans.
- Session memory in `~/.claude/projects/.../memory/` (`MEMORY.md` index) — the
  fast-recall layer for what's deployed, disproven, and why.

Representative research tools (all cost-honest, all in `scripts/`):
`v5_ftmo_xau_diversifiers.py`, `v5_fp_xau_replacement.py`, `v5_xau_weekly_cycle.py`,
`v5_cushion_engine.py`, `v5_martingale_predictor.py`.

## Account roadmap

Buying a **FundingPips Option B** ($100K, 10%/6% targets, 12% max loss, 4% daily,
up to 95% split). Challenge = XAU+BTC+NDX+BRENT weekend-flat @7% (keep BTC to hit
target); funded = XAU+NDX+BRENT weekend-flat @6% (drop BTC). All FundingPips plans
restrict weekend holding, so the remaining build task is a **Friday-flatten** in
the executor. See `fundingpips-b-account-plan` in memory / `V5_FINDINGS.md` §3m.

## Ops notes

- **`np.float64` must never cross the rpyc bridge** — it is shipped to the Wine
  interpreter as literal `np.float64(...)` and eval'd there without numpy →
  `NameError`. Cast every numeric arg/return to plain `float` at the boundary
  (bit the cent bot's trailing stop and sizing twice).
- Local dev MT5 on Wayland can die at launch (`nodrv_CreateWindow`); fix =
  `wineserver -k` + relaunch with the mutter Xwayland cookie
  (`/run/user/1000/.mutter-Xwaylandauth.*`), not `~/.Xauthority`.
- Any systemd unit that launches Wine needs `KillMode=none` or its teardown kills
  the shared wineserver and takes the persistent terminal down with it.
- Prop firms can restrict a symbol mid-challenge without notice — the executor
  checks `trade_mode` before sending and the daily report flags restrictions.
