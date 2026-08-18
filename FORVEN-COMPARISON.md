# Forven vs this repo — approach comparison (pre-adoption due diligence)

**Why this doc exists:** the user asked to adopt `Trader36/Forven-main` into this branch
and borrow its techniques. This documents what Forven actually is and how its approach
differs from this repo's, so specific techniques can be borrowed deliberately instead of
importing a different philosophy wholesale. Written 2026-08-18, from reading Forven's
`README.md`, `AGENTS.md`, `docs/ARCHITECTURE.md`, and its `gauntlet/`, `policy.py`,
`strategies/`, `regime.py`, `portfolio_allocator.py`, `exchange/`, `bot_factory/`, and
`crucible_*` modules directly.

## TL;DR

Forven is a **~150-module full-stack service** (FastAPI + SvelteKit + SQLite + ChromaDB)
that runs an **autonomous AI-agent swarm** generating and validating **crypto** strategy
hypotheses through a **codified, machine-enforced gate pipeline**, trading paper-only on
Hyperliquid testnet by default. This repo is a **loose collection of Python research
scripts** where a human (with Claude as the researcher) hand-drives **one deep question at
a time** about a **prop-firm CFD/FX/gold book**, with rigor enforced by a **written
checklist** (`V5_FINDINGS.md` MANDATORY CONTROLS) that a person or Claude must remember to
apply, and bugs caught by **noticing a suspicious number**, not by a machine assertion.

Both projects independently arrived at the same core insight — *raw backtest Sharpe lies,
and you need automatic checks to catch how* — but Forven built that insight into enforced
code; this repo built it into incident memory and narrative discipline. That's the gap
worth closing, and it's a **narrow, portable** gap: statistical/validation utilities, not
the app around them.

## Side-by-side

| dimension | Forven | this repo (MT5) |
|---|---|---|
| **Asset universe / venue** | Crypto perps/spot, Hyperliquid via CCXT | Gold/FX/indices/commodities/crypto CFDs via MT5 (Wine bridge), prop-firm accounts (FTMO, FundingPips) |
| **Money at risk** | Paper + testnet by default; live exists but is unsupported, opt-in, disabled | Real live-money challenge/funded accounts, cent account, VPS-deployed bots |
| **Strategy generation** | Multi-agent swarm (strategy-developer/quant-researcher/risk/execution) + autonomous hypothesis harvesting from YouTube/Reddit/forums/GitHub | One human + Claude, one hypothesis at a time, driven by conversation |
| **Strategy breadth** | ~75 built-in named TA/quant strategies (`forven/strategies/builtin/`) as raw material for the pipeline to winnow | A handful of bespoke, deeply-scrutinized signal recipes (EWMAC+breakout champion, regime classifiers) |
| **Validation rigor** | Codified, dependency-ordered DAG (`gauntlet/definition.py`): quick_screen → timeframe_sweep → gate → optimization → confirmation → **walk_forward → cost_stress → monte_carlo → regime_split → parameter_jitter** → paper_promotion_gate. Each step's evidence is machine-checked for being non-vacuous (`gauntlet/legitimacy.py`) before the next step runs. | `V5_FINDINGS.md` "MANDATORY CONTROLS": 5 written rules (live-cost floor, buy&hold benchmark, walk-forward the universe, quote an SE, oracle-ceiling-first) that a human/Claude must remember to apply each session. No code enforces them. |
| **Selection-bias correction** | Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014), fully implemented (`gauntlet/deflated_sharpe.py`) — corrects for optimizer trial count **and** for swarm-level sibling-hypothesis attempts (a survivor's effective look-elsewhere count includes its disproven same-cluster siblings) | Lo (2002) standard-error formula quoted ad hoc in individual scripts; no persistent trial-count tracking |
| **Lookahead/leakage detection** | Automated **truncation-invariance probe** (`strategies/lookahead_probe.py`): reruns a strategy's vectorized signal generator on a right-truncated frame and flags any interior-bar signal that changed. Runs at registration time, before any backtest is trusted. | Caught by hand, repeatedly, this session alone: a `merge_asof` holiday-calendar bug, an `fillna(0)` zero-fill that silently under-weighted a portfolio for 3 years, an HMM smoothing-vs-filtering leak. All caught by noticing an implausible number, not by an assertion. |
| **Regime handling** | `regime.py`: ADX + EMA(20/50/200) + ATR-ratio + RSI → 4 discrete states (TREND_UP/DOWN/RANGE_BOUND/HIGH_VOL), 5-min-cached, used as a **hard gate** — a strategy declares `compatible_regimes` and is blocked outside them | This session's regime research used near-identical raw ingredients (Hurst/ADX/Choppiness/ATR-ratio) as **continuous ML features**, not a discrete gate; found a real but small (+2pp) accuracy edge that did not survive as P&L in any trade structure tested (`xau-regime-features-fwd-accuracy` memory) |
| **Position sizing** | One shared module (`strategies/sizing.py`), 5 modes (fixed/full/fraction/atr/kelly incl. a real Kelly-fraction calculator), identical formulas in backtest AND live/paper by construction | One convention throughout: continuous vol-targeted forecast, clipped, with a no-trade buffer band (`engine()` in `v5_xau_turn_prob.py`, reused everywhere) |
| **Portfolio construction** | `portfolio_allocator.py`: risk-parity-style, measures **realized** per-strategy vol + pairwise correlation from live paper trades (falls back to conservative 1.0 correlation when unmeasurable); explicitly never touches paper sizing, only live, behind a default-off flag | `v5_basket_challenge.py`: equal-class-risk + portfolio vol-target + drawdown-scaler over a **hand-picked, fixed** small asset set, correlation measured from **historical closes only** |
| **Hypothesis dedup / "graveyard"** | DB-backed: `disproven_cluster_count()` semantically clusters (family × asset) and blocks re-proposing a disproven thesis within a lookback window | `V5_FINDINGS.md` "Open threads → do not re-open" list + Claude Code memory files — human/AI-readable, narrative, not queried programmatically |
| **Execution risk controls** | Kill-switch at 10%/5% drawdown from high-water mark, loss cooldowns, min risk:reward enforcement, failed-open retry backoff, liquidation-distance monitoring (`exchange/risk.py`) | Prop-firm daily-loss/max-loss guards (`challenge_guards`), 3%-per-position SL, spread-abort thresholds on the fast-trend bot |
| **Persistence** | SQLite (strategies, backtests, lifecycle, trades) + ChromaDB (semantic memory) — structured, queryable | Markdown ledger (`V5_FINDINGS.md`) + Claude Code memory files — narrative, greppable but not queryable; git-versioned |
| **Deployment model** | Always-on FastAPI service + background workflow engine (retry backoff, gate-contention handling for capital slots) + SvelteKit dashboard for operator approval | systemd timers running discrete Python scripts on a VPS; the user runs commands, I run backtests on request |
| **License** | AGPL v3 | (repo's own license/terms, not AGPL) |

## What's genuinely worth borrowing (concrete, ported as utilities — not the app)

Ranked by how directly each addresses a failure mode this repo has actually hit:

1. **The lookahead/truncation-invariance probe.** This is the single highest-value idea
   in Forven for this repo. Every bug this session that mattered (`merge_asof` calendar
   corruption, `fillna(0)` zero-fill dilution, HMM smoothing leakage) was caught because a
   number looked wrong, not because anything checked for it automatically. A stripped-down
   version — rerun a candidate signal function on a right-truncated `close`/`OHLC` series
   and assert interior-bar output is unchanged — is directly applicable to every
   `*_features()` function in `scripts/v5_xau_intermarket_accuracy.py` and would have
   caught the HMM leak class of bug in seconds instead of a manual review pass.
2. **Deflated Sharpe Ratio, formally.** This repo already gestures at the same problem
   (MANDATORY CONTROL #3/#4, Lo(2002) SE) but has no persistent trial-count ledger — every
   SL/TP grid search, Hurst-window sweep, and instrument screen this session picked a
   "best cell" without formally deflating for how many cells were scanned. `gauntlet/
   deflated_sharpe.py` is a complete, dependency-light (stdlib `math` + scipy for the
   inverse-CDF) implementation; porting `deflated_sharpe_ratio()` and
   `probabilistic_sharpe_ratio()` as-is would upgrade "quote an SE" into an actual
   selection-bias-corrected probability.
3. **A codified, ordered validation checklist with non-vacuous evidence checks.** Not the
   async DAG engine (this repo doesn't need a background scheduler for one-off research
   scripts) — just the *idea* in `gauntlet/legitimacy.py`: for each control, a function
   that inspects the actual payload and rejects a check that technically ran but proved
   nothing (e.g. "cost stress reran but produced no finite Sharpe" is caught explicitly,
   not just "cost_stress: done"). This repo's MANDATORY CONTROLS are prose; a `verify_*()`
   helper per control that asserts on vacuous evidence would catch the class of mistake
   where a control is *invoked* but not *satisfied*.
4. **Realized (not just historical) correlation for portfolio construction**, if this repo
   ever runs the same book live long enough to have a return history — `portfolio_
   allocator.py`'s pattern of falling back to a conservative assumption (corr=1.0) when a
   pair's realized-return overlap is too short is a good, honest default this repo's basket
   work doesn't currently have (it only ever measured historical-close correlation).
5. **A structured, queryable disproven-hypothesis ledger**, if research volume ever
   outgrows what `V5_FINDINGS.md` can hold legibly. Not urgent — the file is currently
   well-organized — but the DB-backed semantic-cluster dedup is a real answer to "did we
   basically try this already" at a scale grep can't handle.

## What this repo has that Forven has no equivalent for — don't discard these

- **Prop-firm challenge realism.** FTMO/FundingPips daily-loss and max-loss rules, phase
  targets, weekend-flat handling, and a Monte-Carlo pass-rate simulator (`fp_sim`) tuned to
  those exact rules. Forven trades crypto perps with no analogue to "you fail if you touch
  a 5% daily loss line measured on floating P&L" — this machinery has no Forven counterpart
  to borrow from, because Forven doesn't operate in a domain that needs it.
- **Live-cost floors learned the hard way, per instrument.** The MANDATORY CONTROLS #1
  table (NATGAS CSV understates live cost 50×, HEATOIL 20×, PALL 11×) came from actually
  getting burned on a specific broker's real spreads. Forven's crypto/CCXT venues have
  comparatively transparent, liquid order books; it has no equivalent hard-won cost table
  because it hasn't needed one.
- **Narrative incident memory.** `V5_FINDINGS.md` and Claude Code memory don't just record
  *that* something failed — they record *why*, in enough detail that a future reader (human
  or AI) can judge whether a new, superficially-similar idea is actually the same trap or
  genuinely different (e.g. this session's own explicit reasoning for why option-b's
  longer-horizon test was NOT just re-running the disproven turning-point work). Forven's
  DB stamps (`deflated_sharpe_at`, `disproven` status) are precise but not narrative — they
  answer "was this tried" better than "why did it fail, and does that reasoning still
  apply."
- **A single, deeply-audited signal family.** The champion EWMAC+breakout recipe has
  survived stress tests, cost floors, walk-forward, and five separate attempts to improve
  it (§3q: 8 approaches, all dead) — it is trusted because it has been attacked
  relentlessly, not because it's one of many candidates a pipeline happened to promote.
  Forven's ~75 built-in strategies are breadth-first by design; this repo's few signals are
  depth-first. Neither is wrong, but importing Forven's philosophy wholesale would dilute
  what makes this repo's champion trustworthy.

## Adoption cautions — read before copying code, not after

- **License.** Forven is AGPL v3. AGPL §13 requires that if a modified version is run as a
  network service, the corresponding source must be made available to that service's
  users. This repo currently runs as local scripts + systemd timers (not a network
  service), so straightforward technique-porting (rewriting the DSR formula, adapting the
  lookahead-probe algorithm) is not the same as vendoring Forven's actual source files
  wholesale — but if any borrowed *file* (not just the underlying idea) is copied in and
  this repo's bots are ever exposed as a network service, the AGPL obligation attaches.
  Safest path: re-implement the ALGORITHM (already documented above with enough
  specificity to do so) rather than copy-pasting Forven's files.
- **Venue mismatch is real, not cosmetic.** Forven's regime detector, sizing module, and
  gauntlet all assume a continuously-traded, deep-liquidity crypto perp market. This repo's
  H4/D1 data has weekends, session structure, holiday gaps, and per-broker spread
  floors baked into nearly every script (`load_h4`'s expanding-median spread floor,
  weekend-flat handling, the `merge_asof` calendar-alignment bugs fixed this session). A
  ported technique (e.g. the lookahead probe) needs its synthetic test data to reflect THIS
  market's structure, not crypto's 24/7 one.
- **Scale mismatch.** Forven is a service with a DB schema, a frontend, and a background
  scheduler because it manages an open-ended, growing population of strategies across an
  autonomous swarm. This repo manages a small, fixed set of hand-vetted signals. Adopting
  the *infrastructure* (FastAPI app, SQLite schema, SvelteKit dashboard) would be solving a
  problem this repo doesn't have, at a real maintenance cost. Adopting the *algorithms*
  (items 1-5 above) as standalone `scripts/` or `src/` utilities solves problems this repo
  demonstrably does have, at a small, bounded cost.

## Recommendation

Borrow the five algorithms above as **standalone Python utilities** added to this repo
(likely `src/validation/` or similar — a new module, not a rewrite of anything existing),
starting with the lookahead-truncation probe (highest value, lowest effort, directly
prevents a bug class this repo has hit three times in one session) and the Deflated Sharpe
Ratio (directly upgrades every "grid search, then quote an SE" pattern already in use).
Do **not** adopt Forven's app/service/DB/frontend layer, its crypto-only exchange
integration, or its autonomous-swarm hypothesis-generation model — none of those address a
gap this repo has, and all of them would import a different project's operating philosophy
wholesale rather than a specific, useful technique.
