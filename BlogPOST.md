# A Sharpe 1.0 Gold Trend Follower, and the Forty Ideas That Failed to Improve It

*What we actually found building a systematic XAUUSD strategy — the signal, the portfolio
that beat it, and a graveyard of overlays that looked brilliant until they were measured
properly.*

---

## The short version

We built a long-only trend follower on XAUUSD (gold) 4-hour bars. Net of a real live
broker spread, it does a Sharpe of roughly **1.02–1.10** with a maximum drawdown around
**−11%** at 1% risk per trade. That is a genuine, deployable number, not a backtest
fantasy — it survives walk-forward selection, cost stress, and split-sample tests.

Then we spent months trying to make it better. **Five consecutive families of
improvement — more than forty distinct configurations — all failed.** Regime gates,
averaging down, machine-learned overlays, twelve Smart Money Concepts, volatility-regime
filters, intraday variants, even a frontier LLM reading the chart. Every one either
failed outright or turned out to be leverage in disguise.

The one thing that *did* work was not an overlay at all. Running the **same signal across
four uncorrelated assets** took the book from Sharpe 1.10 to **1.48**, and cut drawdown
from −19.5% to **−8.0%**.

That asymmetry — forty failed refinements versus one structural change — is the actual
lesson, and it is what this post is about.

---

## Part 1: The champion

### The signal

Nothing exotic. Two classic Carver-style trend components, blended:

**Component A — EWMAC (exponentially weighted moving average crossover)** at three
speeds. For each speed pair, take the difference between a fast and slow EMA, normalise
by price volatility, then rescale by its own expanding historical average so the forecast
has a stable unit scale:

```text
raw   = (EMA_fast(close) − EMA_slow(close)) / (close × σ_returns)
scalar= 1 / expanding_mean(|raw|).shift(1)          ← past-only, no lookahead
fc    = clip(raw × scalar, −4, +4)
```

Speeds are `(16,64)`, `(32,128)`, `(64,256)` **days** — implemented as ×6 in H4 bars, so
96/384, 192/768, 384/1536 bars. This is a multi-week to multi-month trend follower, not
an intraday system.

**Component B — Donchian breakout** at 10/20/40 days: where does price sit inside its own
recent range, smoothed, and normalised the same past-only way.

**The blend** takes the positive part of each, applies a concentration exponent of 1.5
(which sharpens conviction — strong signals get proportionally more weight than a linear
mapping would give), and adds a small resting long tilt:

```text
maxewbko = max(EWMAC⁺, breakout⁺)
champion = 0.5 × (conc(maxewbko, 1.5) × 0.8 + 0.15)
         + 0.5 × (conc(breakout⁺, 1.5) × 0.8 + 0.15)
         → clipped to [0, 2]
```

### The single most important design decision: no shorts

The forecast is clipped at zero. It never sells gold. This one constraint is what took the
strategy from mediocre to Sharpe ≥ 1 — internally we call it *"kill-the-shorts is THE
lever."*

We have since re-confirmed it the hard way. Testing a long/short variant across ten
different signal families at a faster horizon, **the short side lost money on all ten**:

| signal | long + short | long-only | short-only |
|---|---|---|---|
| EWMAC (12–60h) | +0.34 | **+0.86** | −0.58 |
| EWMAC (6–48h) | +0.18 | +0.76 | −0.65 |
| Zigzag structure | −0.03 | +0.61 | −0.81 |
| Breakout (6–24h) | −1.22 | −0.19 | −1.48 |

Long-only roughly **doubled** every long/short Sharpe. Gold has a structural upward drift
and violent, mean-reverting downside spikes; a trend follower shorting into that gets
whipsawed. Adding the short side does not diversify the long side, it dilutes it.

### Execution

Signals are decided on completed bar closes and filled at the *next* bar's open — never
intrabar, never on the bar that generated the decision. The live version:

- **Entry** when the forecast clears 0.50
- **Stop** at 3 × ATR(14); **trailing stop** 3 × ATR behind the peak once the trade is
  1 × ATR in profit
- **Sizing** = `risk_frac × equity ÷ (3 × ATR)`, quantised to broker lot steps, scaled by
  a confidence bucket (0.5× / 1.0× / 1.5×)
- Intrabar stop checks use the bar's high/low, and when both stop and target are touched
  in one bar, **the stop wins** — the conservative assumption

Turnover is about **24 position changes per year** (~3.3 trades/month). Positions are held
for weeks. This matters more than it sounds: it makes the strategy nearly immune to
spread, which is why it survives on retail-cost accounts where faster ideas die.

### Risk sizing: a clean dial, and a violent asymmetry

Because sizing is independent of the signal, `risk_frac` is a pure risk dial. Sharpe is
**invariant** across the whole range — same trades, just bigger:

| risk per trade | Sharpe | CAGR | max DD | worst day | P(−10% in 12mo, trailing) |
|---|---|---|---|---|---|
| 0.25% | +0.985 | +2.2% | −2.9% | −$824 | 0.0% |
| **0.50%** | +1.015 | +4.4% | −5.8% | −$1,656 | 0.0% |
| **1.00%** | +1.024 | +9.0% | −11.2% | −$3,286 | 1.9% |
| 1.50% | +1.024 | +13.4% | −16.5% | −$4,902 | 17.9% |
| 2.00% | +1.024 | +17.8% | −21.5% | −$6,517 | 45.9% |
| 4.00% | +1.023 | +34.9% | −39.0% | −$12,724 | 97.5% |

*(on $100k, gold H4, 2018+, costs floored at a live $0.448 spread)*

Return scales linearly. **Breach probability does not.** Going from 1% to 2% doubles CAGR
and multiplies the chance of a 10% drawdown by roughly **24×**. There is no free return
above ~1%, and we cap deployed accounts at 0.5%.

One subtlety worth flagging for anyone doing this against a prop-firm rule set: *static*
drawdown (measured from your starting balance) is far more forgiving than *trailing*
drawdown (from your high-water mark). An 11% dip after you are already up 15% never
touches a static floor. At 1.5% risk those two conventions give 4.4% versus 17.9% breach
probability — sizing off the forgiving one by accident is how books get too big.

---

## Part 2: The portfolio — the only thing that actually worked

Here is the finding that mattered. Take the **exact same signal, unchanged**, and run it
on four assets with different economic drivers, equal-weighted:

| sleeve | standalone Sharpe | one-way cost |
|---|---|---|
| XAUUSD (gold, H4) | +1.098 | 0.75 bp |
| BTC (crypto, D1) | +0.970 | 1.13 bp |
| BRENT (oil, D1) | +0.685 | 5.69 bp |
| NDX (US tech, D1) | +0.523 | 0.38 bp |
| **equal-weight book** | **+1.480** | — |

| | Sharpe | max DD |
|---|---|---|
| Gold alone | +1.098 | −19.5% |
| **4-sleeve book** | **+1.480** | **−8.0%** |

**The book has a higher Sharpe than any of its components and less than half the drawdown
of the best one.** NDX standalone is a mediocre +0.52 and it still earns its place, because
what it contributes is not return, it is *decorrelation*.

Two engineering details that are easy to get wrong here, and that we did get wrong first:

1. **Costs must be in basis points, not dollars.** A $0.10 slippage assumption is 0.002 bp
   on Bitcoin at $60,000 and **12 bp** on Brent at $80. Our first cross-asset run reported
   a Sharpe of −7.7 on EURUSD, which was not a finding, it was a unit error.
2. **Lookback periods must be rescaled per timeframe.** The signal hardcodes 6 bars/day
   for H4 data. Applied naively to daily bars it silently becomes a *96-to-1536-day* trend
   follower instead of 16-to-256. Same code, completely different strategy.

There is also a cost that backtests routinely miss: **financing**. Holding gold long costs
roughly **−$45 per lot per night** at our broker (measured live; interestingly, shorts
*earn* +$25). The 1.480 figure above does **not** include it — it charges spread and
slippage only. A separate study that did model it properly on a comparable four-asset book
measured the drag at **−3.79%/yr of equity**, which took that book's Sharpe from 1.46 to
**1.18** and CAGR from 14.2% to 11.2%. Expect a haircut of that order here too. Any
strategy holding positions for weeks that ignores swap is overstating itself.

---

## Part 3: The graveyard

This is the useful part. Everything below was built, measured honestly, and rejected.

### Overlays on the champion — five families, all dead

**Regime gating.** Cut exposure when a trend-strength indicator (ADX percentile) says
"not trending." Result: Sharpe 1.084 → 1.007, drawdown got *worse* (−17.0% → −19.9%),
paired t = −2.88, and prop-firm pass rate collapsed 95% → 80%. The champion's own
vol-targeting is already a *reactive* risk control; layering a *lagging* one on top just
mistimes the de-risking.

**Averaging down / scale-in.** Add to a losing position so the recovery is amplified. 54
variants. The verdict was not "it fails" but something more useful: **it is leverage in
disguise.** The best variant made +14.28% CAGR at −17.8% drawdown. Simply turning the risk
dial up on the plain champion made +13.41% at −16.5% — better Sharpe, better Calmar,
better worst-trade (−1.0R vs −2.4R). And the risk-neutral version, where the first leg is
sized down so the full stack carries the original risk, was *worse* than a single entry.
When the theoretically clean version of your idea underperforms, the idea is done.

**Twelve Smart Money Concepts.** Break of Structure, Displacement, Order Blocks, Breaker
Blocks, Mitigation Blocks, Fair Value Gaps, Inversion FVGs, Premium/Discount, Optimal Trade
Entry, Power of Three, Judas Swing, SMT Divergence — individually and in confluence
combinations. **Zero survivors.** Several were actively worse than random. Mitigation
Block's canonical "bounce off the zone" reading is *backwards* on gold: mirroring the
signal improved it.

**Machine learning on top of the signal.** A gradient-boosted classifier on regime features
does genuinely predict next-day direction at 52.0% versus a 49.8% persistence baseline —
statistically real, block-bootstrap confirmed. It also **cannot be monetised**, across five
separate trade structures. A 2-percentage-point edge at a one-day horizon does not survive
the spread. Detection accuracy is not trading edge; we have now confirmed this seven times.

**Five untried "smart" refinements.** Range-based volatility estimators, ensemble-agreement
conviction weighting, momentum life-cycle tapers, parameter-cloud averaging, skew
conditioning. 0 of 29 configurations passed. One of them nearly did, and how it failed is
instructive — see below.

### A six-component stack, reviewed

We were handed a well-argued proposal to layer six components onto the champion. The
audit was more interesting than the test:

- **Two were already implemented.** Volatility-scaled position sizing *is* the engine's
  position line. The transaction-cost filter *is* the existing no-trade band — and
  sweeping its width confirmed the deployed setting was already optimal.
- **One was collinear by construction.** The proposed trend-strength ratio
  `EWMA(r)/EWMA(|r|)` correlates with the champion's own forecast at **+0.85**, because
  the champion's EWMAC *is* a signal-to-noise ratio. The proposal warned against adding
  highly correlated indicators; its own component was one.
- **One was a category error.** "Confirm with a higher timeframe" assumes an intraday
  entry signal. The champion's speeds are already 16–256 *days*, so a daily confirmation
  filter sits *inside* what it already blends.
- **One had nothing to filter.** We measured P&L by hour of day rather than assuming:
  every single 4-hour bucket was profitable (+0.34 to +1.74 bp). There was no bad session
  to remove. And a weeks-long holder cannot be hour-filtered anyway — forcing it to
  respect trading hours cut Sharpe from 1.084 to **0.632**.
- **Stacking them all** produced the prettiest numbers on the page: Sharpe 1.236,
  drawdown −7.3%. It achieved that by holding **17% of normal exposure and sitting flat
  66% of the time.** Stacking many "reduce exposure" filters is mostly a decision not to
  trade. At matched risk it was insignificant.

### Two near-misses, and the controls that caught them

**A range-based volatility estimator** looked genuinely real: all four estimators beat the
baseline, each in 7 of 9 years, ranked *exactly* in order of their known statistical
efficiency. Pooled t-statistic +2.25. Then the split-sample: first half t = **−0.25**,
second half t = +3.14. The entire effect lived after 2022. **A pooled t of +2.25 with 7-of-9
years was still an artifact.** And it could not have reached the live system anyway — the
deployed engine sizes off true-range ATR, which is already range-based *and* gap-aware, and
it **beat** all four "better" estimators.

**A high-volatility taper** — cut exposure when volatility is in its top quintile — was the
best single result we found: gold Sharpe 1.098 → **1.311**, drawdown −19.5% → −12.4%, and
unlike everything else it improved in *both* halves. Mechanism verified, not assumed:
volatility targeting alone does *not* neutralise high-vol regimes, because trends and
volatility co-occur, so a strong forecast partly offsets the 1/σ scaling.

So we pre-registered the test that would settle it: same schedule, no per-asset
re-tuning, applied to the other three sleeves and the portfolio.

| sleeve | base | tapered | t |
|---|---|---|---|
| XAU | +1.098 | **+1.311** | +1.29 |
| BTC | +0.970 | +0.823 | −1.43 |
| NDX | +0.523 | +0.439 | −0.75 |
| BRENT | +0.685 | +0.561 | −0.83 |
| **portfolio** | **+1.480** | +1.310 | **−1.18** |

One of four sleeves improved. The portfolio got **worse**. The taper was gold-specific,
and at book level it was actively harmful — because the portfolio's drawdown was already
−8.0%, cross-asset diversification was already doing the smoothing, and volatility regimes
are *not* synchronised across gold, crypto, equities and oil. Tapering each sleeve
independently de-risks at different times and eats the diversification.

**A single-asset risk overlay can be genuinely useful standalone and still destroy value
inside a diversified book.** Pre-registration is what stopped us reporting a win here:
cherry-picking a different schedule per asset would have manufactured a positive story.

---

## Part 4: The methodology, which is the real product

Forty-plus failures are only informative if the measurement is trustworthy. What we
converged on:

**Floor every cost at documented live broker rates.** Our data's spread column understates
the real venue by up to 10×. We measured the live spread by sampling ticks: $0.44 on gold.
Several "edges" existed only in the gap between the CSV and reality.

**Benchmark against the incumbent, not against zero.** The right question is never "is this
profitable" but "does this beat what I already run." Most overlays are positive-Sharpe and
still worthless.

**Gate de-risking overlays at matched risk.** This one cost us a wrong answer before we
caught it. Any overlay that multiplies exposure by ≤1 will show a *negative* raw
paired-return test even when Sharpe and drawdown clearly improve. One variant showed Sharpe
1.155 vs 1.084 and drawdown −13.4% vs −19.8%, with a raw t of −2.52. You must lever the
candidate to the baseline's volatility *first*, then compare returns.

**Split-sample everything, even things that pass.** Pooled significance is not enough. Two
of our most convincing results were confined to one half of the sample.

**Deflated Sharpe is near-vacuous for overlays.** When every candidate is a variation of an
already-good base, they all inherit its real edge, so the cross-trial variance the
deflation corrects against is tiny. We recorded DSR 0.998 alongside a 0.845 probability of
backtest overfitting. For overlays, the paired test against the base is the gate.

**Probe for lookahead mechanically.** A causal function's output at bar *t* must be
identical whether or not bars after *t* exist. We test that by truncation invariance on
synthetic data, and it has caught real leakage that no eyeballing would have.

**Pre-register the decisive test.** Before running the taper replication we fixed the
schedule, the assets, and the success criterion in writing. Without that we would have
found a positive result, because one existed if you were willing to pick per-asset.

**Watch the units.** Dollar constants do not transfer across instruments. Bar-count
lookbacks do not transfer across timeframes. Both produced spectacular nonsense before
being caught.

---

## What we would tell someone starting out

A Sharpe of 1.0 from a boring EWMA crossover on one asset is **not** a signal that you need
more indicators. It is a signal that you have something real and should stop touching it.

The instinct is to refine: add a filter, a regime detector, a machine-learned overlay, a
smarter exit. We tried all of those, thoroughly, with honest measurement. The score is
roughly **0 for 40.**

What moved the needle was structural and almost boring: **run the same proven signal on
more uncorrelated things.** Gold alone, 1.10 at −19.5%. Four assets, 1.48 at −8.0%.

And be more suspicious of your good results than your bad ones. Every idea in the graveyard
above had a moment where it looked like it worked. The difference between a strategy and a
story is entirely in the controls you run *after* the number gets exciting.

---

*All figures are net of live-verified broker costs, evaluated from 2018 onward, walk-forward
selected where a parameter choice was involved. The champion runs live on real accounts;
the failures are documented in full, with code, so they are not re-attempted. Detailed
results live in `V5_FINDINGS.md` in this repository.*
