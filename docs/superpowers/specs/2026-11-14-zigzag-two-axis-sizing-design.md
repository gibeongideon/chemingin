# ZigZag Two-Axis Sizing Design

## Objective

Test one new ZigZag H1 sizing rule that combines the two distinct pieces of
information that have already shown the correct directional effect:

- the downside-volatility size dial from `V5_FINDINGS.md` §3eu; and
- the linear probability multiplier from §3eh.

The experiment asks whether their combination raises return quality at the
same average risk. It does not claim that the ZigZag detector is a validated
source of alpha. The current harness remains a demo research system.

## Scope

The control is the currently deployed Arm V trade specification:

- long-only H1 ZigZag fires at `P(bottom) >= 0.60`;
- 3.0 ATR initial stop and ratcheting trail;
- 4.3R take-profit and 120-bar maximum hold;
- 1.0 daily-ATR circuit breaker;
- one position at a time; and
- 0.25% baseline account risk per accepted trade.

The candidate changes only the lot-size multiplier. It must leave the entry
times, order of accepted trades, exit prices, and exit times unchanged.
Neither a new entry filter nor an exit change belongs in this experiment.

## Candidate sizing rule

At each accepted entry, let `p` be the pre-existing walk-forward ZigZag
probability at the entry bar. Define the probability component:

```text
m_probability = 0.5 + clip((p - 0.60) / 0.20, 0, 1)
```

Thus a fire at 0.60 receives 0.5 before normalisation, a fire at 0.70 receives
1.0, and a fire at 0.80 or above receives 1.5. This is exactly the previously
tested LIN shape; it is not refit or retuned here.

Let `downside_vol_500` be the standard deviation of negative returns over the
causal 500-bar window. Let `median_downside_vol_2000` be its causal rolling
median over 2,000 bars. Define the volatility component:

```text
m_volatility = 0.5  if downside_vol_500 >= median_downside_vol_2000
               1.0  otherwise
```

The unnormalised candidate multiplier is:

```text
m_raw = m_probability * m_volatility
```

To retain the intended 0.25% mean risk without using future trade outcomes,
normalise causally using accepted prior entries only:

```text
m_normalised(i) = m_raw(i) / mean(m_raw(j) for accepted entries j < i)
```

For the first accepted entry the denominator is 1.0. The final multiplier is
bounded to `[0.25, 2.00]`; its account-risk fraction is
`0.0025 * m_final`. The bounds prevent a thin probability tail from producing
a live-size outlier. They do not change exits or the event set.

The backtest must report the realised mean multiplier. All performance
comparisons use a matched-volatility daily return series as the decision
statistic, so a small residual difference in average risk cannot be mistaken
for improvement.

## Research data and controls

Use the existing causal, walk-forward probability streams and the existing
single-source histories:

- CME GC H1, from 2012 onward; and
- Switch XAUUSDs H1, from 2018 onward.

Keep their respective recorded spreads and swap costs. Re-run the candidate at
twice each source's round-trip cost. The event set must be asserted equal to
the Arm V control before calculating any statistic.

The only candidate is the formula above. Do not sweep probability knots,
volatility windows, the 0.5 high-vol multiplier, the bounds, or a take-profit.
The point is to test the interaction of two prior leads, not select a new
maximum.

## Validation and promotion criteria

For each source, report:

1. Arm V and candidate annualised Sharpe, CAGR, maximum drawdown, annual trade
   count, realised average multiplier, and results in chronological halves.
2. The matched-volatility paired daily-return t-statistic against Arm V.
3. A 2,000-draw conditional randomisation test that assigns the complete
   candidate multiplier vector to the same accepted trades at random. The test
   statistic is the candidate's Sharpe, compared with the shuffled
   distribution.
4. The candidate result at twice recorded costs.

Promotion requires all of the following in both CME and Switch:

- a matched-volatility paired t-statistic of at least `+2.0`;
- a candidate Sharpe above the 95th percentile of its conditional-shuffle
  distribution;
- a positive Sharpe improvement in each chronological half;
- CAGR at least equal to Arm V at matched mean risk; and
- maximum drawdown no worse than Arm V.

If any requirement fails, retain Arm V, document the result in
`V5_FINDINGS.md`, and do not alter a live configuration. A passed backtest is
still a research result: the correct next step would be a single demo A/B arm
against unchanged Arm V, not a funded deployment.

## Operational boundaries

The experiment is offline only. It must not send orders, alter MT5 JSON
configs, restart timers, or modify the existing `zigzag_relay_seen.json`
state. The user-owned change already present in that state file is outside this
work.

## Rationale

§3eu established a robust loss concentration in high downside-volatility
entries while holding the event set fixed. §3eh found a small probability-size
relationship with the correct sign in both data sources and both halves, but
not enough evidence to deploy by itself. The candidate tests their interaction
without relitigating exits, gates, or predictor accuracy.

The formula is a capped, conservative use of conditional information rather
than literal Kelly sizing. Kelly-style sizing presumes reliable probabilities
and odds; this detector does not meet that standard. The stringent two-source,
randomisation, and split-half gates protect against the type of backtest
selection bias already documented throughout this repository.
