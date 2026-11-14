# ZigZag probability × downside-vol sizing — CME

Control: Arm V with its existing daily circuit breaker and downside-vol dial.
Candidate: fixed-event causal probability × downside-vol multiplier, capped at 0.25–2.00.

| Metric | Control | Candidate |
| --- | ---: | ---: |
| Sharpe | +0.628 | +0.691 |
| CAGR at matched risk | +1.50% | +1.66% |
| Max drawdown at matched risk | -4.83% | -4.03% |
| 2x-cost Sharpe | +0.438 | +0.511 |

Trades: 2185 (142.1/year); mean candidate multiplier: 0.983.
Matched-volatility paired t: +0.86. Conditional-shuffle p95: +0.656; p-value: 0.017.
Chronological-half Sharpe changes: +0.061, +0.061.

Promotion requires t ≥ +2.0, Sharpe above shuffle p95, both half changes positive, CAGR no lower, and maximum drawdown no worse.
PROMOTION: FAIL — Arm V retained.
Fixed-event assertion passed: entry times, exit times, and R outcomes are shared by control and candidate.
OFFLINE RESEARCH — no live configuration modified.
