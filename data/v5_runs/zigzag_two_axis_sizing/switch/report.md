# ZigZag probability × downside-vol sizing — SWITCH

Control: Arm V with its existing daily circuit breaker and downside-vol dial.
Candidate: fixed-event causal probability × downside-vol multiplier, capped at 0.25–2.00.

| Metric | Control | Candidate |
| --- | ---: | ---: |
| Sharpe | +0.729 | +0.812 |
| CAGR at matched risk | +1.83% | +2.04% |
| Max drawdown at matched risk | -3.57% | -3.32% |
| 2x-cost Sharpe | +0.516 | +0.610 |

Trades: 1396 (152.4/year); mean candidate multiplier: 0.986.
Matched-volatility paired t: +0.91. Conditional-shuffle p95: +0.685; p-value: 0.003.
Chronological-half Sharpe changes: +0.199, +0.001.

Promotion requires t ≥ +2.0, Sharpe above shuffle p95, both half changes positive, CAGR no lower, and maximum drawdown no worse.
PROMOTION: FAIL — Arm V retained.
Fixed-event assertion passed: entry times, exit times, and R outcomes are shared by control and candidate.
OFFLINE RESEARCH — no live configuration modified.
