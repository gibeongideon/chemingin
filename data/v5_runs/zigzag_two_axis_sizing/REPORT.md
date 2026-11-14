# ZigZag two-axis sizing study

## Ledger draft

### §3ev — ZigZag probability × downside-vol sizing (2026-11-14)

Control: Arm V with the existing daily circuit breaker and downside-vol dial.
Candidate: fixed-event causal probability × downside-vol multiplier, capped at 0.25–2.00.

CME: Sharpe +0.691 vs +0.628; matched t +0.86; shuffle p95 +0.656; shuffle p 0.017; 2x-cost Sharpe +0.511; half changes +0.061 / +0.061; CAGR +1.50% → +1.66%; maxDD -4.83% → -4.03%.
Switch: Sharpe +0.812 vs +0.729; matched t +0.91; shuffle p95 +0.685; shuffle p 0.003; 2x-cost Sharpe +0.610; half changes +0.199 / +0.001; CAGR +1.83% → +2.04%; maxDD -3.57% → -3.32%.

Verdict: CLOSED. No live configuration was modified.
