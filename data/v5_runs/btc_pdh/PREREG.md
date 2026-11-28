# BTC-PDH: second-source replication + fill-latency audit, then a live fill probe (pre-registered 2026-11-28)

**Origin.** §3fa #6: the frozen PDH rule on OANDA M5 BTCUSD (2016+) scored net +1.46 (t_a 4.72, halves 2.30/0.48), the strongest leg in a 48-cell family (p 0.0000). One source only, M5 resolution.

## Stage 1: historical (Binance spot BTCUSDT 1-minute, 2017-08 → 2026-10, `data/binance_1m/BTCUSDT.npz`)
- **Rule.** UNCHANGED from the live executor and `scripts/v5_pdh_m1_stress.run`:
  - day = 17:00 → 17:00 New York
  - resting buy stop at the prior day's high; skip a day that opens at/above it
  - stop 1.0 × ATR14(D1)
  - flat at 16:40 New York
- **Entry variants (the latency audit):**
  - `stop` = resting stop, filled at max(level, bar open) (PRIMARY)
  - `mclose` = market at the open of the minute after the first 1-minute close above the level
  - `delay5` = 5 minutes late
- **Day sets.** Weekdays only (PRIMARY, matching the OANDA days) and all 7 days.
- **Costs.** Round trip 4.5bp (PRIMARY), 1.0bp (FTMO-class BTC quote), 2× each.
- **Statistics.** 1/vol weighting and net Sharpe exactly as `v5_pdh_m1_stress.stats`, halves, t_a vs matched hold.
- **Stage-1 PASS (gates Stage 2):**
  - PRIMARY net Sharpe ≥ +0.50
  - both halves positive
  - t_a ≥ 2
  - `mclose` (a bot that is one minute late) still net-positive at 4.5bp
- **KILL.** Primary net ≤ 0, or either half negative. Then BTC-PDH stops here and nothing is deployed.

## Stage 2: live fill probe (only if Stage 1 passes)
- **Where.** The deployed PDH executor, unchanged except for flags: FTMO-Demo (`BTCUSD`, magic 360621, tag `ftmo_btc`), 0.5% risk, so its fills and slippage are logged. Weekdays only, matching Stage 1 PRIMARY.
- **Kill (already in the code).** Mean entry slippage over the last 40 resting-stop fills > 2bp, or −25R.
- **Promotion review** after 40 fills: slippage < 2bp AND realised gross R within the Stage-1 band. Only then is a funded or prop account considered.
