#!/bin/bash
# PDH DAY-TRADE EXECUTOR (V5_FINDINGS §3ex, deployment §3ey) — FTMO-Demo (BTC-PDH fill probe), bridge 18814,
# symbol BTCUSD, magic 360621 (2026-11-28, data/v5_runs/btc_pdh/PREREG.md Stage 2). Weekdays only, no holiday skip.
#
# One resting BUY STOP per server day at the prior server-day high; long only; stop 1.0 x ATR14
# attached server-side; the order expires and the position is closed by 23:40 server = 16:40 New
# York, before the rollover -> no swap, never overnight; 0.5% of BALANCE risked per trade.
# Pre-registered kill rules live in the script: mean entry slip > 2bp over the last 40 fills, or a
# -25R drawdown -> no new entries (an open position is still taken flat).
#
# Trades reach the phone through trade-events (trade-events-ftmo, 360621 in its --bots); this job's own mail carries
# problems only (vanished order, stale or failed exit, kill rule).
# Report:  $PY scripts/v5_pdh_executor.py --report
# Disarm:  systemctl --user disable --now pdh-ftmo-btc.timer   — do it while FLAT: an open position
#          keeps its server-side stop but nothing would close it at 23:40.
set -uo pipefail
cd /home/trader/MT5 || exit 1
export NOTIFY_NO_TELEGRAM=1
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/pdh-ftmo-btc.log
mkdir -p data/v5_runs
exec 9>/tmp/pdh-ftmo-btc.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) previous run still going, skipping" >> "$LOG"; exit 0; }
echo "===== $(date -u +%FT%TZ) =====" >> "$LOG"
$PY scripts/v5_pdh_executor.py --port 18814 --symbol BTCUSD --magic 360621 --tag ftmo_btc --weekdays-only --ignore-holidays --execute >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
if [ "$(wc -l < "$LOG")" -gt 40000 ]; then tail -15000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"; fi
