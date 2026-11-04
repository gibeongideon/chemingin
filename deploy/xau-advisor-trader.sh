#!/bin/bash
# XAUUSD ADVISOR EXECUTOR — Switch 40080430 (DEMO), bridge 18815, XAUUSDs, magic 360602. Every minute.
# Reads data/v5_runs/xau_advisor_state.json (written every 15 min by xau-advisor) and trades it:
# UP->BUY, DOWN->SELL; same direction = no order; on a flip the opposite leg is closed if in profit,
# else kept as a hedge; every stop is server-side and ratchets at 1.0 x H4 ATR20 behind the best
# price. The minute cadence is for the TRAIL and for acting promptly on a new state.
# Disarm: systemctl --user disable --now xau-advisor-trader.timer
set -uo pipefail
cd /home/trader/MT5 || exit 1
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/xau-advisor-trader.log
mkdir -p data/v5_runs
exec 9>/tmp/xau-advisor-trader.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) previous run still going, skipping" >> "$LOG"; exit 0; }
$PY scripts/v5_xau_advisor_trader.py --port 18815 --symbol XAUUSDs --execute >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
if [ "$(wc -l < "$LOG")" -gt 30000 ]; then tail -10000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"; fi
