#!/bin/bash
# TELEGRAM TRADE EVENTS — FTMO-Demo 1515233856 (bridge 18814): ZigZag 360608 (best variant) only.
# TRADABLE / TRADED / CLOSED; with --weekly the weekly P/L per bot.
# Disarm: systemctl --user disable --now trade-events-ftmo.timer trade-events-ftmo-weekly.timer
set -uo pipefail
cd /home/trader/MT5 || exit 1
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/trade-events-ftmo.log
mkdir -p data/v5_runs
exec 9>/tmp/trade-events-ftmo.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) previous run still going, skipping" >> "$LOG"; exit 0; }
echo "===== $(date -u +%FT%TZ) $*" >> "$LOG"
$PY scripts/v5_trade_events_notify.py --port 18814 --label FTMO \
    --bots "360608=ZigZag" "$@" >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
if [ "$(wc -l < "$LOG")" -gt 20000 ]; then tail -8000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"; fi
