#!/bin/bash
# TELEGRAM TRADE EVENTS — the only bot messages on the phone (user request 2026-11-06).
# Read-only watcher of Switch 40080430 (bridge 18815) for the three order-placers:
# Book 360595, ZigZag 360601, Trail-runner 360600. Sends TRADABLE / TRADED / CLOSED.
# With --weekly (Saturday timer) it sends the weekly P/L per bot instead.
# Disarm: systemctl --user disable --now trade-events.timer trade-events-weekly.timer
set -uo pipefail
cd /home/trader/MT5 || exit 1
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/trade-events.log
mkdir -p data/v5_runs
exec 9>/tmp/trade-events.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) previous run still going, skipping" >> "$LOG"; exit 0; }
echo "===== $(date -u +%FT%TZ) $*" >> "$LOG"
$PY scripts/v5_trade_events_notify.py --port 18815 "$@" >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
if [ "$(wc -l < "$LOG")" -gt 20000 ]; then tail -8000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"; fi
