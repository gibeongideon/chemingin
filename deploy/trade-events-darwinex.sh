#!/bin/bash
# TELEGRAM TRADE EVENTS — Darwinex demo 3000112593 (bridge 18817): Book 360605, ZigZag 360606,
# Trail-runner 360607. TRADABLE / TRADED / CLOSED; with --weekly the weekly P/L per bot.
# Disarm: systemctl --user disable --now trade-events-darwinex.timer trade-events-darwinex-weekly.timer
set -uo pipefail
cd /home/trader/MT5 || exit 1
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/trade-events-darwinex.log
mkdir -p data/v5_runs
exec 9>/tmp/trade-events-darwinex.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) previous run still going, skipping" >> "$LOG"; exit 0; }
echo "===== $(date -u +%FT%TZ) $*" >> "$LOG"
$PY scripts/v5_trade_events_notify.py --port 18817 --label Darwinex \
    --bots "360605=Book (XAU champion),360606=ZigZag,360607=Trail-runner" "$@" >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
if [ "$(wc -l < "$LOG")" -gt 20000 ]; then tail -8000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"; fi
