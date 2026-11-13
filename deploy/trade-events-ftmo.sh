#!/bin/bash
# TELEGRAM TRADE EVENTS — FTMO-Demo 1515233856 (bridge 18814): ZigZag 360608 (best variant)
# plus the two advisor arms, 360611 (long/short) and 360612 (long-only).
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
    --bots "360608=ZigZag B · full size,360613=ZigZag V · vol taper,360611=Advisor-LS,360612=Advisor-L" --milestones "50,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500" --no-sl-hint "$@" >> "$LOG" 2>&1
# ^ ONE comma-separated argument. Three separate quoted words (2026-11-10 18:59 -> 11-11 11:00,
#   477 runs) made argparse exit 2 on every pass, so NO FTMO trade event reached Telegram.
echo "exit=$?" >> "$LOG"
if [ "$(wc -l < "$LOG")" -gt 20000 ]; then tail -8000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"; fi
