#!/bin/bash
# ZigZag BEST variant (no lock, 3xATR trail, TP 4.3R) — FTMO-Demo 1515233856, bridge 18814, magic 360608.
# User request 2026-11-09: "place orders for ZigZag best only" on this account. Same rule as Darwinex 360606.
# Disarm: systemctl --user disable --now zigzag-trail-ftmo-b.timer
set -uo pipefail
cd /home/trader/MT5 || exit 1
export NOTIFY_NO_TELEGRAM=1   # phone gets this bot's trades via trade-events-ftmo, not its own chatter
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/zigzag-trail-ftmo-b.log
mkdir -p data/v5_runs
exec 9>/tmp/zigzag-trail-ftmo-b.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) previous run still going, skipping" >> "$LOG"; exit 0; }
echo "===== $(date -u +%FT%TZ) =====" >> "$LOG"
$PY scripts/v5_zigzag_ftmo.py --config configs/v5_zigzag_trail_ftmo_b.json --live --execute >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
