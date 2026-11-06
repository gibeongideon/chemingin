#!/bin/bash
# ZigZag + 3xATR trail — DARWINEX demo 3000112593, bridge 18817, magic 360606. Hourly.
# Parallel to 360601 on Switch: same config, same pooled model (gold + EURUSD/GBPUSD/USDJPY).
# Disarm: systemctl --user disable --now zigzag-trail-darwinex.timer
set -uo pipefail
cd /home/trader/MT5 || exit 1
export NOTIFY_NO_TELEGRAM=1   # 2026-11-06: phone gets only trade events + advisor; this job -> email
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/zigzag-trail-darwinex.log
mkdir -p data/v5_runs
exec 9>/tmp/zigzag-trail-darwinex.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) previous run still going, skipping" >> "$LOG"; exit 0; }
echo "===== $(date -u +%FT%TZ) =====" >> "$LOG"
$PY scripts/v5_zigzag_ftmo.py --config configs/v5_zigzag_trail_darwinex.json --live --execute >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
