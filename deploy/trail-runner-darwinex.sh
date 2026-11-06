#!/bin/bash
# Trail-runner (buy-stop at the prior-day high + 3xATR trail) — DARWINEX demo, bridge 18817,
# magic 360607, own state/log files (--tag darwinex). Every 5 minutes.
# Disarm: systemctl --user disable --now trail-runner-darwinex.timer
set -uo pipefail
cd /home/trader/MT5 || exit 1
export NOTIFY_NO_TELEGRAM=1   # 2026-11-06: phone gets only trade events + advisor; this job -> email
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/trail-runner-darwinex.log
mkdir -p data/v5_runs
exec 9>/tmp/trail-runner-darwinex.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) previous run still going, skipping" >> "$LOG"; exit 0; }
echo "===== $(date -u +%FT%TZ) =====" >> "$LOG"
$PY scripts/v5_trail_runner.py --port 18817 --symbol XAUUSD --magic 360607 --tag darwinex --execute >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
if [ "$(wc -l < "$LOG")" -gt 30000 ]; then tail -10000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"; fi
