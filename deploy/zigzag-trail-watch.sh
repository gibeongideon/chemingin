#!/bin/bash
# Minute-by-minute ratchet for the ZigZag trail (Switch 360601 on 18815, FundingPips 360597 on
# 18816). Ratchet ONLY — entries, max-hold and the model stay in the hourly zigzag-trail-* jobs.
# Exits without touching either bridge while the hourly state file says flat.
set -uo pipefail
cd /home/trader/MT5 || exit 1
export NOTIFY_NO_TELEGRAM=1   # 2026-11-06: phone gets only trade events + advisor; this job -> email
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/zigzag-trail-watch.log
mkdir -p data/v5_runs
exec 9>/tmp/zigzag-trail-watch.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) previous run still going, skipping" >> "$LOG"; exit 0; }
for c in configs/v5_zigzag_trail_switch.json configs/v5_zigzag_trail_fp.json configs/v5_zigzag_trail_darwinex.json; do
    $PY scripts/v5_zigzag_trail_watch.py --config "$c" --execute >> "$LOG" 2>&1 \
        || echo "$(date -u +%FT%TZ) $c exit=$?" >> "$LOG"
done
