#!/bin/bash
# GOAT FUNDED real-time protector, every 2 min: daily guard (-2%), overall halt (-7.5%), phase target (+10%/+5%).
# --guard-only: evaluates the guards and FLATTENS on a breach; never opens. Shares /tmp/gft.lock with gft.sh.
set -uo pipefail
cd /home/trader/MT5 || exit 1
export NOTIFY_NO_TELEGRAM=1
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/gft-guard.log
exec 9>/tmp/gft.lock
flock -w 90 9 || { echo "$(date -u +%FT%TZ) lock busy 90s, skipping" >> "$LOG"; exit 0; }
echo "===== $(date -u +%FT%TZ) =====" >> "$LOG"
$PY scripts/v5_basket_challenge_exec.py --config configs/v5_gft.json --state data/v5_runs/gft_state.json \
    --paper-csv data/v5_runs/gft_guard_log.csv --port 18819 --live --execute --guard-only >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
if [ "$(wc -l < "$LOG")" -gt 40000 ]; then tail -15000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"; fi
