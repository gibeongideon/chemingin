#!/bin/bash
# Daily: (1) record accrued swap on every position -- the 3bd A/B that only Switch can close,
#        (2) report whether any held future needs rolling and whether a successor even exists.
# REPORT ONLY. No --execute, so this can never place an order; a roll is a human decision because
# Switch lists no successor gold contract yet and the fallback is spot XAUUSDs, not "go flat".
set -uo pipefail
cd /home/trader/MT5 || exit 1
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/futures_roll/roll-check.log
mkdir -p data/v5_runs/futures_roll
echo "===== $(date -u +%FT%TZ) =====" >> "$LOG"
$PY scripts/v5_swap_observe.py --port 18815 >> "$LOG" 2>&1
$PY scripts/v5_futures_roll.py --port 18815 >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
