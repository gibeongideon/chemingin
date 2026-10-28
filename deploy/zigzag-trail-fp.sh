#!/bin/bash
# ZigZag bottom detector + 3xATR TRAILING stop on FUNDINGPIPS-Trial (bridge 18816, magic 360597).
#
# RUNNING TO OBSERVE A FIX, NOT BECAUSE THE SIGNAL IS EXPECTED TO PAY. The ratchet was broken from
# deployment until 2026-10-28 (an undefined `bid` raised NameError on every pass with an open
# position: 291 passes, 0 ratchet decisions, process exiting 1 each time), so the trail had never
# actually trailed and the max-hold exit never ran. That is fixed. The CEILING is unchanged: 3bn
# proved a wide trail approaches buy-and-hold FROM BELOW and can never beat it, because the limit
# of an infinitely wide trail IS holding; and 3bv closed the family on mechanism (oracle 0.497 at
# RR=1 against a required 0.503 -- zero headroom before any model exists).
set -uo pipefail
cd /home/trader/MT5 || exit 1
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/zigzag-trail-fp.log
mkdir -p data/v5_runs
exec 9>/tmp/zigzag-trail-fp.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) previous run still going, skipping" >> "$LOG"; exit 0; }
echo "===== $(date -u +%FT%TZ) =====" >> "$LOG"
$PY scripts/v5_zigzag_ftmo.py --config configs/v5_zigzag_trail_fp.json --live --execute >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
