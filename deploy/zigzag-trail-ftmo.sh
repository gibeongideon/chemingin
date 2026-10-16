#!/bin/bash
# TOP/BOTTOM DETECTOR with a 3.0xATR TRAILING STOP and NO take-profit — FTMO 1514968075,
# bridge 18814, magic 360593, risk 0.25%.
#
# *** READ THIS BEFORE TRUSTING IT. *** The detector is NOT a validated edge. V5_FINDINGS 3bl:
# all six fixed-bracket cells are negative GROSS, before any cost; signed max t -0.098 against a
# null-of-the-maximum median of +0.925 (Reality-Check p 0.820); PBO 0.425. A previous live run of
# the fixed-bracket version took a demo from $100,000.00 to $99,564.82 in five orders.
#
# WHY THIS SHAPE IS NONETHELESS THE RIGHT ONE TO TEST. 3bo measured the detector with a trailing
# stop: EV/fire -4.15bp at 1.0 ATR, -2.31bp at 2.0, and +2.22bp at 3.0 -- the only non-negative
# cell in the family. And 3bl showed a FIXED bracket costs even the ORACLE -0.94 to -1.43bp
# against simply holding, because barriers truncate the moves that pay for the losers. A wide
# trail is the one exit that does not truncate, which is the "win much bigger than you lose"
# structure. No take-profit, by design.
#
# HONEST CEILING: +2.22bp/fire sits inside the pre-registered 4.3% exit-change ceiling (3bo A2),
# is measured on overlapping events, and payoff asymmetry cannot manufacture expectancy --
# EV = p*W - (1-p)*L and widening W lowers p, since breakeven is p* = L/(W+L).
# DEMO ONLY: the harness refuses to send on any account not flagged demo.
set -uo pipefail
cd /home/trader/MT5 || exit 1
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/zigzag-trail-ftmo.log
mkdir -p data/v5_runs
exec 9>/tmp/zigzag-trail-ftmo.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) previous run still going, skipping" >> "$LOG"; exit 0; }
echo "===== $(date -u +%FT%TZ) =====" >> "$LOG"
$PY scripts/v5_zigzag_ftmo.py \
  --config configs/v5_zigzag_trail_ftmo.json \
  --live --execute \
  >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
