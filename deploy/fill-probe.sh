#!/bin/bash
# OCTOBER A6 — the fill probe. FundingPips-Trial 40000397948 (DEMO), bridge 18816, magic 360599.
#
# This is a MEASUREMENT, not a strategy. §3ci found that 47% of the prior-high break's +10.161bp
# is a FILL ADVANTAGE (+4.729bp, t +7.45) -- the gap between the prior-day high and where the
# breaking bar closed -- and that the two execution paths disagree: a resting stop at the level
# clears (+8.111bp over drift, t +3.76), a market entry on bar close does not (+3.382bp, t +1.68).
# No further backtesting can settle which one a live order gets. One minimum-lot stop a day can.
#
# It shares the terminal with the live book (magic 360598) and the zigzag (360597) and is scoped
# to its own magic at every read and write. Runs hourly: place early in the session, notice the
# fill, record the slippage, exit or cancel at the session close.
set -uo pipefail
cd /home/trader/MT5 || exit 1
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/fill-probe.log
mkdir -p data/v5_runs
exec 9>/tmp/fill-probe.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) previous run still going, skipping" >> "$LOG"; exit 0; }
echo "===== $(date -u +%FT%TZ) =====" >> "$LOG"
$PY scripts/v5_fill_probe.py --port 18816 --symbol XAUUSD --execute >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
