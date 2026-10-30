#!/bin/bash
# The intraday trail-follower, INTRADAY-TRAIL-SPEC.md — SWITCH 40080430 (DEMO), bridge 18815,
# symbol XAUUSDs, magic 360600. Hourly.
#
# A MEASUREMENT DEPLOYMENT, not a validated edge. §3ck: it LOSES to holding gold on CAGR
# (+11.06% vs +11.96%) and wins only on risk; §3cj: its advantage over a PLACEBO level at the
# same ATR distance sits at p 0.05-0.09; its slippage budget is 3.5bp and is still being measured
# by the fill probe (magic 360599); Bar 1 (DSR/PBO) has never been computed. 0.5% risk on a demo.
#
# Long-only, no take-profit, no move-to-breakeven (§3co: -6.36bp/trade at t -3.55), ratcheting
# 3.0-ATR trail (§3bn), brackets in DAILY ATR never percent. The protective stop is attached to
# the ORDER, so it is server-side from the fill and this wrapper never needs to be alive to
# protect a position -- it only ratchets.
set -uo pipefail
cd /home/trader/MT5 || exit 1
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/trail-runner.log
mkdir -p data/v5_runs
exec 9>/tmp/trail-runner.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) previous run still going, skipping" >> "$LOG"; exit 0; }
echo "===== $(date -u +%FT%TZ) =====" >> "$LOG"
$PY scripts/v5_trail_runner.py --port 18815 --symbol XAUUSDs --execute >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
