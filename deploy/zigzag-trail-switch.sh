#!/bin/bash
# ZigZag bottom detector (threshold 0.6) + 3xATR RATCHETING trail — SWITCH 40080430 (DEMO),
# bridge 18815, symbol XAUUSDs, magic 360601.
#
# PARALLEL TO 360597 ON FUNDINGPIPS, same parameters and the same pooled model, which makes the
# pair a live venue A/B exactly as 360594/360595 are for the book. It is also the safer home:
# 360597 sits on an account being hand-traded, where three fill-probe orders were deleted from
# outside the terminal and equity fell to $93.6k from $100k.
#
# RUNS TO OBSERVE, NOT TO PAY. 3bv closed this family on MECHANISM -- the detector's oracle is
# 0.497 precision at RR=1 against a required 0.503, i.e. zero headroom before any model exists.
# 3bo found the 3.0-ATR trail is the only non-negative exit cell (+2.22bp/fire vs -4.15 at 1.0
# and -2.31 at 2.0), and 3bn fixed the ceiling: a wide trail approaches buy-and-hold FROM BELOW
# and can never beat it, because the limit of an infinitely wide trail IS holding. risk_frac is
# 0.0025, half the champion's, for that reason.
#
# THE `s` SUFFIX APPLIES TO THE AUXILIARIES TOO. The detector is POOLED (gold + the clean H1 FX
# panel) and the executor skips a missing auxiliary SILENTLY, so a verbatim copy of the
# FundingPips config would have trained this on gold alone and turned the venue A/B into a
# comparison of two different models.
set -uo pipefail
cd /home/trader/MT5 || exit 1
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/zigzag-trail-switch.log
mkdir -p data/v5_runs
exec 9>/tmp/zigzag-trail-switch.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) previous run still going, skipping" >> "$LOG"; exit 0; }
echo "===== $(date -u +%FT%TZ) =====" >> "$LOG"
$PY scripts/v5_zigzag_ftmo.py --config configs/v5_zigzag_trail_switch.json --live --execute >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
