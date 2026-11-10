#!/bin/bash
# ZigZag arm C — LIMIT entry (0.5 ATR below the ask, 24h expiry) + the +100/+200/+300 usd profit
# ladder — on FTMO-Demo 1515233856, bridge 18814, magic 360609.
#
# LIVE A/B against arm B. Same signal, same threshold, same size. TWO things differ by user
# request: the ENTRY (resting limit instead of paying the offer) and the EXIT (lock ladder
# instead of none). Because two things differ, read the paired report, not the raw P&L:
#   python scripts/v5_ab_report.py
#
# Priors, stated so the result is not rationalised later: §3dq measured no-lock at Sharpe 1.45
# against the ladder's 1.07 (t +2.16, 2.4x the dollars), and §3bc measured waiting for a dip at
# FILLED -22.8bp vs SKIPPED +17.9bp (t -5.23). Arm B is expected to win; C is here to test that
# on live fills.
#
# Disarm: systemctl --user disable --now zigzag-trail-ftmo-c.timer
set -uo pipefail
cd /home/trader/MT5 || exit 1
export NOTIFY_NO_TELEGRAM=1
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/zigzag-trail-ftmo-c.log
mkdir -p data/v5_runs
exec 9>/tmp/zigzag-trail-ftmo-c.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) previous run still going, skipping" >> "$LOG"; exit 0; }
echo "===== $(date -u +%FT%TZ) =====" >> "$LOG"
$PY scripts/v5_zigzag_ftmo.py --config configs/v5_zigzag_trail_ftmo_c.json --live --execute >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
