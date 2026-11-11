#!/bin/bash
# XAUUSD ADVISOR EXECUTOR — FTMO-Demo 1515233856, bridge 18814, XAUUSD. TWO ARMS, every 5 min.
#
#   360611  arm LS  --side-mode long_short   SELL call -> open a short   <- as the user specified
#   360612  arm L   --side-mode long_only    SELL call -> open nothing
#
# Both trade the BUY/SELL bias the Telegram message announces, both close an opposite leg that is
# in profit, both keep a losing one and close it at +$50, both trail at 1.0 x H4 ATR. They differ
# in ONE thing: whether a SELL call opens a short. V5_FINDINGS 3bz measured that leg at -0.29 to
# -0.52 Sharpe across four thresholds; this is the live test of that claim.
#
# *** NOT A VALIDATED EDGE. *** V5_FINDINGS 3de: this bias made -25.0R over 28 years and -15.1R
# out of sample (shuffle p 0.955). Demo only, 0.25% risk per entry, matching the ZigZag arms that
# share this account. Report: python scripts/v5_advisor_exec_report.py
#
# ONE WRAPPER, ONE FLOCK, BOTH ARMS. Two timers would double the systemd activations on a shared
# bridge and let the arms read DIFFERENT advisor states and DIFFERENT ticks -- at which point the
# A/B is no longer measuring the short leg, it is measuring two instants.
#
# Disarm: systemctl --user disable --now xau-advisor-trader-ftmo.timer
set -uo pipefail
cd /home/trader/MT5 || exit 1
export NOTIFY_NO_TELEGRAM=1   # the phone gets these via trade-events-ftmo; this job -> email
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/xau-advisor-trader-ftmo.log
mkdir -p data/v5_runs
exec 9>/tmp/xau-advisor-trader-ftmo.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) previous run still going, skipping" >> "$LOG"; exit 0; }
echo "===== $(date -u +%FT%TZ) =====" >> "$LOG"

COMMON="--port 18814 --symbol XAUUSD --ref-symbol BTCUSD --rule BIAS --risk-frac 0.0025
        --max-spread 0.80 --rescue-usd 50 --reset-tz FTMO
        --own-lock /tmp/xau-advisor-trader-ftmo.lock --execute"

$PY scripts/v5_xau_advisor_trader.py $COMMON \
    --magic 360611 --tag ftmo_ls --side-mode long_short --label "FTMO demo, long/short" \
    >> "$LOG" 2>&1 || echo "$(date -u +%FT%TZ) arm LS exit=$?" >> "$LOG"

$PY scripts/v5_xau_advisor_trader.py $COMMON \
    --magic 360612 --tag ftmo_lo --side-mode long_only --label "FTMO demo, long-only" \
    >> "$LOG" 2>&1 || echo "$(date -u +%FT%TZ) arm L  exit=$?" >> "$LOG"

if [ "$(wc -l < "$LOG")" -gt 30000 ]; then tail -10000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"; fi
