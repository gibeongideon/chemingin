#!/bin/bash
# FAST intraday XAU trend (magic 360543, M30) on the FTMO demo, bridge 18814.
# SPREAD-GUARDED: v5_xau_fast.spread_guard ABORTS above max_spread_usd ($0.24),
# and FTMO gold is ~$0.45 -> this refuses to trade and exits, so it CANNOT
# affect the champion basket sharing this account. Auto-activates only if the
# account's gold spread ever drops <= $0.24 (raw/ECN). No --force-min-lot: the
# FTMO demo is $100k, real risk_frac applies.
set -u
cd /home/trader/MT5 || exit 1
export MT5_BRIDGE_PORT=18814
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/ftmo-fast-cron.log
echo "=== $(date -u '+%F %T UTC') ftmo fast wake (bridge 18814) ===" >> "$LOG"
$PY scripts/v5_xau_fast.py --bot fast --live --execute --save-data >> "$LOG" 2>&1
echo "$(date -u '+%F %T UTC') ftmo fast done" >> "$LOG"
