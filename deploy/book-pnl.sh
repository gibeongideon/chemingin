#!/bin/bash
# PnL-MANAGED variant of the FTMO book. Overnight holding is NOT permitted on this account,
# so the hard flatten at 20:45 UTC outranks every profit rule -- a breach costs more than any
# target is worth.
#
# MEASURED COST, before it was built (V5_FINDINGS 3bk):
#   champion holding overnight   +0.77 Sharpe / +7.90% CAGR / -22.1% DD
#   flat every night (this rule) +0.50 / +4.75% / -35.5%   <- worse on BOTH axes
#   + the $200 target            -0.26 Sharpe further
# The overnight damage is TURNOVER (132x the crossings), not missed gaps -- gold trades ~23h
# so the overnight gap is only 2% of the move captured.
#
# This REPLACES book-ftmo (same magic 360591, so it adopts those positions). Both must never
# run: book-ftmo.timer is disabled while this is enabled.
set -u
cd /home/trader/MT5 || exit 1
mkdir -p data/v5_runs
exec 9>/tmp/book-pnl.lock
flock -n 9 || exit 0          # a slow run must not be re-entered a minute later
/home/trader/miniconda3/envs/envmt5/bin/python scripts/v5_ftmo_pnl_managed.py \
  --port 18814 --target 200 --harvest 50 --harvest-window 5 --reenter signal \
  --execute --live >> data/v5_runs/book-pnl.log 2>&1
if [ "$(wc -l < data/v5_runs/book-pnl.log)" -gt 20000 ]; then
  tail -6000 data/v5_runs/book-pnl.log > data/v5_runs/book-pnl.log.tmp \
    && mv data/v5_runs/book-pnl.log.tmp data/v5_runs/book-pnl.log
fi
