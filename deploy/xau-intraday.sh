#!/bin/bash
# Short-form intraday action card for XAUUSD: the prior day's levels as a two-sided OCO.
#
# SIGNAL ONLY. No order path exists in the script this runs. The automated book
# (book-ftmo.service) is a SEPARATE and much stronger thing -- net Sharpe 1.18 against this
# card's measured 0.47 -- and nothing here feeds it.
#
# Scheduled at 23:30 UTC: after the broker's daily roll (the gold session break sits at
# ~21:00-22:00 UTC) so the prior day's bar is closed, and before 00:00 when the next day's
# first break can occur.
set -u
cd /home/trader/MT5 || exit 1
mkdir -p data/v5_runs
/home/trader/miniconda3/envs/envmt5/bin/python scripts/v5_intraday_levels.py \
  --port 18814 >> data/v5_runs/xau-intraday.log 2>&1
if [ "$(wc -l < data/v5_runs/xau-intraday.log)" -gt 5000 ]; then
  tail -2000 data/v5_runs/xau-intraday.log > data/v5_runs/xau-intraday.log.tmp \
    && mv data/v5_runs/xau-intraday.log.tmp data/v5_runs/xau-intraday.log
fi
