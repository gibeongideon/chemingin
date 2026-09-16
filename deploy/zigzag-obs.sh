#!/bin/bash
# ZigZag bottom detector — SIGNAL ONLY. Trading DISABLED 2026-09-16 at the user's request
# ("stop zigzag, also just send zigzag signals"), replaced on this account by the gold-tilted
# core-4 champion book.
#
# WHY IT STOPPED: negative expectancy, now measured live as well as in backtest.
# V5_FINDINGS 3x: walk-forward SR -0.76, 0/8 years, DSR 0.000.
# 3as: EV/fire +0.05% against +0.15% for simply holding, mean fire offset +0.771 bars
#      -- it recognises bottoms about a day AFTER they happen.
# LIVE: 5 orders placed 2026-09-10..16 took the demo from $100,000.00 to $99,564.82.
#
# WITHOUT --live --execute the script still fetches the broker feed, scores the detector and
# writes data/v5_runs/zigzag_ftmo_state.json, so both notifiers keep sending signals exactly
# as before. It simply cannot place an order: order_send is only reached under both flags.
set -u
cd /home/trader/MT5 || exit 1
mkdir -p data/v5_runs
/home/trader/miniconda3/envs/envmt5/bin/python scripts/v5_zigzag_ftmo.py \
  >> data/v5_runs/zigzag-obs.log 2>&1
