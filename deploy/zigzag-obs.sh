#!/bin/bash
# ZigZag bottom-detector OBSERVATION HARNESS — FTMO DEMO, bridge 18814.
# NEGATIVE EXPECTANCY by design (V5_FINDINGS 3x: walk-forward SR -0.76, 0/8 years, DSR 0.000).
# Hourly because the 48h max-hold exit is the only exit the bot owns; TP/SL are server-side.
set -u
cd /home/trader/MT5 || exit 1
mkdir -p data/v5_runs
/home/trader/miniconda3/envs/envmt5/bin/python scripts/v5_zigzag_ftmo.py \
  --live --execute \
  >> data/v5_runs/zigzag-obs.log 2>&1
