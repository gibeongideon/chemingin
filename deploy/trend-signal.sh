#!/bin/bash
# CHAMPION TREND FOLLOWER target-position alert. Sends over HTTPS via v5_notify (the VPS
# cannot reach any SMTP port). This is the TRADABLE strategy, unlike the zigzag harness.
# Nothing here places orders; it emails targets for a human.
set -u
cd /home/trader/MT5 || exit 1
mkdir -p data/v5_runs
/home/trader/miniconda3/envs/envmt5/bin/python scripts/v5_trend_signal.py \
  --port 18814 \
  >> data/v5_runs/trend-signal.log 2>&1
