#!/bin/bash
# ZigZag harness notifier — runs ON THE VPS, sends over HTTPS (SMTP is blocked here).
# Harmless until a provider key exists in .env.notify: it just prints what it would send.
# NEGATIVE EXPECTANCY by design (V5_FINDINGS 3x: walk-forward SR -0.76, DSR 0.000).
set -u
cd /home/trader/MT5 || exit 1
mkdir -p data/v5_runs
/home/trader/miniconda3/envs/envmt5/bin/python scripts/v5_zigzag_notify.py \
  >> data/v5_runs/zigzag-notify.log 2>&1
