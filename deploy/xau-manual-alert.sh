#!/bin/bash
# VPS side of the manual-execution alert: compute the ticket and write it as JSON for the
# relay mailer to send. Does NOT email from here — this host cannot reach any SMTP port
# (verified 2026-09-07). Read-only; places no orders.
set -u
cd /home/trader/MT5 || exit 1
mkdir -p data/v5_runs
/home/trader/miniconda3/envs/envmt5/bin/python scripts/v5_manual_alert.py \
  --port 18812 \
  --dial 0.10 \
  --dry \
  --json-out data/v5_runs/manual_alert.json \
  >> data/v5_runs/xau-manual-alert.log 2>&1
