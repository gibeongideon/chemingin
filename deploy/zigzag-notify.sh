#!/bin/bash
# Notifier for the ZigZag harness — sends only when a trade is due, or when the pipeline
# is broken/stale. Runs a minute after the executor.
set -u
cd /home/trader/MT5 || exit 1
/home/trader/miniconda3/envs/envmt5/bin/python scripts/v5_zigzag_notify.py \
  >> data/v5_runs/zigzag-notify.log 2>&1
