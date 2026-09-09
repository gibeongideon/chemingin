#!/bin/bash
# Daily five-finding signal digest. EMAIL ONLY — no order-sending code path.
# Only scripts/v5_zigzag_ftmo.py trades, and only on the demo.
set -u
cd /home/trader/MT5 || exit 1
mkdir -p data/v5_runs
/home/trader/miniconda3/envs/envmt5/bin/python scripts/v5_signals_digest.py \
  >> data/v5_runs/signals-digest.log 2>&1
