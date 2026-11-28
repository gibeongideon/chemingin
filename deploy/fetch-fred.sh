#!/bin/bash
# Daily 2-year Treasury yield (DGS2) for ZigZag arm D's Y2UP size dial: FRED first, US Treasury CSV as fallback.
set -u
cd /home/trader/MT5 || exit 1
echo "===== $(date -u +%FT%TZ)" >> data/v5_runs/fetch-fred.log
/home/trader/miniconda3/envs/envmt5/bin/python scripts/fetch_dgs2.py >> data/v5_runs/fetch-fred.log 2>&1
