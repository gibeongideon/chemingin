#!/bin/bash
# Goat Pay Later PDH end-of-day report -> Telegram + mail (16:50 New York = 23:50 server, after the 23:40 exit).
set -u
cd /home/trader/MT5 || exit 1
echo "===== $(date -u +%FT%TZ) =====" >> data/v5_runs/pdh-goat-report.log
/home/trader/miniconda3/envs/envmt5/bin/python scripts/pdh_prop_report.py --port 18820 --tag goat --magic 360618 \
    --symbol XAUUSD.x --label "Goat Pay Later" --target 0.03 --maxloss 0.08 --halt-dd 0.065 >> data/v5_runs/pdh-goat-report.log 2>&1
