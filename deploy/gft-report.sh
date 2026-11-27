#!/bin/bash
# GOAT FUNDED end-of-day report (user 2026-11-27: "i need reporting every end of day") -> Telegram + mail
# via v5_notify (config report_notify=true). Fires 16:50 New York = 23:50 server, both DST regimes.
set -u
cd /home/trader/MT5 || exit 1
PY=/home/trader/miniconda3/envs/envmt5/bin/python
echo "===== $(date -u +%FT%TZ) =====" >> data/v5_runs/gft-report.log
$PY scripts/challenge_daily_report.py --config configs/v5_gft.json --state data/v5_runs/gft_state.json \
    --port 18819 >> data/v5_runs/gft-report.log 2>&1
