#!/bin/bash
# Intraday advisor -> email. Switch demo (bridge 18815, XAUUSDs). REPORT ONLY: this path holds
# no order functions and cannot place, modify or close anything.
#
# Runs hourly and ALWAYS logs. SENDS only when the tape state changes or 4h have passed, because
# hourly mail is 23 messages a day and an advisor whose usual correct answer is FLAT (§3cp) has
# nothing new to say most hours. Change the cadence with --always or HEARTBEAT_HOURS.
set -uo pipefail
cd /home/trader/MT5 || exit 1
export NOTIFY_NO_TELEGRAM=1   # 2026-11-06: phone gets only trade events + advisor; this job -> email
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/advisor-notify.log
mkdir -p data/v5_runs
exec 9>/tmp/advisor-notify.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) previous run still going, skipping" >> "$LOG"; exit 0; }
echo "===== $(date -u +%FT%TZ) =====" >> "$LOG"
$PY scripts/v5_advisor_notify.py --port 18815 --symbol XAUUSDs --risk-pct 0.5 >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
