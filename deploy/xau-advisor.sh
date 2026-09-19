#!/bin/bash
# XAUUSD ADVISOR — advisory only. It has no order path and no magic number, because it never
# sends an order. See V5_FINDINGS 3az (direction: no edge) and 3ba (the adverse cell, corrected).
#
# COMPUTE AND SEND IN ONE PASS. Two timers would double the systemd activations and open a torn
# read: the notifier could run between the compute step's state write and its cache write. One
# wrapper, two scripts, one lock.
#
# WHAT IT COSTS TO RUN EVERY 5 MINUTES. The model is called once per closed H4 bar -- 6 times a
# day -- and cached by (last_closed_h4, model_id); the other ~282 runs are cache hits that only
# refresh the market panel and check liveness. So the 5-minute cadence buys latency of detection,
# not a fresher probability, and the message says so in as many words.
set -u
cd /home/trader/MT5 || exit 1
mkdir -p data/v5_runs
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/xau-advisor.log

# flock, not a PID file: a run that overlaps its predecessor would step the state machine twice
# for one bar, and the dwell timer and thrash guard are counted in transitions.
exec 9>/tmp/xau-advisor.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) previous run still going — skipping" >> "$LOG"; exit 0; }

echo "===== $(date -u +%FT%TZ) =====" >> "$LOG"

# The compute step is allowed to fail: it writes `broken` into its own state file and the
# notifier turns that into a NO READING alert. Swallowing the exit code here would replace a
# visible alert with silence, which is the one outcome the heartbeat exists to prevent.
$PY scripts/v5_xau_advisor.py --port 18814 >> "$LOG" 2>&1
$PY scripts/v5_xau_advisor_notify.py >> "$LOG" 2>&1

# keep the log from growing without bound (288 runs/day)
if [ "$(wc -l < "$LOG")" -gt 20000 ]; then tail -8000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"; fi
