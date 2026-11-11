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

# NOTIFICATIONS: SWITCH ONLY (user, 2026-11-10: "only sent notification if the prediction change
# from Buy to sell or vice versa not hourly"). The hourly status was also the liveness proof, so
# one heartbeat a day replaces it -- 1 message instead of 23, and total silence still means the
# pipeline is down rather than the market being quiet. Outage and health alerts are untouched.
#
# 2026-11-11 (user: "let the Adviser also send hourly"): `hourly` = the switch at once PLUS a
# status every 60 min while gold is quoting. Still no 3-state announcement. The hourly status is
# the liveness proof again, so the daily heartbeat is unused in this mode (kept for `switch`).
export ADVISOR_ALERTS=hourly
export ADVISOR_HEARTBEAT_HOUR=12
# The card must not claim nobody is trading this. Stated HERE, by the deployment, because the
# message module cannot know -- the old hard-coded "no bot trades these signals" went stale the
# moment an executor was armed. Clear this line if the FTMO arms are ever disarmed.
export ADVISOR_EXECUTORS="FTMO demo 1515233856 - magic 360611 (long/short) and 360612 (long-only), 0.25% risk, bias rule"

echo "===== $(date -u +%FT%TZ) =====" >> "$LOG"

# The compute step is allowed to fail: it writes `broken` into its own state file and the
# notifier turns that into a NO READING alert. Swallowing the exit code here would replace a
# visible alert with silence, which is the one outcome the heartbeat exists to prevent.
#
# MOVED OFF BRIDGE 18814 ON 2026-11-01. The FTMO demo behind it stopped answering at
# 2026-10-28 16:43 UTC and every run since raised ConnectionRefusedError, so the advisor had
# been reporting NO READING for 3.9 days. The heartbeat did its job -- the alert was correct and
# arrived ten times -- but an alert nobody acts on is an outage, so it now reads the Switch
# bridge, which is the same feed the intraday advisor and the zigzag harness already use.
# `s` SUFFIX: every Switch symbol carries it, XAUUSD does not resolve there.
$PY scripts/v5_xau_advisor.py --port 18815 --symbol XAUUSDs >> "$LOG" 2>&1
$PY scripts/v5_xau_advisor_notify.py >> "$LOG" 2>&1

# The ORDER EXECUTOR runs on its own every-minute timer (xau-advisor-trader.timer) and reads
# the state file this writes. Kept out of this wrapper so trailing is not tied to the 5-min cadence.

# keep the log from growing without bound (288 runs/day)
if [ "$(wc -l < "$LOG")" -gt 20000 ]; then tail -8000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"; fi
