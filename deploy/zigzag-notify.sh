#!/bin/bash
# ZigZag notifier — runs ON THE VPS, sends over HTTPS (SMTP is blocked here).
#
# POINTED AT SWITCH (bridge 18815, magic 360601) SINCE 2026-11-01. It previously read
# `zigzag_ftmo_state.json`, written by zigzag-obs against bridge 18814 — an FTMO demo that no
# longer exists. The result was an hourly "state 91h stale" warning and exit 1 for days: the
# notifier was working perfectly and reporting on a dead deployment. The live harness is
# zigzag-trail-switch.timer, which writes zigzag_trail_switch_state.json hourly at :19.
#
# IT NOW SENDS OUTCOMES, NOT ONLY SIGNALS. Each closed trade is announced WIN or LOSS with its
# net, and every message carries the running record, read from the broker's deal history rather
# than the executor's state file.
#
# NEGATIVE EXPECTANCY by design (V5_FINDINGS 3x: walk-forward SR -0.76, DSR 0.000; 3bv closed
# the family on mechanism). This reports what a disproven detector does live; it is not advice.
set -u
cd /home/trader/MT5 || exit 1
mkdir -p data/v5_runs
/home/trader/miniconda3/envs/envmt5/bin/python scripts/v5_zigzag_notify.py \
  --state zigzag_trail_switch_state.json --port 18815 --magic 360601 \
  >> data/v5_runs/zigzag-notify.log 2>&1
