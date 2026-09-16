#!/bin/bash
# GOLD-TILTED core-4 CHAMPION BOOK — TRADED on the FTMO demo (bridge 18814, magic 360591).
# Enabled 2026-09-16 at the user's request, replacing the ZigZag harness on this account.
#
# REFRESH FIRST, TRADE SECOND, AND NEVER TRADE STALE.
# The engine reads LOCAL CSVs, not the broker feed:
#   data/{GOLD,BTC,NDX,BRENT}_D1_long.csv  Yahoo, via refresh_d1_panel.py
#   data/XAUUSD_H4_long.csv                MT5,  via refresh_xau_h4.py
# When this was first wired the D1 files were 2-3 MONTHS stale (2026-06-16) and the H4 file
# 16 days stale, because nothing on this host ever refreshed them. Sizing the book off June
# prices would have been silent and wrong, so both refreshers run first and the staleness
# gate below aborts the whole pass rather than trade on old data.
set -u
cd /home/trader/MT5 || exit 1
mkdir -p data/v5_runs
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/book-ftmo.log

echo "===== $(date -u +%FT%TZ) =====" >> "$LOG"

if ! $PY scripts/refresh_xau_h4.py --port 18814 >> "$LOG" 2>&1; then
  echo "ABORT: XAUUSD H4 refresh failed — not trading on stale data" >> "$LOG"; exit 1
fi
if ! $PY scripts/refresh_d1_panel.py --aliases GOLD,BTC,NDX,BRENT >> "$LOG" 2>&1; then
  echo "ABORT: D1 panel refresh failed — not trading on stale data" >> "$LOG"; exit 1
fi

# Independent gate: trust the FILES, not the exit codes above. A refresher can succeed and
# still return nothing useful (Yahoo outage, weekend, symbol rename).
$PY - >> "$LOG" 2>&1 <<'PYGATE' || { echo "ABORT: staleness gate failed" >> "$LOG"; exit 1; }
import sys, pandas as pd
from pathlib import Path
LIMITS = {"XAUUSD_H4_long": 12, "GOLD_D1_long": 96, "BTC_D1_long": 96,
          "NDX_D1_long": 120, "BRENT_D1_long": 120}   # hours; weekends need the slack
now = pd.Timestamp.utcnow().tz_localize(None)
bad = []
for name, lim in LIMITS.items():
    f = Path("data") / f"{name}.csv"
    if not f.exists():
        bad.append(f"{name} MISSING"); continue
    last = pd.read_csv(f, usecols=["time"], parse_dates=["time"])["time"].max()
    age = (now - last).total_seconds() / 3600
    print(f"  gate {name}: last {last} ({age:.1f}h, limit {lim}h)")
    if age > lim:
        bad.append(f"{name} {age:.0f}h > {lim}h")
if bad:
    print("STALE: " + "; ".join(bad)); sys.exit(1)
print("  gate PASS")
PYGATE

$PY scripts/v5_basket_challenge_exec.py \
  --config configs/v5_ftmo_gold_tilted.json \
  --state data/v5_runs/ftmo_gold_tilted_state.json \
  --paper-csv data/v5_runs/ftmo_gold_tilted_log.csv \
  --port 18814 \
  --live --execute \
  >> "$LOG" 2>&1
