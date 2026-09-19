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
#
# MARKET-CLOSED IS NOT A FAILURE (fixed 2026-09-19). The gate below used to measure XAUUSD H4
# staleness against the WALL CLOCK with a flat 12h limit. Gold prints no bars between Friday
# ~21:00 UTC and Sunday ~22:00 UTC, so the last bar is legitimately 12-70h old all weekend and
# this unit failed on every weekend pass — three consecutive red runs on 2026-09-19 alone. Worse,
# a real dead feed looked identical to a normal Saturday, so the alarm carried no information.
# Now: if the broker is not quoting, the pass exits 0 having done nothing, and freshness is
# judged against THE BROKER'S OWN newest closed bar rather than against the clock.
set -u
cd /home/trader/MT5 || exit 1
mkdir -p data/v5_runs
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/book-ftmo.log

echo "===== $(date -u +%FT%TZ) =====" >> "$LOG"

# ---------------------------------------------------------------- is the market even open?
# Asked of the broker, not of a hard-coded session calendar: holidays, broker-specific hours and
# early closes are all invisible to a weekday/hour rule but obvious in the tick stream.
$PY - >> "$LOG" 2>&1 <<'PYOPEN'
import sys, time
import pandas as pd
from mt5linux import MetaTrader5
mt5 = MetaTrader5(host="localhost", port=18814)
if not mt5.initialize():
    print(f"  market check: bridge init failed {mt5.last_error()}"); sys.exit(2)
mt5.symbol_select("XAUUSD", True)
t = mt5.symbol_info_tick("XAUUSD")
mt5.shutdown()
if t is None:
    print("  market check: no tick object"); sys.exit(2)
age_min = (time.time() - float(t.time)) / 60.0
print(f"  market check: last XAUUSD tick {pd.to_datetime(t.time, unit='s')} "
      f"({age_min:.1f} min old)")
# 10 minutes: gold ticks several times a second when open, so any gap this long means closed.
sys.exit(0 if age_min <= 10 else 1)
PYOPEN
case $? in
  0) : ;;                                    # open — carry on
  1) echo "market CLOSED — no rebalance due, exiting clean" >> "$LOG"; exit 0 ;;
  *) echo "ABORT: could not determine market state" >> "$LOG"; exit 1 ;;
esac

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
from mt5linux import MetaTrader5

now = pd.Timestamp.utcnow().tz_localize(None)
bad = []

# --- XAUUSD H4: judged against the BROKER'S newest closed bar -------------------------------
# The question that matters is "does our file contain everything the broker has?", not "how old
# is the newest bar?". A clock-based limit cannot tell a Saturday from a dead feed. The absolute
# backstop below keeps the safety the clock rule was there to provide: we only reach this code
# when ticks ARE flowing, so a broker whose newest bar is still hours old is genuinely stalled.
mt5 = MetaTrader5(host="localhost", port=18814)
if not mt5.initialize():
    print(f"  gate XAUUSD_H4_long: bridge init failed {mt5.last_error()}"); sys.exit(1)
mt5.symbol_select("XAUUSD", True)
r = mt5.copy_rates_from_pos("XAUUSD", 16388, 0, 4)
mt5.shutdown()
if r is None or len(r) < 2:
    print("  gate XAUUSD_H4_long: broker returned no H4 bars"); sys.exit(1)
bars = pd.to_datetime([x[0] for x in r], unit="s").sort_values()
broker_last = bars[bars + pd.Timedelta(hours=4) <= now].max()   # newest CLOSED bar
newest_any = bars.max()                                         # includes the in-progress bar

f = Path("data/XAUUSD_H4_long.csv")
file_last = pd.read_csv(f, usecols=["time"], parse_dates=["time"])["time"].max()
open_age = (now - newest_any).total_seconds() / 3600
print(f"  gate XAUUSD_H4_long: file {file_last} | newest closed {broker_last} "
      f"| bar in progress {newest_any} ({open_age:.1f}h old)")
if file_last < broker_last:
    bad.append(f"XAUUSD_H4_long behind broker ({file_last} < {broker_last}) — refresh failed")
elif open_age > 8:
    # The stall test is on the bar CURRENTLY FORMING, not on the newest closed one. At the
    # Sunday ~22:00 reopen the newest CLOSED bar is still Friday 20:00 for a couple of hours,
    # so a "closed bar is old" rule would false-alarm at every weekly reopen while the feed is
    # demonstrably alive. A forming bar older than 8h (two H4 periods) with ticks flowing is a
    # real stall.
    bad.append(f"XAUUSD_H4_long broker feed STALLED: bar in progress {open_age:.0f}h old "
               f"while ticks are flowing")

# --- the D1 panel: Yahoo, still wall-clock, weekend slack already in the limits --------------
LIMITS = {"GOLD_D1_long": 96, "BTC_D1_long": 96, "NDX_D1_long": 120, "BRENT_D1_long": 120}
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
