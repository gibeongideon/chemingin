#!/bin/bash
# XAU + BTC, 50/50 equal class risk — SWITCH MARKETS 40080430 (demo), bridge 18815, magic 360595.
# V5_FINDINGS 3bs measured Switch as the cheapest venue for this book: Sharpe +1.286 / CAGR
# +11.6% / maxDD -12.5%, against FTMO's +1.172 / +10.5% / -13.4%, because the BTC sleeve turns
# over only 5.8x/yr so Switch's half-price carry beats its 37x-worse spread by about 4x.
# Runs ALONGSIDE the FTMO deployment of the same book (magic 360594) on the identical forecast,
# which makes the pair a live venue A/B rather than a replacement.
set -uo pipefail
cd /home/trader/MT5 || exit 1
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/xau-btc-switch.log
mkdir -p data/v5_runs
exec 9>/tmp/xau-btc-switch.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) previous run still going, skipping" >> "$LOG"; exit 0; }
echo "===== $(date -u +%FT%TZ) =====" >> "$LOG"

# ---------------------------------------------------------------- is the market even open?
# Asked of the broker's own tick stream and FRAME-INDEPENDENT (does the tick ADVANCE?).
# Comparing MT5's server-stamped tick time against host UTC reads a fresh tick as -180 min old
# and makes the gate a no-op -- that bug shipped once already and was fixed in 5678bd7.
$PY - >> "$LOG" 2>&1 <<'PYOPEN'
import sys, time
from mt5linux import MetaTrader5
mt5 = MetaTrader5(host="localhost", port=18815)
if not mt5.initialize():
    print(f"  market check: bridge init failed {mt5.last_error()}"); sys.exit(2)
open_any = False
for s in ("XAUUSDs", "BTCUSDs"):        # BTC trades weekends; gold does not
    mt5.symbol_select(s, True)
    a = mt5.symbol_info_tick(s); time.sleep(8); b = mt5.symbol_info_tick(s)
    if a and b and (int(b.time) > int(a.time) or float(b.bid) != float(a.bid)):
        print(f"  market check: {s} ticking -> OPEN"); open_any = True
    else:
        print(f"  market check: {s} static -> closed")
mt5.shutdown()
sys.exit(0 if open_any else 1)
PYOPEN
case $? in
  0) : ;;
  1) echo "both symbols closed — nothing to do" >> "$LOG"; exit 0 ;;
  *) echo "ABORT: could not determine market state" >> "$LOG"; exit 1 ;;
esac

# ---------------------------------------------------------------- data, and the splice rule
# *** THE H4 REFRESH READS FTMO (18814), NOT SWITCH, ON PURPOSE. ***
# data/XAUUSD_H4_long.csv already carries one MT5/FTMO splice from 2023-06-28. Appending a THIRD
# broker's quotes is the exact failure recorded in 'multi-broker dataset = leak' -- it faked two
# separate results and is invisible to every timestamp control we run. So the SIGNAL comes from
# the FTMO feed and only EXECUTION happens at Switch. A side benefit: both live books then trade
# the identical forecast, which is what isolates the venue.
$PY scripts/refresh_d1_panel.py --aliases GOLD,BTC >> "$LOG" 2>&1
if ! $PY scripts/refresh_xau_h4.py --port 18814 >> "$LOG" 2>&1; then
  echo "  WARN: H4 refresh via FTMO 18814 failed — falling back to a freshness CHECK" >> "$LOG"
fi

# Independent gate: trust the FILES, not the exit codes above. A refresher can succeed and still
# return nothing useful, and this book must not be hostage to the FTMO terminal being up -- so
# freshness is judged against SWITCH's own newest closed bar. Note the asymmetry that keeps the
# no-splice rule intact: we READ Switch's bars to validate, and never WRITE them to the file.
$PY - >> "$LOG" 2>&1 <<'PYGATE' || { echo "ABORT: staleness gate failed" >> "$LOG"; exit 1; }
import sys, pandas as pd
from pathlib import Path
from mt5linux import MetaTrader5

now = pd.Timestamp.utcnow().tz_localize(None)
bad = []

mt5 = MetaTrader5(host="localhost", port=18815)
if not mt5.initialize():
    print(f"  gate XAUUSD_H4_long: bridge init failed {mt5.last_error()}"); sys.exit(1)
mt5.symbol_select("XAUUSDs", True)
tick = mt5.symbol_info_tick("XAUUSDs")
r = mt5.copy_rates_from_pos("XAUUSDs", 16388, 0, 4)   # 16388 = TIMEFRAME_H4
mt5.shutdown()
if r is None or len(r) < 2:
    print("  gate XAUUSD_H4_long: broker returned no H4 bars"); sys.exit(1)
if tick is None:
    print("  gate XAUUSD_H4_long: no tick, cannot establish the server clock"); sys.exit(1)

# *** EVERY BAR-AGE COMPARISON BELOW IS IN SERVER TIME, NOT HOST UTC. ***
# MT5 stamps bars and ticks on the BROKER's clock (SwitchMarkets-Live and FTMO-Demo are both
# UTC+3, measured 2026-10-19). Subtracting a server-stamped bar from pd.Timestamp.utcnow() makes
# the forming bar read as -3.0h old, so a `> 8` stall test can NEVER fire -- exactly the dead
# gate this book already shipped once on the market check (fixed in 5678bd7), re-created here by
# copying the FTMO wrapper. The fix is to take "now" from the broker's own tick, which is in the
# same frame as the bar stamps. The file-vs-broker test below was never affected: both sides of
# it are server-stamped, so it compares like with like.
srv_now = pd.to_datetime(int(tick.time), unit="s")
bars = pd.to_datetime([x[0] for x in r], unit="s").sort_values()
broker_last = bars[bars + pd.Timedelta(hours=4) <= srv_now].max()   # newest CLOSED bar
newest_any = bars.max()                                            # incl. the in-progress bar

f = Path("data/XAUUSD_H4_long.csv")
file_last = pd.read_csv(f, usecols=["time"], parse_dates=["time"])["time"].max()
open_age = (srv_now - newest_any).total_seconds() / 3600
print(f"  gate XAUUSD_H4_long: file {file_last} | Switch newest closed {broker_last} "
      f"| bar in progress {newest_any} ({open_age:.1f}h old, server clock {srv_now})")
if file_last < broker_last:
    bad.append(f"XAUUSD_H4_long behind the broker ({file_last} < {broker_last}) — "
               f"the FTMO refresh is not keeping up")
elif open_age > 8:
    # The stall test is on the bar CURRENTLY FORMING, not the newest closed one. At the Sunday
    # reopen the newest closed bar is still Friday 20:00 for a couple of hours, so a "closed bar
    # is old" rule false-alarms every week while the feed is demonstrably alive.
    bad.append(f"broker feed STALLED: bar in progress {open_age:.0f}h old while ticks flow")

LIMITS = {"GOLD_D1_long": 96, "BTC_D1_long": 96}
for name, lim in LIMITS.items():
    p = Path("data") / f"{name}.csv"
    if not p.exists():
        bad.append(f"{name} MISSING"); continue
    last = pd.read_csv(p, usecols=["time"], parse_dates=["time"])["time"].max()
    age = (now - last).total_seconds() / 3600
    print(f"  gate {name}: last {last} ({age:.1f}h, limit {lim}h)")
    if age > lim:
        bad.append(f"{name} {age:.0f}h > {lim}h")

if bad:
    print("STALE: " + "; ".join(bad)); sys.exit(1)
print("  gate PASS")
PYGATE

$PY scripts/v5_basket_challenge_exec.py \
  --config configs/v5_xau_btc_switch.json \
  --state data/v5_runs/xau_btc_switch_state.json \
  --paper-csv data/v5_runs/xau_btc_switch_log.csv \
  --port 18815 --live --execute >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
