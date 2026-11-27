#!/bin/bash
# GOAT FUNDED 2-step promo 100K (315531833, GoatFunded-Server) — XAU+BTC book RECONCILE, magic 360617.
# Bridge 18819 (prefix ~/.mt5h). Config configs/v5_gft.json (guards 2% daily / 7.5% halt, target 10% then 5%,
# cushion sizing on the static 8% floor). The 2-minute protector is deploy/gft-guard.sh; both share one lock.
# SIGNAL = data/XAUUSDs_H4_switch.csv, refreshed by the Switch book job at 0/4:50; this job only READS it.
set -uo pipefail
cd /home/trader/MT5 || exit 1
export NOTIFY_NO_TELEGRAM=1   # trades reach the phone via trade-events-gft; this job mails problems only
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/gft.log
mkdir -p data/v5_runs
exec 9>/tmp/gft.lock
flock -w 120 9 || { echo "$(date -u +%FT%TZ) lock busy 120s, skipping" >> "$LOG"; exit 0; }
echo "===== $(date -u +%FT%TZ) =====" >> "$LOG"
$PY - >> "$LOG" 2>&1 <<'PYOPEN'
import sys, time
from mt5linux import MetaTrader5
mt5 = MetaTrader5(host="localhost", port=18819)
if not mt5.initialize():
    print(f"  market check: bridge init failed {mt5.last_error()}"); sys.exit(2)
ai = mt5.account_info()
if ai is None or int(ai.login) != 315531833:
    print(f"  market check: WRONG/NO ACCOUNT on 18819: {getattr(ai, 'login', None)}"); sys.exit(2)
open_any = False
for s in ("XAUUSD.x", "BTCUSD.x"):
    mt5.symbol_select(s, True)
    a = mt5.symbol_info_tick(s); time.sleep(8); b = mt5.symbol_info_tick(s)
    if a and b and (int(b.time) > int(a.time) or float(b.bid) != float(a.bid)):
        print(f"  market check: {s} ticking -> OPEN"); open_any = True
    else:
        print(f"  market check: {s} static -> closed")
sys.exit(0 if open_any else 1)   # NO mt5.shutdown(): shared bridge
PYOPEN
case $? in
  0) : ;;
  1) echo "both symbols closed — nothing to do" >> "$LOG"; exit 0 ;;
  *) echo "ABORT: could not determine market state / account" >> "$LOG"; exit 1 ;;
esac
$PY scripts/refresh_d1_panel.py --aliases GOLD,BTC >> "$LOG" 2>&1
export XAU_H4_CSV="$PWD/data/XAUUSDs_H4_switch.csv"
$PY - >> "$LOG" 2>&1 <<'PYGATE' || { echo "ABORT: staleness gate failed" >> "$LOG"; exit 1; }
import sys, pandas as pd
from pathlib import Path
from mt5linux import MetaTrader5
bad = []
mt5 = MetaTrader5(host="localhost", port=18819)
if not mt5.initialize():
    print("  gate: bridge init failed"); sys.exit(1)
mt5.symbol_select("XAUUSD.x", True)
tick = mt5.symbol_info_tick("XAUUSD.x")
r = mt5.copy_rates_from_pos("XAUUSD.x", 16388, 0, 4)          # 16388 = TIMEFRAME_H4
if r is None or len(r) < 2 or tick is None:
    print("  gate: no H4 bars / no tick from GoatFunded"); sys.exit(1)
srv_now = pd.to_datetime(int(tick.time), unit="s")
bars = pd.to_datetime([x[0] for x in r], unit="s").sort_values()
broker_last = bars[bars + pd.Timedelta(hours=4) <= srv_now].max()
f = Path("data/XAUUSDs_H4_switch.csv")
file_last = pd.read_csv(f, usecols=["time"], parse_dates=["time"])["time"].max()
print(f"  gate H4: signal file {file_last} | GoatFunded newest closed {broker_last} (server {srv_now})")
if file_last < broker_last - pd.Timedelta(hours=4):
    bad.append(f"signal file {file_last} more than one bar behind the broker {broker_last} (Switch refresh stalled?)")
now = pd.Timestamp.utcnow().tz_localize(None)
for name, lim in {"GOLD_D1_long": 96, "BTC_D1_long": 96}.items():
    last = pd.read_csv(Path("data") / f"{name}.csv", usecols=["time"], parse_dates=["time"])["time"].max()
    age = (now - last).total_seconds() / 3600
    print(f"  gate {name}: last {last} ({age:.1f}h, limit {lim}h)")
    if age > lim:
        bad.append(f"{name} {age:.0f}h > {lim}h")
if bad:
    print("STALE: " + "; ".join(bad)); sys.exit(1)
print("  gate PASS")
PYGATE
$PY scripts/v5_basket_challenge_exec.py --config configs/v5_gft.json --state data/v5_runs/gft_state.json \
    --paper-csv data/v5_runs/gft_live_log.csv --port 18819 --live --execute >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
if [ "$(wc -l < "$LOG")" -gt 40000 ]; then tail -15000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"; fi
