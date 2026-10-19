#!/bin/bash
# XAU + BTC, 50/50 equal class risk — FTMO DEMO 1514968075, bridge 18814, magic 360594.
# V5_FINDINGS 3bq's measured A+B cell: Sharpe +1.511, maxDD -11.2%, FTMO phase-1 pass 72.2%
# with a 0.0% daily-breach rate, against 53.6%/17.6% for the XAU champion alone.
# DSR 0.000 / PBO 0.381 both fail CONTROL #6 -- a real but UNPROVEN lead, which is why it is
# on a demo. Reuses v5_basket_challenge_exec.py unchanged (guards, compliance SL, re-anchoring).
set -uo pipefail
cd /home/trader/MT5 || exit 1
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/xau-btc-ftmo.log
mkdir -p data/v5_runs
exec 9>/tmp/xau-btc-ftmo.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) previous run still going, skipping" >> "$LOG"; exit 0; }
echo "===== $(date -u +%FT%TZ) =====" >> "$LOG"

# Market state from the broker's own tick stream, frame-independent (does the tick ADVANCE?).
# Comparing MT5's server-stamped tick time against host UTC reads a fresh tick as -180 min old
# and makes the gate a no-op -- fixed in 5678bd7.
$PY - >> "$LOG" 2>&1 <<'PYOPEN'
import sys, time
from mt5linux import MetaTrader5
mt5 = MetaTrader5(host="localhost", port=18814)
if not mt5.initialize():
    print(f"  market check: bridge init failed {mt5.last_error()}"); sys.exit(2)
open_any = False
for s in ("XAUUSD", "BTCUSD"):          # BTC trades weekends; gold does not
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

$PY scripts/refresh_d1_panel.py --aliases GOLD,BTC >> "$LOG" 2>&1
$PY scripts/refresh_xau_h4.py --port 18814 >> "$LOG" 2>&1
$PY scripts/v5_basket_challenge_exec.py \
  --config configs/v5_xau_btc_ftmo.json \
  --state data/v5_runs/xau_btc_ftmo_state.json \
  --paper-csv data/v5_runs/xau_btc_ftmo_log.csv \
  --port 18814 --live --execute >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
