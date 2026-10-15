#!/bin/bash
# XAUUSD champion, FLAT 0.5% sizing, on the FTMO trial (login 1514962745, bridge 18814,
# magic 360592). This is V5_FINDINGS 3bo's "flat 0.5%" row deployed: 295 trades 2019-2026,
# +36.8% median over 7.2 years, 0.20% chance of a $10,000 drawdown, ~3.4 trades/month.
#
# NO LOT ESCALATION, BY MEASUREMENT. 3bo found the champion's win/loss sign autocorrelation is
# +0.046 (z +0.88 against a shuffled null) -- no serial structure -- so martingale, pyramid and
# recover-and-gain are all pure leverage. At a matched return the best of them ties flat and the
# rest pay 2-19pp of extra ruin; the uncapped recover-and-gain ruins >50% of paths past a 50x cap.
#
# *** book-ftmo.timer MUST STAY DISABLED while this runs. *** Both books hold XAUUSD, and the
# combined gold exposure is what neither was sized for.
set -uo pipefail
cd /home/trader/MT5 || exit 1
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/xau-champ-ftmo.log
mkdir -p data/v5_runs
exec 9>/tmp/xau-champ-ftmo.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) previous run still going, skipping" >> "$LOG"; exit 0; }
echo "===== $(date -u +%FT%TZ) =====" >> "$LOG"

# Market state asked of the BROKER, not a hard-coded calendar: holidays, broker-specific hours
# and early closes are invisible to a weekday/hour rule and obvious in the tick stream.
$PY - >> "$LOG" 2>&1 <<'PYOPEN'
import sys, time
from mt5linux import MetaTrader5
mt5 = MetaTrader5(host="localhost", port=18814)
if not mt5.initialize():
    print(f"  market check: bridge init failed {mt5.last_error()}"); sys.exit(2)
mt5.symbol_select("XAUUSD", True)
t = mt5.symbol_info_tick("XAUUSD")
mt5.shutdown()
if t is None or not getattr(t, "time", 0):
    print("  market check: no tick"); sys.exit(2)
age = (time.time() - float(t.time)) / 60.0
print(f"  market check: last XAUUSD tick {age:.1f} min old")
sys.exit(0 if age <= 10 else 1)
PYOPEN
case $? in
  0) : ;;
  1) echo "market CLOSED — nothing to do" >> "$LOG"; exit 0 ;;
  *) echo "ABORT: could not determine market state" >> "$LOG"; exit 1 ;;
esac

$PY scripts/v5_maven_champ.py \
  --config configs/v5_xau_ftmo_champ.json \
  --port 18814 \
  --live --execute --save-data \
  >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
