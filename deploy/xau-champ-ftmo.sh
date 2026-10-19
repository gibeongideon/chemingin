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
# MARKET-OPEN TEST, FRAME-INDEPENDENT.
#
# The previous version did `age = time.time() - tick.time`, which compared two DIFFERENT clocks:
# MT5 stamps ticks in SERVER time (FTMO runs UTC+3) while time.time() is host UTC. That made a
# fresh tick read as -180.0 min old, and since -180 <= 10 the gate passed unconditionally --
# it only ever blocked on a weekend, when the true age grew past the 180-minute offset. So the
# guard was not guarding.
#
# Rather than hard-code or guess the offset (it shifts an hour with US DST, and 3be had to
# correct exactly that kind of claim), ask the only question that needs no shared clock:
# DOES THE TICK ADVANCE? Gold ticks several times a second when open, so a 15-second gap is
# decisive, and the test is immune to any offset because both readings sit in the same frame.
import sys, time
from mt5linux import MetaTrader5
mt5 = MetaTrader5(host="localhost", port=18814)
if not mt5.initialize():
    print(f"  market check: bridge init failed {mt5.last_error()}"); sys.exit(2)
mt5.symbol_select("XAUUSD", True)
t1 = mt5.symbol_info_tick("XAUUSD")
time.sleep(15)
t2 = mt5.symbol_info_tick("XAUUSD")
mt5.shutdown()
if t1 is None or t2 is None:
    print("  market check: no tick object"); sys.exit(2)
advanced = int(t2.time) > int(t1.time) or float(t2.bid) != float(t1.bid)
print(f"  market check: tick {int(t1.time)} -> {int(t2.time)}, bid {t1.bid} -> {t2.bid}"
      f"  => {'OPEN' if advanced else 'CLOSED'}")
sys.exit(0 if advanced else 1)
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
