#!/bin/bash
# OCTOBER A6 — the fill probe. SWITCH MARKETS 40080430, bridge 18815, symbol XAUUSDs, magic 360599.
#
# *** MOVED OFF FUNDINGPIPS 2026-10-30. *** Three of three probe orders there were cancelled from
# outside the terminal (§3cl): the terminal log carries cancel lines for the two the diagnostic
# cancelled itself and NONE for the probe's, while the account took heavy manual flow — five
# orders at once at 16:47. A `vanished` row is not a fill measurement, so the probe cannot collect
# its sample on a hand-traded account.
#
# SWITCH IS A DEMO DESPITE THE SERVER NAME. account_info().trade_mode == 0 on SwitchMarkets-Live
# 40080430 (verified 2026-10-19 in configs/v5_xau_btc_switch.json, re-verified 2026-10-30), so the
# probe's demo-only lock passes without being relaxed. Balance ~$125k, leverage 1:100; one minimum
# lot of XAUUSDs is 1 oz, about $39 of margin.
#
# SYMBOL IS XAUUSDs, NOT XAUUSD. Every Switch symbol carries an `s` suffix and plain XAUUSD does
# not resolve there at all — symbol_info returns None, which is how this was caught.
# Spread measured 0.280 against FundingPips' 0.150, and swap_mode 1 (POINTS) is irrelevant here
# because the probe is flat before the 00:00 rollover.
#
# This is a MEASUREMENT, not a strategy. §3ci found that 47% of the prior-high break's +10.161bp
# is a FILL ADVANTAGE (+4.729bp, t +7.45) -- the gap between the prior-day high and where the
# breaking bar closed -- and that the two execution paths disagree: a resting stop at the level
# clears (+8.111bp over drift, t +3.76), a market entry on bar close does not (+3.382bp, t +1.68).
# No further backtesting can settle which one a live order gets. One minimum-lot stop a day can.
#
# It shares the terminal with the live book (magic 360598) and the zigzag (360597) and is scoped
# to its own magic at every read and write. Runs hourly: place early in the session, notice the
# fill, record the slippage, exit or cancel at the session close.
set -uo pipefail
cd /home/trader/MT5 || exit 1
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/fill-probe.log
mkdir -p data/v5_runs
exec 9>/tmp/fill-probe.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) previous run still going, skipping" >> "$LOG"; exit 0; }
echo "===== $(date -u +%FT%TZ) =====" >> "$LOG"
$PY scripts/v5_fill_probe.py --port 18815 --symbol XAUUSDs --execute >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
