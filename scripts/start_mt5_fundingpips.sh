#!/bin/bash
# Headless MT5 launcher — FUNDINGPIPS TRIAL instance 5 (prefix ~/.mt5e, display :103, bridge 18816).
# Account 40000397948 / FundingPips-Trial. Mirrors start_mt5_ftmo.sh exactly; only the
# prefix, display and port differ.
#
# PREFIX-SCOPED RUNNING CHECK. A bare `pgrep -f terminal64.exe` would match the Maven, cent and
# FTMO terminals and wrongly skip launching this one. Worse, a bare `pkill -f terminal64.exe`
# anywhere in this file would kill all four. Every process decision below is gated on the
# WINEPREFIX read out of /proc/<pid>/environ.
#
# /config: IS REQUIRED. Passed as a BARE argument the ini is silently ignored and the terminal
# starts logged out; MT5 only honours it with the `/config:` prefix and a WINDOWS path.
#
# AND THE INI IS ONLY A COLD-START FALLBACK. MT5 caches the last good login in common.ini and
# that cache BEATS the startup ini (see memory: account-switch-and-crash-state-traps). This
# terminal was first authorized by a manual GUI login on 2026-10-19 -- which is also what
# fetched SwitchMarkets-Live into servers.dat, something no config file can do. If the server
# entry is ever lost, no amount of ini editing will reconnect it: open the GUI and re-add the
# broker.
set -u
export DISPLAY=:103 WINEPREFIX=$HOME/.mt5e WINEARCH=win64 WINEDEBUG=-all
export WINEDLLOVERRIDES="mscoree,mshtml="
TERM_EXE="$HOME/.mt5e/drive_c/Program Files/MetaTrader 5/terminal64.exe"
WINPY="$HOME/.mt5e/drive_c/Python310/python.exe"
PORT=18816

term_running() {
  local p
  for p in $(pgrep -f terminal64.exe 2>/dev/null); do
    if tr "\0" "\n" < "/proc/$p/environ" 2>/dev/null | grep -qx "WINEPREFIX=$WINEPREFIX"; then
      return 0
    fi
  done
  return 1
}

if ! term_running; then
  if [ -f "$HOME/.mt5e/drive_c/mt5_login.ini" ]; then
    wine "$TERM_EXE" "/config:C:\\mt5_login.ini" >/dev/null 2>&1 &
  else
    wine "$TERM_EXE" >/dev/null 2>&1 &
  fi
  sleep 25
fi
if ! ss -tlnp 2>/dev/null | grep -q "127.0.0.1:$PORT"; then
  wine "$WINPY" -c "from rpyc.utils.server import ThreadedServer; from rpyc.core import SlaveService; ThreadedServer(SlaveService, hostname=\"127.0.0.1\", port=$PORT, reuse_addr=True).start()" >/dev/null 2>&1 &
  sleep 4
fi
