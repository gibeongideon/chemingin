#!/bin/bash
# Start MetaTrader 5 and the mt5linux rpyc bridge on Ubuntu.
#
# What this script does:
#   1. Launches terminal64.exe under Wine (~/.mt5 prefix)
#   2. Launches the rpyc classic server inside Wine Python (port 18812)
#      so native Linux Python can call MT5 via mt5linux
#
# Prerequisites: run ./setup.sh first (installs Wine Python + bridge deps)

set -e

WINEPREFIX="$HOME/.mt5"
MT5_DIR="$WINEPREFIX/drive_c/Program Files/MetaTrader 5"
TERMINAL="$MT5_DIR/terminal64.exe"
BRIDGE_PORT=18812

# Wine Python installed by setup.sh
WINE_PYTHON_DIR="$WINEPREFIX/drive_c/Python310"
WINE_PYTHON="$WINE_PYTHON_DIR/python.exe"
WINE_PIP="$WINE_PYTHON_DIR/Scripts/pip.exe"

export WINEPREFIX

# ─── Wayland/Xwayland env (local desktop only; VPS uses Xvfb, no cookie) ──────
# terminal64.exe dies at launch with `nodrv_CreateWindow` unless it gets the
# mutter Xwayland auth cookie — ~/.Xauthority does NOT work, and the cookie name
# changes every login, so resolve the live one dynamically. On the VPS there is
# no such file, the block is skipped, and the existing Xvfb DISPLAY is used.
_cookie=$(ls -t /run/user/"$(id -u)"/.mutter-Xwaylandauth.* 2>/dev/null | head -1)
if [ -n "$_cookie" ]; then
    export XAUTHORITY="$_cookie"
    export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
    export DISPLAY="${DISPLAY:-:0}"
    echo "Wayland session: XAUTHORITY=$XAUTHORITY  DISPLAY=$DISPLAY"
fi

# Render terminal64 inside a Wine VIRTUAL DESKTOP (explorer /desktop=). Under
# Wayland/XWayland, letting Wine map each window through the native compositor
# produces a garbled ("deformed") UI; a virtual desktop makes Wine draw its own
# top-level window and manage children itself, which renders correctly. Set
# MT5_NO_DESKTOP=1 to fall back to native windowing.
# Sized to the local screen (1536x864) so a maximized MT5 fills it and the Wine
# desktop frame is fully hidden behind MT5. Override with MT5_DESKTOP=WxH.
MT5_DESKTOP="${MT5_DESKTOP:-1536x864}"

# A stale HEADLESS wineserver (e.g. one started by the bridge with no display env)
# poisons every terminal it spawns -> nodrv / deformed UI. Reset it once so the
# fresh wineserver inherits the Wayland env above. Skipped with MT5_KEEP_SERVER=1.
reset_wineserver() {
    if [ "${MT5_KEEP_SERVER:-0}" = "1" ]; then return; fi
    if pgrep -f "wineserver" >/dev/null 2>&1 && ! pgrep -f "terminal64.exe" >/dev/null 2>&1; then
        echo "Resetting stale headless wineserver (frees bridge port, no terminal was running)..."
        wineserver -k 2>/dev/null || true
        sleep 2
    fi
}

# ─── helpers ───────────────────────────────────────────────────────────────
check_wine() {
    if ! command -v wine &>/dev/null; then
        echo "ERROR: wine not found. Install: sudo apt install winehq-stable"
        exit 1
    fi
}

install_terminal() {
    # Download terminal64.exe directly from MetaQuotes CDN — no installer GUI needed
    local dest="$MT5_DIR/terminal64.exe"
    local url="https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/terminal64.exe"
    echo "Downloading terminal64.exe from MetaQuotes CDN..."
    mkdir -p "$MT5_DIR"
    curl -L "$url" -o "$dest" --progress-bar
    if [ -f "$dest" ]; then
        echo "OK  terminal64.exe downloaded ($(du -sh "$dest" | cut -f1))"
    else
        echo "ERROR: Download failed. Check your internet connection."
        exit 1
    fi
}

start_terminal() {
    if pgrep -f "terminal64.exe" >/dev/null 2>&1; then
        echo "MT5 terminal already running."
        return
    fi

    if [ ! -f "$TERMINAL" ]; then
        echo "terminal64.exe not found — downloading from MetaQuotes CDN..."
        install_terminal

        if [ ! -f "$TERMINAL" ]; then
            echo ""
            echo "ERROR: terminal64.exe still not found after install."
            exit 1
        fi
    fi

    echo "Starting MT5 terminal..."
    # Run from a temp copy so the liveupdate can replace the Program Files version
    # (Wine blocks writing to a file that is currently executing — running from a copy avoids this)
    local TEMP_LAUNCHER="$WINEPREFIX/drive_c/users/$USER/AppData/Local/Temp/mt5_launcher"
    mkdir -p "$TEMP_LAUNCHER"
    cp "$TERMINAL" "$TEMP_LAUNCHER/terminal64.exe"
    if [ "${MT5_NO_DESKTOP:-0}" = "1" ]; then
        WINEPREFIX="$WINEPREFIX" WINEDEBUG=-all wine "$TEMP_LAUNCHER/terminal64.exe" &
    else
        echo "Launching in Wine virtual desktop ($MT5_DESKTOP) to avoid deformed XWayland UI..."
        WINEPREFIX="$WINEPREFIX" WINEDEBUG=-all wine explorer /desktop=MT5,"$MT5_DESKTOP" \
            "$TEMP_LAUNCHER/terminal64.exe" &
    fi
    echo "MT5 terminal launched (PID: $!)"
    echo "  → Log in to your broker account"
    echo "  → Enable: Tools > Options > Expert Advisors > Allow algorithmic trading"
    sleep 15
}

start_bridge() {
    if pgrep -f "SlaveService\|rpyc.*18812" >/dev/null 2>&1; then
        echo "rpyc bridge already running on port $BRIDGE_PORT."
        return
    fi

    if [ ! -f "$WINE_PYTHON" ]; then
        echo "ERROR: Wine Python not found at $WINE_PYTHON"
        echo "Run setup.sh first to install Python inside Wine."
        exit 1
    fi

    echo "Starting mt5linux rpyc bridge on port $BRIDGE_PORT..."
    # Run rpyc SlaveService inside Wine Python — mt5linux (Linux) connects to this
    WINEPREFIX="$WINEPREFIX" WINEDEBUG=-all wine "$WINE_PYTHON" -c \
      "from rpyc.utils.server import ThreadedServer; from rpyc.core import SlaveService; ThreadedServer(SlaveService, hostname='127.0.0.1', port=$BRIDGE_PORT, reuse_addr=True).start()" &
    BRIDGE_PID=$!
    sleep 3

    if kill -0 $BRIDGE_PID 2>/dev/null; then
        echo "Bridge running (PID: $BRIDGE_PID) — localhost:$BRIDGE_PORT"
    else
        echo "WARNING: Bridge may have failed to start. Check Wine output above."
    fi
}

# ─── main ──────────────────────────────────────────────────────────────────
check_wine
echo "=== Starting MT5 on Ubuntu ==="
echo "WINEPREFIX: $WINEPREFIX"
echo ""
reset_wineserver
start_terminal
start_bridge
echo ""
echo "=== Ready ==="
echo "  conda activate envmt5"
echo "  python tests/test_connection.py   # verify connection"
echo "  python src/example_bot.py         # run example bot"
