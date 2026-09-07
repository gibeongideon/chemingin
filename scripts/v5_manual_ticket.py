"""MANUAL TRADE TICKET — decision support for accounts where automated execution is not
permitted. Computes what the champion wants to hold today and prints an instruction for a
HUMAN to place. It never sends an order and has no code path that can.

Why this exists: some accounts prohibit EA / automated trading. The compliant answer is not
to disguise a bot but to run the strategy at a cadence a person can actually execute. Measured
cost of that (V5_FINDINGS §3ai): checking once per day at 20:00 UTC with the no-trade band
widened from 0.10 to 0.20 takes net Sharpe from +1.090 to +1.038 and cuts required actions from
136/yr to 64/yr — about one trade a week for ~5% of the edge.

    python scripts/v5_manual_ticket.py --balance 100000 --held 0.15
    python scripts/v5_manual_ticket.py --balance 100000 --port 18812   # read held size live

`--port` uses a READ-ONLY positions query. Reading is not trading; if you would rather not have
any bridge attached to a manual-only account, pass --held and type the number yourself.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v5.xau_dual_signals import champion_signal  # noqa: E402

TARGET_VOL, MAX_LEV = 0.10, 8.0
ANN, VOL_HL = 252 * 6, 42
BUFFER_MANUAL = 0.20          # widened from the automated 0.10 — see docstring
CHECK_HOUR_UTC = 20
CONTRACT = 100.0              # oz per lot on XAUUSD / GoldEternal
MIN_LOT, LOT_STEP = 0.01, 0.01


def load_h4() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data/XAUUSD_H4_long.csv", parse_dates=["time"]).set_index("time")
    return d[~d.index.duplicated(keep="last")].sort_index()


def target_position(df: pd.DataFrame, buffer_mult: float) -> tuple[pd.Series, pd.Series]:
    """Returns (buffered target leverage, raw target leverage). Identical maths to the
    automated engine — only the buffer width and the decision cadence differ."""
    close = df["close"].astype(float)
    ret = close.pct_change()
    vol = ret.ewm(halflife=VOL_HL, min_periods=20).std() * np.sqrt(ANN)
    fc = champion_signal(close)
    raw = (fc.clip(-2.0, 2.0) * (TARGET_VOL / vol)).clip(-MAX_LEV, MAX_LEV)
    band = (buffer_mult * (TARGET_VOL / vol).clip(0, MAX_LEV)).values
    hrs = raw.index.hour.values
    p, out, held = raw.values.copy(), np.zeros(len(raw)), 0.0
    for i in range(len(p)):
        if hrs[i] == CHECK_HOUR_UTC and np.isfinite(p[i]):
            b = band[i] if np.isfinite(band[i]) else 0.0
            if abs(p[i] - held) > b:
                held = p[i] - np.sign(p[i] - held) * b
        out[i] = held
    return pd.Series(out, index=raw.index), raw


def held_from_bridge(port: int, symbol: str) -> float | None:
    try:
        from mt5linux import MetaTrader5
    except ImportError:
        print("  (mt5linux not installed here — run with --held instead)")
        return None
    mt5 = MetaTrader5(host="localhost", port=port)
    if not mt5.initialize():
        print(f"  (bridge {port} not answering: {mt5.last_error()} — use --held)")
        return None
    pos = mt5.positions_get(symbol=symbol) or []
    net = sum((p.volume if p.type == 0 else -p.volume) for p in pos)
    ai = mt5.account_info()
    print(f"  read from bridge {port}: login {ai.login}, {len(pos)} open on {symbol}, "
          f"net {net:+.2f} lots, equity {ai.equity:,.2f}")
    mt5.shutdown()
    return float(net)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--balance", type=float, required=True, help="account equity in USD")
    ap.add_argument("--symbol", default="GoldEternal", help="broker symbol you will trade")
    ap.add_argument("--held", type=float, default=None, help="lots you currently hold (+long)")
    ap.add_argument("--port", type=int, default=None, help="read held size from this bridge")
    ap.add_argument("--buffer", type=float, default=BUFFER_MANUAL)
    ap.add_argument("--dial", type=float, default=None,
                    help="vol dial, e.g. 0.20; default = the engine's own 10%%")
    args = ap.parse_args()

    df = load_h4()
    pos, raw = target_position(df, args.buffer)
    last_bar = pos.index[-1]
    lev = float(pos.iloc[-1])
    if args.dial:
        lev *= args.dial / TARGET_VOL
    px = float(df["close"].iloc[-1])

    tgt_lots = lev * args.balance / (px * CONTRACT)
    tgt_lots = round(tgt_lots / LOT_STEP) * LOT_STEP
    held = args.held if args.held is not None else (
        held_from_bridge(args.port, args.symbol) if args.port else None)

    stale = (datetime.now(timezone.utc) - last_bar.tz_localize("UTC")).total_seconds() / 3600
    print("\n" + "=" * 66)
    print(f"  MANUAL TRADE TICKET   {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
    print("=" * 66)
    print(f"  symbol            {args.symbol}")
    print(f"  last H4 bar       {last_bar:%Y-%m-%d %H:%M}  close ${px:,.2f}"
          f"   ({stale:.0f}h old)")
    if stale > 12:
        print(f"  ** DATA IS {stale:.0f}h STALE — refresh data/XAUUSD_H4_long.csv before acting **")
    print(f"  forecast          {float(champion_signal(df['close']).iloc[-1]):.3f}  (0 = flat, 2 = max)")
    print(f"  target leverage   {lev:.3f}   ->   TARGET {tgt_lots:.2f} lots on ${args.balance:,.0f}")

    if held is None:
        print(f"\n  currently held    unknown — re-run with --held <lots>")
        print(f"  ACTION            hold {tgt_lots:.2f} lots total")
    else:
        delta = round((tgt_lots - held) / LOT_STEP) * LOT_STEP
        print(f"  currently held    {held:+.2f} lots")
        if abs(delta) < MIN_LOT:
            print(f"\n  ACTION            NOTHING. Difference {delta:+.2f} lots is below the "
                  f"{MIN_LOT} minimum.")
        else:
            verb = "BUY" if delta > 0 else "SELL"
            print(f"\n  ACTION            {verb} {abs(delta):.2f} lots   "
                  f"({held:+.2f} -> {tgt_lots:+.2f})")
    print(f"\n  next check        tomorrow {CHECK_HOUR_UTC:02d}:00 UTC")
    print(f"  cadence           once per day, buffer {args.buffer:.2f} — measured 64 actions/yr,")
    print(f"                    net Sharpe +1.038 vs +1.090 for the every-H4-bar automated form")
    print(f"  missed a day?     harmless. This is a multi-week trend follower; it degrades")
    print(f"                    gracefully. Do NOT catch up by trading twice.")
    print("=" * 66 + "\n")
    print("  This script places no orders. You place them.\n")


if __name__ == "__main__":
    main()
