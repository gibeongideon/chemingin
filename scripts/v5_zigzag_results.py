#!/usr/bin/env python3
"""Closed-trade results for a magic — did each trade WIN or LOSE, and what is the running record.

WHY THIS IS SEPARATE FROM THE SIGNAL. The ZigZag notifier reads the executor's state file and
says what the detector THINKS. That is a forecast. This module reads the broker's own deal
history and says what actually HAPPENED. The two must never be built from the same source: a
state file records the order the bot *sent*, which is not evidence that it filled, held, or paid
what the backtest charged.

*** THE MAGIC-ON-CLOSE TRAP, WHICH THIS FILE EXISTS TO AVOID. ***
A deal's `magic` is the magic of the ORDER THAT PRODUCED IT. When a position opened by the bot
is closed by anything else — a server-side SL/TP, or a human clicking Close — the closing deal
carries **magic 0**. Verified in this account set: position 17154140 opened under magic 360597
and closed under magic 0 for -$191.50. So filtering every deal by magic silently drops exactly
the closes we most need to see, and a losing bot reads as a bot with no results at all.

The correct shape, implemented below: find position ids whose OPENING deal carries our magic,
then take **every** deal belonging to those positions regardless of magic.

    python scripts/v5_zigzag_results.py --port 18815 --magic 360601
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEAL_ENTRY_IN = 0
DEAL_ENTRY_OUT = 1
DEAL_ENTRY_INOUT = 2
DEAL_ENTRY_OUT_BY = 3
CLOSING = (DEAL_ENTRY_OUT, DEAL_ENTRY_INOUT, DEAL_ENTRY_OUT_BY)


def _ts(epoch: int) -> datetime:
    """Broker deal stamps are server time. Labelled `srv` everywhere it is printed, because
    this account set runs at UTC+3 and three dead gates have already come from assuming
    otherwise."""
    return datetime.fromtimestamp(int(epoch), timezone.utc)


def closed_trades(m, magic: int, days: int = 3650) -> list[dict]:
    """Every CLOSED position opened under `magic`, oldest first.

    Returns dicts with: position, symbol, side, volume, open/close time and price, held_h,
    profit, swap, commission, net, won.
    """
    start = datetime.now(timezone.utc) - timedelta(days=days)
    end = datetime.now(timezone.utc) + timedelta(days=3)      # server clock runs ahead of UTC
    deals = m.history_deals_get(start, end)
    if not deals:
        return []

    ours, by_pos = set(), {}
    for d in deals:
        pid = getattr(d, "position_id", 0)
        if not pid:
            continue
        by_pos.setdefault(pid, []).append(d)
        if getattr(d, "magic", 0) == magic and getattr(d, "entry", None) == DEAL_ENTRY_IN:
            ours.add(pid)

    out = []
    for pid in ours:
        ds = sorted(by_pos[pid], key=lambda d: d.time)
        opens = [d for d in ds if d.entry == DEAL_ENTRY_IN]
        closes = [d for d in ds if d.entry in CLOSING]
        if not opens or not closes:
            continue                                    # still open — not a result yet
        o, c = opens[0], closes[-1]
        profit = sum(float(getattr(d, "profit", 0.0)) for d in ds)
        swap = sum(float(getattr(d, "swap", 0.0)) for d in ds)
        comm = sum(float(getattr(d, "commission", 0.0)) for d in ds)
        net = profit + swap + comm
        out.append(dict(
            position=pid, symbol=o.symbol, side="BUY" if o.type == 0 else "SELL",
            volume=float(o.volume), open_time=_ts(o.time), close_time=_ts(c.time),
            open_price=float(o.price), close_price=float(c.price),
            held_h=(int(c.time) - int(o.time)) / 3600.0,
            profit=profit, swap=swap, commission=comm, net=net, won=net > 0,
            closed_by_us=getattr(c, "magic", 0) == magic))
    return sorted(out, key=lambda t: t["close_time"])


def render(trades: list[dict], equity: float | None = None, show: int = 6) -> str:
    """The results block, newest trades last. Honest about ties and about costs."""
    if not trades:
        return "RESULTS: no closed trades yet."
    wins = [t for t in trades if t["net"] > 0]
    losses = [t for t in trades if t["net"] < 0]
    flat = len(trades) - len(wins) - len(losses)
    net = sum(t["net"] for t in trades)
    costs = sum(t["swap"] + t["commission"] for t in trades)

    L = [f"RESULTS — {len(trades)} closed trade(s)"]
    for t in trades[-show:]:
        verdict = "WIN " if t["net"] > 0 else ("LOSS" if t["net"] < 0 else "FLAT")
        by = "" if t["closed_by_us"] else "  [closed outside the bot]"
        L.append(f"  {t['close_time']:%Y-%m-%d %H:%M} srv  {verdict}  {t['net']:+.2f}   "
                 f"{t['side']} {t['volume']:g} {t['symbol']}  "
                 f"{t['open_price']:.2f} -> {t['close_price']:.2f}  "
                 f"held {t['held_h']:.0f}h{by}")
    if len(trades) > show:
        L.append(f"  ... {len(trades) - show} earlier trade(s) not shown")

    rate = len(wins) / (len(wins) + len(losses)) if (wins or losses) else float("nan")
    rec = f"  Record: {len(wins)}W / {len(losses)}L"
    if flat:
        rec += f" / {flat} flat"
    rec += f"   win rate {rate:.0%}" if rate == rate else ""
    L.append(rec)
    pct = f" ({net / equity:+.2%} of equity)" if equity else ""
    L.append(f"  Net: {net:+.2f}{pct}    of which financing+commission {costs:+.2f}")
    return "\n".join(L)


def render_short(trades: list[dict]) -> str:
    """One line: the record and the net. Everything else is in the email."""
    if not trades:
        return "no closed trades yet"
    w = sum(1 for t in trades if t["net"] > 0)
    ln = sum(1 for t in trades if t["net"] < 0)
    return f"record {w}W/{ln}L   net {sum(t['net'] for t in trades):+.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18815)
    ap.add_argument("--magic", type=int, default=360601)
    ap.add_argument("--days", type=int, default=3650)
    a = ap.parse_args()

    from src.core.mt5_connector import get_mt5          # noqa: E402 — slow import, keep lazy
    m = get_mt5("localhost", a.port)
    if not m.initialize():
        print(f"MT5 bridge {a.port} not connected"); return 1
    acc = m.account_info()
    print(render(closed_trades(m, a.magic, a.days),
                 equity=float(acc.equity) if acc else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
