#!/usr/bin/env python3
"""End-of-day report for a PDH day-trade arm on a prop account with a TRAILING max loss
(Goat Funded Pay Later 315531834, user 2026-11-27: "i need reporting every end of day").

Reads the account (bridge), the executor's state (initial, peak_equity, halt) and today's rows of its log,
and sends one message via v5_notify (Telegram + Resend + SMTP).

    python scripts/pdh_prop_report.py --port 18820 --tag goat --magic 360618 --symbol XAUUSD.x \
        --label "Goat Pay Later" --target 0.03 --maxloss 0.08 --halt-dd 0.065
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--magic", type=int, required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--label", default="PDH prop")
    ap.add_argument("--target", type=float, default=0.03)
    ap.add_argument("--maxloss", type=float, default=0.08)
    ap.add_argument("--halt-dd", type=float, default=0.065)
    ap.add_argument("--dry", action="store_true", help="print, do not send")
    a = ap.parse_args()

    from mt5linux import MetaTrader5  # noqa: PLC0415
    m = MetaTrader5(host="localhost", port=a.port)
    if not m.initialize():
        print("bridge init failed"); return 1
    ai = m.account_info()
    tick = m.symbol_info_tick(a.symbol)
    pos = [p for p in (m.positions_get(symbol=a.symbol) or []) if int(p.magic) == a.magic]
    ords = [o for o in (m.orders_get(symbol=a.symbol) or []) if int(o.magic) == a.magic]
    # NO m.shutdown(): shared bridge
    import pandas as pd  # noqa: PLC0415
    srv_day = str(pd.to_datetime(int(tick.time), unit="s").date()) if tick else "?"

    runs = ROOT / "data" / "v5_runs"
    try:
        st = json.loads((runs / f"pdh_{a.tag}_state.json").read_text())
    except Exception:
        st = {}
    rows = []
    log = runs / f"pdh_{a.tag}_log.csv"
    if log.exists():
        with log.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
    init = float(st.get("initial") or ai.balance)
    peak = max(float(st.get("peak_equity") or 0.0), float(ai.equity), float(ai.balance))
    floor = min(peak - a.maxloss * init, init)            # firm: trails the high, locks at the initial balance
    bot_halt = peak - a.halt_dd * init
    tgt = init * (1 + a.target)
    prog = (ai.equity - init) / (tgt - init) * 100
    today = [r for r in rows if r.get("day") == srv_day]
    closed = [r for r in rows if r.get("event") == "closed"]
    day_pnl = sum(float(r.get("pnl") or 0) for r in today if r.get("event") == "closed")
    tot_r = sum(float(r.get("R") or 0) for r in closed)
    fills = [float(r["slip_bp"]) for r in rows if r.get("event") == "fill" and r.get("slip_bp") not in (None, "")]
    halt = (st.get("halt") or {}).get("reason")

    lines = [f"{a.label} — PDH day-trade — end of day {srv_day} (server)",
             f"account {ai.login} @ {ai.server}   magic {a.magic}",
             "",
             f"BALANCE / EQUITY   {ai.balance:,.2f} / {ai.equity:,.2f}",
             f"TODAY closed P&L   {day_pnl:+,.2f}",
             f"TOTAL vs start     {ai.equity - init:+,.2f} ({(ai.equity / init - 1) * 100:+.2f}%)",
             "",
             f"TARGET +{a.target * 100:.0f}% = {tgt:,.0f}   progress {prog:5.1f}%",
             f"FIRM FLOOR (trailing {a.maxloss * 100:.0f}%) {floor:,.2f}   headroom {ai.equity - floor:,.2f}"
             f"   (peak {peak:,.2f}{', LOCKED at start' if floor >= init else ''})",
             f"BOT HALT (stops entries) {bot_halt:,.2f}",
             "",
             "TODAY'S EVENTS: " + (", ".join(f"{r['event']}" + (f" {r['level']}" if r.get('level') else "")
                                             + (f" pnl {float(r['pnl']):+.2f}" if r.get('pnl') else "")
                                             for r in today) or "none"),
             "OPEN: " + (", ".join(f"BUY {p.volume} @ {p.price_open:.2f} sl {p.sl:.2f} pnl {p.profit:+.2f}" for p in pos)
                         or "no position") + ("; " + ", ".join(f"BUY STOP {o.volume_current} @ {o.price_open:.2f}"
                                                                for o in ords) if ords else ""),
             f"ALL-TIME: {len(closed)} trades, {tot_r:+.2f}R"
             + (f"; resting-stop slip mean {sum(fills) / len(fills):+.2f}bp over {len(fills)} fills (kill > 2bp over 40)"
                if fills else ""),
             f"STATUS: {'⛔ ' + halt if halt else 'trading'}"]
    body = "\n".join(lines)
    subj = (f"[{a.label}] {srv_day} eq {ai.equity:,.0f} | target {prog:.0f}% | floor headroom {ai.equity - floor:,.0f}"
            + (" | HALTED" if halt else ""))
    if a.dry:
        print(subj); print(body); return 0
    from scripts.v5_notify import notify  # noqa: PLC0415
    notify(subj, body, subj)
    print("sent:", subj)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
