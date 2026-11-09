#!/usr/bin/env python3
"""Telegram trade events for the three Switch order-placers — the ONLY bot messages on the phone.

User request 2026-11-06: "Telegram should send only tradable, traded and closed", plus a weekly
P/L per bot. One read-only watcher replaces each bot's own chatter (their wrappers now export
NOTIFY_NO_TELEGRAM=1, so lock/trail/digest messages go to email only).

EVENTS (diff of the account between runs, magic-filtered):
  TRADABLE  a new pending order (e.g. trail-runner's buy-stop at the prior-day high)
  TRADED    a new position, or a position whose volume changed (book rebalances)
  CLOSED    a position that disappeared — exit price and realised P&L from the deal history
WEEKLY (--weekly, Saturday timer): realised P&L, trades, wins per bot over the last 7 days of
  server time, plus each bot's open floating P&L and the account equity.

Read-only: never sends an order, never calls shutdown() (bridge 18815 is shared), yields to any
peer holding its bridge lock. First run records the current state silently (no flood).

    python scripts/v5_trade_events_notify.py --port 18815
    python scripts/v5_trade_events_notify.py --port 18815 --weekly
"""
from __future__ import annotations

import argparse
import fcntl
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STATE = ROOT / "data" / "v5_runs" / "trade_events_state.json"
BOTS = {360595: "Book (XAU+BTC champion)", 360601: "ZigZag", 360600: "Trail-runner"}
ORDER_TYPES = {2: "BUY LIMIT", 3: "SELL LIMIT", 4: "BUY STOP", 5: "SELL STOP",
               6: "BUY STOP LIMIT", 7: "SELL STOP LIMIT"}
PEERS_BY_PORT = {
    18815: ("/tmp/xau-btc-switch.lock", "/tmp/zigzag-trail-switch.lock",
            "/tmp/zigzag-trail-watch.lock", "/tmp/trail-runner.lock", "/tmp/xau-advisor.lock"),
    18814: ("/tmp/zigzag-trail-ftmo-b.lock", "/tmp/zigzag-trail-watch.lock"),
    18817: ("/tmp/xau-darwinex.lock", "/tmp/zigzag-trail-darwinex.lock",
            "/tmp/zigzag-trail-watch.lock", "/tmp/trail-runner-darwinex.lock"),
}
LABEL = "Switch"
BRIDGE_PEERS = PEERS_BY_PORT[18815]
# ZigZag probability alerts (user 2026-11-09): WATCH at P >= 0.55, SIGNAL at the config threshold
# (0.60, where the bot itself places the order). Read from the bot's own state file, once per H1 bar.
ZZ_STATE_BY_PORT = {18815: "zigzag_trail_switch_state.json", 18817: "zigzag_trail_darwinex_state.json",
                    18814: "zigzag_trail_ftmo_b_state.json"}
ZZ_WATCH = 0.55


def zz_alert(zz: dict | None, last_bar_alerted: str | None) -> tuple[str, str] | None:
    """Pure: (head, body) when the ZigZag state shows a NEW bar with P >= ZZ_WATCH, else None."""
    if not zz or zz.get("prob") is None or zz.get("last_bar") == last_bar_alerted:
        return None
    if zz.get("executed") is False:
        # A DRY run (no --live --execute) also writes this file. 2026-11-09 one did, and the phone
        # got "order placed" for an order that was never sent. Only LIVE passes may alert.
        return None
    p, thr = float(zz["prob"]), float(zz.get("threshold", 0.6))
    if p < ZZ_WATCH:
        return None
    pr = zz.get("prospective") or {}
    px = f"{pr.get('symbol', '')} {pr.get('price', 0):,.2f}"
    if p >= thr:
        sent = [s for s in (zz.get("sent") or []) if "BUY" in str(s) and "10009" in str(s)]   # 10009 = done
        what = (f"skipped: gold trend down (champion {zz.get('trend_fc', 0):.2f} < gate)" if zz.get("trend_gate_blocked") else
                "order placed" if sent else
                "no new order: already in a ZigZag trade" if zz.get("open_positions") else
                "order NOT confirmed — check the bot log")
        return f"🎯 ZigZag SIGNAL · P {p:.2f}", f"{px} · threshold {thr:.2f} · {what}\nbar {zz['last_bar']}"
    return (f"👀 ZigZag WATCH · P {p:.2f}",
            f"{px} · buys at {thr:.2f}; close to a signal\nbar {zz['last_bar']}")


def wait_for_peers(max_wait: float = 45.0) -> str | None:
    """Yield to a peer on the bridge, but WAIT for it instead of skipping the run: with the
    every-minute trail watcher on the same bridge, skip-on-busy skipped 2,031 runs in a row
    (2026-11-06..09) and the Switch watcher sent nothing."""
    import time  # noqa: PLC0415
    t0 = time.time()
    while True:
        busy = peers_busy()
        if busy is None or time.time() - t0 > max_wait:
            return busy
        time.sleep(1.5)


def peers_busy() -> str | None:
    """Test-and-release only: holding a peer's lock would make that peer skip its own run."""
    for path in BRIDGE_PEERS:
        try:
            with open(path, "a") as fh:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(fh, fcntl.LOCK_UN)
                except BlockingIOError:
                    return path
        except OSError:
            continue
    return None


def money(x: float) -> str:
    return f"{'+' if x >= 0 else '-'}${abs(x):,.2f}"


def snapshot(m) -> tuple[dict, dict]:
    orders = {str(o.ticket): dict(magic=int(o.magic), symbol=o.symbol, type=int(o.type),
                                  volume=float(o.volume_initial), price=float(o.price_open),
                                  sl=float(o.sl), tp=float(o.tp))
              for o in (m.orders_get() or []) if int(o.magic) in BOTS}
    positions = {str(p.ticket): dict(magic=int(p.magic), symbol=p.symbol, type=int(p.type),
                                     volume=float(p.volume), price=float(p.price_open),
                                     sl=float(p.sl), profit=float(p.profit))
                 for p in (m.positions_get() or []) if int(p.magic) in BOTS}
    return orders, positions


def diff(prev: dict, orders: dict, positions: dict) -> list[tuple[str, str, dict]]:
    """Pure: (kind, ticket, info) for every event between two snapshots."""
    ev = []
    for t, o in orders.items():
        if t not in prev.get("orders", {}):
            ev.append(("TRADABLE", t, o))
    for t, p in positions.items():
        old = prev.get("positions", {}).get(t)
        if old is None:
            ev.append(("TRADED", t, p))
        elif abs(old["volume"] - p["volume"]) > 1e-9:
            ev.append(("RESIZED", t, dict(p, old_volume=old["volume"])))
    for t, p in prev.get("positions", {}).items():
        if t not in positions:
            ev.append(("CLOSED", t, p))
    return ev


MILESTONES = (50.0, 200.0, 300.0)   # user 2026-11-06: ZigZag floating +$50/+200/+300, so SL can be moved by hand


def milestones(prev: dict, positions: dict) -> list[tuple[str, str, dict]]:
    """Pure: one PROFIT event per ZigZag position per run, for the highest newly-crossed level.
    Levels already hit are carried in positions[t]['hits'] (mutated here) so each fires once."""
    ev = []
    for t, p in positions.items():
        hits = list(prev.get("positions", {}).get(t, {}).get("hits", []))
        new = []
        if "zigzag" in BOTS.get(p["magic"], "").lower():
            new = [lv for lv in MILESTONES if p["profit"] >= lv and lv not in hits]
            hits += new
        p["hits"] = hits
        if new:
            ev.append(("PROFIT", t, dict(p, level=max(new))))
    return ev


def close_details(m, ticket: str) -> tuple[float | None, float | None]:
    deals = m.history_deals_get(position=int(ticket)) or []
    if not deals:
        return None, None
    pnl = sum(float(d.profit) + float(d.swap) + float(d.commission) for d in deals)
    outs = [d for d in deals if int(d.entry) in (1, 2, 3)]          # OUT, INOUT, OUT_BY
    return pnl, (float(outs[-1].price) if outs else None)


def message(m, kind: str, t: str, x: dict) -> tuple[str, str]:
    bot = BOTS[x["magic"]]
    side = "BUY" if x["type"] == 0 else "SELL"
    sl = f" · SL {x['sl']:,.2f}" if x.get("sl") else ""
    if kind == "TRADABLE":
        head = f"🟡 TRADABLE · {bot}"
        line = f"{ORDER_TYPES.get(x['type'], x['type'])} {x['symbol']} {x['volume']} @ {x['price']:,.2f}{sl}"
    elif kind == "TRADED":
        head = f"🟢 TRADED · {bot}"
        line = f"{side} {x['symbol']} {x['volume']} @ {x['price']:,.2f}{sl}"
    elif kind == "RESIZED":
        head = f"🔵 TRADED (resized) · {bot}"
        line = f"{side} {x['symbol']} {x['old_volume']} → {x['volume']} (entry {x['price']:,.2f})"
    elif kind == "PROFIT":
        head = f"💰 +${x['level']:,.0f} REACHED · {bot}"
        line = (f"{side} {x['symbol']} {x['volume']} @ {x['price']:,.2f} · floating {money(x['profit'])}"
                f" · SL {x['sl']:,.2f}\nYou may raise the SL (bot only ever raises it, never lowers)")
    else:
        pnl, exit_px = close_details(m, t)
        head = f"🔴 CLOSED · {bot}"
        ex = f" → {exit_px:,.2f}" if exit_px else ""
        pl = money(pnl) if pnl is not None else f"~{money(x['profit'])} (last seen)"
        line = f"{side} {x['symbol']} {x['volume']} · {x['price']:,.2f}{ex} · P&L {pl}"
    return head, f"{line}\n#{t} · {LABEL}"


def weekly(m) -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    deals = m.history_deals_get(now - timedelta(days=7), now + timedelta(days=1)) or []
    acct = m.account_info()
    rows = []
    for magic, name in BOTS.items():
        mine = [d for d in deals if int(d.magic) == magic]
        closes = [d for d in mine if int(d.entry) in (1, 2, 3)]
        realised = sum(float(d.profit) + float(d.swap) + float(d.commission) for d in mine)
        wins = sum(1 for d in closes if float(d.profit) + float(d.swap) > 0)
        floating = sum(float(p.profit) for p in (m.positions_get() or []) if int(p.magic) == magic)
        rows.append((name, realised, len(closes), wins, floating))
    tot = sum(r[1] for r in rows)
    lines = [f"Week to {now:%a %d %b %Y} (last 7 days)"]
    for name, real, n, w, fl in rows:
        lines.append(f"• {name}: realised {money(real)} · {n} closed ({w} won) · open {money(fl)}")
    lines.append(f"TOTAL realised {money(tot)} · equity ${float(acct.equity):,.2f}")
    return f"📊 WEEKLY P/L · {LABEL}", "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18815)
    ap.add_argument("--weekly", action="store_true")
    ap.add_argument("--dry", action="store_true", help="print, do not send")
    ap.add_argument("--label", default="Switch", help="account name shown in every message")
    ap.add_argument("--bots", default="",
                    help="magic=Name,... e.g. '360605=Book (XAU champion),360606=ZigZag,360607=Trail-runner'")
    a = ap.parse_args()
    global LABEL, BRIDGE_PEERS, STATE
    LABEL = a.label
    BRIDGE_PEERS = PEERS_BY_PORT.get(a.port, ())
    if a.bots:
        BOTS.clear()
        for item in a.bots.split(","):
            k, v = item.split("=", 1)
            BOTS[int(k)] = v.strip()
    if a.port != 18815:                    # one state file per account
        STATE = STATE.with_name(f"trade_events_{a.port}_state.json")
    busy = wait_for_peers() if not a.weekly else None
    if busy:
        print(f"peer busy ({busy}); skipping this run")
        return 0
    from mt5linux import MetaTrader5  # noqa: PLC0415
    from scripts.v5_notify import notify  # noqa: PLC0415
    m = MetaTrader5(host="localhost", port=a.port)
    if not m.initialize():
        print("init failed", m.last_error())
        return 1
    # NO m.shutdown() anywhere: bridge 18815 is shared.
    if a.weekly:
        subj, body = weekly(m)
        print(subj + "\n" + body)
        if not a.dry:
            notify(subj, body, f"{subj}\n{body}")
        return 0
    orders, positions = snapshot(m)
    prev = json.loads(STATE.read_text()) if STATE.exists() else None
    if prev is None:
        print(f"first run: recording {len(orders)} order(s), {len(positions)} position(s) silently")
    else:
        for kind, t, x in diff(prev, orders, positions) + milestones(prev, positions):
            head, body = message(m, kind, t, x)
            print(head, "|", body.replace("\n", " | "))
            if not a.dry:
                notify(head, body, f"{head}\n{body}")
    zz_bar = (prev or {}).get("zz_bar")
    zf = ZZ_STATE_BY_PORT.get(a.port)
    if zf and (ROOT / "data" / "v5_runs" / zf).exists():
        zz = json.loads((ROOT / "data" / "v5_runs" / zf).read_text())
        alert = zz_alert(zz, zz_bar)
        if alert:
            head, body = alert
            print(head, "|", body.replace("\n", " | "))
            if not a.dry:
                notify(head, f"{body}\n{LABEL}", f"{head}\n{body}\n{LABEL}")
        if zz.get("executed") is not False:          # a dry-run state must not consume the bar
            zz_bar = zz.get("last_bar", zz_bar)
    if not a.dry:
        STATE.write_text(json.dumps(dict(orders=orders, positions=positions, zz_bar=zz_bar,
                                         at=datetime.now(timezone.utc).isoformat()), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
