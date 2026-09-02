"""v5_maven_champ.py — MT5 executor for the champion on the Maven account.

Same signal, same discrete engine, same double --live/--execute safety lock
as v5_xau_dual.py (reused directly, not duplicated) — this is a thin sibling
pointed at the Maven bridge (18812) with its own magic (360571) and config
(configs/v5_xau_maven.json).

ONE DELIBERATE DIFFERENCE from v5_xau_dual.py: Maven's rule set includes a
0% daily-loss constraint that the standard discrete trail-stop engine cannot
itself guarantee (backtested honestly in v5_maven_zero_dd_champion.py — no
parameterization of that structure gets close; see V5_FINDINGS). Per the
user's explicit decision (2026-08-31): deploy the champion as-is and manage
exits MANUALLY for Zero-rule compliance, rather than have the bot force a
same-day close.

That creates a real hazard the plain reconcile loop doesn't handle: the
engine's simulated position state has no memory of a manual close, so the
very next hourly pass would see "engine holds position, broker flat" and
REOPEN it — undoing the user's manual profit-take. `once_per_day_guard()`
below closes that gap: if ANY deal (open or close) has already posted under
this magic today (UTC, matching Maven's own reset boundary), a fresh
"open_market" action is refused for the rest of that day. At most one new
entry per UTC day; everything after that is the user's call.

    python scripts/v5_maven_champ.py --live --execute --save-data
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.v5_xau_dual as dual  # noqa: E402
from src.core.mt5_connector import MT5Connector  # noqa: E402
from src.core.trade_journal import TradeJournal  # noqa: E402
from src.v5.xau_dual_signals import SIGNALS  # noqa: E402

CONFIG_FILE = ROOT / "configs" / "v5_xau_maven.json"
MAVEN_PORT = 18812


def once_per_day_guard(conn: MT5Connector, magic: int) -> bool:
    """True if a new entry is ALLOWED this pass — False if any deal (open or
    close) has already posted under this magic today (UTC). Queries MT5
    history directly rather than trusting local state, so it stays correct
    even if a position was closed by hand outside any script."""
    mt5 = conn._mt5
    now_utc = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    day_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    # pad the window like ftmo_daily_report does — server time can differ from UTC
    deals = mt5.history_deals_get(day_start - dt.timedelta(hours=6), now_utc + dt.timedelta(hours=6))
    if not deals:
        return True
    today_utc_date = now_utc.date()
    for d in deals:
        if d.magic != magic:
            continue
        deal_date = dt.datetime.utcfromtimestamp(d.time).date()
        if deal_date == today_utc_date:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(CONFIG_FILE))
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--max-lot", type=float, default=0.05)
    ap.add_argument("--force-min-lot", action="store_true")
    ap.add_argument("--save-data", action="store_true")
    ap.add_argument("--journal", default=str(ROOT / "data" / "live_trades.db"))
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    bot_cfg = cfg["bots"]["champ"]
    magic, run_id = bot_cfg["magic"], bot_cfg["run_id"]
    dual.demo.RUN_ID = run_id
    journal = TradeJournal(args.journal)

    conn = MT5Connector(port=MAVEN_PORT)
    conn.connect()
    try:
        acct, is_demo = dual.require_live(conn, args.live)
        send = args.execute and (is_demo or args.live)
        if args.execute and not send:
            print("  ! --execute ignored on REAL account without --live")

        symbol = dual.demo.resolve_symbol(conn)
        h4 = dual.refresh_h4(conn, symbol, save=args.save_data)

        dual.xt.xau_signal = lambda close: SIGNALS["champ"](close)
        res = dual.xt.run_trades(h4, equity0=float(acct.equity) or 100000.0,
                                 exit_mode=cfg["exit_mode"], flip_mode=cfg["flip_mode"],
                                 params=bot_cfg["params"])
        atr_last = float(dual.xt.wilder_atr(h4, dual.xt.PARAMS["atr_period"]).iloc[-1])
        pos, pending = res["open_position"], res["pending"]
        fc = float(res["signal"].iloc[-1])
        state = ("POSITION " + ("LONG" if pos["dir"] > 0 else "SHORT") if pos
                 else "ENTRY pending" if pending else "flat")
        print(f"  engine[champ]: forecast {fc:+.2f}  {state}")
        if pos:
            print(f"          entry ~{pos['entry']:.2f}  SL {pos['sl']:.2f}"
                  f"{' (trailing)' if pos['trail_on'] else ''}  conf {pos['conf']}")

        mine = [p for p in (conn.get_positions(magic=magic) or []) if p.symbol == symbol]
        held = mine[0] if mine else None
        held_dir = 0 if held is None else (1 if held.type == 0 else -1)
        pendings = dual.demo.my_pendings(conn, symbol, magic)
        print(f"  broker[{magic}]: "
              f"{'flat' if held is None else f'{held.volume} lots dir {held_dir:+d} SL {held.sl}'}"
              f", {len(pendings)} pending order(s)")

        actions = dual.build_plan(res, held, held_dir, pendings,
                                  conn.get_tick(symbol), conn.symbol_info(symbol),
                                  acct, bot_cfg, args, conn, symbol, atr_last)

        opens_before = [a for a in actions if a[0] == "open_market"]
        if opens_before:
            allowed = once_per_day_guard(conn, magic)
            if not allowed:
                actions = [a for a in actions if a[0] != "open_market"]
                print("  ONCE-PER-DAY GUARD: a deal already posted under this magic today (UTC) "
                      "— refusing to open a new position this pass (manual close protected).")

        if not actions:
            print("  PLAN: in sync — nothing to do")
        for act, a in actions:
            printable = {k: v for k, v in a.items() if k != "position"}
            print(f"  PLAN: {act} {printable}")
            if not send:
                continue
            try:
                if act == "close":
                    r = conn.close_position(a["position"])
                elif act == "open_market":
                    r = conn.open_position(symbol, "buy" if a["dir"] > 0 else "sell",
                                           a["lots"], sl=a["sl"], magic=magic, comment=run_id)
                elif act == "modify_sl":
                    r = conn.modify_position(a["position"].ticket, sl=a["sl"],
                                             tp=float(a["position"].tp or 0.0))
                elif act == "cancel":
                    r = dual.demo.cancel_pending(conn, a["ticket"])
                journal.record(dict(
                    bot="v5_maven_champ", symbol=symbol, direction=act,
                    entry_time=str(h4.index[-1]),
                    entry_reason=json.dumps(printable, default=str)[:180],
                    volume=a.get("lots", 0.0), sl_pips=a.get("sl"),
                    magic=magic, run_id=run_id, dry_run=0))
                print(f"    EXECUTED: {r.get('retcode', r) if isinstance(r, dict) else r}")
            except Exception as exc:  # noqa: BLE001
                print(f"    ORDER REJECTED: {exc}")
        if not send and actions:
            print("  (dry plan — rerun with --live --execute to send)")
    finally:
        conn.disconnect()


if __name__ == "__main__":
    main()
