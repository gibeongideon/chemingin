"""v5_xau_variants.py — ONE H4 XAU bot, three risk settings, optional weekend-flat.

    --bot full      champion at full exposure       CAGR +18.6%  maxDD -17.0%  SR 1.084
    --bot trim50    champion x (1 - 0.5*p_bad)      CAGR +15.5%  maxDD -13.2%  SR 1.178
    --bot trim100   champion x (1 - 1.0*p_bad)      CAGR +12.4%  maxDD  -9.4%  SR 1.329

(2018+, live FTMO cost $0.448/oz, continuous engine. The discrete executor here
tracks these closely but is not identical — it trades tickets with ATR trails.)

THESE ARE NOT THREE STRATEGIES. trim50/trim100 are the SAME champion scaled by a
calibrated P(forward P&L < 0): position <= champion's on 100% of bars, daily-return
correlation 0.96. Running two of them on ONE account is leverage on one correlated
bet paying the spread twice — pick ONE per account. Separate accounts with different
drawdown limits is the only sane way to run more than one.

HONESTY (V5_FINDINGS §3p): the trim FAILED its pre-registered gates — paired
t -2.11 vs the champion, positive in only 3 of 9 years. It converts return into
drawdown protection; it does not add alpha. Use it only where a hard DD limit binds.

WEEKEND-FLAT (`"weekend_flat": {"enabled": true}`) closes everything Friday 16:00 UTC
and blocks entries until the Sunday 22:00 re-open. It costs ~2.5-4.4pp of CAGR, makes
maxDD ~1.5-2.9pp WORSE (forced re-entry across the Monday gap), and quadruples
turnover. Enable ONLY when the prop firm forbids weekend exposure.

Reuses v5_xau_dual's validated plumbing (sizing via order_calc_profit, plan
reconciliation, the double --live/--execute safety lock) so the deployed cent/FP
bots are untouched.

    python scripts/v5_xau_variants.py --bot trim100                  # dry plan
    python scripts/v5_xau_variants.py --bot full --live --execute
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import v5_xau_dual as dual  # noqa: E402
import src.v5.xau_trend as xt  # noqa: E402
from src.core.mt5_connector import MT5Connector  # noqa: E402
from src.core.trade_journal import TradeJournal  # noqa: E402
from src.v5.news_filter import NewsFilter, apply_to_plan  # noqa: E402
from src.v5.xau_variant_signals import (VARIANTS, in_flat_window,  # noqa: E402
                                        variant_signal)

demo = dual.demo
CONFIG_FILE = ROOT / "configs" / "v5_xau_variants.json"


def apply_weekend_flat(actions, held, now: pd.Timestamp, wcfg: dict):
    """Drop every new-entry action and force-close any open position while inside
    the weekend window. Returns (actions, note)."""
    if not wcfg.get("enabled"):
        return actions, ""
    fri_h = int(wcfg.get("friday_hour_utc", 16))
    sun_h = int(wcfg.get("sunday_open_hour_utc", 22))
    if not in_flat_window(now, fri_h, sun_h):
        return actions, ""
    kept = [(a, d) for a, d in actions if a not in ("open_market",)]
    n_blocked = len(actions) - len(kept)
    if held is not None and not any(a == "close" for a, _ in kept):
        kept.append(("close", dict(position=held, ticket=held.ticket,
                                   why="weekend-flat: prop firm forbids weekend exposure")))
    note = (f"WEEKEND-FLAT active (Fri {fri_h:02d}:00 -> Sun {sun_h:02d}:00 UTC): "
            f"{n_blocked} entry action(s) blocked"
            + (", closing open position" if held is not None else ""))
    return kept, note


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bot", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--config", default=str(CONFIG_FILE))
    ap.add_argument("--live", action="store_true",
                    help="acknowledge trading a REAL account (required for real)")
    ap.add_argument("--execute", action="store_true",
                    help="actually send orders (needs --live on a real account)")
    ap.add_argument("--max-lot", type=float, default=0.02)
    ap.add_argument("--force-min-lot", action="store_true")
    ap.add_argument("--save-data", action="store_true")
    ap.add_argument("--weekend-flat", action="store_true",
                    help="force weekend-flat on regardless of config")
    ap.add_argument("--journal", default=str(ROOT / "data" / "live_trades.db"))
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    bot_cfg = cfg["bots"][args.bot]
    magic, run_id = bot_cfg["magic"], bot_cfg["run_id"]
    demo.RUN_ID = run_id
    journal = TradeJournal(args.journal)
    wcfg = dict(bot_cfg.get("weekend_flat", {}))
    if args.weekend_flat:
        wcfg["enabled"] = True

    conn = MT5Connector()
    conn.connect()
    try:
        acct, is_demo = dual.require_live(conn, args.live)
        send = args.execute and (is_demo or args.live)
        if args.execute and not send:
            print("  ! --execute ignored on REAL account without --live")

        symbol = demo.resolve_symbol(conn)
        h4 = dual.refresh_h4(conn, symbol, save=args.save_data)

        # variant exposure precomputed on the full frame (p_bad needs the features),
        # then handed to the discrete engine bar-by-bar
        sig = variant_signal(h4, args.bot,
                             model_path=bot_cfg.get("model_path",
                                                    ROOT / "data" / "models" / "xau_pbad_h4"))
        xt.xau_signal = lambda close: sig.reindex(close.index).astype(float)

        res = xt.run_trades(h4, equity0=float(acct.equity) or 3000.0,
                            exit_mode=cfg["exit_mode"], flip_mode=cfg["flip_mode"],
                            params=bot_cfg["params"])
        atr_last = float(xt.wilder_atr(h4, xt.PARAMS["atr_period"]).iloc[-1])
        pos, pending = res["open_position"], res["pending"]
        fc = float(res["signal"].iloc[-1])
        state = ("POSITION " + ("LONG" if pos["dir"] > 0 else "SHORT") if pos
                 else "ENTRY pending" if pending else "flat")
        print(f"  engine[{args.bot}]: forecast {fc:+.2f}  {state}")

        mine = [p for p in (conn.get_positions(magic=magic) or [])
                if p.symbol == symbol]
        held = mine[0] if mine else None
        held_dir = 0 if held is None else (1 if held.type == 0 else -1)
        pendings = demo.my_pendings(conn, symbol, magic)
        print(f"  broker[{magic}]: "
              f"{'flat' if held is None else f'{held.volume} lots dir {held_dir:+d} SL {held.sl}'}"
              f", {len(pendings)} pending order(s)")

        actions = dual.build_plan(res, held, held_dir, pendings,
                                  conn.get_tick(symbol), conn.symbol_info(symbol),
                                  acct, bot_cfg, args, conn, symbol, atr_last)

        nf_cfg = bot_cfg.get("news_filter", {})
        if nf_cfg.get("enabled"):
            verdict = NewsFilter(nf_cfg, root=ROOT).check()
            if verdict["blocked"]:
                print(f"  NEWS WINDOW: {verdict['event']} — entries paused")
            actions, n_blocked, profit_close = apply_to_plan(
                actions, held, verdict, nf_cfg.get("close_in_profit", True))
            if n_blocked:
                print(f"  NEWS: {n_blocked} new-entry action(s) blocked")

        now = pd.Timestamp(h4.index[-1])
        actions, wnote = apply_weekend_flat(actions, held, now, wcfg)
        if wnote:
            print(f"  {wnote}")
        elif wcfg.get("enabled"):
            print(f"  weekend-flat armed (Fri {wcfg.get('friday_hour_utc', 16):02d}:00 UTC) "
                  f"— outside the window, trading normally")

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
                                           a["lots"], sl=a["sl"], magic=magic,
                                           comment=run_id)
                elif act == "modify_sl":
                    r = conn.modify_position(a["position"].ticket, sl=a["sl"],
                                             tp=float(a["position"].tp or 0.0))
                elif act == "cancel":
                    r = demo.cancel_pending(conn, a["ticket"])
                journal.record(dict(
                    bot=f"v5_xau_variants_{args.bot}", symbol=symbol, direction=act,
                    entry_time=str(h4.index[-1]),
                    entry_reason=json.dumps(printable, default=str)[:180],
                    volume=a.get("lots", 0.0), sl_pips=a.get("sl"),
                    magic=magic, run_id=run_id, dry_run=0))
                print(f"    EXECUTED: "
                      f"{r.get('retcode', r) if isinstance(r, dict) else r}")
            except Exception as exc:  # noqa: BLE001
                print(f"    ORDER REJECTED: {exc}")
        if not send and actions:
            print("  (dry plan — rerun with --live --execute to send)")
    finally:
        conn.disconnect()


if __name__ == "__main__":
    main()
