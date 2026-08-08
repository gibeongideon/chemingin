"""v5_intraday_monitor.py — continuous intraday monitor/executor for the high-R:R book.

Unlike the H4 bots (hourly oneshot systemd timers that reconcile and exit), an intraday
high-R:R strategy must WATCH the position: it enters on a bar close with a fixed ATR stop
and an RR target placed broker-side, then manages the trade — banking profit early when a
move stalls, cutting early when it turns, and timing out trades that go nowhere.

    ENTRY   fixed stop = stop_atr x ATR ; broker-side TP at RR x stop  (survives a crash)
    MANAGE  MFE/MAE tracked in R units, evaluated every poll
    EXIT    early-TP on stall  |  early-SL on adverse excursion  |  time stop  |  broker SL/TP

*** THE ENTRY RULE IS DELIBERATELY A NO-OP UNTIL RESEARCH NAMES ONE. ***
Phase-0 measurement (scripts/v5_intraday_lab.py) established that RANDOM intraday entries
on XAU lose -0.11 to -0.18 R per trade net of live cost. Shipping an unvalidated entry
would just automate that loss. `entry_signal()` returns 0 and the bot logs "no validated
entry" until `configs/v5_intraday.json` names a rule that passed the research gates.

The risk layer IS live and is the point of the design:
  - fixed-fractional risk per trade off the ACTUAL stop distance
  - hard daily-loss limit and max concurrent positions (prop-firm aligned)
  - loss-streak aware: at RR2-3 with ~30% win rate, runs of 8-12 losses are NORMAL.
    The bot must not "adapt" after a streak — no martingale, ever (V5_FINDINGS 3l).

    python scripts/v5_intraday_monitor.py --once            # single pass, dry
    python scripts/v5_intraday_monitor.py --minutes 60      # watch an hour, dry
    python scripts/v5_intraday_monitor.py --live --execute  # arm it (needs a real entry)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "configs" / "v5_intraday.json"
MAGIC = 360551


def log(msg: str, path: Path | None = None) -> None:
    line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(line + "\n")


# ------------------------------------------------------------------ signal
def entry_signal(bars, cfg) -> int:
    """Return +1 long / -1 short / 0 flat, decided on the LAST CLOSED bar.

    Returns 0 until `cfg["entry"]["rule"]` names a rule that passed the research gates.
    Measured Phase-0 baseline: random entries = -0.11 to -0.18 R/trade. There is no
    'harmless' placeholder entry — anything not validated is a measured loser.
    """
    rule = (cfg.get("entry") or {}).get("rule")
    if not rule or rule == "none":
        return 0
    raise NotImplementedError(
        f"entry rule {rule!r} is named in the config but not implemented yet — "
        "the research workflow must land it first")


def atr_from(bars, n: int = 14) -> float:
    """Wilder ATR on plain floats (never let numpy cross the rpyc bridge)."""
    trs = []
    for i in range(1, len(bars)):
        h, l, pc = bars[i][2], bars[i][3], bars[i - 1][4]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < n:
        return 0.0
    a = sum(trs[:n]) / n
    for tr in trs[n:]:
        a = (a * (n - 1) + tr) / n
    return float(a)


# -------------------------------------------------------------- management
def manage(pos, bid: float, ask: float, cfg, opened_at: float) -> tuple[str | None, str]:
    """Decide whether to close an open position early. Returns (action, why).

    All thresholds are in R units so they transfer across instruments and volatility.
    """
    m = cfg.get("manage", {})
    d = 1 if pos["type"] == 0 else -1
    entry, risk = float(pos["price_open"]), float(pos["risk"])
    if risk <= 0:
        return None, ""
    px = bid if d > 0 else ask                      # exit at the adverse side
    r_now = d * (px - entry) / risk
    mfe = max(float(pos.get("mfe_r", 0.0)), r_now)
    mae = min(float(pos.get("mae_r", 0.0)), r_now)
    pos["mfe_r"], pos["mae_r"] = mfe, mae
    held_min = (time.time() - opened_at) / 60.0

    # 1) early stop — cut before the full 1R when the trade turns hard
    esl = m.get("early_sl_r")
    if esl is not None and r_now <= -abs(float(esl)):
        return "close", f"early-SL at {r_now:+.2f}R (threshold -{abs(float(esl)):.2f}R)"

    # 2) give-back — banked profit fading from its peak
    gb = m.get("giveback_frac")
    if gb is not None and mfe >= float(m.get("giveback_arm_r", 1.0)):
        if r_now <= mfe * (1.0 - float(gb)):
            return "close", (f"give-back: peak {mfe:+.2f}R -> {r_now:+.2f}R "
                             f"({float(gb):.0%} surrendered)")

    # 3) time stop — an unresolved trade is consuming risk budget for nothing
    tmax = m.get("max_minutes")
    if tmax is not None and held_min >= float(tmax):
        return "close", f"time stop at {held_min:.0f}min (max {float(tmax):.0f})"

    # 4) breakeven ratchet — move the broker stop up, do NOT close
    be = m.get("breakeven_at_r")
    if be is not None and mfe >= float(be) and not pos.get("be_done"):
        return "breakeven", f"MFE {mfe:+.2f}R >= {float(be):.2f}R — stop to entry"
    return None, ""


# ------------------------------------------------------------------ risk
def risk_gate(acct, state, cfg) -> tuple[bool, str]:
    """Prop-firm aligned guards. Blocking is ALWAYS allowed; sizing never increases."""
    r = cfg.get("risk", {})
    eq, bal = float(acct.equity), float(acct.balance)
    day_start = float(state.get("day_start_equity") or bal)
    day_pl = (eq - day_start) / day_start
    lim = float(r.get("daily_loss_limit", 0.03))
    if day_pl <= -lim:
        return False, f"DAILY LOSS LIMIT hit ({day_pl:+.2%} <= -{lim:.2%}) — flat for today"
    tot = (eq - float(state.get("start_equity") or bal)) / float(state.get("start_equity") or bal)
    hard = float(r.get("max_total_loss", 0.06))
    if tot <= -hard:
        return False, f"TOTAL LOSS LIMIT hit ({tot:+.2%}) — halted"
    return True, ""


def size_lots(conn, symbol, direction, price, sl, equity, cfg, si) -> float:
    """Fixed-fractional risk off the ACTUAL stop distance, via the broker's own
    profit calc (correct on any account currency / contract size)."""
    mt5 = conn._mt5
    order = mt5.ORDER_TYPE_BUY if direction > 0 else mt5.ORDER_TYPE_SELL
    loss = abs(mt5.order_calc_profit(order, symbol, 1.0, float(price), float(sl)))
    if not loss:
        return 0.0
    frac = float(cfg.get("risk", {}).get("risk_frac", 0.005))
    step = float(si.volume_step) or 0.01
    lots = round(round((frac * equity) / loss / step) * step, 2)
    lots = float(min(lots, float(cfg.get("risk", {}).get("max_lot", 0.10))))
    return lots if lots >= float(si.volume_min) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--port", type=int, default=int(os.environ.get("MT5_BRIDGE_PORT", 18814)))
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--minutes", type=float, default=0.0)
    ap.add_argument("--poll", type=float, default=20.0)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    logp = ROOT / "data" / "v5_runs" / "intraday_monitor.log"
    statep = ROOT / "data" / "v5_runs" / "intraday_state.json"
    state = json.loads(statep.read_text()) if statep.exists() else {}

    from src.core.mt5_connector import MT5Connector
    conn = MT5Connector(port=args.port)
    conn.connect()
    try:
        acct = conn.account_info()
        is_demo = getattr(acct, "trade_mode", None) == 0 or "demo" in str(acct.server).lower()
        send = args.execute and (is_demo or args.live)
        symbol = cfg.get("symbol", "XAUUSD")
        conn._mt5.symbol_select(symbol, True)
        log(f"=== intraday monitor  {acct.login}@{acct.server}  eq {acct.equity:,.2f}  "
            f"{'DEMO' if is_demo else 'REAL'}  send={send} ===", logp)
        state.setdefault("start_equity", float(acct.balance))
        state["day_start_equity"] = state.get("day_start_equity") or float(acct.balance)

        rule = (cfg.get("entry") or {}).get("rule", "none")
        if rule in (None, "none"):
            log("  ENTRY DISABLED — no validated rule in config. Phase-0 measured random "
                "intraday entries at -0.11..-0.18 R/trade; an unvalidated entry is a "
                "measured loser. Management + risk layers still active on any open trade.",
                logp)

        end = time.time() + (args.minutes * 60 if args.minutes else 0)
        while True:
            acct = conn.account_info()
            ok, why = risk_gate(acct, state, cfg)
            mine = [p for p in (conn.get_positions(magic=MAGIC) or []) if p.symbol == symbol]
            tick = conn.get_tick(symbol)
            bid, ask = float(tick.bid), float(tick.ask)

            for p in mine:
                held = state.get("positions", {}).get(str(p.ticket), {})
                pos = dict(type=p.type, price_open=float(p.price_open),
                           risk=float(held.get("risk") or 0.0),
                           mfe_r=held.get("mfe_r", 0.0), mae_r=held.get("mae_r", 0.0),
                           be_done=held.get("be_done", False))
                act, reason = manage(pos, bid, ask, cfg, held.get("opened_at", time.time()))
                state.setdefault("positions", {})[str(p.ticket)] = {
                    **held, "mfe_r": pos["mfe_r"], "mae_r": pos["mae_r"]}
                d = 1 if p.type == 0 else -1
                r_now = d * ((bid if d > 0 else ask) - float(p.price_open)) / max(pos["risk"], 1e-9)
                log(f"  hold {'LONG' if d>0 else 'SHORT'} {float(p.volume):.2f} @ "
                    f"{float(p.price_open):.2f}  now {r_now:+.2f}R  "
                    f"MFE {pos['mfe_r']:+.2f}R  MAE {pos['mae_r']:+.2f}R  pnl {float(p.profit):+.2f}",
                    logp)
                if act == "close":
                    log(f"    ACTION close — {reason}", logp)
                    if send:
                        conn.close_position(p)
                        state["positions"].pop(str(p.ticket), None)
                elif act == "breakeven":
                    log(f"    ACTION stop->breakeven — {reason}", logp)
                    if send:
                        conn.modify_position(p.ticket, sl=float(p.price_open),
                                             tp=float(p.tp or 0.0))
                        state["positions"][str(p.ticket)]["be_done"] = True

            if not ok:
                log(f"  RISK GATE: {why}", logp)
            elif not mine and int(cfg.get("risk", {}).get("max_positions", 1)) > 0:
                raw = conn._mt5.copy_rates_from_pos(symbol, conn._mt5.TIMEFRAME_M15, 0, 120)
                # `raw or []` would raise: rates come back as a numpy array over the
                # rpyc bridge and its truth value is ambiguous. Test for None explicitly,
                # then materialise to plain floats (np.* must never cross the bridge).
                bars = [] if raw is None else [
                    (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]))
                    for b in raw]
                sig = entry_signal(bars, cfg) if bars else 0
                if sig:
                    a = atr_from(bars)
                    st = float(cfg["entry"]["stop_atr"]) * a
                    price = ask if sig > 0 else bid
                    sl = price - sig * st
                    tp = price + sig * float(cfg["entry"]["rr"]) * st
                    lots = size_lots(conn, symbol, sig, price, sl, float(acct.equity), cfg,
                                     conn.symbol_info(symbol))
                    log(f"  SIGNAL {'LONG' if sig>0 else 'SHORT'} {lots} lots  "
                        f"stop {sl:.2f} ({st:.2f}={cfg['entry']['stop_atr']}xATR)  tp {tp:.2f}",
                        logp)
                    if send and lots > 0:
                        r = conn.open_position(symbol, "buy" if sig > 0 else "sell", lots,
                                               sl=round(sl, 2), tp=round(tp, 2),
                                               magic=MAGIC, comment="v5-intraday")
                        log(f"    SENT: {r}", logp)

            statep.parent.mkdir(parents=True, exist_ok=True)
            statep.write_text(json.dumps(state, indent=2, default=str))
            if args.once or (args.minutes and time.time() >= end):
                break
            time.sleep(args.poll)
    finally:
        conn.disconnect()


if __name__ == "__main__":
    main()
