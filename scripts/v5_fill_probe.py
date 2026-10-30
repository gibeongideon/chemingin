#!/usr/bin/env python3
"""OCTOBER A6 — THE FILL PROBE. Measure what a resting buy-stop actually fills at.

§3ci decomposed the long-only prior-high break and found that **47% of its +10.161bp is a FILL
ADVANTAGE** -- the +4.729bp gap (t +7.45) between the prior-day high and where the breaking bar
actually closed. Two execution paths give opposite verdicts:

    resting STOP at the prior high   -> excess over drift +8.111bp, t +3.76  CLEARS
    detect on bar close, enter market -> excess over drift +3.382bp, t +1.68  FAILS

So the entire lead rests on one operational question, and **no amount of further backtesting can
answer it**: does a resting buy-stop at the prior-day high fill AT the level, or does it slip?
DSR, PBO and the walk-forward blend weight are all downstream of this.

WHAT THIS DOES. Each trading day it places ONE resting BUY STOP at the prior day's high, minimum
lot, dedicated magic. If price never reaches the level the order is cancelled at day's end and
nothing is recorded. If it fills, the probe records **the requested level, the actual fill price and
the difference in bp** -- which is the number §3ci needs -- then closes the position at the session
close, matching the backtested structure.

THIS IS A MEASUREMENT, NOT A STRATEGY. 0.01 lot is ~1oz (~$4,000 notional, ~$40 margin at 1:100), so
P&L is noise by design; the output is the slippage distribution. It uses its own magic (360599) and
touches nothing else -- the FundingPips book runs on 360598 and the zigzag on 360597, and every read
and write here is magic-scoped.

SAFETY. Refuses unless the account reports trade_mode 0 (DEMO), the symbol resolves with
trade_mode 4 (FULL), and the volume equals volume_min. --execute is opt-in; the default is a dry
report.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "v5_runs" / "october" / "a6_fill_probe.csv"
STATE = ROOT / "data" / "v5_runs" / "october" / "a6_fill_probe_state.json"
MAGIC = 360599
FIELDS = ["utc", "event", "symbol", "level", "fill_price", "slip_px", "slip_bp", "volume",
          "ticket", "retcode", "comment", "bid", "ask", "spread_px", "prev_day_high", "note"]
# Server time. MEASURED from the broker's own H1 bars on 2026-10-30: XAUUSD trades server hours
# 01:00-23:59 and hour 00 is the daily break. The backtest exits at the SESSION CLOSE, i.e. the
# 23:xx bar, so 23 is what matches it -- an earlier exit measures a different trade. (An earlier
# comment here guessed the break at 21:00-23:00, which was wrong.)
EXIT_HOUR = 23


def record(**kw) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: kw.get(k, "") for k in FIELDS})


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def save_state(d: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(d, indent=2))


def already_logged(event: str, ticket) -> bool:
    """Has this (event, ticket) pair already been written?

    The slippage number is the whole point of the probe, so it is recorded the MOMENT a fill is
    seen rather than at the exit -- a failed exit must not be able to lose the measurement. That
    makes the writer non-idempotent across an hourly timer, which this guards.
    """
    if not LOG.exists() or not ticket:
        return False
    with LOG.open(newline="") as fh:
        return any(r.get("event") == event and r.get("ticket") == str(ticket)
                   for r in csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18816)
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--execute", action="store_true", help="actually place/cancel/close orders")
    a = ap.parse_args()

    from mt5linux import MetaTrader5
    m = MetaTrader5(host="localhost", port=a.port)
    if not m.initialize():
        print("init failed:", m.last_error()); return 2
    ai = m.account_info()
    if ai is None:
        print("NOT LOGGED IN"); m.shutdown(); return 2
    if ai.trade_mode != 0:
        print(f"REFUSING: account {ai.login} trade_mode {ai.trade_mode} is not DEMO")
        m.shutdown(); return 1

    sym = a.symbol
    m.symbol_select(sym, True)
    si = m.symbol_info(sym)
    tick = m.symbol_info_tick(sym)
    if si is None or tick is None or not tick.ask:
        print("no quote for", sym); m.shutdown(); return 1
    if si.trade_mode != 4:
        print(f"REFUSING: {sym} trade_mode {si.trade_mode} is not FULL"); m.shutdown(); return 1
    vol = float(si.volume_min)

    # ---- is the market actually open? -----------------------------------------------------
    # Asked FRAME-INDEPENDENTLY: does the tick ADVANCE? Comparing a server-stamped tick against
    # host utcnow() reads a fresh quote as three hours old and makes the gate a no-op -- that bug
    # has shipped here twice (see memory: mt5-server-clock-vs-host-clock). On a closed market
    # every order_send below would fail, so this is the difference between a quiet skip and an
    # hourly stream of rejected `place` rows all weekend.
    import time as _time
    _t0 = tick
    _time.sleep(6)
    tick = m.symbol_info_tick(sym) or _t0
    if int(tick.time) == int(_t0.time) and float(tick.bid) == float(_t0.bid):
        print(f"  {sym} quote static over 6s -> market closed; nothing to do")
        m.shutdown(); return 0

    # ---- prior day's HIGH from COMPLETED daily bars (never today's) ------------------------
    r = m.copy_rates_from_pos(sym, 16408, 0, 5)          # D1
    if r is None or len(r) < 2:
        print("no D1 bars"); m.shutdown(); return 1
    import pandas as pd
    D = pd.DataFrame([(x[0], x[2]) for x in r], columns=["ts", "high"])
    D["t"] = pd.to_datetime(D.ts, unit="s")
    srv_now = pd.to_datetime(int(tick.time), unit="s")   # SERVER clock, never host UTC
    done = D[D.t.dt.date < srv_now.date()]
    if done.empty:
        print("no completed prior day"); m.shutdown(); return 1
    pdh = float(done.iloc[-1]["high"])

    mine_pos = [p for p in (m.positions_get(symbol=sym) or []) if p.magic == MAGIC]
    mine_ord = [o for o in (m.orders_get(symbol=sym) or []) if o.magic == MAGIC]
    print(f"account {ai.login} @ {ai.server} (DEMO)   server clock {srv_now}")
    print(f"  {sym}: bid {tick.bid:.2f} ask {tick.ask:.2f} spread {tick.ask-tick.bid:.2f}  "
          f"prior-day HIGH {pdh:.2f}")
    print(f"  magic {MAGIC}: {len(mine_pos)} position(s), {len(mine_ord)} pending order(s)")

    fm = int(si.filling_mode)
    fill_type = (m.ORDER_FILLING_FOK if fm & 1 else
                 m.ORDER_FILLING_IOC if fm & 2 else m.ORDER_FILLING_RETURN)
    base = dict(utc=datetime.now(timezone.utc).isoformat(), symbol=sym,
                bid=tick.bid, ask=tick.ask, spread_px=round(tick.ask - tick.bid, 3),
                prev_day_high=pdh, volume=vol)

    # ---- 1. a position exists: it FILLED. Record the slippage, exit at the session close ----
    if mine_pos:
        p = mine_pos[0]
        slip = p.price_open - pdh
        print(f"  FILLED: {p.volume} @ {p.price_open:.2f} vs level {pdh:.2f}  "
              f"-> slip {slip:+.3f} px = {slip/pdh*1e4:+.3f} bp   (pnl {p.profit:+.2f})")
        if not already_logged("fill", p.ticket):
            record(**base, event="fill", level=pdh, fill_price=p.price_open,
                   slip_px=round(slip, 3), slip_bp=round(slip / pdh * 1e4, 3),
                   ticket=p.ticket, note="recorded on first sight, not at exit")
        # A position that survived midnight must go NOW, not wait for hour 23 of the new day.
        # Without the staleness test `srv_now.hour >= EXIT_HOUR` is false at hour 0-22, so a
        # position carried through the 00:00 break would be held ~23 extra hours and pay a swap --
        # turning the one trade this probe is supposed to replicate into a different one.
        opened = pd.to_datetime(int(p.time), unit="s").date()
        stale = opened < srv_now.date()
        if srv_now.hour >= EXIT_HOUR or stale:
            if a.execute:
                t2 = m.symbol_info_tick(sym)
                rr = m.order_send(dict(action=m.TRADE_ACTION_DEAL, symbol=sym, volume=p.volume,
                                       type=m.ORDER_TYPE_SELL, position=p.ticket, price=t2.bid,
                                       deviation=50, magic=MAGIC, comment="fillprobe-exit",
                                       type_filling=fill_type))
                print(f"  EXIT at session close: retcode {rr.retcode}")
                record(**base, event="exit", level=pdh, fill_price=p.price_open,
                       slip_px=round(slip, 3), slip_bp=round(slip / pdh * 1e4, 3),
                       ticket=p.ticket, retcode=rr.retcode,
                       note="STALE: opened %s, forced out" % opened if stale
                            else f"exit_hour>={EXIT_HOUR}")
            else:
                print(f"  would EXIT ({'STALE' if stale else f'hour {srv_now.hour}'}); "
                      f"--execute to send")
        else:
            print(f"  holding until hour {EXIT_HOUR} server (now {srv_now.hour})")
        m.shutdown(); return 0

    # ---- 2. a pending order exists: cancel it at day's end, else leave it resting -----------
    if mine_ord:
        o = mine_ord[0]
        print(f"  RESTING buy-stop #{o.ticket} at {o.price_open:.2f} "
              f"(level {pdh:.2f}, {100*(o.price_open/tick.ask-1):+.3f}% above ask)")
        if srv_now.hour >= EXIT_HOUR:
            if a.execute:
                rr = m.order_send(dict(action=m.TRADE_ACTION_REMOVE, order=o.ticket))
                print(f"  CANCEL unfilled: retcode {rr.retcode}")
                record(**base, event="cancel_unfilled", level=o.price_open, ticket=o.ticket,
                       retcode=rr.retcode, note="never reached the level today")
            else:
                print("  would CANCEL (unfilled); --execute to send")
        m.shutdown(); return 0

    # ---- 3. nothing outstanding -------------------------------------------------------------
    # ONE ORDER PER SERVER DAY, and the state file is what enforces it. Without this the hourly
    # timer re-places the moment an order disappears for ANY reason -- which is not hypothetical:
    # the first two probe orders (17112033, 17112147) were removed from outside this terminal
    # within 75s and 16s, and the second run duly placed a duplicate. This account is hand-traded
    # concurrently (see memory: manual-trading-trips-the-bot-guard), so a vanished order is a
    # DATA POINT to write down, not a prompt to try again.
    st = load_state()
    day = str(srv_now.date())
    if st.get("day") == day:
        tk = st.get("ticket") or 0
        if not tk:
            print(f"  today's decision was already made and recorded ({st.get('why','skip')})")
        else:
            hist = m.history_orders_get(ticket=tk) or []
            state_code = int(hist[0].state) if hist else -1
            if state_code == 4:                     # ORDER_STATE_FILLED
                print(f"  order #{tk} filled earlier today and the position is already closed")
            else:
                print(f"  order #{tk} is GONE without filling (history state {state_code}) and "
                      f"was not cancelled by this probe — recording, not re-placing")
                if not already_logged("vanished", tk):
                    record(**base, event="vanished", level=pdh, ticket=tk,
                           note=f"disappeared externally, order state {state_code}")
        m.shutdown(); return 0

    if srv_now.hour >= EXIT_HOUR:
        print(f"  past exit hour {EXIT_HOUR}; not opening a new probe today")
        m.shutdown(); return 0
    if tick.ask >= pdh:
        print(f"  ask {tick.ask:.2f} is ALREADY above the level {pdh:.2f} — a stop here would "
              f"fill instantly at market, which is the arm this probe exists to AVOID measuring. "
              f"Skipping today.")
        record(**base, event="skip", level=pdh, note="ask already above level at decision time")
        save_state(dict(day=day, ticket=0, why="ask above level", level=pdh))
        m.shutdown(); return 0

    # The broker's minimum stop distance. A level sitting INSIDE it is rejected (retcode 10016),
    # and those days are exactly the near-misses a naive log would silently drop, so the skip is
    # written down with its reason rather than left as a gap in the series.
    stops_px = float(getattr(si, "trade_stops_level", 0) or 0) * float(si.point)
    if pdh - tick.ask < stops_px:
        print(f"  level {pdh:.2f} is inside the broker's {stops_px:.2f} minimum stop distance "
              f"from ask {tick.ask:.2f}; skipping")
        record(**base, event="skip", level=pdh,
               note=f"inside trade_stops_level ({stops_px:.2f} px)")
        save_state(dict(day=day, ticket=0, why="inside min stop distance", level=pdh))
        m.shutdown(); return 0

    req = dict(action=m.TRADE_ACTION_PENDING, symbol=sym, volume=vol,
               type=m.ORDER_TYPE_BUY_STOP, price=round(pdh, si.digits),
               deviation=0, magic=MAGIC, comment="fillprobe",
               type_time=m.ORDER_TIME_GTC, type_filling=fill_type)
    print(f"  PLACING buy-stop {vol} @ {pdh:.2f} (ask {tick.ask:.2f}, "
          f"{100*(pdh/tick.ask-1):+.3f}% away)")
    if not a.execute:
        print("  dry run — rerun with --execute to send")
        m.shutdown(); return 0
    rr = m.order_send(req)
    print(f"  retcode {rr.retcode}  {getattr(rr,'comment','')}  order {getattr(rr,'order',0)}")
    record(**base, event="place", level=round(pdh, si.digits), ticket=getattr(rr, "order", ""),
           retcode=rr.retcode, comment=getattr(rr, "comment", ""),
           note=f"{100*(pdh/tick.ask-1):+.3f}% above ask at placement")
    if int(rr.retcode) == 10009:
        save_state(dict(day=day, ticket=int(getattr(rr, "order", 0)), level=round(pdh, si.digits)))
    m.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
