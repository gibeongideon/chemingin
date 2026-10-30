#!/usr/bin/env python3
"""The intraday trail-follower, exactly as INTRADAY-TRAIL-SPEC.md specifies. Switch demo.

WHAT IT TRADES. Long-only. One resting BUY STOP per server day at the PRIOR DAY'S HIGH, cancelled
if the prior day's LOW trades first. Initial stop entry - 1.5 x DAILY ATR(14). Thereafter a
RATCHETING trail at high-water - 3.0 x daily ATR, never loosened. **No take-profit and no
move-to-breakeven.** Max hold 5 days.

EVERY PARAMETER IS A MEASUREMENT, NOT A PREFERENCE:
  * long-only          §3bn -- long-only 1.091 vs long/short 0.954, both halves negative for shorts
  * no take-profit     §3bl -- a fixed bracket costs even a PERFECT-FORESIGHT oracle -0.94/-1.43bp
  * trail 3.0 ATR      §3bn -- peaks at 3.0, smooth decay both sides, ratchet worth +0.077 Sharpe
  * NO breakeven move  §3co -- -6.36bp/trade at t -3.55; hit rate collapses 0.5345 -> 0.4044
  * H1                 §3co -- M15/M30/H1 agree to 0.004 Sharpe; H1 is a quarter of the polling
  * DAILY ATR          §3co -- a 1.5-daily-ATR stop is ~20 M15 bar-ranges away, which is what makes
                       bar resolution irrelevant here. Using BAR ATR would reintroduce the
                       unresolvable intrabar ordering that inflated NR7 19x.

*** THIS IS A MEASUREMENT DEPLOYMENT. *** §3ck: the strategy LOSES to holding gold on CAGR
(+11.06% vs +11.96%) and wins only on risk, its advantage over a PLACEBO level sits at p 0.05-0.09
(§3cj), its slippage budget is 3.5bp and is still being measured, and Bar 1 has never been
computed. It runs on a DEMO at 0.5% risk to observe execution, not because it is a validated edge.

SAFETY. Refuses unless the account reports trade_mode 0 (DEMO) and the symbol trade_mode 4 (FULL).
--execute is opt-in. The protective stop is attached to the ORDER so it is server-side from the
moment of the fill; this process only ever RATCHETS an existing stop and never relies on being
alive to protect a position.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "v5_runs" / "trail_runner_log.csv"
STATE = ROOT / "data" / "v5_runs" / "trail_runner_state.json"
MAGIC = 360600                       # 360595 book · 360597 zigzag · 360598 FP book · 360599 probe
INIT_STOP_ATR, TRAIL_ATR = 1.5, 3.0
MAX_HOLD_DAYS = 5
RISK_FRAC = 0.005
DAY_END_HOUR = 23                    # XAUUSD trades server 01:00-23:59; hour 00 is the break
FIELDS = ["utc", "event", "symbol", "level", "price", "sl", "volume", "ticket", "retcode",
          "atr", "hwm", "note"]


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


def daily_frame(m, sym: str, n: int = 60) -> pd.DataFrame:
    r = m.copy_rates_from_pos(sym, 16408, 0, n)          # D1
    if r is None or len(r) < 20:
        return pd.DataFrame()
    d = pd.DataFrame([(x[0], x[1], x[2], x[3], x[4]) for x in r],
                     columns=["ts", "open", "high", "low", "close"])
    d["t"] = pd.to_datetime(d.ts, unit="s")              # SERVER clock, never host UTC
    return d.set_index("t").drop(columns=["ts"])


def wilder_atr(d: pd.DataFrame, n: int = 14) -> float:
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / n, adjust=False).mean().iloc[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18815)
    ap.add_argument("--symbol", default="XAUUSDs")
    ap.add_argument("--risk-frac", type=float, default=RISK_FRAC)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()

    from mt5linux import MetaTrader5
    m = MetaTrader5(host="localhost", port=a.port)
    if not m.initialize():
        print("bridge init failed:", m.last_error()); return 2
    ai = m.account_info()
    if ai is None:
        print("NOT LOGGED IN"); m.shutdown(); return 2
    if ai.trade_mode != 0:
        print(f"REFUSING: account {ai.login} trade_mode {ai.trade_mode} is not DEMO")
        m.shutdown(); return 1

    sym = a.symbol
    m.symbol_select(sym, True)
    si, tick = m.symbol_info(sym), m.symbol_info_tick(sym)
    if si is None or tick is None or not tick.ask:
        print(f"no quote for {sym} (Switch symbols carry an 's' suffix)"); m.shutdown(); return 1
    if si.trade_mode != 4:
        print(f"REFUSING: {sym} trade_mode {si.trade_mode} is not FULL"); m.shutdown(); return 1

    # ---- is the market open? Asked FRAME-INDEPENDENTLY: does the tick ADVANCE? --------------
    # Hour 00 server is XAUUSD's daily break, and `srv_now.hour >= DAY_END_HOUR` is FALSE at
    # hour 0, so without this the runner would try to place into a closed market every night.
    # Comparing a server-stamped tick to host utcnow() instead would read a fresh quote as three
    # hours old and make the gate a no-op -- that bug has shipped here twice.
    import time as _time
    _t0 = tick
    _time.sleep(6)
    tick = m.symbol_info_tick(sym) or _t0
    if int(tick.time) == int(_t0.time) and float(tick.bid) == float(_t0.bid):
        print(f"  {sym} quote static over 6s -> market closed; nothing to do")
        m.shutdown(); return 0

    srv_now = pd.to_datetime(int(tick.time), unit="s")
    D = daily_frame(m, sym)
    if D.empty:
        print("no D1 bars"); m.shutdown(); return 1
    done = D[D.index.normalize() < srv_now.normalize()]
    if len(done) < 20:
        print("not enough completed daily bars"); m.shutdown(); return 1
    pdh, pdl = float(done["high"].iloc[-1]), float(done["low"].iloc[-1])
    atr = wilder_atr(done)
    if not np.isfinite(atr) or atr <= 0:
        print("bad ATR"); m.shutdown(); return 1

    pos = [p for p in (m.positions_get(symbol=sym) or []) if p.magic == MAGIC]
    orders = [o for o in (m.orders_get(symbol=sym) or []) if o.magic == MAGIC]
    print(f"account {ai.login} @ {ai.server} (DEMO)  server {srv_now}  equity ${ai.equity:,.2f}")
    print(f"  {sym} bid {tick.bid:.2f} ask {tick.ask:.2f} | prior day H {pdh:.2f} L {pdl:.2f} | "
          f"daily ATR {atr:.2f}")
    print(f"  magic {MAGIC}: {len(pos)} position(s), {len(orders)} order(s)")

    fm = int(si.filling_mode)
    fill_type = (m.ORDER_FILLING_FOK if fm & 1 else
                 m.ORDER_FILLING_IOC if fm & 2 else m.ORDER_FILLING_RETURN)
    base = dict(utc=datetime.now(timezone.utc).isoformat(), symbol=sym, atr=round(atr, 2))
    st = load_state()
    day = str(srv_now.date())

    # ---- 1. POSITION OPEN: ratchet the stop, never loosen it; exit at max hold ---------------
    if pos:
        p = pos[0]
        opened = pd.to_datetime(int(p.time), unit="s")
        held_days = (srv_now.normalize() - opened.normalize()).days
        # high-water from the broker's own H1 bars since entry -- not from a state file, so a
        # restart or a missed run cannot lose it.
        r = m.copy_rates_from_pos(sym, 16385, 0, 24 * (MAX_HOLD_DAYS + 2))
        hwm = float(p.price_open)
        if r is not None and len(r):
            hb = pd.DataFrame([(x[0], x[2]) for x in r], columns=["ts", "high"])
            hb["t"] = pd.to_datetime(hb.ts, unit="s")
            after = hb[hb.t >= opened]
            if not after.empty:
                hwm = max(hwm, float(after["high"].max()))
        hwm = max(hwm, float(tick.bid))
        want = hwm - TRAIL_ATR * atr
        cur_sl = float(p.sl or 0.0)
        print(f"  POSITION {p.volume} @ {p.price_open:.2f} since {opened} ({held_days}d), "
              f"hwm {hwm:.2f}, sl {cur_sl:.2f}, trail wants {want:.2f}, pnl {p.profit:+.2f}")
        if held_days >= MAX_HOLD_DAYS:
            if a.execute:
                t2 = m.symbol_info_tick(sym)
                rr = m.order_send(dict(action=m.TRADE_ACTION_DEAL, symbol=sym, volume=p.volume,
                                       type=m.ORDER_TYPE_SELL, position=p.ticket, price=t2.bid,
                                       deviation=50, magic=MAGIC, comment="trail-maxhold",
                                       type_filling=fill_type))
                print(f"  MAX HOLD {MAX_HOLD_DAYS}d -> exit, retcode {rr.retcode}")
                record(**base, event="exit_maxhold", price=t2.bid, ticket=p.ticket,
                       retcode=rr.retcode, hwm=round(hwm, 2), volume=p.volume)
            else:
                print(f"  would EXIT on max hold; --execute to send")
        elif want > cur_sl + 1e-6:
            if a.execute:
                rr = m.order_send(dict(action=m.TRADE_ACTION_SLTP, symbol=sym,
                                       position=p.ticket, sl=round(want, si.digits), tp=0.0))
                print(f"  RATCHET sl {cur_sl:.2f} -> {want:.2f}  retcode {rr.retcode}")
                record(**base, event="ratchet", price=p.price_open, sl=round(want, 2),
                       ticket=p.ticket, retcode=rr.retcode, hwm=round(hwm, 2))
            else:
                print(f"  would RATCHET sl {cur_sl:.2f} -> {want:.2f}; --execute to send")
        else:
            print(f"  stop holds at {cur_sl:.2f} (ratchet never loosens)")
        m.shutdown(); return 0

    # ---- 2. PENDING ORDER: cancel if the prior-day LOW went first, or at day end -------------
    if orders:
        o = orders[0]
        todays_low = float(D["low"].iloc[-1]) if D.index[-1].normalize() == srv_now.normalize() \
            else float(tick.bid)
        if todays_low < pdl:
            if a.execute:
                rr = m.order_send(dict(action=m.TRADE_ACTION_REMOVE, order=o.ticket))
                print(f"  prior-day LOW {pdl:.2f} broke first (low {todays_low:.2f}) -> CANCEL "
                      f"#{o.ticket}, retcode {rr.retcode}")
                record(**base, event="cancel_low_first", level=o.price_open, ticket=o.ticket,
                       retcode=rr.retcode, note=f"today low {todays_low:.2f} < pdl {pdl:.2f}")
                save_state(dict(day=day, decided="cancelled_low_first"))
            else:
                print(f"  would CANCEL (prior-day low broke first); --execute to send")
        elif srv_now.hour >= DAY_END_HOUR:
            if a.execute:
                rr = m.order_send(dict(action=m.TRADE_ACTION_REMOVE, order=o.ticket))
                print(f"  day end -> CANCEL unfilled #{o.ticket}, retcode {rr.retcode}")
                record(**base, event="cancel_unfilled", level=o.price_open, ticket=o.ticket,
                       retcode=rr.retcode)
            else:
                print("  would CANCEL unfilled at day end; --execute to send")
        else:
            print(f"  RESTING buy-stop #{o.ticket} at {o.price_open:.2f} "
                  f"({100*(o.price_open/tick.ask-1):+.2f}% above ask), sl {o.sl:.2f}")
        m.shutdown(); return 0

    # ---- 3. NOTHING OUTSTANDING: one decision per server day ---------------------------------
    if st.get("day") == day:
        print(f"  today's decision already made: {st.get('decided')}")
        m.shutdown(); return 0
    if srv_now.hour >= DAY_END_HOUR:
        print("  past day end; no new order today"); m.shutdown(); return 0
    if tick.ask >= pdh:
        print(f"  ask {tick.ask:.2f} already above the level {pdh:.2f} — a stop here fills at "
              f"market, which is not the trade that was measured. Skipping today.")
        record(**base, event="skip", level=pdh, note="ask already above level")
        save_state(dict(day=day, decided="skipped_above_level"))
        m.shutdown(); return 0

    stop_px = pdh - INIT_STOP_ATR * atr
    risk_usd = ai.equity * a.risk_frac
    per_lot = INIT_STOP_ATR * atr * float(si.trade_contract_size)
    step = float(si.volume_step or 0.01)
    lots = max(float(si.volume_min), round((risk_usd / per_lot) / step) * step)
    lots = min(lots, float(si.volume_max))
    print(f"  PLACING buy-stop {lots:.2f} @ {pdh:.2f}, sl {stop_px:.2f} "
          f"({INIT_STOP_ATR}xATR = ${INIT_STOP_ATR*atr:.2f}), risking "
          f"${lots*per_lot:,.2f} = {100*lots*per_lot/ai.equity:.2f}% of equity")
    if not a.execute:
        print("  dry run — rerun with --execute to send")
        m.shutdown(); return 0
    rr = m.order_send(dict(action=m.TRADE_ACTION_PENDING, symbol=sym, volume=lots,
                           type=m.ORDER_TYPE_BUY_STOP, price=round(pdh, si.digits),
                           sl=round(stop_px, si.digits), tp=0.0,     # TP 0.0 = none, by design
                           magic=MAGIC, comment="v5-trail-runner",
                           type_time=m.ORDER_TIME_GTC, type_filling=fill_type))
    print(f"  retcode {rr.retcode} {getattr(rr,'comment','')} order {getattr(rr,'order',0)}")
    record(**base, event="place", level=round(pdh, 2), sl=round(stop_px, 2), volume=lots,
           ticket=getattr(rr, "order", ""), retcode=rr.retcode)
    if int(rr.retcode) == 10009:
        save_state(dict(day=day, decided="placed", ticket=int(getattr(rr, "order", 0)),
                        level=round(pdh, 2)))
    m.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
