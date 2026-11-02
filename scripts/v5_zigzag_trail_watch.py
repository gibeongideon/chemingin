#!/usr/bin/env python3
"""Minute-by-minute TRAIL WATCHER for the ZigZag harness (360601 Switch, 360597 FundingPips).

WHY THIS EXISTS. v5_zigzag_ftmo.py ratchets its 3.0xATR trail, but only once an hour, because
the ratchet lives inside the pass that also retrains the pooled model (~13s of bridge-holding
work). Requested 2026-11-02: track an open ZigZag position every minute and move the stop as
soon as it is earned. This file is the ratchet alone — no model, no entries, no exits.

THE RULE IS THE HOURLY JOB'S RULE, EXACTLY:  SL = peak - trail_atr x ATR(14, closed H1 bars),
raised ONE-WAY only, server-side, via TRADE_ACTION_SLTP (which can never touch volume). The only
difference is resolution: the peak comes from M1 bars since entry plus the live bid, so the stop
catches up within a minute of a new high instead of up to an hour later. The two jobs never
fight: whichever sees the higher peak writes first, and the other finds nothing to raise.

PROFIT LOCK (user rule, 2026-11-02: "once it is +100 move SL to +50, then trail slowly — the
key is to secure quickly"). Config `lock_steps` is a list of [trigger, lock] pairs in
`lock_unit` ("usd" = account profit on the position, "price" = price distance from entry).
Once the PEAK since entry reaches a trigger, the stop is floored at entry + lock; the 3.0xATR
trail takes over whenever it rises above that floor. Both are one-way, so the floor never drops.
*** MEASURED COST, STATED SO IT IS NOT FORGOTTEN. *** §3co: an early lock is the breakeven
family — exact breakeven at +1.0 ATR cost 6.36bp/trade at t -3.55 (hit 0.5345 -> 0.4044),
because it turns dip-then-recover trades into small exits and those recoveries pay for the
losers. +$100 on 0.06 lots is ~0.9 H1 ATR, squarely in the costly zone. Deployed because the
user prefers securing profit over expectancy; this is a demo observation harness.

CHEAP WHEN FLAT. Positions under these magics are only ever opened by the hourly job, which
records `open_positions` in its state file after every pass. If that says 0 the watcher exits
without touching the bridge: 18815 is shared by seven timers and contention has already killed
a firing pass once (2026-11-02). A missing or unreadable state file means connect anyway.

    python scripts/v5_zigzag_trail_watch.py --config configs/v5_zigzag_trail_switch.json
    python scripts/v5_zigzag_trail_watch.py --config ... --execute     # actually raise stops
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TF_M1, TF_H1 = 1, 16385
# Don't resend a stop for a cent of improvement every minute — a broker sees each SLTP as a
# request. 0.1 ATR is ~$1.80 at today's H1 ATR, about 3% of the 54-point trail distance.
MIN_STEP_ATR = 0.10


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%FT%TZ")


def h1_atr(mt5, symbol: str, n_bars: int = 400) -> float:
    """Wilder ATR(14) of CLOSED H1 bars, dropped by the same rule as fetch_h1_closed in
    v5_zigzag_ftmo.py so both jobs compute the same trail distance. 400 bars is plenty: the
    EWM weight left on the first bar is (13/14)^400 ~ 1e-13."""
    r = mt5.copy_rates_from_pos(symbol, TF_H1, 0, n_bars)
    if r is None or len(r) < 50:
        raise SystemExit(f"{now_utc()} {symbol}: H1 history unavailable: {mt5.last_error()}")
    a = np.array([(x[0], x[2], x[3], x[4]) for x in r], dtype=float)
    df = pd.DataFrame(a[:, 1:], columns=["high", "low", "close"],
                      index=pd.to_datetime(a[:, 0], unit="s"))
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    df = df[~(df.index + pd.Timedelta(hours=1) > now)]
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])


def peak_since_entry(mt5, symbol: str, pos, bid: float) -> float:
    """Highest BID since the fill. M1 bars are bid-based, and a long closes at the bid.
    Only bars that START at or after the fill count — the entry minute's bar can hold a
    pre-entry high, which once produced phantom instant trail activation
    (src/v5/xau_m15_exec.py:227-229). Position and bar times are both on the SERVER clock;
    the +240 minutes of slack covers the server-vs-host offset when sizing the fetch."""
    age_min = max(0, int((time.time() - pos.time) / 60)) + 240
    r = mt5.copy_rates_from_pos(symbol, TF_M1, 0, min(age_min, 20_000))
    highs = [float(x[2]) for x in (r if r is not None else []) if int(x[0]) >= int(pos.time)]
    return max([float(pos.price_open), float(bid)] + highs)


def notify(subj: str, body: str) -> None:
    """Never fatal: the stop matters more than the message."""
    try:
        sys.path.insert(0, str(ROOT))
        from scripts.v5_notify import notify as _n                # noqa: PLC0415
        _n(subj, body, body)
    except Exception as exc:                                     # noqa: BLE001
        print(f"  (notify failed: {type(exc).__name__}: {exc})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="connect even if the hourly state file says flat")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    if cfg.get("exit_mode") != "atr_trail":
        print(f"{now_utc()} exit_mode={cfg.get('exit_mode')} — nothing to trail")
        return 0
    port, symbol, magic = cfg["port"], cfg["symbol"], cfg["magic"]
    trail_atr = float(cfg.get("trail_atr", 3.0))
    lock_steps = sorted(cfg.get("lock_steps", []))
    lock_unit = cfg.get("lock_unit", "usd")
    runs = ROOT / "data" / "v5_runs"
    hourly_state = runs / cfg["state_file"]
    my_state = runs / cfg["state_file"].replace("_state.json", "_watch.json")

    if not args.force:
        try:
            if json.loads(hourly_state.read_text()).get("open_positions", 1) == 0:
                return 0                                         # flat: stay off the bridge
        except Exception:
            pass

    from mt5linux import MetaTrader5
    mt5, last = None, None
    for i in range(3):
        if i:
            time.sleep(2 * i)
        mt5 = MetaTrader5(host="localhost", port=port)
        if mt5.initialize():
            break
        last = mt5.last_error()
    else:
        raise SystemExit(f"{now_utc()} bridge {port} init failed after 3 tries: {last}")

    try:
        held = [p for p in (mt5.positions_get(symbol=symbol) or []) if p.magic == magic]
        if not held:
            print(f"{now_utc()} magic {magic}: no open position")
            return 0
        ai = mt5.account_info()
        is_demo = ("demo" in str(getattr(ai, "server", "")).lower()
                   or getattr(ai, "trade_mode", 0) == 0)
        info = tick = None
        for i in range(3):
            if i:
                time.sleep(2 * i)
            mt5.symbol_select(symbol, True)
            info, tick = mt5.symbol_info(symbol), mt5.symbol_info_tick(symbol)
            if info is not None and tick is not None:
                break
        if info is None or tick is None:
            raise SystemExit(f"{now_utc()} {symbol}: symbol_info/tick None after 3 tries")
        bid = float(getattr(tick, "bid", 0.0) or info.bid)
        atr = h1_atr(mt5, symbol)
        min_gap = (info.trade_stops_level or 0) * info.point

        try:
            seen = json.loads(my_state.read_text())
        except Exception:
            seen = {}
        for p in held:
            key = str(p.ticket)
            mark = seen.setdefault(key, {})
            entry, cur_sl = float(p.price_open), float(p.sl)
            peak = peak_since_entry(mt5, symbol, p, bid)
            trail_sl = peak - trail_atr * atr
            # usd -> price: $ per 1.0 price move on this position = volume x contract size
            per_px = float(p.volume) * float(info.trade_contract_size) if lock_unit == "usd" else 1.0
            floor, locked = None, None
            for trig, lock in lock_steps:
                if (peak - entry) * per_px >= trig:
                    floor, locked = entry + lock / per_px, (trig, lock)
            new_sl = round(max(trail_sl, floor if floor is not None else trail_sl), info.digits)
            line = (f"{now_utc()} #{p.ticket} entry {entry:.2f} bid {bid:.2f} "
                    f"({(bid - entry) / atr:+.2f} ATR, {(bid - entry) * per_px:+.0f} {lock_unit}) "
                    f"peak {peak:.2f} ATR {atr:.2f} SL {cur_sl:.2f} -> {new_sl:.2f}"
                    + (f" [lock +{locked[1]} @ +{locked[0]}]" if locked else ""))
            # A newly-reached lock is sent at once; only the trail is held to the minimum step.
            new_lock = locked is not None and mark.get("lock") != list(locked) and new_sl > cur_sl
            if not new_lock and new_sl < cur_sl + MIN_STEP_ATR * atr:
                print(line + "  hold")
                continue
            if new_sl >= bid - min_gap:
                print(line + f"  SKIP: inside stops_level ({min_gap:.2f})")
                continue
            if not (args.execute and is_demo):
                print(line + ("  DRY" if is_demo else "  REFUSED: account not flagged demo"))
                continue
            r = mt5.order_send(dict(action=mt5.TRADE_ACTION_SLTP, symbol=symbol,
                                    position=p.ticket, sl=new_sl, tp=float(p.tp or 0.0)))
            rc = getattr(r, "retcode", r)
            print(line + f"  RAISED retcode={rc}")
            if rc not in (10009, 10008):
                continue
            # Alert on the moments worth a phone buzz, not on every minute's ratchet.
            if new_lock:
                mark["lock"] = list(locked)
                notify(f"[zigzag {magic}] profit locked +{locked[1]} {lock_unit} · {symbol}",
                       f"#{p.ticket} reached +{locked[0]} {lock_unit}; SL {cur_sl:.2f} -> "
                       f"{new_sl:.2f} (entry {entry:.2f}). Trail {trail_atr}xATR continues.")
            if not mark.get("first"):
                mark["first"] = True
                notify(f"[zigzag {magic}] trail engaged · {symbol}",
                       f"#{p.ticket} stop raised {cur_sl:.2f} -> {new_sl:.2f}\n"
                       f"entry {entry:.2f}, peak {peak:.2f}, trail {trail_atr}xATR {atr:.2f}")
            if new_sl >= entry and not mark.get("breakeven"):
                mark["breakeven"] = True
                notify(f"[zigzag {magic}] stop at/above entry · {symbol}",
                       f"#{p.ticket} SL {new_sl:.2f} >= entry {entry:.2f} — "
                       f"the trade can no longer lose (bar a gap)\npeak {peak:.2f}")
        live = {str(p.ticket) for p in held}
        my_state.write_text(json.dumps({k: v for k, v in seen.items() if k in live}, indent=2))
    finally:
        mt5.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
