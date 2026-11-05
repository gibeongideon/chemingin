#!/usr/bin/env python3
"""Trade the XAUUSD advisor's UP / DOWN state. Switch DEMO, magic 360602.

THE ADVISOR ITSELF STAYS ORDER-FREE. `scripts/v5_xau_advisor.py` computes and writes
`data/v5_runs/xau_advisor_state.json`; this separate process READS that file and trades. That
keeps `tests/test_v5_xau_advisor.py`'s import-closure assertion (no order path in the advisor)
true, and means a bug here can never corrupt the advisor's state machine.

THE RULES (as asked, 2026-11-04):
  * advisor UP -> want a BUY, advisor DOWN -> want a SELL. NEUTRAL / UNKNOWN / frozen -> no new
    entries; existing positions keep their stops and keep trailing.
  * a position already open in the wanted direction -> do nothing.
  * the state flipped -> open the new direction. Each OPPOSITE position is CLOSED if it is in
    profit (profit + swap > 0); if it is not, it is LEFT OPEN as a hedge and its stop trails.
  * every position carries a server-side SL from the moment it is sent, and every SL ratchets
    (never loosens) at TRAIL_ATR x H4 ATR from the best price since entry.

ONE ENTRY PER ADVISOR EPISODE. The episode is identified by the state's `since` timestamp. If a
BUY is stopped out while the advisor still says UP, this does NOT re-enter until the advisor
leaves UP and comes back — otherwise a 5-minute loop re-buys after every stop-out on the same
signal, which is a martingale on one reading.

POSITION COUNT IS BOUNDED AT TWO. A losing BUY kept as a hedge + a new SELL; if the state flips
back to UP there is already a BUY (same direction) so nothing new opens. At most one per side.

*** NOT A VALIDATED EDGE. *** The advisor's measurement is a first-touch PATH frequency
(UP bucket 0.603 vs base 0.510; DOWN 0.562 vs 0.490, §3ba) — it has never been run as P&L, and
this repo has repeatedly found accuracy that does not survive cost (control #8: AUC is
payoff-blind; perfect calls one bar late are negative). A hedge on a hedging account is net-flat exposure that pays two
spreads and two swaps. Demo only, 0.5% risk per entry.

    python scripts/v5_xau_advisor_trader.py --port 18815 --symbol XAUUSDs          # dry run
    python scripts/v5_xau_advisor_trader.py --port 18815 --symbol XAUUSDs --execute
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADVISOR_STATE = ROOT / "data" / "v5_runs" / "xau_advisor_state.json"
STATE = ROOT / "data" / "v5_runs" / "xau_advisor_trader_state.json"
LOG = ROOT / "data" / "v5_runs" / "xau_advisor_trader_log.csv"

MAGIC = 360602      # 360595 book · 360597 zz FP · 360598 FP book · 360599 probe · 360600 trail
                    # · 360601 zigzag Switch
SL_ATR = 1.0        # initial stop = the advisor's own barrier (1.0 x H4 ATR20)
TRAIL_ATR = 1.0     # ratchet distance from the best price since entry
RISK_FRAC = 0.005   # per entry; Sharpe-invariant dial, 0.5% under a 10% wall (risk_frac table)
MAX_SPREAD = 0.80   # $ — skip ENTRIES above this; Switch quotes ~$0.28 in normal hours
MAX_STATE_AGE_MIN = 16       # advisor runs every 5 min; allow two missed slots
BUY, SELL = 0, 1    # MT5 POSITION_TYPE_BUY / _SELL
FIELDS = ["utc", "event", "side", "price", "sl", "volume", "ticket", "profit", "retcode",
          "advisor", "atr", "note"]


@dataclass(frozen=True)
class Pos:
    ticket: int
    type: int           # BUY / SELL
    volume: float
    price_open: float
    sl: float
    pnl: float          # profit + swap


def want_side(advisor_name: str) -> int | None:
    return {"UP": BUY, "DOWN": SELL}.get(advisor_name)


def decide(advisor_name: str, episode: str | None, last_entry_episode: str | None,
           positions: list[Pos]) -> dict:
    """Pure. Returns {"open": side|None, "close": [tickets], "hedge": [tickets], "why": str}."""
    side = want_side(advisor_name)
    out = {"open": None, "close": [], "hedge": [], "why": ""}
    if side is None:
        out["why"] = f"advisor {advisor_name}: no new entries, trailing only"
        return out
    if any(p.type == side for p in positions):
        out["why"] = f"already {'BUY' if side == BUY else 'SELL'} — same direction, no order"
        return out
    if episode is not None and episode == last_entry_episode:
        out["why"] = ("already entered for this advisor episode (stopped out since); "
                      "waiting for the next state change")
        return out
    for p in positions:                         # all opposite, since none match `side`
        (out["close"] if p.pnl > 0 else out["hedge"]).append(p.ticket)
    out["open"] = side
    out["why"] = (f"advisor {advisor_name}: open {'BUY' if side == BUY else 'SELL'}; "
                  f"close {len(out['close'])} in profit, keep {len(out['hedge'])} losing as hedge")
    return out


def trail_target(side: int, best: float, atr: float, trail_atr: float = TRAIL_ATR) -> float:
    """Where the stop should be: TRAIL_ATR x ATR behind the best price since entry."""
    return best - trail_atr * atr if side == BUY else best + trail_atr * atr


def tighter(side: int, new_sl: float, cur_sl: float) -> bool:
    """Ratchet only. An unset stop (0.0) is always tightened."""
    if cur_sl <= 0:
        return True
    return new_sl > cur_sl + 1e-6 if side == BUY else new_sl < cur_sl - 1e-6


def lots_for(equity: float, risk_frac: float, stop_dist: float, contract: float,
             vmin: float, vmax: float, step: float) -> float:
    raw = (equity * risk_frac) / (stop_dist * contract)
    lots = round(round(raw / step) * step, 2)
    return min(max(lots, vmin), vmax)


# ------------------------------------------------------------------------------------- I/O ---

def record(**kw) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: kw.get(k, "") for k in FIELDS})
    if kw.get("event") in ("open", "close", "hedge") and kw.get("retcode") in (10009, 10008):
        try:
            sys.path.insert(0, str(ROOT))
            from scripts.v5_notify import notify          # noqa: PLC0415
            line = f"{kw['event']} {kw.get('side', '')} XAU @ {kw.get('price', '')}"
            body = "\n".join(f"  {k} {kw[k]}" for k in FIELDS if kw.get(k) not in (None, ""))
            notify(f"[advisor-trader] {line}", body + f"\n\nmagic {MAGIC}, Switch demo", line)
        except Exception as exc:                          # noqa: BLE001
            print(f"  (notify failed: {type(exc).__name__}: {exc})")


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def save_state(d: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=2))
    tmp.replace(STATE)


def read_advisor(now: datetime) -> tuple[str, str | None, str]:
    """(state name, episode id, note). Anything stale/broken/frozen reads as UNKNOWN.

    TRADES THE BUY/SELL BIAS (user, 2026-11-05), not the 3-state UP/NEUTRAL/DOWN machine, which
    sat in NEUTRAL for 30h and so never placed an order. BUY -> "UP", SELL -> "DOWN"; the
    episode is the bias's own `since`, so each switch allows exactly one entry."""
    a = load_json(ADVISOR_STATE)
    st = a.get("state") or {}
    b = a.get("bias") or {}
    name = {"BUY": "UP", "SELL": "DOWN"}.get(b.get("side"), "UNKNOWN")
    since = b.get("since")
    try:
        age = (now - datetime.fromisoformat(a["computed_utc"])).total_seconds() / 60
    except Exception:
        return "UNKNOWN", None, "advisor state missing or unreadable"
    if age > MAX_STATE_AGE_MIN:
        return "UNKNOWN", None, f"advisor state {age:.0f} min old"
    if a.get("broken") or st.get("frozen") or a.get("health") not in ("ok", "thrashing"):
        return "UNKNOWN", None, f"advisor unhealthy ({a.get('health')}, {a.get('broken')})"
    return name, since, (f"bias {b.get('side')} raw p={b.get('p_raw')} since {since} "
                         f"[3-state {st.get('name')}]")


def h4_atr(m, sym: str) -> float:
    import pandas as pd                                    # noqa: PLC0415
    r = m.copy_rates_from_pos(sym, 16388, 1, 60)           # pos 1 = skip the forming bar
    if r is None or len(r) < 25:
        return float("nan")
    d = pd.DataFrame([(x[2], x[3], x[4]) for x in r], columns=["high", "low", "close"])
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()],
                   axis=1).max(axis=1)
    return float(tr.rolling(20).mean().iloc[-1])           # == v5_range_multitf.atr_series(h4,20)


def best_since(m, sym: str, p, tick) -> float:
    """Best price since entry from the broker's own M5 bars, so a restart can't lose it."""
    r = m.copy_rates_from_pos(sym, 5, 0, 2000)             # M5, ~7 trading days
    best = float(p.price_open)
    if r is not None and len(r):
        after = [x for x in r if int(x[0]) + 300 > int(p.time)]
        if after:
            best = (max(best, max(float(x[2]) for x in after)) if p.type == BUY
                    else min(best, min(float(x[3]) for x in after)))
    return max(best, float(tick.bid)) if p.type == BUY else min(best, float(tick.ask))


# Every other job that talks to bridge 18815, by its wrapper's flock file.
BRIDGE_PEERS = ("/tmp/xau-advisor.lock", "/tmp/xau-btc-switch.lock", "/tmp/zigzag-trail-switch.lock",
                "/tmp/zigzag-trail-watch.lock", "/tmp/trail-runner.lock", "/tmp/fill-probe.lock",
                "/tmp/advisor-notify.lock")


def busy_peer() -> str | None:
    """Name of a bridge peer that is running RIGHT NOW, else None.

    WHY. 18815 is ONE MetaTrader5 session shared by every client through rpyc. Any client's
    shutdown() drops it for all, and this every-minute job used to call shutdown() in the middle
    of the hourly zigzag run (:19:27, ~20s) and the book's market check (:50:27): zigzag got None
    for XAUUSDs and the book saw BTC 'static' on a 24/7 market, from 2026-11-04 12:50 onward.
    A minute skipped here costs nothing; a corrupted book or harness run costs an hour.
    Test-and-release only: HOLDING a peer's lock would make that peer skip its own run."""
    import fcntl
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18815)
    ap.add_argument("--symbol", default="XAUUSDs")
    ap.add_argument("--risk-frac", type=float, default=RISK_FRAC)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    peer = busy_peer()
    if peer:
        print(f"{datetime.now(timezone.utc):%H:%M:%SZ}  bridge peer running ({peer}); skipping")
        return 0

    now = datetime.now(timezone.utc)
    adv_name, episode, adv_note = read_advisor(now)
    print(f"{now:%Y-%m-%d %H:%M:%SZ}  {adv_note}")

    from mt5linux import MetaTrader5
    m, last = None, None
    for i in range(3):                       # 18815 is shared; -6 on collision is transient
        if i:
            time.sleep(3 * i)
        m = MetaTrader5(host="localhost", port=a.port)
        if m.initialize():
            break
        last = m.last_error()
    else:
        print("bridge init failed after 3 tries:", last); return 2

    try:
        ai = m.account_info()
        if ai is None:
            print("NOT LOGGED IN"); return 2
        if ai.trade_mode != 0:
            print(f"REFUSING: account {ai.login} trade_mode {ai.trade_mode} is not DEMO"); return 1
        if int(ai.margin_mode) != 2:
            print(f"REFUSING: margin_mode {ai.margin_mode} is not HEDGING — the keep-the-loser "
                  "rule needs two opposite positions to coexist"); return 1
        sym = a.symbol
        m.symbol_select(sym, True)
        si, t0 = m.symbol_info(sym), m.symbol_info_tick(sym)
        if si is None or t0 is None or not t0.ask:
            print(f"no quote for {sym}"); return 1
        if si.trade_mode != 4:
            print(f"REFUSING: {sym} trade_mode {si.trade_mode} is not FULL"); return 1
        # MARKET OPEN, WITHOUT HOLDING THE BRIDGE. 18815 is a single-client bridge shared by
        # ~8 timers; the old "sleep 6s and see if the tick moved" check held it long enough that
        # zigzag-trail-switch (:19:07) got None for XAUUSDs every hour from 11:19 to 18:19 on
        # 2026-11-04. Instead compare the gold tick to BTC's -- BTC ticks 24/7 on the SAME server
        # clock, so a gold tick >10 min behind BTC means gold is closed. Never use host UTC here.
        tick = t0
        ref = m.symbol_info_tick("BTCUSDs")
        if ref is not None and int(ref.time) - int(tick.time) > 600:
            print(f"  {sym} last tick {int(ref.time) - int(tick.time)}s behind BTCUSDs -> "
                  "market closed; nothing to do"); return 0

        atr = h4_atr(m, sym)
        if not (atr == atr and atr > 0):
            print("bad H4 ATR"); return 1
        raw = [p for p in (m.positions_get(symbol=sym) or []) if p.magic == MAGIC]
        positions = [Pos(int(p.ticket), int(p.type), float(p.volume), float(p.price_open),
                         float(p.sl or 0.0), float(p.profit) + float(p.swap)) for p in raw]
        st = load_json(STATE)
        spread = float(tick.ask) - float(tick.bid)
        print(f"  account {ai.login} (DEMO, hedging) equity ${ai.equity:,.2f} | {sym} "
              f"{tick.bid:.2f}/{tick.ask:.2f} spread ${spread:.2f} | H4 ATR20 {atr:.2f}")
        for p in positions:
            print(f"  pos #{p.ticket} {'BUY' if p.type == BUY else 'SELL'} {p.volume} "
                  f"@ {p.price_open:.2f} sl {p.sl:.2f} pnl {p.pnl:+.2f}")

        fm = int(si.filling_mode)
        fill = (m.ORDER_FILLING_FOK if fm & 1 else
                m.ORDER_FILLING_IOC if fm & 2 else m.ORDER_FILLING_RETURN)
        base = dict(utc=now.isoformat(timespec="seconds"), advisor=adv_name, atr=round(atr, 2))
        stop_gap = max(int(si.trade_stops_level), 0) * float(si.point)

        d = decide(adv_name, episode, st.get("last_entry_episode"), positions)
        print(f"  DECISION: {d['why']}")
        if d["open"] is not None and spread > MAX_SPREAD:
            print(f"  spread ${spread:.2f} > ${MAX_SPREAD:.2f}: entry deferred to a later run")
            d = {**d, "open": None, "close": [], "hedge": []}

        closed = set()
        # 1. close opposite positions that are in profit
        for p in raw:
            if int(p.ticket) not in d["close"]:
                continue
            side = "BUY" if p.type == BUY else "SELL"
            px = tick.bid if p.type == BUY else tick.ask
            if not a.execute:
                print(f"  would CLOSE #{p.ticket} {side} (in profit {p.profit + p.swap:+.2f})")
                continue
            rr = m.order_send(dict(action=m.TRADE_ACTION_DEAL, symbol=sym, volume=p.volume,
                                   type=m.ORDER_TYPE_SELL if p.type == BUY else m.ORDER_TYPE_BUY,
                                   position=p.ticket, price=px, deviation=50, magic=MAGIC,
                                   comment="adv-flip-close", type_filling=fill))
            print(f"  CLOSE #{p.ticket} {side} retcode {rr.retcode}")
            record(**base, event="close", side=side, price=px, ticket=p.ticket,
                   profit=round(p.profit + p.swap, 2), retcode=rr.retcode, volume=p.volume)
            if int(rr.retcode) in (10009, 10008):
                closed.add(int(p.ticket))
        for t in d["hedge"]:
            print(f"  KEEP #{t} (not in profit) as hedge; its stop keeps trailing")
            if a.execute:
                record(**base, event="hedge", ticket=t, note="losing leg kept open")

        # 2. open the new direction, SL attached server-side
        if d["open"] is not None:
            buy = d["open"] == BUY
            px = float(tick.ask if buy else tick.bid)
            dist = max(SL_ATR * atr, stop_gap * 2)
            sl = round(px - dist if buy else px + dist, si.digits)
            lots = lots_for(ai.equity, a.risk_frac, dist, float(si.trade_contract_size),
                            float(si.volume_min), float(si.volume_max),
                            float(si.volume_step or 0.01))
            side = "BUY" if buy else "SELL"
            risk = lots * dist * float(si.trade_contract_size)
            print(f"  OPEN {side} {lots:.2f} @ {px:.2f} sl {sl:.2f} ({SL_ATR}xATR ${dist:.2f}), "
                  f"risk ${risk:,.2f} = {100*risk/ai.equity:.2f}% equity")
            if a.execute:
                rr = m.order_send(dict(action=m.TRADE_ACTION_DEAL, symbol=sym, volume=lots,
                                       type=m.ORDER_TYPE_BUY if buy else m.ORDER_TYPE_SELL,
                                       price=px, sl=sl, tp=0.0, deviation=50, magic=MAGIC,
                                       comment=f"adv-{adv_name}", type_time=m.ORDER_TIME_GTC,
                                       type_filling=fill))
                print(f"  retcode {rr.retcode} {getattr(rr, 'comment', '')}")
                record(**base, event="open", side=side, price=px, sl=sl, volume=lots,
                       ticket=getattr(rr, "order", ""), retcode=rr.retcode)
                if int(rr.retcode) in (10009, 10008):
                    st["last_entry_episode"] = episode
                    st["last_entry_utc"] = now.isoformat(timespec="seconds")
            else:
                print("  dry run — rerun with --execute to send")

        # 3. ratchet every remaining stop
        for p in raw:
            if int(p.ticket) in closed:
                continue
            want = round(trail_target(int(p.type), best_since(m, sym, p, tick), atr),
                         si.digits)
            ok_side = (want < float(tick.bid) - stop_gap if p.type == BUY
                       else want > float(tick.ask) + stop_gap)
            cur = float(p.sl or 0.0)
            if not (ok_side and tighter(int(p.type), want, cur)):
                continue
            side = "BUY" if p.type == BUY else "SELL"
            if not a.execute:
                print(f"  would TRAIL #{p.ticket} {side} sl {cur:.2f} -> {want:.2f}")
                continue
            rr = m.order_send(dict(action=m.TRADE_ACTION_SLTP, symbol=sym, position=p.ticket,
                                   sl=want, tp=float(p.tp or 0.0)))
            print(f"  TRAIL #{p.ticket} {side} sl {cur:.2f} -> {want:.2f} retcode {rr.retcode}")
            record(**base, event="trail", side=side, sl=want, ticket=p.ticket,
                   retcode=rr.retcode, note=f"from {cur:.2f}")

        if a.execute:
            st["last_run_utc"] = now.isoformat(timespec="seconds")
            save_state(st)
        return 0
    finally:
        # NO m.shutdown(): it tears down the terminal session for EVERY client on 18815.
        # Closing our rpyc connection (by exiting) is enough.
        pass


if __name__ == "__main__":
    raise SystemExit(main())
