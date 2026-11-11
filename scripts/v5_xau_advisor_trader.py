#!/usr/bin/env python3
"""Trade the XAUUSD advisor's call. One script, many arms — selected by CLI flags.

THE ADVISOR ITSELF STAYS ORDER-FREE. `scripts/v5_xau_advisor.py` computes and writes
`data/v5_runs/xau_advisor_state.json`; this separate process READS that file and trades. That
keeps `tests/test_v5_xau_advisor.py`'s import-closure assertion (no order path in the advisor)
true, and means a bug here can never corrupt the advisor's state machine.

THE RULES (as asked, 2026-11-04, re-confirmed 2026-11-10):
  * the advisor's call -> a wanted side. `--rule BIAS` trades the BUY/SELL bias the notification
    announces; `--rule STRONG` trades the 3-state UP/DOWN machine (NEUTRAL/UNKNOWN/frozen -> no
    new entries, existing positions keep trailing).
  * ONE FLIP ACTION PER ADVISOR EPISODE, keyed on the call's own `since` timestamp. On the first
    run that sees a new episode:
      - every OPPOSITE position is CLOSED if it is in profit (profit + swap > 0); if it is not,
        it is LEFT OPEN, its stop keeps trailing, and it is given a SERVER-SIDE +$RESCUE_USD
        take-profit so it closes the moment it comes back (user, 2026-11-10: "the one with loss,
        wait as soon as it hit +50, close"). Server-side, not polled, so it fires between runs
        and survives this job being down; the level is recomputed each pass because accrued swap
        moves it.
      - a position is opened on the wanted side, unless one is already held there.
    Later runs in the same episode trail only. Without this a 5-minute loop would re-buy after
    every stop-out on one reading, which is a martingale, and would re-evaluate "close if in
    profit" continuously instead of at the moment of the change.
  * at most ONE position per side, so at most two in total.
  * every position carries a server-side SL from the moment it is sent, and every SL ratchets
    (never loosens) at TRAIL_ATR x H4 ATR from the best price since entry.

TWO ARMS RUN ON FTMO (2026-11-10), differing in exactly one thing:
    --side-mode long_short   magic 360611   SELL call -> open a short   <- as the user specified
    --side-mode long_only    magic 360612   SELL call -> open nothing
Both close and keep identically, so any difference in the live record is attributable to the
short leg and to nothing else. V5_FINDINGS 3bz measured that leg at -0.29 to -0.52 Sharpe across
four thresholds; this is the live test of that claim on the user's own account.

*** NOT A VALIDATED EDGE. *** V5_FINDINGS 3de replayed THIS logic: the BIAS rule made -25.0R over
28 years and -15.1R in the 5-month out-of-sample window (shuffle p 0.955); STRONG's +108.3R is
in-sample, with 14 OOS trades at 0.0R. The advisor's own measurement is a first-touch PATH
frequency (UP bucket 0.603 vs base 0.510; DOWN 0.562 vs 0.490, 3ba), never run as P&L until 3de.
A hedge on a hedging account is net-flat exposure that pays two spreads and two swaps (3bh).
Demo only.

    python scripts/v5_xau_advisor_trader.py --port 18815 --symbol XAUUSDs          # dry run
    python scripts/v5_xau_advisor_trader.py --port 18814 --symbol XAUUSD \
        --magic 360611 --tag ftmo_ls --rule BIAS --side-mode long_short \
        --ref-symbol BTCUSD --risk-frac 0.0025 --execute
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
sys.path.insert(0, str(ROOT))

RUNS = ROOT / "data" / "v5_runs"
ADVISOR_STATE = RUNS / "xau_advisor_state.json"

MAGIC = 360602      # 360595 book - 360597 zz FP - 360598 FP book - 360599 probe - 360600 trail
                    # - 360601 zigzag Switch - 360608/9 zigzag FTMO B/C - 360610 zigzag DWX C
                    # - 360611 advisor FTMO long/short - 360612 advisor FTMO long-only
SL_ATR = 1.0        # initial stop = the advisor's own barrier (1.0 x H4 ATR20)
TRAIL_ATR = 1.0     # ratchet distance from the best price since entry
RISK_FRAC = 0.005   # per entry; Sharpe-invariant dial (risk_frac table). FTMO arms use 0.0025.
MAX_SPREAD = 0.80   # $ - skip ENTRIES above this; Switch ~$0.21, FTMO ~$0.35
MAX_STATE_AGE_MIN = 16       # advisor runs every 5 min; allow two missed slots
RESCUE_USD = 50.0   # a leg kept through a flip closes at +$this. 0 disables (old behaviour).
BUY, SELL = 0, 1    # MT5 POSITION_TYPE_BUY / _SELL
FIELDS = ["utc", "event", "side", "price", "sl", "volume", "ticket", "profit", "retcode",
          "advisor", "atr", "note"]

# Every other job that talks to the bridge, by its wrapper's flock file.
#
# WHY. A bridge is ONE MetaTrader5 session shared by every client through rpyc. Any client's
# shutdown() drops it for all, and this job used to run in the middle of the hourly zigzag run
# and the book's market check: zigzag got None for XAUUSDs and the book saw BTC 'static' on a
# 24/7 market, from 2026-11-04 12:50 onward. A skipped run costs nothing; a corrupted book or
# harness run costs an hour.
BRIDGE_PEERS = {
    18815: ("/tmp/xau-advisor.lock", "/tmp/xau-btc-switch.lock", "/tmp/zigzag-trail-switch.lock",
            "/tmp/zigzag-trail-watch.lock", "/tmp/trail-runner.lock", "/tmp/fill-probe.lock",
            "/tmp/advisor-notify.lock"),
    # NOT /tmp/xau-advisor.lock: the advisor computes on bridge 18815, so it is no contention
    # here, and waiting on it would only make this job skip 5-minute slots. Ordering is handled
    # by the timer offset (the advisor starts at :03/5:20 and writes its state within ~9s) and
    # by MAX_STATE_AGE_MIN, which refuses to trade a stale reading outright.
    18814: ("/tmp/zigzag-trail-ftmo.lock", "/tmp/zigzag-trail-ftmo-b.lock",
            "/tmp/zigzag-trail-ftmo-c.lock", "/tmp/trade-events-ftmo.lock",
            "/tmp/xau-btc-ftmo.lock", "/tmp/xau-champ-ftmo.lock", "/tmp/book-pnl.lock"),
}
# 24/7 reference used to tell "market closed" from "feed dead" on the SERVER clock, per bridge.
REF_SYMBOL = {18815: "BTCUSDs", 18814: "BTCUSD", 18816: "BTCUSD", 18817: ""}


@dataclass(frozen=True)
class Pos:
    ticket: int
    type: int           # BUY / SELL
    volume: float
    price_open: float
    sl: float
    pnl: float          # profit + swap


def side_name(side: int) -> str:
    return "BUY" if side == BUY else "SELL"


def want_side(advisor_name: str) -> int | None:
    return {"UP": BUY, "DOWN": SELL}.get(advisor_name)


def decide(advisor_name: str, episode: str | None, last_handled_episode: str | None,
           positions: list, side_mode: str = "long_short", halted: bool = False) -> dict:
    """Pure. Returns {"open": side|None, "close": [tickets], "hedge": [tickets], "why": str}.

    THE FLIP RULE FIRES ONCE PER EPISODE, not on every pass. `episode` is the advisor call's own
    `since` timestamp, so "close the opposite leg if it is in profit" is evaluated at the moment
    the direction changes -- which is what was asked -- and not re-evaluated every 5 minutes
    afterwards, which would silently turn it into a different rule for whichever arm happens to
    be flat on that side.
    """
    out = {"open": None, "close": [], "hedge": [], "why": ""}
    side = want_side(advisor_name)
    if side is None:
        out["why"] = "advisor %s: no new entries, trailing only" % advisor_name
        return out
    if episode is not None and episode == last_handled_episode:
        out["why"] = ("episode %s already handled (%s call unchanged); trailing only"
                      % (episode, side_name(side)))
        return out

    opposite = [p for p in positions if p.type != side]
    same = [p for p in positions if p.type == side]
    for p in opposite:
        (out["close"] if p.pnl > 0 else out["hedge"]).append(p.ticket)

    bits = ["close %d in profit" % len(out["close"]), "keep %d losing" % len(out["hedge"])]
    if same:
        out["why"] = "advisor %s: already holding that side; %s" % (advisor_name, ", ".join(bits))
        return out
    if side == SELL and side_mode == "long_only":
        out["why"] = ("advisor %s: long-only arm opens no short; %s"
                      % (advisor_name, ", ".join(bits)))
        return out
    if halted:
        out["why"] = "account guard tripped: no new entry; %s" % ", ".join(bits)
        return out
    out["open"] = side
    out["why"] = ("advisor %s: open %s; %s"
                  % (advisor_name, side_name(side), ", ".join(bits)))
    return out


def trail_target(side: int, best: float, atr: float, trail_atr: float = TRAIL_ATR) -> float:
    """Where the stop should be: TRAIL_ATR x ATR behind the best price since entry."""
    return best - trail_atr * atr if side == BUY else best + trail_atr * atr


def rescue_tp(side: int, price_open: float, volume: float, contract: float, swap: float,
              rescue_usd: float, digits: int) -> float:
    """Price at which this position's profit + swap equals +`rescue_usd`.

    profit(BUY) = (close - open) * volume * contract, and `swap` is already accrued, so the
    target solves (tp - open) * volume * contract + swap = rescue_usd. Swap keeps accruing
    (gold is -92 pts/lot/day long on FTMO), which is why the caller recomputes it every pass
    instead of setting it once.
    """
    need = (rescue_usd - swap) / (volume * contract)
    return round(price_open + need if side == BUY else price_open - need, digits)


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


def paths_for(tag: str) -> tuple:
    """(state, log). `--tag` gives a second arm its own files -- v5_trail_runner.py:148."""
    if not tag:
        return RUNS / "xau_advisor_trader_state.json", RUNS / "xau_advisor_trader_log.csv"
    return (RUNS / ("xau_advisor_trader_%s_state.json" % tag),
            RUNS / ("xau_advisor_trader_%s_log.csv" % tag))


# ------------------------------------------------------------------------------------- I/O ---

def record(log: Path, magic: int, label: str, **kw) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    new = not log.exists()
    with log.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: kw.get(k, "") for k in FIELDS})
    if kw.get("event") in ("open", "close", "hedge") and kw.get("retcode") in (10009, 10008):
        try:
            from scripts.v5_notify import notify          # noqa: PLC0415
            line = "%s %s XAU @ %s" % (kw["event"], kw.get("side", ""), kw.get("price", ""))
            body = "\n".join("  %s %s" % (k, kw[k]) for k in FIELDS if kw.get(k) not in (None, ""))
            notify("[advisor-trader] %s" % line,
                   body + ("\n\nmagic %s, %s" % (magic, label)), line)
        except Exception as exc:                          # noqa: BLE001
            print("  (notify failed: %s: %s)" % (type(exc).__name__, exc))


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def save_state(path: Path, d: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=2))
    tmp.replace(path)


def read_advisor(now: datetime, rule: str) -> tuple:
    """(state name, episode id, note). Anything stale/broken/frozen reads as UNKNOWN.

    `--rule BIAS` maps the BUY/SELL bias the notification announces onto the same UP/DOWN names
    (BUY -> "UP", SELL -> "DOWN") and uses the bias's own `since` as the episode, so each switch
    allows exactly one flip action. `--rule STRONG` uses the 3-state machine instead.
    """
    a = load_json(ADVISOR_STATE)
    st = a.get("state") or {}
    b = a.get("bias") or {}
    if rule == "STRONG":
        name, since = st.get("name") or "UNKNOWN", st.get("since")
    else:
        name = {"BUY": "UP", "SELL": "DOWN"}.get(b.get("side"), "UNKNOWN")
        since = b.get("since")
    try:
        age = (now - datetime.fromisoformat(a["computed_utc"])).total_seconds() / 60
    except Exception:
        return "UNKNOWN", None, "advisor state missing or unreadable"
    if age > MAX_STATE_AGE_MIN:
        return "UNKNOWN", None, "advisor state %.0f min old" % age
    if a.get("broken") or st.get("frozen") or a.get("health") not in ("ok", "thrashing"):
        return "UNKNOWN", None, "advisor unhealthy (%s, %s)" % (a.get("health"), a.get("broken"))
    note = ("rule %s: strong %s p=%s | bias %s raw p=%s | episode %s"
            % (rule, st.get("name"), a.get("p"), b.get("side"), b.get("p_raw"), since))
    return name, since, note


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


def busy_peer(port: int, own_lock: str = "") -> str | None:
    """Name of a bridge peer that is running RIGHT NOW, else None.

    Test-and-release only: HOLDING a peer's lock would make that peer skip its own run."""
    import fcntl
    for path in BRIDGE_PEERS.get(port, ()):
        if path == own_lock:
            continue
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


def guard_check(st: dict, balance: float, equity: float, now: datetime,
                reset_tz: str, daily_frac: float, halt_frac: float) -> tuple:
    """(new_state, halted, note). Loss guards only.

    DELIBERATELY NOT challenge_guards.decide(): that also carries the phase machinery, whose
    `phase_complete` latch is sticky and would retire the arm after its first +8% run (3bt).
    Only the two loss tests are wanted here.
    """
    from src.v5 import challenge_guards as G               # noqa: PLC0415
    G.set_reset_tz(reset_tz)
    g = st.get("guard")
    if not g:
        g = G.init_state(balance, equity, now)
    g = G.roll_day(g, balance, equity, now)
    if g.get("halted"):
        return g, True, "HALTED (overall loss guard latched)"
    if G.overall_halt_hit(g, equity, halt_frac):
        g = dict(g)
        g["halted"] = True
        return g, True, ("HALT: equity %.2f <= %.1f%% below initial %.2f"
                         % (equity, 100 * halt_frac, g["initial_balance"]))
    if g.get("day_locked"):
        return g, True, "day locked (daily loss guard tripped today)"
    if G.daily_guard_hit(g, equity, daily_frac):
        g = dict(g)
        g["day_locked"] = True
        return g, True, ("DAY LOCK: equity %.2f <= %.1f%% below today's anchor %.2f"
                         % (equity, 100 * daily_frac, g["day_anchor"]))
    return g, False, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18815)
    ap.add_argument("--symbol", default="XAUUSDs")
    ap.add_argument("--magic", type=int, default=MAGIC)
    ap.add_argument("--tag", default="", help="suffix for the state/log files; '' = the original")
    ap.add_argument("--rule", choices=("BIAS", "STRONG"), default="STRONG")
    ap.add_argument("--side-mode", choices=("long_short", "long_only"), default="long_short")
    ap.add_argument("--ref-symbol", default=None, help="24/7 symbol for the market-open test")
    ap.add_argument("--risk-frac", type=float, default=RISK_FRAC)
    ap.add_argument("--max-spread", type=float, default=MAX_SPREAD)
    ap.add_argument("--rescue-usd", type=float, default=RESCUE_USD,
                    help="close a leg kept through a flip once it reaches +$this; 0 = never")
    ap.add_argument("--label", default="", help="shown in notifications, e.g. 'FTMO demo L/S'")
    ap.add_argument("--own-lock", default="", help="this job's own flock path, so it is not a peer")
    ap.add_argument("--reset-tz", default="FTMO", help="daily anchor reset tz for the loss guard")
    ap.add_argument("--daily-guard-frac", type=float, default=0.035)
    ap.add_argument("--overall-halt-frac", type=float, default=0.08)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()

    magic, label = a.magic, (a.label or ("port %d" % a.port))
    STATE, LOG = paths_for(a.tag)
    ref_sym = a.ref_symbol if a.ref_symbol is not None else REF_SYMBOL.get(a.port, "")

    peer = busy_peer(a.port, a.own_lock)
    if peer:
        print("%s  bridge peer running (%s); skipping"
              % (datetime.now(timezone.utc).strftime("%H:%M:%SZ"), peer))
        return 0

    now = datetime.now(timezone.utc)
    adv_name, episode, adv_note = read_advisor(now, a.rule)
    print("%s  [%s magic %s %s/%s] %s"
          % (now.strftime("%Y-%m-%d %H:%M:%SZ"), label, magic, a.rule, a.side_mode, adv_note))

    from mt5linux import MetaTrader5
    m, last = None, None
    for i in range(3):                       # a shared bridge returns -6 on collision; transient
        if i:
            time.sleep(3 * i)
        m = MetaTrader5(host="localhost", port=a.port)
        if m.initialize():
            break
        last = m.last_error()
    else:
        print("bridge init failed after 3 tries:", last)
        return 2

    try:
        ai = m.account_info()
        if ai is None:
            print("NOT LOGGED IN")
            return 2
        if ai.trade_mode != 0:
            print("REFUSING: account %s trade_mode %s is not DEMO" % (ai.login, ai.trade_mode))
            return 1
        if int(ai.margin_mode) != 2:
            print("REFUSING: margin_mode %s is not HEDGING - the keep-the-loser rule needs two "
                  "opposite positions to coexist" % ai.margin_mode)
            return 1
        sym = a.symbol
        si = tick = None
        for i in range(3):                   # a busy terminal returns None, not an error
            if i:
                time.sleep(2 * i)
            m.symbol_select(sym, True)
            si, tick = m.symbol_info(sym), m.symbol_info_tick(sym)
            if si is not None and tick is not None and tick.ask:
                break
        if si is None or tick is None or not tick.ask:
            print("no quote for %s after 3 tries on bridge %s" % (sym, a.port))
            return 1
        if si.trade_mode != 4:
            print("REFUSING: %s trade_mode %s is not FULL" % (sym, si.trade_mode))
            return 1
        # MARKET OPEN, WITHOUT HOLDING THE BRIDGE. The old "sleep 6s and see if the tick moved"
        # check held a shared bridge long enough to starve the hourly jobs. Instead compare the
        # gold tick to a symbol that ticks 24/7 on the SAME server clock: a gold tick >10 min
        # behind it means gold is closed. NEVER use host UTC here (3dv).
        if ref_sym:
            ref = m.symbol_info_tick(ref_sym)
            if ref is not None and int(ref.time) - int(tick.time) > 600:
                print("  %s last tick %ss behind %s -> market closed; nothing to do"
                      % (sym, int(ref.time) - int(tick.time), ref_sym))
                return 0

        atr = h4_atr(m, sym)
        if not (atr == atr and atr > 0):
            print("bad H4 ATR")
            return 1
        raw = [p for p in (m.positions_get(symbol=sym) or []) if p.magic == magic]
        positions = [Pos(int(p.ticket), int(p.type), float(p.volume), float(p.price_open),
                         float(p.sl or 0.0), float(p.profit) + float(p.swap)) for p in raw]
        st = load_json(STATE)
        spread = float(tick.ask) - float(tick.bid)
        print("  account %s (DEMO, hedging) equity $%s | %s %.2f/%.2f spread $%.2f | H4 ATR20 %.2f"
              % (ai.login, format(ai.equity, ",.2f"), sym, tick.bid, tick.ask, spread, atr))
        for p in positions:
            print("  pos #%s %s %s @ %.2f sl %.2f pnl %+.2f"
                  % (p.ticket, side_name(p.type), p.volume, p.price_open, p.sl, p.pnl))

        guard, halted, gnote = guard_check(st, float(ai.balance), float(ai.equity), now,
                                           a.reset_tz, a.daily_guard_frac, a.overall_halt_frac)
        st["guard"] = guard
        if gnote:
            print("  GUARD: %s - no new entries (positions keep trailing, never force-flattened)"
                  % gnote)

        fm = int(si.filling_mode)
        fill = (m.ORDER_FILLING_FOK if fm & 1 else
                m.ORDER_FILLING_IOC if fm & 2 else m.ORDER_FILLING_RETURN)
        base = dict(utc=now.isoformat(timespec="seconds"), advisor=adv_name, atr=round(atr, 2))
        stop_gap = max(int(si.trade_stops_level), 0) * float(si.point)
        handled = st.get("last_handled_episode", st.get("last_entry_episode"))

        d = decide(adv_name, episode, handled, positions, a.side_mode, halted)
        print("  DECISION: %s" % d["why"])
        deferred = False
        if d["open"] is not None and spread > a.max_spread:
            print("  spread $%.2f > $%.2f: flip deferred to a later run"
                  % (spread, a.max_spread))
            d, deferred = {**d, "open": None, "close": [], "hedge": []}, True

        closed = set()
        # 1. close opposite positions that are in profit
        for p in raw:
            if int(p.ticket) not in d["close"]:
                continue
            nm = side_name(int(p.type))
            px = tick.bid if p.type == BUY else tick.ask
            if not a.execute:
                print("  would CLOSE #%s %s (in profit %+.2f)"
                      % (p.ticket, nm, p.profit + p.swap))
                continue
            rr = m.order_send(dict(action=m.TRADE_ACTION_DEAL, symbol=sym, volume=p.volume,
                                   type=m.ORDER_TYPE_SELL if p.type == BUY else m.ORDER_TYPE_BUY,
                                   position=p.ticket, price=px, deviation=50, magic=magic,
                                   comment="adv-flip-close", type_filling=fill))
            print("  CLOSE #%s %s retcode %s" % (p.ticket, nm, rr.retcode))
            record(LOG, magic, label, **base, event="close", side=nm, price=px, ticket=p.ticket,
                   profit=round(p.profit + p.swap, 2), retcode=rr.retcode, volume=p.volume)
            if int(rr.retcode) in (10009, 10008):
                closed.add(int(p.ticket))
        for t in d["hedge"]:
            tail = ("its stop keeps trailing and it closes at +$%.0f" % a.rescue_usd
                    if a.rescue_usd > 0 else "its stop keeps trailing")
            print("  KEEP #%s (not in profit); %s" % (t, tail))
            if a.execute:
                record(LOG, magic, label, **base, event="hedge", ticket=t,
                       note="losing leg kept open; %s" % tail)

        # 2. open the new direction, SL attached server-side
        opened = False
        if d["open"] is not None:
            buy = d["open"] == BUY
            px = float(tick.ask if buy else tick.bid)
            dist = max(SL_ATR * atr, stop_gap * 2)
            sl = round(px - dist if buy else px + dist, si.digits)
            lots = lots_for(ai.equity, a.risk_frac, dist, float(si.trade_contract_size),
                            float(si.volume_min), float(si.volume_max),
                            float(si.volume_step or 0.01))
            nm = side_name(d["open"])
            risk = lots * dist * float(si.trade_contract_size)
            print("  OPEN %s %.2f @ %.2f sl %.2f (%sxATR $%.2f), risk $%s = %.2f%% equity"
                  % (nm, lots, px, sl, SL_ATR, dist, format(risk, ",.2f"),
                     100 * risk / ai.equity))
            if a.execute:
                rr = m.order_send(dict(action=m.TRADE_ACTION_DEAL, symbol=sym, volume=lots,
                                       type=m.ORDER_TYPE_BUY if buy else m.ORDER_TYPE_SELL,
                                       price=px, sl=sl, tp=0.0, deviation=50, magic=magic,
                                       comment=("adv-%s" % adv_name)[:31],
                                       type_time=m.ORDER_TIME_GTC, type_filling=fill))
                print("  retcode %s %s" % (rr.retcode, getattr(rr, "comment", "")))
                record(LOG, magic, label, **base, event="open", side=nm, price=px, sl=sl,
                       volume=lots, ticket=getattr(rr, "order", ""), retcode=rr.retcode)
                opened = int(rr.retcode) in (10009, 10008)
            else:
                print("  dry run - rerun with --execute to send")

        # The episode is marked handled only when the flip actually completed. A deferred entry
        # (wide spread) or a rejected order must be retried on the next run, not swallowed.
        if a.execute and episode is not None and not deferred:
            if d["open"] is None or opened:
                st["last_handled_episode"] = episode
                st["last_entry_episode"] = episode        # back-compat for anything still reading it
                if opened:
                    st["last_entry_utc"] = now.isoformat(timespec="seconds")

        # 3. RECONCILE STOPS. One SLTP per position carrying BOTH the ratcheted stop and the
        # rescue target. Two separate calls would race: the trail writes back `tp` from the
        # pre-flip snapshot, so a TP set earlier in this same pass would be silently deleted
        # (the exact failure v5_zigzag_ftmo.py:414 warns about).
        live = {int(p.ticket): p for p in raw if int(p.ticket) not in closed}
        orphans = {k: v for k, v in (st.get("orphans") or {}).items() if int(k) in live}
        for t in d["hedge"]:
            orphans.setdefault(str(int(t)), dict(since=now.isoformat(timespec="seconds"),
                                                 episode=episode))
        st["orphans"] = orphans

        for tk, p in sorted(live.items()):
            nm = side_name(int(p.type))
            cur_sl, cur_tp = float(p.sl or 0.0), float(p.tp or 0.0)
            pnl = float(p.profit) + float(p.swap)
            kept = str(tk) in orphans and a.rescue_usd > 0

            # The target is already met and the TP did not fire (never set, or the level moved
            # under it as swap accrued). Take it at market rather than wait another pass.
            if kept and pnl >= a.rescue_usd:
                px = tick.bid if p.type == BUY else tick.ask
                if not a.execute:
                    print("  would CLOSE #%s %s at market: kept leg reached %+.2f >= +$%.0f"
                          % (tk, nm, pnl, a.rescue_usd))
                    continue
                rr = m.order_send(dict(action=m.TRADE_ACTION_DEAL, symbol=sym, volume=p.volume,
                                       type=m.ORDER_TYPE_SELL if p.type == BUY
                                       else m.ORDER_TYPE_BUY,
                                       position=tk, price=px, deviation=50, magic=magic,
                                       comment="adv-rescue", type_filling=fill))
                print("  RESCUE CLOSE #%s %s at %+.2f retcode %s" % (tk, nm, pnl, rr.retcode))
                record(LOG, magic, label, **base, event="close", side=nm, price=px, ticket=tk,
                       profit=round(pnl, 2), retcode=rr.retcode, volume=p.volume,
                       note="kept leg recovered to +$%.0f" % a.rescue_usd)
                if int(rr.retcode) in (10009, 10008):
                    orphans.pop(str(tk), None)
                continue

            want_sl = round(trail_target(int(p.type), best_since(m, sym, p, tick), atr),
                            si.digits)
            ok_side = (want_sl < float(tick.bid) - stop_gap if p.type == BUY
                       else want_sl > float(tick.ask) + stop_gap)
            new_sl = want_sl if (ok_side and tighter(int(p.type), want_sl, cur_sl)) else cur_sl

            new_tp = cur_tp
            if kept:
                new_tp = rescue_tp(int(p.type), float(p.price_open), float(p.volume),
                                   float(si.trade_contract_size), float(p.swap),
                                   a.rescue_usd, si.digits)
                floor_ = float(tick.ask) + stop_gap + float(si.point)
                ceil_ = float(tick.bid) - stop_gap - float(si.point)
                if p.type == BUY and new_tp < floor_:
                    new_tp = round(floor_, si.digits)
                elif p.type == SELL and new_tp > ceil_:
                    new_tp = round(ceil_, si.digits)

            moved_sl = abs(new_sl - cur_sl) > 1e-9
            moved_tp = abs(new_tp - cur_tp) > float(si.point) / 2
            if not (moved_sl or moved_tp):
                continue
            what = " ".join(([("sl %.2f -> %.2f" % (cur_sl, new_sl))] if moved_sl else [])
                            + ([("tp %.2f -> %.2f (+$%.0f)" % (cur_tp, new_tp, a.rescue_usd))]
                               if moved_tp else []))
            if not a.execute:
                print("  would SET #%s %s %s" % (tk, nm, what))
                continue
            rr = m.order_send(dict(action=m.TRADE_ACTION_SLTP, symbol=sym, position=tk,
                                   sl=new_sl, tp=new_tp))
            print("  SET #%s %s %s retcode %s" % (tk, nm, what, rr.retcode))
            record(LOG, magic, label, **base, event="trail", side=nm, sl=new_sl, ticket=tk,
                   retcode=rr.retcode, note=what)
        st["orphans"] = orphans

        if a.execute:
            st["last_run_utc"] = now.isoformat(timespec="seconds")
            st["arm"] = dict(magic=magic, rule=a.rule, side_mode=a.side_mode, port=a.port,
                             symbol=sym, risk_frac=a.risk_frac, label=label)
            save_state(STATE, st)
        return 0
    finally:
        # NO m.shutdown(): it tears down the terminal session for EVERY client on this bridge.
        # Closing our rpyc connection (by exiting) is enough.
        pass


if __name__ == "__main__":
    raise SystemExit(main())
