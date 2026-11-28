#!/usr/bin/env python3
"""PDH DAY-TRADE EXECUTOR — the one intraday rule that passed its pre-registered bars (§3ex).

THE RULE, and the finding behind every line of it
  * ENTRY   one resting BUY STOP per server day at the PRIOR server day's high, i.e. the high of
            yesterday's D1 bar on the broker's own chart. The MT5 server day runs 17:00 -> 17:00
            New York, which is exactly the trading day §3ex tested (OANDA M1 2007-2026, net Sharpe
            +0.76, 17/20 years, placebo level p 0.000).
  * LONG ONLY, no take-profit, no trail, no breakeven move (RANK #3, §3bl, §3co).
  * STOP    1.0 x ATR14(D1) below the level, attached to the ORDER, so it is server-side from the
            instant of the fill and never depends on this process. §3ex part 3 (exploratory): flat
            from 0.5 to 1.5 ATR, i.e. free insurance against a dead bot.
  * EXIT    flat by 23:40 server = 16:40 New York, before the 17:00 rollover: no swap, never held
            overnight (RANK #1). The entry order EXPIRES server-side at the same minute, so it can
            never fill after the last exit attempt. Backtest: exit time is not a lever (+0.71 to
            +0.76 between 16:00 and the last bar).
  * SIZE    0.5% of account BALANCE risked on the 1-ATR stop. BALANCE, not equity: this account is
            hand-traded and floating manual P&L must not move the bot's lot size (memory
            manual-trading-trips-the-bot-guard).
  * SKIP    a day that OPENS at/above the level — a buy stop cannot rest below the market, and the
            same skip is in the trail-runner and the fill probe (in §3ex those 5.6% of days earned
            -8.1bp, but that was seen after the fact and is NOT why they are skipped); US holiday
            early-close days, when 23:40 would be after the market has shut.
  * LATE    if the level breaks in the first 10 minutes of the session, before the order could
            rest, buy at market and say so; later than that the day is recorded as MISSED, never
            chased (§3ex fill audit: every second of latency costs; an hour late nets +0.26).

THE ONE THING IT EXISTS TO MEASURE: entry slippage on a resting stop. §3ex's 5-second audit puts
a gap-aware fill at +0.96bp above the level (median 0.00); at +2bp the Sharpe falls to ~0.48.
Switch has filled 2/2 earlier buy-stops at this level and 42/42 stop-losses at exactly the level.

PRE-REGISTERED KILL RULES (both stop NEW entries only; an open position is still taken flat):
  1. mean entry slip of the last 40 resting-stop fills > 2.0bp;
  2. cumulative-R drawdown from its peak reaches -25R (backtest worst over 20 years: -16.1R).
  Clear a trip by deleting "halt" from the state file once the cause is understood.
Expectation (backtest of exactly this variant): +0.04R per trade, sd 0.57R, 51% wins, ~126 trades
a year; at 0.5% risk CAGR ~+2.4%, maxDD ~-8%. One year of live trades CANNOT distinguish that
from zero (SE ~0.05R), which is why the kill rules are about execution and tail, not P&L.

SAFETY, by construction:
  * refuses unless the account is DEMO (trade_mode 0) and HEDGING (margin_mode 2, so its position
    can never net against another bot's), and the symbol is FULL (trade_mode 4);
  * --execute is opt-in; a dry run writes NOTHING (no state, no log) — a dry run that writes live
    state once made a notifier announce an order that was never sent;
  * every time comparison is on the SERVER clock (tick time), never the host's;
  * never calls shutdown() on the shared bridge, and yields to peer bots' locks first;
  * one decision per server day, kept in a state file: a vanished order is recorded, never
    re-placed; a position that survives the cut-off is closed on the next run.

    python scripts/v5_pdh_executor.py                       # dry plan against the live account
    python scripts/v5_pdh_executor.py --execute             # trade the demo
    python scripts/v5_pdh_executor.py --report              # read the log: fills, slip, R, kill rules
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "v5_runs"

MAGIC = 360614
VENUE = "Switch"               # alert text only; main() sets it from --tag
STOP_ATR = 1.0
ATR_DAYS = 14
RISK_FRAC = 0.005
CUTOFF = (23, 40)              # server clock = 16:40 New York; exits AND order expiry
LATE_WINDOW_MIN = 10           # a break inside this many minutes of the open may be bought at market
MAX_NOTIONAL_FRAC = 0.50       # hard caps on top of the risk budget
HARD_MAX_LOTS = 1.0
KILL_SLIP_BP, KILL_SLIP_N, KILL_DD_R = 2.0, 40, 25.0
COMMENT = "v5-pdh-day"

# US holiday early closes / closures (SERVER dates: the server day ending 17:00 New York that
# date). On these days 23:40 server may be after the market shut, so no order is placed at all.
HOLIDAYS = frozenset({
    "2026-11-26", "2026-11-27",                              # Thanksgiving, Black Friday
    "2026-12-24", "2026-12-25", "2026-12-31", "2027-01-01",  # Christmas, New Year
    "2027-01-18", "2027-02-15", "2027-03-26",                # MLK, Presidents, Good Friday
    "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06",  # Memorial, Juneteenth obs., July 4 obs., Labor
    "2027-11-25", "2027-11-26",                              # Thanksgiving, Black Friday
    "2027-12-23", "2027-12-24", "2027-12-30", "2027-12-31",  # Christmas / New Year (both fall on weekends)
})
CALENDAR_END = "2027-12-31"   # after this the list is stale: every run warns, and one email is sent

# The SHORT jobs on Switch bridge 18815, waited for at most 15 s. Test-and-release only: holding a
# peer's lock would make that peer skip its own run (the trade-events watcher once skipped 2,031
# runs that way). The two hourly ZigZag jobs are deliberately NOT waited for: measured 2026-11-16
# they hold the bridge ~2 and ~5 minutes from :01 every hour, so waiting would only delay this run;
# overlapping them is the norm on this bridge and the 5-try connect absorbs a collision.
PEERS = ("/tmp/zigzag-trail-watch.lock", "/tmp/trade-events.lock", "/tmp/xau-advisor.lock",
         "/tmp/xau-btc-switch.lock", "/tmp/trail-runner.lock", "/tmp/fill-probe.lock", "/tmp/advisor-notify.lock")

FIELDS = ["utc", "srv", "event", "day", "entry", "level", "price", "slip_bp", "sl", "lots", "atr",
          "risk_usd", "pnl", "R", "spread", "ticket", "retcode", "note"]
CLOSE_EVENT = "closed"
ORDER_STATE = {1: "placed", 2: "cancelled", 4: "filled", 5: "rejected", 6: "expired"}
DONE = (10008, 10009)


# ============================================================ pure logic (unit-tested)
def daily_table(rows) -> pd.DataFrame:
    """Broker D1 rows (ts, open, high, low, close, ...) -> DataFrame indexed by SERVER date."""
    d = pd.DataFrame([(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in rows],
                     columns=["ts", "open", "high", "low", "close"])
    d.index = pd.DatetimeIndex(pd.to_datetime(d.ts, unit="s")).normalize()
    return d.drop(columns="ts").sort_index()


def levels(d1: pd.DataFrame, srv_today: pd.Timestamp) -> tuple[float, float, pd.Series | None]:
    """(prior-day high, ATR14, today's forming bar or None) from COMPLETED days only.

    ATR14 = the mean true range of the last 14 completed days — exactly §3ex's
    `TR.rolling(14).mean().shift(1)`. Never includes today's bar."""
    done = d1[d1.index < srv_today]
    if len(done) < ATR_DAYS + 1:
        raise ValueError(f"need {ATR_DAYS + 1} completed daily bars, have {len(done)}")
    pc = done.close.shift(1)
    tr = pd.concat([done.high - done.low, (done.high - pc).abs(), (done.low - pc).abs()], axis=1).max(axis=1)
    atr = float(tr.iloc[-ATR_DAYS:].mean())
    today = d1.loc[srv_today] if srv_today in d1.index else None
    return float(done.high.iloc[-1]), atr, today


def size_lots(balance: float, stop_dist: float, contract: float, step: float, vmin: float, vmax: float,
              price: float, risk_frac: float = RISK_FRAC) -> tuple[float, str]:
    """Lots risking `risk_frac` of balance on the stop distance, floored to the step, capped by a
    notional limit and a hard maximum. 0.0 (with a reason) if even the minimum lot would risk
    more than 1.5x the budget."""
    if stop_dist <= 0 or contract <= 0 or price <= 0:
        return 0.0, "bad inputs"
    budget = balance * risk_frac
    raw = budget / (stop_dist * contract)
    lots = math.floor(raw / step + 1e-9) * step
    cap = min(vmax, HARD_MAX_LOTS, math.floor(MAX_NOTIONAL_FRAC * balance / (price * contract) / step + 1e-9) * step)
    lots = min(lots, cap)
    if lots < vmin:
        if vmin * stop_dist * contract <= 1.5 * budget and vmin <= cap:
            return round(vmin, 2), "minimum lot"
        return 0.0, f"minimum lot {vmin} would risk {vmin * stop_dist * contract:,.0f} > 1.5 x budget {budget:,.0f}"
    return round(lots, 2), "risk budget"


def cutoff_ts(srv_now: pd.Timestamp) -> pd.Timestamp:
    return srv_now.normalize() + pd.Timedelta(hours=CUTOFF[0], minutes=CUTOFF[1])


def expiration_epoch(srv_now: pd.Timestamp) -> int:
    """Today's cut-off as an MT5 time value. MT5 times are SERVER wall-clock seconds encoded like
    epoch seconds (the same frame as tick.time), so this is NOT a host-UTC conversion."""
    return int((cutoff_ts(srv_now) - pd.Timestamp(0)).total_seconds())


def plan_entry(*, srv_now: pd.Timestamp, level: float, ask: float, spread: float, today_high_bid: float | None,
               open_ask: float | None, minutes_open: float | None, halted: str | None,
               weekdays_only: bool = False, ignore_holidays: bool = False) -> tuple[str, str]:
    """What to do when nothing is outstanding and today's decision has not been made.
    Returns (action, reason), action in place | market | skip | missed | halt.
    weekdays_only / ignore_holidays: for 24/7 instruments (BTC-PDH, data/v5_runs/btc_pdh/PREREG.md)."""
    if halted:
        return "halt", halted
    if weekdays_only and srv_now.dayofweek >= 5:
        return "skip", "weekend server day (--weekdays-only)"
    if not ignore_holidays and str(srv_now.date()) in HOLIDAYS:
        return "skip", "US holiday early close / closure"
    if srv_now >= cutoff_ts(srv_now):
        return "skip", "past the 23:40 server cut-off"
    if open_ask is not None and open_ask >= level:
        return "skip", f"opened at/above the level (open ask {open_ask:.2f} >= {level:.2f}); a buy stop cannot rest below the market"
    broke = ask >= level or (today_high_bid is not None and today_high_bid + spread >= level)
    if not broke:
        return "place", f"rest a buy stop at {level:.2f} ({level - ask:.2f} above ask)"
    if minutes_open is not None and minutes_open <= LATE_WINDOW_MIN:
        return "market", f"level broke {minutes_open:.0f} min after the open, before the order could rest"
    return "missed", "level already traded today while no order was resting; not chased"


def exit_action(srv_now: pd.Timestamp, opened_srv: pd.Timestamp) -> str | None:
    if opened_srv.normalize() < srv_now.normalize():
        return "stale"
    if srv_now >= cutoff_ts(srv_now):
        return "exit"
    return None


def kill_status(rows: list[dict]) -> tuple[str | None, dict]:
    """The two pre-registered kill rules, evaluated on the log. Returns (reason or None, stats)."""
    slips = [float(r["slip_bp"]) for r in rows
             if r.get("event") == "fill" and r.get("entry") == "stop" and r.get("slip_bp") not in ("", None)]
    Rs = [float(r["R"]) for r in rows if r.get("event") == CLOSE_EVENT and r.get("R") not in ("", None)]
    cum = np.concatenate([[0.0], np.cumsum(Rs)]) if Rs else np.zeros(1)
    dd = float((cum - np.maximum.accumulate(cum)).min())
    st = dict(fills=len(slips), slip_last=float(np.mean(slips[-KILL_SLIP_N:])) if slips else None,
              trades=len(Rs), cum_R=float(cum[-1]), dd_R=dd)
    if len(slips) >= KILL_SLIP_N and st["slip_last"] > KILL_SLIP_BP:
        return (f"KILL RULE 1: mean entry slip {st['slip_last']:+.2f}bp over the last {KILL_SLIP_N} "
                f"resting-stop fills > {KILL_SLIP_BP}bp"), st
    if dd <= -KILL_DD_R:
        return f"KILL RULE 2: cumulative-R drawdown {dd:.1f}R reached -{KILL_DD_R:.0f}R", st
    return None, st


def slip_bp(fill: float, level: float) -> float:
    return (fill - level) / level * 1e4


# ============================================================ files
class Files:
    def __init__(self, tag: str, base: Path | None = None):
        base = RUNS if base is None else Path(base)
        self.log = base / f"pdh_{tag}_log.csv"
        self.state = base / f"pdh_{tag}_state.json"

    def rows(self) -> list[dict]:
        if not self.log.exists():
            return []
        with self.log.open(newline="") as fh:
            return list(csv.DictReader(fh))

    def load(self) -> dict:
        try:
            return json.loads(self.state.read_text())
        except Exception:
            return {}

    def save(self, st: dict) -> None:
        self.state.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=2, default=str))
        tmp.replace(self.state)

    def append(self, row: dict) -> None:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        new = not self.log.exists()
        with self.log.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in FIELDS})


def alert(subject: str, body: str) -> None:
    """Problems only (vanished order, stale/failed exit, kill rule, stale calendar). Trades reach
    the phone through the trade-events watcher. Never fatal: the exit matters more than the mail."""
    try:
        sys.path.insert(0, str(ROOT))
        from scripts.v5_notify import notify  # noqa: PLC0415
        notify(f"[pdh] {subject}", body + f"\n\nmagic {MAGIC}, {VENUE} demo. scripts/v5_pdh_executor.py", subject)
    except Exception as exc:  # noqa: BLE001
        print(f"  (alert failed: {type(exc).__name__}: {exc})")


# ============================================================ report (no bridge)
def report(rows: list[dict]) -> str:
    L = [f"PDH day-trade executor — {len(rows)} log rows"]
    if not rows:
        return L[0] + "\n  (no log yet)"
    ev = pd.Series([r["event"] for r in rows]).value_counts()
    L.append("  events: " + ", ".join(f"{k} {v}" for k, v in ev.items()))
    reason, st = kill_status(rows)
    fills = [r for r in rows if r["event"] == "fill"]
    stop_f = [float(r["slip_bp"]) for r in fills if r.get("entry") == "stop" and r.get("slip_bp")]
    if stop_f:
        a = np.array(stop_f)
        L.append(f"  resting-stop fills {len(a)}: slip mean {a.mean():+.2f}bp, median {np.median(a):+.2f}, "
                 f"exactly at the level {100 * np.mean(np.abs(a) < 0.01):.0f}%, worst {a.max():+.2f}bp   "
                 f"(backtest 5s audit: mean +0.96, median 0.00; kill at > {KILL_SLIP_BP}bp over {KILL_SLIP_N})")
    late = [r for r in fills if r.get("entry") == "market"]
    if late:
        L.append(f"  late market entries {len(late)}: mean {np.mean([float(r['slip_bp']) for r in late]):+.2f}bp above the level")
    closed = [r for r in rows if r["event"] == CLOSE_EVENT and r.get("R")]
    if closed:
        R = np.array([float(r["R"]) for r in closed])
        pnl = np.array([float(r["pnl"] or 0) for r in closed])
        why = pd.Series([r.get("note", "").split(":")[0] for r in closed]).value_counts()
        L.append(f"  closed trades {len(R)}: mean {R.mean():+.3f}R (sd {R.std():.3f}), wins {100 * np.mean(R > 0):.0f}%, "
                 f"cum {R.sum():+.1f}R, worst drawdown {st['dd_R']:.1f}R, net ${pnl.sum():+,.2f}   "
                 f"(backtest: +0.039R, sd 0.574, 51% wins; kill at -{KILL_DD_R:.0f}R)")
        L.append("  exits: " + ", ".join(f"{k} {v}" for k, v in why.items()))
        if len(R) >= 20:
            L.append(f"  standard error of the mean R: {R.std(ddof=1) / np.sqrt(len(R)):.3f}  "
                     f"(the backtest's +0.039 is {0.039 / (R.std(ddof=1) / np.sqrt(len(R))):.1f} SE — not testable yet "
                     f"below ~800 trades)")
    L.append("  KILL RULES: " + (reason or "none tripped"))
    return "\n".join(L)


# ============================================================ bridge
def wait_for_peers(max_wait: float = 15.0) -> str | None:
    t0 = time.time()
    while True:
        busy = None
        for path in PEERS:
            try:
                with open(path, "a") as fh:
                    try:
                        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fcntl.flock(fh, fcntl.LOCK_UN)
                    except BlockingIOError:
                        busy = path
                        break
            except OSError:
                continue
        if busy is None or time.time() - t0 > max_wait:
            return busy
        time.sleep(1.5)


def connect(port: int, tries: int = 5):
    """5 tries over ~30 s: a terminal already serving another client answers (-6, 'Authorization
    failed'), indistinguishable from a real fault (observed on this bridge 2026-11-01). A run in
    the 23:44-23:54 exit window must not be lost to one collision."""
    from mt5linux import MetaTrader5  # noqa: PLC0415
    last = None
    for i in range(tries):
        if i:
            time.sleep(3 * i)
        m = MetaTrader5(host="localhost", port=port)
        if m.initialize():
            return m
        last = m.last_error()
    raise SystemExit(f"bridge init failed after {tries} tries: {last}")


def market_moving(m, sym: str, tick):
    """Is the market open? Asked frame-independently: does the tick ADVANCE within 30 s? The daily
    break (server 23:58 -> 01:01) and weekends answer no. A server-stamped tick compared with host
    UTC reads three hours old and makes this gate a no-op (shipped here twice)."""
    t0 = tick
    for _ in range(15):
        time.sleep(2)
        tick = m.symbol_info_tick(sym) or t0
        if int(tick.time_msc) != int(t0.time_msc) or float(tick.bid) != float(t0.bid):
            return tick, True
    return tick, False


def ts(sec) -> pd.Timestamp:
    return pd.to_datetime(int(sec), unit="s")


class Runner:
    def __init__(self, m, a, files: Files, st: dict, rows: list[dict]):
        self.m, self.a, self.f, self.st, self.rows = m, a, files, st, rows
        self.execute = a.execute
        self.sym = a.symbol

    # ---- recording
    def logged(self, event: str, ticket) -> bool:
        return any(r.get("event") == event and str(r.get("ticket")) == str(ticket) for r in self.rows)

    def record(self, **kw) -> None:
        kw.setdefault("utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        kw.setdefault("srv", str(self.srv_now))
        kw.setdefault("day", str(self.srv_now.date()))
        line = " ".join(f"{k}={kw[k]}" for k in ("event", "entry", "level", "price", "slip_bp", "sl", "lots",
                                                    "pnl", "R", "ticket", "retcode", "note") if kw.get(k) not in (None, ""))
        print(f"  >> {line}")
        if self.execute:
            self.f.append(kw)
            self.rows.append({k: str(v) for k, v in kw.items()})

    def send(self, req: dict):
        req = {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer,)) else v)
               for k, v in req.items()}                       # np.float64 must never cross rpyc
        if not self.execute:
            print(f"  (dry) would send {req}")
            return None
        r = self.m.order_send(req)
        if r is None:
            print(f"  order_send returned None: {self.m.last_error()}")
        return r

    def persist(self) -> None:
        """Write state the moment it changes, not only at the end of the run: a crash between
        sending an order and saving would otherwise forget the ticket it just created."""
        if self.execute:
            self.f.save(self.st)

    def fail_today(self, day: str, dec: dict, what: str) -> None:
        """A failed send is retried on the next run, at most 3 times a day, then the day is closed."""
        n = (self.st.get("fails") or {}).get(day, 0) + 1
        self.st["fails"] = {day: n}
        if n >= 3:
            self.st.update(dict(dec, decision=what, why=f"{what} 3 times"))

    # ---- one run
    def run(self) -> int:
        m, sym = self.m, self.sym
        ai = m.account_info()
        if ai is None:
            print("NOT LOGGED IN")
            return 2
        if int(ai.trade_mode) != 0:
            allow = getattr(self.a, "allow_real_login", None)
            if allow is None or int(ai.login) != int(allow):
                print(f"REFUSING: account {ai.login} trade_mode {ai.trade_mode} is not DEMO "
                      f"(a prop account must be named explicitly with --allow-real-login)")
                return 1
            print(f"  account {ai.login} trade_mode {ai.trade_mode}: allowed by --allow-real-login (prop challenge)")
        # Prop-rule bookkeeping (Goat Pay Later, 2026-11-27): track the starting balance and the highest
        # equity seen, every run, so decide() can stop entries at the target or before the trailing floor.
        if getattr(self.a, "halt_dd", None) or getattr(self.a, "target_frac", None):
            self.st.setdefault("initial", float(ai.balance))
            self.st["peak_equity"] = max(float(self.st.get("peak_equity") or 0.0), float(ai.equity), float(ai.balance))
        if int(ai.margin_mode) != 2:
            if not getattr(self.a, "allow_netting", False):
                print(f"REFUSING: account {ai.login} margin_mode {ai.margin_mode} is not HEDGING")
                return 1
            # NETTING (Goat Pay Later 315531834): one net position per symbol, so ANY other trade on this
            # symbol would merge into ours and our exit would close it too. Refuse to act if one exists.
            foreign = [p for p in (m.positions_get(symbol=sym) or []) if int(p.magic) != MAGIC]
            if foreign:
                print(f"REFUSING (netting): {len(foreign)} position(s) on {sym} not opened by magic {MAGIC} "
                      f"(magic {[int(p.magic) for p in foreign]}); on a netting account they would merge with ours")
                if self.execute and not self.st.get("netting_warned"):
                    alert("netting conflict", f"Foreign position on {sym}; the PDH bot is standing aside until it is gone.")
                    self.st["netting_warned"] = True
                return 1
            self.st.pop("netting_warned", None)
        m.symbol_select(sym, True)
        si, tick = m.symbol_info(sym), m.symbol_info_tick(sym)
        if si is None or tick is None or not tick.ask:
            print(f"no quote for {sym}")
            return 1
        if int(si.trade_mode) != 4:
            print(f"REFUSING: {sym} trade_mode {si.trade_mode} is not FULL")
            return 1
        tick, moving = market_moving(m, sym, tick)
        self.srv_now = ts(tick.time)
        if not moving:
            print(f"  {sym} quote static over 30s (server {self.srv_now}) -> market closed; nothing to do")
            return 0
        self.si, self.tick = si, tick
        self.fill_type = (m.ORDER_FILLING_FOK if int(si.filling_mode) & 1 else
                          m.ORDER_FILLING_IOC if int(si.filling_mode) & 2 else m.ORDER_FILLING_RETURN)
        srv_today = self.srv_now.normalize()
        day = str(srv_today.date())
        r = m.copy_rates_from_pos(sym, 16408, 0, 40)          # numpy array: never `r or []`
        d1 = daily_table([] if r is None else r)
        try:
            pdh, atr, today_bar = levels(d1, srv_today)
        except ValueError as e:
            print(f"  {e}")
            return 1
        spread = float(tick.ask - tick.bid)
        positions = [p for p in (m.positions_get(symbol=sym) or []) if int(p.magic) == MAGIC]
        orders = [o for o in (m.orders_get(symbol=sym) or []) if int(o.magic) == MAGIC]
        print(f"account {ai.login} @ {ai.server} ({'DEMO' if int(ai.trade_mode) == 0 else 'PROP'}, {'hedging' if int(ai.margin_mode) == 2 else 'NETTING'})  server {self.srv_now}  balance ${ai.balance:,.2f}")
        print(f"  {sym} bid {tick.bid:.2f} ask {tick.ask:.2f} spread {spread:.2f} | prior-day high {pdh:.2f} | "
              f"ATR14 {atr:.2f} | today high so far {float(today_bar.high) if today_bar is not None else float('nan'):.2f}")
        print(f"  magic {MAGIC}: {len(positions)} position(s), {len(orders)} order(s); "
              f"state day {self.st.get('day')} decision {self.st.get('decision')}")
        if day > CALENDAR_END:
            print(f"  !! holiday calendar ended {CALENDAR_END}; early closes are no longer skipped")
            if not self.st.get("calendar_warned") and self.execute:
                alert("holiday calendar expired", f"HOLIDAYS in scripts/v5_pdh_executor.py ends {CALENDAR_END}. Extend it.")
                self.st["calendar_warned"] = True

        self.reconcile({int(o.ticket) for o in orders} | {int(p.ticket) for p in positions})
        for p in positions:
            self.manage_position(p, atr)
        for o in orders:
            self.manage_order(o, day)
        if not positions and not orders and self.st.get("day") != day:
            self.decide(day, pdh, atr, today_bar, spread, ai)
        return 0

    # ---- history: what happened to tickets that are no longer live
    def reconcile(self, live: set[int]) -> None:
        tickets = self.st.setdefault("tickets", {})
        for t, info in list(tickets.items()):
            if info.get("done") or int(t) in live:
                continue
            ho = self.m.history_orders_get(ticket=int(t)) or []
            state = int(ho[0].state) if ho else None
            if state == 4:
                deals = sorted(self.m.history_deals_get(position=int(t)) or [], key=lambda d: d.time_msc)
                ins = [d for d in deals if int(d.entry) == 0]
                outs = [d for d in deals if int(d.entry) in (1, 2, 3)]
                if ins and not info.get("fill_logged"):
                    self.log_fill(t, info, float(ins[0].price), "seen in history")
                if outs:
                    self.log_close(t, info, deals, outs)
                    info["done"] = True
            elif state in (2, 5, 6):
                name = ORDER_STATE[state]
                if state == 2 and info.get("cancelled_by_us"):
                    pass
                elif state == 2:
                    self.record(event="vanished", day=info.get("day"), level=info.get("level"), ticket=t,
                                note="order removed from outside this bot (cancelled, not by us)")
                    if self.execute:
                        alert("order vanished", f"buy stop #{t} at {info.get('level')} was cancelled from outside the bot. "
                                                "Not re-placed (one decision per day).")
                else:
                    self.record(event=name, day=info.get("day"), level=info.get("level"), ticket=t,
                                note="unfilled" if state == 6 else "rejected by the server")
                info["done"] = True
            elif state is None and info.get("day", "") < str(self.srv_now.date()):
                info["missing_runs"] = info.get("missing_runs", 0) + 1
                if info["missing_runs"] >= 3:
                    self.record(event="unknown", day=info.get("day"), ticket=t, note="no history for this ticket")
                    info["done"] = True
        # keep the state small: drop finished tickets older than 10 days
        cutoff = str((self.srv_now - pd.Timedelta(days=10)).date())
        for t in [t for t, i in tickets.items() if i.get("done") and i.get("day", "") < cutoff]:
            del tickets[t]

    def log_fill(self, t, info: dict, price: float, how: str) -> None:
        lvl = float(info.get("level") or price)
        self.record(event="fill", day=info.get("day"), entry=info.get("entry", "stop"), level=round(lvl, 2),
                    price=round(price, 2), slip_bp=round(slip_bp(price, lvl), 3), sl=info.get("sl"),
                    lots=info.get("lots"), atr=info.get("atr"), risk_usd=info.get("risk_usd"), ticket=t, note=how)
        info["fill_logged"] = True
        info["fill_price"] = price

    def log_close(self, t, info: dict, deals, outs) -> None:
        net = sum(float(d.profit) + float(d.swap) + float(getattr(d, "commission", 0.0)) + float(getattr(d, "fee", 0.0))
                  for d in deals)
        last = outs[-1]
        cmt = str(getattr(last, "comment", "") or "")
        why = ("stop" if cmt.startswith("[sl") else "exit" if "pdh-exit" in cmt else
               "stale_exit" if "pdh-stale" in cmt else "external")
        risk = float(info.get("risk_usd") or 0) or None
        R = round(net / risk, 4) if risk else ""
        self.record(event=CLOSE_EVENT, day=info.get("day"), entry=info.get("entry", "stop"), level=info.get("level"),
                    price=round(float(last.price), 2), pnl=round(net, 2), R=R, risk_usd=info.get("risk_usd"),
                    ticket=t, note=f"{why}: closed {ts(last.time)} server ({cmt})")
        if why == "external" and self.execute:
            alert("position closed from outside", f"#{t} closed at {last.price} by someone else ({cmt}); net {net:+.2f}")

    # ---- an open position: record the fill once, take it flat at the cut-off or if stale
    def manage_position(self, p, atr: float) -> None:
        t = str(int(p.ticket))
        info = self.st.setdefault("tickets", {}).setdefault(
            t, dict(day=str(ts(p.time).date()), entry="untracked", level=float(p.price_open), sl=float(p.sl),
                    lots=float(p.volume), atr=round(atr, 2), risk_usd=round(float(p.volume) * self.si.trade_contract_size
                                                                          * STOP_ATR * atr, 2)))
        if not info.get("fill_logged"):
            self.log_fill(t, info, float(p.price_open), "first sight of the position")
        act = exit_action(self.srv_now, ts(p.time))
        print(f"  POSITION #{t} {p.volume} @ {p.price_open:.2f} sl {p.sl:.2f} pnl {p.profit:+.2f} "
              f"opened {ts(p.time)} -> {act or 'hold until 23:40 server'}")
        if not act:
            return
        for attempt in range(3):
            tk = self.m.symbol_info_tick(self.sym)
            r = self.send(dict(action=self.m.TRADE_ACTION_DEAL, symbol=self.sym, volume=float(p.volume),
                               type=self.m.ORDER_TYPE_SELL, position=int(p.ticket), price=float(tk.bid), deviation=50,
                               magic=MAGIC, comment="pdh-exit" if act == "exit" else "pdh-stale",
                               type_filling=self.fill_type))
            rc = int(getattr(r, "retcode", 0) or 0)
            if r is None and not self.execute:
                return
            if rc in DONE:
                self.record(event=act, level=info.get("level"), price=round(float(getattr(r, "price", tk.bid) or tk.bid), 2),
                            spread=round(float(tk.ask - tk.bid), 2), ticket=t, retcode=rc,
                            note="23:40 server cut-off" if act == "exit" else f"held past its day ({ts(p.time).date()})")
                if act == "stale":
                    alert("stale position closed", f"#{t} survived its trading day and was closed at {tk.bid}.")
                return
            print(f"  exit attempt {attempt + 1} failed: retcode {rc} {getattr(r, 'comment', '')}")
            time.sleep(2)
        self.record(event="exit_failed", ticket=t, retcode=rc, note="3 attempts; the next run retries")
        alert("EXIT FAILED", f"#{t} could not be closed (retcode {rc}). Next run retries; check the account.")

    # ---- a resting order: it should expire by itself at 23:40; cancel it if it has not
    def manage_order(self, o, day: str) -> None:
        t = str(int(o.ticket))
        info = self.st.setdefault("tickets", {}).get(t, {})
        stale = info.get("day") not in (None, day)
        late = self.srv_now >= cutoff_ts(self.srv_now)
        print(f"  RESTING buy stop #{t} at {o.price_open:.2f} sl {o.sl:.2f}, expires "
              f"{ts(o.time_expiration) if int(o.time_expiration) else 'never'} (server)")
        if not (stale or late):
            return
        r = self.send(dict(action=self.m.TRADE_ACTION_REMOVE, order=int(o.ticket)))
        rc = int(getattr(r, "retcode", 0) or 0)
        if rc in DONE:
            info["cancelled_by_us"] = True
            self.record(event="cancel", level=round(float(o.price_open), 2), ticket=t, retcode=rc,
                        note="order from an earlier day" if stale else "past the cut-off without expiring")

    # ---- today's decision
    def decide(self, day: str, pdh: float, atr: float, today_bar, spread: float, ai) -> None:
        m, sym, tick = self.m, self.sym, self.tick
        halted = (self.st.get("halt") or {}).get("reason")
        if not halted:
            init, peak, eq = self.st.get("initial"), self.st.get("peak_equity"), float(ai.equity)
            tf, hd = getattr(self.a, "target_frac", None), getattr(self.a, "halt_dd", None)
            prop = None
            if tf and init and eq >= init * (1 + tf):
                prop = f"PROFIT TARGET reached: equity {eq:,.2f} >= {init * (1 + tf):,.2f} (+{100 * tf:.1f}%)"
            elif hd and init and peak and eq <= peak - hd * init:
                prop = (f"TRAILING HALT: equity {eq:,.2f} <= peak {peak:,.2f} - {100 * hd:.1f}% of {init:,.0f} "
                        f"(firm floor sits lower; stopped before it)")
            if prop:
                halted = prop
                if self.execute:
                    self.st["halt"] = dict(reason=prop, since=str(self.srv_now))
                    alert("PROP RULE STOP — no new entries", prop)
        if not halted:
            reason, kst = kill_status(self.rows)
            if reason:
                halted = reason
                if self.execute:
                    self.st["halt"] = dict(reason=reason, since=str(self.srv_now), stats=kst)
                    alert("KILL RULE TRIPPED — no new entries", reason + "\n" + json.dumps(kst, indent=1))
        first_open, minutes_open = None, None
        m1 = m.copy_rates_from_pos(sym, 1, 0, 1500)
        today_m1 = [r for r in ([] if m1 is None else m1) if ts(r[0]).normalize() == self.srv_now.normalize()]
        if today_m1:
            first_open = float(today_m1[0][1])
            minutes_open = (self.srv_now - ts(today_m1[0][0])).total_seconds() / 60.0
        else:                                   # the session's first minute: no M1 bar has formed yet
            first_open, minutes_open = float(tick.bid), 0.0
        today_high = float(today_bar.high) if today_bar is not None else (
            max(float(r[2]) for r in today_m1) if today_m1 else None)
        action, why = plan_entry(srv_now=self.srv_now, level=pdh, ask=float(tick.ask), spread=spread,
                                 today_high_bid=today_high,
                                 open_ask=None if first_open is None else first_open + spread,
                                 minutes_open=minutes_open, halted=halted,
                                 weekdays_only=getattr(self.a, "weekdays_only", False),
                                 ignore_holidays=getattr(self.a, "ignore_holidays", False))
        print(f"  DECISION for server day {day}: {action} — {why}")
        dec = dict(day=day, decision=action, why=why, level=round(pdh, 2), atr=round(atr, 2), at=str(self.srv_now))
        if action in ("skip", "missed", "halt"):
            self.record(event=action, level=round(pdh, 2), atr=round(atr, 2), spread=round(spread, 2), note=why)
            if self.execute:
                self.st.update(dec)
                self.persist()
            return
        stop_dist = STOP_ATR * atr
        ref = pdh if action == "place" else float(tick.ask)
        lots, how = size_lots(float(ai.balance), stop_dist, float(self.si.trade_contract_size), float(self.si.volume_step),
                              float(self.si.volume_min), float(self.si.volume_max), ref)
        if lots <= 0:
            self.record(event="skip", level=round(pdh, 2), atr=round(atr, 2), note=f"size: {how}")
            if self.execute:
                self.st.update(dict(dec, decision="skip", why=how))
                self.persist()
            return
        risk_usd = round(lots * float(self.si.trade_contract_size) * stop_dist, 2)
        digits = int(self.si.digits)
        print(f"  size {lots:.2f} lots ({how}): risk ${risk_usd:,.2f} = {100 * risk_usd / float(ai.balance):.2f}% of balance, "
              f"notional ${lots * self.si.trade_contract_size * ref:,.0f}")
        if action == "place":
            req = dict(action=m.TRADE_ACTION_PENDING, symbol=sym, volume=lots, type=m.ORDER_TYPE_BUY_STOP,
                       price=round(pdh, digits), sl=round(pdh - stop_dist, digits), tp=0.0, deviation=0,
                       magic=MAGIC, comment=COMMENT, type_time=m.ORDER_TIME_SPECIFIED,
                       expiration=expiration_epoch(self.srv_now), type_filling=self.fill_type)
            r = self.send(req)
            rc = int(getattr(r, "retcode", 0) or 0)
            if r is not None and rc == 10022:                  # expiration refused: fall back to good-for-day
                req.update(type_time=m.ORDER_TIME_DAY)
                req.pop("expiration", None)
                r = self.send(req)
                rc = int(getattr(r, "retcode", 0) or 0)
            if r is not None and rc in (10015, 10016):          # price crossed the level while sending
                t2 = m.symbol_info_tick(sym)
                if float(t2.ask) >= pdh:
                    print("  the level was crossed while the order was being placed -> treat as the break")
                    action, why = "market", "crossed the level while the stop was being placed"
                    ref = float(t2.ask)
            if action == "place" and r is None and not self.execute:
                print("  (dry) the buy stop above would rest until it fills or expires at the cut-off")
                return
            if action == "place":
                ok = rc in DONE
                ticket = int(getattr(r, "order", 0) or 0) if r is not None else 0
                self.record(event="place" if ok else "place_failed", entry="stop", level=round(pdh, 2),
                            sl=round(pdh - stop_dist, 2), lots=lots, atr=round(atr, 2), risk_usd=risk_usd,
                            spread=round(spread, 2), ticket=ticket or "", retcode=rc or "",
                            note=f"expires {cutoff_ts(self.srv_now)} server; {why}" if ok else str(getattr(r, "comment", "")))
                if self.execute and ok:
                    self.st["tickets"][str(ticket)] = dict(day=day, entry="stop", level=round(pdh, 2),
                                                           sl=round(pdh - stop_dist, 2), lots=lots, atr=round(atr, 2),
                                                           risk_usd=risk_usd)
                    self.st.update(dict(dec, decision="placed"))
                    self.persist()
                    o = (m.orders_get(ticket=ticket) or [None])[0]
                    if o is not None:
                        print(f"  read back: #{ticket} {o.price_open:.2f} sl {o.sl:.2f} expires {ts(o.time_expiration)} server")
                elif self.execute:
                    self.fail_today(day, dec, "place_failed")
                    self.persist()
                return
        # market entry (late break)
        t2 = m.symbol_info_tick(sym)
        ask = float(t2.ask)
        r = self.send(dict(action=m.TRADE_ACTION_DEAL, symbol=sym, volume=lots, type=m.ORDER_TYPE_BUY, price=ask,
                           sl=round(ask - stop_dist, digits), tp=0.0, deviation=30, magic=MAGIC, comment="pdh-late",
                           type_filling=self.fill_type))
        rc = int(getattr(r, "retcode", 0) or 0)
        if r is None and not self.execute:
            print("  (dry) the market buy above would be sent")
            return
        ok = rc in DONE
        ticket = int(getattr(r, "order", 0) or 0) if r is not None else 0
        self.record(event="enter_market" if ok else "enter_failed", entry="market", level=round(pdh, 2),
                    price=round(float(getattr(r, "price", ask) or ask), 2) if r is not None else round(ask, 2),
                    sl=round(ask - stop_dist, 2), lots=lots, atr=round(atr, 2), risk_usd=risk_usd,
                    spread=round(float(t2.ask - t2.bid), 2), ticket=ticket or "", retcode=rc or "", note=why)
        if self.execute and ok:
            self.st["tickets"][str(ticket)] = dict(day=day, entry="market", level=round(pdh, 2),
                                                   sl=round(ask - stop_dist, 2), lots=lots, atr=round(atr, 2),
                                                   risk_usd=risk_usd)
            self.st.update(dict(dec, decision="entered_market"))
        elif self.execute:
            self.fail_today(day, dec, "enter_failed")
        self.persist()


def main() -> int:
    global MAGIC, VENUE
    ap =argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=18815)
    ap.add_argument("--symbol", default="XAUUSDs")
    ap.add_argument("--magic", type=int, default=MAGIC)
    ap.add_argument("--tag", default="switch", help="state/log file suffix")
    ap.add_argument("--execute", action="store_true", help="send orders and write state/log")
    ap.add_argument("--allow-real-login", type=int, default=None,
                    help="permit THIS login although the broker flags it real (prop challenge accounts)")
    ap.add_argument("--weekdays-only", action="store_true", help="no new entry on Saturday/Sunday server days (24/7 symbols)")
    ap.add_argument("--ignore-holidays", action="store_true", help="do not skip US holiday early closes (crypto)")
    ap.add_argument("--allow-netting", action="store_true",
                    help="permit a NETTING account (refuses to act while any foreign position is on the symbol)")
    ap.add_argument("--halt-dd", type=float, default=None,
                    help="stop new entries once equity <= peak equity - X x initial (trailing max-loss rules)")
    ap.add_argument("--target-frac", type=float, default=None,
                    help="stop new entries once equity >= initial x (1 + X) (challenge profit target)")
    ap.add_argument("--report", action="store_true", help="summarise the log; no bridge")
    a = ap.parse_args()
    MAGIC = a.magic
    VENUE = a.tag.upper() if a.tag != "switch" else "Switch"
    files = Files(a.tag)
    if a.report:
        print(report(files.rows()))
        return 0
    busy = wait_for_peers()
    if busy:
        print(f"  peer {busy} still busy after 15s; proceeding (the connect retries cover a collision)")
    st, rows = files.load(), files.rows()
    m = connect(a.port)                     # NEVER m.shutdown(): bridge 18815 is shared by every Switch job
    runner = Runner(m, a, files, st, rows)
    rc = 1
    try:
        rc = runner.run()
    finally:
        if a.execute:
            files.save(runner.st)          # also on an exception: a half-updated state beats a lost ticket
        else:
            print("  (dry run: nothing sent, no state or log written)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
