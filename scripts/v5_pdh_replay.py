#!/usr/bin/env python3
"""REPLAY the PDH executor's real code path through a simulated MT5 broker on historical M1 bars,
and compare what it traded with the §3ex backtest of the same rule on the same bars.

Why: the unit tests check each function; this checks the WHOLE live path — the server-day logic,
the 23:40 cut-off and order expiry, the open burst, reconciliation from history, P&L and R — on
real data, against the known answer. A live bot whose trades differ from its own backtest is
measuring something else.

FakeMT5 is also what tests/test_v5_pdh_executor.py drives. It is deliberately simple:
  * bars are BID M1 bars on the SERVER clock; ask = bid + spread;
  * a buy stop triggers when a bar's ask-high reaches it and fills at max(level, ask-open);
  * a stop-loss triggers when the bid-low reaches it, at min(sl, open) — or at sl in the entry bar;
  * orders expire at their expiration time before the bar is examined;
  * the tick is the current bar's open; outside a bar (break, weekend) the tick does not move.

    python scripts/v5_pdh_replay.py 2025-05-01 2025-11-01        # OANDA M1, spread 0, vs backtest
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.v5_pdh_executor as X  # noqa: E402


class FakeMT5:
    TRADE_ACTION_DEAL, TRADE_ACTION_PENDING, TRADE_ACTION_SLTP, TRADE_ACTION_REMOVE = 1, 5, 6, 8
    ORDER_TYPE_BUY, ORDER_TYPE_SELL, ORDER_TYPE_BUY_STOP = 0, 1, 4
    ORDER_TIME_GTC, ORDER_TIME_DAY, ORDER_TIME_SPECIFIED = 0, 1, 2
    ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2

    def __init__(self, bars: pd.DataFrame, spread: float = 0.0, balance: float = 100_000.0,
                 contract: float = 100.0, symbol: str = "XAUUSDs"):
        idx = pd.DatetimeIndex(bars.index)
        self.t = ((idx - pd.Timestamp(0)) // pd.Timedelta(seconds=1)).to_numpy(np.int64)
        self.o, self.h, self.l, self.c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        self.dn = self.t // 86400
        self.spread, self.balance, self.contract, self.sym = spread, balance, contract, symbol
        self.k, self.now, self.msc, self.ticket = 0, int(self.t[0]), 0, 500_000
        self.orders, self.positions, self.hist, self.deals = {}, {}, {}, []
        self.sent = []

    # ---- simulation clock
    def set_time(self, now: int) -> None:
        while self.k < len(self.t) and self.t[self.k] + 60 <= now:
            self._bar(self.k)
            self.k += 1
        self.now = int(now)

    def _bar(self, j: int) -> None:
        tj = int(self.t[j])
        for o in list(self.orders.values()):
            if o.time_expiration and tj >= o.time_expiration:
                self._finish(o, 6)
        for o in list(self.orders.values()):
            if self.h[j] + self.spread >= o.price_open:
                px = max(o.price_open, self.o[j] + self.spread)
                self._open(o.ticket, o.magic, o.volume_initial, px, o.sl, tj, o.comment)
                self._finish(o, 4)
        for p in list(self.positions.values()):
            if p.sl and self.l[j] <= p.sl:
                px = p.sl if p.time == tj else min(p.sl, self.o[j])
                self._close(p, px, tj, 0, f"[sl {p.sl:.2f}]")

    def _current(self) -> int | None:
        j = self.k
        return j if j < len(self.t) and self.t[j] <= self.now < self.t[j] + 60 else None

    # ---- book keeping
    def _new_ticket(self) -> int:
        self.ticket += 1
        return self.ticket

    def _finish(self, o, state: int) -> None:
        self.orders.pop(o.ticket, None)
        self.hist[o.ticket] = NS(ticket=o.ticket, state=state)

    def _deal(self, pid, entry, typ, px, vol, profit, t, magic, comment) -> None:
        self.msc += 1
        self.deals.append(NS(ticket=self._new_ticket(), order=pid, position_id=pid, entry=entry, type=typ,
                             price=float(px), volume=float(vol), profit=float(profit), swap=0.0, commission=0.0,
                             fee=0.0, time=int(t), time_msc=int(t) * 1000 + self.msc, magic=magic, comment=comment))

    def _open(self, ticket, magic, vol, px, sl, t, comment) -> None:
        self.positions[ticket] = NS(ticket=ticket, magic=magic, symbol=self.sym, type=0, volume=float(vol),
                                    price_open=float(px), sl=float(sl), tp=0.0, time=int(t), profit=0.0, swap=0.0,
                                    comment=comment)
        self._deal(ticket, 0, 0, px, vol, 0.0, t, magic, comment)

    def _close(self, p, px, t, magic, comment) -> None:
        pnl = (px - p.price_open) * p.volume * self.contract
        self.balance += pnl
        self.positions.pop(p.ticket, None)
        self._deal(p.ticket, 1, 1, px, p.volume, pnl, t, magic, comment)

    # ---- the MetaTrader5 surface the executor uses
    def initialize(self):
        return True

    def last_error(self):
        return (1, "Success")

    def symbol_select(self, sym, enable=True):
        return True

    def account_info(self):
        return NS(login=1, server="FAKE", trade_mode=0, margin_mode=2, leverage=100, balance=self.balance,
                  equity=self.balance, currency="USD")

    def symbol_info(self, sym):
        return NS(trade_mode=4, filling_mode=2, volume_min=0.01, volume_step=0.01, volume_max=35.0,
                  trade_contract_size=self.contract, digits=2, point=0.01, trade_stops_level=0)

    def symbol_info_tick(self, sym):
        j = self._current()
        if j is None:
            last = max(self.k - 1, 0)
            bid = float(self.c[last])
            return NS(time=int(self.t[last]) + 59, time_msc=(int(self.t[last]) + 59) * 1000, bid=bid, ask=bid + self.spread)
        self.msc += 1
        bid = float(self.o[j])
        return NS(time=self.now, time_msc=self.now * 1000 + self.msc % 1000, bid=bid, ask=bid + self.spread)

    def copy_rates_from_pos(self, sym, tf, start, n):
        j = self._current()
        if tf == 1:
            lo = max(0, self.k - n + (1 if j is not None else 0))
            rows = [(int(self.t[i]), self.o[i], self.h[i], self.l[i], self.c[i], 0, 0, 0) for i in range(lo, self.k)]
            if j is not None:
                rows.append((int(self.t[j]), self.o[j], self.o[j], self.o[j], self.o[j], 0, 0, 0))
            return np.array(rows) if rows else None
        if tf == 16408:
            end = self.k + (1 if j is not None else 0)
            first = np.searchsorted(self.dn[:end], self.dn[end - 1] - 3 * n, side="left")
            dn, h, l = self.dn[first:end], self.h[first:end].copy(), self.l[first:end].copy()
            c = self.c[first:end].copy()
            if j is not None:                      # the forming bar only knows its open so far
                h[-1] = l[-1] = c[-1] = self.o[j]
            u, s0 = np.unique(dn, return_index=True)
            s1 = np.append(s0[1:], len(dn))
            rows = [(int(d) * 86400, self.o[first + a], h[a:b].max(), l[a:b].min(), c[b - 1], 0, 0, 0)
                    for d, a, b in zip(u, s0, s1)]
            return np.array(rows[-n:])
        raise ValueError(tf)

    def positions_get(self, symbol=None):
        for p in self.positions.values():
            j = self._current()
            bid = self.o[j] if j is not None else self.c[max(self.k - 1, 0)]
            p.profit = (bid - p.price_open) * p.volume * self.contract
        return tuple(p for p in self.positions.values() if symbol in (None, p.symbol))

    def orders_get(self, symbol=None, ticket=None):
        return tuple(o for o in self.orders.values() if symbol in (None, o.symbol) and ticket in (None, o.ticket))

    def history_orders_get(self, ticket=None):
        if ticket in self.orders:
            return (NS(ticket=ticket, state=1),)
        return (self.hist[ticket],) if ticket in self.hist else ()

    def history_deals_get(self, position=None):
        return tuple(d for d in self.deals if d.position_id == position)

    def order_send(self, req: dict):
        self.sent.append(dict(req))
        tk = self.symbol_info_tick(self.sym)
        a = req["action"]
        if a == self.TRADE_ACTION_PENDING:
            if req["price"] <= tk.ask:
                return NS(retcode=10015, order=0, price=0.0, comment="Invalid price")
            t = self._new_ticket()
            exp = int(req.get("expiration", 0)) if req.get("type_time") == self.ORDER_TIME_SPECIFIED else 0
            self.orders[t] = NS(ticket=t, magic=req["magic"], symbol=req["symbol"], type=req["type"],
                                price_open=float(req["price"]), sl=float(req.get("sl", 0.0)),
                                volume_initial=float(req["volume"]), time_setup=self.now, time_expiration=exp,
                                comment=req.get("comment", ""))
            return NS(retcode=10009, order=t, price=0.0, comment="Request executed")
        if a == self.TRADE_ACTION_DEAL and req["type"] == self.ORDER_TYPE_BUY:
            t = self._new_ticket()
            self._open(t, req["magic"], req["volume"], tk.ask, req.get("sl", 0.0), self.now, req.get("comment", ""))
            self.hist[t] = NS(ticket=t, state=4)              # a market order is a FILLED order in history
            return NS(retcode=10009, order=t, price=tk.ask, comment="Request executed")
        if a == self.TRADE_ACTION_DEAL and req["type"] == self.ORDER_TYPE_SELL:
            p = self.positions.get(req["position"])
            if p is None:
                return NS(retcode=10013, order=0, price=0.0, comment="no position")
            self._close(p, tk.bid, self.now, req["magic"], req.get("comment", ""))
            return NS(retcode=10009, order=self._new_ticket(), price=tk.bid, comment="Request executed")
        if a == self.TRADE_ACTION_REMOVE:
            o = self.orders.get(req["order"])
            if o is None:
                return NS(retcode=10013, order=0, price=0.0, comment="no order")
            self._finish(o, 2)
            return NS(retcode=10009, order=o.ticket, price=0.0, comment="Request executed")
        raise ValueError(req)


def run_times(first_day: pd.Timestamp, last_day: pd.Timestamp) -> list[int]:
    """The deployed timer on the SERVER clock: every 5 min at :04/5:05, plus every minute 01:01-01:12
    at :05 (the open burst). Server = New York + 7h, so the UTC timer lands on the same minutes."""
    out = []
    for d in pd.date_range(first_day.normalize(), last_day.normalize(), freq="D"):
        mins = sorted(set(range(4, 1440, 5)) | set(range(61, 73)))
        base = int((d - pd.Timestamp(0)).total_seconds())
        out += [base + 60 * mm + 5 for mm in mins]
    return out


@contextlib.contextmanager
def quiet(enabled: bool = True):
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def replay(bars: pd.DataFrame, start: str, end: str, spread: float = 0.0, balance: float = 100_000.0,
           workdir: Path | None = None, times: list[int] | None = None, verbose: bool = False,
           fake: FakeMT5 | None = None):
    """Drive X.Runner exactly as main() does, once per timer slot. Returns (rows, fake, alerts)."""
    alerts = []
    saved = X.time.sleep, X.alert
    X.time.sleep = lambda s: None                           # the fake clock does not move while polling
    X.alert = lambda subj, body: alerts.append(subj)        # never let a replay reach a real channel
    try:
        fake = fake or FakeMT5(bars, spread=spread, balance=balance)
        workdir = Path(workdir or tempfile.mkdtemp(prefix="pdh_replay_"))
        files = X.Files("replay", base=workdir)
        a = NS(execute=True, symbol="XAUUSDs")
        times = times or run_times(pd.Timestamp(start), pd.Timestamp(end))
        for T in times:
            fake.set_time(T)
            st, rows = files.load(), files.rows()
            r = X.Runner(fake, a, files, st, rows)
            with quiet(not verbose):
                try:
                    r.run()
                finally:
                    files.save(r.st)
        return files.rows(), fake, alerts
    finally:
        X.time.sleep, X.alert = saved


def main() -> None:
    from scripts.v5_intraday_m1 import load_m1  # noqa: PLC0415
    from scripts.v5_pdh_m1_stress import run, to_days  # noqa: PLC0415
    start, end = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("2025-05-01", "2025-11-01")
    d = load_m1().astype(float)                              # ET-naive OANDA mid
    srv = d.copy()
    srv.index = srv.index + pd.Timedelta(hours=7)            # server clock = New York + 7h
    warm = pd.Timestamp(start) - pd.Timedelta(days=45)
    bars = srv[(srv.index >= warm) & (srv.index < pd.Timestamp(end) + pd.Timedelta(days=1))]
    rows, fake, alerts = replay(bars, start, end)
    R = pd.DataFrame(rows)
    # backtest on the same bars: 1-ATR stop, exit at the last bar before 16:40 ET, gap = OPEN at/above the level
    det = d[(d.index >= warm - pd.Timedelta(hours=7)) & (d.index < pd.Timestamp(end) + pd.Timedelta(days=1))]
    A, mod, s0, s1, bar = to_days(det, 600)
    lv = bar.high.shift(1).to_numpy()
    gap = np.array([np.isfinite(lv[i]) and A["open"][s0[i]] >= lv[i] for i in range(len(bar))])
    hol = np.array([str(x.date()) in X.HOLIDAYS for x in bar.index])
    bt = run(A, mod, s0, s1, bar, stop_k=1.0, exit_at=16 * 60 + 40, block=gap | hol)
    B = pd.DataFrame(dict(day=[str(x.date()) for x in bar.index], on=bt[1], entry=bt[3], ret=bt[0], level=lv,
                          atr=bar.atr.to_numpy()))
    B = B[(B.day >= start) & (B.day < end)]
    fills = R[R.event == "fill"].copy()
    closed = R[R.event == "closed"].copy()
    lv_rep = R[R.event.isin(["place", "skip", "missed"])].groupby("day").first()
    m = B.set_index("day").join(lv_rep[["level", "atr"]].astype(float), rsuffix="_live", how="left")
    print(f"REPLAY {start} -> {end}: {len(B)} backtest days, log rows {len(R)}, alerts {len(alerts)}")
    print("  events:", dict(R.event.value_counts()))
    lvl = m.dropna(subset=["level_live"])
    print(f"  level  : {len(lvl)} days compared, max |live - backtest| {np.nanmax(np.abs(lvl.level_live - lvl.level)):.4f}")
    print(f"  ATR14  : max |live - backtest| {np.nanmax(np.abs(lvl.atr_live - lvl.atr)):.4f}  (live rounds to 0.01)")
    bt_days = set(B[B.on].day)
    live_days = set(fills.day)
    print(f"  trades : backtest {len(bt_days)}, live {len(live_days)}, both {len(bt_days & live_days)}, "
          f"backtest-only {sorted(bt_days - live_days)[:8]}, live-only {sorted(live_days - bt_days)[:8]}")
    f = fills.set_index("day")[["price"]].astype(float).join(B.set_index("day")[["entry"]], how="inner")
    print(f"  entry  : max |live fill - backtest entry| {np.abs(f.price - f.entry).max():.4f}")
    c = closed.set_index("day")
    c = c[["pnl", "R", "note"]].join(B.set_index("day")[["ret", "entry"]], how="inner")
    c["live_ret"] = c.pnl.astype(float) / (fills.set_index("day").lots.astype(float).reindex(c.index) * 100 * c.entry)
    print(f"  return : corr(live, backtest) {c.live_ret.corr(c.ret):.4f}, mean live {1e4 * c.live_ret.mean():+.2f}bp vs "
          f"backtest {1e4 * c.ret.mean():+.2f}bp (live exits 4 min later: 23:44:05 vs the 23:39 close)")
    print("  exits  :", dict(c.note.str.split(":").str[0].value_counts()))
    print(X.report(rows))


if __name__ == "__main__":
    main()
