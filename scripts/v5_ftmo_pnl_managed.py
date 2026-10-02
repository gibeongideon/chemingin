"""PnL-managed variant of the FTMO book: take-profit at a dollar target, harvest before the
close, re-enter on a signal. Emails EVERY action it takes.

*** WHAT THIS COSTS, MEASURED BEFORE IT WAS BUILT (`v5_pnl_overlay_backtest.py`). ***
On the champion's gold sleeve, M15 path, same sizing and costs, 2015-2026:

    champion, no overlay                  Sharpe +0.71   CAGR +7.09%   maxDD -25.3%
    +$200 target, re-enter on a signal    Sharpe +0.45   CAGR +2.98%   maxDD -22.7%
    +$200 target, re-enter immediately    Sharpe +0.54   CAGR +5.14%   maxDD -31.8%

and the fair test, because a de-risking overlay must be judged at MATCHED risk
(`xau-champion-five-lifts-fail`): at the overlay's own -22.7% drawdown, simply holding **0.89x
the size** returns **+6.32%** against the overlay's **+2.98%**. The overlay gives up 3.33pp of
CAGR for a drawdown reduction that position sizing provides free. Its Sharpe is 0.45 against
0.71, and Sharpe is already risk-adjusted, so the reduction is bought by being out of the market
rather than by better timing.

Two further measured facts the request assumed otherwise:
  * WAITING for a "right entry" is WORSE than re-entering at once (+0.45 vs +0.54). The wait
    misses the continuation. `--reenter immediate` switches to the better-measured behaviour.
  * the target fires constantly: 975 take-profits and 133 harvests over the sample, cutting the
    position on **31% of all days**. On $102k, +$200 is +0.195% of equity, and gold's daily range
    is ~2.7% of price -- the target sits inside a fifth of one average day.

This is deployed because it was asked for on a TRIAL account, with the numbers in every email so
the live result can be read against the backtest rather than against hope. §3x already closed
"cash out on first reasonable profit" (48/48 cells negative, walk-forward -0.76 vs buy-and-hold
+0.79) and §3ab's exit sweep had 0 of 64 cells clearing +0.50.

HOW IT WORKS
  IN  -> combined floating P/L >= --target            : close ALL, email, go FLAT
      -> inside the last --harvest-window h before the daily close (21:00 UTC, gold's own
         session break) and P/L >= --harvest          : close ALL, email, lock for the session
  FLAT-> market open, not locked, and the entry signal fires : hand over to the normal executor
         to open the book's target positions, then email what was opened

The entry signal is the champion's own forecast STRENGTHENING versus the previous close. That is
the only measured signal in this repo; bolting on a disproven one would be worse than waiting.

SAFETY. It manages magic 360591 only -- the same magic as `book-ftmo`, so it adopts the existing
positions and the two can never both run (book-ftmo.timer is disabled when this is enabled).
It never opens while the executor's own guards say halt, and it refuses on a non-demo account
unless --live is passed.

    python scripts/v5_ftmo_pnl_managed.py --dry
    python scripts/v5_ftmo_pnl_managed.py --execute
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_notify import channel, env as notify_env, notify  # noqa: E402

STATE = ROOT / "data" / "v5_runs" / "pnl_managed_state.json"
CFG = ROOT / "configs" / "v5_ftmo_gold_tilted.json"
MAGIC = 360591
CLOSE_UTC_H = 21                 # gold's daily session break, measured 21:00-22:00 UTC
FLATTEN_UTC = (20, 45)           # hard flatten: overnight holding is NOT PERMITTED
MEASURED = (
    "MEASURED, gold sleeve 2015-2026: champion holding overnight +0.77 SR / +7.90% CAGR / "
    "-22.1% DD. FLAT EVERY NIGHT (this account's rule): +0.50 / +4.75% / -35.5% -- worse "
    "return AND worse drawdown. The cost is TURNOVER (132x: 21 units of crossings becomes "
    "2,832), not missed gaps: for XAUUSD the overnight gap is only 2% of the move captured, "
    "because gold trades ~23h/day. The +$200 target costs a further 0.26 SR; at matched "
    "drawdown, 0.89x plain size returns +6.32% vs the overlay's +2.98%.")


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def save_state(s: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(s, indent=2, default=str))
    tmp.replace(STATE)


def market_open(conn, symbol="XAUUSD") -> tuple[bool, float]:
    conn.symbol_select(symbol, True)
    t = conn.get_tick(symbol)
    if t is None:
        return False, float("inf")
    age = (time.time() - float(getattr(t, "time", 0) or 0)) / 60.0
    return age <= 10, age


def entry_signal(conn) -> tuple[bool, str]:
    """The champion's own forecast strengthening — the only measured signal here."""
    from src.v5.xau_dual_signals import champion_signal
    r = conn.get_rates("XAUUSD", "D1", 400)
    if r is None or len(r) < 60:
        return False, "no daily history"
    d = pd.DataFrame(r)
    col = "close" if "close" in d.columns else d.columns[4]
    fc = champion_signal(pd.Series(d[col].values, dtype=float))
    now, prev = float(fc.iloc[-1]), float(fc.iloc[-2])
    if abs(now) < 0.05:
        return False, f"forecast flat ({now:+.3f})"
    if abs(now) <= abs(prev):
        return False, f"forecast not strengthening ({prev:+.3f} -> {now:+.3f})"
    return True, f"forecast strengthening {prev:+.3f} -> {now:+.3f}"


def positions(conn):
    ps = conn.get_positions(magic=MAGIC) or []
    return list(ps)


def combined_pnl(ps) -> float:
    return float(sum(p.profit for p in ps))


def send(subject: str, body: str, dry: bool) -> None:
    line = f"\n--- MEASURED ---\n{MEASURED}\nV5_FINDINGS 3bk. This is a de-risking overlay, " \
           "not an edge; position sizing achieves the same drawdown for more return."
    if dry:
        print(f"[dry-email] {subject}\n{body}{line}")
        return
    notify(subject, body + line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18814)
    ap.add_argument("--target", type=float, default=200.0)
    ap.add_argument("--harvest", type=float, default=50.0)
    ap.add_argument("--harvest-window", type=float, default=5.0)
    ap.add_argument("--reenter", choices=["signal", "immediate"], default="signal")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--flatten", default=f"{FLATTEN_UTC[0]:02d}:{FLATTEN_UTC[1]:02d}",
                    help="UTC HH:MM hard flatten. Overnight holding is not permitted, so this "
                         "is a compliance deadline, not a preference. Settable so the path can "
                         "be PROVEN without waiting for the evening.")
    a = ap.parse_args()

    from src.core.mt5_connector import MT5Connector
    conn = MT5Connector(port=a.port)
    conn.connect()
    st = load_state()
    now = datetime.now(timezone.utc)
    try:
        acct = conn.account_info()
        is_demo = "demo" in str(getattr(acct, "server", "")).lower()
        may_send = a.execute and (is_demo or a.live) and not a.dry
        ps = positions(conn)
        pnl = combined_pnl(ps)
        hrs_to_close = (CLOSE_UTC_H - now.hour) - now.minute / 60.0
        in_harvest = 0 < hrs_to_close <= a.harvest_window
        opened, age = market_open(conn)
        print(f"{now:%Y-%m-%d %H:%M}Z acct {acct.login} eq {acct.equity:,.2f} | "
              f"positions {len(ps)} | combined P/L {pnl:+,.2f} | "
              f"{'OPEN' if opened else f'CLOSED ({age/60:.0f}h)'} | "
              f"{hrs_to_close:+.1f}h to close{' [HARVEST WINDOW]' if in_harvest else ''}")

        # OVERNIGHT HOLDING IS NOT PERMITTED on this account, so the hard flatten outranks
        # every P/L rule: a position still open at the cutoff is a compliance breach, and a
        # breach costs more than any target is worth.
        flat_h, flat_m = (int(x) for x in a.flatten.split(":"))
        past_cutoff = (now.hour, now.minute) >= (flat_h, flat_m) or now.hour < 1
        friday_late = now.weekday() == 4 and now.hour >= 12

        action, reason = None, ""
        if ps:
            if past_cutoff:
                action, reason = "close", (f"HARD FLATTEN {flat_h:02d}:{flat_m:02d}Z — overnight "
                                           f"holding is not permitted (P/L {pnl:+,.2f}, closing "
                                           f"regardless)")
            elif pnl >= a.target:
                action, reason = "close", f"combined P/L {pnl:+,.2f} reached the +{a.target:,.0f} target"
            elif in_harvest and pnl >= a.harvest:
                action, reason = "close", (f"combined P/L {pnl:+,.2f} reached the "
                                           f"+{a.harvest:,.0f} harvest inside the last "
                                           f"{a.harvest_window:.0f}h before the close")
            else:
                need = (a.harvest if in_harvest else a.target)
                reason = (f"holding — P/L {pnl:+,.2f}, needs +{need:,.0f} "
                          f"({'harvest window' if in_harvest else 'target'}); "
                          f"hard flatten in {hrs_to_close - (CLOSE_UTC_H - flat_h - flat_m/60):.1f}h")
        else:
            lock = st.get("locked_until")
            locked = bool(lock and now < datetime.fromisoformat(lock))
            # a position opened now must still be closable before the cutoff
            hrs_to_cutoff = (flat_h + flat_m / 60.0) - (now.hour + now.minute / 60.0)
            if not opened:
                reason = "market closed"
            elif past_cutoff or hrs_to_cutoff <= 0.5:
                reason = (f"too close to the {flat_h:02d}:{flat_m:02d}Z flatten "
                          f"({hrs_to_cutoff:+.1f}h) — would have to be closed at once")
            elif friday_late:
                reason = "Friday afternoon — not opening into the weekend break"
            elif locked:
                reason = f"locked until {lock} (already closed this session)"
            elif in_harvest:
                reason = "inside the harvest window — not opening only to harvest"
            else:
                if a.reenter == "immediate":
                    action, reason = "open", "re-entry: immediate (the better-measured arm)"
                else:
                    ok, why = entry_signal(conn)
                    if ok:
                        action, reason = "open", f"entry signal: {why}"
                    else:
                        reason = f"waiting for entry signal — {why}"
        print(f"  decision: {action or 'hold'} — {reason}")

        if action == "close" and ps:
            closed, failed = [], []
            for p in ps:
                if not may_send:
                    closed.append((p.symbol, p.volume, p.profit)); continue
                try:
                    conn.close_position(p)
                    closed.append((p.symbol, p.volume, p.profit))
                except Exception as e:
                    failed.append((p.symbol, f"{type(e).__name__}: {e}"))
            lines = [f"CLOSED ALL — {reason}", "",
                     f"  account {acct.login}  equity {acct.equity:,.2f}"]
            for s_, v, pr in closed:
                lines.append(f"    {s_:12s} {v:5.2f} lots   P/L {pr:+8.2f}")
            lines.append(f"    ------------------------------  total {pnl:+,.2f}")
            if failed:
                lines += ["", "  FAILED to close:"] + [f"    {s_}: {e}" for s_, e in failed]
            if not may_send:
                lines += ["", "  (DRY — nothing was actually closed)"]
            send(f"[book-pnl] CLOSED ALL at {pnl:+,.0f}", "\n".join(lines), a.dry or not may_send)
            # after any close, stay flat until the next session. Re-opening the same evening
            # would pay the spread again for the few hours left before a mandatory flatten.
            st["locked_until"] = (now.replace(hour=0, minute=0, second=0, microsecond=0)
                                  + timedelta(days=1)).isoformat()
            st["last_close"] = now.isoformat()
            st["last_close_pnl"] = pnl
            save_state(st)

        elif action == "open":
            cmd = [str(ROOT / ".." / ".." / ".." / "home" / "trader" / "miniconda3" / "envs"
                       / "envmt5" / "bin" / "python")]
            cmd = ["python3"] if not Path(cmd[0]).exists() else cmd
            cmd += [str(ROOT / "scripts" / "v5_basket_challenge_exec.py"),
                    "--config", str(CFG),
                    "--state", str(ROOT / "data/v5_runs/ftmo_gold_tilted_state.json"),
                    "--paper-csv", str(ROOT / "data/v5_runs/ftmo_gold_tilted_log.csv"),
                    "--port", str(a.port)]
            if may_send:
                cmd += ["--live", "--execute"]
            conn.disconnect()
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            tail = "\n".join([l for l in out.stdout.splitlines()
                              if any(k in l for k in ("symbol", "Opened", "EXECUTED", "guards",
                                                      "compliance", "no change"))][-14:])
            send(f"[book-pnl] OPENED — {reason[:50]}",
                 f"OPENED positions — {reason}\n\n{tail or out.stdout[-900:]}",
                 a.dry or not may_send)
            st["last_open"] = now.isoformat()
            save_state(st)
            return
        else:
            st["last_check"] = now.isoformat()
            st["last_reason"] = reason
            save_state(st)
    finally:
        try:
            conn.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
