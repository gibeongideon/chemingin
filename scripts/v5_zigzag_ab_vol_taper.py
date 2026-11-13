#!/usr/bin/env python3
"""Live A/B: ZigZag arm B (360608, full size) vs arm V (360613, §3eu downside-vol size dial).

FTMO-Demo, bridge 18814, both on XAUUSD, both hourly on the same closed H1 bar (B at :01:40,
V at :02:20). They differ in ONE thing: when downside_vol(500 H1 bars) >= its own rolling
median(2000), arm V takes the same fire at HALF SIZE.

*** THE EVENT SETS SHOULD BE IDENTICAL, AND THE FIRST JOB OF THIS REPORT IS TO PROVE IT. ***
This is not the usual A/B where a filter changes which trades happen — the whole point of §3eu is
that a SIZE dial leaves the event set fixed (§3co). So if the two arms ever diverge in WHICH
trades they take, the experiment has broken (a missed run, a spread skip, a bridge timeout) and
every P&L number below is contaminated. That check runs before anything else.

Because the event set is shared, the comparison is unusually clean: same entries, same exits,
same R per trade — only the dollar size differs. The taper's effect is therefore entirely
visible in the subset where arm V sized down.

    python scripts/v5_zigzag_ab_vol_taper.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_zigzag_results import closed_trades  # noqa: E402

OUT = ROOT / "data" / "v5_runs"
CFG = ROOT / "configs" / "v5_zigzag_trail_ftmo_v.json"
PORT, B, V = 18814, 360608, 360613
PAIR_H, MIN_PAIRS = 3, 20


def armed_since() -> datetime:
    """The day arm V went live. Arm B has been running since 2026-11-09, so comparing the whole
    history would report every pre-deployment B trade as a 'divergence' -- which is exactly what
    the first version of this script did. The A/B only exists from the day both arms were armed.
    """
    try:
        d = json.loads(CFG.read_text())["deployed_on"]
        return datetime.fromisoformat(str(d)).replace(tzinfo=timezone.utc)
    except Exception:
        return datetime(2026, 11, 13, tzinfo=timezone.utc)


def rows(m, magic: int, days: int) -> pd.DataFrame:
    d = pd.DataFrame(closed_trades(m, magic, days))
    if not d.empty:
        d["net_bp"] = (d["net"] / (d["open_price"] * d["volume"] * 100.0)) * 1e4
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3650)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--since", default=None,
                    help="YYYY-MM-DD; default = arm V's deployed_on, i.e. when the A/B began")
    a = ap.parse_args()
    since = (datetime.fromisoformat(a.since).replace(tzinfo=timezone.utc)
             if a.since else armed_since())

    from mt5linux import MetaTrader5
    m = MetaTrader5(host="localhost", port=a.port)
    if not m.initialize():
        print(f"bridge {a.port} not connected: {m.last_error()}")
        return 2

    ai = m.account_info()
    print("=" * 96)
    print("ZigZag A/B — arm B 360608 (full size)  vs  arm V 360613 (half size in high vol)")
    print("=" * 96)
    if ai is not None:
        print(f"  account {ai.login}  balance {ai.balance:,.2f}  equity {ai.equity:,.2f}")

    db_all, dv_all = rows(m, B, a.days), rows(m, V, a.days)
    db = db_all[db_all.open_time >= since] if not db_all.empty else db_all
    dv = dv_all[dv_all.open_time >= since] if not dv_all.empty else dv_all
    pre = len(db_all) - len(db)
    print(f"  A/B window starts {since:%Y-%m-%d} (arm V's deployment); "
          f"{pre} earlier arm-B trade(s) excluded")
    ob = [p for p in (m.positions_get() or []) if p.magic == B]
    ov = [p for p in (m.positions_get() or []) if p.magic == V]
    print(f"  arm B: {len(db)} closed, {len(ob)} open     arm V: {len(dv)} closed, {len(ov)} open")
    if db.empty and dv.empty:
        print("\n  Nothing has closed on either arm yet.")
        return 0

    # ---- 1. INTEGRITY: the event sets must match ------------------------------------------
    print("\n  1. EVENT-SET INTEGRITY (a size dial must not change WHICH trades happen)")
    pairs, lone_b, lone_v = [], [], []
    for _, b in db.iterrows():
        if dv.empty:
            lone_b.append(b); continue
        dt = (dv["open_time"] - b["open_time"]).abs()
        if dt.min() <= timedelta(hours=PAIR_H):
            pairs.append((b, dv.loc[dt.idxmin()]))
        else:
            lone_b.append(b)
    for _, v in dv.iterrows():
        if db.empty or (db["open_time"] - v["open_time"]).abs().min() > timedelta(hours=PAIR_H):
            lone_v.append(v)
    print(f"     matched {len(pairs)}   B-only {len(lone_b)}   V-only {len(lone_v)}")
    if lone_b or lone_v:
        print("     *** THE ARMS HAVE DIVERGED. A size dial cannot do that, so a run was missed")
        print("         or an order was rejected. Investigate before reading the P&L below. ***")
        for x in (lone_b + lone_v)[:5]:
            print(f"       unmatched {x['open_time']:%Y-%m-%d %H:%M}  {x['volume']} lots "
                  f"net {x['net']:+.2f}")
    else:
        print("     clean — every trade has a twin, so the comparison is size and nothing else")

    # ---- 2. the taper's effect -------------------------------------------------------------
    for nm, d in (("B (full size) ", db), ("V (vol taper)", dv)):
        if d.empty:
            continue
        w = int((d["net"] > 0).sum())
        print(f"  {nm}: {len(d)} trades  {w}W/{len(d) - w}L  net {d['net'].sum():+9.2f}  "
              f"mean {d['net'].mean():+7.2f}  {d['net_bp'].mean():+6.1f} bp/trade")

    if pairs:
        lb = np.array([float(b["volume"]) for b, _ in pairs])
        lv = np.array([float(v["volume"]) for _, v in pairs])
        nb = np.array([float(b["net"]) for b, _ in pairs])
        nv = np.array([float(v["net"]) for _, v in pairs])
        tapered = lv < lb * 0.75          # arm V sized down on this signal
        print(f"\n  2. THE DIAL FIRED on {int(tapered.sum())} of {len(pairs)} paired trades "
              f"({100 * tapered.mean():.0f}%)   — backtest said ~45%")
        if tapered.any():
            print(f"     on TAPERED signals:  B {nb[tapered].sum():+8.2f}   "
                  f"V {nv[tapered].sum():+8.2f}   saved {nv[tapered].sum() - nb[tapered].sum():+8.2f}")
        if (~tapered).any():
            print(f"     on FULL-SIZE signals: B {nb[~tapered].sum():+8.2f}   "
                  f"V {nv[~tapered].sum():+8.2f}   (should be ~equal; any gap is slippage/timing)")
        print(f"\n  3. TOTAL: B {nb.sum():+.2f}   V {nv.sum():+.2f}   "
              f"V-B {nv.sum() - nb.sum():+.2f}")
        if len(pairs) < MIN_PAIRS:
            print(f"     *** {len(pairs)} pairs is an observation, not a result. §3eu's")
            print(f"         expectation is ~+0.10 to +0.22 Sharpe and a 1.7-3.0pp smaller")
            print(f"         drawdown — neither is visible in {len(pairs)} trades. ***")

    allrows = ([dict(arm="B", **r) for r in db.to_dict("records")]
               + [dict(arm="V", **r) for r in dv.to_dict("records")])
    if allrows:
        pd.DataFrame(allrows).to_csv(OUT / "ab_zigzag_vol_taper.csv", index=False)
        print(f"\n  wrote {OUT / 'ab_zigzag_vol_taper.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
