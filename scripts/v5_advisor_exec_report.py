#!/usr/bin/env python3
"""Live A/B of the two FTMO advisor arms, plus the only benchmark that matters.

    360611  arm LS  long_short   a SELL call opens a short
    360612  arm L   long_only    a SELL call opens nothing

The arms are identical in every other respect -- same signal, same episode gate, same 1.0 ATR
initial stop and ratchet, same +$50 rescue on a kept leg, same account, same ticks. So the gap
between them is the SHORT LEG and nothing else. V5_FINDINGS 3bz measured that leg at -0.29 to
-0.52 Sharpe over four thresholds; this is the live test of that number.

*** TWO THINGS THIS REPORT REFUSES TO DO. ***
1. It never scores a strategy against zero. V5_FINDINGS 3bh: excess(LONG) + excess(SHORT) =
   -2 x base, so on a drifting asset both sides of a mirror pair lose to simply holding it. The
   `vs HOLD` column is the verdict; `net` on its own is not.
2. It never calls a winner on a handful of fires. Under MIN_PAIRS matched episodes it prints the
   numbers and says they are an observation (scripts/v5_ab_report.py:137 does the same).

    python scripts/v5_advisor_exec_report.py
    python scripts/v5_advisor_exec_report.py --days 30
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_zigzag_results import closed_trades  # noqa: E402

OUT = ROOT / "data" / "v5_runs"
PORT, SYMBOL = 18814, "XAUUSD"
ARMS = [("LS  long/short", 360611, "ftmo_ls"), ("L   long-only ", 360612, "ftmo_lo")]
MIN_PAIRS = 12          # below this the comparison is an observation, not a result
PAIR_WINDOW_H = 6       # two fires belong to the same advisor episode if they open this close

# Pre-registered before the first fill (2026-11-10). Recorded here so the stopping rule is not
# chosen after seeing the numbers.
KILL_TRADES, KILL_DAYS = 60, 90


def arm_rows(m, magic: int, days: int) -> pd.DataFrame:
    df = pd.DataFrame(closed_trades(m, magic, days))
    if not df.empty:
        df["net_bp"] = (df["net"] / (df["open_price"] * df["volume"] * 100.0)) * 1e4
    return df


def hold_return_bp(m, since: datetime, until: datetime) -> float:
    """What buy-and-hold on the same symbol did over the same window, in bp. Control #2."""
    r = m.copy_rates_from_pos(SYMBOL, 16385, 0, 20000)          # H1
    if r is None or not len(r):
        return float("nan")
    d = pd.DataFrame([(x[0], x[4]) for x in r], columns=["ts", "close"])
    d["t"] = pd.to_datetime(d.ts, unit="s", utc=True)
    w = d[(d.t >= since) & (d.t <= until)]
    if len(w) < 2:
        return float("nan")
    return 1e4 * (float(w["close"].iloc[-1]) / float(w["close"].iloc[0]) - 1.0)


def describe(name: str, d: pd.DataFrame) -> None:
    if d.empty:
        print("  %s : no closed trades yet" % name)
        return
    w = int((d["net"] > 0).sum())
    swap = float(d["swap"].sum()) if "swap" in d else 0.0
    comm = float(d["commission"].sum()) if "commission" in d else 0.0
    print("  %s : %3d trades  %3dW/%3dL (%.0f%%)  net %+9.2f  mean %+7.2f  %+7.1f bp/trade"
          % (name, len(d), w, len(d) - w, 100 * w / len(d), d["net"].sum(), d["net"].mean(),
             d["net_bp"].mean()))
    print("  %s   financing: swap %+.2f  commission %+.2f  (the hedge pays two of each, 3bh)"
          % (" " * len(name), swap, comm))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3650)
    ap.add_argument("--port", type=int, default=PORT)
    a = ap.parse_args()

    from mt5linux import MetaTrader5
    m = MetaTrader5(host="localhost", port=a.port)
    if not m.initialize():
        print("bridge %d not connected: %s" % (a.port, m.last_error()))
        return 2

    print("=" * 78)
    print("ADVISOR ARMS ON FTMO  —  the only difference is whether a SELL call opens a short")
    print("=" * 78)
    ai = m.account_info()
    if ai is not None:
        print("  account %s  balance %.2f  equity %.2f" % (ai.login, ai.balance, ai.equity))

    data = {}
    for name, magic, tag in ARMS:
        d = arm_rows(m, magic, a.days)
        data[magic] = d
        open_now = [p for p in (m.positions_get() or []) if p.magic == magic]
        describe(name, d)
        if open_now:
            print("  %s   open now: %s" % (" " * len(name), ", ".join(
                "#%s %s %.2f pnl %+.2f" % (p.ticket, "BUY" if p.type == 0 else "SELL",
                                           p.volume, p.profit + p.swap) for p in open_now)))

    ls, lo = data[360611], data[360612]
    if ls.empty and lo.empty:
        print("\n  Nothing has closed on either arm yet. Nothing to compare.")
        return 0

    # ---------------------------------------------------- the benchmark, before the comparison
    allt = pd.concat([x for x in (ls, lo) if not x.empty])
    since, until = allt["open_time"].min(), allt["close_time"].max()
    hb = hold_return_bp(m, since, until)
    span_d = (until - since).total_seconds() / 86400
    print("\n  WINDOW %s -> %s  (%.1f days)" % (since.strftime("%Y-%m-%d"),
                                                until.strftime("%Y-%m-%d"), span_d))
    print("  holding gold over the same window: %+.1f bp" % hb)
    for name, magic, _ in ARMS:
        d = data[magic]
        if d.empty:
            continue
        tot = float(d["net"].sum()) / float(ai.equity) * 1e4 if ai else float("nan")
        print("  %s : %+.1f bp of equity   vs HOLD %+.1f bp  -> %s"
              % (name, tot, hb, "BEATS hold" if tot > hb else "LOSES to hold"))

    # --------------------------------------------------------- matched episodes, paired t ----
    if not ls.empty and not lo.empty:
        pairs = []
        for _, x in ls.iterrows():
            dt = (lo["open_time"] - x["open_time"]).abs()
            if len(dt) and dt.min() <= timedelta(hours=PAIR_WINDOW_H):
                pairs.append((x, lo.loc[dt.idxmin()]))
        if pairs:
            dls = np.array([float(x["net_bp"]) for x, _ in pairs])
            dlo = np.array([float(y["net_bp"]) for _, y in pairs])
            diff = dls - dlo
            t = (diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
                 if len(diff) > 2 and diff.std(ddof=1) > 0 else float("nan"))
            print("\n  THE SHORT LEG, on %d matched episodes" % len(diff))
            print("    LS %+.1f bp/trade   L %+.1f bp/trade   LS-L %+.1f   paired t %+.2f"
                  % (dls.mean(), dlo.mean(), diff.mean(), t))
            if len(diff) < MIN_PAIRS:
                print("    *** %d pairs is an OBSERVATION, not a result. 3bz's prior is that the"
                      % len(diff))
                print("        short leg costs 0.29-0.52 Sharpe. ***")
        else:
            print("\n  No matched episodes yet — the arms have not both fired on one call.")

    # ------------------------------------------------------------------ the pre-registered rule
    n = max(len(ls), len(lo))
    print("\n  KILL RULE (pre-registered 2026-11-10, before the first fill):")
    print("    at %d closed trades per arm or %d days, whichever first — LS is disabled if it is"
          % (KILL_TRADES, KILL_DAYS))
    print("    net-negative AND below L; BOTH are disabled if both are below buy-and-hold.")
    print("    now: %d trades, %.0f days -> %s"
          % (n, span_d, "DECIDE NOW" if (n >= KILL_TRADES or span_d >= KILL_DAYS)
             else "keep running"))

    if not allt.empty:
        OUT.mkdir(parents=True, exist_ok=True)
        allt.to_csv(OUT / "advisor_arms_ftmo.csv", index=False)
        print("\n  wrote %s" % (OUT / "advisor_arms_ftmo.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
