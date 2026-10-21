#!/usr/bin/env python3
"""Record the accrued swap on every open position. The A/B that closes 3bd's open item.

3ag found swap-free gold and 3bd left ONE thing open: "the open item is no longer a backtest -- it
is a single observed overnight settlement". On Maven that could never happen, because Maven forbids
EAs so no order could be placed. Switch permits automation, so this logs the settlement directly.

THE COMPARISON IS WITHIN ONE ACCOUNT AT ONE MOMENT, which is what makes it clean:
  XAUUSDs (magic 360595, the live book)  -- spot gold, swap_mode POINTS, swap_long -65.07
  GC.Z6s  (magic 360596, the probe)      -- gold future, swap_mode DISABLED, swap_long 0.0
Same account, same underlying, same nights. If spot accrues and the future does not, the finding is
confirmed by observation rather than by a broker field.

WHY A FIELD IS NOT ENOUGH: Maven's GoldEternal reported swap_mode=0 while swap_long/short showed
-30.0, i.e. "disabled" with the fields still populated. A broker can flip the mode without notice,
which is why v5_manual_alert.py re-checks it every run. Observation beats configuration.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "v5_runs" / "futures_roll" / "swap_observations.csv"
TAGS = {360595: "BOOK-spot", 360596: "PROBE-future", 360594: "FTMO-book", 0: "manual"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18815)
    a = ap.parse_args()
    from mt5linux import MetaTrader5
    m = MetaTrader5(host="localhost", port=a.port)
    if not m.initialize():
        print("init failed: %s" % (m.last_error(),)); return 2
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for p in (m.positions_get() or []):
        si = m.symbol_info(p.symbol)
        rows.append(dict(utc=now, symbol=p.symbol, ticket=p.ticket, magic=p.magic,
                         tag=TAGS.get(p.magic, str(p.magic)), volume=p.volume,
                         price_open=p.price_open, swap=p.swap, profit=p.profit,
                         swap_mode=getattr(si, "swap_mode", ""),
                         swap_long=getattr(si, "swap_long", ""),
                         opened=datetime.utcfromtimestamp(p.time).isoformat()))
    m.shutdown()
    if not rows:
        print("no open positions"); return 0
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    print("%s  %d position(s)" % (now, len(rows)))
    print("  %-10s %-13s %8s %9s %10s %10s" % ("symbol", "tag", "vol", "swap", "swap_mode", "swap_long"))
    for r in rows:
        print("  %-10s %-13s %8.2f %+9.2f %10s %10s" % (
            r["symbol"], r["tag"], r["volume"], r["swap"], r["swap_mode"], r["swap_long"]))
    spot = [r for r in rows if r["symbol"] == "XAUUSDs" and r["magic"] == 360595]
    fut = [r for r in rows if r["symbol"].startswith("GC.")]
    if spot and fut:
        print("\n  A/B: spot gold swap %+.2f vs futures gold swap %+.2f  -> %s" % (
            spot[0]["swap"], fut[0]["swap"],
            "FUTURE IS NOT BEING CHARGED" if abs(fut[0]["swap"]) < 1e-9
            else "*** FUTURE IS ACCRUING SWAP -- investigate ***"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
