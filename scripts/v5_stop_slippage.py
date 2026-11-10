#!/usr/bin/env python3
"""How much do our protective stops actually slip, per venue?

WHY MEASURE BEFORE ENGINEERING. A single Darwinex fill on 2026-11-10 slipped $11.30 on a
4102.10 stop (filled 4090.80) and cost $45.20 — 28bp against a 3.5bp budget. One fill is not a
rate. Every remedy for slippage has a cost of its own, and choosing one before knowing the
distribution is how you pay a certain cost to avoid an uncertain one.

THE DATA IS ALREADY THERE. When MT5 closes a position on its stop, the closing deal's comment
records the stop price that triggered, e.g. `[sl 4102.10]`. Comparing that to the deal's actual
`price` gives the realised slippage, exactly, with no modelling.

SIGN CONVENTION. Slippage is reported as ADVERSE-POSITIVE: how much worse than the stop level we
filled. A long filled BELOW its stop, or a short filled ABOVE, is positive. Negative numbers are
positive slippage — fills better than the stop — and they do happen on a gap through a resting
level in our favour, so the mean is what matters, not just the bad tail.

WHAT IT CANNOT TELL YOU. Only stops that FIRED appear here, so this is the slippage distribution
conditional on being stopped out — which is the one that matters for sizing a remedy, but is not
the unconditional cost of using stops.

    python scripts/v5_stop_slippage.py
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "v5_runs"
VENUES = [(18812, "Maven"), (18815, "Switch"), (18816, "FundingPips"), (18817, "Darwinex")]
TAG = re.compile(r"\[(sl|tp)\s+([0-9.]+)\]", re.I)
DEAL_ENTRY_IN, CLOSING = 0, (1, 2, 3)


def scan(m, venue: str, days: int) -> list[dict]:
    start = datetime.now(timezone.utc) - timedelta(days=days)
    end = datetime.now(timezone.utc) + timedelta(days=3)
    deals = m.history_deals_get(start, end)
    if not deals:
        return []
    # Position id -> the opening deal, so we know the SIDE and the entry. The closing deal's own
    # magic can be 0 when the SERVER closes it on the stop, which is exactly our case here, so
    # side must come from the opener and never from the closing deal.
    opens = {}
    for d in deals:
        if getattr(d, "entry", None) == DEAL_ENTRY_IN and getattr(d, "position_id", 0):
            opens.setdefault(d.position_id, d)

    rows = []
    for d in deals:
        if getattr(d, "entry", None) not in CLOSING:
            continue
        mt = TAG.search(str(getattr(d, "comment", "") or ""))
        if not mt:
            continue
        kind, level = mt.group(1).lower(), float(mt.group(2))
        o = opens.get(getattr(d, "position_id", 0))
        if o is None:
            continue
        is_long = (o.type == 0)
        fill = float(d.price)
        # adverse-positive: a long wants to exit HIGH, so filling below the level is adverse
        slip = (level - fill) if is_long else (fill - level)
        contract = float(getattr(d, "volume", 0)) * 100.0        # XAU/most CFDs: 100 units
        rows.append(dict(venue=venue, when=datetime.fromtimestamp(int(d.time), timezone.utc),
                         magic_open=int(getattr(o, "magic", 0)), symbol=d.symbol,
                         kind=kind, side="LONG" if is_long else "SHORT",
                         volume=float(d.volume), level=level, fill=fill,
                         slip_px=slip, slip_bp=slip / level * 1e4,
                         slip_usd=slip * contract, entry=float(o.price)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3650)
    a = ap.parse_args()

    from mt5linux import MetaTrader5
    rows = []
    for port, name in VENUES:
        try:
            m = MetaTrader5(host="localhost", port=port)
            if not m.initialize():
                print(f"{name:<12} bridge {port}: not connected")
                continue
            r = scan(m, name, a.days)
            rows += r
            print(f"{name:<12} bridge {port}: {len(r)} stop/tp-closed deal(s)")
        except Exception as exc:                                  # noqa: BLE001
            print(f"{name:<12} bridge {port}: {type(exc).__name__}: {exc}")
    if not rows:
        print("\nno stop-closed deals with a recorded level found — nothing to measure yet")
        return 0

    df = pd.DataFrame(rows).sort_values("when")
    sl = df[df.kind == "sl"]
    print(f"\n{len(sl)} STOP-LOSS fills with a recorded level\n")
    print(f"{'when':<17}{'venue':<12}{'sym':<9}{'side':<6}{'vol':>6}{'level':>10}{'fill':>10}"
          f"{'slip $':>9}{'slip bp':>9}")
    for _, r in sl.iterrows():
        print(f"{r.when:%Y-%m-%d %H:%M}  {r.venue:<12}{r.symbol:<9}{r.side:<6}{r.volume:>6.2f}"
              f"{r.level:>10.2f}{r.fill:>10.2f}{r.slip_usd:>9.2f}{r.slip_bp:>9.2f}")

    print(f"\n{'venue':<12}{'n':>4}{'mean bp':>10}{'median':>9}{'worst':>9}{'best':>9}"
          f"{'>5bp':>7}{'total $':>10}")
    for v, g in sl.groupby("venue"):
        print(f"{v:<12}{len(g):>4}{g.slip_bp.mean():>10.2f}{g.slip_bp.median():>9.2f}"
              f"{g.slip_bp.max():>9.2f}{g.slip_bp.min():>9.2f}"
              f"{(g.slip_bp > 5).mean():>7.0%}{g.slip_usd.sum():>10.2f}")
    print(f"{'ALL':<12}{len(sl):>4}{sl.slip_bp.mean():>10.2f}{sl.slip_bp.median():>9.2f}"
          f"{sl.slip_bp.max():>9.2f}{sl.slip_bp.min():>9.2f}"
          f"{(sl.slip_bp > 5).mean():>7.0%}{sl.slip_usd.sum():>10.2f}")

    print("\nadverse-positive: + = filled WORSE than the stop level, - = filled better.")
    if len(sl) < 20:
        print(f"*** {len(sl)} fills is not a distribution. Treat every number above as an")
        print("    observation, not a rate, and do not size a remedy off it. ***")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "stop_slippage.csv", index=False)
    print(f"\nwrote {OUT / 'stop_slippage.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
