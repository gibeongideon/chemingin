#!/usr/bin/env python3
"""Live A/B: ZigZag arm B (market entry, no lock) vs arm C (limit entry, profit ladder).

PAIRS
    FTMO      1515233856  bridge 18814   B 360608   C 360609
    Darwinex  3000112593  bridge 18817   B 360606   C 360610

*** TWO THINGS DIFFER AT ONCE, SO A HEAD-TO-HEAD P&L ANSWERS NOTHING. ***
Arm C changes the ENTRY (a resting buy limit 0.5 ATR below the ask) and the EXIT (the
+100/+200/+300 usd lock ladder) together. A raw comparison would credit or blame the lock for
whatever the entry did. This report therefore decomposes into the three effects that are
actually separable from live fills:

  1. FILL RATE.  C only trades when price comes back to it. The fires it misses are not
     neutral — §3bc measured FILLED -22.8bp vs SKIPPED +17.9bp (t -5.23) and §3bz confirmed it:
     waiting for a dip SELECTS LOSERS. A low fill rate with good per-trade numbers is the
     signature of that selection, not of a better rule.
  2. ENTRY ADVANTAGE.  On signals where C filled, how much cheaper did it get in than B? That
     is the part of any C outperformance owed to price, not to the exit rule.
  3. EXIT RULE, ON A MATCHED EVENT SET.  B restricted to the signals C actually filled, against
     C. This is the §3co event-set rule applied live, and it is the only column that speaks to
     the ladder itself.

Until a dozen or so paired fires exist, every number here is an observation. The script says so
rather than pretending otherwise.

    python scripts/v5_ab_report.py
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
PAIRS = [("FTMO", 18814, 360608, 360609), ("Darwinex", 18817, 360606, 360610)]
PAIR_WINDOW_H = 36      # a B and a C trade belong to the same signal if they open within this


def arm_rows(m, magic: int) -> pd.DataFrame:
    tr = closed_trades(m, magic)
    df = pd.DataFrame(tr)
    if not df.empty:
        df["net_bp"] = (df["net"] / (df["open_price"] * df["volume"] * 100.0)) * 1e4
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-h", type=float, default=PAIR_WINDOW_H)
    a = ap.parse_args()

    from mt5linux import MetaTrader5
    allrows = []
    for venue, port, mb, mc in PAIRS:
        print(f"\n{'=' * 78}\n{venue}  —  B {mb} (market, no lock)   vs   C {mc} (limit, ladder)"
              f"\n{'=' * 78}")
        try:
            m = MetaTrader5(host="localhost", port=port)
            if not m.initialize():
                print(f"  bridge {port} not connected"); continue
        except Exception as exc:                                   # noqa: BLE001
            print(f"  bridge {port}: {type(exc).__name__}"); continue

        B, C = arm_rows(m, mb), arm_rows(m, mc)
        openB = [p for p in (m.positions_get() or []) if p.magic == mb]
        openC = [p for p in (m.positions_get() or []) if p.magic == mc]
        pendC = [o for o in (m.orders_get() or []) if o.magic == mc]

        print(f"  B: {len(B)} closed, {len(openB)} open")
        print(f"  C: {len(C)} closed, {len(openC)} open, {len(pendC)} limit resting")

        if B.empty and C.empty:
            print("  no closed trades on either arm yet — nothing to compare")
            continue

        for nm, d in (("B", B), ("C", C)):
            if d.empty:
                print(f"  {nm}: no closed trades")
                continue
            w = int((d["net"] > 0).sum())
            print(f"  {nm}: {len(d)}W/L {w}/{len(d) - w}  net {d['net'].sum():+.2f}  "
                  f"mean {d['net'].mean():+.2f}  bp/trade {d['net_bp'].mean():+.1f}")
            allrows += [dict(venue=venue, arm=nm, **r) for r in d.to_dict("records")]

        # ---- 1. fill rate: how many B entries did C match? -----------------------------------
        if not B.empty:
            matched, unmatched = [], []
            for _, b in B.iterrows():
                if C.empty:
                    unmatched.append(b); continue
                dt = (C["open_time"] - b["open_time"]).abs()
                hit = C[dt <= timedelta(hours=a.window_h)]
                (matched if len(hit) else unmatched).append(b)
            fr = len(matched) / len(B) if len(B) else float("nan")
            print(f"\n  1. FILL RATE   C filled {len(matched)} of B's {len(B)} signals "
                  f"({fr:.0%})")
            if unmatched:
                miss = pd.DataFrame(unmatched)
                print(f"     B on the signals C MISSED: {miss['net'].mean():+.2f} mean "
                      f"({len(miss)} trades)")
                if matched:
                    got = pd.DataFrame(matched)
                    print(f"     B on the signals C FILLED: {got['net'].mean():+.2f} mean "
                          f"({len(got)} trades)")
                    print("     *** If MISSED > FILLED, C is skipping the winners — that is the"
                          "\n         skipped-sample law (§3bc), not evidence about the lock. ***")

            # ---- 2 & 3. entry advantage and the matched-event-set exit comparison ------------
            if matched and not C.empty:
                got = pd.DataFrame(matched).sort_values("open_time")
                pairs = []
                for _, b in got.iterrows():
                    dt = (C["open_time"] - b["open_time"]).abs()
                    j = C.loc[dt.idxmin()]
                    pairs.append((b, j))
                adv = np.mean([float(b["open_price"] - c["open_price"]) for b, c in pairs])
                print(f"\n  2. ENTRY       C got in {adv:+.2f} cheaper on average "
                      f"({len(pairs)} paired signals)")
                db = np.array([float(b["net_bp"]) for b, _ in pairs])
                dc = np.array([float(c["net_bp"]) for _, c in pairs])
                diff = dc - db
                t = (diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
                     if len(diff) > 2 and diff.std(ddof=1) > 0 else float("nan"))
                print(f"\n  3. EXIT RULE   matched event set, {len(diff)} pairs")
                print(f"     B {db.mean():+.1f} bp/trade   C {dc.mean():+.1f} bp/trade   "
                      f"C-B {diff.mean():+.1f}   paired t {t:+.2f}")
                if len(diff) < 12:
                    print(f"     *** {len(diff)} pairs is an observation, not a result. "
                          f"§3dq's prior is B by t +2.16. ***")
        m.shutdown() if hasattr(m, "shutdown") else None

    if allrows:
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(allrows).to_csv(OUT / "ab_zigzag_bc.csv", index=False)
        print(f"\nwrote {OUT / 'ab_zigzag_bc.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
