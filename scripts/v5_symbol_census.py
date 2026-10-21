#!/usr/bin/env python3
"""Full-universe carry/spread census for an MT5 broker. READ-ONLY: symbol_info + ticks, no orders.

WHY: financing is the only lever that has ever paid in this project. 3q measured the book's swap
drag at -3.79%/yr (worth -0.29 Sharpe) and concluded "the only untested way to move it is
EXECUTION, not signal research"; 3ag then found GoldEternal by measuring Maven's live carry and
called it "worth more than every signal result in this repo's history". But that census covered
MAVEN'S 68 SYMBOLS and concluded "there is no swap-free index, crypto or energy". Switch Markets
exposes 2,067 symbols and NOBODY HAS EVER SCANNED THEM -- there was no full-universe sweeper in the
repo at all, only an XAU-filtered spread probe and a per-symbol spec printer taking an explicit
--symbols list.

Spot checks already contradict the Maven conclusion at this venue: BRENTs -0.09%/yr and
WTIs -0.06%/yr are effectively swap-free ENERGY, and JPN225s -0.24%/yr an effectively swap-free
INDEX. This script finds the rest.

TWO TRAPS, handled in src/core/swap_modes.py rather than here:
  * modes 2/3/4/7/8 are NOT convertible to %/yr from (point, contract, price) -- reported UNKNOWN,
    never a number. The previous converter silently ran them through the POINTS formula.
  * mode 0 (DISABLED) can still carry populated swap fields -- Maven's GoldEternal read
    swap_mode=0 with swap_long=-30.0. So raw fields are reported BESIDE the mode, never collapsed.

MIN_NOTIONAL IS REPORTED because a zero-carry instrument you cannot size is useless: JPN225s has
contract 10 and volume_min 0.10, i.e. a ~$64.8k minimum on a $100k account.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.swap_modes import carry_pct_yr, mode_name   # noqa: E402

FIELDS = ["name", "path", "swap_mode", "swap_mode_name", "swap_long", "swap_short",
          "carry_pct_yr", "spread_raw", "spread_bp", "bid", "ask", "trade_contract_size",
          "volume_min", "volume_step", "min_notional", "digits", "trade_mode", "filling_mode"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18815)
    ap.add_argument("--out", default=str(ROOT / "data/v5_runs/symbol_census/switch_census.csv"))
    ap.add_argument("--batch", type=int, default=40, help="symbols to subscribe per settle cycle")
    ap.add_argument("--settle", type=float, default=2.5, help="seconds to wait after subscribing")
    ap.add_argument("--limit", type=int, default=0, help="stop after N symbols (0 = all)")
    a = ap.parse_args()

    from mt5linux import MetaTrader5
    m = MetaTrader5(host="localhost", port=a.port)
    if not m.initialize():
        print("init failed: %s" % (m.last_error(),)); return 2
    ai = m.account_info()
    print("connected: account %s on %s" % (ai.login, ai.server))

    names = sorted(s.name for s in m.symbols_get())
    if a.limit:
        names = names[:a.limit]
    print("universe: %d symbols; batch %d, settle %.1fs" % (len(names), a.batch, a.settle))

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    t0 = time.time()
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for i in range(0, len(names), a.batch):
            chunk = names[i:i + a.batch]
            for n in chunk:
                try:
                    m.symbol_select(n, True)
                except Exception:
                    pass
            time.sleep(a.settle)                       # ticks do not populate until subscribed
            for n in chunk:
                try:
                    si = m.symbol_info(n)
                except Exception:
                    si = None
                if si is None:
                    continue
                try:
                    t = m.symbol_info_tick(n)
                except Exception:
                    t = None
                bid = float(getattr(t, "bid", 0) or getattr(si, "bid", 0) or 0.0)
                ask = float(getattr(t, "ask", 0) or getattr(si, "ask", 0) or 0.0)
                mid = (bid + ask) / 2.0 if (bid and ask) else (ask or bid or 0.0)
                sp_raw = (ask - bid) if (bid and ask) else float("nan")
                sp_bp = (sp_raw / mid * 1e4) if mid else float("nan")
                cy = carry_pct_yr(si, mid) if mid else None
                cs = float(getattr(si, "trade_contract_size", 0) or 0.0)
                vmin = float(getattr(si, "volume_min", 0) or 0.0)
                w.writerow(dict(
                    name=n, path=getattr(si, "path", ""),
                    swap_mode=getattr(si, "swap_mode", ""),
                    swap_mode_name=mode_name(getattr(si, "swap_mode", None)),
                    swap_long=getattr(si, "swap_long", ""),
                    swap_short=getattr(si, "swap_short", ""),
                    carry_pct_yr=("" if cy is None else round(cy, 4)),
                    spread_raw=("" if sp_raw != sp_raw else round(sp_raw, 6)),
                    spread_bp=("" if sp_bp != sp_bp else round(sp_bp, 4)),
                    bid=bid, ask=ask, trade_contract_size=cs,
                    volume_min=vmin, volume_step=getattr(si, "volume_step", ""),
                    min_notional=round(vmin * cs * mid, 2) if mid else "",
                    digits=getattr(si, "digits", ""),
                    trade_mode=getattr(si, "trade_mode", ""),
                    filling_mode=getattr(si, "filling_mode", "")))
                done += 1
            fh.flush()
            el = time.time() - t0
            print("  %5d/%d  %.0fs elapsed, ~%.0fs left"
                  % (done, len(names), el, el / max(done, 1) * (len(names) - done)), flush=True)
    m.shutdown()
    print("wrote %s (%d rows)" % (out, done))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
