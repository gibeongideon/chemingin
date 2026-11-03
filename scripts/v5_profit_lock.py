#!/usr/bin/env python3
"""Lock a FRACTION of the peak profit — "at +200 move the stop to +100" — measured properly.

THE ASK, from a live observation. Position 360601 sat at **+$220 unrealised with only +$50
locked**, because the 3.0-ATR trail is ~$384 wide at that size. The proposal: as profit grows,
drag the stop up proportionally, e.g. hold half the peak.

WHY THIS IS NOT THE SAME QUESTION §3bn ALREADY ANSWERED. §3bn swept the trail DISTANCE in ATR
and found 3.0 the only non-negative cell (+2.22bp/fire against -4.15 at 1.0 and -2.31 at 2.0).
Every cell there is a trail whose width is set by VOLATILITY and is constant through the trade.
A fractional lock is a different family: its width scales with ACCUMULATED PROFIT, so it is
loose early and tightens as the trade works. The two rules cross at peak = 2 x trail width —
below that the fractional lock is TIGHTER, above it looser. That family has not been measured
here, so it gets measured rather than argued about.

HONEST PRIOR, recorded before the run. §3bn says tighter trails lose and §3co says lifting the
stop early costs 6.36bp/trade at t -3.55, both for the same reason: barriers truncate the few
large winners that pay for the many losers. A fractional lock truncates MORE in exactly the
region where most trades live. Expect it to lose, and expect it to look good on win rate while
doing so.

THE EVENT SET IS HELD FIXED (§3co). Every variant takes the same entries on the same days; only
the exit rule differs. Without that, a tighter rule exits sooner, frees capital, takes more
trades, and the comparison measures exposure instead of the exit.

INTRABAR CONVENTION IS ADVERSE-FIRST: within a bar the stop is tested against the LOW before the
high is allowed to ratchet, because the order of the two is unknowable at this resolution.

    python scripts/v5_profit_lock.py --port 18815 --symbol XAUUSDs
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_breakeven_move import (  # noqa: E402
    INIT_STOP_ATR, MAX_HOLD_DAYS, RT_BP, TRAIL_ATR, daily, stats, wilder_atr,
)

OUT = ROOT / "data" / "v5_runs"


def simulate(d: pd.DataFrame, lock_frac: float | None, trigger_atr: float = 0.0,
             bars_per_day: int = 96) -> pd.DataFrame:
    """Long-only break of the prior-day high, 3.0-ATR ratchet, plus an optional fractional lock.

    `lock_frac`   None = the §3bn control, trail only.
                  f    = once peak profit exceeds `trigger_atr`*ATR, the stop may never sit below
                         entry + f*(peak - entry). The ATR trail still applies; the stop is the
                         HIGHER of the two, and never loosens.
    """
    day = d.index.normalize()
    g = d.groupby(day)
    pdh = g["high"].max().shift(1).reindex(day).to_numpy(float)
    atr = wilder_atr(g["high"].max(), g["low"].min(),
                     g["close"].last()).shift(1).reindex(day).to_numpy(float)
    h, l, c = (d[k].to_numpy(float) for k in ("high", "low", "close"))
    nd = day.to_numpy()

    trades, i, n = [], 1, len(d)
    while i < n:
        if not (np.isfinite(pdh[i]) and np.isfinite(atr[i]) and atr[i] > 0) or h[i] <= pdh[i]:
            i += 1
            continue
        entry, a0 = pdh[i], atr[i]
        stop, hwm = entry - INIT_STOP_ATR * a0, entry
        limit = min(n - 1, i + MAX_HOLD_DAYS * bars_per_day)
        j, exit_px, why = i, np.nan, "open"
        while j <= limit:
            if l[j] <= stop:                       # adverse-first
                exit_px, why = stop, "stop"
                break
            if h[j] > hwm:
                hwm = h[j]
            cand = hwm - TRAIL_ATR * a0
            if lock_frac is not None and (hwm - entry) >= trigger_atr * a0:
                cand = max(cand, entry + lock_frac * (hwm - entry))
            if cand > stop:                        # ratchet only
                stop = cand
            j += 1
        if not np.isfinite(exit_px):
            exit_px, why = c[min(j, n - 1)], "max_hold"
        trades.append(dict(entry_time=d.index[i], entry=entry, exit=exit_px, why=why,
                           bars=j - i, ret_bp=(exit_px / entry - 1.0) * 1e4 - RT_BP))
        i = int(np.searchsorted(nd, nd[i], side="right"))       # SAME EVENT SET, always
    return pd.DataFrame(trades)


# *** B/C/D HAVE NO TRIGGER AND ARE THEREFORE DEGENERATE, WHICH IS WORTH SEEING. ***
# With trigger 0 the lock is live from the first bar: hwm starts AT entry, so the stop is lifted
# to entry + f*(hwm-entry) >= entry immediately. They are not "lock half the profit", they are
# "move to breakeven on bar one and then some", i.e. §3co's refuted rule in a stronger form. The
# faithful readings of the request are E-H, which only arm once the profit is real.
#
# H IS THE USER'S ACTUAL NUMBERS: +$200 on 0.08 lots of XAUUSDs is 25 price points, and daily
# ATR was ~16, so the trigger they described is ~1.5 ATR with a 50% lock.
VARIANTS = [
    ("A  trail only, 3.0 ATR   [§3bn control]", None, 0.0),
    ("B  lock 25% of peak, no trigger", 0.25, 0.0),
    ("C  lock 50% of peak, no trigger", 0.50, 0.0),
    ("D  lock 75% of peak, no trigger", 0.75, 0.0),
    ("E  lock 50%, only after +1 ATR", 0.50, 1.0),
    ("H  lock 50% after +1.5 ATR  <- the ask", 0.50, 1.5),
    ("F  lock 50%, only after +2 ATR", 0.50, 2.0),
    ("G  lock 33%, only after +2 ATR", 0.33, 2.0),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18815)
    ap.add_argument("--symbol", default="XAUUSDs")
    ap.add_argument("--bars", type=int, default=80_000)
    a = ap.parse_args()

    from mt5linux import MetaTrader5
    m, last = None, None
    for i in range(3):
        if i:
            time.sleep(3 * i)
        m = MetaTrader5(host="localhost", port=a.port)
        if m.initialize():
            break
        last = m.last_error()
    else:
        print("bridge init failed:", last); return 2
    m.symbol_select(a.symbol, True)
    r = m.copy_rates_from_pos(a.symbol, 15, 0, a.bars)
    m.shutdown()
    d = pd.DataFrame([(x[0], x[1], x[2], x[3], x[4]) for x in r],
                     columns=["ts", "open", "high", "low", "close"])
    d["time"] = pd.to_datetime(d.ts, unit="s")
    d = d.set_index("time").drop(columns=["ts"]).iloc[:-1]
    print(f"{len(d):,} M15 bars  {d.index[0]:%Y-%m-%d} .. {d.index[-1]:%Y-%m-%d}")
    print(f"entry: long-only break of the prior-day high · initial stop {INIT_STOP_ATR} ATR · "
          f"trail {TRAIL_ATR} ATR · max hold {MAX_HOLD_DAYS}d · cost {RT_BP:.3f}bp\n")

    idx = pd.DatetimeIndex(sorted(set(d.index.normalize())))
    ppy = 252.0
    base = None
    print(f"{'variant':<42}{'n':>5}{'win%':>7}{'bp/trade':>10}{'total %':>9}"
          f"{'Sharpe':>8}{'maxDD':>8}{'paired t':>10}")
    rows = []
    for name, f, trig in VARIANTS:
        tr = simulate(d, f, trig)
        if tr.empty:
            continue
        r_d = daily(tr, idx)
        st = stats(r_d, ppy)
        if base is None:
            base, base_tr = r_d, tr
            t = np.nan
        else:
            diff = (r_d - base).dropna()
            t = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))) \
                if diff.std(ddof=1) > 0 else np.nan
        rows.append((name, len(tr), float((tr.ret_bp > 0).mean()), float(tr.ret_bp.mean()),
                     float((1 + r_d).prod() - 1) * 100, st["sharpe"], st["maxdd"], t))
        print(f"{name:<42}{len(tr):>5}{(tr.ret_bp > 0).mean():>7.1%}{tr.ret_bp.mean():>10.2f}"
              f"{((1 + r_d).prod() - 1) * 100:>9.1f}{st['sharpe']:>8.2f}{st['maxdd']:>8.1%}"
              f"{t:>10.2f}")

    print(f"\nevent set held fixed: every variant takes {rows[0][1]} trades on the same days,")
    print("so the paired t measures the EXIT RULE and nothing else (§3co).")
    best = max(rows[1:], key=lambda r: r[3]) if len(rows) > 1 else None
    if best:
        print(f"\nbest non-control by bp/trade: {best[0].strip()} at {best[3]:+.2f}bp "
              f"against the control's {rows[0][3]:+.2f}bp, paired t {best[7]:+.2f}")
    print("\nWATCH THE WIN RATE COLUMN. A profit lock raises it by construction — it converts")
    print("open profit into closed profit more often — and that is not the same as making money.")
    print("WATCH SHARPE AND maxDD TOO, FOR THE OPPOSITE REASON. A tight lock makes the curve")
    print("smooth and small: Sharpe RISES while total return collapses, which is why these are")
    print("judged on the paired t at a fixed event set and not on Sharpe (§3bm, matched vol).")
    print("\nTHE ABSOLUTE COLUMNS ARE NOT A BACKTEST. `same_events` lets holds overlap so the")
    print("daily series double-counts; only the PAIRED comparison between rows is meaningful.")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["variant", "n", "win", "bp_trade", "total_pct", "sharpe",
                                "maxdd", "paired_t"]).to_csv(OUT / "profit_lock.csv", index=False)
    print(f"\nwrote {OUT / 'profit_lock.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
