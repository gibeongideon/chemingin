#!/usr/bin/env python3
"""Scale OUT: bank part of the position at a target, trail the rest at 3.0 ATR.

WHY THIS IS THE RIGHT REVISIT, AND SIZING WAS NOT. The complaint behind all of this is giving
back $170 of a $220 move. Two non-answers were eliminated first:
  * trail WIDTH — §3bn swept it; 3.0 ATR is the only non-negative cell.
  * fractional LOCK — §3cx; every cell loses, and the best of seven sat below its own null median.
  * position SIZE — §3ao measured Sharpe invariant at +1.02 across 0.25-4% risk. It is a pure
    dial: halving size halves the giveback AND halves the profit, so the ratio does not move.
    Suggesting it was a mistake; it cannot address this complaint by construction.

Scaling OUT is genuinely different from all three. A fractional lock tightens the stop on the
WHOLE position, so it truncates the runner. A partial exit banks a slice and leaves the remainder
on the full-width trail — the runner is still intact, just smaller. That is the one shape in this
family nobody here has measured. (§3bs measured scaling IN, which is leverage, not this.)

HONEST PRIOR. It should land between the control and the lock: strictly less harmful than
tightening the whole stop, still a truncation of the part it banks. Expect a small negative,
lower variance, and a higher win rate. If the per-trade EV is flat and the variance falls, that
is a RISK choice and should be stated as one, not sold as an edge.

COSTS. Scaling out does not cost an extra round trip. Entry pays half a spread on the full
position; the two exits pay half a spread on f and on (1-f), which sum to one position. Total is
the same RT_BP as a single exit, so the comparison is clean.

FILL CONVENTION. The partial is filled AT the trigger level (entry + k*ATR), not at the bar
close. The bar's high reached that level, so a resting limit order there is achievable — and
using the close instead would hand the variant a free look at where the bar ended.

EVENT SET HELD FIXED (§3co). Same 459 entries, same days, exit rule the only difference.

    python scripts/v5_partial_exit.py --port 18815 --symbol XAUUSDs
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


def simulate(d: pd.DataFrame, take_frac: float | None, take_atr: float = 2.0,
             move_to_be: bool = False, bars_per_day: int = 96) -> pd.DataFrame:
    """`take_frac` None = control. f = bank f of the position at entry + take_atr*ATR.

    `move_to_be` additionally lifts the stop to entry once the partial is banked — the common
    "risk-free runner" dressing. §3co refuted that move on a whole position; here it is tested on
    the REMAINDER, which is a different (smaller) bet.
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
        target = entry + take_atr * a0 if take_frac else np.inf
        banked, booked = 0.0, 0.0           # fraction already closed, and its return in bp
        limit = min(n - 1, i + MAX_HOLD_DAYS * bars_per_day)
        j, exit_px, why = i, np.nan, "open"
        while j <= limit:
            if l[j] <= stop:                # adverse-first: the stop is tested before the high
                exit_px, why = stop, "stop"
                break
            if take_frac and banked == 0.0 and h[j] >= target:
                banked = take_frac
                booked = (target / entry - 1.0) * 1e4 * take_frac
                if move_to_be and entry > stop:
                    stop = entry
            if h[j] > hwm:
                hwm = h[j]
            trail = hwm - TRAIL_ATR * a0
            if trail > stop:
                stop = trail
            j += 1
        if not np.isfinite(exit_px):
            exit_px, why = c[min(j, n - 1)], "max_hold"
        rest = (exit_px / entry - 1.0) * 1e4 * (1.0 - banked)
        trades.append(dict(entry_time=d.index[i], why=why, bars=j - i, banked=banked,
                           ret_bp=booked + rest - RT_BP))
        i = int(np.searchsorted(nd, nd[i], side="right"))          # SAME EVENT SET
    return pd.DataFrame(trades)


VARIANTS = [
    ("A  no partial, trail only  [control]", None, 0.0, False),
    ("B  bank 25% at +1 ATR", 0.25, 1.0, False),
    ("C  bank 50% at +1 ATR", 0.50, 1.0, False),
    ("D  bank 25% at +2 ATR", 0.25, 2.0, False),
    ("E  bank 50% at +2 ATR", 0.50, 2.0, False),
    ("F  bank 50% at +3 ATR", 0.50, 3.0, False),
    ("G  bank 50% at +2 ATR, rest to BE", 0.50, 2.0, True),
    ("H  bank 33% at +1.5 ATR", 0.33, 1.5, False),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18815)
    ap.add_argument("--symbol", default="XAUUSDs")
    ap.add_argument("--bars", type=int, default=80_000)
    ap.add_argument("--draws", type=int, default=2000)
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
    print(f"entry: break of prior-day high · stop {INIT_STOP_ATR} ATR · trail {TRAIL_ATR} ATR · "
          f"cost {RT_BP:.3f}bp · partial filled AT the trigger level\n")

    idx = pd.DatetimeIndex(sorted(set(d.index.normalize())))
    print(f"{'variant':<40}{'n':>5}{'win%':>7}{'bp/trade':>10}{'sd':>8}{'Sharpe':>8}"
          f"{'maxDD':>8}{'paired t':>10}")
    base, rows, diffs = None, [], []
    for name, f, k, be in VARIANTS:
        tr = simulate(d, f, k, be)
        r_d = daily(tr, idx)
        st = stats(r_d, 252.0)
        if base is None:
            base, t = r_d, np.nan
        else:
            dif = (r_d - base).dropna()
            diffs.append(dif.to_numpy())
            t = float(dif.mean() / (dif.std(ddof=1) / np.sqrt(len(dif)))) \
                if dif.std(ddof=1) > 0 else np.nan
        rows.append((name, len(tr), float((tr.ret_bp > 0).mean()), float(tr.ret_bp.mean()),
                     float(tr.ret_bp.std(ddof=1)), st["sharpe"], st["maxdd"], t))
        print(f"{name:<40}{len(tr):>5}{(tr.ret_bp > 0).mean():>7.1%}{tr.ret_bp.mean():>10.2f}"
              f"{tr.ret_bp.std(ddof=1):>8.1f}{st['sharpe']:>8.2f}{st['maxdd']:>8.1%}{t:>10.2f}")

    # ---- best-of-K floor, identical treatment to §3cx ----------------------------------------
    rng = np.random.default_rng(0)
    L = 20
    nullmax = np.empty(a.draws)
    for q in range(a.draws):
        nb = int(np.ceil(len(diffs[0]) / L))
        signs = np.repeat(rng.choice([-1.0, 1.0], nb), L)[:len(diffs[0])]
        mx = -np.inf
        for v in diffs:
            x = v * signs
            s = x.std(ddof=1)
            if s > 0:
                mx = max(mx, float(x.mean() / (s / np.sqrt(len(x)))))
        nullmax[q] = mx
    obs = max(r[7] for r in rows[1:] if r[7] == r[7])
    print(f"\nbest-of-K floor, K={len(diffs)}, {a.draws} block sign-flip draws (block {L}d):")
    print(f"   null max paired t:  median {np.median(nullmax):+.2f}   "
          f"95th pct {np.quantile(nullmax, 0.95):+.2f}")
    print(f"   observed max:       {obs:+.2f}   p = {float((nullmax >= obs).mean()):.3f}")
    verdict = ("CLEARS" if obs > np.quantile(nullmax, 0.95) else "does NOT clear")
    print(f"   -> the best variant {verdict} the floor.")
    print("\nIf EV is flat and `sd` falls, that is a RISK choice, not an edge. Say so plainly.")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["variant", "n", "win", "bp_trade", "sd", "sharpe", "maxdd",
                                "paired_t"]).to_csv(OUT / "partial_exit.csv", index=False)
    print(f"\nwrote {OUT / 'partial_exit.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
