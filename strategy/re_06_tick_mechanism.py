"""STAGE 6 — the last path to genuine replication, tested on real tick data.

WHY THIS SHAPE OF TEST. Stage 5 proved the vendor's edge is in ENTRY TIMING, not in the
momentum direction rule (their entries: +$19.39/trade gross at 30min, t=+2.12; the
recovered momentum rule: +$0.22 gross against a $4.40 spread). Stage 4 proved the timing
is invisible in M15 bars. The only remaining hypothesis is a TICK-LEVEL trigger.

The literal test — pull ticks for the vendor's own window — is IMPOSSIBLE: measured tick
retention on this broker is ~1 month (1,469 ticks yesterday / 1,522 at 1 week / 904 at
1 month / ZERO at 3, 6, 9 and 11 months, using a download path proven to work on the
available dates). The vendor window is 5-11 months old. No reachable feed has it.

So this tests the MECHANISM instead, on tick data that does exist (last ~4 weeks): if a
momentum scalper's plausible tick trigger — a fast directional burst — genuinely predicts a
short-horizon move, then at the vendor's trade frequency it should produce something near
their +1.94 price points per trade at 30 minutes. If it produces ~zero, then their edge is
not a market property recoverable from ticks on a normal retail feed, which would point to
broker-specific latency or to n=773/t=2.12 having been tail luck.

Runs ON the VPS (ticks stay there; only aggregates come back).

Trigger family: mid moved >= k x sigma over the last W seconds, direction = sign of the
move (buying strength / selling weakness, matching the fingerprint), with a cooldown so
trade frequency and one-at-a-time behaviour match the log.
"""
from __future__ import annotations

import datetime as dt
import sys

import numpy as np

sys.path.insert(0, "/home/trader/MT5")
from src.core.mt5_connector import get_mt5  # noqa: E402

SYM = "XAUUSD"
ACTIVE_H = range(1, 17)          # vendor hours 03-18 minus the +2h offset
HORIZONS = (30, 300, 1800)       # seconds: +30s, +5min, +30min
VENDOR_30M_GROSS = 1.94          # price points/trade the vendor's entries earned at 30min


def day_ticks(mt5, day: dt.date):
    out = []
    for h in ACTIVE_H:
        t0 = dt.datetime(day.year, day.month, day.day, h, 0)
        tk = mt5.copy_ticks_range(SYM, t0, t0 + dt.timedelta(hours=1), mt5.COPY_TICKS_ALL)
        if tk is not None and len(tk):
            out.append(np.array([(t["time_msc"], t["bid"], t["ask"]) for t in tk],
                                dtype=np.float64))
    if not out:
        return None
    a = np.vstack(out)
    return a[np.argsort(a[:, 0])]


def run_day(a: np.ndarray, W: int, k: float, cooldown_s: int):
    """Detect bursts and measure forward mid moves. Returns list of
    (direction, spread_at_entry, fwd_move_h1, fwd_h2, fwd_h3)."""
    ts = a[:, 0] / 1000.0
    mid = (a[:, 1] + a[:, 2]) / 2.0
    spr = a[:, 2] - a[:, 1]
    n = len(ts)
    # index of the tick W seconds before each tick, and at each forward horizon
    back = np.searchsorted(ts, ts - W, side="left")
    fwd = [np.searchsorted(ts, ts + h, side="left") for h in HORIZONS]
    move_w = mid - mid[np.clip(back, 0, n - 1)]
    # sigma of the W-second move over a trailing FIXED-COUNT window, computed with a
    # cumsum identity. A time-based window with np.std() inside a per-tick loop is
    # O(n x window) — hundreds of millions of ops per day — and does not finish.
    LB = 2000
    c1 = np.concatenate(([0.0], np.cumsum(move_w)))
    c2 = np.concatenate(([0.0], np.cumsum(move_w ** 2)))
    i_ar = np.arange(n)
    j_ar = np.maximum(i_ar - LB, 0)
    cnt = (i_ar - j_ar).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        s1 = c1[i_ar] - c1[j_ar]
        s2 = c2[i_ar] - c2[j_ar]
        var = s2 / cnt - (s1 / cnt) ** 2
        sig = np.sqrt(np.maximum(var, 0.0))
    sig[cnt < 200] = np.nan
    events = []
    last_t = -1e18
    for i in range(n):
        if not np.isfinite(sig[i]) or sig[i] <= 0:
            continue
        if ts[i] - last_t < cooldown_s:
            continue
        if abs(move_w[i]) < k * sig[i]:
            continue
        d = 1.0 if move_w[i] > 0 else -1.0
        row = [d, spr[i]]
        okrow = True
        for f in fwd:
            j = f[i]
            if j >= n:
                okrow = False
                break
            row.append((mid[j] - mid[i]) * d)
        if okrow:
            events.append(row)
            last_t = ts[i]
    return events


def main() -> None:
    mt5 = get_mt5(port=18812)
    mt5.initialize(timeout=20000)
    mt5.symbol_select(SYM, True)

    end = dt.date(2026, 9, 2)
    days = [end - dt.timedelta(days=i) for i in range(28)]
    days = [d for d in days if d.weekday() < 5]
    cache = {}
    for d in days:
        a = day_ticks(mt5, d)
        if a is not None and len(a) > 5000:
            cache[d] = a
    tot_ticks = sum(len(v) for v in cache.values())
    print(f"tick data loaded: {len(cache)} trading days, {tot_ticks:,} ticks "
          f"({min(cache)} .. {max(cache)})\n")
    if not cache:
        print("no tick data available — cannot run the mechanism test")
        return

    print("=== DOES A TICK-LEVEL MOMENTUM BURST PREDICT THE VENDOR'S EDGE? ===")
    print(f"vendor benchmark: +{VENDOR_30M_GROSS:.2f} price pts/trade gross at 30min "
          f"(t=+2.12), ~4.3 trades/day\n")
    print(f"{'W(s)':>5} {'k':>5} {'n':>6} {'trades/day':>11} {'spread':>7} "
          f"{'+30s':>8} {'+5min':>8} {'+30min':>9} {'t(30min)':>9} {'net30m':>8}")
    results = []
    for W in (10, 30, 60):
        for k in (2.0, 3.0, 4.0, 5.0):
            ev = []
            for d, a in cache.items():
                ev += run_day(a, W, k, cooldown_s=900)
            if len(ev) < 20:
                print(f"{W:>5} {k:>5.1f} {len(ev):>6}   (too few events)")
                continue
            E = np.array(ev)
            per_day = len(E) / len(cache)
            f30, f5m, f30m = E[:, 2], E[:, 3], E[:, 4]
            t30 = f30m.mean() / (f30m.std() / np.sqrt(len(f30m))) if f30m.std() else np.nan
            net = f30m.mean() - E[:, 1].mean()      # one spread per round trip
            results.append((W, k, len(E), per_day, f30m.mean(), t30, net))
            print(f"{W:>5} {k:>5.1f} {len(E):>6} {per_day:>11.2f} {E[:,1].mean():>7.3f} "
                  f"{f30.mean():>+8.3f} {f5m.mean():>+8.3f} {f30m.mean():>+9.3f} "
                  f"{t30:>+9.2f} {net:>+8.3f}")

    print("\n=== VERDICT ===")
    if results:
        best = max(results, key=lambda r: r[4])
        W, k, n, pd_, m30, t30, net = best
        print(f"best 30-min gross move: {m30:+.3f} pts (W={W}s k={k}, {pd_:.2f} trades/day, "
              f"t={t30:+.2f})")
        print(f"vendor's entries achieved: {VENDOR_30M_GROSS:+.3f} pts")
        print(f"ratio: {m30/VENDOR_30M_GROSS*100:.0f}% of the vendor's edge")
        print(f"net of spread, best config: {net:+.3f} pts/trade "
              f"({'PROFITABLE' if net > 0 else 'LOSS-MAKING'})")
    # frequency-matched view: which config sits nearest 4.3 trades/day?
    if results:
        fm = min(results, key=lambda r: abs(r[3] - 4.3))
        print(f"\nfrequency-matched to the vendor (~4.3/day): W={fm[0]}s k={fm[1]} "
              f"-> {fm[3]:.2f}/day, 30min gross {fm[4]:+.3f} pts, t={fm[5]:+.2f}, "
              f"net {fm[6]:+.3f}")


if __name__ == "__main__":
    main()
