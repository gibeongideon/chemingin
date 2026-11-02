#!/usr/bin/env python3
"""ONE signal out of all the TradingView frames — and a test of how to weight them.

THE QUESTION. The advisor prints a rating per frame (M15..MN1). The user asked for a single
generic signal built from all of them, "with weight on the short period". Whether short weighting
helps is not a matter of taste, so this measures it rather than asserting it.

THE REASON TO DOUBT IT UP FRONT, stated before the run so the result cannot be rationalised: the
per-frame economics already measured in the advisor run the OTHER way — M15 nets -0.46bp after
cost, M30 -0.70, H1 -0.63, while H4 is +1.33 and D1 +10.68. Weighting the short frames more
weights the losing frames more. The honest version of the request is therefore "find the best
weighting", with short-weighted as one pre-registered candidate among several.

*** THE ALIGNMENT TRAP THIS FILE IS BUILT AROUND. *** A D1 bar stamped 2026-10-30 00:00 is NOT
knowable at 00:00 — it is knowable at its CLOSE. Reading a coarse-frame indicator onto a fine
decision grid by plain reindex-and-ffill hands the decision a bar from its own future, which is
how `cross-timeframe-resample-is-lookahead` produced a fake AUC of 0.901 in this repo. Every
frame's rating here is stamped at bar_start + duration before it touches the decision grid, and
a lookahead control is run that must FAIL to see the future when the shift is correct.

CONTROLS. #1 cost at the live round trip. #2 every cell is reported against BUY-AND-HOLD over
the same horizon, never against zero. #4 episode-based SE — ratings persist for many bars, so
bar counts overstate n by the run length. #7 the candidate list is fixed before the run and the
best-of-K floor is printed. #8 EV/fire and mean offset, not AUC.

    python scripts/v5_tv_composite.py --horizon 24 --decide H1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_tv_ratings import votes  # noqa: E402

# frame -> (resample rule, duration). Duration is what the rating must be DELAYED by.
FRAMES = {
    "M15": ("15min", pd.Timedelta("15min")),
    "M30": ("30min", pd.Timedelta("30min")),
    "H1":  ("1h",    pd.Timedelta("1h")),
    "H2":  ("2h",    pd.Timedelta("2h")),
    "H4":  ("4h",    pd.Timedelta("4h")),
    "D1":  ("1D",    pd.Timedelta("1D")),
    "W1":  ("7D",    pd.Timedelta("7D")),
}

# PRE-REGISTERED WEIGHTINGS. Fixed before the run; K is printed with the result so the best cell
# is read against the right noise floor (CONTROL #7). `h` is the frame duration in hours.
SCHEMES = {
    "equal":        lambda h: 1.0,
    "short_1/h":    lambda h: 1.0 / h,              # the user's ask, strongest form
    "short_1/sqrt": lambda h: 1.0 / np.sqrt(h),     # the user's ask, gentler
    "long_sqrt":    lambda h: np.sqrt(h),
    "long_h":       lambda h: h,
}


def resample(m15: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "15min":
        return m15
    g = m15.groupby(m15.index.floor(rule))
    return pd.DataFrame(dict(open=g["open"].first(), high=g["high"].max(),
                             low=g["low"].min(), close=g["close"].last(),
                             tick_volume=g["tick_volume"].sum()))


def frame_rating(m15: pd.DataFrame, rule: str, dur: pd.Timedelta,
                 shift: bool = True) -> pd.Series:
    """Mean TradingView vote per bar, STAMPED AT THE BAR'S CLOSE.

    `shift=False` is the lookahead control: it stamps the rating at the bar's OPEN, which is the
    bug this guards against. That cell must score better than the honest one — if it does not,
    the harness is not sensitive enough for the comparison to mean anything.
    """
    d = resample(m15, rule)
    if len(d) < 250:
        return pd.Series(dtype=float)
    r = votes(d).mean(axis=1)
    return r.set_axis(r.index + dur) if shift else r


def episodes(sig: pd.Series) -> np.ndarray:
    """Contiguous runs of the same sign. SE must be counted in these, not in bars: a rating sits
    in one state for many bars and those bars are one observation, not many."""
    s = np.sign(sig.to_numpy())
    return np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]


def evaluate(score: pd.Series, fwd: pd.Series, base: float, rt_bp: float,
             thr: float = 0.0) -> dict:
    """LONG when score > thr. Everything is reported against HOLDING, never against zero."""
    fire = score > thr
    n = int(fire.sum())
    if n < 50:
        return dict(n=n, ev=np.nan, net=np.nan, vs_hold=np.nan, se=np.nan, h1=np.nan, h2=np.nan)
    r = fwd[fire]
    ev = float(r.mean())
    # Episode SE: one observation per contiguous run, not per bar.
    idx = episodes(fire.astype(int) * 2 - 1)
    runs = [float(fwd[fire].iloc[a:b].mean())
            for a, b in zip(idx, list(idx[1:]) + [len(fire)])
            if b > a and fire.iloc[a:b].any()]
    runs = [x for x in runs if x == x]
    se = float(np.std(runs, ddof=1) / np.sqrt(len(runs))) if len(runs) > 2 else np.nan
    half = len(r) // 2
    # *** THE DECIDING COLUMN. *** `net` (EV minus cost) flatters every cell, because the
    # alternative to firing is not sitting in cash — it is HOLDING, which earns `base` and pays
    # the round trip ONCE for the whole sample while a signal pays it on every fire. So the
    # number that decides anything is EV - base - cost. A cell can show a positive `net` and a
    # positive `vs_hold` and still be a worse way to own gold than owning gold.
    return dict(n=n, n_ep=len(runs), ev=ev, net=ev - rt_bp, vs_hold=ev - base,
                vs_hold_net=ev - base - rt_bp, se=se,
                h1=float(r.iloc[:half].mean()) - base, h2=float(r.iloc[half:].mean()) - base)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/XAUUSD_M15_long.csv")
    ap.add_argument("--decide", default="H1", help="decision grid")
    ap.add_argument("--horizon", type=int, default=24, help="hours held")
    ap.add_argument("--rt-bp", type=float, default=1.08, help="round-trip cost in bp (live)")
    a = ap.parse_args()

    m15 = pd.read_csv(ROOT / a.csv, parse_dates=["time"], index_col="time")
    m15 = m15[~m15.index.duplicated(keep="last")].sort_index()
    print(f"{len(m15):,} M15 bars  {m15.index[0]:%Y-%m-%d} .. {m15.index[-1]:%Y-%m-%d}")

    grid = resample(m15, FRAMES[a.decide][0])
    gi = grid.index + FRAMES[a.decide][1]              # decisions act at the bar's CLOSE
    close = grid["close"].set_axis(gi)
    step = max(1, int(a.horizon / (FRAMES[a.decide][1] / pd.Timedelta("1h"))))
    fwd = (close.shift(-step) / close - 1.0) * 1e4     # bp
    base = float(fwd.mean())
    print(f"decide {a.decide}, hold {a.horizon}h ({step} bars), "
          f"round trip {a.rt_bp:.2f}bp\nBUY-AND-HOLD over the same horizon: "
          f"{base:+.2f}bp — this is the number every cell below must beat (CONTROL #2)\n")

    rat, honest = {}, {}
    for f, (rule, dur) in FRAMES.items():
        s = frame_rating(m15, rule, dur)
        if s.empty:
            continue
        honest[f] = s
        rat[f] = s.reindex(close.index, method="ffill")
    print(f"frames available: {', '.join(rat)}\n")

    hrs = {f: FRAMES[f][1] / pd.Timedelta("1h") for f in rat}
    R = pd.DataFrame(rat).dropna()
    fwd_a = fwd.reindex(R.index)
    ok = fwd_a.notna()
    R, fwd_a = R[ok], fwd_a[ok]

    rows = []
    for f in R.columns:                                # single frames, as controls
        rows.append((f"solo {f}", evaluate(R[f], fwd_a, base, a.rt_bp)))
    for name, fn in SCHEMES.items():
        w = np.array([fn(hrs[f]) for f in R.columns], float)
        w /= w.sum()
        rows.append((name, evaluate(pd.Series(R.to_numpy() @ w, index=R.index),
                                    fwd_a, base, a.rt_bp)))

    print(f"{'cell':<14}{'n':>7}{'ep':>6}{'EV bp':>9}{'vs HOLD':>10}{'vs HOLD net':>12}"
          f"{'SE':>8}{'1st':>8}{'2nd':>8}")
    for name, d in rows:
        print(f"{name:<14}{d['n']:>7}{d.get('n_ep', 0):>6}{d['ev']:>9.2f}"
              f"{d['vs_hold']:>10.2f}{d['vs_hold_net']:>12.2f}"
              f"{d['se']:>8.2f}{d['h1']:>8.2f}{d['h2']:>8.2f}")

    k = len(rows)
    best = max((r for r in rows if r[1]["vs_hold"] == r[1]["vs_hold"]),
               key=lambda r: r[1]["vs_hold_net"], default=None)
    if best:
        d = best[1]
        floor = (d["se"] * np.sqrt(2 * np.log(max(k, 2)))) if d["se"] == d["se"] else np.nan
        print(f"\nbest vs HOLD NET: {best[0]} at {d['vs_hold_net']:+.2f}bp")
        print(f"best-of-K floor (K={k}): a max this large needs ~{floor:+.2f}bp to beat noise "
              f"(CONTROL #7)")
        print(f"halves (gross vs hold): {d['h1']:+.2f} / {d['h2']:+.2f} — both must be positive "
              f"to count")

    # ---- lookahead control: the SAME pipeline with the delay removed must score BETTER -------
    bad = {f: frame_rating(m15, FRAMES[f][0], FRAMES[f][1], shift=False)
           .reindex(close.index, method="ffill") for f in R.columns}
    Rb = pd.DataFrame(bad).reindex(R.index).dropna()
    wq = np.array([1.0 / hrs[f] for f in Rb.columns], float); wq /= wq.sum()
    cheat = evaluate(pd.Series(Rb.to_numpy() @ wq, index=Rb.index),
                     fwd_a.reindex(Rb.index), base, a.rt_bp)
    sw = [r for r in rows if r[0] == "short_1/h"][0][1]
    print(f"\nLOOKAHEAD CONTROL (short_1/h, delay removed): vs HOLD {cheat['vs_hold']:+.2f}bp "
          f"against the honest {sw['vs_hold']:+.2f}bp")
    print("  the cheating cell MUST score better; if it does not, this harness cannot resolve "
          "the difference\n  and no verdict above is trustworthy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
