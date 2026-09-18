"""XAUUSD RANGE / EXTREMES study — the daily and weekly high-low box, not returns.

WHY THIS IS A NEW FRAMING FOR THIS REPO. Every prior study here fed the model RETURNS or
z-scored windows: `features()` (z-scores, RSI, ATR-normalised momentum, wick ratios),
EWMAC/breakout forecasts, `vol_features()`. None of them asked the level-based question the
user raised: **where is price inside the range that today and this week have already
established, how much of the typical range is left, and which extreme gets taken out next?**

The user's own observation is the hypothesis under test: *"even in a day the bottom and top
seem to be almost the same."* If the daily high-low span is genuinely stable, then knowing how
much of it has already been spent is information about what is left — a conditional statement
about the REMAINING move, which is not what a trend forecast gives you.

This part measures. It does not predict, size or trade anything, and deliberately reports the
numbers that would KILL the idea as prominently as the ones that support it.

SECTIONS
  1. Is the range actually stable? CV, by year, in dollars and in percent.
  2. Is tomorrow's range predictable from today's? (autocorrelation + R^2 of the obvious
     predictors). This is the part that decides whether "stable" is exploitable.
  3. WHEN do the daily high and low form? Only intraday bars can answer this, and it is the
     direct test of the user's observation.
  4. Up days vs down days: counts, magnitude, asymmetry, by year.
  5. The weekly box: range, and which weekday holds the weekly extreme.
  6. Trend segmentation of the last year, measured on EXTREMES (higher-highs/higher-lows)
     rather than a moving average.
  7. Close position inside the day's range, and whether it says anything about tomorrow.

    python scripts/v5_range_study.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
P = ROOT / "data" / "v5_runs" / "range_study"


def load(tf: str) -> pd.DataFrame:
    d = pd.read_csv(P / f"XAUUSD_{tf}.csv", parse_dates=["time"]).set_index("time")
    return d[~d.index.duplicated(keep="last")].sort_index()


def sect(n: int, title: str) -> None:
    print(f"\n{'='*78}\n{n}. {title}\n{'='*78}")


def main() -> None:
    d1, m15, w1 = load("D1"), load("M15"), load("W1")
    d1["rng"] = d1["high"] - d1["low"]
    d1["rng_pct"] = d1["rng"] / d1["close"] * 100
    d1["ret"] = d1["close"].pct_change() * 100
    d1["body"] = (d1["close"] - d1["open"]).abs()
    d1["yr"] = d1.index.year

    # ---------------------------------------------------------------- 1
    sect(1, "IS THE DAILY RANGE ACTUALLY STABLE?  (the user's hypothesis)")
    print("CV = coefficient of variation = sd/mean. CV near 0 would mean 'almost the same'.")
    print(f"\n{'year':>6s} {'days':>5s} {'mean $':>8s} {'med $':>8s} {'sd $':>7s} "
          f"{'CV':>6s} {'mean %':>7s} {'CV %':>6s} {'p10 $':>8s} {'p90 $':>8s} {'p90/p10':>8s}")
    for yr, g in d1.groupby("yr"):
        if len(g) < 30:
            continue
        r, rp = g["rng"], g["rng_pct"]
        print(f"{yr:6d} {len(g):5d} {r.mean():8.2f} {r.median():8.2f} {r.std():7.2f} "
              f"{r.std()/r.mean():6.2f} {rp.mean():7.2f} {rp.std()/rp.mean():6.2f} "
              f"{r.quantile(.10):8.2f} {r.quantile(.90):8.2f} "
              f"{r.quantile(.90)/r.quantile(.10):8.2f}")
    last1 = d1.tail(252)
    print(f"\nLAST 252 SESSIONS (your 1-year window):")
    print(f"  range  mean ${last1['rng'].mean():.2f}  median ${last1['rng'].median():.2f}  "
          f"sd ${last1['rng'].std():.2f}  CV {last1['rng'].std()/last1['rng'].mean():.2f}")
    print(f"  as %%   mean {last1['rng_pct'].mean():.2f}%  "
          f"CV {last1['rng_pct'].std()/last1['rng_pct'].mean():.2f}")
    print(f"  p10 ${last1['rng'].quantile(.10):.2f} .. p90 ${last1['rng'].quantile(.90):.2f} "
          f"-> the widest decile is {last1['rng'].quantile(.90)/last1['rng'].quantile(.10):.1f}x "
          f"the narrowest")
    print("\nVERDICT LINE: a CV around 0.4-0.5 means the range is NOT 'almost the same' day to")
    print("day in dollars — it is a wide distribution. In PERCENT it is tighter, which is the")
    print("real content of the observation: the range scales with price, not with the calendar.")

    # ---------------------------------------------------------------- 2
    sect(2, "IS TOMORROW'S RANGE PREDICTABLE?  (stable != exploitable)")
    r = d1["rng_pct"].dropna()
    print(f"{'lag':>4s} {'autocorr':>9s}")
    for k in (1, 2, 3, 5, 10, 20):
        print(f"{k:4d} {r.autocorr(k):9.3f}")
    # obvious predictors of tomorrow's range
    X = pd.DataFrame({
        "today": d1["rng_pct"],
        "ma5": d1["rng_pct"].rolling(5).mean(),
        "ma20": d1["rng_pct"].rolling(20).mean(),
        "ma60": d1["rng_pct"].rolling(60).mean(),
    })
    y = d1["rng_pct"].shift(-1)
    for nm in X.columns:
        z = pd.DataFrame({"x": X[nm], "y": y}).dropna()
        c = z["x"].corr(z["y"])
        print(f"  corr({nm:5s}, tomorrow's range%) = {c:+.3f}   R^2 = {c*c:.3f}  n={len(z)}")
    print("\nCompare with DIRECTION, the thing everyone actually wants:")
    yd = np.sign(d1['ret'].shift(-1))
    zz = pd.DataFrame({"x": d1['ret'], "y": yd}).dropna()
    print(f"  corr(today's return, sign of tomorrow's) = {zz['x'].corr(zz['y']):+.3f}")
    print("  -> RANGE is strongly predictable, DIRECTION is not. That asymmetry is the whole")
    print("     reason a range framing is worth a look, and also its central limitation:")
    print("     a range forecast alone is direction-free and cannot be traded directionally.")

    # ---------------------------------------------------------------- 3
    sect(3, "WHEN DO THE DAILY HIGH AND LOW FORM?  (needs intraday bars)")
    m = m15.copy()
    m["day"] = m.index.normalize()
    m["hour"] = m.index.hour
    g = m.groupby("day")
    hi_h = g["high"].idxmax().dt.hour
    lo_h = g["low"].idxmin().dt.hour
    n_days = len(hi_h)
    print(f"{n_days} sessions of M15 ({m.index[0].date()} -> {m.index[-1].date()})")
    print(f"\n{'hour':>5s} {'high forms':>11s} {'low forms':>10s}   (UTC, share of sessions)")
    for h in range(24):
        ph = (hi_h == h).mean(); pl = (lo_h == h).mean()
        if ph > 0.005 or pl > 0.005:
            bar_h = "#" * int(ph * 200); bar_l = "-" * int(pl * 200)
            print(f"{h:5d} {ph:11.3f} {pl:10.3f}   {bar_h}{bar_l}")
    same = (hi_h == lo_h).mean()
    print(f"\nhigh and low in the SAME hour: {same:.3f} of sessions")
    # how far apart in time
    gap = (g["high"].idxmax() - g["low"].idxmin()).dt.total_seconds().abs() / 3600
    print(f"hours between the high and the low: median {gap.median():.1f}h  "
          f"mean {gap.mean():.1f}h  p10 {gap.quantile(.1):.1f}h  p90 {gap.quantile(.9):.1f}h")
    print("\nThis is the direct test of 'the bottom and top seem almost the same'. If they")
    print("formed at the same time the range would be noise; a wide time gap means the day")
    print("has DIRECTIONAL structure -- one extreme early, the other late.")

    # ---------------------------------------------------------------- 4
    sect(4, "UP DAYS vs DOWN DAYS")
    print(f"{'year':>6s} {'n':>5s} {'up%':>6s} {'mean up':>8s} {'mean dn':>8s} "
          f"{'up/dn':>6s} {'sum up':>8s} {'sum dn':>8s}")
    for yr, gg in d1.groupby("yr"):
        if len(gg) < 30:
            continue
        up, dn = gg.loc[gg["ret"] > 0, "ret"], gg.loc[gg["ret"] < 0, "ret"]
        if not len(up) or not len(dn):
            continue
        print(f"{yr:6d} {len(gg):5d} {(gg['ret']>0).mean()*100:6.1f} {up.mean():8.2f} "
              f"{dn.mean():8.2f} {abs(up.mean()/dn.mean()):6.2f} {up.sum():8.1f} "
              f"{dn.sum():8.1f}")

    # ---------------------------------------------------------------- 5
    sect(5, "THE WEEKLY BOX")
    w = w1.copy()
    w["rng"] = w["high"] - w["low"]
    w["rng_pct"] = w["rng"] / w["close"] * 100
    wl = w.tail(52)
    print(f"last 52 weeks: range mean ${wl['rng'].mean():.2f} ({wl['rng_pct'].mean():.2f}%)  "
          f"CV {wl['rng'].std()/wl['rng'].mean():.2f}")
    print(f"weekly range autocorr lag1 {w['rng_pct'].autocorr(1):+.3f}")
    # daily range summed vs weekly range: how much of the week's travel is 'wasted'
    d1w = d1.copy(); d1w["wk"] = d1w.index.to_period("W")
    agg = d1w.groupby("wk").agg(sum_daily=("rng", "sum"), hi=("high", "max"),
                                lo=("low", "min"), n=("rng", "size"))
    agg = agg[agg["n"] >= 4]
    agg["wk_rng"] = agg["hi"] - agg["lo"]
    eff = (agg["wk_rng"] / agg["sum_daily"])
    print(f"\nweekly range / sum of daily ranges: median {eff.median():.3f}  "
          f"mean {eff.mean():.3f}  (n={len(agg)} weeks)")
    print("  1.0 would be a perfectly one-directional week; 0.2 means the week travelled 5x")
    print("  its net span. This ratio IS the trend-vs-chop measure, built from extremes.")
    print(f"  last 52 weeks median {eff.tail(52).median():.3f}")
    # which weekday holds the weekly extreme
    d1w["dow"] = d1w.index.dayofweek
    wk_hi_dow, wk_lo_dow = [], []
    for wk, gg in d1w.groupby("wk"):
        if len(gg) < 4:
            continue
        wk_hi_dow.append(gg["high"].idxmax().dayofweek)
        wk_lo_dow.append(gg["low"].idxmin().dayofweek)
    names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    print(f"\n{'day':>5s} {'holds wk HIGH':>14s} {'holds wk LOW':>13s}")
    for i, nm in enumerate(names):
        print(f"{nm:>5s} {np.mean(np.array(wk_hi_dow)==i):14.3f} "
              f"{np.mean(np.array(wk_lo_dow)==i):13.3f}")

    # ---------------------------------------------------------------- 6
    sect(6, "TREND SEGMENTATION FROM EXTREMES (last 1y and 3y)")
    for label, sub in (("last 252d", d1.tail(252)), ("last 756d", d1.tail(756))):
        hh = (sub["high"] > sub["high"].shift(1))
        ll = (sub["low"] < sub["low"].shift(1))
        up_struct = (hh & ~ll)      # higher high, higher low = clean up bar
        dn_struct = (ll & ~hh)
        inside = (~hh & ~ll)
        outside = (hh & ll)
        print(f"\n{label}: {len(sub)} sessions, "
              f"net {sub['close'].iloc[-1]/sub['close'].iloc[0]-1:+.1%}")
        print(f"  higher-high & higher-low (up structure) : {up_struct.mean():.3f}")
        print(f"  lower-low & lower-high  (down structure): {dn_struct.mean():.3f}")
        print(f"  inside bars (range contracts)           : {inside.mean():.3f}")
        print(f"  outside bars (range expands both ways)  : {outside.mean():.3f}")

    # ---------------------------------------------------------------- 7
    sect(7, "CLOSE POSITION INSIDE THE DAY'S RANGE -- does it predict tomorrow?")
    pos = ((d1["close"] - d1["low"]) / d1["rng"].replace(0, np.nan)).clip(0, 1)
    fwd = d1["ret"].shift(-1)
    z = pd.DataFrame({"pos": pos, "fwd": fwd}).dropna()
    print(f"n={len(z)}   corr(close position, tomorrow's return) = "
          f"{z['pos'].corr(z['fwd']):+.4f}")
    q = pd.qcut(z["pos"], 5, labels=["Q1 (near low)", "Q2", "Q3", "Q4", "Q5 (near high)"])
    t = z.groupby(q)["fwd"].agg(["mean", "std", "count"])
    t["se"] = t["std"] / np.sqrt(t["count"])
    t["t"] = t["mean"] / t["se"]
    print(t.to_string(float_format=lambda x: f"{x:+.4f}"))
    print("\nAnd on the LAST 252 SESSIONS only (recency, as asked):")
    zl = z.tail(252)
    ql = pd.qcut(zl["pos"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    tl = zl.groupby(ql)["fwd"].agg(["mean", "count"])
    print(tl.to_string(float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()
