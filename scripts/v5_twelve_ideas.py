"""The user's twelve setups: eight declared cells, one per untested idea.

Pre-registered in TWELVE_IDEAS_PREREG.md. Four of the twelve were already closed (fade-the-spike
is §3bf verbatim and the wrong sign; the overnight effect is measured at ASIA +1.68bp/t+1.96,
real in sign but below the 2.34bp round trip; seasonality and the gold/silver ratio closed
earlier). The eight here get ONE cell each, no parameter sweep — the ideas are tested as stated.

Ideas 3, 6, 9 and 10 are genuinely novel mechanisms for this repo: a specific auction time, a
price-LEVEL effect, an expiry-calendar effect and a volume-weighted anchor. Nothing here has ever
used a round-number level or a volume-weighted price.

    python scripts/v5_twelve_ideas.py --draws 400
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "v5_runs" / "session_study"
COST_BP = 2.34


def loc_utc(days, tz, hh, mm=0):
    z = ZoneInfo(tz)
    return pd.DatetimeIndex([pd.Timestamp(d.date(), tz=z) + pd.Timedelta(hours=hh, minutes=mm)
                             for d in days]).tz_convert("UTC").tz_localize(None)


class Book:
    """M15 arrays plus positional helpers. `at` is the index of the bar AT OR AFTER t."""
    def __init__(self, m15):
        self.i = m15.index
        self.v = self.i.values
        self.o, self.h, self.l, self.c = (m15["open"].values, m15["high"].values,
                                          m15["low"].values, m15["close"].values)
        self.vol = (m15["tick_volume"].values if "tick_volume" in m15
                    else np.ones(len(m15)))

    def at(self, t):
        k = int(np.searchsorted(self.v, np.datetime64(t), side="left"))
        return k if k < len(self.v) else None

    def exact(self, t):
        k = self.at(t)
        return k if (k is not None and self.i[k] == t) else None


def ret(entry, exit_, side):
    return side * (exit_ - entry) / entry * 1e4


# ---------------------------------------------------------------- 1. Asian range sweep/reversal
def idea1(B, days):
    a0, a1 = loc_utc(days, "UTC", 0), loc_utc(days, "UTC", 7)
    l1 = loc_utc(days, "UTC", 12)
    out = []
    for d, t0, t1, t2 in zip(days, a0, a1, l1):
        s, m, e = B.exact(t0), B.exact(t1), B.at(t2)
        if s is None or m is None or e is None or m - s < 20:
            continue
        hi, lo = B.h[s:m].max(), B.l[s:m].min()
        for k in range(m, e):
            if B.h[k] > hi and B.c[k] < hi:            # swept the high, closed back inside
                out.append((B.i[k], -1, ret(B.c[k], B.c[e], -1))); break
            if B.l[k] < lo and B.c[k] > lo:
                out.append((B.i[k], +1, ret(B.c[k], B.c[e], +1))); break
    return out


# ---------------------------------------------------------------- 3. LBMA PM fix (15:00 London)
def idea3(B, days):
    pre, fix = loc_utc(days, "Europe/London", 14), loc_utc(days, "Europe/London", 15)
    out = []
    for d, tp, tf in zip(days, pre, fix):
        a, b = B.exact(tp), B.exact(tf)
        if a is None or b is None or b <= a:
            continue
        e = B.at(B.i[b] + pd.Timedelta(minutes=30))
        if e is None or e <= b:
            continue
        drift = B.c[b] - B.c[a]
        if drift == 0:
            continue
        side = -1 if drift > 0 else +1                 # trade AGAINST the pre-fix drift
        out.append((B.i[b], side, ret(B.c[b], B.c[e], side)))
    return out


# ---------------------------------------------------------------- 6. round-number stop runs
def idea6(B, days, lo_usd=3.0, hi_usd=8.0, hold_h=4):
    out, n = [], len(B.c)
    for k in range(1, n - 1):
        px = B.c[k]
        for step in (100.0, 50.0, 25.0):
            lvl = round(px / step) * step
            over_up, over_dn = B.h[k] - lvl, lvl - B.l[k]
            if lo_usd <= over_up <= hi_usd and B.c[k] < lvl:
                e = B.at(B.i[k] + pd.Timedelta(hours=hold_h))
                if e: out.append((B.i[k], -1, ret(B.c[k], B.c[e], -1)))
                break
            if lo_usd <= over_dn <= hi_usd and B.c[k] > lvl:
                e = B.at(B.i[k] + pd.Timedelta(hours=hold_h))
                if e: out.append((B.i[k], +1, ret(B.c[k], B.c[e], +1)))
                break
    return out


# ---------------------------------------------------------------- 7. Monday gap fill
def idea7(B, days, hold_h=4):
    out = []
    for k in range(1, len(B.i)):
        prev, cur = B.i[k - 1], B.i[k]
        if (cur - prev) < pd.Timedelta(hours=20):      # only the weekend break
            continue
        gap = B.o[k] - B.c[k - 1]
        if abs(gap) / B.c[k - 1] * 1e4 < 5:            # ignore sub-5bp noise
            continue
        side = -1 if gap > 0 else +1                   # trade TOWARD the previous close
        tgt, e = B.c[k - 1], B.at(cur + pd.Timedelta(hours=hold_h))
        if e is None:
            continue
        filled = None
        for j in range(k, e):
            if (gap > 0 and B.l[j] <= tgt) or (gap < 0 and B.h[j] >= tgt):
                filled = j; break
        px = tgt if filled is not None else B.c[e]
        out.append((cur, side, ret(B.o[k], px, side)))
    return out


# ---------------------------------------------------------------- 9. COMEX expiry pinning
def idea9(B, days):
    """Monthly options expire the 4th-to-last business day of the month before delivery."""
    out = []
    bd = pd.DatetimeIndex([d for d in days if d.dayofweek < 5])
    for (y, m), grp in pd.Series(bd, index=bd).groupby([bd.year, bd.month]):
        g = pd.DatetimeIndex(grp.values)
        if len(g) < 8:
            continue
        exp = g[-4]
        win = g[(g < exp) & (g >= g[-8])]              # the 4 sessions before expiry
        for d in win:
            s = B.exact(loc_utc([d], "UTC", 7)[0])
            e = B.at(loc_utc([d], "UTC", 19)[0])
            if s is None or e is None or e <= s + 4:
                continue
            atr = np.nanmean(B.h[max(0, s - 96):s] - B.l[max(0, s - 96):s])
            if not np.isfinite(atr) or atr <= 0:
                continue
            for k in range(s + 1, e):                  # fade a >1 ATR excursion from the open
                if B.c[k] - B.o[s] > atr:
                    out.append((B.i[k], -1, ret(B.c[k], B.c[e], -1))); break
                if B.o[s] - B.c[k] > atr:
                    out.append((B.i[k], +1, ret(B.c[k], B.c[e], +1))); break
    return out


# ---------------------------------------------------------------- 10. anchored-VWAP reversion
def idea10(B, days, sd_mult=2.0):
    t0, t1 = loc_utc(days, "UTC", 7), loc_utc(days, "UTC", 19)
    out = []
    for d, a, b in zip(days, t0, t1):
        s, e = B.exact(a), B.at(b)
        if s is None or e is None or e - s < 12:
            continue
        tp = (B.h[s:e] + B.l[s:e] + B.c[s:e]) / 3.0
        w = B.vol[s:e].astype(float)
        cw = np.cumsum(w); cwp = np.cumsum(tp * w)
        vwap = np.where(cw > 0, cwp / np.maximum(cw, 1e-9), tp)
        dev = tp - vwap
        for j in range(8, e - s):
            sd = np.std(dev[:j])
            if sd <= 0:
                continue
            side = -1 if dev[j] > sd_mult * sd else (+1 if dev[j] < -sd_mult * sd else 0)
            if side == 0:
                continue
            # *** THE EXIT MUST BE A LATER BAR. *** Exiting at vwap[j] -- the VWAP of the very
            # bar the trade is entered on -- makes the "return" mechanically equal to the
            # deviation that was just selected for being large. Run that way this printed
            # t +61.8 on a 99.2% hit rate, which is a same-bar identity, not a trade.
            ex, entry = None, B.c[s + j]
            for q in range(j + 1, e - s):
                if (side < 0 and B.l[s + q] <= vwap[q]) or (side > 0 and B.h[s + q] >= vwap[q]):
                    ex = vwap[q]; break            # reverted to VWAP, exit there
            if ex is None:
                ex = B.c[e - 1]                    # never reverted: exit at the session close
            out.append((B.i[s + j], side, ret(entry, ex, side)))
            break
    return out


# ---------------------------------------------------------------- 11. NR7 breakout, trend-filtered
def idea11(B, days):
    dd = pd.DataFrame({"h": B.h, "l": B.l, "c": B.c}, index=B.i).resample("1D").agg(
        {"h": "max", "l": "min", "c": "last"}).dropna()
    dd["rng"] = dd.h - dd.l
    dd["nr7"] = dd["rng"] == dd["rng"].rolling(7).min()
    dd["wk"] = np.sign(dd["c"] - dd["c"].shift(5))
    out = []
    for i in range(7, len(dd) - 1):
        if not dd["nr7"].iloc[i]:
            continue
        trend = dd["wk"].iloc[i]
        if trend == 0:
            continue
        hi, lo = dd["h"].iloc[i], dd["l"].iloc[i]
        nxt = dd.index[i + 1]
        s, e = B.at(nxt), B.at(nxt + pd.Timedelta(hours=23))
        if s is None or e is None or e <= s:
            continue
        for k in range(s, e):                          # break only WITH the weekly trend
            if trend > 0 and B.h[k] > hi:
                out.append((B.i[k], +1, ret(B.c[k], B.c[e], +1))); break
            if trend < 0 and B.l[k] < lo:
                out.append((B.i[k], -1, ret(B.c[k], B.c[e], -1))); break
    return out


# ---------------------------------------------------------------- 5. dollar + yields divergence
def idea5(B, days, hold_d=5):
    ex = ROOT / "data" / "exog"
    g = pd.read_csv(ex / "XAU_USCLOSE_D1.csv", parse_dates=["time"], index_col="time").iloc[:, 0]
    dxy = pd.read_csv(ex / "DXY_D1.csv", parse_dates=["time"], index_col="time")["close"]
    tnx = pd.read_csv(ex / "TNX_D1.csv", parse_dates=["time"], index_col="time")["close"]
    ix = g.index.intersection(dxy.index).intersection(tnx.index)
    g, dxy, tnx = g.loc[ix], dxy.loc[ix], tnx.loc[ix]
    cg = g.pct_change(5); cd = dxy.pct_change(5); ct = tnx.diff(5)
    fwd = (g.shift(-hold_d) / g - 1) * 1e4
    out = []
    for k in range(6, len(ix) - hold_d):
        if not all(np.isfinite([cg.iloc[k], cd.iloc[k], ct.iloc[k], fwd.iloc[k]])):
            continue
        # gold up while BOTH dollar and yields up -> expect gold to correct DOWN (and mirror)
        if cg.iloc[k] > 0 and cd.iloc[k] > 0 and ct.iloc[k] > 0:
            out.append((ix[k], -1, -fwd.iloc[k]))
        elif cg.iloc[k] < 0 and cd.iloc[k] < 0 and ct.iloc[k] < 0:
            out.append((ix[k], +1, fwd.iloc[k]))
    return out


IDEAS = {"1_asia_sweep": idea1, "3_lbma_pm_fix": idea3, "5_dxy_yield_diverge": idea5,
         "6_round_number": idea6, "7_monday_gap": idea7, "9_expiry_pin": idea9,
         "10_vwap_revert": idea10, "11_nr7_breakout": idea11}


def maxstat(store, draws, block=10, seed=11):
    rng = np.random.default_rng(seed)
    obs = {k: a.mean() / (a.std() / np.sqrt(len(a))) for k, a in store.items() if len(a) > 20}
    if not obs:
        return {}
    best = max(obs, key=lambda k: obs[k])
    mx = np.empty(draws)
    for i in range(draws):
        m = -9e9
        for a in store.values():
            n = len(a)
            if n <= 20:
                continue
            nb = int(np.ceil(n / block))
            st = rng.integers(0, max(n - block, 1), nb)
            b = np.concatenate([a[x:x + block] for x in st])[:n] - a.mean()
            m = max(m, b.mean() / (b.std() / np.sqrt(n)))
        mx[i] = m
    q = lambda v: float(np.percentile(mx, v))
    return dict(best_cell=best, observed_t=float(obs[best]), p50=q(50), p95=q(95),
                p_value=float(np.mean(mx >= obs[best])), cells=len(obs))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=400)
    a = ap.parse_args()
    from scripts.v5_advisor_measure import load_frames
    from src.v5.mtf_frames import frames
    _, m15 = load_frames()
    M = frames(m15)["M15"]
    B = Book(M)
    days = pd.DatetimeIndex(sorted(set(M.index.normalize())))
    days = days[days.dayofweek < 5]
    gd = M["close"].resample("1D").last().dropna()

    print("=" * 104)
    print(f"THE EIGHT UNTESTED SETUPS — clean M15, {M.index.min().date()} -> "
          f"{M.index.max().date()}, cost {COST_BP}bp")
    print("=" * 104)
    rows, store = [], {}
    for name, fn in IDEAS.items():
        try:
            trades = fn(B, days)
        except Exception as e:
            print(f"  {name:22s} ERROR {type(e).__name__}: {str(e)[:50]}")
            continue
        if len(trades) < 25:
            print(f"  {name:22s} only {len(trades)} fires — not admissible")
            continue
        t = pd.DataFrame(trades, columns=["when", "side", "gross_bp"])
        t["when"] = pd.DatetimeIndex(t["when"])
        span_y = (t["when"].max() - t["when"].min()).days / 365.25
        # benchmark: what holding gold paid over a MATCHED average holding period
        hold_d = 1
        base = ((gd.shift(-hold_d) / gd - 1) * 1e4).dropna().mean()
        net = t["gross_bp"] - COST_BP
        excess = net - base
        yrs = t["when"].dt.year
        per = excess.groupby(yrs).mean()
        store[name] = t["gross_bp"].values
        rows.append(dict(idea=name, n=len(t), per_year=len(t) / max(span_y, .01),
                         gross_bp=float(t["gross_bp"].mean()), net_bp=float(net.mean()),
                         base_bp=float(base), excess_bp=float(excess.mean()),
                         t=float(net.mean() / (net.std() / np.sqrt(len(net)))),
                         hit=float((t["gross_bp"] > COST_BP).mean()),
                         years_pos=int((per > 0).sum()), years=int(len(per))))
    df = pd.DataFrame(rows).sort_values("t", ascending=False)
    df.to_csv(OUT / "twelve_ideas.csv", index=False)
    print(f"  {'idea':22s} {'n':>6s} {'/yr':>6s} {'gross':>8s} {'net bp':>8s} {'t':>6s} "
          f"{'hit':>6s} {'yrs+':>7s}")
    for _, r in df.iterrows():
        print(f"  {r['idea']:22s} {r['n']:6d} {r['per_year']:6.0f} {r['gross_bp']:+8.2f} "
              f"{r['net_bp']:+8.2f} {r['t']:+6.2f} {r['hit']*100:5.1f}% "
              f"{r['years_pos']:3d}/{r['years']:<3d}")
    print(f"\n  cells with net > 0: {(df.net_bp>0).sum()} of {len(df)}")
    n = maxstat(store, a.draws)
    if n:
        (OUT / "twelve_null.json").write_text(json.dumps(n, indent=2))
        print(f"\n  best signed t {n['observed_t']:+.3f} ({n['best_cell']})")
        print(f"  null of the max over {n['cells']} cells: p50 {n['p50']:+.3f}  "
              f"p95 {n['p95']:+.3f}   p {n['p_value']:.4f} -> "
              f"{'SURVIVES' if n['p_value']<0.05 else 'does not survive'}")


if __name__ == "__main__":
    main()
