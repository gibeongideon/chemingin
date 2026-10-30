#!/usr/bin/env python3
"""OCTOBER D21 part 3 — the ACTUAL YIELD of the prior-high break, and of its placebo.

§3cj decided the statistics: the prior-day high beats an arbitrary matched-distance level on the
FILL axis at p 0.0000 but only at **p 0.05-0.09** on the axis that pays, where the placebo already
earns ~60% of the excess. What none of that says is **what the thing actually makes**, which is the
question an equity curve answers and a basis-point mean does not.

THREE THINGS THIS REPORTS THAT §3cj DID NOT.

1. **Buy-and-hold gold on the same window.** MANDATORY CONTROL #2, and in this file it has killed
   more candidates than any null: `cot-positioning-disproven` was 0/36 against it, the external
   playbook 0/60, and `gold-overnight-decayed` netted +1.07%/yr against hold's +11.96%. A long-only
   intraday rule measured across gold's biggest bull run must clear HOLDING, not zero.

2. **EXPOSURE.** The break is in the market for part of one day, not all year. A sleeve earning
   half of buy-and-hold while exposed a fifth of the time is a different proposition from one
   earning half while fully invested -- it leaves capital, and it is what makes the sleeve
   blendable at all. Both the raw 1x-notional curve and a vol-targeted version are shown, because
   the raw one understates and the targeted one is what the book would actually run.

3. **THE SLIPPAGE LADDER ON THE YIELD, not on the mean.** §3ci: ~47% of the headline is the gap
   between the level and the crossing bar's close, and §3ch's ladder showed the edge surviving ~4bp
   and dying near 8bp. The live probe exists to measure exactly this, so until it reports, every
   row below should be read at the slippage you believe, not at zero.

YEAR BY YEAR, because a mean over 11 years hides regime dependence and this file has been caught by
that before (`smc-bos-h4-promising`: a single-split Sharpe 1.44 that was a 2023-26 bull artifact).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "v5_runs" / "october"
RT_BP = 0.504 * 1.5           # 0.756bp live round trip
SLIP = (0.0, 2.0, 4.0, 8.0)   # §3ch's ladder, in bp
TARGET_VOL = 0.10


def load_m15() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data/XAUUSD_M15_long.csv", parse_dates=["time"], index_col="time")
    return d[~d.index.duplicated(keep="last")].sort_index()


def wilder_atr_daily(hi, lo, cl, n=14):
    pc = cl.shift(1)
    tr = pd.concat([(hi - lo), (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def trades(m15: pd.DataFrame, mode: str, k_atr: float = 0.25, cancel: bool = True) -> pd.DataFrame:
    """One row per day a resting buy-stop would have filled.

    mode 'pdh'   -> level is the prior day's high (the real rule)
    mode 'fixed' -> level is k_atr x prior-day ATR above today's open (the placebo that §3cj
                    showed earns ~60% of it)
    `cancel` applies §3ch's first-break condition, which is implementable in real time: the resting
    order is pulled if the prior-day LOW goes first.
    """
    day = m15.index.normalize()
    g = m15.groupby(day)
    dh, dl, dc = g["high"].max(), g["low"].min(), g["close"].last()
    atr_d = wilder_atr_daily(dh, dl, dc).shift(1)
    pdh_d, pdl_d = dh.shift(1), dl.shift(1)

    h, l, c, o = (m15[x].to_numpy(float) for x in ("high", "low", "close", "open"))
    nd = day.to_numpy()
    pdh = pdh_d.reindex(day).to_numpy(float)
    pdl = pdl_d.reindex(day).to_numpy(float)
    atr = atr_d.reindex(day).to_numpy(float)

    rows, start = [], 0
    for i in range(1, len(m15) + 1):
        if i == len(m15) or nd[i] != nd[start]:
            n = i - start
            if n >= 2 and np.isfinite(pdh[start]) and np.isfinite(pdl[start]) \
                    and np.isfinite(atr[start]) and atr[start] > 0:
                L = pdh[start] if mode == "pdh" else o[start] + k_atr * atr[start]
                if L > o[start]:                       # a stop must sit ABOVE the open
                    sl = slice(start, i)
                    up = np.flatnonzero(h[sl] > L)
                    if len(up):
                        ku = int(up[0])
                        take = True
                        if cancel:
                            dn = np.flatnonzero(l[sl] < pdl[start])
                            if len(dn) and int(dn[0]) < ku:
                                take = False
                        if take:
                            rows.append(dict(date=pd.Timestamp(nd[start]),
                                             gross=c[i - 1] / L - 1.0,
                                             hours=(n - ku) * 0.25, nbars=n, k=ku))
            start = i
    return pd.DataFrame(rows).set_index("date")


def curve(tr: pd.DataFrame, days: pd.DatetimeIndex, slip_bp: float) -> pd.Series:
    r = (tr["gross"] - (RT_BP + slip_bp) / 1e4) if len(tr) else pd.Series(dtype=float)
    return r.reindex(days).fillna(0.0)


def stats(r: pd.Series, ppy: float) -> dict:
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    sh = float(r.mean() / r.std() * np.sqrt(ppy)) if r.std() > 0 else np.nan
    cagr = float(eq.iloc[-1] ** (ppy / len(r)) - 1)
    return dict(sharpe=sh, cagr=cagr, maxdd=float(dd), final=float(eq.iloc[-1]),
                calmar=cagr / abs(dd) if dd < 0 else np.nan)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    m15 = load_m15()
    day = m15.index.normalize()
    dc = m15.groupby(day)["close"].last()
    days = pd.DatetimeIndex(dc.index)
    yrs = (days[-1] - days[0]).days / 365.25
    ppy = len(days) / yrs

    hold = dc.pct_change().fillna(0.0)
    hs = stats(hold, ppy)

    print("=" * 104)
    print("OCTOBER D21 part 3 — ACTUAL YIELD of the prior-high break vs its placebo vs HOLDING gold")
    print("=" * 104)
    print(f"  XAUUSD M15  {days[0].date()} -> {days[-1].date()}   {len(days):,} days, "
          f"{yrs:.1f} years, {ppy:.0f} trading days/yr   round trip {RT_BP:.3f}bp")
    print(f"  gold went {dc.iloc[0]:,.2f} -> {dc.iloc[-1]:,.2f} = {100*(dc.iloc[-1]/dc.iloc[0]-1):+,.1f}%")
    print(f"\n  *** CONTROL #2 — BUY AND HOLD GOLD: CAGR {100*hs['cagr']:+.2f}%  "
          f"Sharpe {hs['sharpe']:+.3f}  maxDD {100*hs['maxdd']:.1f}%  "
          f"x{hs['final']:.2f} on the account ***")

    variants = [
        ("prior-day HIGH, cancel rule", dict(mode="pdh", cancel=True)),
        ("prior-day HIGH, no cancel", dict(mode="pdh", cancel=False)),
        ("PLACEBO 0.25 ATR over open", dict(mode="fixed", k_atr=0.25, cancel=False)),
        ("PLACEBO 0.50 ATR over open", dict(mode="fixed", k_atr=0.50, cancel=False)),
    ]
    rows = []
    for name, kw in variants:
        tr = trades(m15, **kw)
        expo = float(tr["hours"].sum() / (len(days) * 24.0))
        print(f"\n  --- {name} ---")
        print(f"      fires {len(tr):,} days ({len(tr)/len(days):.1%}), "
              f"{len(tr)/yrs:.0f}/yr · median hold {tr['hours'].median():.1f}h · "
              f"EXPOSURE {expo:.1%} of calendar hours")
        print(f"      {'extra slip':>11} {'mean bp':>9} {'hit':>7} {'CAGR':>9} {'Sharpe':>8} "
              f"{'maxDD':>8} {'Calmar':>8} {'x acct':>8} {'vs HOLD':>9}")
        print("      " + "-" * 88)
        for s in SLIP:
            r = curve(tr, days, s)
            st = stats(r, ppy)
            net = r[r != 0]
            tag = "  <- live probe decides this" if s == 4.0 else ""
            print(f"      {s:>8.0f}bp {net.mean()*1e4:>+9.3f} {float((net>0).mean()):>7.4f} "
                  f"{100*st['cagr']:>+8.2f}% {st['sharpe']:>+8.3f} {100*st['maxdd']:>7.1f}% "
                  f"{st['calmar']:>8.2f} {st['final']:>8.2f} "
                  f"{100*(st['cagr']-hs['cagr']):>+8.2f}%{tag}")
            rows.append(dict(variant=name, slip_bp=s, n=len(tr), exposure=expo,
                             mean_bp=float(net.mean() * 1e4), **st))

        # vol-targeted to 10%, which is how the book would actually size it
        r0 = curve(tr, days, 4.0)
        sc = TARGET_VOL / (r0.std() * np.sqrt(ppy))
        rv = r0 * sc
        sv = stats(rv, ppy)
        print(f"      vol-targeted to {TARGET_VOL:.0%} at 4bp slip: leverage x{sc:.1f} -> "
              f"CAGR {100*sv['cagr']:+.2f}%  Sharpe {sv['sharpe']:+.3f}  "
              f"maxDD {100*sv['maxdd']:.1f}%")

    # ---------------- the number the probe has to beat ------------------------------------------
    # §3ch's ladder said "survives ~4bp, dies near 8bp" -- but that was judged against ZERO.
    # Against CONTROL #2 the bar is holding gold, and it is reached far sooner.
    tr = trades(m15, mode="pdh", cancel=True)
    print(f"\n  *** BREAKEVEN SLIPPAGE — the real question the live probe has to answer ***")
    print(f"  {'slip bp':>8} {'CAGR':>9} {'Sharpe':>8} {'maxDD':>8} {'Calmar':>8}   "
          f"beats HOLD on Sharpe? on Calmar?")
    print("  " + "-" * 78)
    be_sh = be_ca = None
    for s in np.arange(0.0, 8.01, 0.5):
        st = stats(curve(tr, days, float(s)), ppy)
        ok_s, ok_c = st["sharpe"] > hs["sharpe"], st["calmar"] > hs["calmar"]
        if be_sh is None and not ok_s:
            be_sh = float(s)
        if be_ca is None and not ok_c:
            be_ca = float(s)
        print(f"  {s:>8.1f} {100*st['cagr']:>+8.2f}% {st['sharpe']:>+8.3f} "
              f"{100*st['maxdd']:>7.1f}% {st['calmar']:>8.2f}        "
              f"{'YES' if ok_s else 'no ':>3}         {'YES' if ok_c else 'no '}")
    print(f"\n  HOLD is Sharpe {hs['sharpe']:+.3f} / Calmar {hs['calmar']:.2f}. The break stops "
          f"beating it on Sharpe at ~{be_sh:.1f}bp and on Calmar at ~{be_ca:.1f}bp of slippage.")
    print(f"  §3ch's ladder said 'survives ~4bp, dies near 8bp' — but that was measured against "
          f"ZERO.\n  Against CONTROL #2 the usable budget is roughly HALF what was filed.")

    print(f"\n  YEAR BY YEAR — prior-day HIGH with the cancel rule, against holding gold")
    print(f"  {'year':>6} {'fires':>6} {'mean bp':>9} {'0bp slip':>10} {'4bp slip':>10} "
          f"{'8bp slip':>10} {'HOLD':>10}")
    print("  " + "-" * 70)
    yr_rows = []
    for y, sub in dc.groupby(dc.index.year):
        d2 = pd.DatetimeIndex(sub.index)
        t2 = tr[tr.index.isin(d2)]
        hy = float((1 + hold.reindex(d2).fillna(0.0)).prod() - 1)
        cells = {}
        for s in (0.0, 4.0, 8.0):
            cells[s] = float((1 + curve(t2, d2, s)).prod() - 1)
        mb = float((t2["gross"] - RT_BP / 1e4).mean() * 1e4) if len(t2) else np.nan
        print(f"  {y:>6} {len(t2):>6} {mb:>+9.3f} {100*cells[0.0]:>+9.2f}% "
              f"{100*cells[4.0]:>+9.2f}% {100*cells[8.0]:>+9.2f}% {100*hy:>+9.2f}%")
        yr_rows.append(dict(year=y, fires=len(t2), mean_bp=mb, r0=cells[0.0],
                            r4=cells[4.0], r8=cells[8.0], hold=hy))
    yd = pd.DataFrame(yr_rows)
    for s, col in ((0.0, "r0"), (4.0, "r4"), (8.0, "r8")):
        w = int((yd[col] > yd["hold"]).sum())
        print(f"    at {s:.0f}bp slip the break beats HOLDING in {w} of {len(yd)} years, "
              f"and is positive in {int((yd[col] > 0).sum())}")

    pd.DataFrame(rows).to_csv(OUT / "d21_yield.csv", index=False)
    yd.to_csv(OUT / "d21_yield_by_year.csv", index=False)
    print(f"""
  HOW TO READ 'vs HOLD'. That column is CONTROL #2 and it is the one that matters: a long-only
  intraday rule measured across gold's biggest bull run is competing with simply owning the metal.
  Note the asymmetry that makes the sleeve interesting even where it loses on raw CAGR -- it is
  exposed only a fraction of the time, so the comparison on Sharpe and on Calmar is the honest one,
  and the leftover capital is what lets it BLEND rather than replace.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
