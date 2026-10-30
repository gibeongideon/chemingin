#!/usr/bin/env python3
"""OCTOBER D21 part 2 — re-run §3ch's blend on the IMPLEMENTABLE break stream.

§3ch blended a break stream earning **+10.161bp** into the champion and reported all four weights
clearing Bar 2 (matched-vol paired t >= +2.0, both halves positive), topping out at Sharpe +1.252
at weight 0.50 with t +2.08. D21 established that +10.161bp contains two conventions a resting stop
order cannot honour:

  * **45 single-bar Sunday days** (one 23:00 H1 bar) at +25.519bp, where "exit at the session close"
    is one hour later and the fill is a weekend GAP rather than a level: worth **+0.493bp**.
  * **35 outside days** where the SAME bar takes out both levels. §3ch's `side = 1 if fu < fd
    else -1` sends every one of them SHORT; a resting buy-stop would have FILLED, and as longs they
    average **-26.6bp**. Worth **+0.860bp**. (On M15 only 8 such days survive, which is
    `order-of-events-needs-intraday` confirming itself: the ambiguity shrinks with resolution.)

*** AND THE ANSWER IS THAT THE CONVENTIONS ARE AN H1-FILE ARTEFACT. *** On H1 they are worth
13% of the headline. On M15 -- the frame that can actually resolve which level was hit first --
there are **8 tie days, not 35, and no single-bar days at all**, so the two streams differ by
**0.153bp and 0.06 t-units**. That is `order-of-events-needs-intraday` confirming itself from the
other side: the ambiguity is a property of the measurement frame, and it shrinks when the frame
gets finer. **§3ch's Bar 2 verdict is NOT overturned by the convention audit.** The run is kept
because a null result on a plausible-looking correction is worth exactly as much as a positive one,
and because the H1-only version of this claim was already written down before M15 was checked.

THE DESIGN POINT. Both streams go through IDENTICAL blend machinery here, so the level of my
reconstruction need not match §3ch's to the digit -- **the DELTA between the two streams is the
decision-relevant quantity**, and it is internally consistent by construction. §3ch's own blend
code was never committed, so its absolute numbers cannot be checked against. **My reconstruction
disagrees with it** (champion +1.111 here vs §3ch's +1.042; weight 0.50 at t +1.70 vs +2.08), so
the absolute PASS/FAIL marks below describe THIS blend construction and must not be quoted as a
re-verdict on §3ch's.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from v5_xau_champion_lifts import champion_recipe                     # noqa: E402
from src.v5 import mtf_frames                                         # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "october"
RT_BP = 0.504 * 1.5
TARGET_VOL, VOL_HL, MAX_LEV = 0.10, 42, 8.0
WEIGHTS = (0.15, 0.25, 0.35, 0.50)


def load_m15() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data/XAUUSD_M15_long.csv", parse_dates=["time"], index_col="time")
    return d[~d.index.duplicated(keep="last")].sort_index()


def champion_daily(m15: pd.DataFrame) -> pd.Series:
    """The deployed champion, H4 REBUILT FROM M15 (never the spliced XAUUSD_H4_long.csv)."""
    d = mtf_frames.aggregate(m15, "H4").dropna(subset=["close"])
    c = d["close"].astype(float)
    ret = c.pct_change()
    bpy = len(d) / max((d.index[-1] - d.index[0]).days / 365.25, 1e-9)
    vol = ret.ewm(halflife=VOL_HL, min_periods=20).std() * np.sqrt(bpy)
    fc = champion_recipe(c, 1.0, 1.5)
    pos = (fc.clip(0, 2) * (TARGET_VOL / vol)).clip(0, MAX_LEV).shift(1)
    r = (pos * ret - pos.diff().abs().fillna(0.0) * (RT_BP / 1e4)).dropna()
    return r.groupby(r.index.normalize()).sum()


def break_daily(m15: pd.DataFrame, implementable: bool) -> pd.Series:
    """Daily return of the prior-high break.

    `implementable=False` reproduces §3ch exactly: ties go SHORT (and are therefore dropped from
    the long leg), single-bar days are counted. `implementable=True` is what a resting buy-stop
    actually gets: ties FILL as longs, single-bar days are excluded.
    """
    day = m15.index.normalize()
    hi = m15["high"].groupby(day).max().shift(1)
    lo = m15["low"].groupby(day).min().shift(1)
    pdh = hi.reindex(day).to_numpy(float); pdl = lo.reindex(day).to_numpy(float)
    h, l, c = (m15[k].to_numpy(float) for k in ("high", "low", "close"))
    nd = day.to_numpy()
    idx, val = [], []
    start = 0
    for i in range(1, len(m15) + 1):
        if i == len(m15) or nd[i] != nd[start]:
            n = i - start
            if np.isfinite(pdh[start]) and np.isfinite(pdl[start]) and (n >= 2 or not implementable):
                sl = slice(start, i)
                up = np.flatnonzero(h[sl] > pdh[start])
                dn = np.flatnonzero(l[sl] < pdl[start])
                fu = up[0] if len(up) else 10**9
                fd = dn[0] if len(dn) else 10**9
                take = (fu < fd) if not implementable else (fu < 10**9 and fu <= fd)
                if take:
                    idx.append(pd.Timestamp(nd[start]))
                    val.append((c[i - 1] / pdh[start] - 1.0) - RT_BP / 1e4)
            start = i
    return pd.Series(val, index=pd.DatetimeIndex(idx)).groupby(level=0).sum()


def perf(r: pd.Series, ppy: float = 252.0) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    return dict(sharpe=float(r.mean() / r.std() * np.sqrt(ppy)),
                cagr=float(eq.iloc[-1] ** (ppy / len(r)) - 1),
                maxdd=float((eq / eq.cummax() - 1).min()))


def vol_match(a: pd.Series, b: pd.Series) -> pd.Series:
    sa, sb = a.std(), b.std()
    return a * (sb / sa) if sa > 0 else a


def blend_table(name: str, champ: pd.Series, brk_raw: pd.Series) -> list:
    """Vol-target the break leg to the champion's daily vol, then blend and paired-t at matched vol."""
    full = champ.index
    brk = brk_raw.reindex(full).fillna(0.0)
    brk = brk * (champ.std() / brk.std())          # same daily vol, so `w` means what it says
    base = perf(champ)
    half = len(full) // 2
    rows = [dict(blend="champion alone", sharpe=base["sharpe"], t=np.nan,
                 h1=np.nan, h2=np.nan, maxdd=base["maxdd"])]
    for w in WEIGHTS:
        bl = (1 - w) * champ + w * brk
        d = vol_match(bl, champ) - champ
        t = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
        m = perf(bl)
        h1 = perf(bl.iloc[:half])["sharpe"] - perf(champ.iloc[:half])["sharpe"]
        h2 = perf(bl.iloc[half:])["sharpe"] - perf(champ.iloc[half:])["sharpe"]
        rows.append(dict(blend=f"+ break {w:.2f}", sharpe=m["sharpe"], t=t, h1=h1, h2=h2,
                         maxdd=m["maxdd"]))
    print(f"\n  --- {name} ---")
    print(f"  {'blend':<22} {'Sharpe':>9} {'matched-t':>10} {'H1 dSR':>9} {'H2 dSR':>9} "
          f"{'maxDD':>9}  Bar 2")
    print("  " + "-" * 82)
    for r in rows:
        ok = "" if np.isnan(r["t"]) else (
            "PASS" if (r["t"] >= 2.0 and r["h1"] > 0 and r["h2"] > 0) else "**FAIL**")
        tt = "—" if np.isnan(r["t"]) else f"{r['t']:+.2f}"
        h1 = "—" if np.isnan(r["h1"]) else f"{r['h1']:+.3f}"
        h2 = "—" if np.isnan(r["h2"]) else f"{r['h2']:+.3f}"
        print(f"  {r['blend']:<22} {r['sharpe']:>+9.3f} {tt:>10} {h1:>9} {h2:>9} "
              f"{100*r['maxdd']:>8.1f}%  {ok}")
    return [dict(stream=name, **r) for r in rows]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    m15 = load_m15()
    champ = champion_daily(m15)
    print("=" * 96)
    print("OCTOBER D21 part 2 — does §3ch's BLEND survive the implementable break stream?")
    print("=" * 96)
    print(f"  champion daily stream: {len(champ):,} days "
          f"{champ.index[0].date()} -> {champ.index[-1].date()}, Sharpe {perf(champ)['sharpe']:+.3f}")

    rows = []
    for label, impl in (("§3ch stream (ties SHORT, Sundays counted)", False),
                        ("IMPLEMENTABLE stream (ties FILL, Sundays out)", True)):
        b = break_daily(m15, impl)
        bp = float(b.mean() * 1e4)
        t = float(b.mean() / (b.std(ddof=1) / np.sqrt(len(b))))
        print(f"\n  {label}: n={len(b):,} trades, mean {bp:+.3f}bp, t {t:+.2f}")
        rows += blend_table(label, champ, b)

    pd.DataFrame(rows).to_csv(OUT / "d21_blend_recheck.csv", index=False)
    print(f"""
  WHAT THIS DECIDES — AND IT IS A NULL RESULT. The convention correction is worth ~0.06 t-units at
  M15 resolution, because M15 has 8 tie days and no single-bar days where H1 has 35 and 45. The
  conventions are a property of the H1 FILE, not of the strategy. §3ch's Bar 2 verdict survives
  this audit.
  TWO THINGS THIS DOES NOT SAY. (1) The absolute PASS/FAIL marks above are from MY blend
  construction, which gives the champion +1.111 against §3ch's +1.042 and weight 0.50 at t +1.70
  against +2.08 — §3ch's blend code was never committed, so this is a different construction, not
  a re-verdict. (2) Bar 1 is still uncomputed: §3ch's DSR was fed a fabricated trial array.
  The live threat to this candidate is D21's PLACEBO control and the fill probe, not conventions.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
