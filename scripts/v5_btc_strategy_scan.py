"""BTCUSD: does ANY trend speed beat simply HOLDING bitcoin? (CONTROL #2, the real bar.)

The champion recipe already runs on BTC as a sleeve of the 4-sleeve book, so the first honest
question is not "invent a BTC strategy" but "what does the incumbent actually deliver, and does
it clear the benchmark that matters". For bitcoin that benchmark is severe: CONTROL #2 exists
because a long-only trend follower in a bull market posts a fine Sharpe with zero timing skill,
and bitcoin has had the largest drift of any instrument in this repo.

THREE BTC-SPECIFIC TRAPS, each already logged:
  * DAY COUNT. BTC quotes WEEKENDS (~365 bars/yr) while every other sleeve quotes ~252. Method
    rule 1.3.6: comparing a 365-day series to a 252-day one under a common sqrt(252) INVERTED a
    book ranking on 2026-09-04. Everything here is annualised on each series' OWN realised
    bars/year, and any cross-series comparison is compounded onto a shared index first.
  * THE `D=6` TRAP. `champion_signal` hardcodes 6 bars/day; on D1 data it silently becomes a
    96-1536 DAY trend follower. Use `champion_recipe(close, scale=1/6, conc_p=1.5)`.
  * COST IN BASIS POINTS, NOT DOLLARS (rule 1.3.4). $1 of BTC spread is 1.2bp at $82,000 and was
    120bp at $800 in 2015 -- a dollar figure is meaningless across this price range, so cost is
    applied as a per-year bp series from the actual price.

Reported for every arm: Sharpe, CAGR, maxDD, paired-t vs BUY-AND-HOLD at matched vol, per-year
sign count, and both sample halves -- because bitcoin's history is two different regimes and a
pooled number hides that (rule 1.3.8).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from v5_xau_champion_lifts import champion_recipe, vol_match          # noqa: E402
from src.cta.bootstrap import block_bootstrap_sharpe                  # noqa: E402

SPREAD_USD = 1.0            # FTMO BTCUSD, from the live config note
TARGET_VOL, VOL_HL, MAX_LEV = 0.10, 42, 8.0


def load() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data/BTC_D1_long.csv", parse_dates=["time"], index_col="time")
    d = d[~d.index.duplicated(keep="last")].sort_index()
    return d


def bars_per_year(idx) -> float:
    yrs = (idx[-1] - idx[0]).days / 365.25
    return len(idx) / max(yrs, 1e-9)


def perf(r: pd.Series, bpy: float) -> dict:
    r = r.dropna()
    if len(r) < 50 or r.std() == 0:
        return dict(sharpe=np.nan, cagr=np.nan, maxdd=np.nan)
    eq = (1 + r).cumprod()
    yrs = len(r) / bpy
    return dict(sharpe=float(r.mean() / r.std() * np.sqrt(bpy)),
                cagr=float(eq.iloc[-1] ** (1 / yrs) - 1),
                maxdd=float((eq / eq.cummax() - 1).min()))


def main() -> int:
    d = load()
    c = d["close"].astype(float)
    bpy = bars_per_year(d.index)
    ret = c.pct_change()
    # cost as a per-bar bp of the CURRENT price, not a fixed dollar amount
    cost_frac = SPREAD_USD / c
    print("=" * 100)
    print("BTCUSD — does any trend speed beat HOLDING bitcoin?")
    print(f"  {len(d):,} D1 bars  {d.index[0].date()}..{d.index[-1].date()}  "
          f"= {bars_per_year(d.index):.1f} bars/yr (BTC quotes weekends; gold quotes ~252)")
    print(f"  spread ${SPREAD_USD} = {1e4*SPREAD_USD/float(c.iloc[-1]):.2f}bp today, "
          f"{1e4*SPREAD_USD/float(c.iloc[0]):.0f}bp at the start of the sample")
    print("=" * 100)

    # ---- the benchmark: vol-targeted buy and hold (CONTROL #2)
    vol = ret.ewm(halflife=VOL_HL, min_periods=20).std() * np.sqrt(bpy)
    bh_pos = (TARGET_VOL / vol).clip(0, MAX_LEV).shift(1)
    bh = (bh_pos * ret).dropna()
    bh_raw = ret.dropna()

    arms = {"BUY & HOLD (vol-targeted)": bh, "BUY & HOLD (unlevered)": bh_raw}
    for name, scale in (("champion (book's own, 1/6)", 1 / 6),
                        ("champion FAST (1/3)", 1 / 3),
                        ("champion SLOW (1/12)", 1 / 12)):
        fc = champion_recipe(c, scale, 1.5)
        pos = (fc.clip(-2, 2) * (TARGET_VOL / vol)).clip(-MAX_LEV, MAX_LEV).shift(1)
        turn = pos.diff().abs().fillna(0.0)
        arms[name] = (pos * ret - turn * cost_frac * 0.75 * 2).dropna()

    idx = sorted(set.intersection(*[set(v.index) for v in arms.values()]))
    idx = pd.DatetimeIndex(idx)
    base = arms["BUY & HOLD (vol-targeted)"].reindex(idx)

    print(f"\n  {'arm':30s} {'Sharpe':>7s} {'CAGR':>9s} {'maxDD':>8s} "
          f"{'t vs B&H':>9s} {'yrs beat':>9s} {'1st half':>9s} {'2nd half':>9s}")
    rows = []
    for name, r in arms.items():
        r = r.reindex(idx)
        m = perf(r, bpy)
        if "BUY & HOLD (vol" in name:
            t, yb = np.nan, "—"
        else:
            dd = vol_match(r, base) - base
            t = float(dd.mean() / dd.std() * np.sqrt(len(dd))) if dd.std() > 0 else np.nan
            yr = r.groupby(r.index.year).sum(); yb_ = base.groupby(base.index.year).sum()
            yb = f"{int((yr > yb_).sum())}/{len(yr)}"
        h1 = r[r.index.year <= 2020]; h2 = r[r.index.year >= 2021]
        s1, s2 = perf(h1, bpy)["sharpe"], perf(h2, bpy)["sharpe"]
        print(f"  {name:30s} {m['sharpe']:+7.3f} {100*m['cagr']:+8.1f}% {100*m['maxdd']:7.1f}% "
              f"{t:+9.2f} {yb:>9s} {s1:+9.3f} {s2:+9.3f}")
        rows.append(dict(arm=name, **m, t_vs_bh=t, half1=s1, half2=s2))

    lo, hi = block_bootstrap_sharpe(base.values, block=21, n=4000, ppy=bpy)[:2] \
        if isinstance(block_bootstrap_sharpe(base.values, block=21, n=4000, ppy=bpy), tuple) \
        else (np.nan, np.nan)
    print(f"\n  buy-and-hold Sharpe 95% CI (block bootstrap): [{lo:+.2f}, {hi:+.2f}]"
          if np.isfinite(lo) else "")
    out = ROOT / "data/v5_runs/btc_scan"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "speed_scan.csv", index=False)
    print(f"\n  wrote {out}/speed_scan.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
