#!/usr/bin/env python3
"""Champion through gold's 2011-2015 BEAR MARKET — the test its 2015+ backtest never faced.

Data: data/GOLD_D1_long.csv (continuous gold futures, 2000-08 ..). The champion's lookbacks are
defined in H4 bars, so on D1 they are rescaled by 1/6 (`champion_recipe(close, 1/6, 1.5)`) to keep
the SAME speeds in days (EWMAC 16-256d, breakout 10-40d). Vol half-life 42 H4 bars = 7 days.
Cost 1.0bp one-way (stress 3bp). Financing: none, and a stress at -3%/yr on long notional.
Benchmarks: gold buy & hold, and CONSTANT LONG at the same vol target (isolates the timing).
Eras: 2002-01..2011-08 bull · 2011-09..2015-12 BEAR (gold -45%) · 2016-01.. recent.
Sanity: the D1 port on 2016+ should look like the H4 champion (Sharpe ~1.0).

    python scripts/v5_champion_bear_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.v5_volregime_taper_crossasset as T  # noqa: E402
from scripts.v5_xau_champion_lifts import champion_recipe  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "champion_bear_test"
ERAS = [("2002-01-01", "2011-08-31", "BULL 2002-11"), ("2011-09-01", "2015-12-31", "BEAR 2011-15"),
        ("2016-01-01", "2026-12-31", "RECENT 2016-26"), ("2002-01-01", "2026-12-31", "FULL 2002-26")]


def run(close: pd.DataFrame, fc: pd.Series, cost_bp: float, fin: float = 0.0) -> pd.Series:
    T.EVAL_START = "2002-01-01"
    r = T.engine_bp(close, fc, cost_bp, 252, 7)
    if fin:
        ret = close["close"].pct_change()
        vol = ret.ewm(halflife=7, min_periods=20).std() * np.sqrt(252)
        lev = (fc.clip(-2, 2) * (T.TARGET_VOL / vol)).clip(-T.MAX_LEV, T.MAX_LEV).shift(1)
        r = r - (lev.reindex(r.index).fillna(0).clip(lower=0) * fin / 252)
    return r


def stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    under = (dd < 0).astype(int)
    longest = under.groupby((under == 0).cumsum()).sum().max()
    yrs = len(r) / 252
    return dict(total=100 * (eq.iloc[-1] - 1), cagr=100 * (eq.iloc[-1] ** (1 / yrs) - 1),
                sharpe=r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0,
                maxdd=100 * dd.min(), underwater_yrs=longest / 252)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(ROOT / "data/GOLD_D1_long.csv", parse_dates=["time"], index_col="time")
    d = d[~d.index.duplicated(keep="last")].sort_index()
    fc = champion_recipe(d["close"], 1 / 6, 1.5)
    streams = {
        "CHAMPION": run(d, fc, 1.0),
        "CHAMPION 3bp cost": run(d, fc, 3.0),
        "CHAMPION -3%/yr swap": run(d, fc, 1.0, fin=0.03),
        "CONSTANT LONG same vol": run(d, pd.Series(1.0, index=d.index), 1.0),
    }
    gold = d["close"].pct_change().loc["2002-01-01":].dropna()
    streams["GOLD buy & hold"] = gold
    lines = []
    for a, b, era in ERAS:
        lines.append(f"\n{era}  (gold {d.close.loc[a:b].iloc[0]:.0f} -> {d.close.loc[a:b].iloc[-1]:.0f})")
        lines.append(f"  {'':24} {'total':>8} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7} {'longest underwater':>19}")
        for k, r in streams.items():
            s = stats(r.loc[a:b])
            lines.append(f"  {k:24} {s['total']:+7.1f}% {s['cagr']:+6.1f}% {s['sharpe']:+7.2f} "
                         f"{s['maxdd']:+6.1f}% {s['underwater_yrs']:>14.1f} yrs")
    # bear-market behaviour in detail
    fcd = fc.loc["2011-09-01":"2015-12-31"]
    lines.append(f"\nBEAR detail: average forecast {fcd.mean():.2f} (resting tilt 0.15); share of days at "
                 f"resting (<0.2) {100*(fcd < 0.2).mean():.0f}%, in strong trend (>=1.5) {100*(fcd >= 1.5).mean():.0f}%")
    ch = streams["CHAMPION"]
    yr = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in
                       (("champion", ch), ("const_long", streams["CONSTANT LONG same vol"]), ("gold", gold))})
    lines.append("\nBY YEAR (%):\n" + (100 * yr).round(1).to_string())
    txt = "\n".join(lines) + "\n"
    print(txt)
    (OUT / "report.txt").write_text(txt)


if __name__ == "__main__":
    sys.exit(main())
