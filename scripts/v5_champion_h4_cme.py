#!/usr/bin/env python3
"""The REAL H4 champion from 2010 on CME gold futures (§3dp) — closes §3dn's open question.

§3dn could only run a D1 port through the 2011-15 bear (no clean pre-2015 H4 gold existed), and it
left one question open: is the live H4 Sharpe ~1.08 real, or partly the in-sample choice made on
2015+ data? Databento GLBX GC.v.0 hourly bars (single provenance, 2010-06..) settle it: everything
before 2015 is OUT-OF-SAMPLE for the H4 recipe.

Data: data/databento/ohlcv_1h/GC.csv.gz, back-adjusted at rolls (OHLC scaled by close_adj/close),
UTC -> Switch server clock (NY+7), resampled to server H4 bars exactly like the live feed.
Futures returns already NET OUT financing (contango ≈ cash rate), so no swap is added; a swap-free
CFD would do better by roughly the cash rate.
Engine T.engine_bp: vol target 10%, H4 vol half-life 42 bars, 1bp one-way (stress 3bp).
Benchmarks: CONSTANT LONG at the same vol target (isolates timing) and GC buy & hold.
Control: on the overlap 2017-2026 the CME run must track the Switch-data run (corr of daily returns).

    python scripts/v5_champion_h4_cme.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.v5_volregime_taper_crossasset as T  # noqa: E402
from scripts.v5_champion_bear_test import stats  # noqa: E402
from scripts.v5_footprint_screen import utc_to_server  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "champion_h4_cme"
ANN, HL = 1512, 42
W0 = "2011-06-01"                      # after the ~1y warm-up of the slowest EWMAC (H4)
ERAS = [("2011-09-01", "2015-12-31", "BEAR 2011-15"),
        (W0, "2014-12-31", "OUT-OF-SAMPLE 2011-14"), ("2015-01-01", "2026-12-31", "IN-SAMPLE era 2015-26"),
        (W0, "2026-12-31", "FULL 2011-26")]


def h4_cme() -> pd.DataFrame:
    g = pd.read_csv(ROOT / "data/databento/ohlcv_1h/GC.csv.gz", parse_dates=["time"], index_col="time")
    k = g.close_adj / g.close
    g = pd.DataFrame(dict(open=g.open * k, high=g.high * k, low=g.low * k, close=g.close_adj))
    g.index = utc_to_server(g.index)
    g = g[~g.index.duplicated(keep="last")].sort_index()
    return g.resample("4h").agg(dict(open="first", high="max", low="min", close="last")).dropna()


def h4_switch() -> pd.DataFrame:
    x = pd.read_csv(ROOT / "data/v5_runs/zigzag_live_wf/XAUUSDs_H1.csv", parse_dates=["time"], index_col="time")
    x = x[~x.index.duplicated()].sort_index()
    return x.resample("4h").agg(dict(open="first", high="max", low="min", close="last")).dropna()


def run(df: pd.DataFrame, fc: pd.Series, cost: float, start: str = W0) -> pd.Series:
    T.EVAL_START = start
    return T.engine_bp(df, fc, cost, ANN, HL)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = h4_cme()
    fc = champion_signal(d.close)
    streams = {"CHAMPION H4": run(d, fc, 1.0), "CHAMPION H4 3bp": run(d, fc, 3.0),
               "CONSTANT LONG same vol": run(d, pd.Series(1.0, index=d.index), 1.0)}
    gd = d.close.resample("D").last().dropna()
    streams["GC buy & hold"] = gd.pct_change().loc[W0:].dropna()
    lines = [f"CME GC H4 (server clock) {d.index[0]} .. {d.index[-1]}, {len(d):,} bars"]
    # provenance control: same recipe on Switch's own H1->H4 over the overlap
    s = h4_switch()
    a, b = run(d, fc, 1.0, "2017-01-01"), run(s, champion_signal(s.close), 1.0, "2017-01-01")
    j = a.index.intersection(b.index)
    lines.append(f"control 2017+: CME vs Switch champion daily-return corr {a[j].corr(b[j]):.3f}; "
                 f"Sharpe CME {stats(a[j])['sharpe']:+.2f} vs Switch {stats(b[j])['sharpe']:+.2f}")
    for lo, hi, era in ERAS:
        px = d.close.loc[lo:hi]
        lines.append(f"\n{era}  (gold {px.iloc[0]:.0f} -> {px.iloc[-1]:.0f})")
        lines.append(f"  {'':24} {'total':>8} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7} {'longest underwater':>19}")
        for k, r in streams.items():
            st = stats(r.loc[lo:hi])
            lines.append(f"  {k:24} {st['total']:+7.1f}% {st['cagr']:+6.1f}% {st['sharpe']:+7.2f} "
                         f"{st['maxdd']:+6.1f}% {st['underwater_yrs']:>14.1f} yrs")
    ch, cl = streams["CHAMPION H4"], streams["CONSTANT LONG same vol"]
    diff = (ch - cl * ch.std() / cl.std()).dropna()                    # matched vol
    t = diff.mean() / diff.std() * np.sqrt(len(diff))
    lines.append(f"\nTIMING vs constant-long at MATCHED vol, full period: paired t {t:+.2f}")
    for lo, hi, era in ERAS[:3]:
        x = diff.loc[lo:hi]
        lines.append(f"  {era:24} t {x.mean() / x.std() * np.sqrt(len(x)):+.2f}")
    fcb = fc.loc["2011-09-01":"2015-12-31"]
    lines.append(f"\nBEAR detail: mean forecast {fcb.mean():.2f}; at resting (<0.2) {100 * (fcb < 0.2).mean():.0f}% "
                 f"of bars; strong (>=1.5) {100 * (fcb >= 1.5).mean():.0f}%")
    yr = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in
                       (("champion", ch), ("const_long", cl), ("gold", streams["GC buy & hold"]))})
    lines.append("\nBY YEAR (%):\n" + (100 * yr).round(1).to_string())
    txt = "\n".join(lines) + "\n"
    print(txt)
    (OUT / "report.txt").write_text(txt)


if __name__ == "__main__":
    sys.exit(main())
