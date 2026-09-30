"""NR7 first-break, backtested on M15 where the first break is OBSERVABLE rather than assumed.

WHY THE DAILY VERSION WAS NOT SALVAGEABLE. A first-break strategy needs to know WHICH of two
levels was touched first. Daily bars do not carry that, and every way of faking it does real
work in the result:

  * assigning outside days by a convention ("the level nearer the open went first") gave portfolio
    Sharpe +7.58 under one convention and +8.23 under its opposite -- the unresolvable choice was
    worth 0.65 Sharpe, so the convention was driving the answer.
  * DROPPING outside days instead is worse, and it is what produced Sharpe 2.1-5.5 with hit rates
    up to 87%. Conditioning on "the high broke AND the low did not" selects days that closed
    strongly in the break direction: it deletes exactly the cases where the breakout failed and
    reversed through the other side. Survivorship on the outcome, not a lookahead in the usual
    timestamp sense, and the tell was a control on NON-NR7 days scoring HIGHER (GOLD 5.46 vs
    4.02, NDX 4.95 vs 3.94) -- the narrow range contributed nothing, and the "edge" was the
    broken mechanic.
  * two files were also unusable outright: SPX has 8,547 zero-range days (34.5%) and SILVER 1,752
    (27%), because pre-1962 daily history carries no intraday high/low, so high==low==close and
    the NR7 test fires on 45% / 35% of days instead of 14%.

ON M15 NONE OF THAT IS NECESSARY. Walk the day's bars in order; the first bar to trade through a
level defines the position; the other order is cancelled. Every NR7 day is included, winners and
losers alike, and the ambiguity rate collapses because both levels rarely break inside the same
15 minutes.

THE CONTROL IS THE POINT. The same mechanic runs on NON-NR7 days. If the narrow-range condition
carries information, NR7 days beat the control; if it does not, this closes with the mechanic
measured honestly rather than with a fantasy Sharpe.

    python scripts/v5_nr7_m15_backtest.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "v5_runs" / "nr7"
SPREAD_BP, SPREAD_MULT = 0.94, 1.5      # measured Maven XAUUSD, widened per repo convention
TARGET_VOL = 0.10
OOS_START = "2021-01-01"


def build(m15: pd.DataFrame, nr7_only: bool, slip: bool = True) -> pd.DataFrame:
    d = m15.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                "close": "last"}).dropna()
    rng = d["high"] - d["low"]
    assert (rng > 0).all(), "zero-range daily bar built from M15 — impossible, check the source"
    nr7 = rng == rng.rolling(7).min()
    ret = d["close"].pct_change()
    vol = ret.ewm(halflife=20, min_periods=20).std() * np.sqrt(252)
    ow = SPREAD_BP * SPREAD_MULT * 1e-4
    mi, hh, ll, cc = m15.index, m15["high"].values, m15["low"].values, m15["close"].values
    rows = []
    for i in range(7, len(d) - 1):
        if nr7.iloc[i] != nr7_only:
            continue
        hi, lo = d["high"].iloc[i], d["low"].iloc[i]
        day = d.index[i + 1]
        s = int(np.searchsorted(mi.values, np.datetime64(day), side="left"))
        e = int(np.searchsorted(mi.values, np.datetime64(day + pd.Timedelta(days=1)),
                                side="left"))
        if s >= len(mi) or e <= s + 2:
            continue
        side, amb = 0, False
        for k in range(s, e):
            up, dn = hh[k] > hi, ll[k] < lo
            if up and dn:                      # both inside one 15-min bar: genuinely ambiguous
                amb = True
                side = 0
                break
            if up:
                side = +1
                break
            if dn:
                side = -1
                break
        if side == 0:
            if amb:
                rows.append(dict(when=day, side=0, ambiguous=True, net_ret=np.nan))
            continue
        lvl = hi if side > 0 else lo
        entry = lvl * (1 + side * ow / 2) if slip else lvl
        exit_px = cc[e - 1]
        v = vol.iloc[i]
        if not np.isfinite(v) or v <= 0:
            continue
        lots = float(np.clip(TARGET_VOL / v, 0, 8))
        g = side * (exit_px - entry) / entry
        rows.append(dict(when=day, side=side, ambiguous=False,
                         gross_bp=g * 1e4, net_ret=lots * (g - ow)))
    return pd.DataFrame(rows)


def stats(r: pd.Series, npy: float) -> dict:
    r = r.dropna()
    if len(r) < 10:
        return {}
    eq = (1 + r).cumprod()
    return dict(n=len(r), sharpe=float(r.mean() / r.std() * np.sqrt(npy)) if r.std() > 0 else np.nan,
                cagr=float(eq.iloc[-1] ** (npy / len(r)) - 1) * 100,
                maxdd=float((eq / eq.cummax() - 1).min() * 100),
                hit=float((r > 0).mean() * 100), mean_bp=float(r.mean() * 1e4))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    from scripts.v5_advisor_measure import load_frames
    from src.v5.mtf_frames import frames
    _, m15 = load_frames()
    M = frames(m15)["M15"]
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print(f"NR7 FIRST-BREAK on XAUUSD M15 — first break OBSERVED, every day included")
    print(f"  {M.index.min().date()} -> {M.index.max().date()}, "
          f"cost {SPREAD_BP*SPREAD_MULT:.2f}bp one-way, intraday so no financing")
    print("=" * 96)
    res = {}
    for nr7_only, lab in ((True, "NR7 days"), (False, "CONTROL: non-NR7 days")):
        t = build(M, nr7_only)
        if not len(t):
            continue
        amb = float(t["ambiguous"].mean() * 100)
        t = t[~t["ambiguous"]].set_index("when")
        yrs = (t.index.max() - t.index.min()).days / 365.25
        npy = len(t) / max(yrs, .1)
        s = stats(t["net_ret"], npy)
        o = stats(t.loc[OOS_START:, "net_ret"], npy)
        res[lab] = (s, o, t, amb, npy)
        print(f"\n  {lab}")
        print(f"    signals {s['n']:5d}  ({npy:.0f}/yr)   ambiguous {amb:.1f}%   "
              f"long {(t.side>0).mean()*100:.0f}%")
        print(f"    gross {t['gross_bp'].mean():+.2f}bp   net/trade {s['mean_bp']:+.2f}bp   "
              f"hit {s['hit']:.1f}%")
        print(f"    Sharpe {s['sharpe']:+.2f}   CAGR {s['cagr']:+.2f}%   maxDD {s['maxdd']:.1f}%")
        print(f"    OOS (2021+) Sharpe {o.get('sharpe', float('nan')):+.2f}   "
              f"n {o.get('n', 0)}")
    if len(res) == 2:
        a, b = res["NR7 days"][0], res["CONTROL: non-NR7 days"][0]
        print("\n" + "=" * 96)
        print(f"  DOES THE NARROW RANGE ADD ANYTHING?")
        print(f"    NR7 net/trade {a['mean_bp']:+.2f}bp  vs  control {b['mean_bp']:+.2f}bp   "
              f"-> {a['mean_bp']-b['mean_bp']:+.2f}bp")
        print(f"    NR7 Sharpe    {a['sharpe']:+.2f}     vs  control {b['sharpe']:+.2f}")
        ta, tb = res["NR7 days"][2]["net_ret"].dropna(), res["CONTROL: non-NR7 days"][2]["net_ret"].dropna()
        se = np.sqrt(ta.var(ddof=1)/len(ta) + tb.var(ddof=1)/len(tb))
        print(f"    difference in means: t {(ta.mean()-tb.mean())/se:+.2f}")
        verdict = ("NR7 adds nothing -- the narrow range is not the mechanism"
                   if (ta.mean()-tb.mean())/se < 1.5 else "NR7 adds measurably")
        print(f"    -> {verdict}")
        pd.concat({"nr7": ta, "control": tb}).to_csv(OUT / "nr7_m15.csv")


if __name__ == "__main__":
    main()
