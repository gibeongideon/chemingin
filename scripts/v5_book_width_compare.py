"""Is XAU+BTC actually better than the 4-sleeve book? Same engine, same dial, same simulation.

3bq measured XAU+BTC at Sharpe 1.511 / FTMO pass 72.2%. The 4-sleeve book (XAU+BTC+NDX+BRENT)
was measured ELSEWHERE at ~88.6% pass -- but on a different dial, cost basis and simulator, so
the two numbers cannot be compared as filed. This puts every width through ONE harness.

`binding-constraint-book-matching` is the governing prior: HRP/RMT/CVaR all lose to equal weight
(0/20), and the right question is not "which weights" but "does the book's WIDTH match the rule
that binds". For FTMO the binding rule is the 5% DAILY limit, and widening took the recorded pass
rate 88.6% -> 94.1%. So the expectation here is that WIDER WINS, and 3bq's 2-sleeve deployment
may be a step backwards.

Scale discipline (3bq's own bug): champion_recipe(scale=1.0) reproduces champion_signal, which
assumes 6 bars/day = H4. XAU is H4 -> 1.0. BTC/NDX/BRENT are D1 -> 1/6.
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

TARGET_VOL, VOL_HL, MAX_LEV = 0.10, 42, 8.0
SPREAD_BP = {"XAU": None, "BTC": None, "NDX": 0.50, "BRENT": 7.58}   # one-way bp, measured
rng = np.random.default_rng(7)


def sleeve_daily(close: pd.Series, scale: float, cost_frac, bpy: float) -> pd.Series:
    ret = close.pct_change()
    vol = ret.ewm(halflife=VOL_HL, min_periods=20).std() * np.sqrt(bpy)
    fc = champion_recipe(close, scale, 1.5)
    pos = (fc.clip(-2, 2) * (TARGET_VOL / vol)).clip(-MAX_LEV, MAX_LEV).shift(1)
    r = (pos * ret - pos.diff().abs().fillna(0.0) * cost_frac).dropna()
    return (1 + r).resample("1D").prod().sub(1)


def perf(r, ppy=252):
    r = r.dropna()
    eq = (1 + r).cumprod()
    return dict(sharpe=float(r.mean()/r.std()*np.sqrt(ppy)), 
                cagr=float(eq.iloc[-1]**(ppy/len(r))-1),
                maxdd=float((eq/eq.cummax()-1).min()))


def ftmo(r, draws=4000, horizon=252, block=21):
    x = r.values; n = len(x); p = fd = ft = 0
    for _ in range(draws):
        idx = np.concatenate([np.arange(s, s+block)
                              for s in rng.integers(0, n-block, horizon//block+1)])[:horizon]
        d = x[idx]; eq = np.cumprod(1+d)
        if (d <= -0.05).any(): fd += 1; continue
        if (eq-1 <= -0.10).any(): ft += 1; continue
        if (eq-1 >= 0.10).any(): p += 1
    return p/draws, fd/draws, ft/draws


def main() -> int:
    S = {}
    # XAU: H4 rebuilt from M15, scale 1.0
    m15 = pd.read_csv(ROOT/"data/XAUUSD_M15_long.csv", parse_dates=["time"], index_col="time")
    m15 = m15[~m15.index.duplicated(keep="last")].sort_index()
    h4 = mtf_frames.aggregate(m15, "H4").dropna(subset=["close"]); xc = h4["close"].astype(float)
    xbpy = len(h4)/max((h4.index[-1]-h4.index[0]).days/365.25, 1e-9)
    S["XAU"] = sleeve_daily(xc, 1.0, 0.672/xc*0.75*2, xbpy)
    # D1 sleeves, scale 1/6
    for al, f in (("BTC", "BTC_D1_long.csv"), ("NDX", "NDX_D1_long.csv"),
                  ("BRENT", "BRENT_D1_long.csv")):
        d = pd.read_csv(ROOT/f"data/{f}", parse_dates=["time"], index_col="time")
        d = d[~d.index.duplicated(keep="last")].sort_index()
        c = d["close"].astype(float)
        bpy = len(d)/max((d.index[-1]-d.index[0]).days/365.25, 1e-9)
        cost = (1.0/c*0.75*2) if al == "BTC" else pd.Series(SPREAD_BP[al]/1e4*2, index=c.index)
        S[al] = sleeve_daily(c, 1/6, cost, bpy)

    lo = max(v.index[0] for v in S.values()); hi = min(v.index[-1] for v in S.values())
    bd = pd.bdate_range(lo.normalize(), hi.normalize())
    S = {k: v.reindex(bd).fillna(0.0) for k, v in S.items()}
    print("=" * 102)
    print("BOOK WIDTH — every combination through ONE harness, same engine, same 10% dial")
    print(f"  {bd[0].date()} .. {bd[-1].date()}  ({len(bd)} business days, {len(bd)/252:.1f}y)")
    print("=" * 102)

    books = {
        "XAU alone":                 {"XAU": 1.0},
        "BTC alone":                 {"BTC": 1.0},
        "XAU+BTC 50/50  (3bq, LIVE)": {"XAU": .5, "BTC": .5},
        "XAU+BTC+NDX":               {"XAU": 1/3, "BTC": 1/3, "NDX": 1/3},
        "core-4 equal":              {"XAU": .25, "BTC": .25, "NDX": .25, "BRENT": .25},
        "core-4 gold-tilted (book)": {"XAU": .50, "BTC": 1/6, "NDX": 1/6, "BRENT": 1/6},
    }
    print(f"\n  {'book':30s} {'Sharpe':>7s} {'CAGR':>8s} {'maxDD':>8s} "
          f"{'pass':>7s} {'failDAY':>8s} {'failTOT':>8s}")
    rows = []
    for name, w in books.items():
        r = sum(S[k]*v for k, v in w.items())
        m = perf(r); pa, fd, ft = ftmo(r)
        star = "  <-- LIVE" if "LIVE" in name else ""
        print(f"  {name:30s} {m['sharpe']:+7.3f} {100*m['cagr']:+7.1f}% {100*m['maxdd']:7.1f}% "
              f"{100*pa:6.1f}% {100*fd:7.1f}% {100*ft:7.1f}%{star}")
        rows.append(dict(book=name, **m, pass_rate=pa, fail_daily=fd, fail_total=ft))
    out = ROOT/"data/v5_runs/btc_vs_xau"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"book_width.csv", index=False)
    print(f"\n  wrote {out}/book_width.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
