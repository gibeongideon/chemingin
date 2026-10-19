"""Can XAU+BTC work at Switch Markets? Re-run the book at each broker's MEASURED cost basis.

Spreads and carry measured live 2026-10-19, 10 tick samples per symbol:

                  spread            carry on a LONG
  SM   XAUUSDs    $0.210 = 0.50bp   -5.70 %/yr   (swap_long -65.07, POINTS)
  SM   BTCUSDs    $36.82 = 4.46bp   -15.00 %/yr  (swap_long -15.0, INTEREST %/yr)
  FTMO XAUUSD     $0.410 = 0.99bp   -6.47 %/yr
  FTMO BTCUSD     $1.00  = 0.12bp   -30.00 %/yr

The two brokers pull in OPPOSITE directions and the answer is not guessable: spread scales with
TURNOVER, carry with HOLDING TIME, and this book holds for weeks. FTMO's BTC spread is 37x
cheaper while Switch Markets' BTC carry is half. So both are priced into the same engine.

CARRY WAS MISSING FROM 3bq/3br. Those runs charged spread only. Financing is the largest measured
drag on this book (-3.79%/yr, worth -0.29 Sharpe), so leaving it out flattered both. It is
included here, charged on the ABSOLUTE EXPOSURE each bar -- carry is paid on notional held, not on
notional traded.
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
rng = np.random.default_rng(7)

BROKERS = {
    "FTMO":           dict(xau_bp=0.99, xau_carry=-6.47, btc_bp=0.12, btc_carry=-30.0),
    "Switch Markets": dict(xau_bp=0.50, xau_carry=-5.70, btc_bp=4.46, btc_carry=-15.0),
    "SM gold + FTMO BTC (best of each)":
                      dict(xau_bp=0.50, xau_carry=-5.70, btc_bp=0.12, btc_carry=-30.0),
}


def sleeve(close, scale, bpy, spread_bp, carry_pct_yr):
    ret = close.pct_change()
    vol = ret.ewm(halflife=VOL_HL, min_periods=20).std() * np.sqrt(bpy)
    fc = champion_recipe(close, scale, 1.5)
    pos = (fc.clip(-2, 2) * (TARGET_VOL / vol)).clip(-MAX_LEV, MAX_LEV).shift(1)
    trade_cost = pos.diff().abs().fillna(0.0) * (spread_bp / 1e4) * 1.5   # round trip = 1.5x spread
    carry_cost = pos.abs() * (abs(carry_pct_yr) / 100.0) / bpy            # paid on what is HELD
    r = (pos * ret - trade_cost - carry_cost).dropna()
    turns = float(pos.diff().abs().sum() / (len(pos) / bpy))
    return (1 + r).resample("1D").prod().sub(1), turns


def perf(r, ppy=252):
    r = r.dropna(); eq = (1 + r).cumprod()
    return dict(sharpe=float(r.mean()/r.std()*np.sqrt(ppy)),
                cagr=float(eq.iloc[-1]**(ppy/len(r))-1),
                maxdd=float((eq/eq.cummax()-1).min()))


def ftmo_pass(r, draws=4000, horizon=252, block=21):
    x = r.values; n = len(x); p = fd = 0
    for _ in range(draws):
        idx = np.concatenate([np.arange(s, s+block)
                              for s in rng.integers(0, n-block, horizon//block+1)])[:horizon]
        d = x[idx]; eq = np.cumprod(1+d)
        if (d <= -0.05).any(): fd += 1; continue
        if (eq-1 <= -0.10).any(): continue
        if (eq-1 >= 0.10).any(): p += 1
    return p/draws, fd/draws


def main() -> int:
    m15 = pd.read_csv(ROOT/"data/XAUUSD_M15_long.csv", parse_dates=["time"], index_col="time")
    m15 = m15[~m15.index.duplicated(keep="last")].sort_index()
    h4 = mtf_frames.aggregate(m15, "H4").dropna(subset=["close"]); xc = h4["close"].astype(float)
    xbpy = len(h4)/max((h4.index[-1]-h4.index[0]).days/365.25, 1e-9)
    b = pd.read_csv(ROOT/"data/BTC_D1_long.csv", parse_dates=["time"], index_col="time")
    b = b[~b.index.duplicated(keep="last")].sort_index(); bc = b["close"].astype(float)
    bbpy = len(b)/max((b.index[-1]-b.index[0]).days/365.25, 1e-9)

    print("=" * 104)
    print("CAN XAU+BTC WORK AT SWITCH MARKETS? — same engine, each broker's MEASURED costs")
    print("  spread AND carry both charged; 3bq/3br charged spread only and so flattered both")
    print("=" * 104)
    print(f"\n  {'broker':34s} {'Sharpe':>7s} {'CAGR':>8s} {'maxDD':>8s} {'pass':>7s} "
          f"{'failDAY':>8s}  {'XAU turns':>9s} {'BTC turns':>9s}")
    base = None
    for name, c in BROKERS.items():
        X, xt = sleeve(xc, 1.0, xbpy, c["xau_bp"], c["xau_carry"])
        B, bt = sleeve(bc, 1/6, bbpy, c["btc_bp"], c["btc_carry"])
        bd = pd.bdate_range(max(X.index[0], B.index[0]).normalize(),
                            min(X.index[-1], B.index[-1]).normalize())
        r = 0.5*X.reindex(bd).fillna(0.0) + 0.5*B.reindex(bd).fillna(0.0)
        m = perf(r); pa, fd = ftmo_pass(r)
        if base is None: base = m["sharpe"]
        print(f"  {name:34s} {m['sharpe']:+7.3f} {100*m['cagr']:+7.1f}% {100*m['maxdd']:7.1f}% "
              f"{100*pa:6.1f}% {100*fd:7.1f}%  {xt:9.1f} {bt:9.1f}")
    print("\n  turns = round trips per year per sleeve (what spread is charged on)")
    print("  NOTE 3bq filed Sharpe +1.511 for this book with NO carry. Carry is the difference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
