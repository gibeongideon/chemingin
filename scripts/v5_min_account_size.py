"""Minimum account size for the XAU+BTC book — from the HISTORICAL position distribution.

Today's snapshot is misleading: the gold sleeve currently targets 0.0120 lots and rounds to the
0.01 floor with a -17% error, but target size moves with the forecast and with volatility, so the
question "what is the minimum account" is really "below what equity does a sleeve spend too much
of its life rounding to ZERO".

A sleeve that rounds to zero is not merely mis-sized -- it is ABSENT, and 3br measured exactly
what that costs: dropping to XAU alone takes the FTMO daily-breach rate from 0.0% to 17.5% and
the pass rate from 72.4% to 53.7%. So the floor is set by the sleeve that disappears first.

Reported per account size: the share of bars each sleeve rounds to zero, the median absolute
rounding error, and the resulting verdict.
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
VMIN, STEP = 0.01, 0.01
SIZES = (250_000, 100_000, 50_000, 25_000, 10_000, 5_000)


def target_lots(close: pd.Series, scale: float, bpy: float, contract: float, weight: float):
    """The engine's own sizing: notional/equity = forecast x (target_vol / realised_vol)."""
    ret = close.pct_change()
    vol = ret.ewm(halflife=VOL_HL, min_periods=20).std() * np.sqrt(bpy)
    fc = champion_recipe(close, scale, 1.5)
    lev = (fc.clip(-2, 2) * (TARGET_VOL / vol)).clip(-MAX_LEV, MAX_LEV) * weight
    return (lev / (contract * close)).dropna(), close


def main() -> int:
    m15 = pd.read_csv(ROOT/"data/XAUUSD_M15_long.csv", parse_dates=["time"], index_col="time")
    m15 = m15[~m15.index.duplicated(keep="last")].sort_index()
    h4 = mtf_frames.aggregate(m15, "H4").dropna(subset=["close"])
    xc = h4["close"].astype(float)
    xbpy = len(h4)/max((h4.index[-1]-h4.index[0]).days/365.25, 1e-9)

    b = pd.read_csv(ROOT/"data/BTC_D1_long.csv", parse_dates=["time"], index_col="time")
    b = b[~b.index.duplicated(keep="last")].sort_index()
    bc = b["close"].astype(float)
    bbpy = len(b)/max((b.index[-1]-b.index[0]).days/365.25, 1e-9)

    # lots-per-dollar-of-equity, per sleeve, 50/50 class weight
    sleeves = {
        "XAUUSD": target_lots(xc, 1.0, xbpy, 100.0, 0.5),      # H4, contract 100 oz
        "BTCUSD": target_lots(bc, 1/6, bbpy, 1.0, 0.5),        # D1, contract 1 BTC
    }
    print("=" * 94)
    print("MINIMUM ACCOUNT SIZE — XAU+BTC 50/50, from the historical position distribution")
    print(f"  lot step {STEP}, minimum {VMIN} on both symbols (FTMO)")
    print(f"  a sleeve rounding to ZERO is ABSENT, and 3br measured that at")
    print(f"  pass 72.4% -> 53.7% and daily-breach 0.0% -> 17.5% when gold runs alone")
    print("=" * 94)
    print(f"\n  {'equity':>9s}  {'XAU drops':>10s} {'XAU err':>9s}  {'BTC drops':>10s} "
          f"{'BTC err':>9s}   verdict")
    rows = []
    for E in SIZES:
        cells, verdict = {}, []
        for name, (lpd, px) in sleeves.items():
            raw = (lpd * E).abs()
            raw = raw[raw > 0]
            rounded = (raw / STEP).round() * STEP
            drop = float((rounded < VMIN).mean())
            live = raw[rounded >= VMIN]
            err = float((((live/STEP).round()*STEP - live).abs() / live).median()) if len(live) else np.nan
            cells[name] = (drop, err)
            if drop > 0.50:
                verdict.append(f"{name} ABSENT {100*drop:.0f}% of the time")
            elif drop > 0.10:
                verdict.append(f"{name} drops {100*drop:.0f}%")
        v = "; ".join(verdict) if verdict else "both sleeves sizable"
        xa, ba = cells["XAUUSD"], cells["BTCUSD"]
        print(f"  {E:>9,}  {100*xa[0]:9.1f}% {100*xa[1]:8.1f}%  {100*ba[0]:9.1f}% "
              f"{100*ba[1]:8.1f}%   {v}")
        rows.append(dict(equity=E, xau_drop=xa[0], xau_err=xa[1],
                         btc_drop=ba[0], btc_err=ba[1], verdict=v))

    print("\n  --- the binding sleeve ---")
    xl, _ = sleeves["XAUUSD"]; bl, _ = sleeves["BTCUSD"]
    for name, lpd in (("XAUUSD", xl), ("BTCUSD", bl)):
        need = (VMIN / 2) / lpd.abs().replace(0, np.nan)      # equity to reach the 0.005 round-up
        print(f"  {name}: equity needed to size at all — median ${need.median():,.0f}, "
              f"25th pct ${need.quantile(.25):,.0f}, 75th pct ${need.quantile(.75):,.0f}")
    out = ROOT/"data/v5_runs/btc_vs_xau"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"min_account_size.csv", index=False)
    print(f"\n  wrote {out}/min_account_size.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
