#!/usr/bin/env python3
"""OCTOBER A5 — forward MAX ADVERSE EXCURSION as a regression target. Oracle first.

Pre-registered in data/v5_runs/october/PREREGISTRATION.md. Design in OCTOBER.MD.

WHY THIS TARGET. `BEST_FINDINGS` #5 names the binding constraint explicitly: *"this project's
binding constraint is not model capacity or feature representation. It is the conversion step --
forecast to net-of-cost position surviving a regime split."* Every approach in this round so far has
attacked the FORECAST. MAE attacks the CONVERSION: it is the quantity a stop actually faces, and it
is what decides position size under any risk-based rule.

It is also only the SECOND regression target ever used here. The first was §3at's pooled volatility
forecast, which produced a real and transferable R² of +0.2299 and then closed **on a negative
oracle** -- ORACLE-VOL −0.146, *"perfect vol foresight makes the book WORSE; no forecast can beat
knowing it exactly."* CONTROL #8's scope note exempts programmes that change the TARGET, so MAE is
admissible where a direction label would not be.

*** MAE IS NOT VOLATILITY, AND THAT IS THE WHOLE HYPOTHESIS. *** Volatility is a symmetric,
distributional quantity the champion already sizes against via `wilder_atr`. MAE is PATH-DEPENDENT
and ONE-SIDED: how far underwater a long goes before its horizon, which is exactly what a stop
faces and what vol-targeting cannot see. If §3at's result generalises to any risk quantity, this
closes the same way. If MAE is genuinely different, this is where it shows.

ORACLE FIRST, AND THE SCRIPT STOPS IF IT FAILS. Two oracle arms, both using the TRUE forward MAE:
  A. SIZE inversely to MAE -- perfect risk foresight, the direct analogue of §3at's ORACLE-VOL.
  B. SKIP the bars whose MAE is worst -- perfect stop-avoidance, which is what a trader means by
     "I would not have taken that trade".
If neither clears CONTROL #5's gate (dSharpe >= +0.20 AND t >= 2.5) the regression is not fitted.
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
HORIZONS = (6, 30)          # H4 bars: 1 day, 5 days


def load_h4() -> pd.DataFrame:
    """H4 rebuilt from M15 — never XAUUSD_H4_long.csv (spliced 2023-06-28, pre-2023 correlates
    0.18-0.40 with real broker quotes)."""
    m15 = pd.read_csv(ROOT / "data/XAUUSD_M15_long.csv", parse_dates=["time"], index_col="time")
    m15 = m15[~m15.index.duplicated(keep="last")].sort_index()
    return mtf_frames.aggregate(m15, "H4").dropna(subset=["close"])


def forward_mae(d: pd.DataFrame, h: int) -> pd.Series:
    """For a LONG opened at the close of t, the worst drawdown reached within the next h bars.

    Returns a POSITIVE magnitude (0 = never underwater). Uses the running minimum of LOW over the
    forward window, so it is the path the stop actually faces, not the close-to-close outcome.
    """
    c = d["close"].astype(float)
    lo = d["low"].astype(float)
    fwd_min = lo.shift(-1).rolling(h, min_periods=1).min().shift(-(h - 1))
    return (1.0 - fwd_min / c).clip(lower=0.0)


def perf(r, ppy):
    r = r.dropna()
    if len(r) < 50 or r.std() == 0:
        return dict(sharpe=np.nan, cagr=np.nan, maxdd=np.nan)
    eq = (1 + r).cumprod()
    return dict(sharpe=float(r.mean() / r.std() * np.sqrt(ppy)),
                cagr=float(eq.iloc[-1] ** (ppy / len(r)) - 1),
                maxdd=float((eq / eq.cummax() - 1).min()))


def vol_match(a, b):
    sa, sb = a.std(), b.std()
    return a * (sb / sa) if sa > 0 else a


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_h4()
    c = d["close"].astype(float)
    ret = c.pct_change()
    bpy = len(d) / max((d.index[-1] - d.index[0]).days / 365.25, 1e-9)
    vol = ret.ewm(halflife=VOL_HL, min_periods=20).std() * np.sqrt(bpy)
    fc = champion_recipe(c, 1.0, 1.5)
    pos = (fc.clip(0, 2) * (TARGET_VOL / vol)).clip(0, MAX_LEV).shift(1)
    base_r = (pos * ret - pos.diff().abs().fillna(0.0) * (RT_BP / 1e4)).dropna()
    mb = perf(base_r, bpy)

    print("=" * 104)
    print("OCTOBER A5 — FORWARD MAX ADVERSE EXCURSION as a regression target (oracle first)")
    print("=" * 104)
    print(f"  XAUUSD H4 rebuilt from M15  {d.index[0].date()} -> {d.index[-1].date()}  "
          f"{len(d):,} bars")
    print(f"  CONTROL — champion: Sharpe {mb['sharpe']:+.3f}  CAGR {100*mb['cagr']:+.1f}%  "
          f"maxDD {100*mb['maxdd']:.1f}%")

    for h in HORIZONS:
        mae = forward_mae(d, h)
        live = pos > 1e-9
        m = mae.where(live)
        print(f"\n  --- horizon {h} bars ({h/6:.0f} day{'s' if h > 6 else ''}) ---")
        print(f"  forward MAE while long: median {100*m.median():.3f}%  "
              f"p75 {100*m.quantile(.75):.3f}%  p95 {100*m.quantile(.95):.3f}%  "
              f"max {100*m.max():.2f}%")
        # how much does MAE differ from what trailing vol already predicts?
        pv = (vol / np.sqrt(bpy) * np.sqrt(h))
        rho = float(m.corr(pv))
        print(f"  corr(forward MAE, trailing vol x sqrt(h)) = {rho:+.4f}  "
              f"<- if ~1.0 the champion already sizes against it")

        print(f"\n  {'oracle arm':<34} {'Sharpe':>9} {'dSharpe':>9} {'t@matched':>10} {'maxDD':>9}")
        print("  " + "-" * 74)
        print(f"  {'champion (incumbent)':<34} {mb['sharpe']:>+9.3f} {'—':>9} {'—':>10} "
              f"{100*mb['maxdd']:>8.1f}%")
        rows = []
        # ---- ARM A: size inversely to the TRUE forward MAE (perfect risk foresight) ----------
        for floor in (0.002, 0.005):
            scale = (m.median() / m.clip(lower=floor)).clip(0.0, 3.0)
            tp = (pos * scale.reindex(pos.index).fillna(1.0)).clip(0, MAX_LEV)
            r = (tp * ret - tp.diff().abs().fillna(0.0) * (RT_BP / 1e4)).dropna()
            ii = base_r.index.intersection(r.index)
            a, b = r.reindex(ii), base_r.reindex(ii)
            dm = vol_match(a, b) - b
            t = float(dm.mean() / (dm.std() / np.sqrt(len(dm)))) if dm.std() > 0 else np.nan
            mm = perf(a, bpy)
            rows.append((f"A size 1/MAE (floor {floor:.1%})", mm, t))
        # ---- ARM B: skip the worst-MAE bars (perfect stop-avoidance) ------------------------
        for q in (0.90, 0.75, 0.50):
            thr = m.quantile(q)
            tp = pos.where(m.reindex(pos.index) <= thr, 0.0)
            r = (tp * ret - tp.diff().abs().fillna(0.0) * (RT_BP / 1e4)).dropna()
            ii = base_r.index.intersection(r.index)
            a, b = r.reindex(ii), base_r.reindex(ii)
            dm = vol_match(a, b) - b
            t = float(dm.mean() / (dm.std() / np.sqrt(len(dm)))) if dm.std() > 0 else np.nan
            mm = perf(a, bpy)
            rows.append((f"B skip MAE worse than p{int(q*100)}", mm, t))
        best = -9.9
        for lbl, mm, t in rows:
            dsh = mm["sharpe"] - mb["sharpe"]
            best = max(best, dsh)
            print(f"  {lbl:<34} {mm['sharpe']:>+9.3f} {dsh:>+9.3f} {t:>+10.2f} "
                  f"{100*mm['maxdd']:>8.1f}%")
        print(f"  best oracle dSharpe at h={h}: {best:+.3f}  (CONTROL #5 needs >= +0.20 at t >= 2.5)")

    print(f"""
  READ. 'corr(forward MAE, trailing vol)' is the question this arm exists to answer: if MAE is
  essentially trailing vol scaled by sqrt(h), then the champion's `wilder_atr` sizing ALREADY
  spends it and §3at's verdict transfers -- "perfect vol foresight makes the book WORSE". If the
  correlation is low, MAE is a genuinely different risk quantity and a negative oracle here is a
  NEW fact rather than a restatement.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
