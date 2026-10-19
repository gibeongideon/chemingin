"""BTC trend vs the XAU champion, head to head. Pre-registered at
data/v5_runs/btc_vs_xau/PREREGISTRATION.md (commit 0962755) before this was run.

SAME ENGINE FOR BOTH. Each signal is run through the identical continuous vol-targeted engine at
the same 10% dial, so the only thing varying is the ASSET + SIGNAL, not the execution machinery.
(The XAU champion's live form is the discrete trail-stop engine; that is a separate axis measured
in 3bn, and mixing the two here would confound asset with engine.)

THE DECIDING STATISTIC IS NOT SHARPE. `book-shootout` established that Sharpe does not predict
prop-firm pass rate -- the dial-scaled LEFT TAIL against the daily limit does. So the FTMO 2-Step
phase-1 rules are simulated directly: +10% target, 5% daily loss, 10% overall loss.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from v5_xau_champion_lifts import champion_recipe, vol_match          # noqa: E402
from src.evaluation.dsr_pbo import deflated_sharpe_ratio, pbo_cscv    # noqa: E402
from src.v5 import mtf_frames                                         # noqa: E402

TARGET_VOL, VOL_HL, MAX_LEV = 0.10, 42, 8.0
rng = np.random.default_rng(7)


def sleeve(close: pd.Series, scale: float, cost_frac: pd.Series, bpy: float) -> pd.Series:
    ret = close.pct_change()
    vol = ret.ewm(halflife=VOL_HL, min_periods=20).std() * np.sqrt(bpy)
    fc = champion_recipe(close, scale, 1.5)
    pos = (fc.clip(-2, 2) * (TARGET_VOL / vol)).clip(-MAX_LEV, MAX_LEV).shift(1)
    return (pos * ret - pos.diff().abs().fillna(0.0) * cost_frac).dropna()


def bh(close: pd.Series, bpy: float) -> pd.Series:
    ret = close.pct_change()
    vol = ret.ewm(halflife=VOL_HL, min_periods=20).std() * np.sqrt(bpy)
    return ((TARGET_VOL / vol).clip(0, MAX_LEV).shift(1) * ret).dropna()


def perf(r, ppy=252):
    r = r.dropna()
    if len(r) < 30 or r.std() == 0:
        return dict(sharpe=np.nan, cagr=np.nan, maxdd=np.nan)
    eq = (1 + r).cumprod()
    return dict(sharpe=float(r.mean()/r.std()*np.sqrt(ppy)),
                cagr=float(eq.iloc[-1]**(ppy/len(r)) - 1),
                maxdd=float((eq/eq.cummax() - 1).min()))


def ftmo_phase1(r: pd.Series, draws=4000, horizon=252, block=21) -> dict:
    """FTMO 2-Step phase 1: +10% target, 5% DAILY loss, 10% overall. Daily limit measured on
    the day's own return, overall from the initial balance -- the prop-firm convention."""
    x = r.values; n = len(x)
    passed = fail_day = fail_tot = 0
    for _ in range(draws):
        idx = np.concatenate([np.arange(s, s+block)
                              for s in rng.integers(0, n-block, horizon//block+1)])[:horizon]
        d = x[idx]
        eq = np.cumprod(1 + d)
        if (d <= -0.05).any():
            fail_day += 1; continue
        if (eq - 1 <= -0.10).any():
            fail_tot += 1; continue
        if (eq - 1 >= 0.10).any():
            passed += 1
    return dict(pass_rate=passed/draws, fail_daily=fail_day/draws, fail_total=fail_tot/draws)


def main() -> int:
    # --- XAU: H4 rebuilt from M15 (the filed H4 is FTMO-spliced from 2023-06-28)
    m15 = pd.read_csv(ROOT/"data/XAUUSD_M15_long.csv", parse_dates=["time"], index_col="time")
    m15 = m15[~m15.index.duplicated(keep="last")].sort_index()
    h4 = mtf_frames.aggregate(m15, "H4").dropna(subset=["close"])
    xc = h4["close"].astype(float)
    xbpy = len(h4) / max((h4.index[-1]-h4.index[0]).days/365.25, 1e-9)
    xcost = pd.Series(0.672/xc * 0.75*2, index=xc.index)      # corrected round trip, Gate A3

    # --- BTC D1
    b = pd.read_csv(ROOT/"data/BTC_D1_long.csv", parse_dates=["time"], index_col="time")
    b = b[~b.index.duplicated(keep="last")].sort_index()
    bc = b["close"].astype(float)
    bbpy = len(b) / max((b.index[-1]-b.index[0]).days/365.25, 1e-9)
    bcost = 1.0/bc * 0.75*2

    # *** THE SCALE IS PER-TIMEFRAME, NOT GLOBAL. *** champion_recipe's docstring:
    # "scale=1.0, conc_p=1.5 reproduces champion_signal", and champion_signal hardcodes 6
    # bars/day = H4. So H4 gold takes scale 1.0 and D1 BTC takes 1/6 (rule 1.3.5, the D=6 trap).
    # The first run of this script passed 1/6 to BOTH, making the GOLD signal six times too slow
    # and collapsing its Sharpe to 0.476 against the 1.091 3bn measured -- the D=6 trap inverted.
    # The tell was that cell A, whose answer was already known, came back wrong.
    A = sleeve(xc, 1.0, xcost, xbpy);  A_bh = bh(xc, xbpy)     # H4  -> scale 1.0
    B = sleeve(bc, 1/6, bcost, bbpy);  B_bh = bh(bc, bbpy)     # D1  -> scale 1/6

    # --- rule 1.3.6: compound onto a COMMON business-day index before annualising
    def daily(r):
        return (1 + r).resample("1D").prod().sub(1)
    lo = max(A.index[0], B.index[0]); hi = min(A.index[-1], B.index[-1])
    bd = pd.bdate_range(lo.normalize(), hi.normalize())
    Ad, Bd = daily(A).reindex(bd).fillna(0.0), daily(B).reindex(bd).fillna(0.0)
    Abh, Bbh = daily(A_bh).reindex(bd).fillna(0.0), daily(B_bh).reindex(bd).fillna(0.0)
    AB = 0.5*Ad + 0.5*Bd

    print("=" * 104)
    print("BTC TREND vs THE XAU CHAMPION — same engine, same 10% dial, common business-day index")
    print(f"  {bd[0].date()} .. {bd[-1].date()}  ({len(bd)} business days, {len(bd)/252:.1f}y)")
    print("=" * 104)
    print(f"\n  {'cell':28s} {'Sharpe':>7s} {'CAGR':>8s} {'maxDD':>8s} "
          f"{'vs own B&H':>11s} {'1st half':>9s} {'2nd half':>9s}")
    rows = {}
    for name, r, bmk in (("A  XAU champion (H4, 1.0)", Ad, Abh), ("B  BTC trend (D1, 1/6)", Bd, Bbh),
                         ("A+B  50/50", AB, 0.5*Abh + 0.5*Bbh)):
        m = perf(r)
        dvb = vol_match(r, bmk) - bmk
        tb = float(dvb.mean()/dvb.std()*np.sqrt(len(dvb))) if dvb.std() > 0 else np.nan
        h1, h2 = r[r.index.year <= 2020], r[r.index.year >= 2021]
        k1, k2 = bmk[bmk.index.year <= 2020], bmk[bmk.index.year >= 2021]
        s1 = perf(h1)["sharpe"] - perf(k1)["sharpe"]
        s2 = perf(h2)["sharpe"] - perf(k2)["sharpe"]
        print(f"  {name:28s} {m['sharpe']:+7.3f} {100*m['cagr']:+7.1f}% {100*m['maxdd']:7.1f}% "
              f"{tb:+11.2f} {s1:+9.3f} {s2:+9.3f}")
        rows[name] = r
    print("   (vs own B&H = paired t at matched vol; halves = Sharpe EDGE over own benchmark)")

    # --- the head-to-head paired test
    dd = vol_match(Bd, Ad) - Ad
    t = float(dd.mean()/dd.std()*np.sqrt(len(dd)))
    yb = (Bd.groupby(Bd.index.year).sum() > Ad.groupby(Ad.index.year).sum()).sum()
    ny = Bd.index.year.nunique()
    print(f"\n  HEAD-TO-HEAD  B vs A at matched vol: paired t {t:+.2f}  "
          f"({yb}/{ny} years B beats A)   gate is t >= +2.0")

    # --- the statistic that actually decides it
    print(f"\n  FTMO 2-Step PHASE 1 simulation (1y horizon, 4,000 block-bootstrap paths):")
    print(f"    {'cell':20s} {'pass':>7s} {'fail DAILY 5%':>15s} {'fail total 10%':>15s}")
    pr = {}
    for name, r in rows.items():
        f = ftmo_phase1(r); pr[name] = f
        print(f"    {name:20s} {100*f['pass_rate']:6.1f}% {100*f['fail_daily']:14.1f}% "
              f"{100*f['fail_total']:14.1f}%")

    # --- CONTROL #6 on the two-cell grid
    M = np.column_stack([Ad.values, Bd.values])
    srs = [perf(Ad)["sharpe"], perf(Bd)["sharpe"]]
    dsr = deflated_sharpe_ratio(Bd.values, srs); pbo = pbo_cscv(M, n_partitions=10)
    print(f"\n  #6 on the declared 2-cell grid: DSR {dsr['dsr']:.3f} (bar >=0.90)  "
          f"PBO {pbo.pbo:.3f} (bar <0.30)")

    out = ROOT/"data/v5_runs/btc_vs_xau"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"daily.csv")
    print(f"\n  wrote {out}/daily.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
