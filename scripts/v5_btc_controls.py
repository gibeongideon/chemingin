"""BTC trend lead -> the MANDATORY CONTROLS. A promising first pass is not a result.

The scan found champion-recipe trend on BTC beating buy-and-hold at t +2.4..+3.0 in 10-11 of 11
years. Before that can be called anything, four controls that have each killed a lead here:

  #3  WALK-FORWARD THE SELECTION, not just the test. The scan compared three speeds on the FULL
      sample and reported the best. Choosing the universe/parameter in-sample roughly doubled an
      apparent Sharpe once (quality-17: +1.03 -> +0.26 under walk-forward re-picking). Here the
      speed is RE-PICKED each January on a trailing 3-year window and applied strictly forward.
  #4  QUOTE A STANDARD ERROR. Lo (2002): SE(SR) = sqrt((1+SR^2/2)/years).
  #6  DSR >= 0.90 AND PBO < 0.30 over the whole grid.
  1.3.8 SPLIT-SAMPLE. A pooled t of +2.25 with 7/9 positive years was still a post-2022 artifact.

Plus a BTC-specific one the scan flagged itself: the spread was 22bp at the start of the sample
and is 0.16bp today, and FTMO did not list BTC in 2014. A cost STRESS arm re-runs everything at
10x the modelled spread to see whether the early-sample edge is even realisable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from v5_xau_champion_lifts import champion_recipe, vol_match         # noqa: E402
from src.evaluation.dsr_pbo import deflated_sharpe_ratio, pbo_cscv  # noqa: E402

SPREAD_USD, TARGET_VOL, VOL_HL, MAX_LEV = 1.0, 0.10, 42, 8.0
SPEEDS = {"1/3": 1/3, "1/6": 1/6, "1/12": 1/12, "1/24": 1/24, "1/48": 1/48}


def build(cost_mult: float = 1.0):
    d = pd.read_csv(ROOT / "data/BTC_D1_long.csv", parse_dates=["time"], index_col="time")
    d = d[~d.index.duplicated(keep="last")].sort_index()
    c = d["close"].astype(float); ret = c.pct_change()
    bpy = len(d) / max((d.index[-1] - d.index[0]).days / 365.25, 1e-9)
    vol = ret.ewm(halflife=VOL_HL, min_periods=20).std() * np.sqrt(bpy)
    cost = (SPREAD_USD * cost_mult) / c * 0.75 * 2
    bh = ((TARGET_VOL / vol).clip(0, MAX_LEV).shift(1) * ret).dropna()
    arms = {}
    for name, s in SPEEDS.items():
        fc = champion_recipe(c, s, 1.5)
        pos = (fc.clip(-2, 2) * (TARGET_VOL / vol)).clip(-MAX_LEV, MAX_LEV).shift(1)
        arms[name] = (pos * ret - pos.diff().abs().fillna(0.0) * cost).dropna()
    idx = pd.DatetimeIndex(sorted(set.intersection(*[set(v.index) for v in arms.values()],
                                                   set(bh.index))))
    return {k: v.reindex(idx) for k, v in arms.items()}, bh.reindex(idx), bpy, idx


def sharpe(r, bpy):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(bpy)) if len(r) > 30 and r.std() > 0 else np.nan


def main() -> int:
    for mult, tag in ((1.0, "modelled spread ($1)"), (10.0, "STRESS: 10x spread")):
        arms, bh, bpy, idx = build(mult)
        print("=" * 100)
        print(f"BTC CONTROLS — {tag}")
        print("=" * 100)

        # ---- #3: walk-forward RE-SELECTION of the speed, re-picked each January on trailing 3y
        picks, wf = {}, []
        for y in range(idx[0].year + 3, idx[-1].year + 1):
            tr = idx[(idx.year >= y - 3) & (idx.year < y)]
            te = idx[idx.year == y]
            if len(tr) < 300 or len(te) < 30:
                continue
            best = max(SPEEDS, key=lambda k: sharpe(arms[k].reindex(tr), bpy))
            picks[y] = best
            wf.append(arms[best].reindex(te))
        wf = pd.concat(wf).dropna()
        wf_sr = sharpe(wf, bpy)
        full_best = max(SPEEDS, key=lambda k: sharpe(arms[k], bpy))
        full_sr = sharpe(arms[full_best], bpy)
        print(f"  full-sample best speed       {full_best:>5s}  Sharpe {full_sr:+.3f}   <- in-sample pick")
        print(f"  WALK-FORWARD re-selected            Sharpe {wf_sr:+.3f}   "
              f"<- the honest number  (cost of in-sample selection: {full_sr-wf_sr:+.3f})")
        print(f"  speeds chosen by year: {', '.join(f'{y}:{p}' for y, p in picks.items())}")

        # vs buy-and-hold, matched vol
        dd = vol_match(wf, bh.reindex(wf.index)) - bh.reindex(wf.index)
        t = float(dd.mean() / dd.std() * np.sqrt(len(dd))) if dd.std() > 0 else np.nan
        yr = wf.groupby(wf.index.year).sum(); yb = bh.reindex(wf.index).groupby(wf.index.year).sum()
        print(f"  walk-forward vs BUY-AND-HOLD at matched vol: t {t:+.2f}, "
              f"{int((yr > yb).sum())}/{len(yr)} years")

        # ---- #4 standard error
        yrs = len(wf) / bpy
        se = np.sqrt((1 + wf_sr**2 / 2) / yrs)
        print(f"  #4  SE(Sharpe) over {yrs:.1f}y = {se:.3f}  -> 95% CI "
              f"[{wf_sr-1.96*se:+.2f}, {wf_sr+1.96*se:+.2f}]")

        # ---- 1.3.8 split halves on the WALK-FORWARD series
        h1 = wf[wf.index.year <= 2020]; h2 = wf[wf.index.year >= 2021]
        b1 = bh.reindex(h1.index); b2 = bh.reindex(h2.index)
        print(f"  1.3.8 halves (walk-forward): <=2020 {sharpe(h1,bpy):+.3f} vs B&H "
              f"{sharpe(b1,bpy):+.3f} | >=2021 {sharpe(h2,bpy):+.3f} vs B&H {sharpe(b2,bpy):+.3f}")

        # ---- #6 DSR / PBO over the grid
        M = np.column_stack([arms[k].values for k in SPEEDS])
        trial_sr = [sharpe(arms[k], bpy) for k in SPEEDS]
        dsr = deflated_sharpe_ratio(arms[full_best].values, trial_sr)
        pbo = pbo_cscv(M, n_partitions=10)
        print(f"  #6  DSR {dsr['dsr']:.3f} (bar >=0.90)   PBO {pbo.pbo:.3f} (bar <0.30)   "
              f"-> {'PASS' if dsr['dsr'] >= 0.90 and pbo.pbo < 0.30 else 'FAIL'}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
