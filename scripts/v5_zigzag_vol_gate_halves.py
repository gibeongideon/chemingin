#!/usr/bin/env python3
"""§3es's keeper, re-checked the hard way: is the gain there in BOTH HALVES, INCREMENTALLY?

USER, 2026-11-13: "re-run the half-sample check incrementally first, then next you wire up".

§3es left exactly one honest gap. Its both-halves check was run on the STANDALONE guard (vol gate
vs the unguarded rule), while the deployment test was INCREMENTAL (vol gate on top of the live
breaker / breaker+champion-gate). Those are different questions, and §3es itself showed how far
apart they can land: "trend only" went from a +0.257 Sharpe win standalone to the 61.5th
percentile — Sharpe −0.122 — once the champion gate was already present. So the standalone
half-sample result does not license the incremental rule.

WHAT IS CHECKED. For each source x live baseline, split the realised daily-return series at the
in-window midpoint and ask, SEPARATELY IN EACH HALF:
    does the vol gate beat a RANDOM veto that drops the same number of fires?
The random control is re-derived per half from the same 200 draws (each draw is one full
backtest, then its daily returns are split), so a half is never scored against the other half's
distribution. Trades are generated on the full sample because the rule is causal; only the
scoring is split.

THE BAR, SET BEFORE LOOKING: 4 of 4 halves at or above the 90th percentile, and none below the
50th. Requiring 95% in every half would be stricter than the §3es bar that produced the candidate
(95th in both FULL samples) and would reject on noise — a half has roughly half the data.
A guard that is real should at least never look like a below-median coin in any half.

    python scripts/v5_zigzag_vol_gate_halves.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_zigzag_downtrend_guards import guards  # noqa: E402
from scripts.v5_zigzag_elliott_veto import stats  # noqa: E402
from scripts.v5_zigzag_long_history import OUT as ZZ, SRC, THR, backtest, load, sharpe  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "elliott"
N_RANDOM = 200
RNG = np.random.default_rng(20261113)


def halves(ret: pd.Series, mid: pd.Timestamp) -> tuple:
    return ret[ret.index < mid], ret[ret.index >= mid]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, cells = [], []
    for src, cfg in SRC.items():
        g = load(src, cfg["gold"])
        p = (pd.read_csv(ZZ / src / "probs.csv", index_col=0, parse_dates=True)
             .iloc[:, 0].reindex(g.index).to_numpy())
        start, swap = cfg["start"], cfg["swap"]
        yrs = (g.index[-1] - pd.Timestamp(start)).days / 365.25
        inw = g.index >= pd.Timestamp(start)
        fire = np.isfinite(p) & (p >= THR)
        G = guards(g)
        vol = G["downside vol below median"]
        live_all = G["circuit breaker k=1.0 LIVE"]
        live_sw = live_all & G["champion fc >= 0.5  LIVE"]
        mid = g.index[inw][int(inw.sum() // 2)]

        print("=" * 114)
        print(f"{src.upper()}  {start} -> {g.index[-1]:%Y-%m-%d} ({yrs:.1f}y)   "
              f"split at {mid:%Y-%m-%d}   adding: downside_vol(500) < rolling_median(2000)")
        print("=" * 114)

        for bname, bmask in (("breaker only (5 of 6 configs)", live_all),
                             ("breaker + champion gate (Switch 360601)", live_sw)):
            fb = fire & bmask
            trb, retb = backtest(g, fb, start, swap)
            sb = stats(trb, retb, yrs)
            fa = fb & vol
            tra, reta = backtest(g, fa, start, swap)
            sa = stats(tra, reta, yrs)
            bidx = np.flatnonzero(fb & inw)
            ndrop = int((fb & ~vol & inw).sum())
            if len(tra) < 20 or not 0 < ndrop < len(bidx):
                print(f"  {bname}: unusable")
                continue

            # ONE set of draws; each is split into the same two halves as the real series.
            d1 = np.empty(N_RANDOM)
            d2 = np.empty(N_RANDOM)
            for i in range(N_RANDOM):
                ff = fb.copy()
                ff[RNG.choice(bidx, size=ndrop, replace=False)] = False
                r = backtest(g, ff, start, swap)[1]
                h1, h2 = halves(r, mid)
                d1[i], d2[i] = sharpe(h1), sharpe(h2)

            bh1, bh2 = halves(retb, mid)
            ah1, ah2 = halves(reta, mid)
            print(f"\n  BASELINE {bname:<42} n {sb['n']:>5}  full {sb['sharpe']:+.3f}  "
                  f"h1 {sharpe(bh1):+.3f}  h2 {sharpe(bh2):+.3f}")
            print(f"  + VOL GATE {'':<40} n {sa['n']:>5}  full {sa['sharpe']:+.3f}  "
                  f"h1 {sharpe(ah1):+.3f}  h2 {sharpe(ah2):+.3f}")
            for lbl, a, b, dist in (("1st half", ah1, bh1, d1), ("2nd half", ah2, bh2, d2)):
                s_, base_ = sharpe(a), sharpe(b)
                pct = float((dist < s_).mean())
                flag = "PASS" if pct >= 0.90 else ("below median" if pct < 0.50 else "weak")
                print(f"      {lbl}: gated {s_:+.3f} vs baseline {base_:+.3f} "
                      f"({s_ - base_:+.3f})   random median {np.median(dist):+.3f} "
                      f"[p5 {np.percentile(dist, 5):+.3f}, p95 {np.percentile(dist, 95):+.3f}]"
                      f"   pctile {100 * pct:5.1f}%  {flag}")
                cells.append((f"{src}/{bname[:14]}/{lbl}", pct))
                rows.append(dict(src=src, baseline=bname, half=lbl, gated=s_, base=base_,
                                 delta=s_ - base_, rnd_med=float(np.median(dist)), pctile=pct))

    print("\n" + "=" * 114)
    print("VERDICT — bar set before looking: 4/4 halves >= 90th percentile, none below the 50th")
    print("=" * 114)
    for name, pct in cells:
        print(f"  {name:<40}{100 * pct:>8.1f}%   "
              f"{'ok' if pct >= 0.90 else ('BELOW MEDIAN' if pct < 0.50 else 'weak')}")
    ok = sum(1 for _, p_ in cells if p_ >= 0.90)
    worst = min(p_ for _, p_ in cells) if cells else 0.0
    print(f"\n  {ok}/{len(cells)} halves at/above the 90th percentile; worst half "
          f"{100 * worst:.1f}%")
    print(f"  -> {'HOLDS — proceed to a live A/B on one arm' if ok == len(cells) and worst >= 0.5 else 'DOES NOT HOLD — do not deploy'}")
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "vol_gate_halves.csv", index=False)
        print(f"\nwrote {OUT / 'vol_gate_halves.csv'}")


if __name__ == "__main__":
    main()
