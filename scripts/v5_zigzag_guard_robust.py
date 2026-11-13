#!/usr/bin/env python3
"""The two guards that cleared both sources: do they hold in both HALVES, and does combining help?

Follow-up to `v5_zigzag_downtrend_guards.py`. Twelve guards were tried there; two reached the
95th percentile against a random same-size drop in BOTH sources:
    * 120-bar return > 0        (trend)
    * downside vol below median (regime)
Everything else -- including both guards that are LIVE today -- cleared CME and missed Switch.

THREE THINGS CHECKED HERE, because a cell that clears a null is not yet a deployable rule.
 1. HALF-SAMPLE STABILITY. Split each source at its midpoint and require the guard to help in
    BOTH halves. §3dn/§3dx killed a 1.58 Sharpe this way -- it was 2024-26 only.
 2. THE TREND FAMILY IS ~ONE BET, NOT FIVE. SMA50, SMA200, EWMAC 16/64, EWMAC 32/128 and the
    120-bar return are near-duplicates (they agree on 80-95% of bars). That "120-bar return"
    cleared Switch at 97.0% while SMA50 missed at 93.5% is within the noise of one correlated
    family, so the family's AGREEMENT RATE is printed -- if they are one bet, the honest reading
    is a family that straddles the bar, not a winner.
 3. COMBINATION. trend AND low-vol, trend OR low-vol, scored on the same random-drop control.

    python scripts/v5_zigzag_guard_robust.py
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
TREND_FAMILY = ("close > SMA50", "close > SMA200", "EWMAC 16/64 > 0",
                "EWMAC 32/128 > 0", "120-bar return > 0")


def main() -> None:
    rows = []
    for src, cfg in SRC.items():
        g = load(src, cfg["gold"])
        p = (pd.read_csv(ZZ / src / "probs.csv", index_col=0, parse_dates=True)
             .iloc[:, 0].reindex(g.index).to_numpy())
        start, swap = cfg["start"], cfg["swap"]
        fire = np.isfinite(p) & (p >= THR)
        inw = g.index >= pd.Timestamp(start)
        fire_idx = np.flatnonzero(fire & inw)
        G = guards(g)
        trend, vol = G["120-bar return > 0"], G["downside vol below median"]
        mid = g.index[inw][int(inw.sum() // 2)]
        yrs = (g.index[-1] - pd.Timestamp(start)).days / 365.25

        print("=" * 112)
        print(f"{src.upper()}  split at {mid:%Y-%m-%d}")
        print("=" * 112)

        # --- 2. how much of ONE bet is the trend family?
        fam = {k: G[k][inw] for k in TREND_FAMILY}
        print("  trend-family agreement (share of in-window bars where the pair agrees):")
        ks = list(fam)
        for i, a in enumerate(ks):
            print("    " + "  ".join(f"{np.mean(fam[a] == fam[b]):.2f}" for b in ks[:i + 1])
                  + f"   {a}")

        base_tr, base_ret = backtest(g, fire, start, swap)
        b0 = stats(base_tr, base_ret, yrs)
        print(f"\n  {'variant':<28}{'n':>6}{'Sharpe':>8}{'dSh':>7}{'CAGR':>8}{'maxDD':>8}"
              f"{'pctile':>8}{'Sh 1st':>8}{'Sh 2nd':>8}  both halves?")
        print(f"  {'BASE (no guard)':<28}{b0['n']:>6}{b0['sharpe']:>8.3f}{0.0:>7.3f}"
              f"{100 * b0['cagr']:>7.2f}%{100 * b0['maxdd']:>7.2f}%{'':>8}"
              f"{sharpe(base_ret[base_ret.index < mid]):>8.3f}"
              f"{sharpe(base_ret[base_ret.index >= mid]):>8.3f}")

        for tag, ok in (("120-bar return > 0", trend),
                        ("downside vol < median", vol),
                        ("trend AND low-vol", trend & vol),
                        ("trend OR low-vol", trend | vol)):
            f = fire & ok
            ndrop = int((fire & ~ok & inw).sum())
            tr, ret = backtest(g, f, start, swap)
            if len(tr) < 20 or not 0 < ndrop < len(fire_idx):
                print(f"  {tag:<28} unusable")
                continue
            s = stats(tr, ret, yrs)
            sh = np.empty(N_RANDOM)
            for i in range(N_RANDOM):
                ff = fire.copy()
                ff[RNG.choice(fire_idx, size=ndrop, replace=False)] = False
                sh[i] = sharpe(backtest(g, ff, start, swap)[1])
            pct = float((sh < s["sharpe"]).mean())
            h1, h2 = sharpe(ret[ret.index < mid]), sharpe(ret[ret.index >= mid])
            b1, b2 = (sharpe(base_ret[base_ret.index < mid]),
                      sharpe(base_ret[base_ret.index >= mid]))
            both = (h1 > b1) and (h2 > b2)
            print(f"  {tag:<28}{s['n']:>6}{s['sharpe']:>8.3f}"
                  f"{s['sharpe'] - b0['sharpe']:>+7.3f}{100 * s['cagr']:>7.2f}%"
                  f"{100 * s['maxdd']:>7.2f}%{100 * pct:>7.1f}%{h1:>8.3f}{h2:>8.3f}"
                  f"   {'YES' if both else 'no'}")
            rows.append(dict(src=src, guard=tag, **s, pctile=pct, sh_h1=h1, sh_h2=h2,
                             base_h1=b1, base_h2=b2, both_halves=both))
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "guard_robustness.csv", index=False)
        print(f"\nwrote {OUT / 'guard_robustness.csv'}")


if __name__ == "__main__":
    main()
