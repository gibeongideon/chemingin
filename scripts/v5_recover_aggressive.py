"""THE AGGRESSIVE RECOVER-AND-GAIN LADDER — how far it escalates, and what it costs.

Spec asked for: after a loss, size the NEXT trade so it recovers the loss AND earns the intended
profit. That is `recover4` in v5_xau_fade_martingale.simulate -- `stake = (deficit + base)/w` --
but §3bo only tested it CAPPED at K = 3/4/6 attempts before giving up. This measures the
UNCAPPED form the spec actually describes, plus the stake ceiling `cap` that is the only thing
standing between it and the wall.

TWO THINGS §3bo DID NOT REPORT AND THIS DOES:
  * HOW BIG THE LOT GETS. A recovery ladder's danger is not the probability of a loss, it is the
    SIZE of the bet by the time the ladder is deep. Reported as a multiple of base and as a
    percentage of equity at risk, because a 50x stake is not an abstraction -- it is a lot size
    the broker has to accept and the account has to margin.
  * WHERE THE LADDER DIES. The distribution of the trade index at which ruin occurs, so the
    failure is described rather than summarised by one probability.

The prior to beat (memory: martingale-predictor-bot): "'recover-all-losses' spec ruins 71-97%
even with +edge". That was on a detector's bet stream; this is the champion's.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from v5_xau_fade_martingale import simulate                       # noqa: E402
from v5_lot_progression_ruin import champion_trades, EQUITY0      # noqa: E402

OUT = ROOT / "data/v5_runs/lot_progression"
WALLS = {"FTMO 8%": 92_000.0, "10%": 90_000.0}


def run(r, engine, base, K, cap, floor, n_mc=2500, block=10, seed=7):
    rng = np.random.default_rng(seed)
    N = len(r)
    ends, dds, busts, maxstakes, deaths = [], [], 0, [], []
    for _ in range(n_mc):
        idx = []
        while len(idx) < N:
            s0 = rng.integers(0, N)
            idx.extend(range(s0, s0 + block))
        seq = r[np.array(idx[:N]) % N]
        c, stk, bust = simulate(seq, engine, base, EQUITY0, floor, K, cap, 1.0)
        ends.append(c[-1] / EQUITY0 - 1.0)
        e = pd.Series(c); dds.append(float((e / e.cummax() - 1).min() * 100))
        maxstakes.append(float(np.nanmax(stk)))
        if bust >= 0:
            busts += 1; deaths.append(bust)
    return dict(p_ruin=busts / n_mc, med_ret=float(np.median(ends) * 100),
                med_dd=float(np.median(dds)),
                med_maxstake=float(np.median(maxstakes)),
                p95_maxstake=float(np.percentile(maxstakes, 95)),
                med_death=int(np.median(deaths)) if deaths else -1)


def main() -> int:
    r = champion_trades("original")
    N = len(r)
    print("\n" + "=" * 104)
    print("AGGRESSIVE RECOVER-AND-GAIN — uncapped ladder (K = N, it never gives up)")
    print(f"  base stake 0.5% of equity (the live risk_frac). {N} trades ~ 7.2 years, "
          f"3.4 trades/month.")
    print("=" * 104)
    base = EQUITY0 * 0.005
    rows = []
    for wall, floor in WALLS.items():
        for cap in (5, 10, 20, 50, 200):
            for K in (6, 20, N):          # K = N means the ladder NEVER resets on give-up
                s = run(r, "recover4", base, K, float(cap), floor)
                rows.append(dict(wall=wall, K=("uncapped" if K == N else K), max_stake_mult=cap,
                                 p_ruin=round(s["p_ruin"], 4),
                                 med_ret_pct=round(s["med_ret"], 1),
                                 med_maxdd_pct=round(s["med_dd"], 1),
                                 med_peak_stake_pct=round(100 * s["med_maxstake"] / EQUITY0, 2),
                                 p95_peak_stake_pct=round(100 * s["p95_maxstake"] / EQUITY0, 2),
                                 med_trade_of_death=s["med_death"]))
        # flat reference at the same base
        f = run(r, "flat", base, 0, 50.0, floor)
        rows.append(dict(wall=wall, K="FLAT ref", max_stake_mult=1,
                         p_ruin=round(f["p_ruin"], 4), med_ret_pct=round(f["med_ret"], 1),
                         med_maxdd_pct=round(f["med_dd"], 1),
                         med_peak_stake_pct=round(100 * f["med_maxstake"] / EQUITY0, 2),
                         p95_peak_stake_pct=round(100 * f["p95_maxstake"] / EQUITY0, 2),
                         med_trade_of_death=f["med_death"]))
    tb = pd.DataFrame(rows)
    for wall in WALLS:
        s = tb[tb.wall == wall]
        print(f"\n--- {wall} wall ---")
        print(s.drop(columns=["wall"]).to_string(index=False))
    tb.to_csv(OUT / "recover_aggressive.csv", index=False)
    print(f"\n  med_peak_stake_pct = the MEDIAN path's largest single-trade risk, as % of equity.")
    print(f"  med_trade_of_death = which trade killed it (-1 = survived); "
          f"at 3.4 trades/month, divide by 3.4 for months.")
    print(f"\n  wrote {OUT}/recover_aggressive.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
