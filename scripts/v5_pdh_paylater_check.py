#!/usr/bin/env python3
"""PDH day-trade on the Goat Funded PAY LATER evaluation (user 2026-11-27): target +3%, no daily limit,
max loss 8% TRAILING (floor = highest balance/equity - 8% of initial, never down, locks at initial).
Per-trade R from the §3ex trade list (data/v5_runs/intraday_day_signal/pdh_trades.parquet):
R = net return x entry / (STOP_ATR x ATR), stop 1.0 ATR. Equity moves by risk x R on trade days.
Intraday adverse excursion is bounded by the server-side stop (-1R + slippage), so the trailing floor is
tested on the realised trade result and on a -1R intraday low for open trades (conservative).
Block bootstrap over DAYS (block 21), 3 x 2000 paths, horizon 36 months. Also reports the bot's own
halt (stop entries at peak - HALT of initial) as 'halted' (account alive, not passed).

    python scripts/v5_pdh_paylater_check.py
"""
import numpy as np
import pandas as pd

d = pd.read_parquet("data/v5_runs/intraday_day_signal/pdh_trades.parquet")
R = np.where(d.on, d.ret * d.entry / d.atr, 0.0)
R = np.nan_to_num(R)
on = d.on.to_numpy()
print(f"{int(on.sum())} trades over {len(d)} days ({on.sum() / len(d) * 5:.2f}/wk); mean R {R[on].mean():+.3f}, "
      f"win {np.mean(R[on] > 0):.1%}, worst {R[on].min():+.2f}R, best {R[on].max():+.2f}R")
TARGET, ML, HALT, BLOCK, N, DAYS = 0.03, 0.08, 0.065, 21, 2000, 756


def sim(risk, seed):
    rng = np.random.default_rng(seed)
    out = {"pass": 0, "breach": 0, "halt": 0, "time": 0}
    days = []
    for _ in range(N):
        idx = []
        while len(idx) < DAYS:
            s = rng.integers(0, len(R)); idx.extend(range(s, s + rng.geometric(1 / BLOCK)))
        idx = np.array(idx[:DAYS]) % len(R)
        eq = peak = 1.0
        res = "time"
        for t, i in enumerate(idx):
            if on[i]:
                low = eq - risk * max(1.0, -R[i])           # intraday worst case: the stop (or a gap through it)
                floor = min(peak - ML, 1.0)
                if low < floor:
                    res = "breach"; break
                eq += risk * R[i]
                peak = max(peak, eq)
            if eq >= 1 + TARGET:
                res = "pass"; days.append(t); break
            if eq <= peak - HALT:
                res = "halt"; break
        out[res] += 1
    return out, days


rows = []
for risk in (0.0025, 0.0035, 0.005, 0.0075, 0.01):
    agg, dd = {"pass": 0, "breach": 0, "halt": 0, "time": 0}, []
    for s in range(3):
        o, x = sim(risk, s)
        for k in agg: agg[k] += o[k]
        dd += x
    n = 3 * N
    eqc = np.cumsum(risk * R); mdd = (eqc - np.maximum.accumulate(eqc)).min()
    print(f"risk {100 * risk:.2f}%/trade: pass {100 * agg['pass'] / n:5.1f}%  median {np.median(dd) / 21:5.1f} mo  "
          f"breach {100 * agg['breach'] / n:4.1f}%  bot-halt {100 * agg['halt'] / n:4.1f}%  timeout {100 * agg['time'] / n:4.1f}%   "
          f"| historical maxDD (additive) {100 * mdd:.1f}%")
