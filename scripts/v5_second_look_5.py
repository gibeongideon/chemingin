#!/usr/bin/env python3
"""Second look #5: PDH day-trade SIZED by the noise-band state (pre-registered addendum in
data/v5_runs/second_look/PREREG.md). §3ef found NY-session noise-band momentum real in two sources
(family p 0.013/0.007) but unable to pay its own spread, and named the untested use: "band state as a
same-day filter on a slower strategy". The PDH (§3ex) already clears cost, so the band only has to
re-weight it, not pay for itself.

  D1 (PRIMARY): previous NY session closed its last 30-min check ABOVE the upper band -> size 1.0, else 0.5
  D2           : same-day NY state at the last check before the entry minute (pre-08:30 entries -> 1.0)
Band arithmetic = scripts/v5_noise_band_momentum.positions (NY 08:00-16:00 ET, LOOKBACK 14, 30-min checks).
Net = w*ret - COST*w*on/px exactly as scripts/v5_pdh_m1_stress.stats (COST $0.30). Control: random dial
halving the same number of trades, 500 draws. Known answer: PDH net SR ~ +0.75.

    python scripts/v5_second_look_5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from scripts.v5_intraday_m1 import load_m1  # noqa: E402
from scripts.v5_noise_band_momentum import LOOKBACK, matrices  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "second_look" / "5"
COST, N_RAND, SIZE = 0.30, 500, 0.5
RNG = np.random.default_rng(20261128)


def sharpe(x):
    x = np.asarray(x, float)
    return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0.0


def band_states() -> tuple[pd.Series, pd.DataFrame]:
    """Per NY session (calendar date): final-check state, and the check-by-check state matrix."""
    d = load_m1()
    d = d[d.index >= "2006-06-01"]
    O, C = matrices(d, 1, 8 * 60, 8 * 60)                 # 480 one-minute slots, 08:00-16:00 ET
    open0, c = O[0].to_numpy(), C.to_numpy()
    move = np.abs(c / open0[:, None] - 1)
    sigma = pd.DataFrame(move).rolling(LOOKBACK, min_periods=10).mean().shift(1).to_numpy()
    last_close = C.ffill(axis=1).iloc[:, -1].shift(1).to_numpy()
    ub = np.fmax(open0, last_close)[:, None] * (1 + sigma)
    lb = np.fmin(open0, last_close)[:, None] * (1 - sigma)
    sig = np.where(c > ub, 1.0, np.where(c < lb, -1.0, 0.0))
    sig[~np.isfinite(c) | ~np.isfinite(ub)] = np.nan
    checks = [j for j in range(c.shape[1]) if (j + 1) % 30 == 0]
    S = pd.DataFrame(sig[:, checks], index=O.index, columns=[(j + 1) for j in checks])   # minutes after 08:00
    final = S.ffill(axis=1).iloc[:, -1]
    return final, S


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_parquet(ROOT / "data/v5_runs/intraday_day_signal/pdh_trades.parquet")
    t = t[t.index >= "2007-01-01"]
    on = t.on.to_numpy()
    gross = (t.w * t.ret).to_numpy()
    c1 = (t.w * t.on / t.px).to_numpy()
    net = lambda m, k=1.0: m * (gross - k * COST * c1)                                    # noqa: E731
    base = net(np.ones(len(t)))
    print(f"KNOWN ANSWER: PDH net SR {sharpe(base):+.3f} (§3ex +0.75), gross {sharpe(gross):+.3f}, trades {int(on.sum())}")

    final, S = band_states()
    days = t.index.normalize()
    prev_final = final.shift(1).reindex(days).to_numpy()       # previous NY session's final state
    sess = S.reindex(days)
    ent_ny = t.t_in.to_numpy() - 15 * 60                       # minutes after 08:00 ET on day D
    d2 = np.full(len(t), np.nan)
    cols = np.array(S.columns)
    for i in np.where(on)[0]:
        if ent_ny[i] >= 30:
            k = cols[cols <= ent_ny[i]]
            if len(k):
                row = sess.iloc[i][k].dropna()
                d2[i] = row.iloc[-1] if len(row) else np.nan
    cells = {"D1 prev-session above band (PRIMARY)": np.where(on & ~(prev_final == 1.0), SIZE, 1.0),
             "D2 same-day above band at entry": np.where(on & np.isfinite(d2) & ~(d2 == 1.0), SIZE, 1.0)}
    half = len(t) // 2
    rows = []
    for name, m in cells.items():
        n_half = int(((m < 1.0) & on).sum())
        sr, sr2 = sharpe(net(m)), sharpe(net(m, 2.0))
        on_idx = np.where(on)[0]
        nl, nl2 = np.empty(N_RAND), np.empty(N_RAND)
        for j in range(N_RAND):
            mm = np.ones(len(t)); mm[RNG.choice(on_idx, n_half, replace=False)] = SIZE
            nl[j], nl2[j] = sharpe(net(mm)), sharpe(net(mm, 2.0))
        pct, pct2 = float((nl < sr).mean()), float((nl2 < sr2).mean())
        h1 = sharpe(net(m)[:half]) - sharpe(base[:half])
        h2 = sharpe(net(m)[half:]) - sharpe(base[half:])
        full = (m == 1.0) & on
        r_full, r_half = t.ret[full].mean() * 1e4, t.ret[(m < 1.0) & on].mean() * 1e4
        print(f"\n{name}: halved {n_half}/{int(on.sum())} trades")
        print(f"  SR {sr:+.3f} (dSR {sr - sharpe(base):+.3f})  pctile vs random dial {100 * pct:.1f}%  "
              f"halves {h1:+.3f}/{h2:+.3f}  2x cost pctile {100 * pct2:.1f}%")
        print(f"  mean gross per trade: full-size group {r_full:+.2f}bp vs half-size group {r_half:+.2f}bp")
        rows.append(dict(cell=name, halved=n_half, trades=int(on.sum()), sr=sr, base=sharpe(base), pctile=pct,
                         h1=h1, h2=h2, pctile_2x=pct2, bp_full=r_full, bp_half=r_half))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "cells.csv", index=False)
    p = R.iloc[0]
    prim = p.pctile >= 0.95 and p.h1 > 0 and p.h2 > 0 and p.pctile_2x >= 0.95
    both = bool(((R.pctile >= 0.975) & (R.h1 > 0) & (R.h2 > 0) & (R.pctile_2x >= 0.975)).all())
    print(f"\nVERDICT: {'REVIVED' if (prim or both) else 'not revived'} (primary {'pass' if prim else 'fail'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
