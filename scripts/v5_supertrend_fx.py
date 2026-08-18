"""SuperTrend on FX — does a different MECHANISM revive what the champ-recipe
found dead? (2026-08-19 follow-up to §3s)

`v5_instrument_search.py` excludes FX with the note "comprehensively dead
post-2016" — that verdict is for the champ EWMAC+breakout recipe. §3s found
SuperTrend real (DSR 0.9948 on XAU) but interchangeable with the champ-recipe on
every trending asset tested so far (corr 0.77-0.88). FX pairs are NOT
structurally trending the way gold/equities/crypto are (no "kill-the-shorts"
prior applies here — a currency pair has no natural long-run drift), so this is
a genuinely different question: does a discrete ATR-band trend-state machine
catch something a continuous EWMAC+breakout blend misses on a range-bound-ish
asset class, or does FX stay dead regardless of mechanism?

Tests BOTH long-only and long/short (unlike XAU/the 2nd basket, there's no prior
reason to expect long-only dominates on FX). Same rigor as the rest of this
line: lookahead-probe already confirmed generically for supertrend_direction;
DSR/PBO on every per-pair grid before trusting a "best" cell.

    python scripts/v5_supertrend_fx.py
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

from scripts.v5_supertrend_basket import load_d1, supertrend_net_returns, sharpe, dd, cagr, EVAL_START  # noqa: E402
from src.evaluation.dsr_pbo import deflated_sharpe_ratio, pbo_cscv  # noqa: E402

FX_PAIRS = (
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
    "EURJPY", "EURGBP", "EURAUD", "EURCHF", "GBPJPY", "GBPAUD", "GBPCHF",
    "AUDJPY", "AUDNZD", "CADJPY", "NZDJPY",
)
ATR_PERIODS = (10, 14, 20)
MULTIPLIERS = (2.5, 3.0, 3.5)


def screen_pair(sym: str, long_only: bool) -> dict:
    df = load_d1(sym)
    grid_rets = {}
    for p in ATR_PERIODS:
        for m in MULTIPLIERS:
            grid_rets[(p, m)] = supertrend_net_returns(df, p, m, long_only=long_only)
    M = pd.DataFrame(grid_rets).dropna(how="all").fillna(0.0).loc[EVAL_START:]
    sh = {k: sharpe(M[k], start=M.index[0].strftime("%Y-%m-%d")) for k in M.columns}
    best_key = max(sh, key=lambda k: sh[k] if np.isfinite(sh[k]) else -1e9)
    trial_sh_daily = np.array([sh[k] / np.sqrt(252) for k in sh])
    dsr = deflated_sharpe_ratio(M[best_key].values, trial_sh_daily)
    pbo = pbo_cscv(M.values, n_partitions=10)
    n_pos = sum(1 for v in sh.values() if np.isfinite(v) and v > 0)
    return dict(sym=sym, long_only=long_only, best=best_key, sr=sh[best_key],
                dd=dd(M[best_key]), dsr=dsr["dsr"], pbo=pbo.pbo, n_pos=n_pos,
                n_cells=len(sh), returns=M[best_key])


def main():
    print("=== SuperTrend on FX — screening 18 pairs, long-only AND long/short ===\n")
    print(f"{'pair':8} {'dir':10} {'best(p,m)':>10} {'SR':>7} {'DD':>7} {'pos/9':>6} {'DSR':>6} {'PBO':>6}")
    rows = []
    for sym in FX_PAIRS:
        for long_only in (True, False):
            r = screen_pair(sym, long_only)
            rows.append(r)
            tag = "long-only" if long_only else "long/short"
            print(f"{sym:8} {tag:10} {str(r['best']):>10} {r['sr']:+7.3f} {r['dd']:+6.1f}% "
                  f"{r['n_pos']:3d}/{r['n_cells']:<3} {r['dsr']:6.3f} {r['pbo']:6.3f}")

    R = pd.DataFrame([{k: v for k, v in r.items() if k != "returns"} for r in rows])
    passing = R[(R.sr > 0.30) & (R.dsr > 0.90) & (R.pbo < 0.30)]
    print(f"\n{len(passing)} of {len(R)} (pair x direction) combos clear ALL THREE bars "
          f"(SR>0.30, DSR>0.90, PBO<0.30):")
    if len(passing):
        print(passing.to_string(index=False))
    else:
        print("  none.")

    out = ROOT / "data" / "v5_runs" / "turning_predict" / "supertrend_fx_screen.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
