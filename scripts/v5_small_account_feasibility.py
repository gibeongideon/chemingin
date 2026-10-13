"""Can the 4-sleeve book run on a $25k account behind a 10% drawdown wall?

TWO INDEPENDENT QUESTIONS, and they have different answers:
  1. SIZING -- can each sleeve be expressed in 0.01-lot steps at that equity?
  2. RISK   -- what vol dial keeps a 10% STATIC drawdown wall from being hit, and what does
               that dial cost in CAGR?

METHOD NOTES THAT MATTER
  * Sharpe is LEVERAGE-INVARIANT (memory: +1.02 flat across risk_frac 0.25-4%), so the dial is a
    pure risk choice -- it cannot buy back what the wall takes away. The only question is how
    much CAGR a survivable dial leaves.
  * SS3af / book-shootout: "Sharpe does not predict pass rate; the dial-scaled LEFT TAIL against
    the limit does." So this reports BREACH PROBABILITY from a block bootstrap, not a single
    historical maxDD -- one realisation of a path is not a risk estimate.
  * Day counts are aligned to a common business-day index before annualising (method rule 1.3.6,
    which inverted a book ranking once because BTC quotes weekends).
  * STATIC vs TRAILING drawdown is a real trap (memory: xau-riskfrac-sizing-table). FTMO-style
    walls are measured from the INITIAL balance, so that is what is used here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from v5_xau_champion_lifts import champion_recipe            # noqa: E402

SLEEVES = {"GOLD": "xau", "BTC": "crypto", "NDX": "eq_us", "BRENT": "energy"}
TARGET_VOL, VOL_HL, MAX_LEV = 0.09, 42, 8.0       # the deployed 9% dial
COST_BP = {"GOLD": 1.17, "BTC": 1.50, "NDX": 0.50, "BRENT": 7.58}   # measured, one-way
WALL = 0.10
# Financing. The book pays -3.79%/yr of overnight swap (memory: financing-is-the-live-lever;
# it is worth -0.29 Sharpe and is the single largest measured drag on this book). It scales
# with NOTIONAL, so it scales with the dial exactly as returns do.
CARRY_YR = -0.0379
rng = np.random.default_rng(7)


def sleeve_returns(sym: str) -> pd.Series:
    f = ROOT / f"data/{sym}_D1_long.csv"
    d = pd.read_csv(f, parse_dates=["time"], index_col="time").sort_index()
    d = d[~d.index.duplicated(keep="last")]
    c = d["close"].astype(float)
    fc = champion_recipe(c, 1 / 6, 1.5)
    ret = c.pct_change()
    vol = ret.ewm(halflife=VOL_HL, min_periods=20).std() * np.sqrt(252)
    pos = (fc.clip(-2, 2) * (TARGET_VOL / vol)).clip(-MAX_LEV, MAX_LEV).shift(1)
    turn = pos.diff().abs().fillna(0.0)
    return (pos * ret - turn * COST_BP[sym] / 1e4).dropna()


def perf(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    dd = (eq / eq.cummax() - 1).min()
    static = (eq / eq.iloc[0] - 1).min()      # vs INITIAL balance, the prop-firm convention
    return dict(sharpe=r.mean() / r.std() * np.sqrt(252), cagr=eq.iloc[-1] ** (1 / yrs) - 1,
                maxdd=dd, static_dd=static)


def breach_prob(r: pd.Series, dial: float, horizon=252, draws=4000, block=21) -> float:
    """P(a STATIC 10% drawdown from the starting balance is hit within one year)."""
    x = (r * dial).values
    n = len(x)
    hits = 0
    for _ in range(draws):
        idx = np.concatenate([np.arange(s, s + block)
                              for s in rng.integers(0, n - block, horizon // block + 1)])[:horizon]
        eq = np.cumprod(1 + x[idx])
        if eq.min() - 1 <= -WALL:
            hits += 1
    return hits / draws


def main() -> int:
    rets = {s: sleeve_returns(s) for s in SLEEVES}
    idx = pd.bdate_range(max(r.index[0] for r in rets.values()),
                         min(r.index[-1] for r in rets.values()))
    R = pd.DataFrame({s: r.reindex(idx).fillna(0.0) for s, r in rets.items()})
    print("=" * 94)
    print("SMALL-ACCOUNT FEASIBILITY — NET of %.2f%%/yr financing, static %.0f%% wall" % (-100 * CARRY_YR, 100 * WALL))
    print(f"  {idx[0].date()} .. {idx[-1].date()}  ({len(idx)/252:.1f}y, business-day aligned)")
    print("=" * 94)

    books = {
        "4 sleeves, gold-tilted (the $100k book)": {"GOLD": .50, "BTC": .1667, "NDX": .1667, "BRENT": .1667},
        "3 sleeves, NO GOLD (what $25k can size)": {"BTC": 1/3, "NDX": 1/3, "BRENT": 1/3},
    }
    for name, w in books.items():
        b = sum(R[s] * wt for s, wt in w.items()) + CARRY_YR / 252
        m = perf(b)
        print(f"\n--- {name} ---")
        print(f"  at the deployed 9% dial: Sharpe {m['sharpe']:+.2f}  CAGR {100*m['cagr']:+.2f}%  "
              f"maxDD {100*m['maxdd']:.1f}%  static DD {100*m['static_dd']:.1f}%")
        print(f"  {'dial':>6} {'CAGR':>8} {'maxDD':>8} {'Sharpe':>7}  P(hit 10% wall in 1y)")
        for dial in (1.00, 0.75, 0.60, 0.50, 0.40, 0.30):
            md = perf(b * dial)
            p = breach_prob(b, dial)
            mark = "  <-- survivable" if p <= 0.05 else ("  breach risk" if p > 0.15 else "")
            print(f"  {dial:>6.2f} {100*md['cagr']:>7.2f}% {100*md['maxdd']:>7.1f}% "
                  f"{md['sharpe']:>7.2f}  {100*p:>5.1f}%{mark}")
    print("\n  Sharpe is leverage-invariant: the dial trades CAGR for survival, nothing else.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
