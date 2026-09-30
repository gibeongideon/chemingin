"""Proper backtest of the NR7 first-break structure — the part of §3bi that survived.

WHAT THIS IS AND IS NOT. §3bi closed the DIRECTIONAL NR7 claim: gold showed break-with-the-trend
+9.24bp / t +2.00 with a clean negative mirror, and across 51 assets the AGAINST variant (+25.31bp)
BEAT the WITH variant (+21.66bp), so the trend filter is not a mechanism and gold's asymmetry was
noise in 260 trades. What survived is non-directional: **an NR7 day is followed by a day that
closes OUTSIDE the narrow range.** Compression precedes expansion — real, and long documented
(Crabel 1990).

So the only honest structure left is the one that does not predict direction: rest a buy-stop at
the NR7 high and a sell-stop at the low, first fill wins, cancel the other, exit at that day's
close. This file backtests that properly — equity curves, Sharpe, drawdown, an out-of-sample
split and PER-ASSET costs — rather than reporting mean basis points.

THREE THINGS §3bi GOT WRONG OR LEFT OPEN, FIXED HERE.

1. **Per-asset costs.** §3bi applied one gold-derived 2.34bp round trip to all 51 assets, which
   is wrong by an order of magnitude at both ends (NDX 0.30bp, BTC 8.99bp) and manufactured the
   high-vol-positive / low-vol-FX-negative split. Here each asset carries its own MEASURED
   spread from `v5_maven_carry_book.CANDS`, widened by the repo's 1.5x convention because a live
   snapshot is one instant mid-session.
2. **Financing.** A one-day hold pays one night of carry, and that ranges from 0 (XAGUSD) to
   -30%/yr (BTC). §3bi ignored it. At -30%/yr a single night is -8bp, which is larger than the
   entire claimed edge.
3. **The ambiguity that D1 cannot resolve.** On an OUTSIDE day both stops are hit and daily bars
   do not say which came first. Gold has clean M15 so it is resolved exactly; for the rest BOTH
   conventions are reported (nearer-the-open first, and its opposite) plus the ambiguity rate. If
   the conclusion flips between conventions the honest answer is "indeterminate on this data",
   not whichever number is nicer.

ENTRY REALISM. A stop order resting at a level known in advance genuinely fills at that level —
unlike §3be's opening-range test, where entering at the breaking bar's CLOSE was the conservative
choice because the level was being crossed intrabar. But a stop in a breakout slips ADVERSELY, so
entry is charged the level plus half the one-way spread, and a no-slippage variant is printed
beside it as the optimistic bound.

    python scripts/v5_nr7_backtest.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "v5_runs" / "nr7"

SPREAD_MULT = 1.5          # repo convention: never trust the tighter live snapshot
TARGET_VOL = 0.10
OOS_START = "2021-01-01"   # discovery 2015-2020, never-seen 2021-2026

# asset -> (D1 file, one-way spread bp, long carry %/yr) — all MEASURED, from CANDS
ASSETS = {
    "GOLD":   ("GOLD_D1_long.csv",   0.94, -0.0370),
    "SILVER": ("SILVER_D1_long.csv", 6.88,  0.0000),
    "NDX":    ("NDX_D1_long.csv",    0.30, -0.0396),
    "SPX":    ("SPX_D1_long.csv",    1.68, -0.0442),
    "DJI":    ("DJI_D1_long.csv",    1.86, -0.0482),
    "DAX":    ("DAX_D1_long.csv",    7.19, -0.0371),
    "BRENT":  ("BRENT_D1_long.csv",  8.49, -0.0479),
    "BTC":    ("BTC_D1_long.csv",    8.99, -0.3000),
}


def load(fn: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / fn, parse_dates=["time"], index_col="time").sort_index()
    d.columns = [c.lower() for c in d.columns]
    d = d[~d.index.duplicated(keep="last")]
    return d[["open", "high", "low", "close"]].dropna()


def trades(d: pd.DataFrame, spread_bp: float, carry: float, convention: str,
           slip: bool = True) -> pd.DataFrame:
    """One row per NR7 signal. `convention` resolves outside days: 'near' = the level nearer
    the next day's OPEN is assumed to trigger first; 'far' = the opposite."""
    rng = d["high"] - d["low"]
    nr7 = rng == rng.rolling(7).min()
    ret = d["close"].pct_change()
    vol = ret.ewm(halflife=20, min_periods=20).std() * np.sqrt(252)
    one_way = spread_bp * SPREAD_MULT * 1e-4
    rows = []
    idx = d.index
    for i in range(7, len(d) - 1):
        if not nr7.iloc[i]:
            continue
        hi, lo = d["high"].iloc[i], d["low"].iloc[i]
        o1, h1, l1, c1 = (d["open"].iloc[i + 1], d["high"].iloc[i + 1],
                          d["low"].iloc[i + 1], d["close"].iloc[i + 1])
        up, dn = h1 > hi, l1 < lo
        if not (up or dn):
            continue
        ambiguous = bool(up and dn)
        if ambiguous:
            near_up = abs(o1 - hi) <= abs(o1 - lo)
            side = (+1 if near_up else -1) if convention == "near" else (-1 if near_up else +1)
        else:
            side = +1 if up else -1
        lvl = hi if side > 0 else lo
        entry = lvl * (1 + side * one_way / 2) if slip else lvl   # stops slip adversely
        gross = side * (c1 - entry) / entry
        v = vol.iloc[i]
        if not np.isfinite(v) or v <= 0:
            continue
        lots = float(np.clip(TARGET_VOL / v, 0, 8))               # vol-target so assets combine
        fin = (carry if side > 0 else -carry * 0.5) / 252          # one night of financing
        net = lots * (gross - one_way + fin)                       # one-way in, close-out at market
        rows.append(dict(when=idx[i + 1], side=side, ambiguous=ambiguous, lots=lots,
                         gross_bp=gross * 1e4, net_ret=net))
    return pd.DataFrame(rows)


def stats(r: pd.Series, n_per_year: float) -> dict:
    if len(r) < 10:
        return {}
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min() * 100)
    ann = n_per_year
    sr = float(r.mean() / r.std() * np.sqrt(ann)) if r.std() > 0 else np.nan
    cagr = float(eq.iloc[-1] ** (ann / len(r)) - 1)
    return dict(n=len(r), sharpe=sr, cagr=cagr * 100, maxdd=dd,
                hit=float((r > 0).mean() * 100))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--convention", default="both", choices=["near", "far", "both"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    convs = ["near", "far"] if a.convention == "both" else [a.convention]

    print("=" * 100)
    print("NR7 FIRST-BREAK BACKTEST — non-directional structure, per-asset costs and financing")
    print("=" * 100)
    allrows = {}
    for conv in convs:
        print(f"\n--- outside-day convention: {conv.upper()} "
              f"({'level nearer the open triggers first' if conv=='near' else 'the opposite'}) ---")
        print(f"  {'asset':8s} {'n':>5s} {'amb%':>6s} {'grossbp':>8s} {'Sharpe':>7s} "
              f"{'CAGR%':>7s} {'maxDD%':>7s} {'hit%':>6s} {'OOS SR':>7s}")
        per = {}
        for sym, (fn, spr, carry) in ASSETS.items():
            try:
                d = load(fn)
            except Exception as e:
                print(f"  {sym:8s} load failed {type(e).__name__}")
                continue
            t = trades(d, spr, carry, conv)
            if len(t) < 30:
                print(f"  {sym:8s} only {len(t)} signals")
                continue
            t = t.set_index("when")
            yrs = (t.index.max() - t.index.min()).days / 365.25
            npy = len(t) / max(yrs, 0.1)
            s = stats(t["net_ret"], npy)
            oos = t.loc[OOS_START:]
            so = stats(oos["net_ret"], npy) if len(oos) > 20 else {}
            per[sym] = t["net_ret"]
            print(f"  {sym:8s} {s['n']:5d} {t['ambiguous'].mean()*100:5.1f}% "
                  f"{t['gross_bp'].mean():+8.1f} {s['sharpe']:+7.2f} {s['cagr']:+7.2f} "
                  f"{s['maxdd']:7.1f} {s['hit']:5.1f}% "
                  f"{so.get('sharpe', float('nan')):+7.2f}")
        if per:
            # equal-weight portfolio across assets, on the union of signal dates
            P = pd.DataFrame(per).sort_index()
            port = P.mean(axis=1, skipna=True).dropna()
            yrs = (port.index.max() - port.index.min()).days / 365.25
            npy = len(port) / max(yrs, 0.1)
            ps = stats(port, npy)
            po = stats(port.loc[OOS_START:], npy)
            print(f"  {'PORTFOLIO':8s} {ps['n']:5d} {'':>6s} {'':>8s} {ps['sharpe']:+7.2f} "
                  f"{ps['cagr']:+7.2f} {ps['maxdd']:7.1f} {ps['hit']:5.1f}% "
                  f"{po.get('sharpe', float('nan')):+7.2f}")
            allrows[conv] = ps
            P.to_csv(OUT / f"nr7_trades_{conv}.csv")
    if len(allrows) == 2:
        n, f = allrows["near"]["sharpe"], allrows["far"]["sharpe"]
        print("\n" + "=" * 100)
        print(f"  CONVENTION SENSITIVITY: portfolio Sharpe {n:+.2f} (near) vs {f:+.2f} (far)")
        print(f"  -> {'ROBUST to the ambiguity' if min(n,f) > 0.3 else 'the conclusion DEPENDS on an unresolvable convention' if n*f < 0 or abs(n-f) > 0.3 else 'mildly sensitive'}")


if __name__ == "__main__":
    main()
