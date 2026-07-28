"""SMC / Supply-&-Demand strategy for XAUUSD — with an OOS-honest grid search.

Smart-Money-Concepts primitives, all strictly causal (a fractal swing at bar j is
only known at bar j+order, so no lookahead):

  swings      confirmed fractal swing highs/lows -> the live S/R levels
  sweep       liquidity grab + reclaim: wick past a swing then close back inside
              (the classic stop-hunt reversal)
  bos         break of structure: close beyond the last swing (continuation)
  sr_bounce   price reacts at a supply/demand level (touch + rejection candle)

Entry next bar open, ATR stop, R:R target, timeout, ONE position at a time.

THE POINT IS THE CONTROLS, NOT THE GRID. A grid over hundreds of configs will
always cough up a great in-sample number by luck (multiple testing). So every
config is scored on a TRAIN half, and what we report is the TEST-half performance
of the train winners, plus the MEDIAN test result across the whole grid — if the
median test Sharpe is ~0, the family has no edge and the top line is noise. Costs
are floored at the live gold spread (default FTMO $0.45; cent $0.34; raw ~$0.12)
and everything is benchmarked against buy&hold. See V5_FINDINGS MANDATORY CONTROLS.

    python scripts/v5_smc_xau.py --tf H4
    python scripts/v5_smc_xau.py --tf H1 --spread-usd 0.34   # cent account
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SLIP_USD = 0.10


def load(tf: str) -> pd.DataFrame:
    df = pd.read_csv(f"{ROOT}/data/XAUUSD_{tf}_long.csv", parse_dates=["time"]).set_index("time")
    return df[~df.index.duplicated(keep="last")].sort_index()


def atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n // 2).mean().values


def swing_levels(df: pd.DataFrame, order: int):
    """Last CONFIRMED swing-high / swing-low level available causally at each bar.

    A fractal high at bar j (high[j] = max over [j-order, j+order]) is confirmed
    `order` bars later, so its level only becomes usable at bar j+order.
    """
    h, l = df["high"].values, df["low"].values
    n = len(h)
    is_hi = np.zeros(n, bool)
    is_lo = np.zeros(n, bool)
    for i in range(order, n - order):
        seg_h = h[i - order:i + order + 1]
        seg_l = l[i - order:i + order + 1]
        if h[i] == seg_h.max():
            is_hi[i] = True
        if l[i] == seg_l.min():
            is_lo[i] = True
    conf_hi = np.full(n, np.nan)
    conf_lo = np.full(n, np.nan)
    last_h = last_l = np.nan
    for i in range(n):
        j = i - order                      # a swing at j is confirmed now
        if j >= 0:
            if is_hi[j]:
                last_h = h[j]
            if is_lo[j]:
                last_l = l[j]
        conf_hi[i] = last_h
        conf_lo[i] = last_l
    return conf_hi, conf_lo


def entries(df, mode, order, tol_atr=0.5):
    """Causal +1/-1/0 entry signal per bar (act at NEXT bar's open)."""
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    a = atr(df)
    hi, lo = swing_levels(df, order)
    sig = np.zeros(len(c))
    for i in range(len(c)):
        if not np.isfinite(a[i]) or not np.isfinite(hi[i]) or not np.isfinite(lo[i]):
            continue
        if mode == "sweep":                       # grab liquidity then reclaim
            if l[i] < lo[i] and c[i] > lo[i]:
                sig[i] = 1
            elif h[i] > hi[i] and c[i] < hi[i]:
                sig[i] = -1
        elif mode == "bos":                       # break of structure (continuation)
            if c[i] > hi[i]:
                sig[i] = 1
            elif c[i] < lo[i]:
                sig[i] = -1
        elif mode == "sr_bounce":                 # supply/demand zone reaction
            if l[i] <= lo[i] + tol_atr * a[i] and c[i] > o[i]:
                sig[i] = 1
            elif h[i] >= hi[i] - tol_atr * a[i] and c[i] < o[i]:
                sig[i] = -1
    return sig, a


def backtest(df, sig, a, stop_atr, rr, max_hold, cost_frac, long_only):
    """Sequential (non-overlapping) trades. Returns array of net per-trade returns."""
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    n = len(c)
    rets, wins = [], 0
    i = 0
    while i < n - 1:
        d = sig[i]
        if d == 0 or (long_only and d < 0) or not np.isfinite(a[i]):
            i += 1
            continue
        entry = o[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            i += 1
            continue
        dist = stop_atr * a[i]
        stop = entry - d * dist
        tgt = entry + d * rr * dist
        exit_px, j = c[min(i + max_hold, n - 1)], min(i + max_hold, n - 1)
        for k in range(i + 1, min(i + 1 + max_hold, n)):
            if (d > 0 and l[k] <= stop) or (d < 0 and h[k] >= stop):
                exit_px, j = stop, k
                break
            if (d > 0 and h[k] >= tgt) or (d < 0 and l[k] <= tgt):
                exit_px, j = tgt, k
                break
            exit_px, j = c[k], k
        net = d * (exit_px - entry) / entry - 2 * cost_frac
        rets.append(net)
        i = j + 1
    return np.array(rets)


def metrics(rets, bars_per_year):
    if len(rets) < 8:
        return None
    m, s = rets.mean(), rets.std()
    tot = float((1 + rets).prod() - 1) * 100
    # per-trade Sharpe annualised by trade frequency
    sr = float(m / s * np.sqrt(len(rets) / (len(rets)))) if s else 0.0  # placeholder
    return dict(n=len(rets), tot=tot, mean=m, std=s, hit=float((rets > 0).mean() * 100))


def sharpe_ann(rets, n_bars, bars_per_year):
    if len(rets) < 8 or rets.std() == 0:
        return 0.0
    trades_per_year = len(rets) / (n_bars / bars_per_year)
    return float(rets.mean() / rets.std() * np.sqrt(trades_per_year))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tf", default="H4", choices=["H1", "H4", "M30"])
    ap.add_argument("--spread-usd", type=float, default=0.45,
                    help="live gold spread: FTMO 0.45 / cent 0.34 / raw 0.12")
    ap.add_argument("--max-hold", type=int, default=24)
    args = ap.parse_args()

    df = load(args.tf)
    bpy = {"H1": 252 * 24, "H4": 252 * 6, "M30": 252 * 48}[args.tf]
    px = df["close"].mean()
    cost_frac = (args.spread_usd / 2 + SLIP_USD) / px      # one-way, fraction of price
    split = int(len(df) * 0.70)
    tr, te = df.iloc[:split], df.iloc[split:]
    bh_tr = (tr["close"].iloc[-1] / tr["close"].iloc[0] - 1) * 100
    bh_te = (te["close"].iloc[-1] / te["close"].iloc[0] - 1) * 100

    print(f"SMC grid — XAUUSD {args.tf}  {df.index[0].date()}..{df.index[-1].date()}  "
          f"spread ${args.spread_usd}  cost {cost_frac*1e4:.1f}bp/side")
    print(f"  train {tr.index[0].date()}..{tr.index[-1].date()} (buy&hold {bh_tr:+.0f}%)  "
          f"| test {te.index[0].date()}..{te.index[-1].date()} (buy&hold {bh_te:+.0f}%)\n")

    grid = list(itertools.product(
        ["sweep", "bos", "sr_bounce"],       # mode
        [3, 5],                              # swing order
        [1.0, 1.5, 2.0],                     # stop_atr
        [1.0, 1.5, 2.0, 3.0],                # rr
        [True, False],                       # long_only
    ))
    rows = []
    # cache signals per (mode, order) — independent of stop/rr/dir
    sigcache = {}
    for mode, order, stop_atr, rr, lo in grid:
        key = (mode, order)
        if key not in sigcache:
            sigcache[key] = entries(df, mode, order)
        sig, a = sigcache[key]
        rtr = backtest(tr, sig[:split], a[:split], stop_atr, rr, args.max_hold, cost_frac, lo)
        rte = backtest(te, sig[split:], a[split:], stop_atr, rr, args.max_hold, cost_frac, lo)
        s_tr = sharpe_ann(rtr, len(tr), bpy)
        s_te = sharpe_ann(rte, len(te), bpy)
        rows.append(dict(mode=mode, order=order, stop=stop_atr, rr=rr, lo=lo,
                         n_tr=len(rtr), sr_tr=s_tr, net_tr=(1 + rtr).prod() * 100 - 100 if len(rtr) else 0,
                         n_te=len(rte), sr_te=s_te, net_te=(1 + rte).prod() * 100 - 100 if len(rte) else 0,
                         hit_te=float((rte > 0).mean() * 100) if len(rte) else 0))
    R = pd.DataFrame(rows)
    R = R[R.n_tr >= 20]

    print(f"{'rank':>4} {'mode':10} {'ord':>3} {'stop':>4} {'rr':>4} {'dir':>5} "
          f"{'TRAIN_sr':>8} {'tr_net%':>8} | {'TEST_sr':>8} {'te_net%':>8} {'te_hit%':>7} {'te_n':>5}")
    top = R.sort_values("sr_tr", ascending=False).head(12)
    for r_ in range(len(top)):
        x = top.iloc[r_]
        print(f"{r_+1:>4} {x['mode']:10} {int(x['order']):>3} {x['stop']:>4.1f} {x['rr']:>4.1f} "
              f"{'long' if x['lo'] else 'both':>5} {x['sr_tr']:>8.2f} {x['net_tr']:>+8.0f} | "
              f"{x['sr_te']:>8.2f} {x['net_te']:>+8.0f} {x['hit_te']:>7.1f} {int(x['n_te']):>5}")

    print("\n--- OOS honesty checks (this is the verdict, not the top row) ---")
    print(f"  grid size (n_tr>=20): {len(R)}")
    print(f"  MEDIAN test Sharpe across grid: {R.sr_te.median():+.2f}   "
          f"(mean {R.sr_te.mean():+.2f})   -> ~0 means no family edge")
    print(f"  configs with test Sharpe > 0.5: {int((R.sr_te > 0.5).sum())}/{len(R)} "
          f"({(R.sr_te>0.5).mean()*100:.0f}%)   [~expected by luck if no edge]")
    tw = top.iloc[0]
    print(f"  train winner holds OOS? train {tw.sr_tr:+.2f} -> test {tw.sr_te:+.2f}   "
          f"{'YES' if tw.sr_te > 0.5 else 'NO — overfit'}")
    corr = R[["sr_tr", "sr_te"]].corr().iloc[0, 1]
    print(f"  train-vs-test Sharpe correlation across grid: {corr:+.2f}   "
          f"(>0.3 = train rank carries OOS; ~0 = pure noise)")
    out = ROOT / "data" / "v5_runs" / f"smc_grid_{args.tf}.csv"
    R.to_csv(out, index=False)
    print(f"\nfull grid -> {out}")


if __name__ == "__main__":
    main()
