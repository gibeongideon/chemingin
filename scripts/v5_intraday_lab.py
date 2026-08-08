"""Intraday high-R:R lab — few wins, positive expectancy. PHASE 0 = can we measure it?

The thesis (measured, not asserted): required edge above a random entry is

    p_breakeven = (1 + cost/R) / (1 + RR)     random = 1/(1+RR)
    REQUIRED EDGE = (cost/R) / (1 + RR)

so a WIDER stop shrinks the hurdle linearly. On XAU M15 at a 2xATR stop and RR 3 that
is ~1.9pp above random — versus the M15 fade (V5_FINDINGS §1) which faced a spread
~10x its edge. That 20-40x easier hurdle is the only reason to reopen intraday here.

Phase 0 answers whether the question is even measurable, BEFORE any model:

  --stage bias    P0.1 intrabar-path bias. Every simulator in this repo resolves the
                  STOP before the TARGET (conservative). When both barriers sit inside
                  one bar the win rate is understated. If that bias is the same size as
                  the 1-2pp edge we hunt, no result at this timeframe can be trusted.
                  Measured by re-resolving the SAME trades on finer bars.
  --stage random  P0.2 EMPIRICAL random-entry baseline. The benchmark is NOT 1/(1+RR):
                  that ignores wick path and spread-on-entry. Measure it.
  --stage oracle  P0.3 perfect-hindsight ceiling + the (precision, recall) contour that
                  clears net-positive expectancy  [MANDATORY CONTROL #5, V5_FINDINGS].

Trades are resolved by ONE engine (`resolve_trades`) which also records MFE/MAE in R
units — Phase 2's early-TP / early-SL research needs them and nothing in this repo
records them today.

    python scripts/v5_intraday_lab.py --stage bias
    python scripts/v5_intraday_lab.py --stage random --tf M15 --tf M30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# live FTMO round-trip spread + per-side slippage, PRICE units (V5_FINDINGS control #1)
LIVE_COST = {"XAUUSD": (0.45, 0.10), "EURUSD": (0.00008, 0.00002),
             "GBPUSD": (0.00010, 0.00002), "USDJPY": (0.010, 0.002)}


def load(sym: str, tf: str) -> pd.DataFrame:
    for cand in (f"{sym}_{tf}_long.csv", f"{sym}_{tf}.csv"):
        p = ROOT / "data" / cand
        if p.exists():
            d = pd.read_csv(p, parse_dates=["time"], index_col="time").sort_index()
            return d[~d.index.duplicated(keep="last")]
    raise FileNotFoundError(f"no data for {sym} {tf}")


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def resolve_trades(df: pd.DataFrame, entry_idx: np.ndarray, direction: np.ndarray,
                   stop_atr: float, rr: float, max_bars: int, cost: float,
                   a: np.ndarray | None = None, stop_first: bool = True) -> pd.DataFrame:
    """Resolve fixed stop / RR-target trades and RECORD MFE & MAE in R units.

    Entry at the NEXT bar's open (causal). `stop_first=True` is this repo's universal
    conservative convention: when both barriers fall inside one bar, assume the stop.
    Flipping it to False brackets the optimistic side — the gap between the two IS the
    intrabar-path bias (P0.1).
    """
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    a = atr(df).values if a is None else a
    n = len(c)
    rows = []
    for i, d in zip(entry_idx, direction):
        if i + 1 >= n or not np.isfinite(a[i]) or a[i] <= 0:
            continue
        entry = o[i + 1]
        risk = stop_atr * a[i]                      # 1R in price
        stop = entry - d * risk
        tgt = entry + d * rr * risk
        mfe = mae = 0.0
        out_r, bars, reason = None, 0, "open"
        for k in range(i + 1, min(i + 1 + max_bars, n)):
            up = (h[k] - entry) * d / risk          # excursions in R units
            dn = (l[k] - entry) * d / risk
            if d < 0:
                up, dn = (entry - l[k]) / risk, (entry - h[k]) / risk
            mfe, mae = max(mfe, up), min(mae, dn)
            hit_sl = (l[k] <= stop) if d > 0 else (h[k] >= stop)
            hit_tp = (h[k] >= tgt) if d > 0 else (l[k] <= tgt)
            bars = k - i
            if stop_first:
                if hit_sl:
                    out_r, reason = -1.0, "sl"; break
                if hit_tp:
                    out_r, reason = rr, "tp"; break
            else:
                if hit_tp:
                    out_r, reason = rr, "tp"; break
                if hit_sl:
                    out_r, reason = -1.0, "sl"; break
        if out_r is None:                            # time barrier
            j = min(i + max_bars, n - 1)
            out_r, reason = d * (c[j] - entry) / risk, "time"
        rows.append(dict(i=int(i), time=df.index[i], dir=int(d), entry=float(entry),
                         risk=float(risk), r_gross=float(out_r),
                         r_net=float(out_r - cost / risk),   # round-trip cost in R
                         mfe_r=float(mfe), mae_r=float(mae), bars=int(bars),
                         reason=reason, cost_r=float(cost / risk)))
    return pd.DataFrame(rows)


def summarise(t: pd.DataFrame, label: str) -> dict:
    if t.empty:
        return dict(label=label, n=0)
    win = (t.r_net > 0).mean()
    se = float(np.sqrt(win * (1 - win) / len(t)))
    return dict(label=label, n=len(t), win=win, se=se, exp_r=float(t.r_net.mean()),
                cost_r=float(t.cost_r.mean()), tp=float((t.reason == "tp").mean()),
                sl=float((t.reason == "sl").mean()), time=float((t.reason == "time").mean()),
                mfe=float(t.mfe_r.mean()), mae=float(t.mae_r.mean()))


def random_entries(df: pd.DataFrame, n: int, seed: int, warm: int = 200) -> tuple:
    rng = np.random.default_rng(seed)
    idx = rng.integers(warm, len(df) - 2, size=n)
    return idx, rng.choice([-1, 1], size=n)


# --------------------------------------------------------------- P0.1 bias
def stage_bias(syms, tfs, stops, rrs, max_bars, n) -> None:
    print("\n=== P0.1 INTRABAR-PATH BIAS ===")
    print("Same random trades resolved stop-first (conservative) vs target-first")
    print("(optimistic). The GAP brackets how much the bar resolution can distort the")
    print("win rate. If the gap is >= the 1-2pp edge we hunt, that timeframe is unusable.\n")
    print(f"  {'sym':8s} {'tf':4s} {'stop':>5s} {'RR':>4s} {'n':>6s} "
          f"{'win(stop1st)':>12s} {'win(tgt1st)':>12s} {'BIAS pp':>8s} {'edge needed':>12s}")
    for sym in syms:
        spread, slip = LIVE_COST[sym]
        cost = spread + 2 * slip
        for tf in tfs:
            try:
                df = load(sym, tf)
            except FileNotFoundError:
                continue
            a = atr(df).values
            idx, dirs = random_entries(df, n, seed=11)
            for st in stops:
                for rr in rrs:
                    lo = resolve_trades(df, idx, dirs, st, rr, max_bars, cost, a, True)
                    hi = resolve_trades(df, idx, dirs, st, rr, max_bars, cost, a, False)
                    if lo.empty:
                        continue
                    w1, w2 = (lo.r_net > 0).mean(), (hi.r_net > 0).mean()
                    need = float(lo.cost_r.mean()) / (1 + rr)
                    flag = "  <-- UNUSABLE" if (w2 - w1) >= need else ""
                    print(f"  {sym:8s} {tf:4s} {st:5.1f} {rr:4.1f} {len(lo):6d} "
                          f"{w1:11.1%} {w2:11.1%} {(w2-w1)*100:7.2f} {need*100:11.2f}{flag}")


# ------------------------------------------------------------- P0.2 random
def stage_random(syms, tfs, stops, rrs, max_bars, n) -> None:
    print("\n=== P0.2 EMPIRICAL RANDOM-ENTRY BASELINE ===")
    print("The real benchmark. Theory says 1/(1+RR); reality embeds wick path, the")
    print("time barrier and spread-on-entry. 'need' = empirical random + cost hurdle.\n")
    print(f"  {'sym':8s} {'tf':4s} {'stop':>5s} {'RR':>4s} {'n':>6s} {'theory':>7s} "
          f"{'EMPIRICAL':>10s} {'expR':>7s} {'cost/R':>7s} {'need':>7s} {'edge pp':>8s} {'tp/sl/time':>16s}")
    rows = []
    for sym in syms:
        spread, slip = LIVE_COST[sym]
        cost = spread + 2 * slip
        for tf in tfs:
            try:
                df = load(sym, tf)
            except FileNotFoundError:
                continue
            a = atr(df).values
            idx, dirs = random_entries(df, n, seed=7)
            for st in stops:
                for rr in rrs:
                    t = resolve_trades(df, idx, dirs, st, rr, max_bars, cost, a)
                    if t.empty:
                        continue
                    s = summarise(t, f"{sym}/{tf}")
                    theory = 1 / (1 + rr)
                    need = (1 + s["cost_r"]) / (1 + rr)
                    edge = (need - s["win"]) * 100
                    rows.append(dict(sym=sym, tf=tf, stop=st, rr=rr, **s,
                                     theory=theory, need=need, edge_pp=edge))
                    print(f"  {sym:8s} {tf:4s} {st:5.1f} {rr:4.1f} {s['n']:6d} "
                          f"{theory:6.1%} {s['win']:9.1%} {s['exp_r']:+7.3f} "
                          f"{s['cost_r']:6.1%} {need:6.1%} {edge:7.2f} "
                          f"{s['tp']:.2f}/{s['sl']:.2f}/{s['time']:.2f}")
    if rows:
        R = pd.DataFrame(rows)
        out = ROOT / "data" / "v5_runs" / "intraday"
        out.mkdir(parents=True, exist_ok=True)
        R.to_csv(out / "p0_random_baseline.csv", index=False)
        print(f"\n  wrote {out/'p0_random_baseline.csv'}")
        ok = R[(R.edge_pp <= 3.0)].sort_values("edge_pp")
        print(f"\n  CELLS PASSING THE <=3pp GATE: {len(ok)}/{len(R)}")
        for _, x in ok.head(10).iterrows():
            print(f"    {x['sym']}/{x['tf']} stop {x['stop']}xATR RR{x['rr']:.0f} "
                  f"-> need +{x['edge_pp']:.2f}pp over random {x['win']:.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="random", choices=["bias", "random", "oracle"])
    ap.add_argument("--sym", action="append", default=None)
    ap.add_argument("--tf", action="append", default=None)
    ap.add_argument("--stops", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    ap.add_argument("--rrs", type=float, nargs="+", default=[2.0, 3.0])
    ap.add_argument("--max-bars", type=int, default=96)
    ap.add_argument("-n", type=int, default=10000)
    args = ap.parse_args()
    syms = args.sym or ["XAUUSD"]
    tfs = args.tf or ["M15", "M30"]
    if args.stage == "bias":
        stage_bias(syms, tfs, args.stops, args.rrs, args.max_bars, args.n)
    elif args.stage == "random":
        stage_random(syms, tfs, args.stops, args.rrs, args.max_bars, args.n)
    else:
        raise SystemExit("oracle stage lands next")


if __name__ == "__main__":
    main()
