"""THE CORRECTED TEST: unresolved trades marked to market, and selection done on H1 only.

Two fixes to my own harness, both of which flatter the result if left in:
  1. UNRESOLVED TRADES WERE DROPPED. 37 of 110 pivots (34%) at the strongest cell never touched
     TP or SL inside the horizon and were excluded from precision. Dropping them conditions on the
     outcome -- the survivorship bug filed in 'order-of-events needs intraday' (dropping outside
     days was survivorship on the outcome). Every trade now settles: TP, SL, or marked to market
     at the horizon.
  2. SELECTION WAS POOLED. Cells chosen on full-sample performance then "confirmed" on both halves
     is near-circular, since pooled is the average of the halves. Selection now happens on H1 and
     is evaluated on H2 only (CONTROL #3, walk-forward the selection).

EV is reported in bp of NOTIONAL per trade, net of the ratio's 7.44bp two-leg round trip, and
benchmarked against the same bracket at random bars.
"""
import sys, numpy as np, pandas as pd, importlib.util
spec = importlib.util.spec_from_file_location(
    "zz", "/home/rock/Desktop/2026_Projects/Trader36/MT5/scripts/v5_zigzag_ratio_rr.py")
zz = importlib.util.module_from_spec(spec); spec.loader.exec_module(zz)
RNG = np.random.default_rng(11)
HORIZON = 30

def settle(d, atr, entries, sl_atr, rr, horizon=HORIZON):
    """Every trade settles. Returns per-trade bp return (net of cost) and win flag."""
    hi, lo, cl = (d["high"].to_numpy(float), d["low"].to_numpy(float), d["close"].to_numpy(float))
    av = atr.to_numpy(float); rows=[]
    for ci, side in entries:
        if ci+1 >= len(cl) or not np.isfinite(av[ci]) or av[ci] <= 0: continue
        entry = cl[ci]; trade=int(side)
        sl_d, tp_d = sl_atr*av[ci], sl_atr*rr*av[ci]
        sl_px, tp_px = entry - trade*sl_d, entry + trade*tp_d
        exit_px, resolved = None, False
        end = min(ci+1+horizon, len(cl))
        for j in range(ci+1, end):
            if trade > 0: tp_hit, sl_hit = hi[j] >= tp_px, lo[j] <= sl_px
            else:         tp_hit, sl_hit = lo[j] <= tp_px, hi[j] >= sl_px
            if tp_hit and sl_hit: exit_px, resolved = sl_px, True; break   # pessimistic
            if tp_hit:            exit_px, resolved = tp_px, True; break
            if sl_hit:            exit_px, resolved = sl_px, True; break
        if exit_px is None:                       # MARKED TO MARKET, not dropped
            exit_px = cl[end-1]
        ret_bp = trade*(exit_px-entry)/entry*1e4 - zz.COST_BP["RATIO"]
        rows.append((d.index[ci], ret_bp, 1 if ret_bp > 0 else 0, resolved))
    return pd.DataFrame(rows, columns=["time","ret_bp","win","resolved"]).set_index("time")

gold, btc = zz.load("GOLD"), zz.load("BTC")
common = gold.index.intersection(btc.index)
ratio = pd.DataFrame({
    "open": btc["open"].reindex(common)/gold["open"].reindex(common),
    "high": btc["high"].reindex(common)/gold["low"].reindex(common),
    "low":  btc["low"].reindex(common)/gold["high"].reindex(common),
    "close":btc["close"].reindex(common)/gold["close"].reindex(common)}).dropna()
atr = zz.wilder_atr(ratio["high"], ratio["low"], ratio["close"])
mid = ratio.index[len(ratio)//2]

CELLS = [(k, sl, rr) for k in (1.0,2.0,3.0) for sl,rr in
         ((0.5,1.0),(1.0,1.0),(1.0,2.0),(2.0,1.0),(2.0,2.0),(3.0,1.0),(3.0,1.5))]
print("="*100); print("CORRECTED: all trades settle (none dropped); selection on H1, verdict on H2")
print("="*100)
print(f"  {'k':>4} {'SL':>4} {'RR':>4} | {'H1 n':>5} {'H1 p':>6} {'H1 EV':>8} | "
      f"{'H2 n':>5} {'H2 p':>6} {'H2 EV':>8} | {'unres%':>7}")
recs=[]
for k, sl, rr in CELLS:
    piv = zz.zigzag_pivots(ratio["high"], ratio["low"], ratio["close"], atr, k)
    if len(piv) < 20: continue
    ent = [(int(c), int(s)) for _, c, s in piv.itertuples(index=False)]
    t = settle(ratio, atr, ent, sl, rr)
    if t.empty: continue
    h1, h2 = t[t.index < mid], t[t.index >= mid]
    if len(h1) < 15 or len(h2) < 15: continue
    rec = dict(k=k, sl=sl, rr=rr, n1=len(h1), p1=h1.win.mean(), ev1=h1.ret_bp.mean(),
               n2=len(h2), p2=h2.win.mean(), ev2=h2.ret_bp.mean(),
               unres=1-t.resolved.mean())
    recs.append(rec)
    print(f"  {k:>4.1f} {sl:>4.1f} {rr:>4.1f} | {rec['n1']:>5} {rec['p1']:>6.3f} "
          f"{rec['ev1']:>+8.1f} | {rec['n2']:>5} {rec['p2']:>6.3f} {rec['ev2']:>+8.1f} | "
          f"{rec['unres']:>6.1%}")
R = pd.DataFrame(recs)
best = R.loc[R.ev1.idxmax()]
print(f"\n  SELECTED ON H1 ONLY: k={best.k} SL={best.sl} RR={best.rr}  (H1 EV {best.ev1:+.1f}bp)")
print(f"  OUT-OF-SAMPLE H2    : EV {best.ev2:+.1f}bp on {int(best.n2)} trades, "
      f"precision {best.p2:.3f}")
print(f"\n  cells with BOTH halves EV>0: {int(((R.ev1>0)&(R.ev2>0)).sum())} of {len(R)}")
print(f"  mean H1 EV {R.ev1.mean():+.1f}bp   mean H2 EV {R.ev2.mean():+.1f}bp   "
      f"corr(H1,H2) {R.ev1.corr(R.ev2):+.3f}")
# random-entry benchmark for the H1-selected cell, on H2 only
piv = zz.zigzag_pivots(ratio["high"], ratio["low"], ratio["close"], atr, best.k)
ent = [(int(c), int(s)) for _, c, s in piv.itertuples(index=False)]
t = settle(ratio, atr, ent, best.sl, best.rr); h2 = t[t.index >= mid]
side_mix = float((piv["side"]>0).mean())
av = atr.to_numpy(float); valid = np.flatnonzero(np.isfinite(av)&(av>0))
i_mid = ratio.index.get_loc(mid)
valid = valid[(valid > i_mid) & (valid < len(ratio)-HORIZON-1)]
null=[]
for _ in range(400):
    idx = RNG.choice(valid, size=len(h2), replace=False)
    sd = RNG.choice([-1,1], size=len(h2), p=[1-side_mix, side_mix])
    null.append(settle(ratio, atr, list(zip(idx, sd)), best.sl, best.rr).ret_bp.mean())
null=np.array(null)
print(f"\n  H2 random-entry control: median {np.median(null):+.1f}bp  "
      f"p95 {np.percentile(null,95):+.1f}bp   vs pivot {h2.ret_bp.mean():+.1f}bp")
print(f"  VERDICT: {'beats random OOS' if h2.ret_bp.mean() > np.percentile(null,95) else 'INSIDE the random-entry noise band'}")
