"""RANDOM-ENTRY CONTROL: does knowing a ZigZag pivot just confirmed help AT ALL?

The grid's `headroom` compares oracle precision to breakeven, but says nothing about whether the
PIVOT is doing the work. This places the IDENTICAL bracket at randomly chosen bars, matched on
count and on side mix, and asks whether pivot-located entries beat them. If they do not, the label
carries no information at the only time it could be acted on, and no model over it can either.
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, "/home/rock/Desktop/2026_Projects/Trader36/MT5")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "zz", "/home/rock/Desktop/2026_Projects/Trader36/MT5/scripts/v5_zigzag_ratio_rr.py")
zz = importlib.util.module_from_spec(spec); spec.loader.exec_module(zz)

RNG = np.random.default_rng(20261021)
N_NULL = 400

def random_entry_precision(d, atr, n_entries, side_mix, sl_atr, rr, horizon=30):
    hi, lo, cl = (d["high"].to_numpy(float), d["low"].to_numpy(float), d["close"].to_numpy(float))
    av = atr.to_numpy(float)
    valid = np.flatnonzero(np.isfinite(av) & (av > 0))
    valid = valid[(valid > 20) & (valid < len(cl) - horizon - 1)]
    out = np.empty(N_NULL)
    for t in range(N_NULL):
        idx = RNG.choice(valid, size=n_entries, replace=False)
        sides = RNG.choice([-1, 1], size=n_entries, p=[1 - side_mix, side_mix])
        w = l_ = 0
        for i, trade in zip(idx, sides):
            entry = cl[i]; sl_d = sl_atr * av[i]; tp_d = sl_atr * rr * av[i]
            o, _ = zz.bracket_outcome(entry, int(trade), entry - trade * sl_d,
                                      entry + trade * tp_d, hi[i+1:], lo[i+1:], horizon)
            if o > 0: w += 1
            elif o < 0: l_ += 1
        out[t] = w / (w + l_) if (w + l_) else np.nan
    return out

gold, btc = zz.load("GOLD"), zz.load("BTC")
common = gold.index.intersection(btc.index)
ratio = pd.DataFrame({
    "open": btc["open"].reindex(common)/gold["open"].reindex(common),
    "high": btc["high"].reindex(common)/gold["low"].reindex(common),
    "low":  btc["low"].reindex(common)/gold["high"].reindex(common),
    "close":btc["close"].reindex(common)/gold["close"].reindex(common)}).dropna()

print("="*104)
print("RANDOM-ENTRY CONTROL -- pivot-located brackets vs the SAME bracket at random bars")
print("="*104)
for name, d in (("RATIO (BTC/XAU)", ratio), ("GOLD [control]", gold.loc[common])):
    atr = zz.wilder_atr(d["high"], d["low"], d["close"])
    print(f"\n{name}")
    print(f"  {'k':>4} {'SL':>5} {'RR':>5} {'n':>5} {'pivot p':>8} {'random p50':>11} "
          f"{'random p95':>11} {'lift':>8}  verdict")
    for k in (1.0, 2.0, 3.0):
        piv = zz.zigzag_pivots(d["high"], d["low"], d["close"], atr, k)
        if len(piv) < 20: continue
        side_mix = float((piv["side"] > 0).mean())
        for sl_atr, rr in ((1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (3.0, 1.0), (3.0, 1.5)):
            orc = zz.oracle_precision(d, atr, piv, sl_atr, rr)
            if not np.isfinite(orc["precision"]) or orc["resolved"] < 20: continue
            null = random_entry_precision(d, atr, orc["resolved"], side_mix, sl_atr, rr)
            p50, p95 = np.nanpercentile(null, 50), np.nanpercentile(null, 95)
            lift = orc["precision"] - p50
            v = "beats random" if orc["precision"] > p95 else (
                "INSIDE noise" if orc["precision"] > p50 else "WORSE than random")
            print(f"  {k:>4.1f} {sl_atr:>5.1f} {rr:>5.1f} {orc['resolved']:>5} "
                  f"{orc['precision']:>8.3f} {p50:>11.3f} {p95:>11.3f} {lift:>+8.3f}  {v}")
