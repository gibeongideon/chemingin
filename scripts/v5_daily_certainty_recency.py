#!/usr/bin/env python3
"""Does RECENCY WEIGHTING change the daily-pattern verdict? (follow-up to 3bu)

The question: gold's regime genuinely changed -- 2026 daily range is 2.73%, about 8x its 2015
level -- so should the 761-pattern search weight recent data more heavily, and does the answer move?

*** THE ARITHMETIC TRAP THIS SCRIPT EXISTS TO PRICE. *** Weighting recent data does not add
information, it DISCARDS old information. Exponential weights with half-life h reduce the effective
sample to N_eff = (sum w)^2 / sum(w^2), and Gate A established that the best-of-K noise floor RISES
as N falls: at K=500 the p95 floor is 0.644 on 250 fires but 0.880 on 25. So a shorter half-life
must produce MORE apparent survivors even when nothing is there. Any honest comparison therefore has
to move the bar at the same time it moves the weights, which is exactly what naive recency weighting
fails to do.

Three things are measured, in increasing order of how much they matter:

  1. SURVIVOR COUNT at a FIXED bar          -- what a naive re-run would report.
  2. SURVIVOR COUNT at the N_eff-ADJUSTED bar -- the same test with the goalposts moved honestly.
  3. TRUE OUT-OF-SAMPLE                      -- select the best pattern using data up to a cutoff
                                                under each half-life, then score it on the held-out
                                                future. This is the only number that can settle it,
                                                because it is the only one recency weighting cannot
                                                flatter: the test data is not weighted at all.

PRIOR, recorded so the result is not read as a surprise either way: recency weighting was tried once
here before (range/extremes features) and did not help; 'recent data looks better' has twice been
the signature of an artifact (SMC-BOS 1.44 single-split vs 0.54 walk-forward; range-vol t +2.25
pooled but a 2022+ artifact). Note also that the pattern grid is almost entirely SCALE-FREE --
quantile buckets, streaks and z-scores -- so a volatility regime change should NOT break it, which
weakens the mechanistic case for reweighting before the statistics are even consulted.
"""
from __future__ import annotations

import importlib.util
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "v5_runs" / "daily_certainty"
RNG = np.random.default_rng(20261021)

_spec = importlib.util.spec_from_file_location("scan", ROOT / "scripts" / "v5_daily_certainty_scan.py")
scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scan)

HALF_LIVES = [None, 2000, 1000, 500, 250]      # trading days; None = flat (the 3bu baseline)
OOS_CUTOFF = "2024-10-21"                       # last 2 years held out, never weighted
MIN_FIRES = 25


def weights(index: pd.DatetimeIndex, half_life: int | None) -> np.ndarray:
    if half_life is None:
        return np.ones(len(index))
    age = np.arange(len(index))[::-1].astype(float)     # 0 = most recent bar
    return 0.5 ** (age / half_life)


def n_eff(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w ** 2).sum())


def noise_floor(base: float, n: int, k: int, trials: int = 3000, q: float = 0.95) -> float:
    n = max(int(n), 5)
    draws = RNG.binomial(n, base, size=(trials, k)) / n
    return float(np.quantile(draws.max(axis=1), q))


def weighted_precision(P: dict, nxt_up: pd.Series, w: np.ndarray) -> pd.DataFrame:
    y = nxt_up.to_numpy(float)
    ok = ~np.isnan(y)
    rows = []
    base = float(np.average(y[ok], weights=w[ok]))
    for name, m in P.items():
        mm = m.fillna(False).to_numpy(bool) & ok
        raw = int(mm.sum())
        if raw < MIN_FIRES:
            continue
        ww = w[mm]
        prec = float(np.average(y[mm], weights=ww))
        side = 1 if prec >= base else -1
        rows.append(dict(pattern=name, raw_fires=raw, eff_fires=n_eff(ww), side=side,
                         precision=prec if side == 1 else 1 - prec,
                         base=base if side == 1 else 1 - base))
    d = pd.DataFrame(rows)
    if not d.empty:
        d["excess"] = d["precision"] - d["base"]
        d.attrs["global_base"] = base
    return d


def main() -> None:
    gold, btc = scan.load("GOLD"), scan.load("BTC")
    f = scan.features(gold)
    x = scan.features(btc).reindex(f.index)
    P = scan.build_patterns(f, x, use_intrabar=True)
    nxt_ret = f["ret"].shift(-1)
    nxt_up = (nxt_ret > 0).astype(float).where(nxt_ret.notna())

    cut = pd.Timestamp(OOS_CUTOFF)
    in_mask = f.index < cut
    oos_mask = f.index >= cut

    print("=" * 104)
    print("DOES RECENCY WEIGHTING CHANGE THE VERDICT?  (GOLD daily, follow-up to 3bu)")
    print("=" * 104)
    print(f"  patterns K={len(P)}   bars {len(f):,}   "
          f"in-sample < {cut.date()} ({in_mask.sum():,})   "
          f"HELD OUT >= {cut.date()} ({oos_mask.sum():,}, never weighted)")

    print(f"\n--- 1 & 2. SURVIVORS: naive fixed bar vs the N_eff-adjusted bar ---")
    print(f"  {'half-life':>10} {'N_eff':>9} {'base':>7} {'best p':>8} {'fixed bar':>10} "
          f"{'surv@fixed':>11} {'adj bar':>9} {'surv@adj':>9}")
    print("  " + "-" * 88)
    fixed_bar = None
    summary = []
    for hl in HALF_LIVES:
        w = weights(f.index, hl)
        w_in = np.where(in_mask, w, 0.0)                     # fit only on in-sample
        res = weighted_precision(P, nxt_up, w_in)
        if res.empty:
            continue
        ne = float(np.median(res["eff_fires"]))
        # BUG FIXED: res["base"].iloc[0] is the FIRST pattern's SIDE-ADJUSTED base (0.477 if that
        # row chose the short side), not the global up-rate (0.525). The noise floor was being
        # built on the wrong base. Use the global base, and since every pattern is free to pick
        # whichever side favours it, the floor must be built on the FAVOURED side: max(b, 1-b).
        gb = float(res.attrs["global_base"])
        base = max(gb, 1.0 - gb)
        if fixed_bar is None:                                 # the flat-weight bar, held constant
            fixed_bar = noise_floor(base, np.median(res["raw_fires"]), len(res))
        adj_bar = noise_floor(base, ne, len(res))
        s_fixed = int((res["precision"] > fixed_bar).sum())
        s_adj = int((res["precision"] > adj_bar).sum())
        best = res.loc[res["precision"].idxmax()]
        print(f"  {str(hl):>10} {ne:>9.0f} {base:>7.3f} {best['precision']:>8.3f} "
              f"{fixed_bar:>10.3f} {s_fixed:>11} {adj_bar:>9.3f} {s_adj:>9}")
        summary.append(dict(half_life=hl, n_eff=ne, best=best["pattern"],
                            best_p=best["precision"], surv_fixed=s_fixed, surv_adj=s_adj,
                            adj_bar=adj_bar))

    print("\n  Reading: `surv@fixed` is what a naive recency re-run would report. `surv@adj` holds")
    print("  the FALSE-POSITIVE RATE constant instead of the threshold, which is the honest")
    print("  comparison -- a shorter half-life buys apparent winners by shrinking the sample.")

    print(f"\n--- 3. TRUE OUT-OF-SAMPLE: select under each half-life, score on held-out future ---")
    print(f"  {'half-life':>10} {'selected pattern':<34} {'IS p':>7} {'OOS p':>7} "
          f"{'OOS n':>6} {'OOS base':>9} {'OOS excess':>11}")
    print("  " + "-" * 92)
    y = nxt_up.to_numpy(float)
    oos_base = float(np.nanmean(y[oos_mask & ~np.isnan(y)]))
    oos_rows = []
    for rec in summary:
        hl = rec["half_life"]
        name = rec["best"]
        m = P[name].fillna(False).to_numpy(bool) & oos_mask & ~np.isnan(y)
        n_o = int(m.sum())
        if n_o < 5:
            print(f"  {str(hl):>10} {name[:34]:<34} {rec['best_p']:>7.3f} "
                  f"{'--':>7} {n_o:>6} {'--':>9} {'(too few OOS fires)':>11}")
            continue
        # score the side the in-sample fit chose, never re-chosen out of sample
        p_oos = float(np.mean(y[m]))
        side = 1 if rec["best_p"] >= 0.5 else 1
        res_in = weighted_precision(P, nxt_up, np.where(in_mask, weights(f.index, hl), 0.0))
        sd = int(res_in.loc[res_in["pattern"] == name, "side"].iloc[0])
        p_oos_sided = p_oos if sd == 1 else 1 - p_oos
        base_sided = oos_base if sd == 1 else 1 - oos_base
        print(f"  {str(hl):>10} {name[:34]:<34} {rec['best_p']:>7.3f} {p_oos_sided:>7.3f} "
              f"{n_o:>6} {base_sided:>9.3f} {p_oos_sided - base_sided:>+11.3f}")
        oos_rows.append(dict(half_life=hl, pattern=name, is_p=rec["best_p"],
                             oos_p=p_oos_sided, oos_n=n_o, oos_excess=p_oos_sided - base_sided))

    O = pd.DataFrame(oos_rows)
    O.to_csv(OUT / "recency_oos.csv", index=False)
    print("\n" + "=" * 104)
    print("VERDICT")
    print("=" * 104)
    if not O.empty:
        print(f"  mean OOS excess over base across half-lives: {O['oos_excess'].mean():+.3f}")
        print(f"  half-lives with POSITIVE OOS excess: {int((O['oos_excess'] > 0).sum())} of {len(O)}")
        c = O["is_p"].corr(O["oos_excess"])
        print(f"  corr(in-sample precision, OOS excess): {c:+.3f}"
              if pd.notna(c) else "  corr: n/a")
        best_hl = O.loc[O["oos_excess"].idxmax()]
        print(f"  best OOS half-life: {best_hl['half_life']} "
              f"({best_hl['oos_excess']:+.3f} excess on {int(best_hl['oos_n'])} fires)")
    print(f"\n  -> {OUT / 'recency_oos.csv'}")


if __name__ == "__main__":
    main()
