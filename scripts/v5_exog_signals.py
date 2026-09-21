"""Forward-looking and macro inputs as XAUUSD signals: implied vol, real yields, the dollar.

WHY THESE THREE, AND WHY THEY ARE NOT MORE PRICE. §3bg tested positioning — a new data type —
and it lost to buy-and-hold in 0 of 36 cells. The ranked remainder of the standard external
playbook for gold beyond price is (1) options-implied information, (2) real yields, (3) flows.
This tests all of it. GVZ is the genuinely different KIND of input: it is forward-looking, a
market forecast rather than a realised outcome, and no study in this file has ever used one.

ALIGNMENT, WHICH WAS WRONG ON THE FIRST ATTEMPT. Every series here is stamped at the US close.
Sampling gold at the UTC day end instead put the two ~4 hours apart and dragged the GLD-vs-gold
daily correlation to 0.9565; sampling at **16:00 ET, DST-correct** lifts it to 0.9696. The
residual is the ETF's NAV basis and its single closing print, not a timing error. Everything
below uses the 16:00 ET sample.

SANITY, CHECKED BEFORE TESTING (this is what the data means, verified not assumed):
    GVZ vs 21d realised gold vol   corr +0.79, and implied averages **+3.1 vol points above
                                   realised** -- the volatility risk premium, textbook
    TIP return vs 10y yield change corr -0.71  (a bond ETF falls when yields rise)
    gold vs DXY daily returns      corr -0.41  (gold is a short-dollar asset)

THE BENCHMARK IS HOLDING GOLD, NOT ZERO. §3bg's decisive finding, and §3az's before it: over this
sample gold's unconditional drift is +24.1bp / 5d (t +5.97), +47.8 / 10d (t +8.53), **+93.9bp /
20d (t +12.00)**. A t of 12 is the thing to beat. So the statistic here is the conditional mean
MINUS the unconditional mean at the same horizon -- "are these days better than average", not
"are these days positive". Comparing to zero in a sample like this flatters anything.

DECLARED GRID -- 60 cells, fixed before running:

    measure   GVZ level / GVZ-minus-realised (the risk premium itself) /
              TIP 20d change / TNX 20d change / DXY 20d change              5
    rule      HIGH->LONG / HIGH->SHORT (exact mirrors)                      2
    threshold extreme 20% / 30% of the trailing 3-year distribution         2
    horizon   5 / 10 / 20 US trading days                                   3
                                                                         = 60

Null on GROSS and on the SIGNED maximum (§3bf), skipped-sample column
(`measure-the-skipped-sample`), mirror check, per-year stability reported not optimised.

    python scripts/v5_exog_signals.py --draws 400
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "v5_runs" / "session_study"
EXOG = ROOT / "data" / "exog"

RULES = ("HIGH_LONG", "HIGH_SHORT")
EXTREME = (0.20, 0.30)
HORIZON = (5, 10, 20)
TRAIL = 756          # ~3 years of US trading days
COST_BP = 2.34


def _series(path: Path) -> pd.Series:
    """First data column, whatever it is called. The XAU file was written from an unnamed
    Series so its column is '0', not 'close' -- read positionally instead of guessing."""
    d = pd.read_csv(path, parse_dates=["time"], index_col="time")
    return d.iloc[:, 0].astype(float)


def measures() -> pd.DataFrame:
    g = _series(EXOG / "XAU_USCLOSE_D1.csv")
    E = {n: _series(EXOG / f"{n}_D1.csv") for n in ("GVZ", "TIP", "TNX", "DXY")}
    rg = np.log(g).diff()
    rv = rg.rolling(21).std() * np.sqrt(252) * 100          # realised vol in GVZ's units
    d = pd.DataFrame(index=g.index)
    d["gold"] = g
    d["GVZ_level"] = E["GVZ"].reindex(g.index)
    d["GVZ_minus_RV"] = d["GVZ_level"] - rv                  # the volatility risk premium
    for n in ("TIP", "TNX", "DXY"):
        s = E[n].reindex(g.index)
        d[f"{n}_chg20"] = s.diff(20) if n == "TNX" else np.log(s).diff(20) * 100
    for h in HORIZON:
        d[f"fwd{h}"] = (g.shift(-h) / g - 1) * 1e4
    return d.dropna(subset=["gold"])


def cells(d: pd.DataFrame):
    res, store = [], {}
    names = ["GVZ_level", "GVZ_minus_RV", "TIP_chg20", "TNX_chg20", "DXY_chg20"]
    for meas in names:
        v = d[meas]
        pct = np.full(len(v), np.nan)
        vv = v.values
        for i in range(TRAIL, len(vv)):
            w = vv[i - TRAIL:i]
            w = w[np.isfinite(w)]
            if len(w) > 100 and np.isfinite(vv[i]):
                pct[i] = float((w < vv[i]).mean())
        d[f"{meas}_pct"] = pct
        for ex in EXTREME:
            hi = d[f"{meas}_pct"] >= 1 - ex
            lo = d[f"{meas}_pct"] <= ex
            for rule in RULES:
                sgn = +1.0 if rule == "HIGH_LONG" else -1.0
                side = pd.Series(0.0, index=d.index)
                side[hi] = sgn
                side[lo] = -sgn
                for h in HORIZON:
                    m = (side != 0) & d[f"fwd{h}"].notna()
                    if m.sum() < 100:
                        continue
                    sub = d[m]
                    gross = (side[m] * sub[f"fwd{h}"]).values
                    base = d.loc[d[f"fwd{h}"].notna(), f"fwd{h}"].mean()
                    skipped = d.loc[~m & d[f"fwd{h}"].notna(), f"fwd{h}"]
                    excess = gross - base                    # vs HOLDING gold, not vs zero
                    yrs = pd.DatetimeIndex(sub.index).year
                    per = pd.Series(excess, index=yrs).groupby(level=0).mean()
                    key = f"{meas}_{rule}_{int(ex*100)}_{h}d"
                    store[key] = excess
                    res.append(dict(
                        cell=key, measure=meas, rule=rule, extreme=ex, horizon=h,
                        n=int(m.sum()), per_week=m.sum() / ((sub.index.max()-sub.index.min()).days/7),
                        gross_bp=float(gross.mean()), base_bp=float(base),
                        excess_bp=float(excess.mean()),
                        t_excess=float(excess.mean()/(excess.std()/np.sqrt(len(excess)))),
                        skipped_bp=float(skipped.mean()) if len(skipped) else np.nan,
                        years_pos=int((per > 0).sum()), years=int(len(per))))
    return pd.DataFrame(res), store


def maxstat_signed(store: dict, draws: int, block: int = 25, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    obs = {k: a.mean()/(a.std()/np.sqrt(len(a))) for k, a in store.items()}
    best = max(obs, key=lambda k: obs[k])
    mx = np.empty(draws)
    for i in range(draws):
        m = -9e9
        for a in store.values():
            n = len(a); nb = int(np.ceil(n/block))
            st = rng.integers(0, max(n-block, 1), nb)
            b = np.concatenate([a[x:x+block] for x in st])[:n] - a.mean()
            m = max(m, b.mean()/(b.std()/np.sqrt(n)))
        mx[i] = m
    q = lambda v: float(np.percentile(mx, v))
    return dict(cells=len(store), draws=draws, best_cell=best, observed_t=float(obs[best]),
                p50=q(50), p95=q(95), p99=q(99), p_value=float(np.mean(mx >= obs[best])))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=400)
    a = ap.parse_args()
    d = measures()
    df, store = cells(d)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "exog_cells.csv", index=False)
    print("=" * 110)
    print(f"EXOGENOUS SIGNALS — {len(df)} of 60 declared cells, {len(d):,} US trading days "
          f"{d.index.min().date()} -> {d.index.max().date()}")
    print("=" * 110)
    print("  BASE RATES (what holding gold pays -- the benchmark):")
    for h in HORIZON:
        v = d[f"fwd{h}"].dropna()
        print(f"    {h:2d}d: {v.mean():+7.1f}bp  t {v.mean()/(v.std()/np.sqrt(len(v))):+6.2f}")
    print(f"\n  {'cell':32s} {'n':>5s} {'gross':>8s} {'base':>7s} {'EXCESS':>8s} {'t':>6s} "
          f"{'skipped':>8s} {'yrs+':>7s}")
    for _, r in df.sort_values("t_excess", ascending=False).head(10).iterrows():
        print(f"  {r['cell']:32s} {r['n']:5d} {r['gross_bp']:+8.1f} {r['base_bp']:+7.1f} "
              f"{r['excess_bp']:+8.1f} {r['t_excess']:+6.2f} {r['skipped_bp']:+8.1f} "
              f"{r['years_pos']:3d}/{r['years']:<3d}")
    print(f"  ... worst excess {df.excess_bp.min():+.1f}bp")
    print(f"\n  cells with EXCESS > 0 (beat holding gold): {(df.excess_bp>0).sum()} of {len(df)}")
    print(f"  cells beating the SKIPPED days:            "
          f"{(df.gross_bp>df.skipped_bp).sum()} of {len(df)}")
    for rule in RULES:
        s = df[df.rule == rule]
        print(f"    {rule:10s} mean excess {s.excess_bp.mean():+8.2f}bp  "
              f"positive {(s.excess_bp>0).sum()}/{len(s)}")
    print("\n  by measure (is any ONE input carrying information?):")
    for meas, s in df.groupby("measure"):
        print(f"    {meas:14s} best excess {s.excess_bp.max():+7.1f}bp  "
              f"best t {s.t_excess.max():+5.2f}  positive {(s.excess_bp>0).sum()}/{len(s)}")
    n = maxstat_signed(store, a.draws)
    (OUT / "exog_null.json").write_text(json.dumps(n, indent=2))
    print("=" * 110)
    print(f"  best signed t {n['observed_t']:+.3f} ({n['best_cell']})")
    print(f"  null of the max signed t: p50 {n['p50']:+.3f}  p95 {n['p95']:+.3f}  "
          f"p99 {n['p99']:+.3f}")
    print(f"  Reality-Check p {n['p_value']:.4f}  -> "
          f"{'SURVIVES' if n['p_value']<0.05 else 'does not survive'}")


if __name__ == "__main__":
    main()
