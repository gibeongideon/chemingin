#!/usr/bin/env python3
"""Downside-vol as a SIZE DIAL instead of a veto: take every fire, but half-size the risky ones.

USER, 2026-11-13: test the size-reduction variant before wiring anything.

WHY THIS IS THE BETTER SHAPE OF THE SAME IDEA.
  * §3cy/§3cx closed the giveback family with one instruction: "to shrink swings cut LOT SIZE,
    never scale out". A veto is the crudest possible size dial — size 0 — and it throws away the
    bull-market upside that §3et showed the veto costs (−0.080 Sharpe on CME 2019-26).
  * THE EVENT SET STAYS FIXED. Every fire is still taken, so §3co is satisfied by construction
    and the whole "a filter changes exposure" problem disappears. The veto could not say this:
    skipping a fire frees the one-at-a-time harness to take a LATER one, which is why size 0 here
    is NOT the same thing as the §3es veto.
  * The exit path is size-independent — stop and trail are in ATR, TP in R — so a trade's R is
    unchanged by its size. The variants below are therefore exact, not re-simulated.

THE CONTROL. A conditional size cut must beat a RANDOM conditional size cut that half-sizes the
SAME NUMBER of trades (200 resamples). Uniform risk is a pure dial — `xau-riskfrac-sizing-table`
measured Sharpe invariant at +1.02 across 0.25-4% — so any Sharpe change here must come from the
CONDITIONING, and the random control is what isolates it.

PRIOR, recorded before the run: `xau-per-trade-prob-sizing-disproven` ("vol-targeting only cuts
drawdown") and `xau-volregime-taper-lead` (taper closed at sleeve level, gold-specific 1/4).
Expect drawdown to improve and Sharpe to move little.

    python scripts/v5_zigzag_vol_taper.py
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
from scripts.v5_zigzag_downtrend_guards import guards  # noqa: E402
from scripts.v5_zigzag_long_history import OUT as ZZ, RISK, SRC, THR, backtest, load, sharpe  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "elliott"
N_RANDOM = 200
MULTS = (1.0, 0.75, 0.5, 0.25, 0.0)
RNG = np.random.default_rng(20261113)


def daily(tr: pd.DataFrame, mult: np.ndarray, start: str, last: pd.Timestamp) -> pd.Series:
    """Daily return series with a per-trade size multiplier. R is size-independent, so this is
    exact — the same construction as v5_zigzag_long_history.backtest, with `mult` folded in."""
    days = pd.date_range(pd.Timestamp(start).normalize(), last.normalize(), freq="D")
    ret = pd.Series(0.0, index=days)
    for m, r in zip(mult, tr.itertuples()):
        ret[r.t_out.normalize()] += RISK * m * r.R
    return ret[ret.index.dayofweek < 5]


def shape(ret: pd.Series, yrs: float) -> tuple:
    eq = (1 + ret).cumprod()
    return (sharpe(ret), float(eq.iloc[-1] ** (1 / yrs) - 1),
            float((eq / eq.cummax() - 1).min()))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for src, cfg in SRC.items():
        g = load(src, cfg["gold"])
        p = (pd.read_csv(ZZ / src / "probs.csv", index_col=0, parse_dates=True)
             .iloc[:, 0].reindex(g.index).to_numpy())
        start, swap = cfg["start"], cfg["swap"]
        yrs = (g.index[-1] - pd.Timestamp(start)).days / 365.25
        inw = g.index >= pd.Timestamp(start)
        fire = np.isfinite(p) & (p >= THR)
        G = guards(g)
        calm = G["downside vol below median"]                 # True = calm, size stays full
        bar_of = pd.Series(np.arange(len(g)), index=g.index)
        mid = g.index[inw][int(inw.sum() // 2)]

        print("=" * 116)
        print(f"{src.upper()}  {start} -> {g.index[-1]:%Y-%m-%d} ({yrs:.1f}y)   split {mid:%Y-%m-%d}"
              f"   dial: size x M when downside_vol(500) >= median(2000)")
        print("=" * 116)

        for bname, bmask in (("breaker only (5 of 6)", G["circuit breaker k=1.0 LIVE"]),
                             ("breaker + champion gate (Switch 360601)",
                              G["circuit breaker k=1.0 LIVE"] & G["champion fc >= 0.5  LIVE"])):
            tr, _ = backtest(g, fire & bmask, start, swap)
            if len(tr) < 50:
                print(f"  {bname}: only {len(tr)} trades"); continue
            ent = bar_of.reindex(tr.t_in).to_numpy()          # entry bar of each trade
            risky = ~calm[ent]                                # high downside vol at entry
            nrisk = int(risky.sum())
            print(f"\n  BASELINE {bname:<44} {len(tr)} trades, "
                  f"{nrisk} ({100 * nrisk / len(tr):.0f}%) taken in high downside vol")
            print(f"    {'size on risky trades':<24}{'Sharpe':>9}{'CAGR':>9}{'maxDD':>9}"
                  f"{'Sh 1st':>9}{'Sh 2nd':>9}{'pctile':>9}{'both halves':>13}")

            base = None
            for m in MULTS:
                mult = np.where(risky, m, 1.0)
                r = daily(tr, mult, start, g.index[-1])
                sh, cg, dd = shape(r, yrs)
                h1, h2 = shape(r[r.index < mid], yrs / 2)[0], shape(r[r.index >= mid], yrs / 2)[0]
                if m == 1.0:
                    base = (sh, cg, dd, h1, h2)
                    print(f"    {'1.00  (unchanged)':<24}{sh:>9.3f}{100 * cg:>8.2f}%"
                          f"{100 * dd:>8.2f}%{h1:>9.3f}{h2:>9.3f}{'—':>9}{'—':>13}")
                    continue
                # random control: half-size the SAME NUMBER of trades, chosen at random
                sh_r = np.empty(N_RANDOM)
                for i in range(N_RANDOM):
                    pick = np.zeros(len(tr), dtype=bool)
                    pick[RNG.choice(len(tr), size=nrisk, replace=False)] = True
                    sh_r[i] = sharpe(daily(tr, np.where(pick, m, 1.0), start, g.index[-1]))
                pct = float((sh_r < sh).mean())
                both = (h1 > base[3]) and (h2 > base[4])
                print(f"    {m:<24.2f}{sh:>9.3f}{100 * cg:>8.2f}%{100 * dd:>8.2f}%"
                      f"{h1:>9.3f}{h2:>9.3f}{100 * pct:>8.1f}%"
                      f"{('YES' if both else 'no'):>13}")
                rows.append(dict(src=src, baseline=bname, mult=m, sharpe=sh, cagr=cg, maxdd=dd,
                                 sh_h1=h1, sh_h2=h2, base_sharpe=base[0], base_dd=base[2],
                                 pctile=pct, both_halves=both, n=len(tr), n_risky=nrisk))
    if rows:
        d = pd.DataFrame(rows)
        d.to_csv(OUT / "vol_taper.csv", index=False)
        print("\n" + "=" * 116)
        print("SUMMARY — size dial on high-downside-vol fires, event set FIXED")
        print("=" * 116)
        for m in [x for x in MULTS if x != 1.0]:
            s = d[d.mult == m]
            print(f"  size {m:.2f}:  dSharpe " + "  ".join(
                f"{r.src}/{'champ' if 'champion' in r.baseline else 'brkr'} "
                f"{r.sharpe - r.base_sharpe:+.3f} (pct {100 * r.pctile:.0f}%, "
                f"dDD {100 * (r.maxdd - r.base_dd):+.2f}pp)" for r in s.itertuples()))
        print(f"\nwrote {OUT / 'vol_taper.csv'}")


if __name__ == "__main__":
    main()
