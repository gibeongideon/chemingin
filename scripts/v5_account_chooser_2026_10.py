#!/usr/bin/env python3
"""Which 5%ers EVALUATION to buy, re-run against the CURRENT terms and the CURRENTLY DEPLOYED book.

3al answered this on 2026-09-09, but TWO THINGS HAVE CHANGED and both matter:

  1. HIGH STAKES' DAILY LIMIT WENT 5% -> 3%, and its price $491 -> $149 ("100K Summer Plan").
     That is not a discount, it is a different product. 3al's whole finding was that the binding
     constraint decides, and it measured the 3% daily limit DESTROYING the 1-Step Growth plan
     (fail_day 36.3%, pass 61.5% at a 5% dial) while 5%-daily High Stakes passed 98.9%.
     So the cheap plan now carries the constraint that killed the other one.

  2. THE DEPLOYED BOOK IS NO LONGER GOLD-ONLY. 3al simulated the manual gold champion
     (net Sharpe +1.084, vol 13.9%, maxDD -19.8%). What runs now is XAU+BTC (3bs: Sharpe +1.286 at
     Switch costs, maxDD -12.5%). A book with a THINNER LEFT TAIL is exactly what a daily-loss rule
     rewards -- book-shootout established that Sharpe does not predict pass rate, the DIAL-SCALED
     LEFT TAIL against the daily limit does -- so the ranking could move.

Bootcamp max-loss is also re-read from the current page and it is NOT what 3al encoded: the
screenshot shows $10,000 on a 100K first step (10%), then $7,500 on 150K and $10,000 on 200K
(5% each). 3al had [5%, 5%, 5%]. The first step is more generous than previously modelled.

AMBIGUITY DECLARED: the Bootcamp page lists THREE balance steps but only TWO profit targets
(5%, 6%). Both readings are simulated -- a 3-step and a 2-step evaluation -- and both are reported,
because guessing here would change the answer silently.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import v5_account_chooser as ac                                   # noqa: E402
from v5_xau_champion_lifts import champion_recipe                 # noqa: E402
from src.v5 import mtf_frames                                     # noqa: E402

NSIM, BLOCK, MAXD, DAY_SAFETY = 1200, 21, 756, 1.5
ac.NSIM, ac.BLOCK, ac.MAXD, ac.DAY_SAFETY = NSIM, BLOCK, MAXD, DAY_SAFETY

PLANS = {
    "3-Step Bootcamp ($225+$350)": dict(
        fee=225.0, funded_fee=350.0, balances=[100_000, 150_000, 200_000],
        funded_balance=250_000, targets=[0.05, 0.06, 0.06],
        maxloss=[0.10, 0.05, 0.05], dayloss=[None, None, None], consistency=None,
        funded_maxloss=0.04, split=1.00, leverage=30),
    "  ^ same, read as 2-step eval": dict(
        fee=225.0, funded_fee=350.0, balances=[100_000, 150_000],
        funded_balance=250_000, targets=[0.05, 0.06],
        maxloss=[0.10, 0.05], dayloss=[None, None], consistency=None,
        funded_maxloss=0.04, split=1.00, leverage=30),
    "High Stakes ($149, 3% daily)": dict(
        fee=149.0, funded_fee=0.0, balances=[100_000, 100_000], funded_balance=100_000,
        targets=[0.10, 0.05], maxloss=[0.10, 0.10], dayloss=[0.03, 0.03], consistency=0.50,
        funded_maxloss=0.10, split=1.00, leverage=100),
    "  ^ at the OLD 5% daily (3al)": dict(
        fee=491.0, funded_fee=0.0, balances=[100_000, 100_000], funded_balance=100_000,
        targets=[0.10, 0.05], maxloss=[0.10, 0.10], dayloss=[0.05, 0.05], consistency=0.50,
        funded_maxloss=0.10, split=1.00, leverage=100),
}


def xau_btc_book() -> pd.Series:
    """The DEPLOYED book (magic 360595): XAU champion H4 + BTC D1, 50/50, Switch costs."""
    m15 = pd.read_csv(ROOT / "data/XAUUSD_M15_long.csv", parse_dates=["time"], index_col="time")
    m15 = m15[~m15.index.duplicated(keep="last")].sort_index()
    h4 = mtf_frames.aggregate(m15, "H4").dropna(subset=["close"])
    legs = []
    for close, scale, sp, carry in ((h4["close"].astype(float), 1.0, 0.50, -5.76),
                                    (pd.read_csv(ROOT / "data/BTC_D1_long.csv",
                                                 parse_dates=["time"], index_col="time")
                                     .sort_index()["close"].astype(float), 1 / 6, 4.46, -15.0)):
        close = close[~close.index.duplicated(keep="last")]
        bpy = len(close) / max((close.index[-1] - close.index[0]).days / 365.25, 1e-9)
        ret = close.pct_change()
        vol = ret.ewm(halflife=42, min_periods=20).std() * np.sqrt(bpy)
        pos = (champion_recipe(close, scale, 1.5).clip(-2, 2) * (0.10 / vol)).clip(-8, 8).shift(1)
        r = (pos * ret - pos.diff().abs().fillna(0.0) * (sp / 1e4) * 1.5
             - pos.abs() * (abs(carry) / 100.0) / bpy).dropna()
        legs.append((1 + r).resample("1D").prod().sub(1))
    bd = pd.bdate_range(max(l.index[0] for l in legs).normalize(),
                        min(l.index[-1] for l in legs).normalize())
    return (0.5 * legs[0].reindex(bd).fillna(0.0) + 0.5 * legs[1].reindex(bd).fillna(0.0)).dropna()


def main() -> int:
    for book_name, r in (("GOLD champion (what 3al used)", ac.manual_gold_returns()),
                         ("XAU+BTC book (DEPLOYED now)", xau_btc_book())):
        rv = r.to_numpy()
        sd = rv.std() * np.sqrt(252)
        sr = rv.mean() / rv.std() * np.sqrt(252)
        eq = (1 + r).cumprod()
        print("\n" + "=" * 108)
        print("BOOK: %s   Sharpe %+.3f  vol %.1f%%  maxDD %.1f%%  (%d days)"
              % (book_name, sr, 100 * sd, 100 * (eq / eq.cummax() - 1).min(), len(r)))
        print("=" * 108)
        print("  %-30s %6s %7s %8s %10s %9s %9s" %
              ("plan", "dial", "pass%", "med mo", "fail_day%", "fail_dd%", "timeout%"))
        print("  " + "-" * 92)
        for name, plan in PLANS.items():
            for dial in (0.03, 0.05, 0.07, 0.10):
                k = dial / sd
                out = [ac.sim_plan(rv, k, plan, seed) for seed in range(3)]
                agg = {kk: float(np.mean([o[kk] for o in out])) for kk in out[0]}
                print("  %-30s %6.0f%% %7.1f %8.1f %10.1f %9.1f %9.1f" %
                      (name if dial == 0.03 else "", 100 * dial, agg["passpct"],
                       agg["med_mo"], agg["fail_day"], agg["fail_dd"], agg["timeout"]))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
