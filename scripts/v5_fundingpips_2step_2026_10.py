#!/usr/bin/env python3
"""Which FundingPips 2-Step plan, and can this book pass it? Simulated on the DEPLOYED book.

THE THREE PLANS (screenshot 2026-10-28, $100k 2-Step, MetaTrader 5):

  Standard  "Biggest Daily Loss"   P1 8% / P2 5%   daily 3% OR 5% (toggle)  max loss 10% STATIC
  Flex      "Biggest Max Loss"     P1 10% / P2 8%  daily 4%                 max loss 12% STATIC
  Pro       "Lowest Profit Target" P1 6% / P2 6%   daily 3%                 max loss  6% STATIC

STATIC max loss means measured from the INITIAL balance, never trailing. That is materially more
generous than a trailing wall and the repo has a logged trap about conflating the two
(risk_frac sizing table: "static-vs-trailing DD convention trap").

TWO PAID OPTIONS, BOTH OF WHICH THIS BOOK ACTUALLY NEEDS:

  SWAP FREE (+$55.50) -- financing is the single largest measured drag on this book: -3.79%/yr,
    worth -0.29 Sharpe (3q), and 3ag called a swap-free instrument "worth more than every signal
    result in this repo's history". Here it is a checkbox.
  SWING (+$55.50) -- weekend holding. 3d/xau-weekly-cycle measured that being forced flat over
    weekends costs ~45% of the edge and took an FTMO pass rate to 43.5%. Without swing this book
    is a different, much worse strategy.

Both are priced below rather than assumed worth it.

THE SELECTION RULE IS NOT THE HEADLINE TARGET. book-shootout established that Sharpe does not
predict pass rate -- the DIAL-SCALED LEFT TAIL against the daily limit does. Pro has the lowest
profit target (6%/6%) and is the cheapest, which makes it look easiest; it also has the TIGHTEST
max loss (6%) and a 3% daily. Those pull in opposite directions and the answer is not guessable.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import v5_account_chooser as ac                                    # noqa: E402
from v5_account_chooser_2026_10 import xau_btc_book                # noqa: E402

ac.NSIM, ac.BLOCK, ac.MAXD, ac.DAY_SAFETY = 1500, 21, 756, 1.5

BASE = dict(funded_balance=100_000, consistency=None, split=1.00)
PLANS = {
    "Standard @5% daily ($652)": dict(
        fee=652.0, funded_fee=0.0, balances=[100_000, 100_000],
        targets=[0.08, 0.05], maxloss=[0.10, 0.10], dayloss=[0.05, 0.05],
        funded_maxloss=0.10, **BASE),
    "Standard @3% daily ($652)": dict(
        fee=652.0, funded_fee=0.0, balances=[100_000, 100_000],
        targets=[0.08, 0.05], maxloss=[0.10, 0.10], dayloss=[0.03, 0.03],
        funded_maxloss=0.10, **BASE),
    "Flex ($555)": dict(
        fee=555.0, funded_fee=0.0, balances=[100_000, 100_000],
        targets=[0.10, 0.08], maxloss=[0.12, 0.12], dayloss=[0.04, 0.04],
        funded_maxloss=0.12, **BASE),
    "Pro ($466)": dict(
        fee=466.0, funded_fee=0.0, balances=[100_000, 100_000],
        targets=[0.06, 0.06], maxloss=[0.06, 0.06], dayloss=[0.03, 0.03],
        funded_maxloss=0.06, **BASE),
}


def main() -> int:
    r = xau_btc_book()
    rv = r.to_numpy()
    sd = rv.std() * np.sqrt(252)
    sr = rv.mean() / rv.std() * np.sqrt(252)
    eq = (1 + r).cumprod()
    mdd = abs((eq / eq.cummax() - 1).min())

    print("=" * 112)
    print("FUNDINGPIPS 2-STEP $100k — simulated on the DEPLOYED XAU+BTC book")
    print("=" * 112)
    print("  book: Sharpe %+.3f   vol %.1f%%   maxDD %.1f%%   maxDD/vol %.2fx   (%d days)"
          % (sr, 100 * sd, 100 * mdd, mdd / sd, len(r)))
    print("  1,500 sims x 3 seeds, block bootstrap, day_safety 1.5 (floating-P&L proxy)\n")
    print("  %-28s %6s %7s %8s %10s %9s %9s" %
          ("plan", "dial", "pass%", "med mo", "fail_day%", "fail_dd%", "timeout%"))
    print("  " + "-" * 92)
    best = {}
    for name, plan in PLANS.items():
        for dial in (0.03, 0.04, 0.05, 0.06, 0.08):
            k = dial / sd
            out = [ac.sim_plan(rv, k, plan, s) for s in range(3)]
            agg = {kk: float(np.mean([o[kk] for o in out])) for kk in out[0]}
            if agg["passpct"] > best.get(name, (0,))[0]:
                best[name] = (agg["passpct"], dial, agg)
            print("  %-28s %6.0f%% %7.1f %8.1f %10.1f %9.1f %9.1f" %
                  (name if dial == 0.03 else "", 100 * dial, agg["passpct"], agg["med_mo"],
                   agg["fail_day"], agg["fail_dd"], agg["timeout"]))
        print()

    print("=" * 112)
    print("BEST DIAL PER PLAN")
    print("=" * 112)
    print("  %-28s %7s %7s %8s %10s %12s" %
          ("plan", "dial", "pass%", "med mo", "fail_day%", "cost"))
    ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
    for name, (p, dial, agg) in ranked:
        fee = PLANS[name]["fee"]
        print("  %-28s %6.0f%% %7.1f %8.1f %10.1f %12s" %
              (name, 100 * dial, p, agg["med_mo"], agg["fail_day"], f"${fee:,.0f}"))

    print("\n" + "=" * 112)
    print("THE TWO PAID OPTIONS, PRICED")
    print("=" * 112)
    top = ranked[0]
    dial = top[1][1]
    print("  SWAP FREE (+$55.50). Financing on this book is -3.79%%/yr (3q), worth -0.29 Sharpe.")
    print("    At a %.0f%% dial the book earns ~%.2f%%/yr gross of financing; carry would take"
          % (100 * dial, 100 * dial * sr))
    print("    roughly %.2f%% of that back, i.e. ~$%.0f/yr on 100k."
          % (3.79 * (dial / 0.10), 1000 * 3.79 * (dial / 0.10)))
    print("    $55.50 once against ~$%.0f/yr recurring -> pays back in ~%.1f WEEKS."
          % (1000 * 3.79 * (dial / 0.10), 52 * 55.5 / max(1000 * 3.79 * (dial / 0.10), 1)))
    print("\n  SWING (+$55.50). Without it the book is flat every weekend. 3d measured a")
    print("    weekend-flat constraint costing ~45%% of the edge and an FTMO pass rate of 43.5%%.")
    print("    This is NOT an optimisation, it is a precondition: the champion is an H4 trend")
    print("    follower whose average hold is days, so forced Friday exits change the strategy.")
    print("\n  Both together: +$111 on a $%s base. Buy both."
          % f"{PLANS[top[0]]['fee']:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
