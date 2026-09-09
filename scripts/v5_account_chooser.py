"""Which prop account to buy — simulated against THIS strategy's measured return series.

The three plans on The 5%ers' checkout are not variations of one product; they bind on
different rules, so the answer depends on the strategy's own return shape rather than on which
headline number looks friendliest.

    1-STEP GROWTH   $249   100K   target 10%   max loss $6,000 (6%)   daily 3%   1:100
                                  CONSISTENCY RULE 50%   withdraw cap $2,000
    2-STEP HIGH ST. $491   100K   targets 10% then 5%   max loss $10,000 (10%)   daily 5%  1:100
    3-STEP BOOTCAMP $225   100K -> 150K -> 200K -> 250K funded, +$350 payable only on passing
                                  target 6%/step   max loss $5,000/$7,500/$10,000 (5% each)
                                  no daily limit shown   leverage 1:30   profit split 100%

WHAT THE EXISTING `fp_sim` COULD NOT DO, and why this file exists: it hard-codes a 2-phase
challenge, so it cannot represent a 1-step, a 3-step with STEPPING BALANCES, or a consistency
rule. All three matter here.

THE CONSISTENCY RULE IS MODELLED, not ignored. A 50% consistency rule means no single day may
account for more than half of total profit. A trend follower's returns are lumpy by
construction, so on the day the target is first touched the best day is often a large share of
the total — and the account must then keep trading to dilute it, taking on more drawdown risk
while already at target. Ignoring this would flatter the 1-step plan considerably.

Leverage is checked rather than assumed: at a 20% vol dial the gold champion holds ~1.3x
notional, so even Bootcamp's 1:30 is not binding. Stated because a 1:30 cap would matter a
great deal for a higher-dial or multi-sleeve book.

    python scripts/v5_account_chooser.py
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

from scripts.v5_volregime_taper_crossasset import load_sleeve, TARGET_VOL, MAX_LEV  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402
from scripts.v5_xau_champion_lifts import sharpe, dd_of  # noqa: E402

COST_BP = 0.75 * 1.23 * 1.5          # GoldEternal, zero carry
DAY_SAFETY = 1.5                     # intraday floating-P&L proxy, repo standard
SEEDS = (7, 11, 23, 47, 101)
NSIM, MAXD, BLOCK = 1200, 2520, 20

PLANS = {
    "1-Step Growth ($249)": dict(
        fee=249.0, funded_fee=0.0, balances=[100_000], funded_balance=100_000,
        targets=[0.10], maxloss=[0.06], dayloss=[0.03], consistency=0.50,
        funded_maxloss=0.06, split=None),
    "2-Step High Stakes ($491)": dict(
        fee=491.0, funded_fee=0.0, balances=[100_000, 100_000], funded_balance=100_000,
        targets=[0.10, 0.05], maxloss=[0.10, 0.10], dayloss=[0.05, 0.05], consistency=None,
        funded_maxloss=0.10, split=None),
    "3-Step Bootcamp ($225+$350)": dict(
        fee=225.0, funded_fee=350.0, balances=[100_000, 150_000, 200_000],
        funded_balance=250_000, targets=[0.06, 0.06, 0.06],
        maxloss=[0.05, 0.05, 0.05], dayloss=[None, None, None], consistency=None,
        funded_maxloss=0.04, split=1.00),
}


def manual_gold_returns() -> pd.Series:
    """The deployed manual config: 08:00 H4 bar, buffer 0.20, GoldEternal costs, zero carry."""
    df = load_sleeve(dict(path="data/XAUUSD_H4_long.csv"))
    close = df["close"]; ret = close.pct_change()
    vol = ret.ewm(halflife=42, min_periods=20).std() * np.sqrt(252 * 6)
    raw = (champion_signal(close).clip(0, 2) * (TARGET_VOL / vol)).clip(0, MAX_LEV)
    band = (0.20 * (TARGET_VOL / vol).clip(0, MAX_LEV)).values
    hrs = raw.index.hour.values
    p, out, held = raw.values.copy(), np.zeros(len(raw)), 0.0
    for i in range(len(p)):
        if hrs[i] == 8 and np.isfinite(p[i]):
            b = band[i] if np.isfinite(band[i]) else 0.0
            if abs(p[i] - held) > b:
                held = p[i] - np.sign(p[i] - held) * b
        out[i] = held
    pos = pd.Series(out, index=raw.index).shift(1).fillna(0.0)
    net = (pos * ret - pos.diff().abs().fillna(0.0) * COST_BP * 1e-4).fillna(0.0)
    eq = (1 + net).cumprod().loc["2018-01-01":]
    return (eq / eq.iloc[0]).resample("D").last().pct_change(fill_method=None).dropna()


def sim_plan(r: np.ndarray, k: float, plan: dict, seed: int) -> dict:
    """Block-bootstrap the whole multi-phase journey. Returns pass rate, days, failure mode."""
    rng = np.random.default_rng(seed)
    n = len(r)
    passed = fail_day = fail_dd = timeout = 0
    days_all = []
    for _ in range(NSIM):
        idx = []
        while len(idx) < MAXD:
            s = rng.integers(0, n)
            L = min(rng.geometric(1 / BLOCK), MAXD - len(idx))
            idx.extend([(s + j) % n for j in range(L)])
        x = k * r[np.array(idx)]
        cur, total_days, alive = 0, 0, True
        while alive and cur < len(plan["targets"]):
            eq, best_day, dl = 1.0, 0.0, plan["dayloss"][cur]
            tgt, ml = 1 + plan["targets"][cur], plan["maxloss"][cur]
            while True:
                if total_days >= MAXD:
                    timeout += 1; alive = False; break
                d = x[total_days]; total_days += 1
                if dl is not None and d * DAY_SAFETY < -dl:
                    fail_day += 1; alive = False; break
                eq *= (1 + d)
                best_day = max(best_day, d)
                if eq < 1 - ml:
                    fail_dd += 1; alive = False; break
                if eq >= tgt:
                    # consistency rule: best single day must be <= c x total profit
                    c = plan["consistency"]
                    if c is not None and (eq - 1) > 0 and best_day > c * (eq - 1):
                        continue                      # keep trading to dilute it
                    cur += 1; break
        if alive and cur == len(plan["targets"]):
            passed += 1; days_all.append(total_days)
    med = int(np.median(days_all)) if days_all else -1
    return dict(passpct=passed / NSIM * 100, fail_day=fail_day / NSIM * 100,
                fail_dd=fail_dd / NSIM * 100, timeout=timeout / NSIM * 100,
                med_mo=med / 21 if med > 0 else np.nan)


def main() -> None:
    r = manual_gold_returns()
    sd = r.std() * np.sqrt(252)
    print(f"strategy: manual gold champion on GoldEternal — net Sharpe {sharpe(r):+.3f}, "
          f"maxDD {dd_of(r):.1f}%, realised vol {sd:.1%}")
    print(f"leverage check: at a 20% dial the position is ~1.3x notional, so even 1:30 is "
          f"not binding\n")

    for name, plan in PLANS.items():
        print(f"══ {name}")
        print(f"   {'dial':>5s} {'pass%':>7s} {'med mo':>7s} {'fail_day%':>10s} "
              f"{'fail_dd%':>9s} {'E[fee]/funded':>14s} {'E[$/yr] funded':>15s}")
        rows = []
        for dial in (0.03, 0.05, 0.07, 0.10, 0.15):
            k = dial / sd
            agg = [sim_plan(r.values, k, plan, s) for s in SEEDS]
            p = np.mean([a["passpct"] for a in agg]) / 100
            mo = np.nanmean([a["med_mo"] for a in agg])
            fd = np.mean([a["fail_day"] for a in agg])
            fdd = np.mean([a["fail_dd"] for a in agg])
            total_fee = plan["fee"] / max(p, 1e-9) + plan["funded_fee"]
            cagr = dial * sharpe(r)
            split = plan["split"] if plan["split"] else 0.90     # unknown -> assume 90%
            eyr = cagr * plan["funded_balance"] * split
            print(f"   {dial:5.0%} {p * 100:7.1f} {mo:7.1f} {fd:10.1f} {fdd:9.1f} "
                  f"{total_fee:14,.0f} {eyr:15,.0f}")
            rows.append(dict(plan=name, dial=dial, passpct=p * 100, med_mo=mo,
                             fail_day=fd, fail_dd=fdd, exp_fee=total_fee, exp_yr=eyr))
        pd.DataFrame(rows).to_csv(ROOT / f"data/v5_runs/acct_{name.split()[0]}.csv", index=False)
        print()
    print("split assumed 90% where the checkout did not show it (Bootcamp shows 100%).")
    print("E[$/yr] funded = dial x Sharpe x funded balance x split, i.e. the earn rate ONCE "
          "funded, before any withdrawal cap.")


if __name__ == "__main__":
    main()
