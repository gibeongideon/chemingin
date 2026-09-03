"""BOOK SHOOTOUT — FundingPips (XAUmicro+ETH+DJI30) vs FTMO (XAU+BTC+NDX+BRENT).

Both books already have documented numbers, but from DIFFERENT harnesses, vol dials and
windows, so they are not comparable as recorded. This runs both through ONE identical
harness: same champion recipe, same engine, same period, equal weights, per-instrument
one-way bp costs built from live-verified spreads.

COSTS (one-way bp = 0.75 x spread, i.e. half-spread + slippage at 50% of half-spread;
anchored by reproducing XAU's 0.75bp at the $0.448/$4400 the dollar engine charges):
  FP sleeves   XAUmicro 0.3bp spread -> 0.225   ETH 2.7bp -> 2.03   DJI30 0.2bp -> 0.15
  FTMO sleeves XAU 1.0bp -> 0.75  BTC 1.5bp -> 1.13  NDX 0.5bp -> 0.38  BRENT 7.58bp -> 5.69
  (FP spreads from configs/v5_fp_flex_10k.json _specs; FTMO from configs/v5_ftmo_challenge.json
   _broker, BRENT verified live 2026-07-24)

Champion speeds are rescaled by 1/6 on D1 sleeves — `champion_signal` hardcodes D=6 (H4
bars/day) and would otherwise become a 96-1536 DAY trend follower on daily bars. XAU stays
H4 as deployed.

RULE SETS ARE NOT THE SAME EITHER, so each book is scored under BOTH firms' rules to
separate "is the book better" from "are the rules easier":
  FundingPips Flex : P1 +10% / P2 +6%, daily 4%, overall 12% STATIC, dial 7%
  FTMO 2-Step      : P1 +10% / P2 +5%, daily 5%, overall 10% STATIC, dial 9%

    python scripts/v5_book_shootout.py
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
sys.path.insert(0, str(ROOT / "data" / "v5_runs" / "challenge-lab"))

from scripts.v5_volregime_taper_crossasset import engine_bp, load_sleeve  # noqa: E402
from scripts.v5_xau_champion_lifts import champion_recipe, sharpe, dd_of  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402
from scripts.v5_xau_turn_prob import paired  # noqa: E402
from challenge_lab import fp_sim  # noqa: E402

EVAL_START = "2018-01-01"     # ETH history starts 2017-11, so this is the common window
SPLITS = [("2018-01-01", "2021-12-31", "2018-21"), ("2022-01-01", "2026-12-31", "2022-26")]

SLEEVES = {
    # label: (path, ann, one-way bp, champion speed scale, vol halflife)
    "XAU":       ("data/XAUUSD_H4_long.csv", 252 * 6, 0.75, 1.0, 42),
    "XAUmicro":  ("data/XAUUSD_H4_long.csv", 252 * 6, 0.225, 1.0, 42),
    "ETH":       ("data/ETH_D1_long.csv", 252, 2.03, 1 / 6, 7),
    "DJI30":     ("data/DJI_D1_long.csv", 252, 0.15, 1 / 6, 7),
    "BTC":       ("data/BTC_D1_long.csv", 252, 1.13, 1 / 6, 7),
    "NDX":       ("data/NDX_D1_long.csv", 252, 0.38, 1 / 6, 7),
    "BRENT":     ("data/BRENT_D1_long.csv", 252, 5.69, 1 / 6, 7),
}

BOOKS = {
    "FundingPips 10K (XAUmicro+ETH+DJI30)": ["XAUmicro", "ETH", "DJI30"],
    "FTMO 100K (XAU+BTC+NDX+BRENT)":        ["XAU", "BTC", "NDX", "BRENT"],
}

RULES = {
    "FundingPips Flex": dict(p1=0.10, p2=0.06, dayloss=0.04, maxloss=0.12, dial=0.07),
    "FTMO 2-Step":      dict(p1=0.10, p2=0.05, dayloss=0.05, maxloss=0.10, dial=0.09),
}


def sleeve_returns() -> dict[str, pd.Series]:
    out = {}
    for name, (path, ann, bp, scale, hl) in SLEEVES.items():
        df = load_sleeve(dict(path=path))
        if df is None:
            print(f"  !! {name}: data missing")
            continue
        fc = (champion_signal(df["close"]) if scale == 1.0
              else champion_recipe(df["close"], scale, 1.5))
        r = engine_bp(df, fc, bp, ann, hl)
        out[name] = r.loc[EVAL_START:]
    return out


def main() -> None:
    R = sleeve_returns()
    print(f"eval from {EVAL_START}, per-instrument bp costs, equal weights\n")

    print("=== PER-SLEEVE (standalone, same harness) ===")
    print(f"{'sleeve':10} {'cost bp':>8} {'n days':>7} {'Sharpe':>8} {'maxDD':>8} {'CAGR%':>7}")
    for name, r in R.items():
        yrs = max((r.index[-1] - r.index[0]).days / 365.25, 0.1)
        cagr = ((1 + r).prod() ** (1 / yrs) - 1) * 100
        print(f"{name:10} {SLEEVES[name][2]:>8.2f} {len(r):>7} {sharpe(r):>+8.3f} "
              f"{dd_of(r):>+8.1f} {cagr:>+7.2f}")

    books = {}
    print(f"\n{'='*96}\n=== BOOK LEVEL ===\n{'='*96}")
    for bname, legs in BOOKS.items():
        legs = [x for x in legs if x in R]
        common = None
        for x in legs:
            common = R[x].index if common is None else common.intersection(R[x].index)
        b = pd.concat([R[x].reindex(common).fillna(0.0) for x in legs], axis=1).mean(axis=1)
        books[bname] = b
        yrs = max((b.index[-1] - b.index[0]).days / 365.25, 0.1)
        cagr = ((1 + b).prod() ** (1 / yrs) - 1) * 100
        vol = b.std() * np.sqrt(252) * 100
        print(f"\n{bname}")
        print(f"  {len(legs)} sleeves, {len(common)} common days "
              f"({common[0].date()}..{common[-1].date()})")
        print(f"  Sharpe {sharpe(b):+.3f}   maxDD {dd_of(b):+.1f}%   CAGR {cagr:+.2f}%   "
              f"realised vol {vol:.1f}%   Calmar {cagr/abs(dd_of(b)):.3f}")
        # correlation matrix = diversification quality
        M = pd.concat([R[x].reindex(common).fillna(0.0).rename(x) for x in legs], axis=1)
        cm = M.corr()
        pairs = [(a, c, cm.loc[a, c]) for i, a in enumerate(legs) for c in legs[i + 1:]]
        print("  pairwise correlations: " +
              ", ".join(f"{a}/{c} {v:+.2f}" for a, c, v in pairs))
        print(f"  mean |corr| {np.mean([abs(v) for _,_,v in pairs]):.3f}")
        for s0, s1, lbl in SPLITS:
            seg = b.loc[s0:s1]
            if len(seg) > 60 and seg.std() > 0:
                print(f"    {lbl}: Sharpe {sharpe(seg):+.3f}  maxDD {dd_of(seg):+.1f}%")

    print(f"\n{'='*96}\n=== HEAD TO HEAD ===\n{'='*96}")
    a, bb = list(books)
    j = books[a].index.intersection(books[bb].index)
    x, y = books[a].reindex(j), books[bb].reindex(j)
    print(f"common overlap: {len(j)} days ({j[0].date()}..{j[-1].date()})")
    print(f"  {a}: Sharpe {sharpe(x):+.3f}  maxDD {dd_of(x):+.1f}%")
    print(f"  {bb}: Sharpe {sharpe(y):+.3f}  maxDD {dd_of(y):+.1f}%")
    print(f"  correlation between the two books: {x.corr(y):+.3f}")
    # matched-vol paired test: at equal risk, which earns more?
    xm = x * (y.std() / x.std())
    _, t, _ = paired(xm, y)
    print(f"  paired t at matched vol ({a.split()[0]} vs {bb.split()[0]}): {t:+.2f}"
          f"   (>0 favours {a.split()[0]})")
    combo = (x + y) / 2
    print(f"  50/50 COMBINATION of both books: Sharpe {sharpe(combo):+.3f}  "
          f"maxDD {dd_of(combo):+.1f}%")

    print(f"\n{'='*96}\n=== PASS SIMULATION under BOTH rule sets (block-bootstrap fp_sim) ===\n{'='*96}")
    print("each book normalised to the firm's own vol dial before its rules are applied\n")
    print(f"{'book':40} {'rules':18} {'pass%':>7} {'fail_day%':>10} {'fail_dd%':>9} {'med mo':>7}")
    for bname, b in books.items():
        for rname, rc in RULES.items():
            realised = b.std() * np.sqrt(252)
            k = rc["dial"] / realised if realised > 0 else 1.0
            s = fp_sim(b.values, k, p1=rc["p1"], p2=rc["p2"],
                       dayloss=rc["dayloss"], maxloss=rc["maxloss"], day_safety=1.5)
            print(f"{bname:40} {rname:18} {s['passpct']:>7.1f} {s['fail_day']:>10.1f} "
                  f"{s['fail_dd']:>9.1f} {s['med_mo']:>7.1f}")

    print(f"\n{'='*96}\n=== PRACTICAL CONSTRAINT (documented, not modelled above) ===\n{'='*96}")
    print("configs/v5_fp_flex_10k.json: 'The 100K book (XAU+BTC+NDX) CANNOT be sized here —")
    print("  on FundingPips specs BTC ($642) and NDX100 ($5,746) min-lots round target lots")
    print("  to ZERO.'  Verified FP sizing at $10,000: XAUmicro 0.01 lot (1.18x target),")
    print("  ETH 0.13 lot (0.98x), DJI30 0.01 lot (0.89x) -> gross 0.92x target.")
    print("So the books are NOT interchangeable: the FP book exists because the FTMO book")
    print("is un-sizeable on a 10K account. Any 'which is better' answer is account-size")
    print("conditional.")


if __name__ == "__main__":
    main()
