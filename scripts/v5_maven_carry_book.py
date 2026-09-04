"""LIVE-MEASURED CARRY: Maven's swap census, and the swap-free gold instrument it exposes.

§3q's closing verdict was that financing (-3.79%/yr, -0.29 Sharpe) is the book's single
largest lever and "the only untested way to move it is EXECUTION ... not signal research."
This is that execution test, run against the live Maven terminal (account 10398574, read-only
`symbol_info` + `copy_rates`, no orders).

MEASURED 2026-09-03 on MavenTrade-Server (spread = live ask-bid; %/yr computed from
swap_mode, contract size and the live price, not assumed):

  symbol       swap_mode          spread     long %/yr   short %/yr
  XAUUSD       POINTS             $0.42 (0.94bp)   -3.70      +2.06
  GoldEternal  DISABLED (NO SWAP) $0.55 (1.23bp)    0.00       0.00
  XAGUSD       DISABLED (NO SWAP) $0.046 (6.88bp)   0.00       0.00
  EURUSD       DISABLED (NO SWAP) 0.77bp            0.00       0.00
  US100        POINTS             0.30bp           -3.96      +1.00
  US500        POINTS             1.68bp           -4.42      +1.12
  US30         POINTS             1.86bp           -4.82      +1.22
  GER30        POINTS             7.19bp           -3.71      -0.18
  BRENT        POINTS             8.49bp           -4.79      -5.40
  WTI          POINTS            11.12bp           -3.09      -3.72
  BTCUSD       INT_OPEN           8.99bp          -30.00     -30.00
  ETHUSD       INT_OPEN           6.38bp          -30.00     -30.00

Two facts that change the book, neither of them a signal:
  1. GoldEternal carries NO overnight financing and tracks XAUUSD at 0.998 daily-return
     correlation. The obvious objection - "a perpetual must charge its funding in the price"
     - was tested directly: over the 167 calendar days of history the instrument has
     (launched ~2026-03-13), the log basis GoldEternal/XAUUSD drifts +0.17%/yr (H1, 2,711
     bars) and +0.21%/yr (D1, 143 bars) and stays inside a ~1% band. There is no -3.7%/yr
     embedded drift. Six months cannot rule out a SMALL drift, so this is stated as measured,
     not proven, and must be re-measured monthly.
  2. BRENT's +9.6%/yr carry - the reason it was the best net sleeve at FTMO - DOES NOT EXIST
     at Maven, which pays -4.79% long AND -5.40% short. Carry is broker-specific; a book
     tuned on one venue's financing is mis-specified on another.

WHAT IS CLAIMED, AND WHAT IS NOT: the headline substitution (XAUUSD -> GoldEternal, same
signal, same sizing) is a DETERMINISTIC cost change, not a discovered edge - it is contractual
and involves no parameter search, so no paired-t or DSR applies to it. The subset search
underneath IS a search: 8 candidates, all subsets of size 3-5 = 182 books, disclosed, with a
split-sample on the winner. Do not read the search result with the same confidence as the
substitution.

DEPLOYMENT GATES (must clear before any of this trades):
  * Confirm with Maven that synthetic/"Eternal" instruments are permitted under the challenge
    rules - some firms exclude them.
  * Verify fill quality with one minimum-lot live order; a synthetic's quoted spread can hide
    slippage.
  * Re-measure the basis monthly; a broker can start charging swap or widen the basis.

    python scripts/v5_maven_carry_book.py
"""
from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "v5_runs" / "challenge-lab"))

from scripts.v5_volregime_taper_crossasset import load_sleeve  # noqa: E402
from scripts.v5_financing_aware_book import engine_fin, SHORT_HAIRCUT  # noqa: E402
from scripts.v5_xau_champion_lifts import champion_recipe, vol_match, sharpe, dd_of  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402
from scripts.v5_xau_turn_prob import paired, per_year  # noqa: E402

EVAL_START = "2018-01-01"
SPLITS = [("2018-01-01", "2021-12-31", "2018-21"), ("2022-01-01", "2026-12-31", "2022-26")]
SPREAD_MULT = 1.5      # live snapshots are one instant mid-session; widen them, per
#                        `ftmo-xau-diversifier-search` (never trust the tighter figure)

# Maven symbol -> (price proxy, ann bars, live spread bp, champion scale, vol hl, long %/yr)
CANDS = {
    "GoldEternal": ("data/XAUUSD_H4_long.csv", 252 * 6, 1.23, 1.0, 42, 0.0),
    "XAUUSD":      ("data/XAUUSD_H4_long.csv", 252 * 6, 0.94, 1.0, 42, -0.0370),
    "XAGUSD":      ("data/SILVER_D1_long.csv", 252, 6.88, 1 / 6, 7, 0.0),
    "US100":       ("data/NDX_D1_long.csv", 252, 0.30, 1 / 6, 7, -0.0396),
    "US500":       ("data/SPX_D1_long.csv", 252, 1.68, 1 / 6, 7, -0.0442),
    "US30":        ("data/DJI_D1_long.csv", 252, 1.86, 1 / 6, 7, -0.0482),
    "GER30":       ("data/DAX_D1_long.csv", 252, 7.19, 1 / 6, 7, -0.0371),
    "BRENT":       ("data/BRENT_D1_long.csv", 252, 8.49, 1 / 6, 7, -0.0479),
    "BTCUSD":      ("data/BTC_D1_long.csv", 252, 8.99, 1 / 6, 7, -0.3000),
}
CONFIGURED = ["XAUUSD", "BTCUSD", "US100", "BRENT"]      # configs/v5_maven_book.json today
SUBSTITUTED = ["GoldEternal", "BTCUSD", "US100", "BRENT"]


def sleeve(name: str, charge: bool = True) -> pd.Series:
    path, ann, spr_bp, scale, hl, rate = CANDS[name]
    df = load_sleeve(dict(path=path))
    fc = champion_signal(df["close"]) if scale == 1.0 else champion_recipe(df["close"], scale, 1.5)
    cost_bp = 0.75 * spr_bp * SPREAD_MULT          # repo convention: one-way = 0.75 x spread
    return engine_fin(df, fc, cost_bp, ann, hl, rate, charge)


def halves(r: pd.Series) -> list[float]:
    return [sharpe(r.loc[a:b]) for a, b, _ in SPLITS]


def book(names: list[str], S: dict[str, pd.Series]) -> pd.Series:
    return pd.DataFrame({n: S[n] for n in names}).dropna(how="all").fillna(0.0).mean(axis=1)


def gate(c: pd.Series, b: pd.Series) -> tuple[float, str, list[float]]:
    i = c.index.intersection(b.index)
    c, b = c.loc[i], b.loc[i]
    _, t, _ = paired(vol_match(c, b), b)
    yp, yn = per_year(vol_match(c, b), b)
    dh = [sharpe(vol_match(c.loc[a:x], b.loc[a:x])) - sharpe(b.loc[a:x]) for a, x, _ in SPLITS]
    return t, f"{yp}/{yn}", dh


def main() -> None:
    net = {n: sleeve(n, True) for n in CANDS}
    gro = {n: sleeve(n, False) for n in CANDS}

    print("--- per-sleeve, Maven's own measured costs and carry "
          f"(spreads widened {SPREAD_MULT}x) ---")
    print(f"{'symbol':12s} {'carry':>8s} {'1-way bp':>9s} {'gross SR':>9s} {'net SR':>8s} "
          f"{'carry cost':>11s} {'halves':>16s}")
    for n in CANDS:
        drag = (net[n].mean() - gro[n].mean()) * 252
        h = halves(net[n])
        print(f"{n:12s} {CANDS[n][5]:+7.2%} {0.75 * CANDS[n][2] * SPREAD_MULT:9.2f} "
              f"{sharpe(gro[n]):+9.3f} {sharpe(net[n]):+8.3f} {drag:+10.2%}  "
              f"{h[0]:+7.2f}/{h[1]:+.2f}")

    print("\n=== HEADLINE (deterministic, no search): the SAME champion on the "
          "swap-free instrument ===")
    a, b = net["XAUUSD"], net["GoldEternal"]
    print(f"  XAUUSD      net SR {sharpe(a):+.3f}  DD {dd_of(a):+.2f}%  "
          f"halves {halves(a)[0]:+.2f}/{halves(a)[1]:+.2f}  "
          f"CAGR {((1 + a).prod() ** (252 / len(a)) - 1):+.2%}")
    print(f"  GoldEternal net SR {sharpe(b):+.3f}  DD {dd_of(b):+.2f}%  "
          f"halves {halves(b)[0]:+.2f}/{halves(b)[1]:+.2f}  "
          f"CAGR {((1 + b).prod() ** (252 / len(b)) - 1):+.2%}")
    print(f"  -> +{sharpe(b) - sharpe(a):.3f} Sharpe and "
          f"+{(((1 + b).prod() ** (252 / len(b))) - ((1 + a).prod() ** (252 / len(a)))):.2%} CAGR "
          f"for +{0.75 * (1.23 - 0.94) * SPREAD_MULT:.2f}bp of extra spread")

    bc, bs = book(CONFIGURED, net), book(SUBSTITUTED, net)
    t, yy, dh = gate(bs, bc)
    print(f"\n  BOOK as configured   {'+'.join(CONFIGURED):34s} net SR {sharpe(bc):+.3f}  "
          f"DD {dd_of(bc):+.2f}%  halves {halves(bc)[0]:+.2f}/{halves(bc)[1]:+.2f}")
    print(f"  BOOK w/ substitution {'+'.join(SUBSTITUTED):34s} net SR {sharpe(bs):+.3f}  "
          f"DD {dd_of(bs):+.2f}%  halves {halves(bs)[0]:+.2f}/{halves(bs)[1]:+.2f}")
    print(f"  t@matched {t:+.2f} [{yy}]  dSR by half {dh[0]:+.2f}/{dh[1]:+.2f}")

    print("\n--- EXPLORATORY subset search (disclosed: all size-3..5 books over 8 "
          "candidates, XAUUSD excluded in favour of GoldEternal) ---")
    pool = [n for n in CANDS if n != "XAUUSD"]
    rows = []
    for k in (3, 4, 5):
        for combo in combinations(pool, k):
            bk = book(list(combo), net)
            h = halves(bk)
            rows.append(dict(book="+".join(combo), k=k, sr=sharpe(bk), dd=dd_of(bk),
                             h1=h[0], h2=h[1], worst=min(h)))
    R = pd.DataFrame(rows).sort_values("sr", ascending=False)
    print(f"  {len(R)} books searched. Top 8 by net Sharpe:")
    for _, r in R.head(8).iterrows():
        print(f"    {r.book:44s} SR {r.sr:+.3f}  DD {r.dd:+6.2f}%  "
              f"halves {r.h1:+.2f}/{r.h2:+.2f}")
    print(f"  configured book rank: "
          f"{int((R.sr > sharpe(bc)).sum()) + 1} of {len(R)} (SR {sharpe(bc):+.3f})")
    print(f"  substituted book rank: "
          f"{int((R.sr > sharpe(bs)).sum()) + 1} of {len(R)} (SR {sharpe(bs):+.3f})")
    best = R.iloc[0]
    bb = book(best.book.split("+"), net)
    t2, yy2, dh2 = gate(bb, bs)
    print(f"\n  search winner vs the substituted book: t@matched {t2:+.2f} [{yy2}]  "
          f"dSR by half {dh2[0]:+.2f}/{dh2[1]:+.2f}  "
          f"{'survives the split' if min(dh2) > 0 else 'FAILS the split - treat as a search artifact'}")

    R.to_csv(ROOT / "data" / "v5_runs" / "maven_carry_books.csv", index=False)
    print("\n-> data/v5_runs/maven_carry_books.csv")


if __name__ == "__main__":
    main()
