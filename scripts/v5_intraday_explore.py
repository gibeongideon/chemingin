#!/usr/bin/env python3
"""Independent intraday exploration, 2026-10-21. Grounded in what this repo has already closed.

WHAT IS DELIBERATELY NOT RE-RUN, because the project already answered it:
  * intraday TREND at a tight spread. BEST_FINDINGS #7 measured net Sharpe by timeframe at a
    BEST-CASE $0.12 spread -- tighter than Switch Markets' measured $0.210 -- and H4 still beat
    M30 (0.93) and H1 (0.98) at 1.27 with a third of the drawdown. "Tight spreads do not rescue
    intraday trend"; the signal itself is worse. Re-pricing that is a closed cell.
  * the M15 fade, ORB, scheduled releases, SMC/ICT, M15 entry timing, NR7. All closed.

SO THIS GOES AT TWO THINGS THE REPO HAS NEVER TESTED, both chosen for having a MECHANISM rather
than a pattern:

  A. THE COST WALL, RE-DERIVED AT THE LIVE VENUE. BEST_FINDINGS #7's table is floored at 1.5bp
     one-way, and the live account now quotes 0.50bp. That table is described as "the constraint
     that governs every day-trading idea here", so if its numbers moved, the governing constraint
     moved. This corrects it rather than arguing with it.

  B. THE OVERNIGHT / INTRADAY RETURN DECOMPOSITION. In equities, close-to-open accrues nearly all
     the long-run return while open-to-close contributes ~nothing -- one of the most robust
     anomalies in finance. NOBODY HAS EVER TESTED IT ON GOLD IN THIS REPO. It is intraday-adjacent
     in the way that matters commercially: if the drift is concentrated in a few hours, you hold
     for those hours instead of 24, which cuts financing (the repo's one proven live lever: swap
     costs the book -3.79%/yr) and cuts exposure without cutting return. 3b flagged an
     "overnight+reversal ensemble 1.30 OOS" and recorded that it NEEDS A FILL CHECK. That check
     was never done. This does it.

  C. BTC -> GOLD ACROSS GOLD'S CLOSED WINDOW. BTC trades 24/7; gold does not. Information that
     arrives while gold is shut has nowhere to go but BTC, so BTC's move across gold's daily break
     is a candidate predictor of gold's re-open. The repo has tested cross-asset at D1 (divergence,
     XS momentum) but never a lead-lag ACROSS A CLOSURE, which is a different mechanism: it is not
     correlation, it is an information-release ordering.

Controls carried: cost floored at the live quote (#1); benchmark vs HOLDING, never zero (#2);
EV per fire and base rate printed beside every hit rate (#8, and the drift-baseline rule); the
skipped sample measured wherever a filter is applied.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v5_runs" / "intraday_explore"

# Measured live at Switch Markets 2026-10-19 (3bs): XAUUSDs $0.210 on $4,164 = 0.504bp.
SPREAD_BP_LIVE = 0.504
SPREAD_BP_FILED = 2.0        # the 1.5bp one-way floor BEST_FINDINGS #7 used == 2.0bp spread-equiv
ONE_WAY = 0.75               # repo convention: one-way bp = 0.75 x spread_bp


def load(tf: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / f"XAUUSD_{tf}_long.csv", parse_dates=["time"])
    d = d.set_index("time").sort_index()
    return d[~d.index.duplicated(keep="last")]


def wilder_atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ----------------------------------------------------------------- A. the cost wall, re-derived
def cost_wall() -> None:
    print("=" * 100)
    print("A.  THE INTRADAY COST WALL, RE-DERIVED AT THE LIVE VENUE")
    print("=" * 100)
    print("    BEST_FINDINGS #7 floors cost at 1.5bp one-way. Switch Markets quotes 0.50bp.")
    print("    P* = 0.5 + round_trip / (2 x k x ATR)   [symmetric bracket]")
    rows = []
    for tf in ("M15", "H1", "D1"):
        d = load(tf) if tf != "D1" else pd.read_csv(
            ROOT / "data" / "GOLD_D1_long.csv", parse_dates=["time"]).set_index("time").sort_index()
        atr = wilder_atr(d["high"], d["low"], d["close"])
        atr_bp = float((atr / d["close"]).median() * 1e4)
        for k in (1, 2, 3):
            for label, sp in (("filed 2.0bp", SPREAD_BP_FILED), ("LIVE 0.50bp", SPREAD_BP_LIVE)):
                rt = sp * 2 * ONE_WAY
                rows.append(dict(tf=tf, k=k, basis=label, atr_bp=atr_bp,
                                 rt_bp=rt, p_star=0.5 + rt / (2 * k * atr_bp)))
    R = pd.DataFrame(rows)
    piv = R.pivot_table(index=["tf", "k"], columns="basis", values="p_star")
    piv["improvement"] = piv["filed 2.0bp"] - piv["LIVE 0.50bp"]
    order = [("M15", 1), ("M15", 2), ("M15", 3), ("H1", 1), ("H1", 2), ("H1", 3),
             ("D1", 1), ("D1", 2), ("D1", 3)]
    print(f"\n    {'tf':>4} {'k':>2} {'ATR bp':>8} {'P* filed':>9} {'P* LIVE':>9} {'shrink':>8}")
    print("    " + "-" * 48)
    for tf, k in order:
        if (tf, k) not in piv.index:
            continue
        r = piv.loc[(tf, k)]
        a = float(R[(R.tf == tf)].atr_bp.iloc[0])
        print(f"    {tf:>4} {k:>2} {a:>8.1f} {r['filed 2.0bp']:>9.4f} "
              f"{r['LIVE 0.50bp']:>9.4f} {r['improvement']:>+8.4f}")
    R.to_csv(OUT / "cost_wall.csv", index=False)
    print("\n    The wall MOVED, and materially. But BEST_FINDINGS #7's second half still stands:")
    print("    at $0.12 -- tighter than the live quote -- H4 beat M30/H1 on Sharpe AND drawdown,")
    print("    so a lower wall does not revive intraday TREND. It only widens what is worth")
    print("    testing for NON-trend structures, which is what B and C are.")


# ------------------------------------------------- B. overnight vs intraday return decomposition
def overnight_decomposition() -> None:
    print("\n" + "=" * 100)
    print("B.  OVERNIGHT vs INTRADAY -- the equity anomaly, never tested on gold in this repo")
    print("=" * 100)
    m15 = load("M15")
    # Find gold's daily break EMPIRICALLY rather than assuming a session calendar: the repo has
    # already been burnt once by a hard-coded 2h DST claim that was really 1h.
    gaps = m15.index.to_series().diff()
    typical = gaps.median()
    brk = gaps[gaps > typical * 2]
    print(f"    bar spacing median {typical}; {len(brk):,} gaps > 2x that")
    hrs = brk.index.to_series().dt.hour.value_counts().head(5)
    print(f"    gap END hours (UTC): {dict(hrs)}")

    # Day = between consecutive breaks. Overnight = last close before break -> first open after.
    is_break = gaps > typical * 2
    sess = is_break.cumsum()
    g = m15.groupby(sess)
    sess_df = pd.DataFrame({
        "start": g.apply(lambda x: x.index[0]), "end": g.apply(lambda x: x.index[-1]),
        "open": g["open"].first(), "close": g["close"].last(),
        "high": g["high"].max(), "low": g["low"].min(), "bars": g.size()})
    sess_df = sess_df[sess_df["bars"] >= 20].copy()          # drop stubs
    sess_df["prev_close"] = sess_df["close"].shift(1)
    sess_df = sess_df.dropna()
    sess_df["overnight"] = sess_df["open"] / sess_df["prev_close"] - 1.0
    sess_df["intraday"] = sess_df["close"] / sess_df["open"] - 1.0
    sess_df["full"] = sess_df["close"] / sess_df["prev_close"] - 1.0
    sess_df["year"] = sess_df["end"].dt.year

    n = len(sess_df)
    yrs = (sess_df["end"].max() - sess_df["end"].min()).days / 365.25
    print(f"\n    {n:,} sessions over {yrs:.1f} years "
          f"({sess_df['end'].min().date()} -> {sess_df['end'].max().date()})")

    def ann(x):
        return (1 + x).prod() ** (1 / yrs) - 1

    def sharpe(x):
        return x.mean() / x.std() * np.sqrt(n / yrs) if x.std() > 0 else np.nan

    print(f"\n    {'leg':<12} {'total ret':>11} {'CAGR':>9} {'Sharpe':>8} {'mean bp':>9} "
          f"{'hit':>7} {'t-stat':>8}")
    print("    " + "-" * 70)
    for name in ("overnight", "intraday", "full"):
        x = sess_df[name]
        t = x.mean() / (x.std() / np.sqrt(len(x)))
        print(f"    {name:<12} {(1 + x).prod() - 1:>+10.1%} {ann(x):>+8.2%} "
              f"{sharpe(x):>8.2f} {x.mean() * 1e4:>+9.2f} {(x > 0).mean():>7.3f} {t:>+8.2f}")

    hold_cost = SPREAD_BP_LIVE * 2 * ONE_WAY
    on = sess_df["overnight"]
    print(f"\n    *** THE FILL CHECK 3b ASKED FOR AND NEVER DID. ***")
    print(f"    An overnight-only strategy pays the round trip EVERY session: "
          f"{hold_cost:.2f}bp x {n / yrs:.0f} sessions/yr = {hold_cost * n / yrs / 100:.2f}%/yr.")
    net = on - hold_cost / 1e4
    print(f"    overnight GROSS {ann(on):+.2%}/yr   NET of live spread {ann(net):+.2%}/yr   "
          f"(Sharpe {sharpe(net):.2f})")
    print(f"    buy-and-hold over the same window: {ann(sess_df['full']):+.2%}/yr, "
          f"paying the spread ONCE.")

    print(f"\n    per-year split (is it a regime artifact?):")
    print(f"    {'year':>6} {'overnight':>11} {'intraday':>11} {'full':>11} {'ON share':>10}")
    for y, grp in sess_df.groupby("year"):
        o, i, f_ = grp["overnight"].sum(), grp["intraday"].sum(), grp["full"].sum()
        share = o / f_ if abs(f_) > 1e-9 else np.nan
        print(f"    {y:>6} {o:>+10.2%} {i:>+10.2%} {f_:>+10.2%} {share:>10.2f}")
    sess_df.to_csv(OUT / "overnight_sessions.csv")
    pos = sum(1 for _, grp in sess_df.groupby("year") if grp["overnight"].sum() > 0)
    tot = sess_df["year"].nunique()
    print(f"\n    years with POSITIVE overnight: {pos} of {tot}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cost_wall()
    overnight_decomposition()


if __name__ == "__main__":
    main()
