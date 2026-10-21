#!/usr/bin/env python3
"""Book WIDTH re-derived on the REGIME-U criterion, with carry charged.

Pre-registered in data/v5_runs/regime_u_width/PREREGISTRATION.md (committed before the first fit).

THE QUESTION. The live Switch book (magic 360595) is 2 sleeves. 3br chose 2 sleeves using
`pass` / `fail_day` from an FTMO PHASE-1 SIMULATOR, and its stopping rule is explicitly a
daily-breach rule: "width helps only while it is still buying a reduction in the daily-breach rate
-- and that reaches ZERO at TWO sleeves". Switch Markets has NO DAILY LIMIT.
RESEARCH-REGIMES.md:124-126 files that exact criterion as "meaningless in regime U", whose rule is
"net Sharpe, full stop" (L187). 3br's own table, spread-only, puts XAU+BTC+NDX at +1.696 / -9.5%
against XAU+BTC's +1.511 / -11.2%. So the regime-U criterion selects the THREE-sleeve book -- but
3bs proved carry is worth 0.22-0.34 Sharpe and those rows charged SPREAD ONLY.

WHAT IS NOT CLAIMED. The wide 7-9 sleeve books stay rejected: 3ah measured their added sleeves as
net-negative after carry (GER30 net Sharpe -0.003), a finding written AFTER prop limits were
dropped from scope, so it holds in regime U. This tests 2 vs 3 vs 4 ONLY. The Sharpe-1.60 drift
portfolio is retracted to 1.561 by its own causality audit (full-sample vol scaling) and is
excluded.

BRENT is in because its MEASURED Switch carry is -0.09%/yr -- effectively free to hold. 3ae valued
BRENT as tail insurance (fail_day 13.9% -> 0%), a regime-P reason that is moot here, so it must
earn its place on net Sharpe alone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from v5_xau_champion_lifts import champion_recipe          # noqa: E402
from src.v5 import mtf_frames                              # noqa: E402

VOL_HL, MAX_LEV = 42, 8.0
OUT = ROOT / "data" / "v5_runs" / "regime_u_width"

# Carry %/yr on a LONG, MEASURED LIVE at Switch 2026-10-21 via symbol_info (mode-aware).
# Spreads in bp: XAU/BTC from 3bs's tick sample; NDX/BRENT filled from the symbol census.
SLEEVES = {
    "XAU":   dict(src="H4",  scale=1.0, spread_bp=0.50, carry=-5.76, sym="XAUUSDs"),
    "BTC":   dict(src="BTC", scale=1/6, spread_bp=4.46, carry=-15.00, sym="BTCUSDs"),
    "NDX":   dict(src="NDX", scale=1/6, spread_bp=None, carry=-5.01, sym="USTECs"),
    "BRENT": dict(src="BRENT", scale=1/6, spread_bp=None, carry=-0.09, sym="BRENTs"),
}
BOOKS = {
    "XAU alone":            ["XAU"],
    "XAU+BTC  [INCUMBENT]": ["XAU", "BTC"],
    "XAU+BTC+NDX":          ["XAU", "BTC", "NDX"],
    "XAU+BTC+BRENT":        ["XAU", "BTC", "BRENT"],
    "XAU+BTC+NDX+BRENT":    ["XAU", "BTC", "NDX", "BRENT"],
}
DIALS = (0.06, 0.09, 0.12, 0.15, 0.20)


def load_census_spreads() -> dict:
    """Fill NDX/BRENT spreads from the live census rather than guessing."""
    f = OUT.parent / "symbol_census" / "switch_census.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f)
    return {r["name"]: float(r["spread_bp"])
            for _, r in d.iterrows() if pd.notna(r.get("spread_bp"))}


def close_series(src: str) -> tuple[pd.Series, float]:
    if src == "H4":
        m15 = pd.read_csv(ROOT / "data/XAUUSD_M15_long.csv", parse_dates=["time"], index_col="time")
        m15 = m15[~m15.index.duplicated(keep="last")].sort_index()
        h4 = mtf_frames.aggregate(m15, "H4").dropna(subset=["close"])
        c = h4["close"].astype(float)
    else:
        d = pd.read_csv(ROOT / f"data/{src}_D1_long.csv", parse_dates=["time"], index_col="time")
        d = d[~d.index.duplicated(keep="last")].sort_index()
        c = d["close"].astype(float)
    bpy = len(c) / max((c.index[-1] - c.index[0]).days / 365.25, 1e-9)
    return c, bpy


def sleeve(close, scale, bpy, spread_bp, carry_pct_yr, target_vol):
    """3bs's engine verbatim: spread on TURNOVER, carry on HOLDINGS."""
    ret = close.pct_change()
    vol = ret.ewm(halflife=VOL_HL, min_periods=20).std() * np.sqrt(bpy)
    fc = champion_recipe(close, scale, 1.5)
    pos = (fc.clip(-2, 2) * (target_vol / vol)).clip(-MAX_LEV, MAX_LEV).shift(1)
    trade_cost = pos.diff().abs().fillna(0.0) * (spread_bp / 1e4) * 1.5
    carry_cost = pos.abs() * (abs(carry_pct_yr) / 100.0) / bpy
    r = (pos * ret - trade_cost - carry_cost).dropna()
    return (1 + r).resample("1D").prod().sub(1)


def perf(r, ppy=252):
    r = r.dropna(); eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1)
    cagr = float(eq.iloc[-1] ** (ppy / len(r)) - 1)
    mdd = float(dd.min())
    return dict(sharpe=float(r.mean() / r.std() * np.sqrt(ppy)), cagr=cagr, maxdd=mdd,
                calmar=(cagr / abs(mdd)) if mdd else np.nan,
                worst_m=float(((1 + r).resample("ME").prod() - 1).min()))


def vol_match(a: pd.Series, b: pd.Series) -> pd.Series:
    """Lever `a` to `b`'s realised vol before any paired-t (method rule 1.3.1)."""
    sa, sb = a.std(), b.std()
    return a * (sb / sa) if sa > 0 else a


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    spreads = load_census_spreads()
    for k, v in SLEEVES.items():
        if v["spread_bp"] is None:
            v["spread_bp"] = spreads.get(v["sym"])
            if v["spread_bp"] is None:
                print("MISSING spread for %s (%s) -- census not ready" % (k, v["sym"]))
                return 2

    print("=" * 108)
    print("BOOK WIDTH ON THE REGIME-U CRITERION (net Sharpe), WITH CARRY — pre-registered")
    print("=" * 108)
    print("  sleeve costs, measured at Switch:")
    for k, v in SLEEVES.items():
        print("    %-6s %-9s spread %5.2fbp   carry %+7.2f%%/yr" %
              (k, v["sym"], v["spread_bp"], v["carry"]))

    streams = {}
    for name, v in SLEEVES.items():
        c, bpy = close_series(v["src"])
        streams[name] = (c, bpy)
        print("    %-6s %s -> %s (%d bars, %.0f/yr)" %
              (name, c.index[0].date(), c.index[-1].date(), len(c), bpy))

    results, daily = [], {}
    for bname, members in BOOKS.items():
        for dial in DIALS:
            legs = []
            for mname in members:
                c, bpy = streams[mname]
                v = SLEEVES[mname]
                legs.append(sleeve(c, v["scale"], bpy, v["spread_bp"], v["carry"], dial))
            lo = max(l.index[0] for l in legs).normalize()
            hi = min(l.index[-1] for l in legs).normalize()
            bd = pd.bdate_range(lo, hi)                      # rule 1.3.6: common business days
            w = 1.0 / len(legs)
            r = sum(w * l.reindex(bd).fillna(0.0) for l in legs)
            m = perf(r)
            results.append(dict(book=bname, n_sleeves=len(members), dial=dial, **m))
            if dial == 0.09:
                daily[bname] = r
    R = pd.DataFrame(results)
    R.to_csv(OUT / "width_frontier.csv", index=False)

    print("\n--- CONTROL CELLS (known answers from 3bs; if these are off the harness is wrong) ---")
    c1 = R[(R.book == "XAU alone") & (R.dial == 0.09)].iloc[0]
    c2 = R[(R.book.str.startswith("XAU+BTC ")) & (R.dial == 0.09)].iloc[0]
    print("  XAU alone  @0.09: Sharpe %+.3f   (3bs/3br reference ~+0.92)" % c1.sharpe)
    print("  XAU+BTC    @0.09: Sharpe %+.3f   (3bs reference      ~+1.29)" % c2.sharpe)

    print("\n--- NET SHARPE BY BOOK (Sharpe is leverage-invariant, so one column suffices) ---")
    print("  %-22s %3s %9s %9s %9s %9s %9s" %
          ("book", "n", "Sharpe", "CAGR@.09", "maxDD@.09", "Calmar", "worst mo"))
    print("  " + "-" * 76)
    for bname in BOOKS:
        row = R[(R.book == bname) & (R.dial == 0.09)].iloc[0]
        print("  %-22s %3d %+9.3f %+8.1f%% %8.1f%% %9.2f %+8.1f%%" %
              (bname, row.n_sleeves, row.sharpe, 100 * row.cagr, 100 * row.maxdd,
               row.calmar, 100 * row.worst_m))

    print("\n--- THE PRE-REGISTERED GATE vs the incumbent (matched-vol paired-t, both halves) ---")
    inc = daily["XAU+BTC  [INCUMBENT]"]
    print("  %-22s %9s %9s %9s %9s  verdict" % ("book", "t(matched)", "H1 dSR", "H2 dSR", "dSharpe"))
    print("  " + "-" * 74)
    for bname, r in daily.items():
        if bname == "XAU+BTC  [INCUMBENT]":
            continue
        idx = inc.index.intersection(r.index)
        a, b = r.reindex(idx).fillna(0.0), inc.reindex(idx).fillna(0.0)
        am = vol_match(a, b)
        d = am - b
        t = float(d.mean() / (d.std() / np.sqrt(len(d)))) if d.std() > 0 else np.nan
        mid = len(idx) // 2
        h1 = perf(a.iloc[:mid])["sharpe"] - perf(b.iloc[:mid])["sharpe"]
        h2 = perf(a.iloc[mid:])["sharpe"] - perf(b.iloc[mid:])["sharpe"]
        ds = perf(a)["sharpe"] - perf(b)["sharpe"]
        ok = (t >= 2.0) and (h1 > 0) and (h2 > 0) and (ds > 0)
        print("  %-22s %+9.2f %+9.3f %+9.3f %+9.3f  %s" %
              (bname, t, h1, h2, ds, "** ADOPT **" if ok else "fails gate"))

    print("\n--- DIAL FRONTIER (loss tolerance, NOT edge: Sharpe is leverage-invariant) ---")
    print("  %-22s %6s %9s %9s %9s" % ("book", "dial", "CAGR", "maxDD", "Calmar"))
    print("  " + "-" * 60)
    for bname in BOOKS:
        for dial in DIALS:
            row = R[(R.book == bname) & (R.dial == dial)].iloc[0]
            print("  %-22s %6.2f %+8.1f%% %8.1f%% %9.2f" %
                  (bname if dial == DIALS[0] else "", dial, 100 * row.cagr,
                   100 * row.maxdd, row.calmar))
    print("\n  -> %s" % (OUT / "width_frontier.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
