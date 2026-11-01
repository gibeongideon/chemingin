#!/usr/bin/env python3
"""FUTURES LAB — what the swap-free futures at Switch actually cost, and which can be rolled.

Retail-accessible by construction (`retail-accessible-only`): these are ordinary symbols on an
account we hold, trade_mode 4, and Switch permits EAs.

WHY FUTURES AT ALL. `financing-is-the-live-lever` measured swap at **-3.79%/yr on the book, which
is -0.29 Sharpe** -- larger than any signal result in this project's history, and §3ag's words
were that one contractual change was "worth more than the 64 signal trials in that round".
Futures carry `swap_mode = DISABLED`, so the nightly debit disappears.

*** THE TRAP, AND IT IS THE WHOLE POINT OF THIS SCRIPT. *** Futures are not free money: **the
financing is in the BASIS**, not in a nightly line item. A future trading above spot decays toward
spot by expiry, and that decay IS the carry, merely invisible. §3by measured it once at -4.13%/yr
realised against spot's -5.76%/yr, i.e. a saving of 1.6-3.2pp, **not the 5.76 a naive reading
suggests**. So the question is never "is it swap-free" but "what does the basis actually cost".

FOUR EXPERIMENTS:
  F1  ROLL CONTINUITY. A contract you cannot roll is not a position you can hold. Which families
      list a successor and which do not?
  F2  THE BASIS, measured as a time series rather than as a snapshot -- the decay rate IS the
      implied carry.
  F3  TRACKING. Correlation and tracking error against the spot the book actually models.
  F4  THE INVERSION. `financing-is-the-live-lever` records BRENT at **+9.6%/yr, a sleeve that
      EARNS its carry**. For that one, going swap-free would FORFEIT income. Swap-free is not
      universally better and this checks which way each sleeve points.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "v5_runs"
H1 = 16385
# MT5/CME-style month codes. Z=Dec, F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun,
# N=Jul, Q=Aug, U=Sep, V=Oct, X=Nov.
MONTH = dict(F=1, G=2, H=3, J=4, K=5, M=6, N=7, Q=8, U=9, V=10, X=11, Z=12)


def decode(sym: str):
    """'GC.Z6s' -> ('GC', 2026, 12). Returns (root, None, None) if it is not a dated contract."""
    s = sym[:-1] if sym.endswith("s") else sym
    if "." not in s:
        return s, None, None
    root, tail = s.rsplit(".", 1)
    if len(tail) >= 2 and tail[0] in MONTH and tail[1].isdigit():
        return root, 2020 + int(tail[1]), MONTH[tail[0]]
    return root, None, None


def bars(m, sym: str, n: int = 60000) -> pd.DataFrame:
    r = m.copy_rates_from_pos(sym, H1, 0, n)
    if r is None or not len(r):
        return pd.DataFrame()
    d = pd.DataFrame([(x[0], x[4]) for x in r], columns=["ts", "close"])
    d["t"] = pd.to_datetime(d.ts, unit="s")
    return d.set_index("t")[["close"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18815)
    a = ap.parse_args()
    from mt5linux import MetaTrader5
    m = MetaTrader5(host="localhost", port=a.port)
    if not m.initialize():
        print("bridge init failed:", m.last_error()); return 2

    syms = m.symbols_get()
    free = []
    for x in syms:
        si = m.symbol_info(x.name)
        if si is not None and si.swap_mode == 0:
            free.append(x.name)

    print("=" * 100)
    print("FUTURES LAB — Switch Markets 40080430 (DEMO), bridge 18815")
    print("=" * 100)
    print(f"  {len(free)} swap-free symbols of {len(syms):,}")

    # ---------------- F1 — ROLL CONTINUITY --------------------------------------------------
    fam = {}
    for n in free:
        root, yr, mo = decode(n)
        if yr:
            fam.setdefault(root, []).append((yr, mo, n))
    print(f"\n  F1 — ROLL CONTINUITY. A contract you cannot roll is not a position you can hold.")
    print(f"  {'root':>8} {'contracts':>10}  listed (soonest first)            verdict")
    print("  " + "-" * 90)
    rollable, orphan = [], []
    for root in sorted(fam):
        ks = sorted(fam[root])
        lst = ", ".join(f"{n} ({y}-{mth:02d})" for y, mth, n in ks)
        ok = len(ks) >= 2
        (rollable if ok else orphan).append(root)
        print(f"  {root:>8} {len(ks):>10}  {lst:<34} "
              f"{'ROLLABLE' if ok else '** NO SUCCESSOR LISTED **'}")
    print(f"\n    rollable families: {', '.join(rollable) if rollable else 'none'}")
    print(f"    ORPHANS (single contract, nothing to roll into): "
          f"{', '.join(orphan) if orphan else 'none'}")

    # ---------------- F2/F3 — BASIS AND TRACKING ---------------------------------------------
    # Spot names verified live 2026-11-01 — Switch's are not the obvious ones (USTECs not
    # NAS100s, BRENTs not UKOILs), and a wrong name fails SILENTLY as "one leg does not resolve".
    pairs = [("GC.Z6s", "XAUUSDs", "gold"), ("SI.Z6s", "XAGUSDs", "silver"),
             ("NQ100.Z6s", "USTECs", "nasdaq"), ("US30.Z6s", "US30s", "dow"),
             ("ES500.Z6s", "US500s", "sp500"), ("DE40.Z6s", "GER40s", "dax"),
             ("BRN.F7s", "BRENTs", "brent F7"), ("BRN.G7s", "BRENTs", "brent G7"),
             ("CL.Z6s", "WTIs", "wti"), ("HG.Z6s", "COPPERs", "copper")]
    print(f"\n  F2/F3 — BASIS (the hidden carry) AND TRACKING, from H1 closes")
    print(f"  {'pair':<20} {'bars':>6} {'win':>6} {'corr':>7} {'TE bp':>6} "
          f"{'basis now':>10} {'REALISED':>11} {'FORWARD':>11} {'DTE':>5}")
    print("  " + "-" * 96)
    rows = []
    for fut, spot, lab in pairs:
        if m.symbol_info(fut) is None or m.symbol_info(spot) is None:
            print(f"  {lab + ' ' + fut:<22} (one leg does not resolve)")
            continue
        m.symbol_select(fut, True); m.symbol_select(spot, True)
        F, S = bars(m, fut), bars(m, spot)
        if F.empty or S.empty:
            print(f"  {lab + ' ' + fut:<22} (no bars)"); continue
        j = F.join(S, lsuffix="_f", rsuffix="_s", how="inner").dropna()
        if len(j) < 200:
            print(f"  {lab + ' ' + fut:<22} {len(j):>6}  too little overlap"); continue
        bas = (j["close_f"] / j["close_s"] - 1.0) * 100.0
        rf = j["close_f"].pct_change(); rs = j["close_s"].pct_change()
        corr = float(rf.corr(rs))
        te = float((rf - rs).std() * 1e4)
        # TWO ESTIMATORS, because one of them is wrong on short windows and it is not obvious
        # which until you see both.
        #   REALISED = the observed basis decay, annualised. Backward-looking, and on a 30-day
        #     window it is dominated by TERM-STRUCTURE MOVES rather than convergence -- which is
        #     why the energy contracts first printed +32%/yr "carry", a number that is really the
        #     oil curve steepening.
        #   FORWARD  = basis_now / time-to-expiry. The basis MUST reach zero at expiry, so this
        #     is what holding from here actually costs, and it is immune to curve moves.
        # §3by reported exactly this pair for gold (-4.13% realised, -2.60% forward); reproducing
        # both is the control cell.
        days = (j.index[-1] - j.index[0]).days
        slope30 = float((bas.iloc[-1] - bas.iloc[0]) / max(days, 1) * 30.0)
        carry = slope30 * 12.0
        _, ey, em = decode(fut)
        fwd_carry, dte = np.nan, np.nan
        if ey:
            # expiration_time reads 0 for every contract here, so expiry is ASSUMED at the 25th
            # of the contract month -- stated rather than hidden, and it only scales the number.
            exp = dt.datetime(ey, em, 25)
            dte = (exp - j.index[-1].to_pydatetime()).days
            if dte > 5:
                fwd_carry = -float(bas.iloc[-1]) * 365.0 / dte
        warn = " ⚠short" if days < 60 else ""
        print(f"  {lab + ' ' + fut:<20} {len(j):>6} {days:>5}d {corr:>7.4f} {te:>6.2f} "
              f"{bas.iloc[-1]:>+9.3f}% {carry:>+10.2f}% {fwd_carry:>+10.2f}% "
              f"{dte if np.isfinite(dte) else 0:>5.0f}{warn}")
        rows.append(dict(pair=lab, fut=fut, spot=spot, n=len(j), days=days, corr=corr,
                         te_bp=te, basis_now=float(bas.iloc[-1]),
                         carry_realised_yr=carry, carry_forward_yr=fwd_carry, days_to_exp=dte))

    # ---------------- F4 — THE DECISION TABLE -----------------------------------------------
    # *** A BUG IN THE FIRST VERSION OF THIS SECTION, FIXED. *** It compared the spot swap against
    # ZERO and concluded "spot pays carry -> swap-free is a saving" for every row. That is the
    # wrong comparison: the alternative to paying a swap is not paying nothing, it is paying the
    # FUTURE'S BASIS. Brent's spot swap is -0.08%/yr while its future costs -3.07%/yr, so going
    # swap-free there LOSES 3pp. The decision is spot_swap vs futures_forward_carry, pairwise.
    print(f"\n  F4 — THE DECISION: spot swap vs the FUTURE'S OWN CARRY, and whether it can be rolled")
    print(f"  {'sleeve':<10} {'spot':<9} {'spot %/yr':>10} {'fut %/yr':>9} {'saving':>9} "
          f"{'rollable':>9}  verdict")
    print("  " + "-" * 92)
    dec = []
    for r in sorted(rows, key=lambda x: -(x.get("carry_forward_yr") or -99)):
        spot = r["spot"]
        si = m.symbol_info(spot); t = m.symbol_info_tick(spot)
        if si is None or t is None or not t.bid or si.swap_mode != 1:
            continue
        pt = float(si.point) or 0.01
        spot_yr = (float(si.swap_long) * pt / float(t.bid)) * 365 * 100
        fut_yr = r.get("carry_forward_yr")
        if fut_yr is None or not np.isfinite(fut_yr):
            continue
        saving = fut_yr - spot_yr
        root, _, _ = decode(r["fut"])
        can_roll = root in rollable
        verdict = ("** BEST AVAILABLE **" if saving > 0 and can_roll else
                   "saves, but ORPHAN — cannot hold past expiry" if saving > 0 else
                   "WORSE than spot — swap-free COSTS here")
        print(f"  {r['pair']:<10} {spot:<9} {spot_yr:>+9.2f}% {fut_yr:>+8.2f}% "
              f"{saving:>+8.2f}% {('yes' if can_roll else 'NO'):>9}  {verdict}")
        dec.append(dict(sleeve=r["pair"], spot=spot, spot_yr=spot_yr, fut_yr=fut_yr,
                        saving_pp=saving, rollable=can_roll))
    if dec:
        pd.DataFrame(dec).to_csv(OUT / "futures_decision.csv", index=False)
        helps = [d for d in dec if d["saving_pp"] > 0]
        both = [d for d in helps if d["rollable"]]
        print(f"\n    futures that SAVE vs their spot : {len(helps)} of {len(dec)}"
              f"  ({', '.join(d['sleeve'] for d in helps) if helps else 'none'})")
        print(f"    of those, ones we can ROLL      : {len(both)} of {len(helps)}"
              f"  ({', '.join(d['sleeve'] for d in both) if both else 'NONE'})")

    if rows:
        pd.DataFrame(rows).to_csv(OUT / "futures_lab.csv", index=False)
    print(f"""
  HOW TO READ 'implied carry'. It is the basis decay annualised from the OBSERVED series, which is
  what holding the future actually costs — the number a swap table never shows you. Compare it to
  the spot leg's own swap in F4 and the difference, not the swap figure alone, is what switching
  is worth. §3by: gold's saving was 1.6-3.2pp/yr, NOT the 5.76 the swap line implies.
  And F1 gates all of it: an ORPHAN contract cannot be held past expiry at any price.""")
    m.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
