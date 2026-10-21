#!/usr/bin/env python3
"""Swap-free GOLD at a venue that permits EAs: substitute GC.Z6s for XAUUSDs and re-price the book.

THE FIND (census, 2026-10-21). Switch Markets exposes 38 symbols with `swap_mode = DISABLED`,
including GOLD, NASDAQ, DOW, S&P, DAX, FTSE, EuroStoxx, Brent, WTI and SILVER futures. That
directly contradicts 3ah's "there is NO swap-free index, crypto or energy" -- which was measured on
MAVEN'S 68 SYMBOLS, not here.

WHY THIS MATTERS MORE THAN IT LOOKS. 3ag found GoldEternal at Maven and called it "worth more than
every signal result in this repo's history" (+0.20 Sharpe on the gold sleeve), and it has sat
UNHARVESTED FOR SEVEN WEEKS for one reason: Maven forbids EAs, so no order could ever be placed and
no overnight settlement was ever observed (3bd). Switch permits automation -- the book is already
trading there on magic 360595.

AND THE TRAP, WHICH IS THE WHOLE POINT. A future is swap-free because futures DO NOT PAY SWAP --
the financing is embedded in the BASIS. Buying a contract above spot and holding to expiry pays the
carry through convergence instead of through a nightly debit. So `swap_mode = DISABLED` is NOT free
money and the honest cost is the MEASURED basis decay, not zero.

MEASURED, not assumed (H1 bars, 2026-07-21 -> 2026-10-21, 1,518 common bars):
  * return correlation GC.Z6s vs XAUUSDs = 0.9990  (GoldEternal needed 0.998 -- this is better)
  * basis +1.532% -> +0.491%, i.e. -1.041% over 92 days = -4.13%/yr REALISED
  * forward basis +0.491% with ~69 days to a Dec expiry = -2.60%/yr
  * spot XAUUSDs swap = -5.76%/yr
So the saving is roughly 1.6 to 3.2 percentage points a year, NOT the full 5.76.

Both the realised and forward estimates are priced below, plus the extra spread (0.89bp vs 0.50bp)
and a quarterly roll. The signal is unchanged and still computed on ELEVEN YEARS of spot history --
only the execution ticket differs. GC.Z6s itself has just 3 months of bars and could never be
backtested; that is exactly the GoldEternal pattern (trade the spot-derived signal, execute on the
tracking instrument), and it is why the 0.9990 correlation is load-bearing.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from v5_regime_u_width import sleeve, perf, close_series, vol_match   # noqa: E402

BTC = dict(spread_bp=4.46, carry=-15.00, scale=1/6)
# gold execution options, all measured
GOLD = {
    "XAUUSDs spot (incumbent)":      dict(spread_bp=0.50, carry=-5.76),
    "GC.Z6s futures, REALISED basis": dict(spread_bp=0.89, carry=-4.13, roll_bp_yr=5.34),
    "GC.Z6s futures, FORWARD basis":  dict(spread_bp=0.89, carry=-2.60, roll_bp_yr=5.34),
    "hypothetical ZERO carry (3ag ceiling)": dict(spread_bp=0.89, carry=0.0, roll_bp_yr=5.34),
}


def main() -> int:
    xc, xbpy = close_series("H4")
    bc, bbpy = close_series("BTC")
    print("=" * 100)
    print("SWAP-FREE GOLD AT SWITCH — substitute GC.Z6s for XAUUSDs in the LIVE book")
    print("=" * 100)
    print("  signal unchanged (11y of spot history); only the execution ticket differs.")
    print("  GC.Z6s tracks XAUUSDs at return-correlation 0.9990 over 1,518 H1 bars.\n")
    print("  %-40s %9s %9s %9s %9s" % ("gold leg", "SR gold", "SR book", "CAGR", "maxDD"))
    print("  " + "-" * 82)
    base_book = base_gold = None
    rows = []
    for name, g in GOLD.items():
        roll = g.get("roll_bp_yr", 0.0) / 1e4
        X = sleeve(xc, 1.0, xbpy, g["spread_bp"], g["carry"], 0.09)
        X = X - roll / 252.0                     # quarterly roll, amortised per business day
        B = sleeve(bc, BTC["scale"], bbpy, BTC["spread_bp"], BTC["carry"], 0.09)
        bd = pd.bdate_range(max(X.index[0], B.index[0]).normalize(),
                            min(X.index[-1], B.index[-1]).normalize())
        r = 0.5 * X.reindex(bd).fillna(0.0) + 0.5 * B.reindex(bd).fillna(0.0)
        mg, mb = perf(X.reindex(bd).fillna(0.0)), perf(r)
        if base_book is None:
            base_book, base_gold = mb["sharpe"], mg["sharpe"]
        rows.append((name, mg["sharpe"], mb["sharpe"], mb["cagr"], mb["maxdd"], r))
        print("  %-40s %+9.3f %+9.3f %+8.1f%% %8.1f%%" %
              (name, mg["sharpe"], mb["sharpe"], 100 * mb["cagr"], 100 * mb["maxdd"]))

    print("\n  --- gain over the incumbent ---")
    inc = rows[0][5]
    for name, sg, sb, _, _, r in rows[1:]:
        idx = inc.index.intersection(r.index)
        a, b = r.reindex(idx).fillna(0.0), inc.reindex(idx).fillna(0.0)
        d = vol_match(a, b) - b
        t = float(d.mean() / (d.std() / np.sqrt(len(d)))) if d.std() > 0 else np.nan
        print("    %-40s dSR gold %+.3f  dSR book %+.3f  matched-vol t %+.2f" %
              (name, sg - base_gold, sb - base_book, t))
    print("\n  *** READ THE t-STATS CORRECTLY -- THEY ARE HUGE AND THEY MEAN NOTHING. ***")
    print("  +54 to +62 is not evidence of edge. It is the signature of a DETERMINISTIC cost")
    print("  substitution: the two streams run the SAME signal on the SAME prices and differ only")
    print("  by a constant applied to |position|, so the difference is almost always positive with")
    print("  near-zero variance, and the t explodes. Method rule 1.3.2 exists for exactly this")
    print("  (DSR 0.998 alongside PBO 0.845 on an overlay). The gain here is ARITHMETIC -- a known")
    print("  carry saving on a known exposure -- and the number to trust is the CARRY MEASUREMENT,")
    print("  not the t. Which is why the basis was measured over 92 days rather than assumed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
