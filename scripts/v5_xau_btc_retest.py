"""Re-test of the XAU+BTC trend book on FRESH data, requested 2026-11-02 before redeploying it on
Switch (the HFM cent account it began on is at 0.12 USC and cannot be revived).

Same engine as v5_btc_vs_xau.py (sleeve/bh/perf imported verbatim), different inputs:
  * gold = Switch's OWN H4 feed (data/XAUUSDs_H4_switch.csv, one broker, 2003->today). The
    original test rebuilt H4 from XAUUSD_M15_long.csv, which ends 2026-06-15.
  * BTC  = data/BTC_D1_long.csv, refreshed daily on the VPS.
NOTHING IS RE-FIT. The recipe has no trained parameters; "retraining" a fixed-rule trend
follower means re-measuring it. The window after 2026-06-15 is the only part these signals
have never been scored on, and at ~100 business days its Sharpe SE is ~1.6 — read it as a
sanity check that nothing broke, never as evidence for or against.

    python scripts/v5_xau_btc_retest.py --xau-h4 data/XAUUSDs_H4_switch.csv \
                                         --btc-d1 data/BTC_D1_long.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from v5_btc_vs_xau import sleeve, bh, perf, ftmo_phase1, TARGET_VOL, VOL_HL, MAX_LEV  # noqa: E402
from src.v5.xau_dual_signals import ls_signal                                       # noqa: E402

LAST_SEEN = pd.Timestamp("2026-06-15")      # end of the M15 file the original test used


def ls_sleeve(close: pd.Series, cost_frac: pd.Series, bpy: float) -> pd.Series:
    """The HFM `ls` bot's long/short forecast through the SAME continuous engine."""
    ret = close.pct_change()
    vol = ret.ewm(halflife=VOL_HL, min_periods=20).std() * np.sqrt(bpy)
    pos = (ls_signal(close).clip(-2, 2) * (TARGET_VOL / vol)).clip(-MAX_LEV, MAX_LEV).shift(1)
    return (pos * ret - pos.diff().abs().fillna(0.0) * cost_frac).dropna()


def load(path: str) -> pd.Series:
    d = pd.read_csv(path, parse_dates=["time"], index_col="time")
    d = d[~d.index.duplicated(keep="last")].sort_index()
    return d["close"].astype(float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xau-h4", default=str(ROOT / "data/XAUUSDs_H4_switch.csv"))
    ap.add_argument("--btc-d1", default=str(ROOT / "data/BTC_D1_long.csv"))
    a = ap.parse_args()

    xc, bc = load(a.xau_h4), load(a.btc_d1)
    xbpy = len(xc) / ((xc.index[-1] - xc.index[0]).days / 365.25)
    bbpy = len(bc) / ((bc.index[-1] - bc.index[0]).days / 365.25)
    xcost = pd.Series(0.672 / xc * 0.75 * 2, index=xc.index)   # same charge as v5_btc_vs_xau
    bcost = 1.0 / bc * 0.75 * 2

    A, A_ls, A_bh = sleeve(xc, 1.0, xcost, xbpy), ls_sleeve(xc, xcost, xbpy), bh(xc, xbpy)
    B, B_bh = sleeve(bc, 1 / 6, bcost, bbpy), bh(bc, bbpy)

    def daily(r):
        return (1 + r).resample("1D").prod().sub(1)
    lo, hi = max(A.index[0], B.index[0]), min(A.index[-1], B.index[-1])
    bd = pd.bdate_range(lo.normalize(), hi.normalize())
    D = {k: daily(v).reindex(bd).fillna(0.0) for k, v in
         dict(A=A, A_ls=A_ls, A_bh=A_bh, B=B, B_bh=B_bh).items()}
    cells = {
        "XAU champion  long-only (H4)": D["A"],
        "XAU ls  long/short (HFM bot)": D["A_ls"],
        "BTC champion  long-only (D1)": D["B"],
        "XAU+BTC 50/50  <- DEPLOYED":   0.5 * D["A"] + 0.5 * D["B"],
        "XAU ls + BTC 50/50":           0.5 * D["A_ls"] + 0.5 * D["B"],
        "hold XAU (vol-targeted)":      D["A_bh"],
        "hold BTC (vol-targeted)":      D["B_bh"],
    }
    print(f"gold {xc.index[0].date()}..{xc.index[-1]}  BTC {bc.index[0].date()}..{bc.index[-1].date()}")
    print(f"common business days {bd[0].date()}..{bd[-1].date()} ({len(bd)/252:.1f}y), dial {TARGET_VOL:.0%}")
    wins = [("full", bd[0]), ("2021+", pd.Timestamp("2021-01-01")),
            ("last 12m", bd[-1] - pd.Timedelta(days=365)), ("NEW since 06-15", LAST_SEEN)]
    print(f"\n  {'cell':30s}" + "".join(f"{w:>17s}" for w, _ in wins) + f"{'maxDD':>8s}")
    print(f"  {'':30s}" + "".join(f"{'Sharpe  ret':>17s}" for _ in wins))
    for name, r in cells.items():
        out = ""
        for _, t0 in wins:
            s = r[r.index >= t0]
            m = perf(s)
            tot = float((1 + s).prod() - 1)
            out += f"  {m['sharpe']:+6.2f} {100*tot:+7.1f}%"
        print(f"  {name:30s}{out}{100*perf(r)['maxdd']:7.1f}%")

    print("\n  per-year return (Sharpe):")
    yrs = sorted(set(bd.year))[-6:]
    print(f"  {'cell':30s}" + "".join(f"{y:>16d}" for y in yrs))
    for name, r in cells.items():
        print(f"  {name:30s}" + "".join(
            f"  {100*float((1+r[r.index.year==y]).prod()-1):+6.1f}% ({perf(r[r.index.year==y])['sharpe']:+5.2f})"
            for y in yrs))

    se_new = np.sqrt(252 / max(1, (D['A'].index >= LAST_SEEN).sum()))
    print(f"\n  SE of an annualised Sharpe on the NEW window ~ {se_new:.2f} — a sanity check, not evidence.")
    print("\n  FTMO-style 1y bootstrap (10% target / 5% daily / 10% total) for the two book forms:")
    for name in ("XAU+BTC 50/50  <- DEPLOYED", "XAU ls + BTC 50/50"):
        f = ftmo_phase1(cells[name])
        print(f"    {name:30s} pass {100*f['pass_rate']:5.1f}%  fail-daily {100*f['fail_daily']:4.1f}%"
              f"  fail-total {100*f['fail_total']:4.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
