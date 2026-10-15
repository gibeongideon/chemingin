"""GATE C1 — the lookback-window question, answered descriptively before any model is fitted.

THE USER'S QUESTION: "days have highs and lows, also weeks — what is the sweet spot of
prediction, previous two days, same day, last five days?"

THE CONTROL THAT ANSWERS IT (§3ay, V5_FINDINGS:3328-3330), verbatim:

    "any statistic about WHERE an extreme falls in a window is mechanically determined by the
     sign of that window's return. Split by outcome first; if it inverts, there is nothing there."

Three for three so far:
  * Mon-low / Fri-high  -> UP weeks Mon-low 0.508, DOWN weeks 0.088 (and Mon holds the HIGH 0.467)
  * the close-position U-shape -> collapsed to corr +0.0245, t +1.34, driven by 2026 alone
  * session timing -> P(low forms 00:00-04:00) = 0.547 on UP days vs 0.045 on DOWN days

So for each candidate window this asks: **is "we are at the window's extreme" a statement about
the market, or a restatement of which way the window happened to move?** If the conditional
probabilities invert when split by the window's own return sign, the pattern is drift wearing a
pattern's clothes and no classifier is warranted.

It also prints, per window, the fraction of bars that ARE at the extreme — because a window whose
extreme is touched on 40% of bars is not identifying anything rare.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/v5_runs/trail_decomp"

WINDOWS = [("same-day", 24), ("2 days", 48), ("5 days", 120),
           ("10 days", 240), ("20 days", 480)]
TOL_BARS = 1          # "at the extreme" = within 1 bar of it


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    h1 = pd.read_csv(ROOT / "data/XAUUSD_H1_long.csv", parse_dates=["time"],
                     index_col="time").sort_index()
    h1 = h1[~h1.index.duplicated(keep="last")]
    hi, lo, c = h1["high"].values, h1["low"].values, h1["close"].values
    n = len(h1)

    print("=" * 104)
    print("GATE C1 — is 'we are at the window's extreme' a market statement, or a restatement")
    print("          of which way the window moved?   (§3ay's outcome-split control)")
    print(f"  XAUUSD H1, {n:,} bars, {h1.index[0].date()} .. {h1.index[-1].date()}")
    print("=" * 104)

    rows = []
    for name, W in WINDOWS:
        # causal rolling extremes over the trailing W bars, inclusive of now
        rmax = pd.Series(hi).rolling(W, min_periods=W).max().values
        rmin = pd.Series(lo).rolling(W, min_periods=W).min().values
        # "the window's HIGH is here" = this bar's high is the window max (within TOL bars)
        at_hi = np.zeros(n, bool)
        at_lo = np.zeros(n, bool)
        for k in range(TOL_BARS + 1):
            s = slice(k, n)
            at_hi[s] |= (hi[: n - k] >= rmax[s] - 1e-9)
            at_lo[s] |= (lo[: n - k] <= rmin[s] + 1e-9)
        wret = np.full(n, np.nan)
        wret[W:] = c[W:] / c[:-W] - 1.0                 # the window's OWN return
        ok = np.isfinite(rmax) & np.isfinite(wret)
        up, dn = ok & (wret > 0), ok & (wret <= 0)

        p_hi_up, p_hi_dn = at_hi[up].mean(), at_hi[dn].mean()
        p_lo_up, p_lo_dn = at_lo[up].mean(), at_lo[dn].mean()
        # an INVERSION = the high is far more likely on up-windows and the low on down-windows
        inverted = (p_hi_up > p_hi_dn) and (p_lo_dn > p_lo_up)
        ratio_hi = p_hi_up / max(p_hi_dn, 1e-9)
        ratio_lo = p_lo_dn / max(p_lo_up, 1e-9)
        rows.append(dict(window=name, bars=W, pooled_at_hi=round(float(at_hi[ok].mean()), 4),
                         pooled_at_lo=round(float(at_lo[ok].mean()), 4),
                         hi_on_UP=round(float(p_hi_up), 4), hi_on_DOWN=round(float(p_hi_dn), 4),
                         lo_on_UP=round(float(p_lo_up), 4), lo_on_DOWN=round(float(p_lo_dn), 4),
                         ratio_hi=round(float(ratio_hi), 1), ratio_lo=round(float(ratio_lo), 1),
                         inverted=bool(inverted), n=int(ok.sum())))

    tb = pd.DataFrame(rows)
    print("\nPOOLED — how often a bar sits at the window's extreme (before any split):")
    print(tb[["window", "bars", "pooled_at_hi", "pooled_at_lo", "n"]].to_string(index=False))

    print("\nSPLIT BY THE SIGN OF THE WINDOW'S OWN RETURN — the control:")
    print(tb[["window", "hi_on_UP", "hi_on_DOWN", "ratio_hi",
              "lo_on_UP", "lo_on_DOWN", "ratio_lo", "inverted"]].to_string(index=False))

    all_inv = bool(tb.inverted.all())
    print("\n" + "=" * 104)
    if all_inv:
        print("VERDICT: INVERTS AT EVERY WINDOW — four-for-four with §3ay's three prior kills.")
        print("  'We are at the N-day high' is not a market state; it is a restatement of the")
        print("  fact that the last N days went up. There is no window sweet spot to find,")
        print("  because the quantity being swept carries no information to begin with.")
        print("  C2 (the day-scale classifier) is CANCELLED by the pre-registered kill.")
    else:
        keep = tb[~tb.inverted]["window"].tolist()
        print(f"VERDICT: survives at {keep} — C2 may run on those windows only.")
    print("=" * 104)

    tb.to_csv(OUT / "c1_window_drift.csv", index=False)
    json.dump(dict(all_inverted=all_inv, rows=rows),
              open(OUT / "c1_verdict.json", "w"), indent=2)
    print(f"\n  wrote {OUT}/c1_window_drift.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
