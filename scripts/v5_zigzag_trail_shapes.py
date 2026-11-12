#!/usr/bin/env python3
"""Profit-scaled trailing stop WITH A DISTANCE FLOOR on ZigZag variant B (user 2026-11-12: "keep the
ZigZag as it is, now can we try trailing loss to go upward slowly as the profit goes up but avoid
being too close, try it"). Research only — the live bots are untouched. PRE-REGISTERED.

What is new vs §3bn/§3cx/§3dq: the fractional lock (§3cx) failed with NO floor, so it could sit
right under price; here every rule has a MINIMUM distance to the running high.

EVENT SET FIXED (breakeven-stop rule): the entries are CTRL's trades (variant B: P >= 0.60, entry
next open + $0.25, initial stop 3 ATR, TP 4.3R, 120h max, swap). Every variant re-simulates exactly
those entries; all variants are tighter-or-equal to CTRL, so they exit no later and never overlap.
RULES (P = (high since entry - entry) / ATR, ATR of the current bar):
  CTRL            stop = high - 3.0 ATR                                           (live)
  TAPER g/D       stop = high - max(D, 3.0 - g*P) ATR      g 0.25/0.5, D 1.5/2.0   (slow tightening)
  FRAC f/D        stop = max(CTRL, min(entry + f*(high-entry), high - D ATR))  f 0.33/0.5, D 1.5/2.0
  The stop only ever rises.
PASS: paired t (variant R - CTRL R, per trade) > 2.7 (8 variants) in BOTH sources, positive
in both halves. Otherwise the family is closed.

    python scripts/v5_zigzag_trail_shapes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_zigzag_long_history import (MAX_HOLD, OUT as ZZ, RISK, SPREAD, SRC, THR, TP_R,  # noqa: E402
                                            TRAIL, load, sharpe)
from scripts.v5_zigzag_regime import bt  # noqa: E402
from scripts.v5_zigzag_signal import atr  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "zigzag_trail_shapes"
RULES = {"CTRL": ("ctrl", None, None)}
for g in (0.25, 0.5):
    for D in (1.5, 2.0):
        RULES[f"TAPER g{g} D{D}"] = ("taper", g, D)
for f in (0.33, 0.5):
    for D in (1.5, 2.0):
        RULES[f"FRAC f{f} D{D}"] = ("frac", f, D)


def sim(o, h, l, a, t, i, kind, p1, p2, swap):
    """One long trade entered at the open of bar i+1. Mirrors v5_zigzag_regime.bt bar for bar."""
    n = len(o)
    s = i + 1
    risk = TRAIL * a[i]
    e = o[s] + SPREAD
    sl, ext = e - risk, e
    tp = e + TP_R * risk
    j, px = n - 1, o[-1]
    for j in range(s, n):
        if l[j] <= sl:
            px = min(sl, o[j]); break
        if h[j] >= tp:
            px = max(tp, o[j]); break
        if j - s >= MAX_HOLD:
            px = o[j]; break
        ext = max(ext, h[j])
        cand = ext - TRAIL * a[j]
        if kind == "taper":
            P = (ext - e) / a[j]
            cand = ext - max(p2, TRAIL - p1 * P) * a[j]
        elif kind == "frac":
            cand = max(cand, min(e + p1 * (ext - e), ext - p2 * a[j]))
        sl = max(sl, cand)
    days = (t[j] - t[s]).total_seconds() / 86400
    return (px - e + e * swap * days / 365) / risk, t[j], (ext - e) / risk


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []
    for src, cfg in SRC.items():
        g = load(src, cfg["gold"])
        p = pd.read_csv(ZZ / src / "probs.csv", index_col=0, parse_dates=True).iloc[:, 0].reindex(g.index).to_numpy()
        fires = np.isfinite(p) & (p >= THR)
        st = pd.Timestamp(cfg["start"])
        tr0, r0 = bt(g, fires, np.zeros(len(g), bool), st, cfg["swap"], 0)
        o, h, l = (g[x].to_numpy() for x in ("open", "high", "low"))
        a, t = atr(g).to_numpy(), g.index
        pos = {ts: k for k, ts in enumerate(t)}
        entries = [pos[x] - 1 for x in tr0.t_in]                # signal bar = entry bar - 1
        res = {}
        for name, (kind, p1, p2) in RULES.items():
            out = [sim(o, h, l, a, t, i, kind, p1, p2, cfg["swap"]) for i in entries]
            res[name] = pd.DataFrame(out, columns=["R", "t_out", "peakR"])
        c = res["CTRL"]
        chk = float(np.abs(c.R.to_numpy() - tr0.R.to_numpy()).max())
        assert chk < 1e-9, f"CTRL does not reproduce bt: {chk}"
        days = pd.date_range(st.normalize(), t[-1].normalize(), freq="D")
        tin = pd.to_datetime(tr0.t_in).to_numpy()
        mid = tin[len(tin) // 2]
        rows = []
        for name, d in res.items():
            ret = pd.Series(0.0, index=days)
            for r in d.itertuples():
                ret[r.t_out.normalize()] += RISK * r.R
            ret = ret[ret.index.dayofweek < 5]
            eq = ret.cumsum()
            diff = d.R.to_numpy() - c.R.to_numpy()
            se = diff.std(ddof=1) / np.sqrt(len(diff)) if diff.std() > 0 else np.nan
            h1, h2 = diff[tin < mid], diff[tin >= mid]
            rows.append(dict(rule=name, trades=len(d), win=round((d.R > 0).mean(), 3),
                             meanR=round(d.R.mean(), 3), giveback_R=round((d.peakR - d.R).mean(), 2),
                             SR=round(sharpe(ret), 2), maxDD_pct=round(100 * (eq - eq.cummax()).min(), 1),
                             dR=round(diff.mean(), 4), paired_t=round(diff.mean() / se, 2) if se == se else 0.0,
                             dR_h1=round(h1.mean(), 4), dR_h2=round(h2.mean(), 4)))
        A = pd.DataFrame(rows)
        lines += [f"\n{src.upper()} {st:%Y}..: {len(entries)} fixed entries (CTRL reproduces bt, max |dR| {chk:.1e})",
                  A.to_string(index=False)]
        print("done", src, flush=True)
    txt = "Profit-scaled trail with a distance floor on ZigZag B (see docstring)\n" + "\n".join(lines) + "\n"
    print(txt)
    (OUT / "report.txt").write_text(txt)


if __name__ == "__main__":
    sys.exit(main())
