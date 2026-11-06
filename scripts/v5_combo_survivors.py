#!/usr/bin/env python3
"""Combination study (user: "combine the best ideas"): the XAU+BTC champion book plus the two
intraday leads that cleared their own tests, at PRE-REGISTERED risk shares.

SLEEVES
  BOOK   XAU+BTC champion, equal weight (`v5_book_overlays.sleeve_returns`)
  ORB    BTC 9:30-ET opening-range breakout, both stop orders, 10%-ATR stop, flat 16:00 ET,
         M1 path, cost 0.05%/side (Switch BTCUSDs 0.045% measured 2026-11-06)       — §3di
  PDH    XAU long-only first break of the PRIOR-DAY HIGH (stop order), exit at the day's close,
         M15 (UTC days), cost = live round trip 0.756bp + 2bp stop slippage            — §3ch
ZigZag variants are NOT included: none beat their random-timing null (§3dg, §3dk, BERT).

MIXES (frozen before running): BOOK + 10% ORB, + 20% ORB, + 10% PDH, + 20% PDH, + 10% ORB + 10% PDH.
Each added sleeve is scaled to the book's vol, then mixed by risk share.
GATE: paired t at MATCHED VOL vs BOOK (>= 2.0), years better, both halves, and the same mix at
2x cost (ORB 0.10%/side, PDH slippage 4bp) must still beat the book.

    python scripts/v5_combo_survivors.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_book_overlays import sleeve_returns  # noqa: E402
from scripts.v5_xau_champion_lifts import dd_of, sharpe, vol_match  # noqa: E402
from scripts.v5_xau_turn_prob import paired, per_year  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "combo_survivors"
LIVE_RT_BP = 0.756


def pdh_stream(slip_bp: float) -> pd.Series:
    d = pd.read_csv(ROOT / "data/XAUUSD_M15_long.csv", parse_dates=["time"], index_col="time")
    d = d[~d.index.duplicated(keep="last")].sort_index()
    day = d.index.normalize()
    pdh = d.high.groupby(day).max().shift(1).reindex(day).to_numpy(float)
    pdl = d.low.groupby(day).min().shift(1).reindex(day).to_numpy(float)
    h, l, c = (d[k].to_numpy(float) for k in ("high", "low", "close"))
    o = d.open.to_numpy(float)
    nd = day.to_numpy()
    out, start = {}, 0
    for i in range(1, len(d) + 1):
        if i == len(d) or nd[i] != nd[start]:
            if np.isfinite(pdh[start]) and np.isfinite(pdl[start]):
                up = np.flatnonzero(h[start:i] > pdh[start])
                dn = np.flatnonzero(l[start:i] < pdl[start])
                fu = up[0] if len(up) else 10**9
                fd = dn[0] if len(dn) else 10**9
                if fu < fd:                                         # LONG: high broke first
                    k = start + fu
                    entry = max(pdh[start], o[k])                    # gap through -> open
                    out[nd[start]] = (c[i - 1] / entry - 1.0) - (LIVE_RT_BP + slip_bp) / 1e4
            start = i
    return pd.Series(out).sort_index()


def orb_stream(cost: float) -> pd.Series:
    os.environ["ORB_COST"] = str(cost)
    import importlib  # noqa: PLC0415
    import scripts.v5_orb_book_integration as m  # noqa: PLC0415
    m = importlib.reload(m)
    return m.btc_orb(False)["BOTH"]


def mix(book: pd.Series, sleeves: dict, shares: dict) -> pd.Series:
    out = book * (1 - sum(shares.values()))
    for k, w in shares.items():
        s = sleeves[k].reindex(book.index, fill_value=0.0)
        out = out + w * s * (book.std() / s.std())
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    book = sleeve_returns()[["XAU", "BTC"]].mean(axis=1)
    sets = {"1x": dict(ORB=orb_stream(0.0005), PDH=pdh_stream(2.0)),
            "2x": dict(ORB=orb_stream(0.0010), PDH=pdh_stream(4.0))}
    end = min(book.index[-1], sets["1x"]["PDH"].index[-1])
    book = book[:end]
    lines = [f"window {book.index[0]:%Y-%m-%d}..{end:%Y-%m-%d}; BOOK SR {sharpe(book):+.3f} "
             f"DD {dd_of(book):+.1f}% halves {sharpe(book[:'2021']):+.2f}/{sharpe(book['2022':]):+.2f}"]
    for tag, sl in sets.items():
        for k, s in sl.items():
            s = s.reindex(book.index, fill_value=0.0)
            lines.append(f"[{tag}] {k}: SR {sharpe(s):+.2f} halves {sharpe(s[:'2021']):+.2f}/"
                         f"{sharpe(s['2022':]):+.2f} corr book {s.corr(book):+.2f}")
    a = sets["1x"]["ORB"].reindex(book.index, fill_value=0)
    b = sets["1x"]["PDH"].reindex(book.index, fill_value=0)
    lines.append(f"corr ORB-PDH {a.corr(b):+.2f}")
    MIXES = {"+10% ORB": {"ORB": .1}, "+20% ORB": {"ORB": .2}, "+10% PDH": {"PDH": .1},
             "+20% PDH": {"PDH": .2}, "+10% ORB +10% PDH": {"ORB": .1, "PDH": .1}}
    rows = []
    for name, sh in MIXES.items():
        r = {"mix": name}
        for tag, sl in sets.items():
            m = mix(book, sl, sh)
            mm = vol_match(m, book)
            _, t, _ = paired(mm, book)
            yp, yn = per_year(mm, book)
            r.update({f"SR_{tag}": round(sharpe(m), 3), f"t_{tag}": round(t, 2),
                      f"yrs_{tag}": f"{yp}/{yn}", f"DD_{tag}": round(dd_of(m), 1),
                      f"halves_{tag}": f"{sharpe(m[:'2021']):+.2f}/{sharpe(m['2022':]):+.2f}"})
        r["PASS"] = bool(r["t_1x"] >= 2.0 and r["SR_2x"] > sharpe(book))
        rows.append(r)
    tab = pd.DataFrame(rows)
    txt = "\n".join(lines) + "\n\n" + tab.to_string(index=False) + (
        "\n\nPASS = matched-vol paired t >= 2.0 at 1x cost AND Sharpe at 2x cost still above the book.\n")
    print(txt)
    (OUT / "report.txt").write_text(txt)


if __name__ == "__main__":
    sys.exit(main())
