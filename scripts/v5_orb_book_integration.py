#!/usr/bin/env python3
"""Does the BTC opening-range breakout (§3di) improve the live XAU+BTC champion book?

Sleeve: BTC, 9:30 ET 5-minute opening range, stop orders at its high/low, stop 10% of ATR14,
flat 16:00 ET, M1 path (bar-order robust: PATH and PESS agree within 0.05 Sharpe), cost 0.05%
of price per side (Switch ~$35). Three versions:
  DIR    the paper's rule (first candle's direction only)
  BOTH   both stop orders live (the structure, which §3di found carries the edge)
  LONG   only the bullish-candle days (drift-asset convention)
Book: `v5_book_overlays.sleeve_returns()` XAU + BTC champion, equal weight (the live pair).
Gate: `paired` t at MATCHED VOL vs the book, years better, both halves, plus Sharpe/DD.

    python scripts/v5_orb_book_integration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_book_overlays import sleeve_returns  # noqa: E402
from scripts.v5_orb_stocks_in_play import sim  # noqa: E402
from scripts.v5_orb_xau_btc import STOP_ATR, events  # noqa: E402
from scripts.v5_xau_champion_lifts import dd_of, sharpe, vol_match  # noqa: E402
from scripts.v5_xau_turn_prob import paired, per_year  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "orb_xau_btc"
import os
ORW, COST = 5, float(os.environ.get("ORB_COST", 0.0005))


def btc_orb(pess: bool = False) -> dict[str, pd.Series]:
    z = np.load(ROOT / "data" / "binance_1m" / "BTCUSDT.npz")
    b = pd.DataFrame({k: z[k] for k in "ohlc"}, index=pd.to_datetime(z["t"], unit="s"))
    b = b[~b.index.duplicated()]
    ev, (o, h, l, c, s) = events(b, 1, lambda d: d.c * COST)
    g = ev.s0.to_numpy() - 1
    orh = np.array([h[a:a + ORW].max() for a in g])
    orl = np.array([l[a:a + ORW].min() for a in g])
    side = np.sign([c[a + ORW - 1] - o[a] for a in g]).astype(int)
    s0 = g + ORW
    R = STOP_ATR * ev.atr.to_numpy()
    args = (orh + s[s0 - 1], orl, R, s0, ev.s1.to_numpy(), o, h, l, c, s, pess)
    L = sim(np.full(len(ev), 1), *args)
    S = sim(np.full(len(ev), -1), *args)
    day = pd.DatetimeIndex(ev.day)
    mk = lambda r: pd.Series(np.nan_to_num(r), index=day).groupby(level=0).sum()  # noqa: E731
    return dict(DIR=mk(np.where(side == 1, L, np.where(side == -1, S, np.nan))),
                BOTH=mk(0.5 * np.nan_to_num(L) + 0.5 * np.nan_to_num(S)),
                LONG=mk(np.where(side == 1, L, np.nan)))


def main() -> None:
    R = sleeve_returns()[["XAU", "BTC"]]
    book = R.mean(axis=1)
    lines = [f"BOOK XAU+BTC champion {book.index[0]:%Y-%m-%d}..{book.index[-1]:%Y-%m-%d}: "
             f"SR {sharpe(book):+.3f}  DD {dd_of(book):+.1f}%  halves "
             f"{sharpe(book[:'2021']):+.2f}/{sharpe(book['2022':]):+.2f}"]
    for pess in (False, True):
        orb = btc_orb(pess)
        tag = "PESS" if pess else "PATH"
        for name, r in orb.items():
            r = r.reindex(book.index, fill_value=0.0)
            k = book.std() / r.std()
            lines.append(f"\n[{tag}] BTC-ORB {name}: standalone SR {sharpe(r):+.2f} (halves "
                         f"{sharpe(r[:'2021']):+.2f}/{sharpe(r['2022':]):+.2f}), corr to book "
                         f"{r.corr(book):+.2f}, to BTC champ {r.corr(R.BTC):+.2f}")
            for w in (0.1, 0.2, 0.33):
                mix = (1 - w) * book + w * k * r
                mm = vol_match(mix, book)
                _, t, _ = paired(mm, book)
                yp, yn = per_year(mm, book)
                lines.append(f"   {int(w*100):2d}% risk: SR {sharpe(mix):+.3f}  DD {dd_of(mix):+.1f}% "
                             f"(matched {dd_of(mm):+.1f}%)  t@matched {t:+.2f}  years {yp}/{yn}  "
                             f"halves {sharpe(mix[:'2021']):+.2f}/{sharpe(mix['2022':]):+.2f}")
    txt = "\n".join(lines) + "\n"
    print(txt)
    (OUT / "book_integration.txt").write_text(txt)


if __name__ == "__main__":
    sys.exit(main())
