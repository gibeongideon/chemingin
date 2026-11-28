#!/usr/bin/env python3
"""Second look #9: index OVERNIGHT (16:00 -> 09:30 ET) and 1-day REVERSAL sleeves, synchronised, on 24h
CFD prices (pre-registered addendum in data/v5_runs/second_look/PREREG.md).
§3b's overnight ensemble (OOS 1.19) was 'verified dead' on HFM futures-CFDs using their D1 bar boundary
(17:00 -> 18:00 ET), which is not the cash-session window. On a 24h CFD the 16:00 -> 09:30 window IS
tradeable, so this re-tests it on OANDA M5 NAS100/SPX500/US30/US2000 (same 16:00 close = synchronised).

    python scripts/v5_second_look_9.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from scripts.v5_pdh_panel import load  # noqa: E402
from v5_account_chooser_2026_10 import xau_btc_book  # noqa: E402
from v5_second_look_1 import matched_t, sharpe  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "second_look" / "9"
UNIV = {"NAS100USD": 1.0, "SPX500USD": 2.0, "US30USD": 1.0, "US2000USD": 5.0}   # round-trip bp
FIN, EVAL = 0.04, "2008-01-01"


def legs(sym: str, cost_bp: float, k: float = 1.0) -> pd.DataFrame:
    d = load(sym)
    t = d.index
    c16 = d.close[(t.hour == 15) & (t.minute == 55)]
    o930 = d.open[(t.hour == 9) & (t.minute == 30)]
    c16.index, o930.index = c16.index.normalize(), o930.index.normalize()
    c16, o930 = c16[~c16.index.duplicated()], o930[~o930.index.duplicated()]
    days = c16.index.intersection(o930.index)
    c16, o930 = c16.reindex(days), o930.reindex(days)
    nxt_open = o930.shift(-1)
    nights = pd.Series((days[1:] - days[:-1]).days, index=days[:-1]).reindex(days)
    on_g = nxt_open / c16 - 1                                       # 16:00 D -> 09:30 next day
    on_net = on_g - k * cost_bp * 1e-4 - FIN * nights / 365
    r16 = c16.pct_change()                                          # 16:00 -> 16:00, stamped at its end
    pos = -np.sign(r16.shift(1))
    rev_g = pos * r16
    flips = pos.diff().abs().fillna(1.0) / 2.0
    rev_net = rev_g - k * cost_bp * 1e-4 * flips - FIN * (pos > 0) * nights.shift(1).fillna(1) / 365
    hold24 = r16 - FIN * nights.shift(1).fillna(1) / 365
    out = pd.DataFrame(dict(on_g=on_g.shift(1), on_net=on_net.shift(1), rev_net=rev_net, hold24=hold24)).dropna()
    return out[out.index >= EVAL]                                   # all stamped on the day the leg ENDS


def basket(cols: dict[str, pd.Series]) -> pd.Series:
    W = []
    for s in cols.values():
        w = 1.0 / s.rolling(60, min_periods=20).std().shift(1)
        W.append((s * w / w.median()).rename(s.name))
    return pd.concat(W, axis=1).mean(axis=1, skipna=True).dropna()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    L1 = {s: legs(s, c) for s, c in UNIV.items()}
    L2 = {s: legs(s, c, 2.0) for s, c in UNIV.items()}
    spx = L1["SPX500USD"]
    print(f"KNOWN ANSWER: SPX gross overnight drift {spx.on_g.mean() * 1e4:+.2f}bp/night, "
          f"gross overnight SR {sharpe(spx.on_g):+.3f}")
    for s, x in L1.items():
        print(f"  {s:10s} overnight net SR {sharpe(x.on_net):+.3f}  reversal net SR {sharpe(x.rev_net):+.3f}  "
              f"24h hold SR {sharpe(x.hold24):+.3f}")
    O1 = basket({s: x.on_net.rename(s) for s, x in L1.items()})
    R1 = basket({s: x.rev_net.rename(s) for s, x in L1.items()})
    H = basket({s: x.hold24.rename(s) for s, x in L1.items()})
    O1x2 = basket({s: x.on_net.rename(s) for s, x in L2.items()})
    R1x2 = basket({s: x.rev_net.rename(s) for s, x in L2.items()})
    book = xau_btc_book()
    rows = []
    for name, sl, sl2 in (("O1 overnight (PRIMARY)", O1, O1x2), ("R1 1-day reversal", R1, R1x2)):
        j = sl.index.intersection(book.index).intersection(H.index)
        s_, b_, h_ = sl.reindex(j), book.reindex(j), H.reindex(j)
        t1 = matched_t(s_, h_)[0]
        comb = 0.5 * s_ * (b_.std() / s_.std()) + 0.5 * b_
        t2, da = matched_t(comb, b_)
        mid = j[len(j) // 2]
        h1, h2 = matched_t(comb[:mid], b_[:mid])[0], matched_t(comb[mid:], b_[mid:])[0]
        s2 = sl2.reindex(j).fillna(0)
        t2x = matched_t(0.5 * s2 * (b_.std() / s2.std()) + 0.5 * b_, b_)[0]
        print(f"\n{name}: {j[0].date()} -> {j[-1].date()} ({len(j)} days)")
        print(f"  sleeve net SR {sharpe(s_):+.3f} (2x cost {sharpe(s2):+.3f})  vs 24h hold basket {sharpe(h_):+.3f}: "
              f"T1 t {t1:+.2f}   corr to book {np.corrcoef(s_, b_)[0, 1]:+.2f}")
        print(f"  book SR {sharpe(b_):+.3f} -> 50/50 {sharpe(comb):+.3f}: T2 t {t2:+.2f} halves {h1:+.2f}/{h2:+.2f}  "
              f"2x cost T2 {t2x:+.2f}")
        rows.append(dict(sleeve=name, sr=sharpe(s_), sr_2x=sharpe(s2), t1=t1, book_sr=sharpe(b_), comb_sr=sharpe(comb),
                         t2=t2, h1=h1, h2=h2, t2_2x=t2x, corr=float(np.corrcoef(s_, b_)[0, 1])))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "cells.csv", index=False)
    for r in R.itertuples():
        ok = r.t2 >= 2 and r.h1 > 0 and r.h2 > 0 and r.t2_2x > 0
        print(f"VERDICT {r.sleeve}: {'REVIVED' if ok else 'not revived'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
