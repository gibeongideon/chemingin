#!/usr/bin/env python3
"""NASDAQ-100 (NDX) study (§3du, user request 2026-11-09).

Executability (Switch, live 2026-11-09): USTECs cash CFD, $1/pt, min 0.1 lot (~$2.7k notional),
spread 1.9 pt = 0.70bp, LONG SWAP -5.47%/yr of notional; NQ100.Z6s futures CFD $20/pt, min 0.01
(~$5.5k), 1.10bp, swap-free but expires quarterly. Darwinex: no Nasdaq symbol.

Data: Databento CME NQ / GC / BTC hourly (single provenance, back-adjusted), server-clock H4 as live.
Futures returns already net out carry (~cash rate minus dividends); the CFD's swap is ~1.7%/yr worse,
charged as a stress on long notional ("+swap" rows).

1  champion recipe on NQ H4 (identical code/params to gold) vs NQ buy & hold and constant long at
   the same vol; eras incl. the 2022 tech bear; timing paired t at matched vol.
2  BOOK: equal-risk sleeves, daily. XAU (GC H4 champion) + BTC (D1 champion, CME BTC) = the live book
   vs + NDX. Gate (memory: binding-constraint / matched-vol rule): the 3-sleeve book must beat the
   2-sleeve book at MATCHED VOL (paired t >= 2) and in both halves.

    python scripts/v5_ndx_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.v5_volregime_taper_crossasset as T  # noqa: E402
from scripts.v5_champion_bear_test import stats  # noqa: E402
from scripts.v5_footprint_screen import utc_to_server  # noqa: E402
from scripts.v5_xau_champion_lifts import champion_recipe  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "ndx_study"
EXTRA_SWAP = 0.017


def bars(root: str, rule: str) -> pd.DataFrame:
    g = pd.read_csv(ROOT / f"data/databento/ohlcv_1h/{root}.csv.gz", parse_dates=["time"], index_col="time")
    k = g.close_adj / g.close
    g = pd.DataFrame(dict(open=g.open * k, high=g.high * k, low=g.low * k, close=g.close_adj))
    g.index = utc_to_server(g.index)
    g = g[~g.index.duplicated(keep="last")].sort_index()
    return g.resample(rule).agg(dict(open="first", high="max", low="min", close="last")).dropna()


def run(df, fc, cost, ann, hl, start, fin=0.0):
    T.EVAL_START = start
    r = T.engine_bp(df, fc, cost, ann, hl)
    if fin:
        ret = df.close.pct_change()
        vol = ret.ewm(halflife=hl, min_periods=20).std() * np.sqrt(ann)
        lev = (fc.clip(-2, 2) * (T.TARGET_VOL / vol)).clip(-T.MAX_LEV, T.MAX_LEV).shift(1)
        r = r - lev.resample("D").mean().reindex(r.index).fillna(0).clip(lower=0) * fin / 365
    return r


def tstat(a: pd.Series, b: pd.Series) -> float:
    j = a.index.intersection(b.index)
    a, b = a[j], b[j]
    d = a * (b.std() / a.std()) - b
    return float(d.mean() / d.std() * np.sqrt(len(d)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    W0 = "2011-06-01"
    nq = bars("NQ", "4h")
    fc = champion_signal(nq.close)
    S = {"CHAMPION NQ H4": run(nq, fc, 1.0, 1512, 42, W0),
         "CHAMPION NQ H4 +swap": run(nq, fc, 1.0, 1512, 42, W0, EXTRA_SWAP),
         "CONSTANT LONG same vol": run(nq, pd.Series(1.0, index=nq.index), 1.0, 1512, 42, W0),
         "NQ buy & hold": nq.close.resample("D").last().dropna().pct_change().loc[W0:].dropna()}
    eras = [(W0, "2014-12-31", "2011-14"), ("2015-01-01", "2019-12-31", "2015-19"),
            ("2020-01-01", "2021-12-31", "2020-21"), ("2022-01-01", "2022-12-31", "2022 BEAR (NQ -33%)"),
            ("2023-01-01", "2026-12-31", "2023-26"), (W0, "2026-12-31", "FULL 2011-26")]
    L = ["1. CHAMPION RECIPE ON NASDAQ-100 FUTURES (H4, server clock)"]
    for a, b, name in eras:
        L.append(f"\n{name}\n  {'':26} {'total':>8} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7}")
        for k, r in S.items():
            s = stats(r.loc[a:b])
            L.append(f"  {k:26} {s['total']:+7.1f}% {s['cagr']:+6.1f}% {s['sharpe']:+7.2f} {s['maxdd']:+6.1f}%")
    ch, cl = S["CHAMPION NQ H4"], S["CONSTANT LONG same vol"]
    L.append("\ntiming vs constant long at matched vol: " + ", ".join(
        f"{n} t {tstat(ch.loc[a:b], cl.loc[a:b]):+.2f}" for a, b, n in eras))

    # ---------------- 2. book
    W1 = "2019-01-01"
    gc = bars("GC", "4h")
    btc = bars("BTC", "D")
    sl = {"XAU": run(gc, champion_signal(gc.close), 1.0, 1512, 42, W1),
          "BTC": run(btc, champion_recipe(btc.close, 1 / 6, 1.5), 5.0, 252, 7, W1),
          "NDX": run(nq, fc, 1.0, 1512, 42, W1, EXTRA_SWAP)}
    D = pd.DataFrame(sl).fillna(0.0)
    D = D[D.index.dayofweek < 5]
    books = {"XAU+BTC (live book)": D[["XAU", "BTC"]].mean(axis=1),
             "XAU+BTC+NDX": D.mean(axis=1), "XAU+NDX": D[["XAU", "NDX"]].mean(axis=1),
             "XAU alone": D.XAU, "NDX alone": D.NDX}
    L.append(f"\n\n2. BOOK, equal-risk sleeves, {W1}..  (NDX sleeve charged the CFD's extra swap)")
    L.append("sleeve correlations:\n" + D.corr().round(2).to_string())
    base = books["XAU+BTC (live book)"]
    mid = D.index[len(D) // 2]
    L.append(f"\n  {'':22} {'Sharpe':>7} {'CAGR':>7} {'maxDD':>7} {'SR h1':>6} {'SR h2':>6} {'t vs live @vol':>15}")
    for k, r in books.items():
        s = stats(r)
        h1, h2 = stats(r.loc[:mid]), stats(r.loc[mid:])
        t = tstat(r, base) if k != "XAU+BTC (live book)" else 0.0
        L.append(f"  {k:22} {s['sharpe']:+7.2f} {s['cagr']:+6.1f}% {s['maxdd']:+6.1f}% {h1['sharpe']:+6.2f} {h2['sharpe']:+6.2f} {t:+15.2f}")
    txt = "\n".join(L) + "\n"
    print(txt)
    (OUT / "report.txt").write_text(txt)


if __name__ == "__main__" and "--hold" not in sys.argv:
    sys.exit(main())


def hold_sleeve(start: str = "2019-01-01") -> pd.Series:
    """NDX as a plain LONG HOLD at ~10% vol: daily B&H return x (0.10 / trailing 1y vol, lagged),
    leverage capped at 1.5, minus the CFD's extra swap. No timing at all."""
    d = bars("NQ", "D").close.pct_change().dropna()
    vol = d.rolling(252, min_periods=120).std().shift(1) * np.sqrt(252)
    lev = (0.10 / vol).clip(upper=1.5)
    r = (lev * d - lev * EXTRA_SWAP / 252).loc[start:].dropna()
    return r[r.index.dayofweek < 5]


def extra() -> None:
    W1 = "2019-01-01"
    gc, btc = bars("GC", "4h"), bars("BTC", "D")
    D = pd.DataFrame({"XAU": run(gc, champion_signal(gc.close), 1.0, 1512, 42, W1),
                      "BTC": run(btc, champion_recipe(btc.close, 1 / 6, 1.5), 5.0, 252, 7, W1),
                      "NDXhold": hold_sleeve(W1)}).fillna(0.0)
    D = D[D.index.dayofweek < 5]
    base = D[["XAU", "BTC"]].mean(axis=1)
    mid = D.index[len(D) // 2]
    L = ["\n3. NDX as a plain long HOLD sleeve (~10% vol, no timing, CFD swap charged)",
         "correlations:\n" + D.corr().round(2).to_string()]
    for k, r in {"XAU+BTC (live book)": base, "XAU+BTC+NDXhold": D.mean(axis=1), "NDXhold alone": D.NDXhold}.items():
        s = stats(r)
        L.append(f"  {k:22} SR {s['sharpe']:+.2f} CAGR {s['cagr']:+.1f}% maxDD {s['maxdd']:+.1f}% | h1 "
                 f"{stats(r.loc[:mid])['sharpe']:+.2f} h2 {stats(r.loc[mid:])['sharpe']:+.2f} | t vs live @vol "
                 f"{tstat(r, base) if k != 'XAU+BTC (live book)' else 0:+.2f}")
    txt = "\n".join(L) + "\n"
    print(txt)
    with open(OUT / "report.txt", "a") as f:
        f.write(txt)


if __name__ == "__main__" and "--hold" in sys.argv:
    extra()
