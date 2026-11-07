#!/usr/bin/env python3
"""Footprint screen (§3do part A): does AGGRESSOR FLOW predict returns beyond PRICE?

Assets: XAU = Switch XAUUSDs H1 prices with PAXGUSDT footprint (UTC hours mapped to the Switch
server clock, NY+7); BTC = Binance BTCUSDT H1 with its own footprint.
Features at the close of bar i (pre-registered, 8):
  delta     (2tb - v)/v of bar i
  cvd6/24   cumulative delta over 6 / 24 bars, / volume
  div24     z(ret24) - z(cvd24): price up while flow down -> negative
  poc       point-of-control position in the bar (0 low .. 1 high)
  buy_lo    taker-buy share of volume traded in the bar's bottom 25% (absorption at the low)
  buy_hi    same in the top 25%
  vol_z     log-volume z vs trailing 240 bars
Target: open[i+1] -> close[i+h], h in {1, 6, 24}.
Reported per cell: raw Spearman IC, PARTIAL IC (feature residualised on the bar's own return and
ret6/ret24 — flow is mostly a restatement of price), t on non-overlapping h-spaced rows, the sign in
each half, and EV: top-decile forward return minus the all-bar mean (bp) beside the round-trip cost.
Pass bar (48 cells): |partial t| >= 3.3 (Bonferroni 0.05/48) AND same sign in both halves AND
top-decile EV above cost.

    python scripts/v5_footprint_screen.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v5_runs" / "footprint"
COST_BP = {"XAU": 0.25 / 4000 * 1e4 * 2, "BTC": 10.0}   # XAU $0.25 x2 sides; BTC ~5bp/side (Switch)
FEATS = ["delta", "cvd6", "cvd24", "div24", "poc", "buy_lo", "buy_hi", "vol_z"]
HS = (1, 6, 24)


def load_fp(sym: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / f"data/binance_1m/{sym}_fp_H1.csv.gz", parse_dates=["time"], index_col="time")


def utc_to_server(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Switch/Darwinex server clock = New York wall time + 7h (memory: mt5-server-clock)."""
    return (idx.tz_localize("UTC").tz_convert("America/New_York").tz_localize(None)
            + pd.Timedelta(hours=7))


def features(fp: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    d = 2 * fp.tb - fp.v
    f = pd.DataFrame(index=fp.index)
    f["delta"] = d / fp.v
    for n in (6, 24):
        f[f"cvd{n}"] = d.rolling(n).sum() / fp.v.rolling(n).sum()
    r24 = np.log(close).diff(24)
    z = lambda x: (x - x.rolling(240).mean()) / x.rolling(240).std()  # noqa: E731
    f["div24"] = z(f.cvd24) - z(r24)
    f["poc"] = fp.poc
    f["buy_lo"] = fp.lo_tb / fp.lo_v
    f["buy_hi"] = fp.hi_tb / fp.hi_v
    lv = np.log(fp.v.clip(lower=1e-9))
    f["vol_z"] = (lv - lv.rolling(240).mean()) / lv.rolling(240).std()
    return f


def assets() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    out = {}
    g = pd.read_csv(ROOT / "data/v5_runs/zigzag_live_wf/XAUUSDs_H1.csv", parse_dates=["time"], index_col="time")
    g = g[~g.index.duplicated()].sort_index()
    p = load_fp("PAXGUSDT")
    p.index = utc_to_server(p.index)
    p = p[~p.index.duplicated()]
    # alignment control: PAXG vs XAU hourly returns must peak at lag 0
    j = g.index.intersection(p.index)
    gr, pr = g.close.reindex(j).pct_change(), p.c.reindex(j).pct_change()
    lags = {k: round(gr.corr(pr.shift(k)), 3) for k in range(-2, 3)}
    print("PAXG vs Switch XAU hourly-return corr by lag (must peak at 0):", lags)
    assert max(lags, key=lags.get) == 0, "clock misaligned"
    out["XAU"] = (g.loc[j], features(p.reindex(j), g.close.reindex(j)))
    b = load_fp("BTCUSDT")
    px = b[["o", "h", "l", "c"]].rename(columns=dict(o="open", h="high", l="low", c="close"))
    out["BTC"] = (px, features(b, b.c))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (px, F) in assets().items():
        o, c = px.open, px.close
        r1 = np.log(c).diff()
        ctrl = pd.concat([r1, np.log(c).diff(6), np.log(c).diff(24)], axis=1)
        for h in HS:
            fwd = np.log(c.shift(-h) / o.shift(-1)) * 1e4               # bp, enter next open
            for f in FEATS:
                df = pd.concat([F[f], fwd, ctrl], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
                df.columns = ["x", "y", "a", "b", "c"]
                A = np.column_stack([np.ones(len(df)), df[["a", "b", "c"]].values])
                res = df.x - A @ np.linalg.lstsq(A, df.x.values, rcond=None)[0]
                sub = np.arange(0, len(df), h)                           # non-overlapping rows
                ic = spearmanr(df.x, df.y)[0]
                pic = spearmanr(res.iloc[sub], df.y.iloc[sub])[0]
                t = pic * np.sqrt(len(sub) - 2) / np.sqrt(max(1e-12, 1 - pic ** 2))
                half = len(df) // 2
                s1 = spearmanr(res.iloc[:half], df.y.iloc[:half])[0]
                s2 = spearmanr(res.iloc[half:], df.y.iloc[half:])[0]
                q = res.rank(pct=True)
                top, bot = df.y[q >= 0.9].mean(), df.y[q <= 0.1].mean()
                ev = max(top, -bot) - df.y.mean() if np.sign(pic) >= 0 else -bot - (-df.y.mean())
                ev = (top - df.y.mean()) if pic >= 0 else (df.y.mean() - bot)
                ok = abs(t) >= 3.3 and np.sign(s1) == np.sign(s2) == np.sign(pic) and ev > COST_BP[name]
                rows.append(dict(asset=name, h=h, feat=f, n=len(df), ic=round(ic, 4), partial_ic=round(pic, 4),
                                 t=round(t, 2), half1=round(s1, 4), half2=round(s2, 4),
                                 ev_top_bp=round(ev, 2), cost_bp=round(COST_BP[name], 2), PASS=ok))
                print(rows[-1], flush=True)
    R = pd.DataFrame(rows)
    txt = (R.to_string(index=False) + f"\n\nPASS cells: {int(R.PASS.sum())} / {len(R)}\n"
           "partial_ic = Spearman of the feature RESIDUALISED on the bar's own return, ret6 and ret24,\n"
           "on non-overlapping rows; ev_top_bp = mean forward bp of the feature's favourable decile minus\n"
           "the all-bar mean, entry at the next open, before costs.\n")
    print(txt)
    (OUT / "screen.txt").write_text(txt)


if __name__ == "__main__":
    sys.exit(main())
