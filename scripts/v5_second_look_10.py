#!/usr/bin/env python3
"""Second look #10: (a) G10 FX CARRY (§3ah "open", never run) and (b) the CRYPTO TREND panel vs each
coin's own hold (§3bp left BTC trend "unproven" only because DSR cannot judge agreeing cells).
Pre-registered addendum in data/v5_runs/second_look/PREREG.md.

    python scripts/v5_second_look_10.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from v5_account_chooser_2026_10 import xau_btc_book  # noqa: E402
from v5_second_look_1 import matched_t  # noqa: E402
from v5_xau_champion_lifts import champion_recipe  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "second_look" / "10"
PAIRS = {"EUR": ("EURUSD", +1), "GBP": ("GBPUSD", +1), "AUD": ("AUDUSD", +1), "NZD": ("NZDUSD", +1),
         "JPY": ("USDJPY", -1), "CAD": ("USDCAD", -1), "CHF": ("USDCHF", -1)}
COINS = ("BTC", "ETH", "SOL", "LTC", "AVA")


def d1(name: str) -> pd.Series:
    s = pd.read_csv(ROOT / f"data/{name}_D1_long.csv", parse_dates=["time"], index_col="time")["close"]
    return s[~s.index.duplicated(keep="last")].sort_index().astype(float)


def sr(x, ann=252):
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std() * np.sqrt(ann)) if x.std() > 0 else 0.0


# ------------------------------------------------------------------ 10a FX carry
def fx_carry() -> pd.Series:
    rates = pd.read_csv(ROOT / "data/rates_3m.csv", parse_dates=["date"], index_col="date").ffill() / 100.0
    rates.index = rates.index + pd.offsets.MonthBegin(1)               # usable from the following month
    px = pd.DataFrame({c: d1(p) for c, (p, _) in PAIRS.items()}).dropna()
    px = px[px.index >= "2007-06-01"]
    # currency-vs-USD return: +1 for XXXUSD, inverse for USDXXX
    ret = pd.DataFrame({c: (px[c].pct_change() if sgn > 0 else (1 / px[c]).pct_change())
                        for c, (_, sgn) in PAIRS.items()})
    ret["USD"] = 0.0
    R = rates.reindex(ret.index, method="ffill")[list(ret.columns)]
    month = ret.index.to_period("M")
    pos = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    for m in month.unique():
        rows = month == m
        r0 = R[rows].iloc[0].dropna()
        if len(r0) < 6:
            continue
        rk = r0.sort_values()
        w = pd.Series(0.0, index=ret.columns)
        w[rk.index[-3:]] = 1.0 / 3
        w[rk.index[:3]] = -1.0 / 3
        pos.loc[rows] = w.values
    usd = R["USD"]
    carry = pos.mul(R.sub(usd, axis=0).fillna(0.0)).sum(axis=1) / 252.0
    markup = pos.abs().sum(axis=1) * 0.01 / 252.0
    turn = pos.diff().abs().sum(axis=1).fillna(0.0)
    gross = (pos.shift(1) * ret).sum(axis=1) + carry.shift(1)
    net = gross - markup.shift(1) - turn * 2e-4
    vol = net.rolling(60, min_periods=20).std().shift(1) * np.sqrt(252)
    out = (net * (0.10 / vol)).dropna()
    return out[out.index >= "2008-02-01"]


# ------------------------------------------------------------------ 10b crypto trend
def crypto_leg(close: pd.Series, trend: bool) -> pd.Series:
    bpy = len(close) / max((close.index[-1] - close.index[0]).days / 365.25, 1e-9)
    ret = close.pct_change()
    vol = ret.ewm(halflife=42, min_periods=20).std() * np.sqrt(bpy)
    fc = champion_recipe(close, 1 / 6, 1.5).clip(0, 2) if trend else pd.Series(1.0, index=close.index)
    pos = (fc * (0.10 / vol)).clip(0, 8).shift(1)
    r = pos * ret - pos.diff().abs().fillna(0.0) * (4.5e-4 * 1.5) - pos.abs() * 0.15 / bpy
    return r.dropna()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    book = xau_btc_book()
    lines = []

    def say(s=""):
        print(s, flush=True); lines.append(s)

    # 10a
    fx = fx_carry()
    j = fx.index.intersection(book.index)
    f, b = fx.reindex(j).fillna(0), book.reindex(j).fillna(0)
    comb = 0.5 * f * (b.std() / f.std()) + 0.5 * b
    t2 = matched_t(comb, b)[0]
    mid = j[len(j) // 2]
    h1, h2 = matched_t(comb[:mid], b[:mid])[0], matched_t(comb[mid:], b[mid:])[0]
    say(f"10a G10 FX CARRY {fx.index[0].date()} -> {fx.index[-1].date()}: standalone net SR {sr(fx):+.3f} "
        f"(halves {sr(fx[:len(fx) // 2]):+.2f}/{sr(fx[len(fx) // 2:]):+.2f})")
    say(f"    as a sleeve on the XAU+BTC book ({j[0].date()}+): book {sr(b):+.3f} -> 50/50 {sr(comb):+.3f}  "
        f"T2 t {t2:+.2f} halves {h1:+.2f}/{h2:+.2f}  corr {np.corrcoef(f, b)[0, 1]:+.2f}")
    ok_a = sr(fx) > 0 and t2 >= 2 and h1 > 0 and h2 > 0
    say(f"    VERDICT 10a: {'REVIVED' if ok_a else 'not revived'}")

    # 10b
    diffs, per = [], []
    trends = {}
    for c in COINS:
        cl = d1(c)
        tr, hd = crypto_leg(cl, True), crypto_leg(cl, False)
        jj = tr.index.intersection(hd.index)
        tr, hd = tr.reindex(jj), hd.reindex(jj)
        trends[c] = tr
        tm = tr * (hd.std() / tr.std())
        d = (tm - hd).rename(c)
        diffs.append(d)
        t_c = d.mean() / d.std() * np.sqrt(len(d))
        per.append((c, sr(tr, 365), sr(hd, 365), t_c, jj[0].date()))
        say(f"10b {c}: from {jj[0].date()}  trend SR {sr(tr, 365):+.3f}  hold SR {sr(hd, 365):+.3f}  matched-vol t {t_c:+.2f}")
    D = pd.concat(diffs, axis=1)
    pooled = D.mean(axis=1, skipna=True).dropna()
    tp = pooled.mean() / pooled.std() * np.sqrt(len(pooled))
    ph = len(pooled) // 2
    t_h1 = pooled[:ph].mean() / pooled[:ph].std() * np.sqrt(ph)
    t_h2 = pooled[ph:].mean() / pooled[ph:].std() * np.sqrt(len(pooled) - ph)
    npos = sum(1 for p in per if p[3] > 0)
    say(f"    POOLED trend-minus-hold (matched vol): t {tp:+.2f}  halves {t_h1:+.2f}/{t_h2:+.2f}  coins positive {npos}/5")
    # T_b: 5-coin sleeve vs BTC-only sleeve inside the book (reported, not a pass condition)
    sleeve5 = pd.concat(trends.values(), axis=1).mean(axis=1, skipna=True)
    btc = trends["BTC"]
    jb = sleeve5.index.intersection(btc.index)
    s5, s1 = sleeve5.reindex(jb), btc.reindex(jb)
    tb = matched_t(s5, s1)[0]
    say(f"    T_b 5-coin trend sleeve SR {sr(s5, 365):+.3f} vs BTC-only {sr(s1, 365):+.3f}: matched t {tb:+.2f} (reported only)")
    ok_b = tp >= 2 and npos >= 4 and t_h1 > 0 and t_h2 > 0
    say(f"    VERDICT 10b: {'REVIVED' if ok_b else 'not revived'}")
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    pd.DataFrame(per, columns=["coin", "trend_sr", "hold_sr", "t", "start"]).to_csv(OUT / "crypto_cells.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
