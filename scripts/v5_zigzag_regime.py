#!/usr/bin/env python3
"""Regime-adaptive ZigZag (user request 2026-11-10): adapt when the market turns the other way.

PRE-REGISTERED (8 variants, nothing else is tried):
  V0    baseline: long on every bottom fire (variant B)
  L200  long only when gold's previous daily close > its 200-day SMA
  LEMA  long only when the 50-day EMA (previous day) is above its value 10 days earlier
  LEQ   long only when the last 20 ungated V0 trades closed before the fire sum to > 0 R
  S200  SHORT only: top fires when previous close < 200-day SMA
  LS200 long bottoms above the 200-day SMA, short tops below it (one position at a time)
  LSEMA same, regime = 50-day EMA slope
  LSEQ  long bottoms while the long equity filter is on, short tops while it is off
TOP detector = the identical live model trained on MIRRORED prices (p' = C/p, high' = C/low, ...), so a
swing high of the real series is a swing low of the mirror; walk-forward, pooling and purge unchanged.
Short trade: sell next open - spread, stop 3 ATR above, trail 3 ATR above the lowest low, TP 4.3R,
hold 120. Short swap: 0 on CME (carry is in the back-adjustment), -1%/yr on Switch (conservative).
Regime inputs use only the PREVIOUS day's close (no same-day look-ahead).
Per variant: Sharpe, halves, 2013-15 bear, random-timing null (raw fires shifted, regime mask held
fixed), paired t of daily returns vs V0. Family: max over variants vs max over the shifted replays.

    python scripts/v5_zigzag_regime.py cme | switch
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_zigzag_long_history import (MAX_HOLD, OUT, RISK, SPREAD, SRC, THR, TP_R, TRAIL,  # noqa: E402
                                            load, sharpe, walk_forward)
from scripts.v5_zigzag_signal import LEAK_MAX, atr, intrabar_leak  # noqa: E402

N_NULL = 200
SHORT_SWAP = {"cme": 0.0, "switch": -0.01}


def mirror(d: pd.DataFrame) -> pd.DataFrame:
    c = float(d.close.iloc[0]) ** 2
    return pd.DataFrame({"open": c / d.open, "high": c / d.low, "low": c / d.high, "close": c / d.close},
                        index=d.index)


def bt(g, fl, fs, start, swl, sws, spread=SPREAD):
    """One position at a time; long on fl, short on fs (long wins a tie)."""
    o, h, l = (g[x].to_numpy() for x in ("open", "high", "low"))
    a, t, n = atr(g).to_numpy(), g.index, len(g)
    ok = (t >= start) & np.isfinite(a)
    fl, fs = fl & ok, fs & ok
    trades, i = [], 0
    while i < n - 1:
        side = 1 if fl[i] else (-1 if fs[i] else 0)
        if side == 0:
            i += 1
            continue
        s = i + 1
        risk = TRAIL * a[i]
        e = o[s] + side * spread
        sl, ext = e - side * risk, e
        tp = e + side * TP_R * risk
        j, px = n - 1, o[-1]
        for j in range(s, n):
            if (side == 1 and l[j] <= sl) or (side == -1 and h[j] >= sl):
                px = min(sl, o[j]) if side == 1 else max(sl, o[j])
                break
            if (side == 1 and h[j] >= tp) or (side == -1 and l[j] <= tp):
                px = max(tp, o[j]) if side == 1 else min(tp, o[j])
                break
            if j - s >= MAX_HOLD:
                px = o[j]
                break
            if side == 1:
                ext = max(ext, h[j])
                sl = max(sl, ext - TRAIL * a[j])
            else:
                ext = min(ext, l[j])
                sl = min(sl, ext + TRAIL * a[j])
        days = (t[j] - t[s]).total_seconds() / 86400
        sw = swl if side == 1 else sws
        trades.append((t[s], t[j], side, (side * (px - e) + e * sw * days / 365) / risk))
        i = j
    tr = pd.DataFrame(trades, columns=["t_in", "t_out", "side", "R"])
    days = pd.date_range(pd.Timestamp(start).normalize(), t[-1].normalize(), freq="D")
    ret = pd.Series(0.0, index=days)
    for r in tr.itertuples():
        ret[r.t_out.normalize()] += RISK * r.R
    return tr, ret[ret.index.dayofweek < 5]


def daily_regimes(g: pd.DataFrame) -> dict:
    dc = g.close.resample("D").last().dropna()
    up200 = (dc > dc.rolling(200).mean()).shift(1)            # known at the start of the next day
    e50 = dc.ewm(span=50, adjust=False).mean()
    upema = (e50 > e50.shift(10)).shift(1)
    day = g.index.normalize()
    return {k: v.reindex(day, method="ffill").fillna(False).to_numpy(bool)
            for k, v in (("200", up200), ("EMA", upema))}


def equity_gate(g, tr0: pd.DataFrame, n=20) -> np.ndarray:
    """True at bar t when the last n V0 trades that CLOSED before t sum to > 0 R."""
    t_out = tr0.t_out.to_numpy()
    cs = np.concatenate([[0.0], np.cumsum(tr0.R.to_numpy())])
    k = np.searchsorted(t_out, g.index.to_numpy(), side="left")   # trades closed strictly before t
    s = cs[k] - cs[np.maximum(k - n, 0)]
    return (k >= n) & (s > 0)


def variants(g, fb, ft, start, swl, sws, reg):
    tr0, _ = bt(g, fb, np.zeros_like(fb), start, swl, sws)
    eq = equity_gate(g, tr0)
    z = np.zeros_like(fb)
    return {
        "V0": (fb, z), "L200": (fb & reg["200"], z), "LEMA": (fb & reg["EMA"], z), "LEQ": (fb & eq, z),
        "S200": (z, ft & ~reg["200"]), "LS200": (fb & reg["200"], ft & ~reg["200"]),
        "LSEMA": (fb & reg["EMA"], ft & ~reg["EMA"]), "LSEQ": (fb & eq, ft & ~eq),
    }


def main() -> None:
    src = sys.argv[1]
    cfg = SRC[src]
    start = pd.Timestamp(cfg["start"])
    out = OUT / src
    gold = load(src, cfg["gold"])
    aux = {s: load(src, s) for s in cfg["aux"]}
    aux = {s: d for s, d in aux.items() if intrabar_leak(d) <= LEAK_MAX}
    pb = pd.read_csv(out / "probs.csv", index_col=0, parse_dates=True).iloc[:, 0]
    cache = out / "probs_top.csv"
    if cache.exists():
        pt = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        pt = walk_forward(mirror(gold), {k: mirror(v) for k, v in aux.items()}, start)
        pt.rename("prob_top").to_csv(cache)
    b, tp_ = (x.reindex(gold.index).to_numpy() for x in (pb, pt))
    fb, ft = np.isfinite(b) & (b >= THR), np.isfinite(tp_) & (tp_ >= THR)
    swl, sws = cfg["swap"], SHORT_SWAP[src]
    reg = daily_regimes(gold)

    V = variants(gold, fb, ft, start, swl, sws, reg)
    res = {k: bt(gold, fl, fs, start, swl, sws) for k, (fl, fs) in V.items()}
    s0 = int((gold.index < start).sum())
    rng = np.random.default_rng(13)
    shifts = rng.integers(500, len(gold) - s0 - 500, N_NULL)
    null = {k: [] for k in V}
    for sh in shifts:
        rb, rt = fb.copy(), ft.copy()
        rb[s0:], rt[s0:] = np.roll(fb[s0:], int(sh)), np.roll(ft[s0:], int(sh))
        for k, (fl, fs) in variants(gold, rb, rt, start, swl, sws, reg).items():
            null[k].append(sharpe(bt(gold, fl, fs, start, swl, sws)[1]))
    nullm = np.array([null[k] for k in V])                     # variants x draws
    base = res["V0"][1]
    mid = start + (gold.index[-1] - start) / 2
    rows = []
    for k, (tr, ret) in res.items():
        d = (ret - base).to_numpy()
        s = sharpe(ret)
        rows.append(dict(
            variant=k, trades=len(tr), per_wk=round(len(tr) / ((gold.index[-1] - start).days / 7), 2),
            short_pct=round(100 * (tr.side < 0).mean(), 0) if len(tr) else 0, win=round(100 * (tr.R > 0).mean(), 1),
            SR=round(s, 2), h1=round(sharpe(ret[:mid]), 2), h2=round(sharpe(ret[mid:]), 2),
            bear_13_15=round(sharpe(ret["2013":"2015"]), 2) if src == "cme" else None,
            y24_26=round(sharpe(ret["2024":]), 2),
            null_med=round(float(np.median(null[k])), 2), p=round(float((np.array(null[k]) >= s).mean()), 3),
            t_vs_V0=round(d.mean() / d.std() * np.sqrt(len(d)), 2) if d.std() > 0 else 0.0,
            SR_2x=round(sharpe(bt(gold, *V[k], start, swl, sws, 2 * SPREAD)[1]), 2)))
    A = pd.DataFrame(rows)
    obs = A.SR.max()
    mx = nullm.max(axis=0)
    gd = gold[gold.index >= start].close.resample("D").last().dropna()
    gr = gd.pct_change().dropna() + swl / 252
    gr = gr[gr.index.dayofweek < 5]
    yr = pd.DataFrame({k: r.groupby(r.index.year).apply(sharpe) for k, (_, r) in res.items()})
    yr["gold_%"] = gd.groupby(gd.index.year).apply(lambda x: 100 * (x.iloc[-1] / x.iloc[0] - 1))
    txt = (f"Regime-adaptive ZigZag, {src.upper()} gold H1, test {start.date()} -> {gold.index[-1]}; "
           f"top-detector fire rate {100 * ft[s0:].mean():.1f}% vs bottom {100 * fb[s0:].mean():.1f}%\n"
           f"hold gold same days: Sharpe {sharpe(gr):+.2f}\n\n" + A.to_string(index=False)
           + f"\n\nfamily max-stat: best {A.variant[A.SR.idxmax()]} {obs:+.2f} vs max-over-variants of random "
           f"timing: median {np.median(mx):+.2f}, 95th {np.percentile(mx, 95):+.2f}, p {(mx >= obs).mean():.3f}\n\n"
           "Sharpe by year:\n" + yr.round(2).to_string() + "\n")
    print(txt)
    (out / "regime_report.txt").write_text(txt)


if __name__ == "__main__":
    sys.exit(main())
