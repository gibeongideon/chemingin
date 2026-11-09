#!/usr/bin/env python3
"""BTC-SPECIFIC ZigZag (§3dv, user request 2026-11-09): redesign the gold ZigZag for Bitcoin, incl.
gold (XAUUSD) features as inputs to the BTC model.

DATA: Binance BTCUSDT hourly (24/7, real exchange; from the footprint build), 2017-08..2026-10.
Costs = Switch BTCUSDs live 2026-11-09: spread 4.33bp -> 6bp round trip incl. slippage; LONG SWAP
-15%/yr of notional, charged per hour held.
MODEL VARIANTS (live ZigZag features/label/HistGB, quarterly walk-forward from 2020-01, 20k trailing bars):
  M1 BTC-only H1, zigzag theta 1.5 ATR (live label)
  M2 BTC-only H1, theta 2.5 ATR (BTC dips are deeper -> fewer, larger swings)
  M3 BTC+ETH pooled H1, theta 1.5
  M4 BTC-only H1 + GOLD features (CME GC hourly, UTC: ret 1/6/24h, z20, vol ratio, gold-closed flag;
     forward-filled <=72h across gold's weekend)
  M5 BTC-only H4, theta 1.5
RULE GRID (96 per model): thr .55/.60/.65/.70 x trail k 3/4/6 ATR x TP none/4.3R x max hold 120/336 h
  x gate none / BTC champion forecast >= 0.5 (D1 recipe, last closed day).
PROTOCOL: DISCOVERY 2020-01..2022-12 picks ONE (model, rule) by Sharpe among all 480; HOLDOUT
2023-01..2026-10 is read ONCE for that pick and for the best rule of each model (reported, not chosen),
against a 200-shift random-timing null, BTC buy & hold, and the live gold ZigZag rule on BTC.

    python scripts/v5_zigzag_btc.py probs      # stage 1 (writes data/v5_runs/zigzag_btc/probs_*.csv.gz)
    python scripts/v5_zigzag_btc.py study      # stage 2
"""
from __future__ import annotations

import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.v5_zigzag_signal as ZS  # noqa: E402
from scripts.v5_zigzag_signal import MIN_TRAIN, TOL, TRAIN_BARS, atr  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "zigzag_btc"
START, DISC_END, HOLD_START = pd.Timestamp("2020-01-01"), "2022-12-31", "2023-01-01"
COST, SWAP_YR, RISK = 6e-4, 0.15, 0.0025
MODELS = ("M1", "M2", "M3", "M4", "M5")


def binance_h1(sym: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / f"data/binance_1m/{sym}_fp_H1.csv.gz", parse_dates=["time"], index_col="time")
    d = d.rename(columns=dict(o="open", h="high", l="low", c="close"))[["open", "high", "low", "close"]]
    return d[~d.index.duplicated()].sort_index().dropna()


def to_h4(d: pd.DataFrame) -> pd.DataFrame:
    return d.resample("4h").agg(dict(open="first", high="max", low="min", close="last")).dropna()


def gold_feats(idx: pd.DatetimeIndex) -> pd.DataFrame:
    g = pd.read_csv(ROOT / "data/databento/ohlcv_1h/GC.csv.gz", parse_dates=["time"], index_col="time")
    c = g.close_adj[~g.index.duplicated()]
    lr = np.log(c)
    f = pd.DataFrame({"g_r1": lr.diff(), "g_r6": lr.diff(6), "g_r24": lr.diff(24),
                      "g_z20": (c - c.rolling(20).mean()) / c.rolling(20).std(),
                      "g_volr": lr.diff().rolling(24).std() / lr.diff().rolling(240).std()})
    # a gold bar stamped t (UTC, bar open) is known at t+1h -> shift by one hour before joining
    f.index = f.index + pd.Timedelta(hours=1)
    out = f.reindex(idx, method="ffill", limit=72)
    out["g_closed"] = (~idx.isin(f.index)).astype(float)
    return out


def frame(d: pd.DataFrame, theta: float, extra: pd.DataFrame | None = None):
    old = ZS.ZZ_THETA
    ZS.ZZ_THETA = theta
    try:
        F, y, ok = ZS._frame(d)
    finally:
        ZS.ZZ_THETA = old
    if extra is not None:
        F = pd.concat([F, extra.reindex(F.index)], axis=1)          # NaNs allowed (HistGB)
    return F, y, ok


def wf_probs(series: dict, theta: float, extras: dict | None = None) -> pd.Series:
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415
    extras = extras or {}
    full = {k: frame(d, theta, extras.get(k)) for k, d in series.items()}
    btc = series["BTC"]
    out = pd.Series(np.nan, index=btc.index)
    cuts = pd.date_range(START, btc.index[-1] + pd.offsets.QuarterBegin(1), freq="QS")
    kw = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0)
    for c0, c1 in zip(cuts[:-1], cuts[1:]):
        X, Y = [], []
        for k, d in series.items():
            n = int((d.index < c0).sum())
            if n < MIN_TRAIN:
                continue
            F, y, ok = frame(d.iloc[:n], theta, extras.get(k))
            hi = n - (TOL + 1)
            lo = max(0, hi - TRAIN_BARS)
            m = np.zeros(n, bool)
            m[lo:hi] = True
            m &= ok
            X.append(F.values[m])
            Y.append(y[m])
        clf = HistGradientBoostingClassifier(**kw).fit(np.vstack(X), np.concatenate(Y))
        Fa, _, oka = full["BTC"]
        rows = (btc.index >= c0) & (btc.index < c1) & oka
        if rows.any():
            out[rows] = clf.predict_proba(Fa.values[rows])[:, 1]
    return out[out.index >= START]


def stage_probs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    btc, eth = binance_h1("BTCUSDT"), binance_h1("ETHUSDT")
    specs = {"M1": (dict(BTC=btc), 1.5, None), "M2": (dict(BTC=btc), 2.5, None),
             "M3": (dict(BTC=btc, ETH=eth), 1.5, None),
             "M4": (dict(BTC=btc), 1.5, dict(BTC=gold_feats(btc.index))),
             "M5": (dict(BTC=to_h4(btc)), 1.5, None)}
    for name, (series, theta, extras) in specs.items():
        t0 = time.time()
        p = wf_probs(series, theta, extras)
        p.to_frame("p").to_csv(OUT / f"probs_{name}.csv.gz", float_format="%.4f")
        print(f"{name}: {p.notna().sum():,} probs, mean {p.mean():.3f}, {time.time() - t0:.0f}s", flush=True)


@njit(cache=True)
def sim(o, h, l, a, fire, k, tp_r, hold_bars, cost, swap_per_bar):
    n = len(o)
    ex_i = np.full(n, -1)
    R = np.zeros(n)
    i = 0
    while i < n - 1:
        if fire[i] and np.isfinite(a[i]) and a[i] > 0:
            s = i + 1
            e = o[s] * (1 + cost)
            risk = k * a[i]
            sl, best = e - risk, e
            tp = e + tp_r * risk if tp_r > 0 else np.inf
            j, out = s, np.nan
            while j < n:
                if l[j] <= sl:
                    out = min(sl, o[j])
                    break
                if h[j] >= tp:
                    out = max(tp, o[j])
                    break
                if j - s >= hold_bars:
                    out = o[j]
                    break
                best = max(best, h[j])
                if np.isfinite(a[j]):
                    sl = max(sl, best - k * a[j])
                j += 1
            if j >= n:
                break
            ex_i[i] = j
            R[i] = (out - e - e * swap_per_bar * (j - s)) / risk
            i = j
        i += 1
    return ex_i, R


class Book:
    def __init__(self, d: pd.DataFrame, p: pd.Series, bar_h: int, gate: np.ndarray):
        self.d, self.bar_h = d, bar_h
        self.o, self.h, self.l = (d[x].to_numpy(float) for x in ("open", "high", "low"))
        self.a = atr(d).to_numpy(float)
        self.p = p.reindex(d.index).to_numpy(float)
        self.gate = gate
        self.days = d.index.normalize()
        self.uday = pd.DatetimeIndex(sorted(set(self.days)))
        self.did = self.uday.get_indexer(self.days)

    def run(self, thr, k, tp, hold_h, gated, fire=None):
        f = (self.p >= thr) & np.isfinite(self.p) if fire is None else fire
        if gated and fire is None:
            f = f & self.gate
        ex, R = sim(self.o, self.h, self.l, self.a, f, k, tp, hold_h // self.bar_h, COST,
                    SWAP_YR * self.bar_h / 8760)
        m = ex >= 0
        ret = np.zeros(len(self.uday))
        np.add.at(ret, self.did[ex[m]], RISK * R[m])
        return pd.Series(ret, index=self.uday), R[m], f


def sharpe(r: pd.Series, a=None, b=None) -> float:
    r = r.loc[a:b]
    return float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else 0.0     # BTC trades 365d


def stage_study() -> None:
    from scripts.v5_xau_champion_lifts import champion_recipe  # noqa: PLC0415
    btc = binance_h1("BTCUSDT")
    d1 = btc.resample("D").agg(dict(open="first", high="max", low="min", close="last")).dropna()
    fc = champion_recipe(d1.close, 1 / 6, 1.5).shift(1)          # last CLOSED day
    books = {}
    for mname in MODELS:
        p = pd.read_csv(OUT / f"probs_{mname}.csv.gz", index_col=0, parse_dates=True)["p"]
        d = to_h4(btc) if mname == "M5" else btc
        d = d.loc[START - pd.Timedelta(days=60):]
        gate = (fc.reindex(d.index.normalize()).to_numpy() >= 0.5)
        books[mname] = Book(d, p, 4 if mname == "M5" else 1, gate)
    grid = list(itertools.product((0.55, 0.60, 0.65, 0.70), (3.0, 4.0, 6.0), (0.0, 4.3), (120, 336), (False, True)))
    rows = []
    for mname, bk in books.items():
        for thr, k, tp, hold, gated in grid:
            ret, R, _ = bk.run(thr, k, tp, hold, gated)
            rows.append(dict(model=mname, thr=thr, k=k, tp=tp, hold=hold, gate=gated,
                             trades=int(len(R)), disc=round(sharpe(ret, str(START.date()), DISC_END), 3),
                             hold_out=round(sharpe(ret, HOLD_START), 3)))
    G = pd.DataFrame(rows)
    pick = G.sort_values("disc", ascending=False).iloc[0]
    rng = np.random.default_rng(9)
    bh = btc.close.resample("D").last().pct_change()

    def detail(r):
        bk = books[r.model]
        ret, R, f = bk.run(r.thr, r.k, r.tp, int(r.hold), bool(r.gate))
        seg = bk.d.index >= HOLD_START
        f0 = f.copy()
        nl = []
        for s in rng.integers(500, len(f0) - 500, 200):
            g = f0.copy()
            g[seg] = np.roll(f0[seg], int(s) % max(1, seg.sum()))
            nl.append(sharpe(bk.run(0, r.k, r.tp, int(r.hold), False, fire=g)[0], HOLD_START))
        ho = sharpe(ret, HOLD_START)
        return ho, float((np.array(nl) >= ho).mean()), len(R)

    L = [f"BTC ZigZag (Binance H1, Switch costs: {COST * 1e4:.0f}bp round trip, swap -{SWAP_YR:.0%}/yr) "
         f"discovery {START.date()}..{DISC_END}, holdout {HOLD_START}..",
         f"BTC buy & hold Sharpe: discovery {sharpe(bh, str(START.date()), DISC_END):.2f}, holdout {sharpe(bh, HOLD_START):.2f}\n"]
    live = G[(G.model == "M1") & (G.thr == 0.60) & (G.k == 3.0) & (G.tp == 4.3) & (G.hold == 120) & (~G.gate)].iloc[0]
    L.append(f"LIVE GOLD RULE on BTC (M1, .60, 3ATR, TP4.3R, 120h): discovery {live.disc:+.2f}, holdout {live.hold_out:+.2f}")
    ho, pn, nt = detail(pick)
    L.append(f"\nPICK (best of {len(G)} on discovery): {pick[['model', 'thr', 'k', 'tp', 'hold', 'gate']].to_dict()}"
             f"\n  discovery Sharpe {pick.disc:+.2f} -> HOLDOUT {ho:+.2f} (random-timing p {pn:.3f})")
    L.append("\nbest rule per model (by discovery), holdout shown for information:")
    for mname in MODELS:
        r = G[G.model == mname].sort_values("disc", ascending=False).iloc[0]
        ho2, p2, _ = detail(r)
        L.append(f"  {mname}: thr {r.thr} k {r.k} tp {r.tp} hold {r.hold} gate {r.gate} | disc {r.disc:+.2f} -> "
                 f"holdout {ho2:+.2f} (p {p2:.3f})")
    L.append("\nmean holdout Sharpe over ALL 96 rules per model (robustness, no selection):")
    L.append("  " + "  ".join(f"{m} {G[G.model == m].hold_out.mean():+.2f}" for m in MODELS))
    L.append("gate effect (mean holdout, gated vs ungated, all models): "
             f"{G[G.gate].hold_out.mean():+.2f} vs {G[~G.gate].hold_out.mean():+.2f}")
    L.append("gold features (M4 vs M1, mean over identical rules): disc "
             f"{G[G.model == 'M4'].disc.mean() - G[G.model == 'M1'].disc.mean():+.3f}, holdout "
             f"{G[G.model == 'M4'].hold_out.mean() - G[G.model == 'M1'].hold_out.mean():+.3f}")
    txt = "\n".join(L) + "\n"
    G.to_csv(OUT / "grid.csv", index=False)
    print(txt)
    (OUT / "report.txt").write_text(txt)


if __name__ == "__main__":
    {"probs": stage_probs, "study": stage_study}[sys.argv[1]]()
