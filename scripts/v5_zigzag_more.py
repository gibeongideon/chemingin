#!/usr/bin/env python3
"""ZigZag on 15 UNTESTED markets (§3dw, user request 2026-11-09) — the batch left over from §3dt.

Markets: Switch broker H1 2016+ (DE40s, UK100s, STOXX50s, AUS200s — spot CFDs, swap charged), CME
(Databento) H1 2010+ (NKD Nikkei, BZ Brent, HO, RB, PA, KE, ZL, ZM, GF, UB — futures, carry embedded),
Binance SOLUSDT H1 2020-08+ (Switch SOLUSDs costs: 21bp spread, -15%/yr swap).
Model: the live ZigZag model pooled over these 15 series, quarterly walk-forward from 2018-01
(each series starts once it has MIN_TRAIN bars). Rule V0 = best live variant (P>=.60, 3 ATR trail,
TP 4.3R, hold 120h, long only, 0.25% risk).
Per market: Sharpe, random-timing null p (200 circular shifts), buy & hold, halves 2018-21 / 2022-26,
2x-cost Sharpe. PASS = null p < 0.05 AND Sharpe > 0 in both halves AND Sharpe > buy & hold.
Family: max-stat of the best market vs the max over markets of random-timing replays.

    python scripts/v5_zigzag_more.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_zigzag_btc import sim  # noqa: E402   (numba sim with per-bar swap)
from scripts.v5_zigzag_panel_probs import load_cme  # noqa: E402
from scripts.v5_zigzag_signal import LEAK_MAX, MIN_TRAIN, TOL, TRAIN_BARS, _frame, atr, intrabar_leak  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "zigzag_more"
START, H2 = pd.Timestamp("2018-01-01"), "2022-01-01"
RISK, N_NULL = 0.0025, 200
# (source, round-trip cost bp, extra long swap %/yr — 0 for futures, which embed carry)
MKTS = {"DE40s": ("switch", 0.9 * 1.5, 4.22), "UK100s": ("switch", 5.6, 6.32), "STOXX50s": ("switch", 3.8, 4.32),
        "AUS200s": ("switch", 3.3, 5.33), "NKD": ("cme", 3.2, 0), "BZ": ("cme", 6.5, 0), "HO": ("cme", 8, 0),
        "RB": ("cme", 8, 0), "PA": ("cme", 15, 0), "KE": ("cme", 10, 0), "ZL": ("cme", 10, 0),
        "ZM": ("cme", 10, 0), "GF": ("cme", 10, 0), "UB": ("cme", 2, 0), "SOL": ("binance", 21, 15.0)}
V0 = (0.60, 3.0, 4.3, 120)


def load(name: str, src: str) -> pd.DataFrame:
    if src == "cme":
        return load_cme(name)
    f = ROOT / "data/zigzag_more" / ("SOLUSDT_H1.csv" if name == "SOL" else f"{name}_H1.csv")
    d = pd.read_csv(f, parse_dates=["time"], index_col="time")[["open", "high", "low", "close"]]
    return d[~d.index.duplicated()].sort_index().dropna()


def probs(data: dict) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415
    frames = {k: _frame(d) for k, d in data.items()}
    out = {k: pd.Series(np.nan, index=d.index) for k, d in data.items()}
    end = max(d.index[-1] for d in data.values())
    cuts = pd.date_range(START, end + pd.offsets.QuarterBegin(1), freq="QS")
    kw = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0)
    for c0, c1 in zip(cuts[:-1], cuts[1:]):
        t0, X, Y = time.time(), [], []
        for k, d in data.items():
            n = int((d.index < c0).sum())
            if n < MIN_TRAIN:
                continue
            F, y, ok = _frame(d.iloc[:n])
            hi = n - (TOL + 1)
            lo = max(0, hi - TRAIN_BARS)
            m = np.zeros(n, bool)
            m[lo:hi] = True
            m &= ok
            X.append(F.values[m])
            Y.append(y[m])
        clf = HistGradientBoostingClassifier(**kw).fit(np.vstack(X), np.concatenate(Y))
        for k, d in data.items():
            if int((d.index < c0).sum()) < MIN_TRAIN:
                continue                       # no prediction before a series has its own history
            Fa, _, oka = frames[k]
            rows = (d.index >= c0) & (d.index < c1) & oka
            if rows.any():
                out[k][rows] = clf.predict_proba(Fa.values[rows])[:, 1]
        print(f"{c0:%Y-%m}: {sum(len(y) for y in Y):,} rows {time.time() - t0:.0f}s", flush=True)
    return {k: v[v.index >= START] for k, v in out.items()}


class M:
    def __init__(self, d, p, cost_bp, swap_pct):
        d = d.loc[p.dropna().index[0]:]
        self.d, self.o, self.h, self.l = d, *(d[x].to_numpy(float) for x in ("open", "high", "low"))
        self.a = atr(d).to_numpy(float)
        self.p = p.reindex(d.index).to_numpy(float)
        self.cost, self.swap = cost_bp * 1e-4, swap_pct / 100 / 8760
        day = d.index.normalize()
        wd = day + pd.to_timedelta(np.where(day.dayofweek == 5, 2, np.where(day.dayofweek == 6, 1, 0)), unit="D")
        self.days = pd.DatetimeIndex(sorted(set(wd)))
        self.did = self.days.get_indexer(wd)

    def run(self, fire=None, cost_mult=1.0):
        thr, k, tp, hold = V0
        f = ((self.p >= thr) & np.isfinite(self.p)) if fire is None else fire
        ex, R = sim(self.o, self.h, self.l, self.a, f, k, tp, hold, self.cost * cost_mult, self.swap)
        m = ex >= 0
        ret = np.zeros(len(self.days))
        np.add.at(ret, self.did[ex[m]], RISK * R[m])
        return pd.Series(ret, index=self.days), R[m], f


def sr(r, a=None, b=None):
    r = r.loc[a:b]
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    for k, (src, _, _) in MKTS.items():
        d = load(k, src)
        lk = intrabar_leak(d)
        print(f"{k}: {len(d):,} bars {d.index[0].date()}..{d.index[-1].date()} leak {lk:.3f}", flush=True)
        if lk <= LEAK_MAX:
            data[k] = d
    P = probs(data)
    pd.DataFrame(P).to_csv(OUT / "probs.csv.gz", float_format="%.4f")
    rng = np.random.default_rng(13)
    rows, mk = [], {}
    for k, d in data.items():
        m = M(d, P[k], MKTS[k][1], MKTS[k][2])
        mk[k] = m
        ret, R, f = m.run()
        s = sr(ret)
        nl = np.array([sr(m.run(fire=np.roll(f, int(x)))[0]) for x in rng.integers(500, len(f) - 500, N_NULL)])
        bh = d.close.loc[m.d.index[0]:].resample("D").last().dropna().pct_change()
        bh = bh - MKTS[k][2] / 100 / 365                         # CFD swap on a held long
        bh = bh[bh.index.dayofweek < 5] if k != "SOL" else bh
        row = dict(mkt=k, src=MKTS[k][0], start=str(m.d.index[0].date()), trades=len(R),
                   win=round(100 * (R > 0).mean(), 1), SR=round(s, 2), SR_2x=round(sr(m.run(cost_mult=2)[0]), 2),
                   p=round((nl >= s).mean(), 3), SR_h1=round(sr(ret, None, "2021-12-31"), 2),
                   SR_h2=round(sr(ret, H2), 2), BH=round(float(bh.mean() / bh.std() * np.sqrt(252)), 2))
        row["PASS"] = row["p"] < 0.05 and row["SR_h1"] > 0 and row["SR_h2"] > 0 and row["SR"] > row["BH"]
        rows.append(row)
        print(row, flush=True)
    A = pd.DataFrame(rows).sort_values("SR", ascending=False)
    obs, mx = A.SR.max(), []
    for _ in range(100):
        mx.append(max(sr(m.run(fire=np.roll(m.run()[2], int(rng.integers(500, len(m.p) - 500))))[0]) for m in mk.values()))
    mx = np.array(mx)
    txt = (f"ZigZag V0 on {len(mk)} untested markets (pooled quarterly WF from 2018; costs incl. CFD swap where spot)\n"
           + A.to_string(index=False)
           + f"\n\nPASS: {int(A.PASS.sum())} / {len(A)}\nfamily max-stat: best SR {obs:.2f} vs random-timing max over markets "
           f"median {np.median(mx):.2f}, 95th {np.percentile(mx, 95):.2f}, p {(mx >= obs).mean():.2f}\n")
    print(txt)
    (OUT / "report.txt").write_text(txt)


if __name__ == "__main__":
    sys.exit(main())
