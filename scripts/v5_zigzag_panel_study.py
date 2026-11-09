#!/usr/bin/env python3
"""ZigZag PANEL study (§3dt stage 2): can the best ZigZag variant be improved, and where else does it work?

Probabilities from `v5_zigzag_panel_probs.py` (pooled, quarterly walk-forward, 2018-01..2026-11, 26 CME
futures). Trade rule = the live ZigZag long-only harness: fire when P >= thr on a closed H1 bar, buy the
next open (+ round-trip cost), stop k x ATR14 trailing the highest high, optional TP at tp_r x risk,
max hold, one position at a time. 0.25% risk per trade. Futures returns already embed carry.
Costs: estimated CFD round trips in bp (COST_BP) — stress 2x is reported.

A  every market, the BEST LIVE VARIANT V0 (thr .60, k 3, TP 4.3R, hold 120): Sharpe, trades, timing-null
   p (200 circular shifts of the fire mask, same rule), buy & hold Sharpe, corr with GC.
B  48-variant grid (thr .55/.60/.65/.70 x k 2/3/4 x TP none/4.3R x hold 120/240). SELECTION CONTROL:
   the variant is chosen by mean Sharpe over the OTHER 25 markets in 2018-21, then read on GC 2022-26.
C  GC gates from the champion (H4 champion forecast, causal, at the signal bar): keep fires when the
   champion is >= 0.5 / skip otherwise — kept AND skipped samples reported, each vs its timing null.
D  portfolio: markets whose V0 had Sharpe > 0 AND null p < 0.10 in 2018-21 are added at equal risk to
   GC; result read on 2022-26 vs GC alone. Max-stat over markets reported for the family claim.

    python scripts/v5_zigzag_panel_study.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_zigzag_panel_probs import load_cme  # noqa: E402
from scripts.v5_zigzag_signal import atr  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "zigzag_panel"
RISK, N_NULL, SEED = 0.0025, 200, 5
H1_END, H2_START = "2021-12-31", "2022-01-01"
COST_BP = {"GC": 1.2, "SI": 6, "HG": 6, "PL": 10, "CL": 6, "NG": 15, "ES": 1.5, "NQ": 1.5, "YM": 2,
           "RTY": 4, "ZN": 2, "ZB": 2, "ZF": 2, "6E": 1, "6J": 1.5, "6B": 1.5, "6A": 2, "6C": 2, "6S": 2,
           "BTC": 10, "ETH": 20, "ZC": 10, "ZS": 8, "ZW": 10, "LE": 8, "HE": 12}
V0 = (0.60, 3.0, 4.3, 120)
GRID = list(itertools.product((0.55, 0.60, 0.65, 0.70), (2.0, 3.0, 4.0), (0.0, 4.3), (120, 240)))


@njit(cache=True)
def sim(o, h, l, a, fire, k, tp_r, hold, cost):
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
            j = s
            out = np.nan
            while j < n:
                if l[j] <= sl:
                    out = min(sl, o[j])
                    break
                if h[j] >= tp:
                    out = max(tp, o[j])
                    break
                if j - s >= hold:
                    out = o[j]
                    break
                best = max(best, h[j])
                if np.isfinite(a[j]):
                    sl = max(sl, best - k * a[j])
                j += 1
            if j >= n:
                break
            ex_i[i] = j
            R[i] = (out - e) / risk
            i = j
        i += 1
    return ex_i, R


class Mkt:
    def __init__(self, root: str, p: pd.Series):
        d = load_cme(root).loc[p.index[0]:]
        self.root, self.idx = root, d.index
        self.o, self.h, self.l = (d[x].to_numpy(float) for x in ("open", "high", "low"))
        self.a = atr(d).to_numpy(float)
        self.p = p.reindex(d.index).to_numpy(float)
        self.close = d.close
        day = d.index.normalize()
        wd = day + pd.to_timedelta(np.where(day.dayofweek == 5, 2, np.where(day.dayofweek == 6, 1, 0)), unit="D")
        self.days = pd.DatetimeIndex(sorted(set(wd)))
        self.day_id = self.days.get_indexer(wd)
        self.cost = COST_BP[root] * 1e-4

    def run(self, v, fire=None, cost_mult=1.0):
        thr, k, tp, hold = v
        f = (self.p >= thr) if fire is None else fire
        ex, R = sim(self.o, self.h, self.l, self.a, f & np.isfinite(self.p), k, tp, hold, self.cost * cost_mult)
        m = ex >= 0
        ret = np.zeros(len(self.days))
        np.add.at(ret, self.day_id[ex[m]], RISK * R[m])
        return pd.Series(ret, index=self.days), R[m], np.flatnonzero(m)

    def null(self, v, n=N_NULL, seg=None):
        f0 = (self.p >= v[0]) & np.isfinite(self.p)
        rng = np.random.default_rng(SEED)
        out = []
        for k in rng.integers(500, len(f0) - 500, n):
            out.append(sharpe(self.run(v, fire=np.roll(f0, int(k)))[0], seg))
        return np.array(out)


def sharpe(r: pd.Series, seg=None) -> float:
    if seg is not None:
        r = r.loc[seg[0]:seg[1]]
    s = r.std()
    return float(r.mean() / s * np.sqrt(252)) if s > 0 else 0.0


def bh_sharpe(close: pd.Series, seg) -> float:
    r = close.resample("D").last().dropna().pct_change().loc[seg[0]:seg[1]]
    return float(r.mean() / r.std() * np.sqrt(252))


def main() -> None:
    cal = "--cal" in sys.argv                     # model WITH hour/weekday features
    P = pd.read_csv(OUT / ("probs_cal.csv.gz" if cal else "probs.csv.gz"), index_col=0, parse_dates=True)
    M = {r: Mkt(r, P[r].dropna()) for r in P.columns if P[r].notna().sum() > 1000}
    FULL, H1, H2 = ("2018-01-01", "2026-12-31"), ("2018-01-01", H1_END), (H2_START, "2026-12-31")
    lines = [f"ZigZag panel: {len(M)} CME markets, pooled quarterly walk-forward {P.index[0]:%Y-%m}..{P.index[-1]:%Y-%m}",
             f"V0 = best live variant (thr {V0[0]}, trail {V0[1]} ATR, TP {V0[2]}R, hold {V0[3]}h); costs = est. CFD round trip\n"]

    # ---------------- A
    gc_ret = M["GC"].run(V0)[0]
    rows, streams = [], {}
    for r, m in M.items():
        ret, R, _ = m.run(V0)
        streams[r] = ret
        nf, n1 = m.null(V0), m.null(V0, seg=H1)
        rows.append(dict(mkt=r, trades=len(R), win=round(100 * (R > 0).mean(), 1), meanR=round(R.mean(), 3),
                         SR=round(sharpe(ret), 2), SR_2x_cost=round(sharpe(m.run(V0, cost_mult=2)[0]), 2),
                         null_p=round((nf >= sharpe(ret)).mean(), 3), SR_h1=round(sharpe(ret, H1), 2),
                         p_h1=round((n1 >= sharpe(ret, H1)).mean(), 3), SR_h2=round(sharpe(ret, H2), 2),
                         BH=round(bh_sharpe(m.close, FULL), 2),
                         corr_GC=round(ret.reindex(gc_ret.index).fillna(0).corr(gc_ret), 2)))
        print(rows[-1], flush=True)
    A = pd.DataFrame(rows).sort_values("SR", ascending=False)
    lines.append("A. V0 on every market (null_p: share of 200 random-timing replays with Sharpe >= it)\n" + A.to_string(index=False))
    # family max-stat: is the BEST market's Sharpe beyond what random timing produces across markets?
    obs = A.SR.max()
    rng = np.random.default_rng(SEED + 1)
    mx = []
    for _ in range(100):
        best = -9
        for m in M.values():
            f0 = (m.p >= V0[0]) & np.isfinite(m.p)
            best = max(best, sharpe(m.run(V0, fire=np.roll(f0, int(rng.integers(500, len(f0) - 500))))[0]))
        mx.append(best)
    lines.append(f"\nfamily max-stat: best market SR {obs:.2f}; random-timing max over markets: median "
                 f"{np.median(mx):.2f}, 95th {np.percentile(mx, 95):.2f}, p {(np.array(mx) >= obs).mean():.2f}\n")

    # ---------------- B
    gs = []
    for v in GRID:
        others = [sharpe(m.run(v)[0], H1) for r, m in M.items() if r != "GC"]
        gret = M["GC"].run(v)[0]
        gs.append(dict(thr=v[0], trail=v[1], tp=v[2], hold=v[3], others_SR_h1=round(np.mean(others), 3),
                       GC_SR_h1=round(sharpe(gret, H1), 2), GC_SR_h2=round(sharpe(gret, H2), 2),
                       GC_SR_full=round(sharpe(gret), 2)))
    B = pd.DataFrame(gs).sort_values("others_SR_h1", ascending=False)
    pick = B.iloc[0]
    v_pick = (pick.thr, pick.trail, pick.tp, int(pick.hold))
    n2 = M["GC"].null(v_pick, seg=H2)
    s2 = sharpe(M["GC"].run(v_pick)[0], H2)
    v0_2 = sharpe(M["GC"].run(V0)[0], H2)
    lines.append("B. 48-variant grid. others_SR_h1 = mean Sharpe over the other 25 markets 2018-21 (the selector)\n"
                 + B.head(12).to_string(index=False) + "\n...\n" + B.tail(4).to_string(index=False)
                 + f"\nSELECTED on other markets: {v_pick} -> GC 2022-26 Sharpe {s2:.2f} vs V0 {v0_2:.2f}; "
                 f"timing-null p {(n2 >= s2).mean():.3f}\n"
                 f"GC-only best cell (in-sample, for reference): {B.sort_values('GC_SR_full').iloc[-1][['thr', 'trail', 'tp', 'hold', 'GC_SR_full']].to_dict()}\n")

    # ---------------- C
    g = M["GC"]
    h4 = load_cme("GC").resample("4h").agg(dict(open="first", high="max", low="min", close="last")).dropna()
    fc = champion_signal(h4.close).shift(1).reindex(g.idx, method="ffill").to_numpy()   # last CLOSED H4 bar
    f0 = (g.p >= V0[0]) & np.isfinite(g.p)
    cl = []
    for name, gate in (("champion >= 0.5 (trend up)", fc >= 0.5), ("champion < 0.5", fc < 0.5),
                       ("champion >= 1.0", fc >= 1.0), ("close > 200-day EMA", (g.close > g.close.ewm(span=24 * 200).mean()).to_numpy())):
        f = f0 & gate
        ret, R, _ = g.run(V0, fire=f)
        rng = np.random.default_rng(SEED + 2)
        nl = np.array([sharpe(g.run(V0, fire=np.roll(f, int(k)))[0]) for k in rng.integers(500, len(f) - 500, N_NULL)])
        cl.append(dict(gate=name, fires=int(f.sum()), trades=len(R), meanR=round(R.mean(), 3) if len(R) else np.nan,
                       SR=round(sharpe(ret), 2), null_p=round((nl >= sharpe(ret)).mean(), 3)))
    lines.append("C. GC gates (V0 rule; ungated V0 GC SR " f"{sharpe(gc_ret):.2f}, meanR {M['GC'].run(V0)[1].mean():.3f})\n"
                 + pd.DataFrame(cl).to_string(index=False) + "\n")

    # ---------------- D
    sel = A[(A.mkt != "GC") & (A.SR_h1 > 0) & (A.p_h1 < 0.10)].mkt.tolist()
    S = pd.DataFrame(streams).fillna(0.0)
    port = S[["GC"] + sel].mean(axis=1) if sel else S["GC"]
    gc = S["GC"]
    corr = S.loc[:, ["GC"] + sel].corr().round(2) if sel else None
    lines.append(f"D. markets selected on 2018-21 (SR>0, null p<0.10): {sel or 'none'}\n"
                 f"   2022-26: GC alone SR {sharpe(gc, H2):.2f} | GC + selected (equal risk) SR {sharpe(port, H2):.2f}\n"
                 f"   full: GC {sharpe(gc):.2f} | portfolio {sharpe(port):.2f}\n" + (f"{corr}\n" if corr is not None else ""))
    txt = "\n".join(lines)
    print(txt)
    (OUT / ("report_cal.txt" if cal else "report.txt")).write_text(txt)


if __name__ == "__main__":
    sys.exit(main())
