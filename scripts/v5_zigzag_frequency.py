#!/usr/bin/env python3
"""ZigZag variant B at HIGHER TRADE FREQUENCY (user request 2026-11-11: "make more frequent
trades, the ZigZag one, with high accuracy and the most profitable way").

WHERE THE FREQUENCY CAP IS. Variant B opens ONE position at a time and holds up to 120 H1 bars,
so every fire that arrives while a trade is open is thrown away. That rule, not the detector,
sets ~3.3 trades/week. Markets (41 tested), BTC, trend gates, long/short, exit variants and
calendar features are all closed (3dt/3dv/3dw/3dy); tighter targets raise the win rate and cut
the money (3cx/3cy). The untested lever is the one-at-a-time rule itself.

THE IDEA (outside this repo: entry diversification / tranching, the fix for "timing luck").
Allow up to K positions open at once, at least GAP bars apart, each sized at 1/K of the risk so
the BUDGET is unchanged. Sharpe is scale-free, so "K at full risk each" is the same cell at K x
leverage — reported as a dial, never as a second finding (3ao).

PRE-REGISTERED FAMILY, 9 cells, fixed before the first run:
  V0          variant B as live (K=1)                         <- known answer 0.21 CME / 0.42 Switch
  CB          V0 + same-day circuit breaker k=1.0 (3dz)       <- known answer 0.56 / 0.53
  K2g24  K3g24  K5g24   K concurrent, entries >= 24 bars apart
  K3g8        K=3, entries >= 8 bars apart
  T50         threshold 0.50, K=1
  T50K3g24    threshold 0.50 + K=3/24
  CBK3g24     circuit breaker + K=3/24
TEST. Each cell vs V0 as a paired t on daily returns at MATCHED VOL (3as: an overlay is judged at
matched vol), on BOTH sources. Family significance = White reality check: block-bootstrap the
centred difference matrix, statistic = max t. A cell is a keeper only if it is positive in both
sources and the family p clears 0.05 in both.

    python scripts/v5_zigzag_frequency.py cme
    python scripts/v5_zigzag_frequency.py switch
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
from scripts.v5_zigzag_fall_gate import circuit, daily  # noqa: E402
from scripts.v5_zigzag_long_history import MAX_HOLD, OUT, RISK, SPREAD, SRC, TP_R, TRAIL, load, sharpe  # noqa: E402
from scripts.v5_zigzag_signal import atr  # noqa: E402

CELLS = {  # name: (threshold, K, gap_bars, circuit_breaker)
    "V0": (0.60, 1, 0, False), "CB": (0.60, 1, 0, True),
    "K2g24": (0.60, 2, 24, False), "K3g24": (0.60, 3, 24, False), "K5g24": (0.60, 5, 24, False),
    "K3g8": (0.60, 3, 8, False), "T50": (0.50, 1, 0, False),
    "T50K3g24": (0.50, 3, 24, False), "CBK3g24": (0.60, 3, 24, True),
}


class Sim:
    """One trade per entry bar, cached: every cell that enters on bar i gets the same trade."""

    def __init__(self, g: pd.DataFrame, start, swap: float):
        self.o, self.h, self.l = (g[x].to_numpy() for x in ("open", "high", "low"))
        self.a, self.t, self.n = atr(g).to_numpy(), g.index, len(g)
        self.start, self.swap, self.cache = start, swap, {}
        self.ok = (self.t >= start) & np.isfinite(self.a)
        days = pd.date_range(pd.Timestamp(start).normalize(), self.t[-1].normalize(), freq="D")
        self.days = days

    def trade(self, i: int, spread: float):
        key = (i, spread)
        if key in self.cache:
            return self.cache[key]
        o, h, l, a, n = self.o, self.h, self.l, self.a, self.n
        s = i + 1
        e = o[s] + spread
        risk = TRAIL * a[i]
        sl, best, tp = e - risk, e, e + TP_R * risk
        j, px = n - 1, o[-1]
        for j in range(s, n):
            if l[j] <= sl:
                px = min(sl, o[j]); break
            if h[j] >= tp:
                px = max(tp, o[j]); break
            if j - s >= MAX_HOLD:
                px = o[j]; break
            best = max(best, h[j])
            sl = max(sl, best - TRAIL * a[j])
        d = (self.t[j] - self.t[s]).total_seconds() / 86400
        out = (s, j, (px - e + e * self.swap * d / 365) / risk)
        self.cache[key] = out
        return out

    def run(self, fire: np.ndarray, K: int, gap: int, spread: float = SPREAD):
        fire = fire & self.ok
        exits, last, rows = [], -10**9, []
        for i in np.flatnonzero(fire):
            if i >= self.n - 1:
                break
            exits = [j for j in exits if j > i]              # a trade exiting ON bar i frees its slot
            if len(exits) >= K or i - last < gap:
                continue
            s, j, R = self.trade(int(i), spread)
            exits.append(j); last = i
            rows.append((self.t[s], self.t[j], R, j - s))
        tr = pd.DataFrame(rows, columns=["t_in", "t_out", "R", "bars"])
        ret = pd.Series(0.0, index=self.days)
        np.add.at(ret.values, self.days.get_indexer(tr.t_out.dt.normalize()), RISK / K * tr.R.to_numpy())
        return tr, ret[ret.index.dayofweek < 5]


def paired_t(v: pd.Series, base: pd.Series) -> tuple[float, pd.Series]:
    d = v * (base.std() / v.std()) - base if v.std() > 0 else v - base
    return (float(d.mean() / d.std() * np.sqrt(len(d))) if d.std() > 0 else 0.0), d


def reality_check(D: np.ndarray, n=2000, blk=20, seed=11) -> tuple[float, np.ndarray]:
    """D: days x cells of matched-vol differences vs V0. Null: centred columns, block bootstrap,
    statistic = max over cells of the t-stat."""
    T = len(D)
    C = D - D.mean(axis=0)
    rng, mx = np.random.default_rng(seed), []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + blk) % T for s in rng.integers(0, T, T // blk + 1)])[:T]
        y = C[idx]
        mx.append((y.mean(axis=0) / y.std(axis=0) * np.sqrt(T)).max())
    obs = (D.mean(axis=0) / D.std(axis=0) * np.sqrt(T)).max()
    return float(obs), np.array(mx)


def main() -> None:
    src = sys.argv[1]
    cfg = SRC[src]
    start = pd.Timestamp(cfg["start"])
    out = OUT / src
    g = load(src, cfg["gold"])
    p = pd.read_csv(out / "probs.csv", index_col=0, parse_dates=True).iloc[:, 0].reindex(g.index).to_numpy()
    cb = circuit(g, daily(g), 1.0, "day")
    sim = Sim(g, start, cfg["swap"])
    weeks = (g.index[-1] - start).days / 7

    rets, rows = {}, []
    for name, (thr, K, gap, use_cb) in CELLS.items():
        fire = np.isfinite(p) & (p >= thr) & (~cb if use_cb else True)
        tr, r = sim.run(fire, K, gap)
        r2 = sim.run(fire, K, gap, 2 * SPREAD)[1]
        rets[name] = r
        eq = (1 + r).cumprod()
        # mean number of positions open, from bar-level occupancy
        occ = np.zeros(sim.n)
        for a, b in zip(g.index.get_indexer(tr.t_in), g.index.get_indexer(tr.t_out)):
            occ[a:b] += 1
        rows.append(dict(cell=name, trades=len(tr), per_week=round(len(tr) / weeks, 2),
                         win_pct=round(100 * (tr.R > 0).mean(), 1), exp_R=round(tr.R.mean(), 3),
                         avg_open=round(occ[sim.ok].mean(), 2),
                         SR=round(sharpe(r), 2), SR_2x=round(sharpe(r2), 2),
                         pre24=round(sharpe(r[:"2023"]), 2), y24_26=round(sharpe(r["2024":]), 2),
                         cagr_budget=round(100 * (eq.iloc[-1] ** (52 / weeks) - 1), 2),
                         cagr_fullrisk=round(100 * (((1 + K * r).cumprod().iloc[-1]) ** (52 / weeks) - 1), 2),
                         maxDD_budget=round(100 * (1 - eq / eq.cummax()).max(), 2),
                         maxDD_fullrisk=round(100 * (1 - (1 + K * r).cumprod() / (1 + K * r).cumprod().cummax()).max(), 2)))
    A = pd.DataFrame(rows).set_index("cell")
    base = rets["V0"]
    diffs = {}
    for name in CELLS:
        if name == "V0":
            continue
        t, d = paired_t(rets[name], base)
        A.loc[name, "t_vs_V0"] = round(t, 2)
        diffs[name] = d
    t_cb, _ = paired_t(rets["CBK3g24"], rets["CB"])
    D = np.column_stack([diffs[k].to_numpy() for k in diffs])
    obs, mx = reality_check(D)
    best = list(diffs)[int(np.argmax(D.mean(axis=0) / D.std(axis=0)))]
    txt = "\n".join([
        f"ZigZag FREQUENCY study, {src.upper()} gold H1, test {start.date()} -> {g.index[-1]} ({weeks / 52:.1f}y)",
        "budget = each position risks 0.25%/K (same total risk as live); fullrisk = 0.25% each (K x leverage)",
        "", A.to_string(), "",
        f"CBK3g24 vs CB (does tranching add to the circuit breaker?): paired t {t_cb:+.2f}",
        f"family reality check vs V0 (8 cells): best {best} t {obs:+.2f}; null max-t median {np.median(mx):+.2f}, "
        f"95th {np.percentile(mx, 95):+.2f}, p {(mx >= obs).mean():.3f}"])
    print(txt)
    (out / "frequency_report.txt").write_text(txt + "\n")


if __name__ == "__main__":
    sys.exit(main())
