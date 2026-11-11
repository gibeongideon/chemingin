#!/usr/bin/env python3
"""Does sizing ZigZag trades by the model's PROBABILITY raise the Sharpe? (user, 2026-11-11:
"apply capital management based on prediction probability ... using our best ZigZag so far")

BASE = the best ZigZag as deployed 2026-11-11: variant B (P >= .60, 3 ATR stop + trail, TP 4.3R,
hold 120, one at a time) WITH the same-day circuit breaker k = 1.0 (3dz/3ee). Flat 0.25% risk.

PRIORS, recorded before the run so the result cannot be read generously: probability sizing
failed twice on the trend bot (xau-per-trade-prob-sizing-disproven) and on this detector the
high-probability bars did not out-earn the rest (3bz: P > 0.65 bars −4.93bp vs +6.68 base;
3bk: best economics = worst AUC). So the INVERSE rule is in the family as the sign control.

FAMILY, 5 cells, fixed before the first run. m(p) is a multiplier on the 0.25% risk, rescaled so
the mean multiplier over the trades is 1 — the same average risk as FLAT (Sharpe is scale-free;
the rescale only makes CAGR and drawdown comparable):
  FLAT    m = 1
  LIN     0.5 at p = .60 rising linearly to 1.5 at p >= .80
  STEP    1 below .70, 2 at or above
  KELLY   m proportional to (p - 0.5)          (edge-proportional, the textbook rule)
  INV     1.5 at p = .60 falling to 0.5 at p >= .80   (the sign control)
TESTS. (1) expectancy by probability bucket + Spearman rank correlation of R with p — the direct
question. (2) each rule vs FLAT, paired t on daily returns at matched vol. (3) SHUFFLE NULL: the
same multipliers dealt to the trades at random, 2,000 times — a sizing rule must beat the same
set of sizes assigned without the probability. Family p = max over the 4 rules. Both sources.

    python scripts/v5_zigzag_prob_sizing.py cme
    python scripts/v5_zigzag_prob_sizing.py switch
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_zigzag_fall_gate import circuit, daily  # noqa: E402
from scripts.v5_zigzag_frequency import Sim  # noqa: E402
from scripts.v5_zigzag_long_history import OUT, RISK, SPREAD, SRC, THR, load, sharpe  # noqa: E402

N_NULL = 2000
RULES = {
    "LIN": lambda p: 0.5 + np.clip((p - 0.60) / 0.20, 0, 1),
    "STEP": lambda p: np.where(p >= 0.70, 2.0, 1.0),
    "KELLY": lambda p: p - 0.5,
    "INV": lambda p: 1.5 - np.clip((p - 0.60) / 0.20, 0, 1),
}


def trades(sim: Sim, fire: np.ndarray, p: np.ndarray, spread: float = SPREAD) -> pd.DataFrame:
    """One at a time, exactly as Sim.run(K=1, gap=0), but keeping the entry bar's probability."""
    rows, i, idx = [], 0, np.flatnonzero(fire & sim.ok)
    busy_until = -1
    for i in idx:
        if i >= sim.n - 1 or i < busy_until:
            continue
        s, j, R = sim.trade(int(i), spread)
        busy_until = j
        rows.append((sim.t[j].normalize(), p[i], R))
    return pd.DataFrame(rows, columns=["day_out", "p", "R"])


def daily_ret(sim: Sim, tr: pd.DataFrame, mult: np.ndarray) -> pd.Series:
    ret = pd.Series(0.0, index=sim.days)
    np.add.at(ret.values, sim.days.get_indexer(tr.day_out), RISK * mult * tr.R.to_numpy())
    return ret[ret.index.dayofweek < 5]


def main() -> None:
    src = sys.argv[1]
    cfg = SRC[src]
    start = pd.Timestamp(cfg["start"])
    out = OUT / src
    g = load(src, cfg["gold"])
    p = pd.read_csv(out / "probs.csv", index_col=0, parse_dates=True).iloc[:, 0].reindex(g.index).to_numpy()
    sim = Sim(g, start, cfg["swap"])
    fire = np.isfinite(p) & (p >= THR) & ~circuit(g, daily(g), 1.0, "day")
    tr = trades(sim, fire, p)
    tr2 = trades(sim, fire, p, 2 * SPREAD)
    pp, R = tr.p.to_numpy(), tr.R.to_numpy()
    flat = daily_ret(sim, tr, np.ones(len(tr)))
    s_flat = sharpe(flat)

    lines = [f"ZigZag PROBABILITY SIZING, {src.upper()} gold H1, test {start.date()} -> {g.index[-1]}",
             f"base = variant B + circuit breaker, flat 0.25% risk: {len(tr)} trades, Sharpe {s_flat:+.2f}, "
             f"expectancy {R.mean():+.3f}R, win {100 * (R > 0).mean():.1f}%", "",
             "1. DOES A HIGHER PROBABILITY MEAN A BETTER TRADE?"]
    bins = [0.60, 0.65, 0.70, 0.75, 0.80, 1.01]
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (pp >= lo) & (pp < hi)
        if m.sum():
            r = R[m]
            lines.append(f"   P {lo:.2f}-{min(hi, 1):.2f}: n {m.sum():5d} ({100 * m.mean():4.1f}%)  win {100 * (r > 0).mean():5.1f}%  "
                         f"expectancy {r.mean():+.3f}R  (t {r.mean() / r.std() * np.sqrt(len(r)):+.2f})")
    rho, prho = stats.spearmanr(pp, R)
    half = len(tr) // 2
    r1 = stats.spearmanr(pp[:half], R[:half])[0]
    r2 = stats.spearmanr(pp[half:], R[half:])[0]
    lines += [f"   Spearman(P, R) {rho:+.4f} (p {prho:.3f});  first half {r1:+.4f}, second half {r2:+.4f}", "",
              "2. SIZING RULES vs FLAT (same average risk)"]

    rng = np.random.default_rng(17)
    rows, null_max = [], np.full(N_NULL, -np.inf)
    perms = [rng.permutation(len(tr)) for _ in range(N_NULL)]
    day_idx = sim.days.get_indexer(tr.day_out)
    wk = (sim.days.dayofweek < 5)

    def fast_sharpe(mult):
        x = np.zeros(len(sim.days))
        np.add.at(x, day_idx, RISK * mult * R)
        x = x[wk]
        return x.mean() / x.std() * np.sqrt(252)

    for name, f in RULES.items():
        m = f(pp).astype(float)
        m = m / m.mean()
        r = daily_ret(sim, tr, m)
        s = sharpe(r)
        d = r * (flat.std() / r.std()) - flat
        nl = np.array([fast_sharpe(m[k]) for k in perms])
        null_max = np.maximum(null_max, nl)
        eq, eqf = (1 + r).cumprod(), (1 + flat).cumprod()
        rows.append(dict(rule=name, mult_min=round(m.min(), 2), mult_max=round(m.max(), 2),
                         SR=round(s, 2), dSR=round(s - s_flat, 2),
                         t_vs_flat=round(float(d.mean() / d.std() * np.sqrt(len(d))), 2),
                         SR_2x=round(sharpe(daily_ret(sim, tr2, (f(tr2.p.to_numpy()) / f(tr2.p.to_numpy()).mean()))), 2),
                         h1=round(sharpe(r[:len(r) // 2]) - sharpe(flat[:len(r) // 2]), 2),
                         h2=round(sharpe(r[len(r) // 2:]) - sharpe(flat[len(r) // 2:]), 2),
                         maxDD=round(100 * (1 - eq / eq.cummax()).max(), 2),
                         maxDD_flat=round(100 * (1 - eqf / eqf.cummax()).max(), 2),
                         shuffle_med=round(float(np.median(nl)), 2), shuffle95=round(float(np.percentile(nl, 95)), 2),
                         p=round(float((nl >= s).mean()), 3)))
    A = pd.DataFrame(rows)
    best = A.loc[A.SR.idxmax()]
    lines += [A.to_string(index=False), "",
              f"   h1/h2 = Sharpe change vs FLAT in each half.  FLAT at 2x spread: "
              f"{sharpe(daily_ret(sim, tr2, np.ones(len(tr2)))):+.2f}",
              f"   family (4 rules): best {best.rule} {best.SR:+.2f} vs shuffled-size max median "
              f"{np.median(null_max):+.2f}, 95th {np.percentile(null_max, 95):+.2f}, p {(null_max >= best.SR).mean():.3f}"]
    txt = "\n".join(lines)
    print(txt)
    (out / "prob_sizing_report.txt").write_text(txt + "\n")


if __name__ == "__main__":
    sys.exit(main())
