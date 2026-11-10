#!/usr/bin/env python3
"""ZigZag variant B on ALL available gold history (user request 2026-11-10).

Exactly the live model and rule re-proved in `v5_zigzag_champion_reproof.py` (Sharpe 1.58 on
2024-26), extended back in time:
  cme     Databento GC H1 2010+ (back-adjusted, single source), pooled with CME 6E/6B/6J — the
          futures equivalents of EURUSD/GBPUSD/USDJPY. Test 2012-01 -> now (incl. the 2012-15 bear).
          Futures embed carry in the back-adjustment, so no extra swap; spread $0.25/oz as on the CFD.
  switch  Switch XAUUSDs H1 2016-09+ pooled with EURUSDs (2017+) and GBPUSDs/USDJPYs (2022+ only).
          Test 2018-01 -> now. Swap -6%/yr as live.
Model: monthly walk-forward refit, trailing 20,000 bars per series, labels on truncated data,
TOL+1 purge. Rule: P >= .60 -> buy next open + spread, stop 3 ATR, trail 3 ATR under the high,
TP 4.3R, hold 120, one at a time, 0.25% risk.
Reports full-period, per-year, 2024-26 (overlap with the re-proof), random-timing null (circular
shift inside the test window), block-bootstrap CI and buy & hold.

    python scripts/v5_zigzag_long_history.py cme
    python scripts/v5_zigzag_long_history.py switch
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
from scripts.v5_zigzag_panel_probs import load_cme  # noqa: E402
from scripts.v5_zigzag_signal import LEAK_MAX, MIN_TRAIN, TOL, TRAIN_BARS, _frame, atr, intrabar_leak  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "zigzag_long_history"
THR, TRAIL, TP_R, MAX_HOLD, RISK, SPREAD = 0.60, 3.0, 4.3, 120, 0.0025, 0.25
SRC = {
    "cme": dict(start="2012-01-01", swap=0.0, gold="GC", aux=("6E", "6B", "6J")),
    "switch": dict(start="2018-01-01", swap=-0.06, gold="XAUUSDs", aux=("EURUSDs", "GBPUSDs", "USDJPYs")),
}


def load(src: str, sym: str) -> pd.DataFrame:
    if src == "cme":
        return load_cme(sym)
    d = pd.read_csv(ROOT / f"data/v5_runs/zigzag_live_wf/{sym}_H1.csv", parse_dates=["time"]).set_index("time")
    return d[~d.index.duplicated()].sort_index()[["open", "high", "low", "close"]]


def walk_forward(gold, aux, start):
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415
    Fg, _, okg = _frame(gold)
    out = pd.Series(np.nan, index=gold.index)
    months = pd.date_range(start, gold.index[-1] + pd.offsets.MonthBegin(1), freq="MS")
    for m0, m1 in zip(months[:-1], months[1:]):
        test = (gold.index >= m0) & (gold.index < m1)
        if not test.any():
            continue
        X, Y = [], []
        for d in [gold] + [aux[k] for k in sorted(aux)]:
            d = d[d.index < m0]
            if len(d) < MIN_TRAIN + TOL + 1:
                continue
            F, y, ok = _frame(d)
            hi = len(d) - (TOL + 1)
            m = np.zeros(len(d), bool)
            m[max(0, hi - TRAIN_BARS):hi] = True
            m &= ok
            X.append(F.values[m])
            Y.append(y[m])
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                             random_state=0).fit(np.vstack(X), np.concatenate(Y))
        rows = test & okg
        out[rows] = clf.predict_proba(Fg.values[rows])[:, 1]
        print(f"{m0:%Y-%m}: {sum(len(y) for y in Y):,} rows, {len(X)} series, "
              f"{int((out[test] >= THR).sum())} fires", flush=True)
    return out


def backtest(g, fire, start, swap, spread=SPREAD):
    o, h, l = (g[x].to_numpy() for x in ("open", "high", "low"))
    a, t, n = atr(g).to_numpy(), g.index, len(g)
    fire = fire & (t >= start) & np.isfinite(a)
    trades, i = [], 0
    while i < n - 1:
        if not fire[i]:
            i += 1
            continue
        s = i + 1
        e = o[s] + spread
        risk = TRAIL * a[i]
        sl, best, tp = e - risk, e, e + TP_R * risk
        j, px, why = n - 1, o[-1], "open"
        for j in range(s, n):
            if l[j] <= sl:
                px, why = min(sl, o[j]), "stop"
                break
            if h[j] >= tp:
                px, why = max(tp, o[j]), "tp"
                break
            if j - s >= MAX_HOLD:
                px, why = o[j], "time"
                break
            best = max(best, h[j])
            sl = max(sl, best - TRAIL * a[j])
        days = (t[j] - t[s]).total_seconds() / 86400
        trades.append((t[s], t[j], (px - e + e * swap * days / 365) / risk, j - s, why))
        i = j
    tr = pd.DataFrame(trades, columns=["t_in", "t_out", "R", "bars", "exit"])
    days = pd.date_range(pd.Timestamp(start).normalize(), t[-1].normalize(), freq="D")
    ret = pd.Series(0.0, index=days)
    for r in tr.itertuples():
        ret[r.t_out.normalize()] += RISK * r.R
    return tr, ret[ret.index.dayofweek < 5]


def sharpe(r):
    return float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 2 and r.std() > 0 else 0.0


def boot(r, n=2000, blk=20, seed=7):
    x, rng, L, out = r.to_numpy(), np.random.default_rng(seed), len(r), []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + blk) % L for s in rng.integers(0, L, L // blk + 1)])[:L]
        y = x[idx]
        out.append(y.mean() / y.std() * np.sqrt(252))
    return np.percentile(out, [2.5, 97.5])


def main() -> None:
    src = sys.argv[1]
    cfg = SRC[src]
    start = pd.Timestamp(cfg["start"])
    out = OUT / src
    out.mkdir(parents=True, exist_ok=True)
    gold = load(src, cfg["gold"])
    aux = {}
    for s in cfg["aux"]:
        d = load(src, s)
        lk = intrabar_leak(d)
        print(f"{s}: {len(d):,} bars {d.index[0].date()}..{d.index[-1].date()} leak {lk:.3f}")
        if lk <= LEAK_MAX:
            aux[s] = d
    print(f"{cfg['gold']}: {len(gold):,} bars {gold.index[0].date()}..{gold.index[-1].date()} "
          f"leak {intrabar_leak(gold):.3f}")
    cache = out / "probs.csv"
    if cache.exists():
        p = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        p = walk_forward(gold, aux, start)
        p.rename("prob").to_csv(cache)
    pr = p.reindex(gold.index).to_numpy()
    fires = np.isfinite(pr) & (pr >= THR)
    tr, ret = backtest(gold, fires, start, cfg["swap"])
    tr.to_csv(out / "trades.csv", index=False)

    s_full = sharpe(ret)
    eq = (1 + ret).cumprod()
    weeks = (gold.index[-1] - start).days / 7
    lo, hi = boot(ret)
    s0 = int((gold.index < start).sum())
    seg, rng, nl = fires[s0:], np.random.default_rng(13), []
    for k in rng.integers(500, len(seg) - 500, 300):
        z = fires.copy()
        z[s0:] = np.roll(seg, int(k))
        nl.append(sharpe(backtest(gold, z, start, cfg["swap"])[1]))
    nl = np.array(nl)
    gd = gold[gold.index >= start].close.resample("D").last().dropna()
    gr = gd.pct_change().dropna() + cfg["swap"] / 252
    gr = gr[gr.index.dayofweek < 5]
    yr = pd.DataFrame({"zigzag": ret.groupby(ret.index.year).apply(sharpe),
                       "hold_gold": gr.groupby(gr.index.year).apply(sharpe),
                       "gold_ret_%": gd.groupby(gd.index.year).apply(lambda x: 100 * (x.iloc[-1] / x.iloc[0] - 1)),
                       "trades": tr.groupby(tr.t_out.dt.year).size(),
                       "R": tr.groupby(tr.t_out.dt.year).R.sum()}).round(2)
    mid = start + (gold.index[-1] - start) / 2
    w, ls = tr.R[tr.R > 0], tr.R[tr.R <= 0]
    txt = "\n".join([
        f"ZigZag variant B, {src.upper()} gold H1, test {start.date()} -> {gold.index[-1]} ({weeks / 52:.1f} years), "
        f"pooled with {sorted(aux)}",
        f"trades {len(tr)} = {len(tr) / weeks:.2f}/week; win {100 * len(w) / len(tr):.1f}%  avg win {w.mean():+.2f}R  "
        f"avg loss {ls.mean():+.2f}R  expectancy {tr.R.mean():+.3f}R  exits {tr.exit.value_counts().to_dict()}",
        f"at 0.25% risk: {100 * (eq.iloc[-1] - 1):+.1f}% total, {100 * (eq.iloc[-1] ** (52 / weeks) - 1):+.2f}%/yr, "
        f"maxDD {100 * (1 - eq / eq.cummax()).max():.2f}%",
        f"SHARPE {s_full:+.2f}  (95% CI [{lo:+.2f}, {hi:+.2f}])   first half {sharpe(ret[:mid]):+.2f}  "
        f"second half {sharpe(ret[mid:]):+.2f}   2024-26 {sharpe(ret['2024':]):+.2f}   pre-2024 {sharpe(ret[:'2023']):+.2f}",
        f"hold gold same days: Sharpe {sharpe(gr):+.2f} ({100 * ((1 + gr).prod() - 1):+.0f}%), daily corr "
        f"{pd.concat([ret, gr], axis=1).dropna().corr().iloc[0, 1]:+.2f}",
        f"random timing (300 shifts in window): median {np.median(nl):+.2f}, 95th {np.percentile(nl, 95):+.2f}, "
        f"p {(nl >= s_full).mean():.3f}",
        f"2x spread: {sharpe(backtest(gold, fires, start, cfg['swap'], 2 * SPREAD)[1]):+.2f}",
        "", "per year:", yr.to_string()])
    print(txt)
    (out / "report.txt").write_text(txt + "\n")


if __name__ == "__main__":
    sys.exit(main())
