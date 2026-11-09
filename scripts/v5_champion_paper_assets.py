#!/usr/bin/env python3
"""Statistics and figures for the §3dp write-ups (blog post + working paper).

Re-runs `v5_champion_h4_cme` and adds: block-bootstrap Sharpe CIs, monthly return statistics,
exposure over time, rolling 3-year Sharpe, the bear-market forecast path, and a cost sweep.
Outputs: docs/paper/figs/*.svg and docs/paper/stats.json

    python scripts/v5_champion_paper_assets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.v5_champion_h4_cme as C  # noqa: E402
import scripts.v5_volregime_taper_crossasset as T  # noqa: E402
from scripts.v5_champion_bear_test import stats  # noqa: E402

OUT = ROOT / "docs" / "paper"
FIG = OUT / "figs"
COL = {"champion": "#b5791a", "const": "#3f6fa6", "gold": "#8a8a8a"}
NAME = {"champion": "Champion (H4, long-only)", "const": "Constant long, same vol", "gold": "Gold buy & hold"}
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "savefig.bbox": "tight"})


def sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0


def boot_ci(r: pd.Series, n: int = 2000, block: int = 21, seed: int = 3) -> tuple[float, float]:
    """Stationary-ish moving-block bootstrap of daily returns (21-day blocks)."""
    rng, x = np.random.default_rng(seed), r.to_numpy()
    k = int(np.ceil(len(x) / block))
    out = []
    for _ in range(n):
        st = rng.integers(0, len(x) - block, k)
        s = np.concatenate([x[a:a + block] for a in st])[:len(x)]
        out.append(s.mean() / s.std() * np.sqrt(252))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def boot_diff_ci(a: pd.Series, b: pd.Series, n: int = 2000, block: int = 21, seed: int = 4) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    j = a.index.intersection(b.index)
    x, y = a[j].to_numpy(), b[j].to_numpy()
    k = int(np.ceil(len(x) / block))
    out = []
    for _ in range(n):
        st = rng.integers(0, len(x) - block, k)
        idx = np.concatenate([np.arange(s, s + block) for s in st])[:len(x)]
        out.append(x[idx].mean() / x[idx].std() * np.sqrt(252) - y[idx].mean() / y[idx].std() * np.sqrt(252))
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float((out <= 0).mean())


def leverage(d: pd.DataFrame, fc: pd.Series) -> pd.Series:
    ret = d.close.pct_change()
    vol = ret.ewm(halflife=C.HL, min_periods=20).std() * np.sqrt(C.ANN)
    return (fc.clip(-2, 2) * (T.TARGET_VOL / vol)).clip(-T.MAX_LEV, T.MAX_LEV)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    d = C.h4_cme()
    fc = C.champion_signal(d.close)
    S = {"champion": C.run(d, fc, 1.0), "const": C.run(d, pd.Series(1.0, index=d.index), 1.0),
         "gold": d.close.resample("D").last().dropna().pct_change().loc[C.W0:].dropna()}
    st: dict = {"period": [str(S["champion"].index[0].date()), str(S["champion"].index[-1].date())],
                "bars_h4": len(d), "streams": {}}
    for k, r in S.items():
        s = stats(r)
        m = (1 + r).resample("ME").prod() - 1
        lo, hi = boot_ci(r)
        st["streams"][k] = dict(total=s["total"], cagr=s["cagr"], sharpe=s["sharpe"], maxdd=s["maxdd"],
                                underwater=s["underwater_yrs"], vol=float(r.std() * np.sqrt(252) * 100),
                                sharpe_ci=[lo, hi], pct_pos_months=float((m > 0).mean() * 100),
                                best_month=float(m.max() * 100), worst_month=float(m.min() * 100),
                                skew_monthly=float(m.skew()), calmar=float(s["cagr"] / abs(s["maxdd"])),
                                sortino=float(r.mean() / r[r < 0].std() * np.sqrt(252)))
    st["diff_vs_const"] = dict(zip(("lo", "hi", "p_le0"), boot_diff_ci(S["champion"], S["const"])))
    st["diff_vs_gold"] = dict(zip(("lo", "hi", "p_le0"), boot_diff_ci(S["champion"], S["gold"])))
    st["cost_sweep"] = {str(c): sharpe(C.run(d, fc, c)) for c in (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)}
    lev = leverage(d, fc)
    ld = lev.resample("D").mean().loc[C.W0:]
    st["exposure"] = dict(mean=float(ld.mean()), median=float(ld.median()), p95=float(ld.quantile(.95)),
                          bear_mean=float(ld.loc["2011-09":"2015-12"].mean()),
                          post_mean=float(ld.loc["2016":].mean()))
    fcb = fc.loc["2011-09-01":"2015-12-31"]
    st["bear_forecast"] = dict(mean=float(fcb.mean()), resting=float((fcb < .2).mean() * 100),
                               strong=float((fcb >= 1.5).mean() * 100))
    st["turnover_per_year"] = float(lev.diff().abs().resample("YE").sum().loc[C.W0:].mean())
    yr = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in S.items()}) * 100
    st["by_year"] = {int(y): {k: round(float(v), 1) for k, v in row.items()} for y, row in yr.iterrows()}
    st["pct_years_beat_gold"] = float((yr.champion > yr.gold).mean() * 100)
    st["pct_years_positive"] = float((yr.champion > 0).mean() * 100)

    # ---- figures
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for k, r in S.items():
        ax.plot((1 + r).cumprod() * 10000, color=COL[k], lw=1.8 if k == "champion" else 1.1, label=NAME[k])
    ax.set_yscale("log")
    ax.set_yticks([5000, 10000, 20000, 30000])
    ax.set_yticklabels(["$5k", "$10k", "$20k", "$30k"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.axvspan(pd.Timestamp("2011-09-01"), pd.Timestamp("2015-12-31"), color="#c0392b", alpha=.06, lw=0)
    ax.text(pd.Timestamp("2013-10-01"), 5600, "gold bear market", ha="center", fontsize=8, color="#a33")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.30, 1.0))
    ax.set_ylabel("growth of $10,000 (log)")
    fig.savefig(FIG / "equity.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    for k, r in S.items():
        eq = (1 + r).cumprod()
        ax.fill_between(eq.index, (eq / eq.cummax() - 1) * 100, 0, color=COL[k], alpha=.35 if k == "champion" else .18,
                        lw=0, label=NAME[k])
    ax.set_ylabel("drawdown %")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(FIG / "drawdown.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for k in ("champion", "const", "gold"):
        r = S[k]
        rs = r.rolling(756).mean() / r.rolling(756).std() * np.sqrt(252)
        ax.plot(rs, color=COL[k], lw=1.6 if k == "champion" else 1.0, label=NAME[k])
    ax.axhline(0, color="k", lw=.6)
    ax.set_ylabel("rolling 3-year Sharpe")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(FIG / "rolling_sharpe.svg")
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1]))
    seg = slice("2011-01-01", "2016-06-30")
    a1.plot(d.close.loc[seg], color=COL["gold"], lw=1)
    a1.set_ylabel("gold (back-adj.)")
    a2.plot(fc.loc[seg], color=COL["champion"], lw=.9)
    a2.axhline(.15, color="k", lw=.6, ls="--")
    a2.text(pd.Timestamp("2015-10-01"), .25, "resting tilt 0.15", fontsize=8)
    a2.set_ylabel("forecast (0-2)")
    fig.savefig(FIG / "bear_forecast.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.plot(ld, color=COL["champion"], lw=.8)
    ax.axhline(ld.mean(), color="k", lw=.6, ls="--")
    ax.set_ylabel("position (x equity)")
    fig.savefig(FIG / "exposure.svg")
    plt.close(fig)

    (OUT / "stats.json").write_text(json.dumps(st, indent=1))
    print(json.dumps({k: v for k, v in st.items() if k != "by_year"}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
