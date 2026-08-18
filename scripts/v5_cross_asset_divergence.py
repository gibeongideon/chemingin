"""Cross-asset divergence mean-reversion — a genuinely DIFFERENT edge source.

Borrowed from Forven's `cross_asset_divergence` builtin (re-implemented from
the described mechanism, not copied — AGPL). Every prior borrowed idea this
line tried (SuperTrend, on XAU/2nd-basket/FX) was still trend-following and
converged to what the champ-recipe already knows. This is not: it's a
relative-value MEAN-REVERSION signal — fade the z-scored divergence between two
historically-correlated assets' returns, betting the excess reverts.

This exact mechanism was already tried on ONE pair (gold/silver — see
[[gold-silver-spread-disproven]]: corr 0.79, but the rolling z-spread edge is
pre-2015-only and dead OOS 2017+). That result is used here as a NEGATIVE
CONTROL: if the method is applied correctly, XAU/SILVER should come back dead,
confirming the harness isn't the reason it failed before. The actual question
is whether the SAME technique works on pairs never tested this way.

Pairs tested (all with real, measured historical correlation in this repo):
  GOLD / SILVER     (corr ~0.6)   — NEGATIVE CONTROL, expect dead
  WTI / BRENT       (corr ~0.85)  — same trade, cheapest possible convergence
  PLAT / PALL       (corr ~0.44)  — precious-metals complex, not gold itself
  EURUSD / GBPUSD   (corr, FX)    — never tested this way
  SPX / NDX         (corr ~0.86)  — US equity complex
  DAX / STOXX       (corr ~0.87)  — EU equity complex
  AUDUSD / NZDUSD   (corr, FX)    — "Aussie/Kiwi", classically linked

Single-leg framing (trade the FIRST-named asset only, sized by its divergence
from the second) for a direct fit with this repo's existing vol-targeted
`engine()`/`_buffered_pos` machinery — not a dollar-neutral pairs trade. Same
rigor as every prior test in this line: lookahead-probed first (via a custom
two-asset synthetic frame builder), DSR/PBO on every per-pair grid, fitness-
ranked at the end.

    python scripts/v5_cross_asset_divergence.py
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

import scripts.v5_basket_challenge as vbc  # noqa: E402
from src.evaluation.lookahead_probe import probe_lookahead, print_report, synthetic_ohlc  # noqa: E402
from src.evaluation.dsr_pbo import deflated_sharpe_ratio, pbo_cscv  # noqa: E402
from src.evaluation.fitness import rank_candidates  # noqa: E402

EVAL_START = "2018-01-01"
WINDOWS = (20, 30, 50)
THRESHOLDS = (1.5, 2.0, 2.5, 3.0)

PAIRS = (
    ("GOLD", "SILVER"),   # negative control — known dead (gold-silver-spread-disproven)
    ("WTI", "BRENT"),
    ("PLAT", "PALL"),
    ("EURUSD", "GBPUSD"),
    ("SPX", "NDX"),
    ("DAX", "STOXX"),
    ("AUDUSD", "NZDUSD"),
)


def load_close(sym: str) -> pd.Series:
    fp = ROOT / "data" / f"{sym}_D1_long.csv"
    df = pd.read_csv(fp, parse_dates=["time"], index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df["close"]


def divergence_zscore(close_a: pd.Series, close_b: pd.Series, window: int) -> pd.Series:
    ret_a = close_a.pct_change()
    ret_b = close_b.pct_change()
    excess = (ret_a - ret_b).reindex(close_a.index)
    m = excess.rolling(window).mean()
    sd = excess.rolling(window).std()
    return (excess - m) / sd


def divergence_fc_from_frame(df: pd.DataFrame, window: int, threshold: float, cap: float = 2.0) -> pd.Series:
    """Causal: fades asset A when its recent excess return vs B is extreme.
    `df` needs `close_a`/`close_b` columns of equal length/index."""
    z = divergence_zscore(df["close_a"], df["close_b"], window)
    fc = (-z).clip(-cap, cap)
    return fc.where(z.abs() >= threshold, 0.0)


def build_divergence_probe_frame(n: int = 900, seed: int = 7) -> pd.DataFrame:
    """Two INDEPENDENT synthetic legs (different seeds) — correlation between
    them is irrelevant for a truncation-invariance check; only causality of
    the computation is being tested."""
    a = synthetic_ohlc(n=n, seed=seed)
    b = synthetic_ohlc(n=n, seed=seed + 1000)
    return pd.DataFrame({"close_a": a["close"].values, "close_b": b["close"].values}, index=a.index)


def net_returns_for_pair(sym_a: str, sym_b: str, window: int, threshold: float) -> pd.Series:
    """Trade sym_a with a vol-targeted, buffered position sized by its
    divergence from sym_b — same cost/vol-target machinery as every other
    D1 signal in this repo (`vbc._buffered_pos`, live spread-median floor)."""
    df_a = pd.read_csv(ROOT / "data" / f"{sym_a}_D1_long.csv", parse_dates=["time"], index_col="time").sort_index()
    df_a = df_a[~df_a.index.duplicated(keep="last")]
    close_b = load_close(sym_b)
    j = pd.concat([df_a["close"].rename("close_a"), close_b.rename("close_b")], axis=1).dropna()
    fc = divergence_fc_from_frame(j, window, threshold)
    fc = fc.reindex(df_a.index).fillna(0.0)

    close = df_a["close"]
    spread_px = df_a["spread"].clip(lower=df_a["spread"].median())
    ret = close.pct_change()
    vol = ret.ewm(halflife=42, min_periods=20).std() * np.sqrt(252)
    pos = vbc._buffered_pos(fc, vol, spread_px, close, 252).shift(1).fillna(0.0)
    cost = pos.diff().abs().fillna(0.0) * (spread_px / close)
    net = (pos * ret - cost).fillna(0.0).resample("D").sum()
    return net[net.index.dayofweek < 5]


def sharpe(d: pd.Series, start: str = EVAL_START) -> float:
    d = d.loc[start:].dropna()
    return float(d.mean() / d.std() * np.sqrt(252)) if len(d) > 20 and d.std() > 0 else float("nan")


def dd(d: pd.Series, start: str = EVAL_START) -> float:
    e = (1 + d.loc[start:]).cumprod()
    return float((e / e.cummax() - 1).min() * 100) if len(e) else float("nan")


def main():
    print("=== STAGE 0: lookahead probe (custom two-asset synthetic frame) ===")
    probes = {
        f"divergence_fc(window={w},thr={t})": (lambda d, w=w, t=t: divergence_fc_from_frame(d, w, t))
        for w in (30,) for t in (2.0,)
    }
    results = {
        name: probe_lookahead(fn, n=900, offsets=(150, 100, 50, 20), frame_builder=build_divergence_probe_frame)
        for name, fn in probes.items()
    }
    print_report(results)
    if not all(v.ok for v in results.values()):
        print("!! lookahead probe FAILED — refusing to trust the backtest below.")
        return

    print("\n=== STAGE 1: per-pair grid (window x threshold), trading the FIRST-named leg ===\n")
    rows = []
    for sym_a, sym_b in PAIRS:
        grid_rets = {}
        for w in WINDOWS:
            for t in THRESHOLDS:
                grid_rets[(w, t)] = net_returns_for_pair(sym_a, sym_b, w, t)
        M = pd.DataFrame(grid_rets).dropna(how="all").fillna(0.0).loc[EVAL_START:]
        sh = {k: sharpe(M[k], start=M.index[0].strftime("%Y-%m-%d")) for k in M.columns}
        best_key = max(sh, key=lambda k: sh[k] if np.isfinite(sh[k]) else -1e9)
        n_pos = sum(1 for v in sh.values() if np.isfinite(v) and v > 0)
        trial_sh_daily = np.array([sh[k] / np.sqrt(252) for k in sh])
        dsr = deflated_sharpe_ratio(M[best_key].values, trial_sh_daily)
        pbo = pbo_cscv(M.values, n_partitions=10)
        control = " <-- NEGATIVE CONTROL (expect dead)" if (sym_a, sym_b) == ("GOLD", "SILVER") else ""
        print(f"{sym_a}/{sym_b}: best(window={best_key[0]},thr={best_key[1]})  "
              f"SR={sh[best_key]:+.3f}  DD={dd(M[best_key]):+.1f}%  pos={n_pos}/{len(sh)}  "
              f"DSR={dsr['dsr']:.3f}  PBO={pbo.pbo:.3f}{control}")
        rows.append(dict(pair=f"{sym_a}/{sym_b}", best=best_key, sharpe=sh[best_key],
                         max_drawdown_pct=dd(M[best_key]), dsr=dsr["dsr"], pbo=pbo.pbo,
                         n_pos=n_pos, n_cells=len(sh)))

    print("\n=== STAGE 2: cross-pair-screen-deflated DSR (correcting for scanning 7 pairs) ===")
    all_best_sharpes = np.array([r["sharpe"] for r in rows])
    trial_sh_screen = all_best_sharpes / np.sqrt(252)
    for r in rows:
        # recompute using the actual best cell's daily returns for THIS pair
        sym_a, sym_b = r["pair"].split("/")
        best_ret = net_returns_for_pair(sym_a, sym_b, r["best"][0], r["best"][1]).loc[EVAL_START:]
        dsr_screen = deflated_sharpe_ratio(best_ret.values, trial_sh_screen)
        r["dsr_screen_level"] = dsr_screen["dsr"]
        print(f"  {r['pair']:16s} within-pair DSR={r['dsr']:.3f}   screen-level DSR={dsr_screen['dsr']:.3f}")

    print("\n=== STAGE 3: fitness-ranked summary (screen-level DSR, no book to diversify yet) ===")
    ranked = rank_candidates(
        [{"sharpe": r["sharpe"], "max_drawdown_pct": r["max_drawdown_pct"], "dsr": r["dsr_screen_level"],
          "pair": r["pair"]} for r in rows]
    )
    for r in ranked:
        print(f"  fitness={r['fitness']:5.1f}  {r['pair']:16s}  SR={r['sharpe']:+.3f}  "
              f"DD={r['max_drawdown_pct']:+.1f}%  DSR(screen)={r['dsr']:.3f}")


if __name__ == "__main__":
    main()
