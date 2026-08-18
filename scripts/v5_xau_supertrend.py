"""SuperTrend — a genuinely different signal SHAPE, borrowed as a technique
(re-implemented from the public formula, not copied) from Forven's builtin
strategy library, tested "the MT5 way": lookahead-probed first
(`src/evaluation/lookahead_probe.py`), backtested through the same vol-targeted
engine as the champion (`v5_xau_turn_prob.engine`), at live cost, paired against
buy&hold and the champion, and — since a parameter grid gets scanned — deflated
with the DSR/PBO this repo already has (`src/evaluation/dsr_pbo.py`) rather than
just quoting the best cell's raw Sharpe.

The champion is an EWMAC (moving-average crossover) + smoothed-Donchian-position
blend — continuous, no discrete state. SuperTrend is a discrete ATR-band
trend-state machine: price above the trailing upper band -> uptrend; below the
trailing lower band -> downtrend; the band only moves in the trend's favour
(never against it) until price crosses it, which is a structurally different
mechanism for deciding "trend is on" than an EWMA crossover, and worth testing
on its own terms rather than assumed similar because both are "trend following."

    python scripts/v5_xau_supertrend.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_xau_turn_prob import load_h4, atr, engine, paired, per_year  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402
from src.evaluation.lookahead_probe import probe_lookahead, print_report  # noqa: E402
from src.evaluation.dsr_pbo import deflated_sharpe_ratio, pbo_cscv  # noqa: E402

PNL_EVAL_START = "2018-01-01"
ATR_PERIODS = (7, 10, 14, 21)
MULTIPLIERS = (2.0, 2.5, 3.0, 3.5, 4.0)


def supertrend_direction(df: pd.DataFrame, atr_period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """Classic SuperTrend trend-state machine, +1 (up) / -1 (down). Purely causal:
    each bar only ever references bar i-1's already-computed band/state."""
    close, high, low = df["close"].values, df["high"].values, df["low"].values
    a = atr(df, atr_period).values
    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * a
    basic_lower = hl2 - multiplier * a
    n = len(df)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    direction = np.ones(n)
    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    for i in range(1, n):
        final_upper[i] = (
            basic_upper[i]
            if (basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1])
            else final_upper[i - 1]
        )
        final_lower[i] = (
            basic_lower[i]
            if (basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1])
            else final_lower[i - 1]
        )
        if close[i] > final_upper[i - 1]:
            direction[i] = 1.0
        elif close[i] < final_lower[i - 1]:
            direction[i] = -1.0
        else:
            direction[i] = direction[i - 1]
    return pd.Series(direction, index=df.index, name="st_dir")


def main():
    print("=== STAGE 0: lookahead probe (before trusting anything below) ===")
    probes = {
        f"supertrend(p={p},m={m})": (lambda d, p=p, m=m: supertrend_direction(d, p, m))
        for p in (10,) for m in (3.0,)
    }
    results = {name: probe_lookahead(fn, n=1200, offsets=(150, 100, 50, 20)) for name, fn in probes.items()}
    print_report(results)
    if not all(v.ok for v in results.values()):
        print("!! lookahead probe FAILED — refusing to trust the backtest below.")
        return

    print("\n=== STAGE 1: parameter grid, standalone, long-only (kill-the-shorts prior) ===")
    df = load_h4(0.448)
    close = df["close"]
    champ_fc = champion_signal(close)
    bh_m, bh_d = engine(df, pd.Series(1.0, index=df.index), eval_start=PNL_EVAL_START)
    ch_m, ch_d = engine(df, champ_fc, eval_start=PNL_EVAL_START)
    print(f"  buy&hold vt10%   SR {bh_m['sharpe']:+.3f}  CAGR {bh_m['cagr']:+.1f}%  DD {bh_m['dd']:+.1f}%")
    print(f"  champion (live)  SR {ch_m['sharpe']:+.3f}  CAGR {ch_m['cagr']:+.1f}%  DD {ch_m['dd']:+.1f}%")

    daily_rets = {}
    metrics = {}
    for p in ATR_PERIODS:
        for m in MULTIPLIERS:
            direction = supertrend_direction(df, p, m)
            fc_lo = direction.clip(lower=0.0)          # long-only variant
            mm, dd = engine(df, fc_lo, eval_start=PNL_EVAL_START)
            daily_rets[(p, m)] = dd
            metrics[(p, m)] = mm

    print(f"\n  {'atr_p':>6} {'mult':>5} {'SR':>7} {'CAGR':>7} {'DD':>7} {'turn':>7}")
    for (p, m), mm in sorted(metrics.items(), key=lambda kv: -kv[1]["sharpe"]):
        print(f"  {p:6d} {m:5.1f} {mm['sharpe']:+7.3f} {mm['cagr']:+6.1f}% {mm['dd']:+6.1f}% {mm['turnover']:7.1f}")

    M = pd.DataFrame(daily_rets).dropna(how="all").fillna(0.0)
    sharpes = {k: float(M[k].mean() / M[k].std() * np.sqrt(252)) if M[k].std() > 0 else np.nan for k in M.columns}
    best_key = max(sharpes, key=lambda k: sharpes[k] if np.isfinite(sharpes[k]) else -1e9)
    best_p, best_m = best_key
    print(f"\n  best cell: atr_period={best_p} multiplier={best_m}  SR={sharpes[best_key]:+.4f}")

    trial_sharpes_daily = np.array([sharpes[k] / np.sqrt(252) for k in sharpes])
    dsr = deflated_sharpe_ratio(M[best_key].values, trial_sharpes_daily)
    pbo = pbo_cscv(M.values, n_partitions=10)
    print(f"  DSR (n_trials={dsr['n_trials']}) = {dsr['dsr']:.4f}   PBO = {pbo.pbo:.3f} (n_splits={pbo.n_splits})")

    print("\n=== STAGE 2: best cell — paired vs buy&hold and champion, per-year ===")
    best_direction = supertrend_direction(df, best_p, best_m)
    fc_lo = best_direction.clip(lower=0.0)
    fc_ls = best_direction  # long/short variant, for comparison
    for label, fc in (("long-only", fc_lo), ("long/short", fc_ls)):
        mm, dd = engine(df, fc, eval_start=PNL_EVAL_START)
        _, t_bh, _ = paired(dd, bh_d)
        _, t_ch, _ = paired(dd, ch_d)
        yp_bh, yn = per_year(dd, bh_d)
        yp_ch, _ = per_year(dd, ch_d)
        print(f"  {label:11s} SR {mm['sharpe']:+.3f}  CAGR {mm['cagr']:+.1f}%  DD {mm['dd']:+.1f}%  "
              f"turn {mm['turnover']:.1f}  t_vs_bh {t_bh:+.2f}  t_vs_champ {t_ch:+.2f}  "
              f"yrs+ vsBH {yp_bh}/{yn} vsChamp {yp_ch}/{yn}")

    print("\n=== STAGE 3: overlay — tilt the champion by the best SuperTrend cell ===")
    tilt = fc_lo.clip(0, 1).reindex(df.index).fillna(0.0)
    for b in (0.25, 0.5, 1.0):
        mm, dd = engine(df, (champ_fc * (1.0 + b * (tilt - tilt.mean()))).clip(-2, 2), eval_start=PNL_EVAL_START)
        _, t_ch, _ = paired(dd, ch_d)
        yp_ch, yn = per_year(dd, ch_d)
        print(f"  tilt b={b:.2f}  SR {mm['sharpe']:+.3f}  CAGR {mm['cagr']:+.1f}%  DD {mm['dd']:+.1f}%  "
              f"t_vs_champ {t_ch:+.2f}  yrs+ {yp_ch}/{yn}")


if __name__ == "__main__":
    main()
