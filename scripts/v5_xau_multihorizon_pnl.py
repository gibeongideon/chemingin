"""Option (b): does a LONGER horizon fix what killed the fwd6 signal? (2026-08-11)

`scripts/v5_xau_fwd6_regime_pnl.py` found the fwd6 (~1 trading day) price+regime
classifier loses money standalone (SR -0.44 live) and fails as a champion overlay,
and that gating to only the highest-confidence subset does NOT rescue it — the
paired t-stat vs buy&hold/champion stays significantly negative even at the
sparsest, most-accurate threshold tested (55.6% accuracy, n=24 signals/yr).

The diagnosed mechanism there was NOT (only) turnover: it was that trading a
~1-day mean-reversion-flavoured signal sits the position near-flat much of the
time, forfeiting XAUUSD's strong structural drift over 2018-2026 (champion CAGR
~18-19%) — the OPPORTUNITY COST outweighs the thin edge, at every threshold.

This script asks a different question: does the SAME regime feature recipe
(the one validated, block-bootstrap-confirmed accuracy edge) still show an edge
at LONGER forward horizons — 12/24/48/96 H4 bars (~2/4/8/16 trading days) as well
as the original 6 — and if so, does a longer horizon's naturally lower turnover
and (with CONTINUOUS, not gated, sizing) near-always-in exposure avoid the
opportunity-cost trap that killed fwd6? Same walk-forward discipline (expanding
yearly refit, purge=H), same P&L engine/costs/benchmarks as the fwd6 test.

    python scripts/v5_xau_multihorizon_pnl.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_xau_turn_prob import load_h4, engine, paired, per_year  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402
from scripts.v5_xau_intermarket_accuracy import (  # noqa: E402
    atr, price_features, regime_features, build_models, EVAL_START_YEAR,
)

PNL_EVAL_START = "2018-01-01"
HORIZONS = [6, 12, 24, 48, 96]   # ~1, 2, 4, 8, 16 trading days at H4


def make_fwd_target(close: pd.Series, h: int) -> tuple[np.ndarray, np.ndarray]:
    """sign(close[t+h]-close[t]) as {0,1}; -1 sentinel where no forward data."""
    fwd_ret = close.shift(-h) - close
    y = (fwd_ret > 0).astype(int)
    y[fwd_ret.isna()] = -1
    persist = np.sign(close - close.shift(h)).fillna(0).clip(lower=0).astype(int).values
    return y.values, persist


def oos_probabilities(df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray, h: int,
                       years: list[int]) -> pd.Series:
    """Same expanding-per-year walk-forward as fwd6, purge=h (this horizon's
    own label window, not the fwd6 default)."""
    zoo = build_models(["histgb"])
    _, model = zoo["histgb"]
    idx = X.index
    valid = np.isfinite(X.values).all(axis=1) & (y >= 0)
    p_series = pd.Series(np.nan, index=idx)
    for yr in years:
        test_mask = (idx.year == yr) & valid
        if test_mask.sum() == 0:
            continue
        cutoff = pd.Timestamp(f"{yr}-01-01")
        train_end_pos = np.searchsorted(idx.values, np.datetime64(cutoff)) - h
        train_mask = np.zeros(len(idx), dtype=bool)
        train_mask[:train_end_pos] = True
        train_mask &= valid
        if train_mask.sum() < 200 or len(np.unique(y[train_mask])) < 2:
            continue
        m = model.__class__(**model.get_params())
        m.fit(X.values[train_mask], y[train_mask])
        p_series.loc[test_mask] = m.predict_proba(X.values[test_mask])[:, 1]
    return p_series


def main():
    print("Loading XAUUSD H4...")
    df0 = load_h4(None)
    a = atr(df0)
    price = price_features(df0)
    reg = regime_features(df0, a)
    X = pd.concat([price, reg], axis=1)
    years = list(range(EVAL_START_YEAR, df0.index.year.max() + 1))
    close0 = df0["close"]

    df = load_h4(0.448)   # live cost throughout — the decision-relevant number
    close = df["close"]
    champ_fc = champion_signal(close)
    bh_m, bh_d = engine(df, pd.Series(1.0, index=df.index), eval_start=PNL_EVAL_START)
    ch_m, ch_d = engine(df, champ_fc, eval_start=PNL_EVAL_START)
    print(f"benchmarks (live $0.448):  buy&hold-vt SR {bh_m['sharpe']:+.3f}  "
          f"CAGR {bh_m['cagr']:+.1f}%   champion SR {ch_m['sharpe']:+.3f}  "
          f"CAGR {ch_m['cagr']:+.1f}%  DD {ch_m['dd']:+.1f}%\n")

    hdr = (f"{'H(bars)':>7} {'~days':>6} {'n':>6} {'acc':>7} {'auc':>6} {'persist':>8} {'|':>1} "
          f"{'SR':>7} {'CAGR':>7} {'DD':>7} {'turn':>7} {'avg|pos|':>9} "
          f"{'t_vs_bh':>8} {'t_vs_champ':>11} {'yrs+bh':>7} {'yrs+ch':>7}")
    print(hdr)
    print("-" * len(hdr))

    for h in HORIZONS:
        y, persist = make_fwd_target(close0, h)
        p = oos_probabilities(df0, X, y, h, years)

        valid = np.isfinite(X.values).all(axis=1) & (y >= 0)
        idx = X.index
        mask = valid & p.notna()
        yv = y[mask.values]
        pv = p[mask].values
        persv = persist[mask.values]
        acc = float(((pv >= 0.5).astype(int) == yv).mean())
        auc = float(roc_auc_score(yv, pv)) if len(np.unique(yv)) > 1 else np.nan
        persist_acc = float((persv == yv).mean())

        fc = ((p - 0.5) * 4.0).clip(-2, 2).fillna(0.0).reindex(df.index).fillna(0.0)
        m, d = engine(df, fc, eval_start=PNL_EVAL_START)
        _, t_bh, _ = paired(d, bh_d)
        _, t_ch, _ = paired(d, ch_d)
        yp_bh, yn = per_year(d, bh_d)
        yp_ch, _ = per_year(d, ch_d)

        print(f"{h:7d} {h/6:6.1f} {mask.sum():6d} {acc*100:6.2f}% {auc:6.3f} "
              f"{persist_acc*100:7.2f}% {'|':>1} "
              f"{m['sharpe']:+7.3f} {m['cagr']:+6.1f}% {m['dd']:+6.1f}% {m['turnover']:7.1f} "
              f"{m['avg_pos']:9.2f} {t_bh:+8.2f} {t_ch:+11.2f} {yp_bh:4d}/{yn} {yp_ch:4d}/{yn}")


if __name__ == "__main__":
    main()
