"""Regime-GATE the champion (not a new signal) — genuinely untried this session.

Every regime study so far ([[xau-regime-features-fwd-accuracy]]) fed regime
features (Hurst/ADX/Choppiness) into a NEW ML classifier trying to predict
direction — real accuracy edge, never survived as P&L. This is a different
application, borrowed from Forven's `regime_gate.py` concept (flagged in
FORVEN-COMPARISON.md, never implemented): don't predict anything new — GATE
the CHAMPION's own already-proven trend signal, cutting its size when a
trend-strength regime (ADX percentile-ranked, causal) says "not trending,"
full size when it does. The champion's own vol-target/DD-scaler already does
soft, reactive risk reduction; this tests an explicit, interpretable,
regime-conditioned size cut on top of it.

Learns directly from §3t: walk-forward SELECTION is built in from the start,
not bolted on after an exciting full-sample number. Every reported figure
below is the walk-forward-honest one.

    python scripts/v5_xau_regime_gate.py
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

from scripts.v5_xau_turn_prob import load_h4, engine, paired, per_year  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402
from src.features.indicators import adx as adx_fn  # noqa: E402
from src.evaluation.lookahead_probe import probe_lookahead, print_report  # noqa: E402

PNL_EVAL_START = "2018-01-01"
ADX_PERIOD = 14
LOOKBACK = 252 * 6          # ~252 trading days at H4 (6 bars/day)
THRESHOLDS = (30, 40, 50, 60, 70)   # ADX trailing-percentile cut
REDUCTIONS = (0.0, 0.25, 0.5, 0.75)  # forecast multiplier when "not trending"
YEARS = list(range(2018, 2027))
SELECT_WINDOW_BARS = 3 * 252 * 6    # trailing 3y, matching this repo's own precedent


def adx_percentile(df: pd.DataFrame, adx_period: int = ADX_PERIOD, lookback: int = LOOKBACK) -> pd.Series:
    a = adx_fn(df, adx_period)
    return a.rolling(lookback).apply(lambda x: (x[-1] > x).mean(), raw=True)


def gated_champion_fc(df: pd.DataFrame, threshold: float, reduction: float) -> pd.Series:
    champ_fc = champion_signal(df["close"])
    pctile = adx_percentile(df)
    gate = np.where(pctile.values >= (threshold / 100.0), 1.0, reduction)
    return champ_fc * pd.Series(gate, index=df.index)


def main():
    print("=== STAGE 0: lookahead probe ===")
    results = {
        "gated_champion(thr=50,red=0.25)": probe_lookahead(
            lambda d: gated_champion_fc(d, 50, 0.25), n=4200, offsets=(300, 200, 100, 50), warmup_buffer=1700
        )
    }
    print_report(results)
    if not all(v.ok for v in results.values()):
        print("!! lookahead probe FAILED — refusing to trust the backtest below.")
        return

    print("\n=== STAGE 1: precompute all (threshold, reduction) cells' daily returns ===")
    df = load_h4(0.448)
    close = df["close"]
    champ_fc = champion_signal(close)
    grid_rets = {}
    for thr in THRESHOLDS:
        for red in REDUCTIONS:
            fc = gated_champion_fc(df, thr, red)
            m, d = engine(df, fc, eval_start="2015-01-01")
            grid_rets[(thr, red)] = d

    idx = next(iter(grid_rets.values())).index
    bh_m, bh_d = engine(df, pd.Series(1.0, index=df.index), eval_start=PNL_EVAL_START)
    ch_m, ch_d = engine(df, champ_fc, eval_start=PNL_EVAL_START)
    print(f"buy&hold-vt SR={bh_m['sharpe']:+.3f}   champion SR={ch_m['sharpe']:+.3f}  DD={ch_m['dd']:+.1f}%")

    full_sample_sh = {k: float(v.loc[PNL_EVAL_START:].mean() / v.loc[PNL_EVAL_START:].std() * np.sqrt(252))
                      for k, v in grid_rets.items()}
    best_full = max(full_sample_sh, key=full_sample_sh.get)
    print(f"\n(reference only — NOT trusted) full-sample-best cell: thr={best_full[0]} red={best_full[1]}  "
          f"SR={full_sample_sh[best_full]:+.3f}")

    print("\n=== STAGE 2: WALK-FORWARD selection (trailing 3y, re-picked each Jan) — the honest number ===")
    oos_parts = []
    selections = {}
    for yr in YEARS:
        cutoff = pd.Timestamp(f"{yr}-01-01")
        train_end_pos = int(np.searchsorted(idx.values, np.datetime64(cutoff)))
        train_start_pos = max(0, train_end_pos - SELECT_WINDOW_BARS)
        if train_end_pos - train_start_pos < 100:
            continue
        train_sh = {}
        for k, ret in grid_rets.items():
            r = ret.iloc[train_start_pos:train_end_pos]
            train_sh[k] = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else -np.inf
        best_k = max(train_sh, key=train_sh.get)
        selections[yr] = (best_k, train_sh[best_k])
        test_mask = idx.year == yr
        oos_parts.append(grid_rets[best_k][test_mask])

    for yr, (k, sr_train) in selections.items():
        print(f"  {yr}: thr={k[0]:2d} red={k[1]:.2f}  (trailing-3y train SR={sr_train:+.2f})")

    oos = pd.concat(oos_parts).sort_index()
    sr_wf = float(oos.mean() / oos.std() * np.sqrt(252))
    e = (1 + oos).cumprod()
    dd_wf = float((e / e.cummax() - 1).min() * 100)
    print(f"\nWALK-FORWARD honest gated-champion: SR={sr_wf:+.3f}  DD={dd_wf:+.1f}%")
    print(f"vs CHAMPION alone (same window):     SR={ch_m['sharpe']:+.3f}  DD={ch_m['dd']:+.1f}%")

    print("\nper-year:")
    for yr, g in oos.groupby(oos.index.year):
        sr_y = float(g.mean() / g.std() * np.sqrt(252)) if g.std() > 0 else float("nan")
        g_ch = ch_d.loc[g.index]
        sr_ch_y = float(g_ch.mean() / g_ch.std() * np.sqrt(252)) if g_ch.std() > 0 else float("nan")
        print(f"  {yr}  gated {sr_y:+.2f}   champion {sr_ch_y:+.2f}   delta {sr_y - sr_ch_y:+.2f}")

    j = pd.concat([oos.rename("gated"), ch_d.rename("champ")], axis=1).dropna()
    delta = j.gated - j.champ
    t = float(delta.mean() / delta.std() * np.sqrt(len(delta)))
    yrp = (delta.groupby(delta.index.year).mean() > 0).sum()
    yrn = delta.groupby(delta.index.year).mean().shape[0]
    print(f"\npaired t (gated vs champion alone) = {t:+.2f}   years better {yrp}/{yrn}")

    import scripts.v5_basket_challenge as vbc
    m = vbc.MODELS["ftmo"]
    def fp(series):
        r10 = (series * (0.10 / (series.std() * np.sqrt(252)))).values
        return vbc.fp_sim(r10, m["vol"] / vbc.TARGET_VOL, day_safety=1.5, p1=m["p1"], p2=m["p2"],
                          dayloss=m["daily"], maxloss=m["maxloss"])
    sf = fp(j.champ)
    sg = fp(j.gated)
    print(f"\nFTMO pass-sim   champion alone: {sf['passpct']:.1f}%  median {sf['med_mo']:.1f}mo")
    print(f"FTMO pass-sim   gated champion: {sg['passpct']:.1f}%  median {sg['med_mo']:.1f}mo")


if __name__ == "__main__":
    main()
