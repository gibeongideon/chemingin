"""Does the fwd6/regime accuracy edge survive as P&L? (2026-08-11)

`scripts/v5_xau_intermarket_accuracy.py` found ONE real, block-bootstrap-confirmed
accuracy edge in six rounds of XAU turning-point/trend research: price+regime
features / HistGBoost predict the next ~1 trading day's direction (fwd6) at 52.0%
vs a 49.8% persistence baseline (+2.12pp, positive 8/9 years). "Detection accuracy
!= trading edge" is this repo's own repeated lesson
(`xau-turning-point-detectability`, `oracle-ceiling-control-tops-not-bottoms`) —
this script asks the P&L question honestly, reusing the SAME engine/cost
machinery as `v5_xau_turn_prob.py` so results are directly comparable to the
champion (eval SR ~1.04) and buy&hold-vt benchmarks already on record.

Three stages, same convention as v5_xau_turn_prob.py (paired daily-delta t-stat
+ per-year win count, at TWO cost levels: blessed-CSV and live $0.448 floor):

  STAGE 0  ORACLE — trade the ACTUAL fwd6 sign with perfect hindsight, standalone.
           Calibrates the mechanics/turnover of this signal type & horizon before
           trusting anything built on top of it (V5_FINDINGS MANDATORY CONTROL #5).
  STAGE 1  HONEST STANDALONE — the real out-of-sample walk-forward probability
           (re-run here to capture per-date values, not just pooled accuracy)
           turned into a forecast, traded on its own. Continuous AND binary
           (+-1 unit conviction) forecast mappings, since HistGBoost's predict_proba
           is not a calibrated probability and a linear scaling could over/under-size.
  STAGE 2  OVERLAY — the same honest signal as a tilt on the existing long-only
           champion (the pattern every other "does this detector make money"
           question in this repo uses), since a standalone failure doesn't rule
           out value as a timing filter.

    python scripts/v5_xau_fwd6_regime_pnl.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_xau_turn_prob import load_h4, engine, paired, per_year  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402
from scripts.v5_xau_intermarket_accuracy import (  # noqa: E402
    atr, price_features, regime_features, make_targets, build_models,
    FWD_H, EVAL_START_YEAR,
)

PNL_EVAL_START = "2018-01-01"   # matches the walk-forward's own OOS window


def oos_probabilities(df: pd.DataFrame, years: list[int]) -> pd.Series:
    """Re-run the EXACT price+regime/HistGBoost walk-forward (the round-1
    champion config, same expanding-per-year refit + purge) but keep the
    per-date OOS probability instead of only pooled accuracy metrics, so it
    can be turned into a tradeable forecast."""
    a = atr(df)
    price = price_features(df)
    reg = regime_features(df, a)
    X = pd.concat([price, reg], axis=1)
    targets, _ = make_targets(df, a)
    y = targets["fwd6"]
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
        train_end_pos = np.searchsorted(idx.values, np.datetime64(cutoff)) - FWD_H
        train_mask = np.zeros(len(idx), dtype=bool)
        train_mask[:train_end_pos] = True
        train_mask &= valid
        if train_mask.sum() < 200 or len(np.unique(y[train_mask])) < 2:
            continue
        m = model.__class__(**model.get_params())
        m.fit(X.values[train_mask], y[train_mask])
        p_series.loc[test_mask] = m.predict_proba(X.values[test_mask])[:, 1]
    return p_series


def confidence_sweep(df0: pd.DataFrame, p: pd.Series, true_sign: pd.Series,
                     thresholds: list[float]) -> None:
    """STAGE 1b — option (a) from the P&L verdict: only trade the classifier's
    highest-confidence subset (|p-0.5| >= threshold), flat otherwise. Cuts
    turnover a lot AND raises accuracy on the traded subset (both measured
    below) — tests whether that combination clears the cost bar that the
    unfiltered signal (SR -0.24/-0.44) did not. Two forecast flavours per
    threshold: CONTINUOUS (magnitude still scales with confidence beyond the
    gate) and BINARY (full +-2 conviction once gated in, since the whole
    point of gating is "when confident, act on it").

    NOT block-bootstrapped: this is a threshold SWEEP (12 cells), and picking
    the best cell post-hoc without a held-out check is exactly the
    look-elsewhere trap flagged in the Hurst-window sweep. Read directionally,
    then verify the single most promising cell with `paired()`'s t-stat AND
    a sanity check that coverage/n is not so small the number is noise."""
    ok = p.notna() & (true_sign != 0)
    conf = (p - 0.5).abs()
    pred_sign = np.sign(p - 0.5)
    n_total = int(ok.sum())

    print("=== STAGE 1b: CONFIDENCE-GATED — trade only when |p-0.5| >= threshold ===")
    print(f"{'thr':>5} {'coverage':>9} {'n/yr':>6} {'subset_acc':>11} {'|':>1} "
          f"{'SR_cont':>8} {'SR_bin':>7} {'CAGR_cont':>10} {'DD_cont':>8} "
          f"{'turn_cont':>10} {'t_vs_bh':>8} {'t_vs_champ':>11} {'yrs+':>6}")

    df = load_h4(0.448)   # live cost — the decision-relevant number
    close = df["close"]
    champ_fc = champion_signal(close)
    bh_m, bh_d = engine(df, pd.Series(1.0, index=df.index), eval_start=PNL_EVAL_START)
    ch_m, ch_d = engine(df, champ_fc, eval_start=PNL_EVAL_START)

    for thr in thresholds:
        m = ok & (conf.reindex(p.index).fillna(0) >= thr)
        n = int(m.sum())
        if n < 30:
            print(f"{thr:5.2f}  too few signals (n={n}) to evaluate")
            continue
        subset_acc = float((pred_sign[m] == true_sign[m]).mean())

        raw = ((p - 0.5) * 4.0).clip(-2, 2)
        fc_cont = raw.where(conf >= thr, 0.0).fillna(0.0).reindex(df.index).fillna(0.0)
        fc_bin = (np.sign(p - 0.5) * 2.0).where(conf >= thr, 0.0).fillna(0.0) \
            .reindex(df.index).fillna(0.0)

        mc, dc = engine(df, fc_cont, eval_start=PNL_EVAL_START)
        mb, _ = engine(df, fc_bin, eval_start=PNL_EVAL_START)
        _, t_bh, _ = paired(dc, bh_d)
        _, t_ch, _ = paired(dc, ch_d)
        yp_ch, yn = per_year(dc, ch_d)

        print(f"{thr:5.2f} {n/n_total*100:8.1f}% {n/9:6.0f} {subset_acc*100:10.2f}% {'|':>1} "
              f"{mc['sharpe']:+8.3f} {mb['sharpe']:+7.3f} {mc['cagr']:+9.1f}% "
              f"{mc['dd']:+7.1f}% {mc['turnover']:9.1f}  {t_bh:+7.2f} {t_ch:+10.2f} "
              f"{yp_ch:3d}/{yn}")


def main():
    print("Loading XAUUSD H4 (turn_prob's honest expanding-median spread floor)...")
    df0 = load_h4(None)
    years = list(range(EVAL_START_YEAR, df0.index.year.max() + 1))
    p = oos_probabilities(df0, years)
    n_oos = int(p.notna().sum())
    print(f"  OOS probabilities: {n_oos} bars, {p.first_valid_index().date()} "
          f"..{p.last_valid_index().date()}\n")

    close0 = df0["close"]
    fwd_ret = close0.shift(-FWD_H) / close0 - 1.0
    true_sign = np.sign(fwd_ret).fillna(0.0)

    if "--confidence-sweep" in sys.argv:
        confidence_sweep(df0, p, true_sign,
                         [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30])
        return

    fc_oracle = (true_sign * 2.0).clip(-2, 2)
    fc_honest_cont = ((p - 0.5) * 4.0).clip(-2, 2).fillna(0.0)
    fc_honest_bin = np.sign(p - 0.5).fillna(0.0)   # unit conviction, not max lev

    for cost_usd, tag in ((None, "blessed CSV"), (0.448, "live $0.448 floor")):
        df = load_h4(cost_usd)
        close = df["close"]
        champ_fc = champion_signal(close)
        bh_m, bh_d = engine(df, pd.Series(1.0, index=df.index), eval_start=PNL_EVAL_START)
        ch_m, ch_d = engine(df, champ_fc, eval_start=PNL_EVAL_START)
        print(f"=== cost = {tag} ===  eval from {PNL_EVAL_START}  "
              f"(mean spread ${df['spread_px'].mean():.4f}/oz)")
        print(f"  {'buy&hold vol-targeted 10%':32s} SR {bh_m['sharpe']:+.3f}  "
              f"CAGR {bh_m['cagr']:+6.1f}%  DD {bh_m['dd']:+6.1f}%  turn {bh_m['turnover']:7.1f}")
        print(f"  {'CHAMPION (incumbent, live)':32s} SR {ch_m['sharpe']:+.3f}  "
              f"CAGR {ch_m['cagr']:+6.1f}%  DD {ch_m['dd']:+6.1f}%  turn {ch_m['turnover']:7.1f}"
              "   <-- deltas below vs this")

        for name, fc in (
            ("ORACLE fwd6 (perfect hindsight)", fc_oracle),
            ("HONEST standalone (continuous)", fc_honest_cont),
            ("HONEST standalone (binary +-1)", fc_honest_bin),
        ):
            fc = fc.reindex(df.index).fillna(0.0)
            m, d = engine(df, fc, eval_start=PNL_EVAL_START)
            _, t_bh, _ = paired(d, bh_d)
            _, t_ch, _ = paired(d, ch_d)
            yp_bh, yn = per_year(d, bh_d)
            yp_ch, _ = per_year(d, ch_d)
            print(f"  {name:32s} SR {m['sharpe']:+.3f}  CAGR {m['cagr']:+6.1f}%  "
                  f"DD {m['dd']:+6.1f}%  turn {m['turnover']:7.1f}  "
                  f"t_vs_bh {t_bh:+5.2f}  t_vs_champ {t_ch:+5.2f}  "
                  f"yrs+ vsBH {yp_bh}/{yn}  vsChamp {yp_ch}/{yn}")

        tilt = ((p - 0.5) * 2.0).clip(-1, 1).reindex(df.index).fillna(0.0)
        for b in (0.25, 1.0):
            m, d = engine(df, (champ_fc * (1.0 + b * tilt)).clip(-2, 2), eval_start=PNL_EVAL_START)
            _, t_ch, _ = paired(d, ch_d)
            yp_ch, yn = per_year(d, ch_d)
            print(f"  {'OVERLAY tilt b=' + str(b):32s} SR {m['sharpe']:+.3f}  "
                  f"CAGR {m['cagr']:+6.1f}%  DD {m['dd']:+6.1f}%  turn {m['turnover']:7.1f}  "
                  f"{'':11}t_vs_champ {t_ch:+5.2f}  {'':17}vsChamp {yp_ch}/{yn}")
        print()


if __name__ == "__main__":
    main()
