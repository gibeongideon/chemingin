"""ICT/SMC concept sweep, Phase 1 (Bucket A — expect bull-beta replication):
BOS baseline, Displacement, OTE-filtered BOS. XAU H4.

Plan: `~/.claude/plans/i-wanttt-you-to-playful-widget.md`. BOS itself was
already disproven (`v5_smc_xau.py`, V5_FINDINGS §3o — walk-forward SR 0.54,
loses money 2021-22 flat). It's reproduced here ONLY as the regime-split
comparison reference for the two genuinely new concepts in this file, using
the SAME continuous vol-target `engine()` (not the old discrete SL/TP
simulator) so all three are directly comparable.

  Displacement  — P9: a large ATR-normalized-body candle sets a directional
                  state, held until the next opposite displacement event or
                  `hold_bars` bars elapse (ICT teaches displacement as a short
                  burst of institutional momentum, not an indefinite hold).
  OTE-filtered  — P1+P2+P8: BOS signal, full size only if price spent time in
  BOS             the classic 62-79% OTE retracement zone of the CURRENT
                  swing range in the `lookback` bars before the break fires
                  (the ICT-canonical "buy the discount pullback, breakout
                  confirms" reading) — reduced size otherwise.

    python scripts/v5_ict_structure_xau.py
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
from src.features.ict_primitives import (  # noqa: E402
    swing_levels, break_of_structure, displacement, ote_zone, equal_levels,
    liquidity_sweep, REGIME_BOUNDARIES,
)
from src.features.smc_signals import _atr14, _ohlcv  # noqa: E402
from src.evaluation.lookahead_probe import probe_lookahead, print_report  # noqa: E402
from src.evaluation.dsr_pbo import deflated_sharpe_ratio, pbo_cscv  # noqa: E402
from src.evaluation.walk_forward_grid import run_concept_pipeline, print_summary_table  # noqa: E402

PNL_EVAL_START = "2018-01-01"
YEARS = list(range(2018, 2027))
SELECT_WINDOW_BARS = 3 * 252 * 6   # trailing 3y, H4
BOS_ORDERS = (3, 5, 8)
DISP_ATR_MULTS = (1.5, 2.0, 2.5, 3.0)
DISP_HOLD_BARS = (6, 12, 24, 48)
OTE_ORDERS = (3, 5, 8)
OTE_REDUCTIONS = (0.0, 0.25, 0.5)
OTE_LOOKBACKS = (5, 10, 20)

EQHL_TOL_ATRS = (0.1, 0.2, 0.3)
EQHL_ORDERS = (3, 5, 8)
EQHL_HOLD_BARS = (4, 8, 16)


# ============================================================== forecasts
def bos_fc(df: pd.DataFrame, order: int) -> pd.Series:
    sh, sl = swing_levels(df, order=order)
    return break_of_structure(df, sh, sl).astype(float)


def displacement_fc(df: pd.DataFrame, atr_mult: float, hold_bars: int) -> pd.Series:
    disp = displacement(df, atr_mult=atr_mult).values
    n = len(disp)
    state = np.zeros(n)
    cur, since = 0.0, 10**9
    for i in range(n):
        if disp[i] != 0:
            cur, since = disp[i], 0
        else:
            since += 1
            if since > hold_bars:
                cur = 0.0
        state[i] = cur
    return pd.Series(state, index=df.index, name="displacement_fc")


def bos_ote_fc(df: pd.DataFrame, order: int, reduction: float, lookback: int) -> pd.Series:
    sh, sl = swing_levels(df, order=order)
    bos = break_of_structure(df, sh, sl)
    in_ote_bull = ote_zone(df, sh, sl, direction=1)
    in_ote_bear = ote_zone(df, sh, sl, direction=-1)
    recently_bull = in_ote_bull.rolling(lookback, min_periods=1).max().astype(bool).values
    recently_bear = in_ote_bear.rolling(lookback, min_periods=1).max().astype(bool).values
    b = bos.values
    gate = np.ones(len(df))
    gate[(b > 0) & ~recently_bull] = reduction
    gate[(b < 0) & ~recently_bear] = reduction
    return bos * pd.Series(gate, index=df.index)


def eqhl_fade_fc(df: pd.DataFrame, tol_atr: float, order: int, hold_bars: int) -> pd.Series:
    """Equal Highs/Lows liquidity-pool fade (Bucket B): only take a
    liquidity-sweep (P3) signal when the level being swept was itself an
    EQUAL level (P1's `equal_levels` — two near-duplicate swing points, the
    literal ICT 'liquidity resting at equal highs/lows' read), a stricter
    filter than raw `liquidity_sweep` alone (already implicitly tested as
    `v5_smc_xau.py`'s noisy 'sweep' mode)."""
    o, h, l, c, idx = _ohlcv(df)
    a = pd.Series(_atr14(h, l, c), index=idx)
    sh, sl = swing_levels(df, order=order)
    eq_hi = equal_levels(sh, tol_atr, a)
    eq_lo = equal_levels(sl, tol_atr, a)
    sweep = liquidity_sweep(df, sh, sl)
    s = sweep.values
    raw = np.where((s > 0) & eq_lo.values, 1.0, np.where((s < 0) & eq_hi.values, -1.0, 0.0))
    n = len(df)
    state = np.zeros(n)
    cur, since = 0.0, 10**9
    for i in range(n):
        if raw[i] != 0:
            cur, since = raw[i], 0
        else:
            since += 1
            if since > hold_bars:
                cur = 0.0
        state[i] = cur
    return pd.Series(state, index=df.index, name="eqhl_fade_fc")


# ============================================================== pipeline
def run_concept(name: str, df: pd.DataFrame, grid_fns: dict, champ_fc: pd.Series,
                bh_m: dict, bh_d: pd.Series, ch_m: dict, ch_d: pd.Series) -> dict:
    return run_concept_pipeline(
        name, df, grid_fns,
        engine=engine, paired=paired, per_year=per_year,
        deflated_sharpe_ratio=deflated_sharpe_ratio, pbo_cscv=pbo_cscv,
        bh_m=bh_m, bh_d=bh_d, ch_m=ch_m, ch_d=ch_d,
        regime_boundaries=REGIME_BOUNDARIES["XAUUSD"],
        pnl_eval_start=PNL_EVAL_START, years=YEARS, select_window_bars=SELECT_WINDOW_BARS,
    )


def main():
    print("=== STAGE 0: lookahead probes ===")
    results = {
        "bos(order=5)": probe_lookahead(lambda d: bos_fc(d, 5), n=900, offsets=(60, 40, 20, 10), warmup_buffer=400),
        "displacement(2.0,12)": probe_lookahead(lambda d: displacement_fc(d, 2.0, 12),
                                                n=900, offsets=(60, 40, 20, 10), warmup_buffer=400),
        "bos_ote(5,0.25,10)": probe_lookahead(lambda d: bos_ote_fc(d, 5, 0.25, 10),
                                              n=900, offsets=(60, 40, 20, 10), warmup_buffer=400),
        "eqhl_fade(5,0.2,8)": probe_lookahead(lambda d: eqhl_fade_fc(d, 0.2, 5, 8),
                                              n=900, offsets=(60, 40, 20, 10), warmup_buffer=400),
    }
    print_report(results)
    if not all(v.ok for v in results.values()):
        print("!! lookahead probe FAILED — refusing to trust the backtest below.")
        return

    df = load_h4(0.448)
    champ_fc = champion_signal(df["close"])
    bh_m, bh_d = engine(df, pd.Series(1.0, index=df.index), eval_start=PNL_EVAL_START)
    ch_m, ch_d = engine(df, champ_fc, eval_start=PNL_EVAL_START)

    all_results = []

    bos_grid = {("order", o): (lambda o=o: bos_fc(df, o)) for o in BOS_ORDERS}
    all_results.append(run_concept("BOS baseline (regime-split reference)", df, bos_grid,
                                   champ_fc, bh_m, bh_d, ch_m, ch_d))

    disp_grid = {(a, h): (lambda a=a, h=h: displacement_fc(df, a, h))
                for a in DISP_ATR_MULTS for h in DISP_HOLD_BARS}
    all_results.append(run_concept("Displacement", df, disp_grid, champ_fc, bh_m, bh_d, ch_m, ch_d))

    ote_grid = {(o, r, l): (lambda o=o, r=r, l=l: bos_ote_fc(df, o, r, l))
               for o in OTE_ORDERS for r in OTE_REDUCTIONS for l in OTE_LOOKBACKS}
    all_results.append(run_concept("OTE-filtered BOS", df, ote_grid, champ_fc, bh_m, bh_d, ch_m, ch_d))

    eqhl_grid = {(t, o, h): (lambda t=t, o=o, h=h: eqhl_fade_fc(df, t, o, h))
                for t in EQHL_TOL_ATRS for o in EQHL_ORDERS for h in EQHL_HOLD_BARS}
    all_results.append(run_concept("EQH/EQL liquidity-pool fade", df, eqhl_grid, champ_fc, bh_m, bh_d, ch_m, ch_d))

    print(f"\n{'='*70}\nSUMMARY (structure, Phase 1+2)\n{'='*70}")
    print_summary_table(all_results)

    out = ROOT / "data" / "v5_runs" / "ict_phase1_structure_xau_summary.csv"
    pd.DataFrame([{k: v for k, v in r.items() if k != "regimes"} for r in all_results]).to_csv(out, index=False)
    print(f"\nsummary -> {out}")


if __name__ == "__main__":
    main()
