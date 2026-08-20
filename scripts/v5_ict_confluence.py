"""ICT/SMC concept sweep, Phase 5: confluence combos. XAU H4/H1, quick sanity checks.

Plan: `~/.claude/plans/i-wanttt-you-to-playful-widget.md` — 6 bounded
ICT-canonical combos. Combo #2 (BOS+OTE entry) and #6 (Premium/Discount as a
champion filter) were ALREADY implemented as their own standalone tests
(`v5_ict_structure_xau.py`'s OTE-filtered-BOS, `v5_ict_premium_discount_filter.py`)
— not repeated here. The other four all have EVERY ingredient already
individually disproven in Phases 1-3, so per the plan's own rule ("a combo
with an already-failed Bucket-A ingredient gets one cheap sanity check only,
not a full pipeline") this file runs one full-sample Sharpe check per combo,
not the full mandatory pipeline — confluence occasionally rescues a
component, worth a quick look, not a re-investment.

    python scripts/v5_ict_confluence.py
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

from scripts.v5_xau_turn_prob import load_h4, engine  # noqa: E402
from src.features.ict_primitives import (  # noqa: E402
    swing_levels, break_of_structure, liquidity_sweep, session_windows,
)
from scripts.v5_ict_blocks_xau import breaker_fc, ifvg_fc  # noqa: E402
from scripts.v5_ict_sessions_xau import judas_fc, po3_fc  # noqa: E402
from scripts.v5_ict_smt_divergence import net_returns as smt_net_returns  # noqa: E402

EVAL_START = "2018-01-01"


def sanity(name: str, fc: pd.Series, df: pd.DataFrame) -> None:
    m, d = engine(df, fc, eval_start=EVAL_START)
    n_active = int((fc != 0).sum())
    print(f"  {name:45s} SR={m['sharpe']:+.3f}  DD={m['dd']:+.1f}%  active_bars={n_active}")


def main():
    df = load_h4(0.448)

    print("=== Combo 1: Unicorn (Breaker Block ^ IFVG agreement) ===")
    b = breaker_fc(df, 2.0, 1000)
    f = ifvg_fc(df, 2.0, 1000)
    agree = np.sign(b.values) == np.sign(f.values)
    fc1 = pd.Series(np.where(agree & (b.values != 0), np.sign(b.values), 0.0), index=df.index)
    sanity("Unicorn (breaker & ifvg agree)", fc1, df)

    print("\n=== Combo 3: Judas Swing + BOS confirmation (H4 approx of the H1 Judas signal) ===")
    sh, sl = swing_levels(df, order=5)
    bos = break_of_structure(df, sh, sl)
    # H4 approximation of the H1 Judas signal's direction (session-anchored reversal
    # isn't meaningful to recompute at H4 resolution — reuse liquidity_sweep as the
    # nearest H4-native "fakeout+reversal" proxy for this quick confluence check).
    sweep = liquidity_sweep(df, sh, sl)
    fc3 = pd.Series(np.where((sweep.values != 0) & (bos.values == np.sign(sweep.values)),
                             sweep.values, 0.0), index=df.index)
    sanity("Judas-proxy(sweep) confirmed by BOS", fc3, df)

    print("\n=== Combo 4: SMT Divergence + liquidity sweep (SPX/NDX, D1) ===")
    from scripts.v5_ict_smt_divergence import load_ohlc, smt_fc
    df_a, df_b = load_ohlc("SPX"), load_ohlc("NDX")
    j = df_a.index.intersection(df_b.index)
    smt = smt_fc(df_a.loc[j], df_b.loc[j], order=5, hold_bars=5)
    sh_a, sl_a = swing_levels(df_a.loc[j], order=5)
    sweep_a = liquidity_sweep(df_a.loc[j], sh_a, sl_a)
    fc4 = pd.Series(np.where((smt.values != 0) & (sweep_a.values == smt.values), smt.values, 0.0), index=j)
    import scripts.v5_basket_challenge as vbc
    close = df_a.loc[j, "close"]
    spread_px = df_a.loc[j, "spread"].clip(lower=df_a.loc[j, "spread"].median())
    ret = close.pct_change()
    vol = ret.ewm(halflife=42, min_periods=20).std() * np.sqrt(252)
    pos = vbc._buffered_pos(fc4, vol, spread_px, close, 252).shift(1).fillna(0.0)
    cost = pos.diff().abs().fillna(0.0) * (spread_px / close)
    net = (pos * ret - cost).fillna(0.0).resample("D").sum()
    net = net[net.index.dayofweek < 5].loc[EVAL_START:]
    sr4 = float(net.mean() / net.std() * np.sqrt(252)) if net.std() > 0 else float("nan")
    e4 = (1 + net).cumprod()
    dd4 = float((e4 / e4.cummax() - 1).min() * 100)
    print(f"  {'SMT confirmed by liquidity sweep (SPX)':45s} SR={sr4:+.3f}  DD={dd4:+.1f}%  "
          f"active_days={int((fc4.reindex(j).fillna(0)!=0).sum())}")

    print("\n=== Combo 5: PO3/AMD session-filtered BOS (NY-open-only BOS) ===")
    sess = session_windows(df)
    fc5 = bos.where(sess["ny_open"].values, 0.0)
    sanity("BOS filtered to NY-open window only", fc5, df)

    print("\n=== Verdict ===")
    print("All four combos are quick sanity checks (single config, full-sample only,")
    print("no walk-forward/DSR/PBO) per the plan's rule for combos where every")
    print("ingredient already failed individually — a positive number here would")
    print("warrant a full pipeline re-investment; a negative/near-zero one does not.")


if __name__ == "__main__":
    main()
