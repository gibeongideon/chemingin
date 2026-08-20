"""ICT/SMC concept sweep — zone-based concepts. XAU H4.

Plan: `~/.claude/plans/i-wanttt-you-to-playful-widget.md`.

  Breaker Block (Phase 1, Bucket A)  — an Order Block (P4) that later gets
      breached and, per ICT's standard reading, flips role and trades as S/R
      in the OPPOSITE direction. `zone_lifecycle` (P6) already tracks this
      flip; the signal here is `nearest_zone_dir` itself (the nearest active
      zone's CURRENT, possibly-flipped, role), proximity-weighted.

  Mitigation Block (Phase 2, Bucket B) — a DIFFERENT hypothesis from Breaker
      Block, not a near-duplicate: Breaker bets on CONTINUATION after a zone
      flips role (tested, failed badly, SR -0.67/DD -42%). Mitigation instead
      bets on REJECTION *at* a zone that is STILL ACTIVE (not yet broken) —
      the classic "smart money re-enters to mitigate a losing position"
      read, i.e. trade the zone's ORIGINAL (unflipped) direction, and only
      while price is actually inside its bounds, not proximity-weighted.

  IFVG (Phase 2, Bucket B) — mechanistically identical to Breaker Block
      (same `zone_lifecycle` flip-and-continue read) but fed by Fair Value
      Gaps (P5, bug-fixed 2026-08-19) instead of Order Blocks (P4) — a
      different, information-denser zone source worth testing on its own
      merits even though the OB-sourced version failed.

    python scripts/v5_ict_blocks_xau.py
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
from src.features.smc_signals import order_blocks, fair_value_gaps  # noqa: E402
from src.features.ict_primitives import zone_lifecycle, REGIME_BOUNDARIES  # noqa: E402
from src.evaluation.lookahead_probe import probe_lookahead, print_report  # noqa: E402
from src.evaluation.dsr_pbo import deflated_sharpe_ratio, pbo_cscv  # noqa: E402
from src.evaluation.walk_forward_grid import run_concept_pipeline, print_summary_table  # noqa: E402

PNL_EVAL_START = "2018-01-01"
YEARS = list(range(2018, 2027))
SELECT_WINDOW_BARS = 3 * 252 * 6

SCALES = (1.0, 2.0, 4.0)
EXPIRY_BARS = (500, 1000, 2000)
MB_HOLD_BARS = (4, 8, 16)
IFVG_SCALES = (1.0, 2.0, 4.0)
IFVG_EXPIRY_BARS = (500, 1000, 2000)


def ob_zones(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Adapt `order_blocks()`'s 0/1 formation flags into `zone_lifecycle`'s
    (formed, top, bottom, kind) shape. The order-block candle is bar i-1 (the
    last opposite-direction bar before the confirming breakout at bar i, per
    `order_blocks()`'s own docstring) — its [low,high] range is the zone,
    known causally at bar i via `.shift(1)`."""
    ob_bull, ob_bear = order_blocks(df)
    m_bull, m_bear = ob_bull.astype(bool), ob_bear.astype(bool)
    high_prev, low_prev = df["high"].shift(1), df["low"].shift(1)
    formed = m_bull | m_bear
    top = pd.Series(np.nan, index=df.index)
    bottom = pd.Series(np.nan, index=df.index)
    kind = pd.Series(0.0, index=df.index)
    top[m_bull], bottom[m_bull], kind[m_bull] = high_prev[m_bull], low_prev[m_bull], 1.0
    top[m_bear], bottom[m_bear], kind[m_bear] = high_prev[m_bear], low_prev[m_bear], -1.0
    return formed, top, bottom, kind


def breaker_fc(df: pd.DataFrame, scale: float, expiry_bars: int) -> pd.Series:
    formed, top, bottom, kind = ob_zones(df)
    z = zone_lifecycle(df, formed, top, bottom, kind, expiry_bars=expiry_bars)
    dist = z["nearest_zone_dist_atr"].fillna(np.inf)
    weight = np.exp(-dist.values / scale)
    return z["nearest_zone_dir"] * pd.Series(weight, index=df.index)


def mitigation_fc(df: pd.DataFrame, expiry_bars: int, hold_bars: int) -> pd.Series:
    """Fade AT a still-active (not-yet-flipped) order block: `nearest_zone_dir`
    while `in_active_zone` (inside the zone's bounds, not just near it) equals
    the zone's original formation direction — since `zone_lifecycle` only
    flips `nearest_zone_dir` on a confirmed break, `in_active_zone==1` implies
    the zone hasn't flipped yet (a broken zone that's later re-approached
    would show as a NEW active zone in its flipped kind, which is exactly the
    Breaker read, not this one). Persist `hold_bars` after leaving the zone
    (avoids single-bar flicker), reset on the opposite side."""
    formed, top, bottom, kind = ob_zones(df)
    z = zone_lifecycle(df, formed, top, bottom, kind, expiry_bars=expiry_bars)
    raw = (z["nearest_zone_dir"] * z["in_active_zone"]).values
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
    return pd.Series(state, index=df.index, name="mitigation_fc")


def fvg_zones(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Adapt `fair_value_gaps()` (bug-fixed 2026-08-19) into `zone_lifecycle`'s
    (formed, top, bottom, kind) shape. Bull FVG gap = [high[i-2], low[i]]
    (per its own docstring), confirmed and known at bar i; mirror for bear."""
    fvg_bull, fvg_bear, _ = fair_value_gaps(df)
    m_bull, m_bear = fvg_bull.astype(bool), fvg_bear.astype(bool)
    high_prev2, low_prev2 = df["high"].shift(2), df["low"].shift(2)
    low_now, high_now = df["low"], df["high"]
    formed = m_bull | m_bear
    top = pd.Series(np.nan, index=df.index)
    bottom = pd.Series(np.nan, index=df.index)
    kind = pd.Series(0.0, index=df.index)
    top[m_bull], bottom[m_bull], kind[m_bull] = low_now[m_bull], high_prev2[m_bull], 1.0
    top[m_bear], bottom[m_bear], kind[m_bear] = low_prev2[m_bear], high_now[m_bear], -1.0
    return formed, top, bottom, kind


def ifvg_fc(df: pd.DataFrame, scale: float, expiry_bars: int) -> pd.Series:
    """Inversion FVG: same flip-and-continue read as `breaker_fc`, FVG-sourced
    zones instead of order blocks."""
    formed, top, bottom, kind = fvg_zones(df)
    z = zone_lifecycle(df, formed, top, bottom, kind, expiry_bars=expiry_bars)
    dist = z["nearest_zone_dist_atr"].fillna(np.inf)
    weight = np.exp(-dist.values / scale)
    return z["nearest_zone_dir"] * pd.Series(weight, index=df.index)


def main():
    print("=== STAGE 0: lookahead probes ===")
    results = {
        "breaker(scale=2,exp=1000)": probe_lookahead(
            lambda d: breaker_fc(d, 2.0, 1000), n=900, offsets=(60, 40, 20, 10), warmup_buffer=400
        ),
        "mitigation(exp=1000,hold=8)": probe_lookahead(
            lambda d: mitigation_fc(d, 1000, 8), n=900, offsets=(60, 40, 20, 10), warmup_buffer=400
        ),
        "ifvg(scale=2,exp=1000)": probe_lookahead(
            lambda d: ifvg_fc(d, 2.0, 1000), n=900, offsets=(60, 40, 20, 10), warmup_buffer=400
        ),
    }
    print_report(results)
    if not all(v.ok for v in results.values()):
        print("!! lookahead probe FAILED — refusing to trust the backtest below.")
        return

    df = load_h4(0.448)
    champ_fc = champion_signal(df["close"])
    bh_m, bh_d = engine(df, pd.Series(1.0, index=df.index), eval_start=PNL_EVAL_START)
    ch_m, ch_d = engine(df, champ_fc, eval_start=PNL_EVAL_START)

    def pipe(name, grid):
        return run_concept_pipeline(
            name, df, grid,
            engine=engine, paired=paired, per_year=per_year,
            deflated_sharpe_ratio=deflated_sharpe_ratio, pbo_cscv=pbo_cscv,
            bh_m=bh_m, bh_d=bh_d, ch_m=ch_m, ch_d=ch_d,
            regime_boundaries=REGIME_BOUNDARIES["XAUUSD"],
            pnl_eval_start=PNL_EVAL_START, years=YEARS, select_window_bars=SELECT_WINDOW_BARS,
        )

    results_all = []
    breaker_grid = {(s, e): (lambda s=s, e=e: breaker_fc(df, s, e)) for s in SCALES for e in EXPIRY_BARS}
    results_all.append(pipe("Breaker Block (standard reading)", breaker_grid))

    mb_grid = {(e, h): (lambda e=e, h=h: mitigation_fc(df, e, h)) for e in EXPIRY_BARS for h in MB_HOLD_BARS}
    results_all.append(pipe("Mitigation Block (fade at unflipped zone)", mb_grid))

    ifvg_grid = {(s, e): (lambda s=s, e=e: ifvg_fc(df, s, e)) for s in IFVG_SCALES for e in IFVG_EXPIRY_BARS}
    results_all.append(pipe("Inversion FVG (flip-and-continue)", ifvg_grid))

    print(f"\n{'='*70}\nSUMMARY (blocks)\n{'='*70}")
    print_summary_table(results_all)
    out = ROOT / "data" / "v5_runs" / "ict_blocks_xau_summary.csv"
    pd.DataFrame([{k: v for k, v in r.items() if k != "regimes"} for r in results_all]).to_csv(out, index=False)
    print(f"\nsummary -> {out}")


if __name__ == "__main__":
    main()
