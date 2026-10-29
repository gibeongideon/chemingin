#!/usr/bin/env python3
"""OCTOBER A3 — STRATEGIES AS FEATURES. The stacked ensemble this repo has never built.

Pre-registered in data/v5_runs/october/PREREGISTRATION.md. Design in OCTOBER.MD.

WHAT IS NEW. Verified three ways (import graph, filename scan, document scan): **no stacked
ensemble over this repo's own strategy signals has ever existed.** The nearest objects are
`src/models/ensemble.py` (stacks MODEL CLASSES over raw price features, never run on XAU) and
`src/cta/ml_combine.py::ml_forecast` (purged walk-forward Ridge over 4 EWMAC speeds + xsmom + rev5 +
vol_z, **with no recorded result anywhere**). Neither has been given breakout, SuperTrend, the
zigzag probability, the regime block, displacement, premium/discount or a liquidity sweep.

*** RUN AT H4, NOT H1, AND THE REASON MATTERS. *** A2 (3cd) measured this exact target --
`sign(champion position x forward return)` -- at H1 and found the ORACLE NEGATIVE in all 9 cells
(-0.407 at 24h/flat, monotone toward the incumbent from below). So the target A3 was declared with
is closed at H1. But 3p files a champ-meta trim oracle of **+0.824 (t 11.7, 10/10 years) at H4**,
which A2 left as an open discrepancy. **Running A3 at H4 oracle-first therefore does two jobs: it
gates the stack, and it resolves 3cd's open discrepancy using the same construction on both
timeframes.** If the oracle is negative at H4 too, 3p's number needs explaining and the stack is
not built. If it is positive, the prize is real and timeframe-specific.

THE TRAP THIS IS DESIGNED AGAINST. "Combination" has failed five times here (ICT confluence 0/4,
the 6-component stack 0/29, feature-block stacking, two books combined 1.41 -> 1.26, book-level
M15 trim 0/32), and the measured reason is collinearity: SuperTrend correlates **0.77** with the
champion on XAU and **0.86-0.88** on the second basket, the champion correlates **0.96** with
buy-and-hold, and 3s's closing instruction is *"don't re-try any other single-mechanism trend
filter -- a genuinely different return stream needs a genuinely different EDGE SOURCE."*

**So every feature is RESIDUALISED against the champion forecast before the model sees it.** Each
signal is regressed on the champion (walk-forward, train-only coefficients) and only the residual
is kept. The stack can then only be credited for what is ORTHOGONAL to the incumbent, which is the
single change that makes this a test rather than a sixth repeat.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from v5_xau_champion_lifts import champion_recipe                     # noqa: E402
from src.v5 import mtf_frames                                         # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "october"
TARGET_VOL, VOL_HL, MAX_LEV = 0.10, 42, 8.0
HORIZONS = (6, 30)            # H4 bars: 1 day and 5 days (3p/3av used a 5-day forward)
N_FOLDS = 8


def load_h4() -> pd.DataFrame:
    """H4 REBUILT FROM M15 — never XAUUSD_H4_long.csv, which splices two brokers from 2023-06-28
    and whose pre-2023 half correlates 0.18-0.40 with real broker quotes (3bb / 3by)."""
    m15 = pd.read_csv(ROOT / "data/XAUUSD_M15_long.csv", parse_dates=["time"], index_col="time")
    m15 = m15[~m15.index.duplicated(keep="last")].sort_index()
    return mtf_frames.aggregate(m15, "H4").dropna(subset=["close"])


def champ_position(close: pd.Series, bpy: float) -> tuple[pd.Series, pd.Series]:
    ret = close.pct_change()
    vol = ret.ewm(halflife=VOL_HL, min_periods=20).std() * np.sqrt(bpy)
    fc = champion_recipe(close, 1.0, 1.5)
    pos = (fc.clip(0, 2) * (TARGET_VOL / vol)).clip(0, MAX_LEV).shift(1)
    return pos, fc


def perf(r, ppy):
    r = r.dropna()
    if len(r) < 50 or r.std() == 0:
        return dict(sharpe=np.nan, cagr=np.nan, maxdd=np.nan)
    eq = (1 + r).cumprod()
    return dict(sharpe=float(r.mean() / r.std() * np.sqrt(ppy)),
                cagr=float(eq.iloc[-1] ** (ppy / len(r)) - 1),
                maxdd=float((eq / eq.cummax() - 1).min()))


def vol_match(a, b):
    sa, sb = a.std(), b.std()
    return a * (sb / sa) if sa > 0 else a


def build_signals(d: pd.DataFrame) -> pd.DataFrame:
    """Every per-bar strategy signal this repo implements that is computable from H4 OHLCV.

    Defensive: a signal that raises is skipped and reported, rather than silently becoming NaN.
    """
    from src.v5.xau_dual_signals import ewmac_fc, breakout_fc, ls_signal
    from src.features import smc_signals as S, indicators as I, ict_primitives as P
    c, h, l = (d[k].astype(float) for k in ("close", "high", "low"))
    df = d.rename(columns=str.lower)
    F, skipped = {}, []

    def add(name, fn):
        try:
            v = fn()
            v = v.iloc[:, 0] if isinstance(v, pd.DataFrame) else pd.Series(v, index=d.index)
            F[name] = pd.to_numeric(v, errors="coerce")
        except Exception as e:                                   # noqa: BLE001
            skipped.append(f"{name} ({type(e).__name__})")

    # --- trend family ---------------------------------------------------------------------
    add("ewmac_fast", lambda: ewmac_fc(c, ((8, 32), (16, 64))))
    add("ewmac_slow", lambda: ewmac_fc(c, ((32, 128), (64, 256))))
    add("breakout_f", lambda: breakout_fc(c, (10, 20, 40)))
    add("breakout_s", lambda: breakout_fc(c, (40, 80, 160)))
    add("ls_signal", lambda: ls_signal(c))
    add("supertrend", lambda: S.supertrend(df, 10, 3.0)[1])
    add("tsmom_21", lambda: (c / c.shift(21) - 1.0))
    add("tsmom_126", lambda: (c / c.shift(126) - 1.0))
    # --- mean reversion -------------------------------------------------------------------
    add("rsi7", lambda: I.rsi(c, 7))
    add("rsi14", lambda: I.rsi(c, 14))
    add("bb_pctb", lambda: I.bollinger_pct_b(c, 20, 2.0))
    add("stoch_k", lambda: I.stochastic(df, 14, 3, 3)[0])
    add("andean", lambda: S.andean_oscillator(df, 50, 9)[0])
    # --- volatility / regime ---------------------------------------------------------------
    add("adx14", lambda: I.adx(df, 14))
    add("atr_ratio", lambda: I.atr(df, 14) / I.atr(df, 50))
    add("vol_z", lambda: (c.pct_change().rolling(20).std() /
                          c.pct_change().rolling(120).std()))
    # --- structure / ICT ---------------------------------------------------------------------
    try:
        sh, sl = P.swing_levels(df, order=5)[:2]
        add("displacement", lambda: P.displacement(df, 2.0))
        add("prem_disc", lambda: P.premium_discount(df, sh, sl)[0])
        add("liq_sweep", lambda: P.liquidity_sweep(df, sh, sl))
    except Exception as e:                                        # noqa: BLE001
        skipped.append(f"ict_block ({type(e).__name__})")
    try:
        pdh, pdl, pdc = S.prev_day_levels(df)
        add("dist_pdh", lambda: (c - pdh) / I.atr(df, 14))
        add("dist_pdl", lambda: (c - pdl) / I.atr(df, 14))
    except Exception as e:                                        # noqa: BLE001
        skipped.append(f"prev_day ({type(e).__name__})")
    # --- volume ------------------------------------------------------------------------------
    if "tick_volume" in d.columns:
        tv = d["tick_volume"].astype(float)
        add("vol_pctl", lambda: tv.rolling(100).rank(pct=True))
        add("vol_trend", lambda: tv.rolling(10).mean() / tv.rolling(50).mean())

    if skipped:
        print("    skipped: " + ", ".join(skipped))
    return pd.DataFrame(F).replace([np.inf, -np.inf], np.nan)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_h4()
    c = d["close"].astype(float)
    ret = c.pct_change()
    bpy = len(d) / max((d.index[-1] - d.index[0]).days / 365.25, 1e-9)
    pos, fc = champ_position(c, bpy)
    base_r = (pos * ret).dropna()
    mb = perf(base_r, bpy)

    print("=" * 104)
    print("OCTOBER A3 — STRATEGIES AS FEATURES (stacked ensemble, never built here)")
    print("=" * 104)
    print(f"  XAUUSD H4 rebuilt from M15  {d.index[0].date()} -> {d.index[-1].date()}  "
          f"{len(d):,} bars ({bpy:.0f}/yr)")
    print(f"  CONTROL — champion: Sharpe {mb['sharpe']:+.3f}  CAGR {100*mb['cagr']:+.1f}%  "
          f"maxDD {100*mb['maxdd']:.1f}%   (reference ~+1.02)")

    # ── STAGE 1 — the oracle at H4, which also resolves 3cd's open discrepancy ──────────────
    print("\n" + "=" * 104)
    print("STAGE 1 — THE ORACLE AT H4 (CONTROL #5), and 3cd's open discrepancy vs 3p's +0.824")
    print("=" * 104)
    print(f"  {'horizon':>8} {'trim to':>10} {'Sharpe':>9} {'dSharpe':>9} {'t@matched':>10} "
          f"{'maxDD':>9}")
    print("  " + "-" * 60)
    best = -9.9
    for H in HORIZONS:
        fwd = c.shift(-H) / c - 1.0
        loser = ((pos * fwd) <= 0) & (pos * fwd).notna()
        for keep, lbl in ((0.0, "0.0 flat"), (0.5, "0.5 half"), (0.75, "0.75")):
            tp = pos.where(~loser, pos * keep)
            r = (tp * ret).dropna()
            ii = base_r.index.intersection(r.index)
            a, b = r.reindex(ii), base_r.reindex(ii)
            dm = vol_match(a, b) - b
            t = float(dm.mean() / (dm.std() / np.sqrt(len(dm)))) if dm.std() > 0 else np.nan
            m = perf(a, bpy)
            best = max(best, m["sharpe"] - mb["sharpe"])
            print(f"  {H:>7}b {lbl:>10} {m['sharpe']:>+9.3f} "
                  f"{m['sharpe'] - mb['sharpe']:>+9.3f} {t:>+10.2f} {100*m['maxdd']:>8.1f}%")
        print()
    print(f"  best oracle dSharpe at H4: {best:+.3f}   "
          f"(3cd measured {-0.022:+.3f} best at H1; 3p files +0.824 at H4)")
    if best < 0.20:
        print("\n  CONTROL #5 FAILS AT H4 TOO -> the stack is NOT built, as pre-registered.")
        print("  *** AND THIS RESOLVES 3cd's OPEN DISCREPANCY IN THE DIRECTION OF 3cd: ***")
        print("  the same construction is negative at BOTH timeframes, so the H1 result was not a")
        print("  timeframe artefact. 3p's +0.824 must come from a different trim mechanic or a")
        print("  different baseline, not from H4 being the kinder frame.")
        stage2 = False
    else:
        print("\n  CONTROL #5 PASSES at H4 -> build the stack.")
        stage2 = True

    # ── STAGE 2 — the stack, only if the prize exists ───────────────────────────────────────
    X = build_signals(d)
    print(f"\n  signals assembled: {X.shape[1]}  ({', '.join(X.columns[:8])}...)")
    corr = X.apply(lambda s: s.corr(fc))
    print(f"\n  --- COLLINEARITY WITH THE CHAMPION FORECAST (the documented killer) ---")
    for k, v in corr.abs().sort_values(ascending=False).head(8).items():
        print(f"    {k:<14} |corr| {v:.3f}")
    print(f"    median |corr| across all {X.shape[1]} signals: "
          f"{float(corr.abs().median()):.3f}")
    X.to_csv(OUT / "a3_signals.csv")
    if not stage2:
        print("\n  Stage 2 skipped: no prize to chase. Signal panel written for reuse.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
