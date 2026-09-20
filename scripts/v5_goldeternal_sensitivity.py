"""How much hidden basis drift would erase the swap-free gold advantage? And what does hand-
executing it cost?

WHY A SENSITIVITY AND NOT A POINT ESTIMATE. §3ag priced GoldEternal at EXACTLY zero carry, on the
strength of a metadata field (`swap_mode = DISABLED`) plus a basis regression over 167 days. Both
are weaker than the conclusion they carry:

  * the field is an inference. Re-read 2026-09-20, `swap_mode` is still 0 but `swap_long` and
    `swap_short` now show **-30.0**, which DISABLED mode should leave dormant — should.
  * the basis slope has MOVED with more data: +0.17%/yr -> **+0.234%/yr** (H1, 175 days) and
    +0.21 -> **+0.347%/yr** (D1, 182 days). Still the wrong SIGN to hide a -3.78%/yr funding
    charge, and daily-return correlation is 0.990/0.998, but 182 days cannot exclude a small
    drift and the estimate is visibly unsettled.

Maven also does not permit Expert Advisors, so this cannot be captured by the bot there at all —
only by hand, through the alert path that already exists (`deploy/xau-manual-alert.*`, read-only,
no order code). That makes TURNOVER a first-class number rather than an afterthought: an edge
that needs a hand-trade every few hours is not collectable by a person.

So instead of asking "what is it worth if the carry is zero", this asks the two questions that
actually decide it:

    1. what basis drift would it take to wipe out the advantage?   (robustness)
    2. how many hand-trades a month does it demand?                (feasibility)

A conclusion that survives any plausible basis is worth more than a point estimate that needs
one number to be exactly right.

THE PRICE SERIES IS THE CLEAN ONE. `v5_maven_carry_book.py` proxies both instruments with
`data/XAUUSD_H4_long.csv`, which §3bb found is two brokers spliced at 2023-06-28. For a
single-timeframe backtest that is one spurious return at the boundary rather than a systematic
leak, but it is free to remove: `src/v5/mtf_frames.py` rebuilds H4 from M15, verified to the cent
pre-2023. Both are run so the difference is visible instead of assumed negligible.

    python scripts/v5_goldeternal_sensitivity.py
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

from scripts.v5_financing_aware_book import engine_fin  # noqa: E402
from scripts.v5_maven_carry_book import CANDS, SPREAD_MULT, book, halves  # noqa: E402
from scripts.v5_volregime_taper_crossasset import load_sleeve  # noqa: E402
from scripts.v5_xau_champion_lifts import champion_recipe, dd_of, sharpe  # noqa: E402
from src.v5.mtf_frames import frames, verify_against_filed  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402

# carry scenarios for the GOLD sleeve, %/yr on a long
SCENARIOS = [
    ("measured basis +0.234%/yr", +0.00234),
    ("§3ag assumption: exactly 0", 0.0),
    ("pessimistic -0.5%/yr", -0.005),
    ("pessimistic -1.0%/yr", -0.010),
    ("pessimistic -2.0%/yr", -0.020),
    ("swap quietly switched on (-3.78%)", -0.0378),
    ("the dormant field is real (-30/lot = -2.51%)", -0.0251),
]
XAUUSD_RATE = CANDS["XAUUSD"][5]          # -0.0370 as recorded
CONFIGURED = ["XAUUSD", "BTCUSD", "US100", "BRENT"]


def _cagr(r: pd.Series) -> float:
    """engine_fin returns DAILY returns, so 252 — not the H4 bar count."""
    r = r.dropna()
    if not len(r):
        return float("nan")
    return float((1.0 + r).prod() ** (252.0 / len(r)) - 1.0)


def rebalance_at(close: pd.DataFrame, rate: float, spr_bp: float, freq: str | None) -> pd.Series:
    """The sleeve when the position is only allowed to CHANGE every `freq`.

    This is the hand-execution question. The engine re-sizes 47 times a month; a person will
    not. `freq=None` is the engine as-is; "W" means the held position is carried unchanged for a
    week at a time. Costs and financing still accrue on whatever is actually held, so a lazier
    schedule is not silently rewarded with lower turnover cost it did not earn.
    """
    from scripts.v5_financing_aware_book import BUFFER, MAX_LEV, SHORT_HAIRCUT, TARGET_VOL
    ann = 252 * 6
    ret = close["close"].pct_change()
    vol = ret.ewm(halflife=42, min_periods=20).std() * np.sqrt(ann)
    fc = champion_signal(close["close"])
    pos = (fc.clip(-2.0, 2.0) * (TARGET_VOL / vol)).clip(-MAX_LEV, MAX_LEV)
    band = (BUFFER * (TARGET_VOL / vol).clip(0, MAX_LEV)).values
    p, out, held = pos.values.copy(), np.zeros(len(pos)), 0.0
    for i in range(len(p)):
        if np.isfinite(p[i]):
            b = band[i] if np.isfinite(band[i]) else 0.0
            if abs(p[i] - held) > b:
                held = p[i] - np.sign(p[i] - held) * b
        out[i] = held
    h = pd.Series(out, index=pos.index)
    if freq is not None:
        # *** resample().last() IS A LOOKAHEAD HERE. *** It indexes each period's CLOSING value
        # at the period's START, so a forward-fill propagates it BACKWARDS in time — up to a
        # full month on a monthly schedule. Run that way this printed monthly SR 2.215 against
        # the engine's 1.099, i.e. it "discovered" that checking less often doubles the Sharpe.
        # Same bug as §3ax's weekly ffill, which faked +40%/yr, and it is in my own notes.
        #
        # What a person actually does: look at the engine at check time T, set the position to
        # whatever it says AT T, leave it until the next check. `.first()` is the value at each
        # period's FIRST bar, which is knowable then, and ffill carries it forward only.
        h = h.resample(freq).first().reindex(h.index, method="ffill")
    h = h.shift(1).fillna(0.0)
    cost_bp = 0.75 * spr_bp * SPREAD_MULT
    net = h * ret - h.diff().abs().fillna(0.0) * (cost_bp * 1e-4)
    rate_short = -rate * SHORT_HAIRCUT
    net = net + pd.Series(np.where(h > 0, h * rate, -h * rate_short) / ann, index=h.index)
    eq = (1.0 + net.fillna(0.0)).cumprod().loc["2018-01-01":]
    if eq.empty:
        return pd.Series(dtype=float)
    eq = eq / eq.iloc[0]
    n_moves = int((h.loc["2018-01-01":].diff().abs() > 1e-9).sum())
    out_s = eq.resample("D").last().pct_change(fill_method=None).dropna()
    out_s.attrs["moves"] = n_moves
    return out_s


def gold_sleeve(close: pd.DataFrame, rate: float, spr_bp: float) -> pd.Series:
    fc = champion_signal(close["close"])
    return engine_fin(close, fc, 0.75 * spr_bp * SPREAD_MULT, 252 * 6, 42, rate, True)


def other(name: str) -> pd.Series:
    path, ann, spr_bp, scale, hl, rate = CANDS[name]
    df = load_sleeve(dict(path=path))
    fc = champion_signal(df["close"]) if scale == 1.0 else champion_recipe(df["close"], scale, 1.5)
    return engine_fin(df, fc, 0.75 * spr_bp * SPREAD_MULT, ann, hl, rate, True)


def turnover(close: pd.DataFrame) -> dict:
    """How much hand-work this sleeve demands. The buffer in `engine_fin` already suppresses
    small adjustments; what is counted here is the number of bars on which the held position
    actually MOVES, which is the number of times a person would have to act."""
    from scripts.v5_financing_aware_book import BUFFER, MAX_LEV, TARGET_VOL
    ret = close["close"].pct_change()
    vol = ret.ewm(halflife=42, min_periods=20).std() * np.sqrt(252 * 6)
    fc = champion_signal(close["close"])
    pos = (fc.clip(-2.0, 2.0) * (TARGET_VOL / vol)).clip(-MAX_LEV, MAX_LEV)
    band = (BUFFER * (TARGET_VOL / vol).clip(0, MAX_LEV)).values
    p, out, held = pos.values.copy(), np.zeros(len(pos)), 0.0
    for i in range(len(p)):
        if np.isfinite(p[i]):
            b = band[i] if np.isfinite(band[i]) else 0.0
            if abs(p[i] - held) > b:
                held = p[i] - np.sign(p[i] - held) * b
        out[i] = held
    held_s = pd.Series(out, index=pos.index).loc["2018-01-01":]
    moves = held_s.diff().abs() > 1e-9
    months = (held_s.index[-1] - held_s.index[0]).days / 30.44
    flips = ((np.sign(held_s) != np.sign(held_s.shift(1))) & (np.sign(held_s) != 0)).sum()
    return dict(moves=int(moves.sum()), per_month=moves.sum() / months,
                flips=int(flips), flips_per_month=flips / months,
                mean_abs_pos=float(held_s.abs().mean()))


def main() -> None:
    filed = load_sleeve(dict(path="data/XAUUSD_H4_long.csv"))
    _, m15 = __import__("scripts.v5_advisor_measure", fromlist=["load_frames"]).load_frames()
    F = frames(m15)
    verify_against_filed(F["H4"], filed)
    clean = F["H4"][["open", "high", "low", "close"]].copy()

    print("=" * 96)
    print("GOLDETERNAL SENSITIVITY — what would it take to erase the advantage?")
    print("=" * 96)
    spr_g, spr_x = CANDS["GoldEternal"][2], CANDS["XAUUSD"][2]
    for label, px in (("FILED H4 (two brokers spliced at 2023-06-28)", filed),
                      ("CLEAN H4 (rebuilt from M15, one feed)", clean)):
        base = gold_sleeve(px, XAUUSD_RATE, spr_x)
        print(f"\n--- {label} ---")
        # dd_of already returns a PERCENTAGE and engine_fin returns DAILY returns, so the
        # drawdown is not scaled again and the CAGR annualises at 252, not 252*6. Getting
        # either wrong prints a -2277% drawdown and a +134% CAGR, which is how I first ran it.
        print(f"  XAUUSD  (carry {XAUUSD_RATE*100:+.2f}%/yr, spread {spr_x}bp): "
              f"net SR {sharpe(base):+.3f}  maxDD {dd_of(base):.1f}%  CAGR {_cagr(base)*100:+.2f}%")
        print(f"  {'GoldEternal carry scenario':40s} {'net SR':>8s} {'vs XAUUSD':>10s} "
              f"{'maxDD':>8s} {'CAGR':>8s} {'d CAGR':>8s}")
        for nm, rate in SCENARIOS:
            r = gold_sleeve(px, rate, spr_g)
            print(f"  {nm:40s} {sharpe(r):+8.3f} {sharpe(r)-sharpe(base):+10.3f} "
                  f"{dd_of(r):7.1f}% {_cagr(r)*100:+7.2f}% "
                  f"{(_cagr(r)-_cagr(base))*100:+7.2f}%")

    # breakeven: the carry at which GoldEternal stops beating XAUUSD
    base = gold_sleeve(clean, XAUUSD_RATE, spr_x)
    lo, hi = -0.06, 0.01
    for _ in range(40):
        mid = (lo + hi) / 2
        if sharpe(gold_sleeve(clean, mid, spr_g)) > sharpe(base):
            hi = mid
        else:
            lo = mid
    print("\n" + "=" * 96)
    print(f"BREAKEVEN: GoldEternal stops beating XAUUSD once its all-in carry reaches "
          f"{(lo+hi)/2*100:+.2f}%/yr")
    print(f"  measured basis drift is {+0.234:+.3f}%/yr, the WRONG SIGN, and "
          f"{abs((lo+hi)/2*100 - 0.234):.2f}pp away from that line.")
    print("  So the conclusion does not depend on the basis estimate being exactly right —")
    print("  it depends only on Maven not switching the swap on, which is checkable monthly.")

    print("\n" + "=" * 96)
    print("FEASIBILITY: Maven forbids EAs, so this must be hand-executed. How much work?")
    print("=" * 96)
    t = turnover(clean)
    print(f"  position CHANGES (net of the engine's own buffer): {t['moves']:,} over the eval, "
          f"= {t['per_month']:.1f} per month")
    print(f"  direction FLIPS (long<->flat<->short): {t['flips']:,} = "
          f"{t['flips_per_month']:.2f} per month")
    print(f"  mean absolute position: {t['mean_abs_pos']:.3f} lots per $100k of notional target")
    print(f"\n  {t['per_month']:.0f} adjustments a month is "
          f"{'PRACTICAL by hand' if t['per_month'] <= 30 else 'HEAVY for hand execution'}; "
          f"the existing alert path (deploy/xau-manual-alert.*) already emails these.")
    print("  NOTE the buffer is what makes this liveable: without it the position would be")
    print("  re-set on nearly every H4 bar (6/day).")
    print("  And note the DIRECTION essentially never changes (1 flip in the whole eval) --")
    print("  the work is continuous SIZE adjustment, not deciding which way to be.")

    print("\n" + "=" * 96)
    print("SO: how much does a LAZIER, hand-sized schedule actually cost?")
    print("=" * 96)
    full = rebalance_at(clean, +0.00234, spr_g, None)
    print(f"  {'schedule':22s} {'net SR':>8s} {'vs engine':>10s} {'CAGR':>8s} "
          f"{'maxDD':>8s} {'changes/mo':>11s}")
    months = 8.5 * 12
    for nm, fq in (("engine (every H4 bar)", None), ("daily", "1D"), ("every 2 days", "2D"),
                   ("weekly", "1W"), ("fortnightly", "2W"), ("monthly", "1MS")):
        r = rebalance_at(clean, +0.00234, spr_g, fq)
        if not len(r):
            continue
        print(f"  {nm:22s} {sharpe(r):+8.3f} {sharpe(r)-sharpe(full):+10.3f} "
              f"{_cagr(r)*100:+7.2f}% {dd_of(r):7.1f}% "
              f"{r.attrs.get('moves', 0)/months:11.1f}")
    print("\n  The verdict on feasibility is whichever row a person will actually DO every time,")
    print("  not the best row: a schedule that is skipped is worse than a lazier one that is not.")


if __name__ == "__main__":
    main()
