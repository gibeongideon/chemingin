"""Risk variants of the H4 long-only XAU champion, plus weekend-flat scheduling.

ONE strategy, three risk settings. `champ-meta` is the champion multiplied by
(1 - beta * p_bad), where p_bad is a calibrated probability that the champion's own
forward 24-bar P&L is negative. It is NOT an independent strategy: its position is
<= the champion's on 100% of bars, daily-return correlation 0.96. Choose a variant,
do not stack them on one account.

Backtested 2018+ at live FTMO cost ($0.448/oz), continuous engine:

    variant      beta   CAGR    maxDD    Sharpe   avg exposure
    full         0.0   +18.6%   -17.0%   1.084    0.78x
    trim50       0.5   +15.5%   -13.2%   1.178    0.61x
    trim100      1.0   +12.4%    -9.4%   1.329    0.44x

HONESTY: the trim FAILED its pre-registered research gates (paired t -2.11 vs the
champion, positive in only 3 of 9 years) — see V5_FINDINGS §3p. It buys drawdown
with return; it does not add alpha. Deploy it only where a hard DD limit binds.

WEEKEND-FLAT costs real money and makes drawdown WORSE (forced re-entry across the
Monday gap), measured on the same window:

    variant   hold        Fri-16:00 flat     delta
    full      +18.6%      +14.2% / DD -19.9%  -4.4pp CAGR, DD 2.9pp WORSE
    trim100   +12.4%      + 9.9% / DD -10.9%  -2.5pp CAGR, DD 1.5pp worse
    turnover  ~24/yr      ~100/yr             4x the spread bill

Friday 16:00 UTC is the best cutoff tested (Thursday 20:00 is markedly worse).
Enable it only when the prop firm forbids weekend exposure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.v5.xau_dual_signals import champion_signal

__all__ = ["variant_signal", "load_pbad", "predict_pbad", "weekend_flat_mask",
           "in_flat_window", "VARIANTS"]

# beta = how hard the p_bad probability trims the champion's exposure
VARIANTS = {"full": 0.0, "trim50": 0.5, "trim100": 1.0}

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "data" / "models" / "xau_pbad_h4"


# ------------------------------------------------------------------ features
def pbad_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal feature matrix for the p_bad model.

    Must stay byte-identical to what `scripts/v5_train_pbad.py` trained on — the
    saved artifact records the column order and loading asserts it matches.
    """
    from scripts.v5_xau_turn_prob import build_features
    return build_features(df)


# --------------------------------------------------------------- model I/O
def load_pbad(path: str | Path = DEFAULT_MODEL):
    """Load the trained p_bad artifact, or return None if absent.

    Returning None is deliberate: a missing model must degrade to the plain
    champion (a known-good strategy), never to a silently different one.
    """
    import joblib
    p = Path(path)
    mf, jf = p.with_suffix(".joblib"), p.with_suffix(".json")
    if not mf.exists() or not jf.exists():
        return None
    bundle = joblib.load(mf)
    bundle["meta"] = json.loads(jf.read_text())
    return bundle


def predict_pbad(bundle, df: pd.DataFrame) -> pd.Series:
    """Calibrated P(champion's forward P&L < 0) per bar. NaN where features are
    incomplete — callers must treat NaN as 0.0 (no trim), never as 1.0."""
    F = pbad_features(df)
    cols = bundle["meta"]["features"]
    missing = [c for c in cols if c not in F.columns]
    if missing:
        raise ValueError(f"p_bad feature mismatch, missing: {missing}")
    X = F[cols].values
    ok = np.isfinite(X).all(axis=1)
    out = pd.Series(np.nan, index=df.index)
    if ok.sum():
        p = bundle["clf"].predict_proba(X[ok])[:, 1]
        if bundle.get("iso") is not None:
            p = bundle["iso"].predict(p)
        out.iloc[np.where(ok)[0]] = p
    return out


# ------------------------------------------------------------------ signal
def variant_signal(df: pd.DataFrame, variant: str = "full",
                   model_path: str | Path = DEFAULT_MODEL,
                   verbose: bool = True) -> pd.Series:
    """Champion exposure for the named risk variant, in [0, 2]."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected {list(VARIANTS)}")
    close = df["close"]
    champ = champion_signal(close)
    beta = VARIANTS[variant]
    if beta == 0.0:
        return champ
    bundle = load_pbad(model_path)
    if bundle is None:
        if verbose:
            print(f"  !! p_bad model not found at {model_path} — "
                  f"FALLING BACK to the full champion (variant {variant} inactive). "
                  f"Train it: python scripts/v5_train_pbad.py")
        return champ
    p = predict_pbad(bundle, df).fillna(0.0)     # NaN -> no trim (fail-safe)
    if verbose:
        print(f"  variant {variant}: beta {beta}  p_bad last {float(p.iloc[-1]):.3f}  "
              f"trim {beta * float(p.iloc[-1]) * 100:.0f}%  "
              f"(model trained {bundle['meta'].get('trained_on', '?')})")
    return (champ * (1.0 - beta * p)).clip(0.0, 2.0)


# ------------------------------------------------------------ weekend flat
def in_flat_window(ts: pd.Timestamp, fri_hour: int = 16,
                   sun_open_hour: int = 22) -> bool:
    """True when a weekend-flat bot must hold NO position.

    Gold trades Sun ~22:00 to Fri ~21:00 UTC. Flat from Friday `fri_hour` through
    the Sunday re-open. `ts` must be broker/UTC time, consistent with the H4 bars.
    """
    wd = ts.dayofweek                      # Mon=0 .. Sun=6
    if wd == 4 and ts.hour >= fri_hour:
        return True
    if wd == 5:
        return True
    if wd == 6 and ts.hour < sun_open_hour:
        return True
    return False


def weekend_flat_mask(index: pd.DatetimeIndex, fri_hour: int = 16,
                      sun_open_hour: int = 22) -> pd.Series:
    """Vectorised `in_flat_window` for backtests."""
    wd = index.dayofweek
    return pd.Series(((wd == 4) & (index.hour >= fri_hour)) | (wd == 5)
                     | ((wd == 6) & (index.hour < sun_open_hour)), index=index)
