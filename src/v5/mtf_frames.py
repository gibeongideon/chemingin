"""One provenance-clean set of H4 / H1 / M15 frames for XAUUSD. Pure functions, no I/O of its own.

WHY THIS EXISTS, AND WHY IT IS NOT OPTIONAL FOR A MULTI-TIMEFRAME STUDY.
A stack that reads direction from H4, structure from H1 and entries from M15 is precisely the
shape of study that §3ba and §3bb say cannot be run on the files in `data/` as they stand. The
audit's verdict for XAUUSD:

    consistent core : M15, H1        (agreement 1.00 at every shared instant, every year)
    OUTLIERS        : H4  — 0.04 from 2023 onward
                      M30 — 0.31 in 2026

`XAUUSD_H4_long.csv` carries FTMO's quotes from 2023-06-28 (the reach of
`refresh_xau_h4.py --bars 5000`) and the original feed before that, while the M15 file was never
refreshed. Reading an H4 "bias" from one broker's bars and confirming it on another's M15 is the
same defect that manufactured §3ao's M15 result and §3az's 0.646 operating point — a leak in the
data's PROVENANCE, which no timestamp control in this repo detects.

THE FIX IS CHEAP AND EXACTLY VALIDATED. H4 and H1 are exact aggregations of M15, and the M15 file
is one feed throughout. Rebuilding both from M15 makes all three frames one series BY
CONSTRUCTION, and the rebuild is not an approximation to be taken on trust: it reproduces the
filed H4 to the cent for every year through 2022 (12,790 bars, 100% within $0.10, median
difference exactly 0.0000) and diverges only where FTMO overwrote it. `verify_against_filed`
asserts that on every run, so a future feed change breaks the build instead of the result.

Cost of the substitution, measured in §3ba rather than assumed: scoring the frozen advisor model
on rebuilt-H4 versus the broker's own H4 disagrees on **0.33% of decisions (98.9% recall)** in
the era where the prices are identical, so the aggregation itself is faithful. The remaining
6.20% disagreement in the spliced era is the broker difference, not the method.

PARTIAL BARS ARE FLAGGED, NOT DROPPED. `legs` counts the M15 bars inside each coarse bar (16 for
a full H4, 4 for a full H1). ~20% of H4 groups have fewer than 16, so a model trained on partial
bars and served on complete ones has a train/serve skew. Returning the count lets a study gate on
it and report the delta; silently dropping them would change the sample, and silently keeping
them would hide the skew.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEGS = {"H4": 16, "H1": 4, "M30": 2}
RULE = {"H4": "4h", "H1": "1h", "M30": "30min"}


def aggregate(m15: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Exact OHLC aggregation of M15 up to `tf`. One feed by construction."""
    if tf not in RULE:
        raise ValueError(f"tf must be one of {sorted(RULE)}, got {tf!r}")
    d = m15.copy()
    d.index = pd.DatetimeIndex(d.index)
    g = d.groupby(d.index.floor(RULE[tf]))
    out = pd.DataFrame(dict(open=g["open"].first(), high=g["high"].max(), low=g["low"].min(),
                            close=g["close"].last(), legs=g["close"].count()))
    if "tick_volume" in d.columns:
        out["tick_volume"] = g["tick_volume"].sum()
    if "spread" in d.columns:
        out["spread"] = g["spread"].median()
    out["full_bar"] = out["legs"] == LEGS[tf]
    return out


def frames(m15: pd.DataFrame) -> dict:
    """The clean stack: {'M15','H1','H4'}, all derived from the single M15 series."""
    return {"M15": m15.copy(), "H1": aggregate(m15, "H1"), "H4": aggregate(m15, "H4")}


def verify_against_filed(rebuilt: pd.DataFrame, filed: pd.DataFrame, clean_until: int = 2022,
                         tol: float = 0.10) -> dict:
    """Assert the rebuild reproduces the filed file in the era before the splice.

    Raises rather than warns. A rebuild that no longer matches means either the M15 file has
    itself been refreshed from a different broker or the aggregation is wrong, and both of those
    silently invalidate every downstream number. §3bb's whole lesson is that this check has to
    run, not be assumed.
    """
    ix = rebuilt.index.intersection(filed.index)
    pre = ix[ix.year <= clean_until]
    if len(pre) < 1000:
        raise AssertionError(f"only {len(pre)} pre-{clean_until} bars overlap — cannot verify")
    d = (rebuilt.loc[pre, "close"] - filed.loc[pre, "close"]).abs()
    med, frac = float(d.median()), float((d <= tol).mean())
    if med > 0.01 or frac < 0.99:
        raise AssertionError(
            f"rebuilt frame does not reproduce the filed one before {clean_until}: "
            f"median |d| {med:.4f}, {frac*100:.1f}% within ${tol:.2f}. Either the M15 file has "
            "been refreshed from a different broker or the aggregation changed. See §3ba/§3bb.")
    return dict(n=int(len(pre)), median_abs=med, frac_within_tol=frac)


def align_to(coarse_index, fine_index, coarse_minutes: int) -> np.ndarray:
    """For each fine bar, the position of the newest coarse bar whose CLOSE precedes it.

    THE POINT OF THIS FUNCTION. A coarse bar stamped t does not close until t + its duration, so
    an H4 bar stamped 12:00 is not knowable at 12:15 — its own close is still forming. Reading an
    H4 "bias" onto an M15 bar inside that window is a lookahead, and it is the exact bug that
    produced AUC 0.901 in §3az before `dec_bar_minutes` was added. Returns -1 where no coarse bar
    has closed yet, which callers must treat as "no bias available" rather than as index -1.

    Three separate results in this file's history died to cross-timeframe alignment (§3av's
    forward-stamped FX D1, §3ax's weekly ffill, §3az's stamp-vs-close), so the alignment lives in
    one audited function instead of being re-derived per script.
    """
    ci = pd.DatetimeIndex(coarse_index) + pd.Timedelta(minutes=coarse_minutes)
    fi = pd.DatetimeIndex(fine_index)
    pos = np.searchsorted(ci.values, fi.values, side="right") - 1
    return pos


def broadcast(coarse: pd.Series, fine_index, coarse_minutes: int) -> pd.Series:
    """Carry a coarse-frame series onto a fine index, using only CLOSED coarse bars."""
    pos = align_to(coarse.index, fine_index, coarse_minutes)
    vals = np.full(len(pos), np.nan, dtype=float)
    ok = pos >= 0
    vals[ok] = np.asarray(coarse.values, dtype=float)[pos[ok]]
    return pd.Series(vals, index=pd.DatetimeIndex(fine_index))
