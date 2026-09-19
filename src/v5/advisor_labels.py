"""Labels and baselines for the XAUUSD direction advisor. Pure functions, no I/O, no MT5.

Two label families, both declared in `data/v5_runs/xau_advisor/PREREGISTRATION.md` (6df4b12):

  DIRECTION   sign of the close-to-close move h decision-bars ahead, plus the three baselines
              the advisor is judged against.
  ADVERSE     symmetric +/- k*ATR first-touch, RESOLVED ON A FINER TIMEFRAME.

WHY THE ADVERSE LABEL NEEDS A FINER PATH SERIES. A decision bar cannot say which of its own
barriers was touched first — that is §3ax's lesson, where resolving a first-touch label on daily
bars forced a 46% "ambiguous" rate and the model still lost to pure geometry (AUC 0.8659 for the
distance-implied null alone). Here the decision index is H4 closes and the path is walked on M15
bars strictly AFTER each close, so "which came first" is answered rather than guessed.

WHY THE BARRIER IS ATR-SCALED AND SYMMETRIC.
  * ATR-scaled: ATR14/price on XAUUSD H4 has a median of 0.348% in 2018 and 0.936% in 2026, so
    §3ao's fixed `BARRIER = 0.02` is 5.7 ATRs at the start of the sample and 2.1 ATRs at the end
    — a different label in each era. §3ax's rule ("re-derive in ATR or range units, never
    percent") applies to the label, not only to trade brackets.
  * Symmetric: the distance/geometry null is then exactly 0.5 by construction. And at a 6-hour
    horizon drift cannot inflate the base rate (measured 0.5068), so §3ay's "positive EV with
    negative skill" trap — where a 3-ATR upside barrier was touched first 67% of the time on
    drift alone — is structurally absent.

THE THREE BASELINES, AND WHY DRIFT IS CO-PRIMARY. §3r's headline beat PERSISTENCE by +2.12pp.
Measured against a plain "gold rises" forecast the same model is 1.3-1.6pp WORSE (4h 51.68 vs
51.39, 8h 51.63 vs 52.05, 24h 52.23 vs 53.56). An advisor is read by someone whose default is
drift, not persistence, so `drift_baseline` is returned alongside and §3ay's rule — always print
the base rate beside the hit rate — is enforced by making it impossible to get one without the
other.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- direction
def fwd_dir(close: pd.Series, h: int) -> pd.DataFrame:
    """Direction label at horizon `h` decision-bars, with all three baselines aligned to it.

    Returns columns:
      y            1 if close[t+h] > close[t], 0 if lower, NaN at exact ties and the tail
      persistence  1 if close[t] > close[t-h]  — the §3r baseline, the expression at
                   `v5_xau_intermarket_accuracy.py:670`
      fwd_ret      the raw forward return, for magnitude-conditional reporting

    DRIFT and CLIMATOLOGY are not columns because they are TRAIN-SLICE constants and computing
    them here would peek. Use `drift_baseline` / `climatology` inside the walk-forward.
    """
    c = pd.Series(close).astype(float)
    fwd = c.shift(-h) - c
    y = pd.Series(np.where(fwd > 0, 1.0, np.where(fwd < 0, 0.0, np.nan)), index=c.index)
    y[fwd.isna()] = np.nan                                     # tail
    per = np.sign(c - c.shift(h)).fillna(0.0).clip(lower=0.0)
    return pd.DataFrame(dict(y=y, persistence=per.astype(float),
                             fwd_ret=(c.shift(-h) / c - 1.0)), index=c.index)


def drift_baseline(y_train: np.ndarray) -> float:
    """The constant 0/1 DRIFT forecast: predict 'up' iff the TRAIN up-rate exceeds 0.5.

    Train-only by construction — passing out-of-sample labels here is a peek, and on gold it is
    the peek that matters, because the OOS up-rate is exactly the thing the advisor is trying to
    beat.
    """
    m = np.isfinite(y_train)
    if not m.any():
        return float("nan")
    return 1.0 if float(np.mean(y_train[m])) > 0.5 else 0.0


def climatology(y_train: np.ndarray) -> float:
    """The constant PROBABILITY forecast at the train up-rate — the Brier/log-loss reference."""
    m = np.isfinite(y_train)
    return float(np.mean(y_train[m])) if m.any() else float("nan")


# --------------------------------------------------------------------------- adverse move
def first_touch_atr(dec_index, dec_close, dec_atr, path_high, path_low, path_index,
                    k_atr: float, horizon_bars: int, dec_bar_minutes: int = 0) -> pd.DataFrame:
    """Symmetric +/- k*ATR first-touch, resolved on a finer path series.

    For each decision timestamp t with close C and ATR A:
        up   = C + k_atr * A
        down = C - k_atr * A
    Walk the path bars STRICTLY AFTER t, up to `horizon_bars` of them, and record which level is
    reached first.

      y = 1  the DOWN (adverse) barrier is touched first
      y = 0  the UP barrier is touched first
      y = NaN  neither within the horizon, OR both inside the same path bar (genuinely
               ambiguous — dropped, never guessed, the `first_touch` convention at
               `v5_repr_ceiling.py:75`)

    `resolved` and `ambiguous` are returned as separate columns so the unconditional resolution
    rate can be reported model-free. That matters for the user-facing card: "82% of 6h windows
    resolve one way or the other" is a fact about gold, not about the model, and conflating them
    would invent an unconditional probability out of a conditional one.

    *** `dec_bar_minutes` IS NOT OPTIONAL WHEN THE PATH IS FINER THAN THE DECISION FRAME. ***
    A bar STAMPED at t does not close until t + its duration: an H4 bar stamped 12:00 closes at
    16:00. Its close is therefore not knowable at 12:00, and starting the path at the first M15
    bar after 12:00 walks the very bars that FORMED that close.

    This is not hypothetical — it is the bug this parameter was added to fix. With the path
    started at t, the adverse label scored **AUC 0.901 and 82.8% accuracy** against a 0.514 base
    rate, and the tell was that AUC ROSE as the horizon SHORTENED (9h 0.813 -> 6h 0.844 ->
    4h 0.901), because at a 4-hour horizon the entire 16-bar path sits inside the decision bar.
    Pass `dec_bar_minutes=240` for an H4 decision index so the path starts at the CLOSE.

    Note that the same-timeframe equivalence test against `label_symmetric` cannot catch this:
    when the path and decision frames are identical, starting at the next bar IS correct, so the
    test passed at 1.000000 while the cross-timeframe caller was leaking. `src/v5/xau_m15_exec.py`
    documents the same rule ("H4 keyed off COMPLETION time").

    Strictness: `searchsorted(..., side="right")` then guarantees the first path bar considered
    opens strictly after the decision bar has closed.
    """
    di = pd.DatetimeIndex(dec_index)
    pi = pd.DatetimeIndex(path_index)
    C = np.asarray(dec_close, float)
    A = np.asarray(dec_atr, float)
    PH = np.asarray(path_high, float)
    PL = np.asarray(path_low, float)

    # the decision is made at the bar's CLOSE, not its stamp
    dec_time = di + pd.Timedelta(minutes=int(dec_bar_minutes))
    start = np.searchsorted(pi.values, dec_time.values, side="right")
    n = len(di)
    y = np.full(n, np.nan)
    amb = np.zeros(n, bool)
    for i in range(n):
        a = A[i]
        if not np.isfinite(a) or a <= 0 or not np.isfinite(C[i]):
            continue
        up, dn = C[i] + k_atr * a, C[i] - k_atr * a
        s = start[i]
        e = min(s + horizon_bars, len(pi))
        for j in range(s, e):
            tu, td = PH[j] >= up, PL[j] <= dn
            if tu and td:
                amb[i] = True
                break
            if td:
                y[i] = 1.0
                break
            if tu:
                y[i] = 0.0
                break
    return pd.DataFrame(dict(y=y, ambiguous=amb,
                             resolved=np.isfinite(y),
                             up_level=C + k_atr * A,
                             dn_level=C - k_atr * A), index=di)


def resolution_rate(ft: pd.DataFrame) -> dict:
    """Model-free facts about an adverse label: what fraction resolves, and the base rate."""
    n = len(ft)
    res = int(ft["resolved"].sum())
    amb = int(ft["ambiguous"].sum())
    dec = ft.loc[ft["resolved"], "y"]
    return dict(n=n, resolved=res, resolved_frac=res / n if n else float("nan"),
                ambiguous=amb, neither=n - res - amb,
                base_rate=float(dec.mean()) if len(dec) else float("nan"))


# --------------------------------------------------------------------------- frames
def build_h1_from_m15(m15: pd.DataFrame, require_full: bool = True) -> pd.DataFrame:
    """Aggregate M15 to H1, enforcing complete legs.

    A partial hour is not an H1 bar: its high/low/volume are drawn from fewer observations, so a
    model trained on partial bars and served on complete ones has a train/serve skew. With
    `require_full` only 4-leg hours survive, and `legs` is returned either way so the skew can be
    measured rather than assumed away (~20% of H4 groups in this repo have fewer than 16 M15
    legs, which is exactly this problem one timeframe up).
    """
    d = m15.copy()
    d.index = pd.DatetimeIndex(d.index)
    g = d.groupby(d.index.floor("1h"))
    out = pd.DataFrame(dict(
        open=g["open"].first(), high=g["high"].max(), low=g["low"].min(),
        close=g["close"].last(), legs=g["close"].count(),
    ))
    for c in ("tick_volume", "spread"):
        if c in d.columns:
            out[c] = g[c].sum() if c == "tick_volume" else g[c].median()
    return out[out["legs"] == 4] if require_full else out


def bar_legs(fine: pd.DataFrame, freq: str) -> pd.Series:
    """How many fine bars fall inside each coarse bar — the train/serve skew diagnostic."""
    ix = pd.DatetimeIndex(fine.index)
    return pd.Series(1, index=ix).groupby(ix.floor(freq)).sum()


def minutes_since(coarse_index, at) -> float:
    """Age in minutes of the newest closed coarse bar at wall-clock `at`.

    Feeds the card's honesty stamp: the advisor refreshes every 5 minutes but its features only
    move on a coarse close, so a printed number can legitimately be hours old and must say so.
    """
    ix = pd.DatetimeIndex(coarse_index)
    t = pd.Timestamp(at)
    past = ix[ix <= t]
    return float((t - past[-1]).total_seconds() / 60.0) if len(past) else float("nan")
