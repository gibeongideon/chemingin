"""Lookahead / leakage probe via truncation invariance.

Fills a gap next to `dsr_pbo.py` (is the edge real?) and `bootstrap.py` (how
uncertain is it?): this asks a cheaper, prior question — **is the computation
even causal in the first place?** A genuinely causal feature function's output
at bar t must be identical whether or not bars after t exist in the frame. This
probes that directly: build a deterministic synthetic OHLC panel, compute the
function on the full panel and on several right-truncated copies, and compare
the interior bars both copies share.

Three separate leakage-shaped bugs got caught THIS SESSION only by noticing an
implausible backtest number and tracing it back by hand (a `merge_asof`
calendar-alignment bug that let one instrument's holiday silently corrupt
another's fresh value; an `al[m].fillna(0.0)` zero-fill that quietly
under-weighted a portfolio for years before a newer asset's data began; an HMM
decode that would have used forward-backward smoothing — future bars informing
an earlier bar's state — had it not been hand-written to filter instead). None
of those needed a suspicious number to be catchable in principle; each is
exactly a truncation-invariance violation. Run new causal feature/signal
functions through `probe_lookahead` BEFORE trusting a backtest built on them,
not after.

Deliberately narrow: only right-truncation invariance, the property that
actually matters for a bar-by-bar backtest/live engine. It says nothing about
whether an edge is real (`dsr_pbo.py`) or how uncertain a Sharpe is
(`bootstrap.py`) — a probe PASS is necessary, not sufficient.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

_PROBE_OFFSETS: tuple[int, ...] = (60, 40, 20, 10)
_WARMUP_BUFFER = 400          # clears every rolling window used in this repo (max ~300)
_SYNTHETIC_ROWS = 900


def synthetic_ohlc(n: int = _SYNTHETIC_ROWS, seed: int = 7, freq: str = "4h") -> pd.DataFrame:
    """Deterministic, regime-segmented random-walk OHLC panel.

    Cycles trend-up / trend-down / high-vol-chop / quiet-range segments so a
    trend-follower, a breakout, and a mean-reverter all find something to fire
    on — a probe window where nothing ever fires would pass vacuously (it
    compared nothing, not "compared and matched")."""
    rng = np.random.default_rng(seed)
    segments = ((0.0006, 0.0040), (-0.0006, 0.0050), (0.0000, 0.0120), (0.0000, 0.0015))
    seg_len = max(n // (len(segments) * 2), 1)
    drift = np.empty(n)
    scale = np.empty(n)
    for i in range(n):
        mu, sd = segments[(i // seg_len) % len(segments)]
        drift[i] = mu
        scale[i] = sd
    log_ret = rng.normal(0.0, 1.0, n) * scale + drift
    close = 2000.0 * np.exp(np.cumsum(log_ret))
    rng2 = np.random.default_rng(seed + 1)
    wick = np.abs(rng2.normal(0.0, 1.0, n)) * scale * close
    high = close + wick * rng2.uniform(0.3, 1.0, n)
    low = close - wick * rng2.uniform(0.3, 1.0, n)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    idx = pd.date_range("2020-01-01", periods=n, freq=freq)
    idx.name = "time"   # matches this repo's `set_index("time")` convention — several
                        # feature functions merge_asof on a column named "time"
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "tick_volume": rng2.integers(100, 5000, n).astype(float)},
        index=idx,
    )


@dataclass(frozen=True)
class LookaheadVerdict:
    ok: bool
    reason: str | None
    n_compared: int
    max_abs_diff: float
    offending_columns: tuple[str, ...] = field(default_factory=tuple)


def probe_lookahead(
    feature_fn: Callable[[pd.DataFrame], "pd.Series | pd.DataFrame"],
    *,
    n: int = _SYNTHETIC_ROWS,
    seed: int = 7,
    offsets: tuple[int, ...] = _PROBE_OFFSETS,
    warmup_buffer: int = _WARMUP_BUFFER,
    atol: float = 1e-9,
    frame_builder: Callable[..., pd.DataFrame] = synthetic_ohlc,
) -> LookaheadVerdict:
    """Right-truncation-invariance probe for one causal feature/signal function.

    `feature_fn` must take a single frame and return a Series or DataFrame of
    the same length. A hard error inside `feature_fn` — on the full frame OR
    any truncated copy — is reported as a probe FAILURE rather than
    propagated: a function that only raises under truncation would otherwise
    evade the very check this exists to run.

    `frame_builder(n=.., seed=..) -> DataFrame` defaults to `synthetic_ohlc`
    (a single-asset OHLC panel). Pass a custom builder for a signal that needs
    a different shape — e.g. a cross-asset function reading `close_a`/
    `close_b` columns from two independent synthetic series — so the same
    truncation-invariance check covers it without duplicating this logic.
    """
    df_full = frame_builder(n=n, seed=seed)
    try:
        out_full = feature_fn(df_full)
    except Exception as exc:  # noqa: BLE001 — deliberately broad, see docstring
        return LookaheadVerdict(False, f"feature_fn raised on the full frame: {exc}", 0, float("nan"))
    if isinstance(out_full, pd.Series):
        out_full = out_full.to_frame(name=out_full.name or "value")

    worst = 0.0
    offenders: set[str] = set()
    n_compared = 0
    ran_any = False
    for k in offsets:
        cutoff = n - k
        if cutoff - warmup_buffer < 50:
            continue
        df_trunc = df_full.iloc[:cutoff]
        try:
            out_trunc = feature_fn(df_trunc)
        except Exception as exc:  # noqa: BLE001
            return LookaheadVerdict(
                False, f"feature_fn raised on a right-truncated frame (k={k}): {exc}", n_compared, worst
            )
        if isinstance(out_trunc, pd.Series):
            out_trunc = out_trunc.to_frame(name=out_trunc.name or "value")
        ran_any = True

        compare_idx = df_trunc.index[warmup_buffer:]
        common_cols = [c for c in out_full.columns if c in out_trunc.columns]
        if not common_cols:
            continue
        a = out_full.loc[compare_idx, common_cols].apply(pd.to_numeric, errors="coerce")
        b = out_trunc.loc[compare_idx, common_cols].apply(pd.to_numeric, errors="coerce")
        diff = (a - b).abs()
        col_max = diff.max(skipna=True)
        for col, v in col_max.items():
            if pd.notna(v) and float(v) > atol:
                offenders.add(str(col))
                worst = max(worst, float(v))
        n_compared += int(diff.notna().sum().sum())

    if not ran_any:
        return LookaheadVerdict(False, "no offset cleared the warmup buffer — probe misconfigured", 0, float("nan"))
    if offenders:
        return LookaheadVerdict(
            False, "signal changed when future bars were withheld", n_compared, worst, tuple(sorted(offenders))
        )
    if n_compared == 0:
        return LookaheadVerdict(False, "no comparable columns/bars found (probe compared nothing)", 0, float("nan"))
    return LookaheadVerdict(True, None, n_compared, worst)


def probe_many(named_fns: dict[str, Callable], **kwargs) -> dict[str, LookaheadVerdict]:
    """Convenience batch runner. Returns {name: LookaheadVerdict}."""
    return {name: probe_lookahead(fn, **kwargs) for name, fn in named_fns.items()}


def print_report(results: dict[str, LookaheadVerdict]) -> None:
    width = max((len(k) for k in results), default=10)
    print(f"{'function':{width}}  {'verdict':7}  {'n_cmp':>7}  {'max|diff|':>10}  reason")
    for name, v in results.items():
        verdict = "PASS" if v.ok else "FAIL"
        cols = f" [{', '.join(v.offending_columns)}]" if v.offending_columns else ""
        print(f"{name:{width}}  {verdict:7}  {v.n_compared:7d}  {v.max_abs_diff:10.3g}  {v.reason or ''}{cols}")
