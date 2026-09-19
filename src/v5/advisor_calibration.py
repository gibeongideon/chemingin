"""Calibration measurement — does a stated probability mean what it says?

WHY THIS FILE EXISTS. `grep -rni "brier"` over this repo returns nothing. Every prior study
scored AUC, Spearman IC, log-loss in bits, or precision/recall — all of which measure whether a
score RANKS well. None of them measures whether a number printed as "0.58" is right 58% of the
time. For a research script that gap is tolerable; for an advisor whose entire product is a
probability shown to a human, it is the product.

THE MEASUREMENT THAT MADE THIS NECESSARY (recorded in
`data/v5_runs/xau_advisor/PREREGISTRATION.md`, committed 6df4b12): on the fwd2 direction label,
no calibrator reaches positive Brier skill — raw HistGB **-1.37%**, isotonic **-0.26%**, Platt
**-0.15%**, 5-bin OOS **-0.34%** — while the same model's *accuracy* beats a persistence
baseline by +2.79pp with 9/9 years positive. Ranking informative, number not. Without a Brier
skill score that contradiction is invisible, and the advisor would have printed 0.58 while the
truth was 0.54.

WHY BRIER SKILL CAN BE NEGATIVE ON A REAL EDGE — read before treating a negative as a bug.
Brier decomposes (Murphy 1973) as `reliability - resolution + uncertainty`. `resolution` is
bounded by the variance of the TRUE conditional probability: if truth only moves 0.52 -> 0.54,
`resolution <= ~2.5e-4` against `uncertainty ~0.25`, i.e. a ceiling near **0.04% skill**. Any
estimation noise in `reliability` then dominates and the total goes negative. So a small negative
skill on a genuinely-informative-but-weak signal is the EXPECTED result, not evidence of a bug —
which is exactly why `brier_decomposition` is here and not just the scalar: it separates
"the probabilities lie" (reliability) from "the probabilities carry little" (resolution).

EVERYTHING IS PURE. No I/O, no MT5, no file paths, no sklearn. That keeps it unit-testable
without a bridge and importable by both the measurement harness and the live service, which is
the same separation `src/v5/xau_trend.py` keeps from its executors.

OVERLAPPING LABELS. Every CI here is a MOVING-BLOCK bootstrap, never an analytic binomial
interval. A 6-hour label sampled at H4 closes overlaps its neighbours, so `sqrt(p(1-p)/n)`
understates the error — the same reason §3ak's per-event t-stats were demoted to diagnostics.
`src/cta/bootstrap.py:9` has a Sharpe-specific block bootstrap; this is the missing general
sibling and deliberately does not touch it (18 call sites).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Declared in PREREGISTRATION.md §8 so the table cannot be re-cut after the counts are seen.
_EDGES = {
    "direction": (0.00, 0.45, 0.50, 0.525, 0.55, 0.60, 0.65, 1.00),
    "adverse": (0.00, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 1.00),
    "decile": tuple(round(x, 2) for x in np.linspace(0.0, 1.0, 11)),
}
MIN_BUCKET_N = 200          # PREREGISTRATION.md §6/§8


def prob_edges(kind: str = "direction") -> tuple:
    """The pre-registered bucket edges. One source, so no run can re-cut them."""
    if kind not in _EDGES:
        raise KeyError(f"unknown edge set {kind!r}; declared: {sorted(_EDGES)}")
    return _EDGES[kind]


# --------------------------------------------------------------------------- scalars
def brier(y, p) -> float:
    """Mean squared error of a probabilistic forecast. Lower is better."""
    y, p = _clean(y, p)
    return float(np.mean((p - y) ** 2))


def brier_skill(y, p, p_ref) -> float:
    """1 - BS/BS_ref. Positive means better than the reference forecast.

    `p_ref` MUST be the train-prior climatology (a constant), not 0.5 and not the out-of-sample
    base rate — scoring against the OOS base rate is a peek, and scoring against 0.5 flatters
    any forecast on a drifting asset, which is precisely the mistake §3ay's "print the base rate
    beside the hit rate" rule exists to prevent.
    """
    y, p = _clean(y, p)
    ref = np.full_like(p, float(p_ref)) if np.isscalar(p_ref) else np.asarray(p_ref, float)
    bs, bs_ref = np.mean((p - y) ** 2), np.mean((ref - y) ** 2)
    return float(1.0 - bs / bs_ref) if bs_ref > 0 else float("nan")


def brier_decomposition(y, p, edges=None) -> dict:
    """Murphy's three terms: brier = reliability - resolution + uncertainty.

    reliability  mean squared gap between the forecast and the observed frequency in its
                 bucket. **This is the "are the numbers honest" term** — 0 is perfect.
    resolution   variance of bucket frequencies about the base rate. **This is the "are the
                 numbers informative" term** — bigger is better, and it is what a weak signal
                 cannot supply.
    uncertainty  p_bar * (1 - p_bar). Irreducible; depends only on the label.

    Reporting all three is the point: a near-zero Brier skill with reliability ~0 and
    resolution ~0 means "honest but uninformative", which is a shippable advisor. The same skill
    with a large reliability term means "dishonest", which is not.

    THE IDENTITY IS EXACT FOR THE *BINNED* FORECAST, NOT THE CONTINUOUS ONE. Murphy's terms
    replace each forecast by its bucket mean, so `reliability - resolution + uncertainty`
    reconstructs `brier_binned`, and `brier` (on the raw continuous `p`) differs by the
    within-bucket variance of `p`. Both are returned plus `binning_gap`, because a large gap
    means the buckets are too wide to describe the forecast being shown to the user.
    """
    y, p = _clean(y, p)
    e = np.asarray(edges if edges is not None else prob_edges("decile"), float)
    idx = np.clip(np.digitize(p, e[1:-1]), 0, len(e) - 2)
    n = len(y)
    obar = float(np.mean(y))
    rel = res = 0.0
    binned = np.empty(n)
    for b in np.unique(idx):
        m = idx == b
        nk = int(m.sum())
        fk, ok = float(np.mean(p[m])), float(np.mean(y[m]))
        binned[m] = fk
        rel += nk / n * (fk - ok) ** 2
        res += nk / n * (ok - obar) ** 2
    unc = obar * (1.0 - obar)
    bs_binned = rel - res + unc
    bs_cont = float(np.mean((p - y) ** 2))
    return dict(reliability=rel, resolution=res, uncertainty=unc,
                brier_binned=bs_binned, brier=bs_cont,
                binning_gap=bs_cont - bs_binned, base_rate=obar)


def ece(y, p, edges=None) -> float:
    """Expected calibration error: n-weighted mean |forecast - observed| across buckets."""
    return _cal_err(y, p, edges, reduce="mean")


def mce(y, p, edges=None) -> float:
    """Maximum calibration error: the worst bucket. Catches a single lying bucket that ECE
    averages away — which matters here because the advisor's alerts fire from the tails."""
    return _cal_err(y, p, edges, reduce="max")


# --------------------------------------------------------------------------- bootstrap
def block_bootstrap_ci(x, block: int = 60, n_boot: int = 2000, alpha: float = 0.10,
                       seed: int = 7, stat=np.mean) -> tuple:
    """Moving-block bootstrap CI for any per-event statistic. Returns (lo, hi, se).

    Blocks of `block` consecutive observations are resampled WITH REPLACEMENT, preserving
    short-range autocorrelation while breaking long-range structure. `block=60` is the repo's
    convention (`block_shuffle` at `v5_xau_turn_prob.py:446`) and must exceed the label horizon
    so an overlapping label cannot straddle two independent blocks.

    Pass the per-event DIFFERENCE series (model minus baseline) to get a paired CI, which is what
    every gate in PREREGISTRATION.md §7 is stated in terms of.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < block * 2:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(len(x) / block))
    starts_hi = len(x) - block
    out = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, starts_hi, nb)
        out[i] = stat(np.concatenate([x[j:j + block] for j in s])[:len(x)])
    return (float(np.quantile(out, alpha / 2)),
            float(np.quantile(out, 1 - alpha / 2)),
            float(np.std(out)))


# --------------------------------------------------------------------------- the table
def merge_thin_buckets(counts, edges, base_rate: float, min_n: int = MIN_BUCKET_N) -> list:
    """Collapse buckets below `min_n` toward the base rate, iteratively.

    Declared in advance (PREREGISTRATION.md §8) so the table cannot be re-cut once the counts
    are visible. Returns a list of (lo, hi) pairs. Merging TOWARD the base rate keeps the tails
    — which is where an advisor's alerts come from — from being absorbed into the middle.
    """
    buckets = [[float(edges[i]), float(edges[i + 1]), int(counts[i])]
               for i in range(len(edges) - 1)]
    buckets = [b for b in buckets if b[2] > 0]
    if not buckets:
        return []
    guard = 0
    while len(buckets) > 1 and guard < 100:
        guard += 1
        thin = [i for i, b in enumerate(buckets) if b[2] < min_n]
        if not thin:
            break
        # merge the thinnest bucket into the neighbour lying toward the base rate
        i = min(thin, key=lambda k: buckets[k][2])
        mid = 0.5 * (buckets[i][0] + buckets[i][1])
        j = i + 1 if (mid < base_rate and i + 1 < len(buckets)) else i - 1
        if j < 0:
            j = 1
        if j >= len(buckets):
            j = len(buckets) - 2
        lo = min(buckets[i][0], buckets[j][0])
        hi = max(buckets[i][1], buckets[j][1])
        nn = buckets[i][2] + buckets[j][2]
        for k in sorted((i, j), reverse=True):
            buckets.pop(k)
        buckets.insert(min(i, j), [lo, hi, nn])
    return [(b[0], b[1]) for b in buckets]


def reliability_table(y, p, edges=None, index=None, half_at=None,
                      min_n: int = MIN_BUCKET_N, block: int = 60,
                      n_boot: int = 2000, seed: int = 7) -> pd.DataFrame:
    """The deliverable: is a stated P auditable?

    One row per displayed bucket with the observed frequency, its block-bootstrap CI, the
    calibration error, and — critically — **the observed frequency in each half of the sample.**

    The half columns are not decoration. §3ap reported a pooled t +2.39 with *every cell negative
    in 2018-21 and positive in 2022-26*; a reliability table that is only calibrated in the
    recent half is that same failure wearing a different hat, and it is invisible in the pooled
    column. PREREGISTRATION.md §7 gate 5 requires the top bucket to sit above the base rate in
    BOTH halves, and this is the column that decides it.
    """
    y, p = _clean(y, p)
    e = np.asarray(edges if edges is not None else prob_edges("direction"), float)
    raw_idx = np.clip(np.digitize(p, e[1:-1]), 0, len(e) - 2)
    counts = [int((raw_idx == b).sum()) for b in range(len(e) - 1)]
    kept = merge_thin_buckets(counts, e, float(np.mean(y)), min_n)

    halves = None
    if index is not None and len(index) == len(y):
        ix = pd.DatetimeIndex(index)
        cut = pd.Timestamp(half_at) if half_at is not None else ix[len(ix) // 2]
        halves = (ix < cut, ix >= cut)

    rows = []
    for lo, hi in kept:
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= 1.0)
        if not m.any():
            continue
        yk = y[m]
        ci_lo, ci_hi, se = block_bootstrap_ci(yk, block=block, n_boot=n_boot, seed=seed)
        row = dict(lo=lo, hi=hi, n=int(m.sum()), share=float(m.mean()),
                   mean_fc=float(np.mean(p[m])), observed=float(np.mean(yk)),
                   ci_lo=ci_lo, ci_hi=ci_hi, se=se)
        row["calib_err"] = row["observed"] - row["mean_fc"]
        if halves is not None:
            for nm, hm in zip(("obs_first_half", "obs_second_half"), halves):
                sub = y[m & hm]
                row[nm] = float(np.mean(sub)) if len(sub) >= 20 else float("nan")
                row[f"n_{nm.split('_')[1]}"] = int(len(sub))
        rows.append(row)
    return pd.DataFrame(rows)


def operating_points(p, y, thresholds=(0.52, 0.55, 0.58, 0.60, 0.65),
                     index=None, hysteresis: float = 0.0,
                     block: int = 60, n_boot: int = 2000, seed: int = 7) -> pd.DataFrame:
    """Coverage, observed frequency, and how OFTEN a threshold would fire.

    `crossings_per_month` and `median_dwell_h` are what tell the live state machine whether
    "P >= 0.60 -> UPTREND" fires twice a year or forty times a month. With `hysteresis > 0` the
    state is entered at `tau` and only released below `tau - hysteresis`, which is the dead-band
    the service actually implements — so the dwell reported here is the dwell the user will see.
    """
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    ix = pd.DatetimeIndex(index)[ok] if index is not None else None
    rows = []
    for tau in thresholds:
        on = _latch(p, tau, hysteresis)
        n = int(on.sum())
        if n < 10:
            rows.append(dict(threshold=tau, coverage=float(on.mean()), n=n))
            continue
        ci_lo, ci_hi, _ = block_bootstrap_ci(y[on], block=block, n_boot=n_boot, seed=seed)
        r = dict(threshold=tau, coverage=float(on.mean()), n=n,
                 observed=float(np.mean(y[on])), ci_lo=ci_lo, ci_hi=ci_hi)
        starts = int(np.sum(on & ~np.r_[False, on[:-1]]))
        if ix is not None and len(ix) > 1:
            months = max((ix[-1] - ix[0]).days / 30.437, 1e-9)
            r["crossings_per_month"] = starts / months
            spans, cur = [], 0
            step_h = np.median(np.diff(ix.values).astype("timedelta64[m]")
                               .astype(float)) / 60.0
            for v in on:
                if v:
                    cur += 1
                elif cur:
                    spans.append(cur); cur = 0
            if cur:
                spans.append(cur)
            r["median_dwell_h"] = float(np.median(spans) * step_h) if spans else 0.0
        rows.append(r)
    return pd.DataFrame(rows)


def staleness_curve(y, p, minutes_since, bins=(0, 60, 120, 180, 240)) -> pd.DataFrame:
    """Brier and accuracy as a function of how stale the underlying bar is.

    The advisor refreshes every 5 minutes but its features only change on an H4 close, so a
    printed number can be up to 4 hours old. This is what makes the card's age stamp honest
    rather than cosmetic: if accuracy decays materially within the bar, the card should say so.
    """
    y, p = _clean(y, p)
    ms = np.asarray(minutes_since, float)[:len(y)]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (ms >= lo) & (ms < hi)
        if m.sum() < 30:
            continue
        rows.append(dict(min_lo=lo, min_hi=hi, n=int(m.sum()),
                         brier=brier(y[m], p[m]),
                         acc=float(np.mean((p[m] >= 0.5) == (y[m] >= 0.5))),
                         base=float(np.mean(y[m]))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- internals
def _clean(y, p):
    y = np.asarray(y, float).ravel()
    p = np.asarray(p, float).ravel()
    if len(y) != len(p):
        raise ValueError(f"length mismatch: y={len(y)} p={len(p)}")
    ok = np.isfinite(y) & np.isfinite(p)
    return y[ok], p[ok]


def _cal_err(y, p, edges, reduce: str) -> float:
    y, p = _clean(y, p)
    e = np.asarray(edges if edges is not None else prob_edges("decile"), float)
    idx = np.clip(np.digitize(p, e[1:-1]), 0, len(e) - 2)
    gaps, ws = [], []
    for b in np.unique(idx):
        m = idx == b
        gaps.append(abs(float(np.mean(p[m])) - float(np.mean(y[m]))))
        ws.append(m.sum())
    if not gaps:
        return float("nan")
    return float(max(gaps)) if reduce == "max" else float(np.average(gaps, weights=ws))


def _latch(p, tau: float, hysteresis: float) -> np.ndarray:
    """Threshold with an optional release band: enter at `tau`, leave below `tau - hysteresis`."""
    if hysteresis <= 0:
        return p >= tau
    on = np.zeros(len(p), bool)
    state = False
    rel = tau - hysteresis
    for i, v in enumerate(p):
        state = v >= tau if not state else v >= rel
        on[i] = state
    return on
