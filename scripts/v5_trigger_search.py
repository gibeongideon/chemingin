"""EXHAUSTIVE CONDITIONAL TRIGGER SEARCH on XAUUSD H4, with a maximum-statistic bootstrap null.

THE QUESTION: is there any market-structure CONDITION such that "when you see this, act this
way" carries a real edge? Not a continuous forecast (§3aj's RL agent, the champion itself), and
not a hand-named pattern from a trading course (§3v-3w's twelve ICT concepts). A systematic
sweep of the conjunction space, with the multiple-testing problem handled properly instead of
being waved at.

WHY THE MULTIPLE-TESTING CONTROL IS THE WHOLE POINT. Search 25,000 conditions and the best one
will look excellent by construction — the expected maximum t-statistic under pure noise across
25k correlated tests is around 4. Every previous "found edge" in this repo died exactly here:
§3w's SMT divergence looked like +0.27 full-sample and collapsed to -0.40 walk-forward; §3v's
EQH/EQL fade posted +1.54 in one regime segment out of four. So the test here is not "is my best
trigger significant" but **"is my best trigger better than the best trigger I would expect to
find in data with no edge at all"** — White's Reality Check / Hansen's SPA, implemented as a
STATIONARY BLOCK BOOTSTRAP over the maximum standardised statistic.

The block bootstrap is not decoration. Forward returns over h bars OVERLAP, so they are
autocorrelated by construction and an analytic t-statistic is invalid — it would overstate
significance by roughly sqrt(h). Resampling in blocks longer than h reproduces that exact
dependence under the null, so the comparison is apples to apples.

THE TRIGGER SPACE — 11 causal, scale-free state variables, each ranked against its own trailing
history (expanding percentile, so no full-sample leakage) and cut into terciles:

    trend       EWMAC forecast, momentum over 24 bars
    location    position in the 40-bar Donchian range, distance from MA in vol units
    volatility  realised vol percentile, range expansion (TR / ATR)
    path        consecutive-direction streak, overnight gap in vol units
    activity    tick-volume percentile          <- the data type §3ag found barely used
    calendar    session (Asian / London / NY), day-of-week bucket

Conjunctions to depth 3 (~5,000 conditions) x 5 horizons (1, 3, 6, 12, 24 H4 bars = 4h to 4
days) = ~25,000 tests, every one of them counted in the null.

DELIBERATE SEPARATION OF TWO QUESTIONS, because conflating them is how cost-doomed "edges" get
published: the statistical test runs on RAW forward returns (is any conditional mean real?),
then survivors alone are priced at 2.76bp round trip (is it tradeable?).

AND THEN THE PART THAT USUALLY KILLS IT: whatever survives the null is re-tested out of sample.
Triggers are discovered on 2015-2020 only and evaluated on 2021-2026, which the search never
saw. A trigger that survives the max-statistic null but not the holdout was still noise.

    python scripts/v5_trigger_search.py --boot 300
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_volregime_taper_crossasset import load_sleeve  # noqa: E402
from src.v5.xau_dual_signals import ewmac_fc, EWMAC_MID  # noqa: E402

HORIZONS = (1, 3, 6, 12, 24)          # H4 bars: 4 hours to 4 days
ANN, VOL_HL = 252 * 6, 42
COST_RT_BP = 2.0 * 0.75 * 1.23 * 1.5  # round trip on GoldEternal, 2.76bp
MIN_SUPPORT = 150                     # a trigger must fire at least this often
DISCOVER_END, HOLDOUT_START = "2020-12-31", "2021-01-01"
RANK_WIN = 252 * 6 * 2                # 2y trailing window for the causal percentile


def state_vars(df: pd.DataFrame) -> pd.DataFrame:
    """Eleven causal, scale-free state variables. Everything continuous is converted to a
    percentile against its OWN trailing 2-year history before binning, so no threshold is
    ever fitted on data the trigger would not have had."""
    c, h, l, o = (df[k].astype(float) for k in ("close", "high", "low", "open"))
    ret = c.pct_change()
    vol = ret.ewm(halflife=VOL_HL, min_periods=20).std()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(halflife=VOL_HL, min_periods=20).mean()
    hi40, lo40 = c.rolling(240).max(), c.rolling(240).min()
    up = (ret > 0).astype(int)
    streak = up.groupby((up != up.shift()).cumsum()).cumcount() + 1
    raw = {
        "trend":   ewmac_fc(c, EWMAC_MID),
        "mom24":   np.log(c / c.shift(24)) / (vol.replace(0, np.nan) * np.sqrt(24)),
        "donch":   (c - lo40) / (hi40 - lo40).replace(0, np.nan),
        "ma_dist": (c / c.ewm(span=120, min_periods=60).mean() - 1) / vol.replace(0, np.nan),
        "vol":     vol,
        "rng_exp": tr / atr.replace(0, np.nan),
        "streak":  np.where(ret > 0, streak, -streak),
        "gap":     (o - c.shift()) / (c.shift() * vol.replace(0, np.nan)),
        "tickvol": df["tick_volume"].astype(float),
    }
    S = {}
    for k, v in raw.items():
        v = pd.Series(v, index=c.index).replace([np.inf, -np.inf], np.nan)
        pct = v.rolling(RANK_WIN, min_periods=252).rank(pct=True)
        S[k] = np.clip((pct * 3).astype(float), 0, 2.999) // 1
    hour = c.index.hour
    S["session"] = pd.Series(np.select([hour < 8, hour < 16], [0, 1], default=2), index=c.index)
    dow = c.index.dayofweek
    S["dow"] = pd.Series(np.select([dow == 0, dow >= 4], [0, 2], default=1), index=c.index)
    return pd.DataFrame(S)


def build_triggers(S: pd.DataFrame, min_support: int) -> tuple[np.ndarray, list[str]]:
    """Depth-1, -2 and -3 conjunctions of (variable == bin). Anything firing fewer than
    `min_support` times in the discovery window is dropped before it can be selected."""
    cols = list(S.columns)
    atoms, names = [], []
    for cvar in cols:
        for b in (0, 1, 2):
            m = (S[cvar].values == b)
            if m.sum() >= min_support:
                atoms.append(m); names.append(f"{cvar}={b}")
    A = np.array(atoms)
    masks, labels = list(A), list(names)
    idx_by_var = {}
    for i, n in enumerate(names):
        idx_by_var.setdefault(n.split("=")[0], []).append(i)
    for depth in (2, 3):
        for vs in combinations(cols, depth):
            for combo in _cartesian([idx_by_var.get(v, []) for v in vs]):
                m = A[combo[0]].copy()
                for j in combo[1:]:
                    m &= A[j]
                if m.sum() >= min_support:
                    masks.append(m)
                    labels.append(" & ".join(names[j] for j in combo))
    return np.array(masks, dtype=np.float32), labels


def _cartesian(lists):
    if not all(lists):
        return []
    out = [[]]
    for L in lists:
        out = [p + [x] for p in out for x in L]
    return out


def fwd_returns(close: pd.Series, h: int) -> pd.Series:
    """Log return over the NEXT h bars, entered at this bar's close."""
    return np.log(close.shift(-h) / close)


def demean(y: np.ndarray) -> np.ndarray:
    """EXCESS return over the unconditional mean. Without this the search does not measure
    conditional edge at all — it measures drift. Gold rose over both windows here, so ANY
    long-biased trigger scores well against zero, which is precisely the bull-beta artifact
    that killed §3o's break-of-structure engine (single-split Sharpe 1.44 -> walk-forward
    0.54 vs buy-and-hold's +209%). Subtracting the unconditional mean asks the only question
    worth asking: does this condition beat simply being in the market?"""
    return y - y.mean()


def trigger_null_p(mask: np.ndarray, y_all: np.ndarray, sign: float, h: int,
                   n_boot: int = 3000, seed: int = 11) -> tuple[float, float, int]:
    """Correct null for ONE trigger, two ways.

    (a) BLOCK-BOOTSTRAP NULL: hold the trigger's firing mask fixed and block-resample the
        forward-return series, which breaks the state->return link while preserving the
        overlap and the volatility clustering. p = P(null conditional excess >= observed).
        This is the single-trigger analogue of the max-statistic null above.
    (b) NON-OVERLAPPING t: keep only fires at least h bars apart, so an ordinary t-test is
        actually valid. Costs power, but needs no assumptions.

    An earlier version of this function was wrong: it compared a bootstrap resample against
    its own expectation via `permutation(samp).mean()`, which cannot change a mean, so it
    returned ~0.5 regardless of the data."""
    rng = np.random.default_rng(seed)
    n = len(y_all)
    idx_fire = np.flatnonzero(mask)
    obs = sign * (y_all[idx_fire].mean() - y_all.mean())
    block = max(2 * h, 24)
    hits = 0
    for _ in range(n_boot):
        starts = rng.integers(0, n - block, n // block + 2)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        yb = y_all[idx]
        if sign * (yb[idx_fire].mean() - yb.mean()) >= obs:
            hits += 1
    keep, last = [], -10**9
    for i in idx_fire:
        if i - last >= h:
            keep.append(i); last = i
    kn = np.array(keep)
    exc = sign * (y_all[kn] - y_all.mean())
    t_nl = exc.mean() / (exc.std(ddof=1) / np.sqrt(len(kn))) if len(kn) > 3 else np.nan
    return hits / n_boot, float(t_nl), len(kn)


def max_stat(M: np.ndarray, sup: np.ndarray, y: np.ndarray, sd: float) -> tuple[float, int]:
    """Standardised conditional mean for every trigger; returns (max |z|, argmax)."""
    z = (M @ y) / np.sqrt(sup) / sd
    i = int(np.abs(z).argmax())
    return float(abs(z[i])), i


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--min-support", type=int, default=MIN_SUPPORT)
    args = ap.parse_args()
    t0 = time.time()

    df = load_sleeve(dict(path="data/XAUUSD_H4_long.csv"))
    S = state_vars(df)
    close = df["close"].astype(float)
    Y = {h: fwd_returns(close, h) for h in HORIZONS}
    ok = S.notna().all(axis=1) & pd.concat(Y, axis=1).notna().all(axis=1)
    S, close = S[ok], close[ok]
    Y = {h: Y[h][ok] for h in HORIZONS}
    disc = close.index <= DISCOVER_END
    print(f"XAU H4: {ok.sum():,} usable bars {close.index.min():%Y-%m-%d} -> "
          f"{close.index.max():%Y-%m-%d}")
    print(f"discovery {disc.sum():,} bars (<= {DISCOVER_END}) | "
          f"holdout {(~disc).sum():,} bars (>= {HOLDOUT_START})")

    M_all, labels = build_triggers(S[disc], args.min_support)
    print(f"trigger space: {len(labels):,} conditions x {len(HORIZONS)} horizons = "
          f"{len(labels) * len(HORIZONS):,} tests  ({time.time() - t0:.0f}s)")
    sup = M_all.sum(1)

    # ---- observed max statistic across the ENTIRE search ----
    obs, best = -1.0, None
    per_h = {}
    for h in HORIZONS:
        y = demean(Y[h][disc].values.astype(np.float32))
        sd = float(y.std())
        z, i = max_stat(M_all, sup, y, sd)
        per_h[h] = (z, labels[i])
        if z > obs:
            obs, best = z, (h, i)
    print(f"\nobserved max |z| over all tests: {obs:.3f}")
    for h in HORIZONS:
        print(f"    h={h:2d}: best |z| {per_h[h][0]:5.2f}   {per_h[h][1]}")

    # ---- stationary block bootstrap null on the MAXIMUM ----
    rng = np.random.default_rng(7)
    n = int(disc.sum())
    null = np.zeros(args.boot)
    for b in range(args.boot):
        mx = -1.0
        for h in HORIZONS:
            block = max(2 * h, 24)                 # must exceed the overlap length
            starts = rng.integers(0, n - block, n // block + 1)
            idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
            y = demean(Y[h][disc].values.astype(np.float32)[idx])
            z, _ = max_stat(M_all, sup, y, float(y.std()))
            mx = max(mx, z)
        null[b] = mx
        if (b + 1) % 50 == 0:
            print(f"  bootstrap {b + 1}/{args.boot}  null max|z| p50={np.median(null[:b+1]):.2f}"
                  f"  p95={np.quantile(null[:b+1], .95):.2f}  ({time.time() - t0:.0f}s)")
    p = float((null >= obs).mean())
    print(f"\nNULL DISTRIBUTION of the maximum: p50 {np.median(null):.2f}  "
          f"p95 {np.quantile(null, .95):.2f}  p99 {np.quantile(null, .99):.2f}  "
          f"max {null.max():.2f}")
    print(f"OBSERVED {obs:.3f}  ->  Reality-Check p = {p:.3f}  "
          f"{'*** SURVIVES the search-corrected null ***' if p < 0.05 else 'INDISTINGUISHABLE from the best trigger a no-edge dataset yields'}")

    # ---- holdout, regardless of the null's verdict ----
    h_best, i_best = best
    print(f"\n--- OUT-OF-SAMPLE test of the single best discovery-window trigger ---")
    print(f"trigger: {labels[i_best]}   horizon {h_best} bars")
    Sh = S[~disc]
    mask_h = np.ones(len(Sh), dtype=bool)
    for part in labels[i_best].split(" & "):
        v, b = part.split("=")
        mask_h &= (Sh[v].values == float(b))
    all_d, all_h = Y[h_best][disc].values, Y[h_best][~disc].values
    yd = all_d[M_all[i_best].astype(bool)]
    yh = all_h[mask_h]
    sign = np.sign(yd.mean() - all_d.mean())
    cost = COST_RT_BP * 1e-4
    print(f"{'window':11s} {'fires':>6s} {'EXCESS bp':>10s} {'net bp':>8s} "
          f"{'naive t':>8s} {'boot p':>7s} {'non-ovl n':>10s} {'non-ovl t':>10s}")
    masks = {"discovery": M_all[i_best].astype(bool), "HOLDOUT": mask_h}
    for tag, allr in (("discovery", all_d), ("HOLDOUT", all_h)):
        mk = masks[tag]
        arr = allr[mk]
        if len(arr) < 5:
            print(f"{tag:11s} {len(arr):6d}  too few fires"); continue
        exc = sign * (arr - allr.mean())
        net = exc - cost
        t = net.mean() / (net.std(ddof=1) / np.sqrt(len(net)))
        bp_, t_nl, kn = trigger_null_p(mk, allr, sign, h_best)
        print(f"{tag:11s} {len(arr):6d} {exc.mean() * 1e4:10.2f} {net.mean() * 1e4:8.2f} "
              f"{t:8.2f} {bp_:7.3f} {kn:10d} {t_nl:10.2f}")
    print(f"\ndirection: {'LONG' if sign > 0 else 'SHORT'} on fire, hold {h_best} bars, "
          f"cost {COST_RT_BP:.2f}bp round trip")
    pd.DataFrame(dict(horizon=[h for h in HORIZONS],
                      best_z=[per_h[h][0] for h in HORIZONS],
                      best_trigger=[per_h[h][1] for h in HORIZONS])).to_csv(
        ROOT / "data/v5_runs/trigger_search.csv", index=False)
    print(f"elapsed {time.time() - t0:.0f}s -> data/v5_runs/trigger_search.csv")


if __name__ == "__main__":
    main()
