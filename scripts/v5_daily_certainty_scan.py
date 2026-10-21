#!/usr/bin/env python3
"""GATE B -- the daily-pattern search for XAUUSD / BTCUSD / the BTC-XAU ratio.

Pre-registered in data/v5_runs/daily_certainty/PREREGISTRATION.md. Gate A established the bar:
the best-of-K p95 noise floor, which at 25 fires and K=500 is already 0.880 precision on a 0.525
base rate. This script runs the search against that bar.

DESIGN NOTE -- why this is fast enough to null properly. Every pattern is a function of data up to
and including day t and predicts day t+1, so the pattern matrix NEVER CHANGES when the outcome is
re-aligned. The boolean matrix P (n_days x K) is therefore built once, and each null draw only
rolls the outcome vector and recomputes precisions by matrix arithmetic. That makes re-running the
ENTIRE search inside every null draw affordable, which is the only way a max-stat p-value prices
the search rather than the winner.

THE NULL IS A CIRCULAR SHIFT, NOT A PERMUTATION. These patterns fire in clusters (a 4-day streak
implies neighbouring days are also streak days), and returns are themselves serially structured.
Permuting returns destroys that structure and understates the floor. Rolling the outcome by a
random lag >= 20 days preserves the autocorrelation of both series and destroys only their
alignment, which is exactly the relationship under test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v5_runs" / "daily_certainty"
RNG = np.random.default_rng(20261021)

# Round trip in bp, measured live at Switch Markets 2026-10-19 (3bs). one-way = 0.75 x spread.
COST_BP = {"GOLD": 0.50 * 1.5, "BTC": 4.46 * 1.5, "RATIO": (0.50 + 4.46) * 1.5}
N_NULL = 600
MIN_FIRES = 25          # below this, Gate A says the claim is unfalsifiable whatever it shows


def load(alias: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / f"{alias}_D1_long.csv", parse_dates=["time"])
    d = d.set_index("time").sort_index()
    return d[~d.index.duplicated(keep="last")]


def intrabar_leak(d: pd.DataFrame) -> float:
    """|corr(clpos[t], ret[t+1])| -- the forward-stamping harm itself. Threshold 0.15."""
    c = d["close"].pct_change()
    x = (d["close"] - d["low"]) / (d["high"] - d["low"]).replace(0, np.nan)
    nx = c.shift(-1)
    ok = x.notna() & nx.notna()
    return float(abs(x[ok].corr(nx[ok]))) if ok.sum() > 300 else 0.0


def features(d: pd.DataFrame) -> pd.DataFrame:
    """Strictly causal: everything here is known at the CLOSE of day t."""
    f = pd.DataFrame(index=d.index)
    c, h, l, o = d["close"], d["high"], d["low"], d["open"]
    r = c.pct_change()
    f["ret"] = r
    f["up"] = (r > 0).astype(float)
    rng = (h - l) / c.shift(1)
    f["rng"] = rng
    f["rng_z"] = (rng - rng.rolling(20).mean()) / rng.rolling(20).std()
    f["clpos"] = (c - l) / (h - l).replace(0, np.nan)
    f["gap"] = o / c.shift(1) - 1.0
    for n in (2, 3, 5, 10, 20, 60):
        f[f"mom{n}"] = c / c.shift(n) - 1.0
        f[f"z{n}"] = (c - c.rolling(n).mean()) / c.rolling(n).std()
    f["vol20"] = r.rolling(20).std()
    f["vol_pct"] = f["vol20"].rolling(250).rank(pct=True)
    # streaks of same-signed closes, counted causally
    sign = np.sign(r).fillna(0.0)
    streak = sign.groupby((sign != sign.shift()).cumsum()).cumcount() + 1
    f["streak"] = streak * sign
    f["dow"] = d.index.dayofweek
    f["dom"] = d.index.day
    return f


def build_patterns(f: pd.DataFrame, x: pd.DataFrame | None,
                   use_intrabar: bool) -> dict[str, pd.Series]:
    """The pre-registered candidate grid. `x` is the cross-asset frame (may be None).

    Every entry is a boolean Series true on days where the pattern is OBSERVED at the close.
    Closed families are deliberately absent -- see the pre-registration's exclusion list.
    """
    P: dict[str, pd.Series] = {}

    # --- A. streaks ------------------------------------------------------------------------
    for k in (2, 3, 4, 5, 6):
        P[f"streak_up_{k}"] = f["streak"] >= k
        P[f"streak_dn_{k}"] = f["streak"] <= -k

    # --- B. gaps ---------------------------------------------------------------------------
    for t in (0.003, 0.006, 0.010):
        P[f"gap_up_{t}"] = f["gap"] > t
        P[f"gap_dn_{t}"] = f["gap"] < -t

    # --- C. range regime (NON-directional NR7 is closed; the CONDITIONAL direction is not) ---
    for lo, hi in ((-9, -1.0), (-1.0, -0.3), (-0.3, 0.3), (0.3, 1.0), (1.0, 9)):
        P[f"rngz_{lo}_{hi}"] = (f["rng_z"] > lo) & (f["rng_z"] <= hi)

    # --- D/E. momentum and reversal buckets --------------------------------------------------
    for n in (2, 3, 5, 10, 20, 60):
        m = f[f"mom{n}"]
        q = m.rolling(250).rank(pct=True)
        for lo, hi in ((0.0, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 1.0)):
            P[f"mom{n}_q{lo}_{hi}"] = (q > lo) & (q <= hi)
        for lo, hi in ((-9, -2), (-2, -1), (1, 2), (2, 9)):
            P[f"z{n}_{lo}_{hi}"] = (f[f"z{n}"] > lo) & (f[f"z{n}"] <= hi)

    # --- H. volatility regime ----------------------------------------------------------------
    for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)):
        P[f"vol_q{lo}_{hi}"] = (f["vol_pct"] > lo) & (f["vol_pct"] <= hi)

    # --- I. calendar ---------------------------------------------------------------------------
    for dw in range(5):
        P[f"dow_{dw}"] = f["dow"] == dw
    P["turn_of_month"] = (f["dom"] >= 28) | (f["dom"] <= 3)
    P["mid_month"] = (f["dom"] >= 12) & (f["dom"] <= 18)

    # --- J. close position, ONLY if the leak screen passed -------------------------------------
    if use_intrabar:
        for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.6, 0.8), (0.8, 1.0)):
            P[f"clpos_{lo}_{hi}"] = (f["clpos"] > lo) & (f["clpos"] <= hi)

    # --- F/G. cross-asset joint states and the ratio (the genuinely new dimension) -------------
    if x is not None:
        P["xa_both_up"] = (f["up"] == 1) & (x["up"] == 1)
        P["xa_both_dn"] = (f["up"] == 0) & (x["up"] == 0)
        P["xa_split_a"] = (f["up"] == 1) & (x["up"] == 0)
        P["xa_split_b"] = (f["up"] == 0) & (x["up"] == 1)
        for n in (5, 20):
            for lo, hi in ((-9, -1.5), (-1.5, -0.5), (0.5, 1.5), (1.5, 9)):
                P[f"xa_z{n}_{lo}_{hi}"] = (x[f"z{n}"] > lo) & (x[f"z{n}"] <= hi)
        # divergence: this asset trending one way while the other trends the other
        for n in (5, 20):
            P[f"xa_div_pos_{n}"] = (f[f"mom{n}"] > 0) & (x[f"mom{n}"] < 0)
            P[f"xa_div_neg_{n}"] = (f[f"mom{n}"] < 0) & (x[f"mom{n}"] > 0)

    # --- two-way conjunctions of the strongest single families ---------------------------------
    base_keys = [k for k in P if k.startswith(("streak_", "rngz_", "vol_q"))]
    trend_keys = [k for k in P if k.startswith(("mom20_q", "mom60_q"))]
    for a in base_keys:
        for b in trend_keys:
            P[f"{a}&{b}"] = P[a] & P[b]

    return P


def evaluate(P: dict[str, pd.Series], nxt_up: pd.Series, nxt_ret: pd.Series,
             cost_bp: float, half: pd.Series) -> pd.DataFrame:
    base_up = float(nxt_up.mean())
    base_ret_bp = float(nxt_ret.mean() * 1e4)
    rows = []
    for name, m in P.items():
        m = m.fillna(False).astype(bool) & nxt_up.notna()
        n = int(m.sum())
        if n < MIN_FIRES:
            continue
        prec = float(nxt_up[m].mean())
        ev = float(nxt_ret[m].mean() * 1e4)
        # Direction the pattern implies: whichever side its own precision favours. Scoring the
        # favoured side is what a trader would do, and the max-stat null prices that free choice.
        side = 1 if prec >= base_up else -1
        prec_side = prec if side == 1 else 1.0 - prec
        ev_side = ev * side - cost_bp
        # excess over HOLDING over the same days (control #2 / cot-positioning-disproven)
        excess_vs_hold = ev_side - (base_ret_bp * side)
        se = float(np.sqrt(prec * (1 - prec) / n))
        h1 = m & (half == 0)
        h2 = m & (half == 1)
        p1 = float(nxt_up[h1].mean()) if h1.sum() >= 10 else np.nan
        p2 = float(nxt_up[h2].mean()) if h2.sum() >= 10 else np.nan
        rows.append(dict(pattern=name, fires=n, side=side, precision=prec_side,
                         base=base_up if side == 1 else 1 - base_up,
                         excess=prec_side - (base_up if side == 1 else 1 - base_up),
                         se=se, ev_bp=ev_side, excess_vs_hold_bp=excess_vs_hold,
                         h1=p1 if side == 1 else 1 - p1,
                         h2=p2 if side == 1 else 1 - p2))
    return pd.DataFrame(rows)


def maxstat_null(P: dict[str, pd.Series], nxt_up: pd.Series, n_null: int) -> np.ndarray:
    """Re-run the WHOLE search under circular shifts; record the max precision each time."""
    keys = list(P)
    M = np.column_stack([P[k].fillna(False).to_numpy(dtype=bool) for k in keys])
    y = nxt_up.to_numpy(dtype=float)
    ok = ~np.isnan(y)
    M, y = M[ok], y[ok]
    counts = M.sum(axis=0).astype(float)
    keep = counts >= MIN_FIRES
    M, counts = M[:, keep], counts[keep]
    n = len(y)
    base = y.mean()
    out = np.empty(n_null)
    for i in range(n_null):
        lag = int(RNG.integers(20, n - 20))
        ys = np.roll(y, lag)
        hits = ys @ M
        prec = hits / counts
        # the search is free to pick a side, so the null must be too
        out[i] = np.maximum(prec, 1.0 - prec).max() if base else np.nan
    return out


def run_asset(name: str, d: pd.DataFrame, cross: pd.DataFrame | None, cost_bp: float) -> None:
    leak = intrabar_leak(d)
    f = features(d)
    x = features(cross).reindex(f.index) if cross is not None else None
    P = build_patterns(f, x, use_intrabar=(leak <= 0.15))

    nxt_ret = f["ret"].shift(-1)
    nxt_up = (nxt_ret > 0).astype(float).where(nxt_ret.notna())
    half = pd.Series((np.arange(len(f)) >= len(f) // 2).astype(int), index=f.index)

    res = evaluate(P, nxt_up, nxt_ret, cost_bp, half)
    K = len(res)
    null = maxstat_null(P, nxt_up, N_NULL)
    floor95 = float(np.nanquantile(null, 0.95))

    print(f"\n{'=' * 98}\n{name}   leak {leak:.3f} "
          f"({'intrabar features ALLOWED' if leak <= 0.15 else 'intrabar features EXCLUDED'})"
          f"   base up {float(nxt_up.mean()):.3f}   K={K}   cost {cost_bp:.2f}bp\n{'=' * 98}")
    print(f"  max-stat null (circular shift, {N_NULL} draws): median "
          f"{float(np.nanmedian(null)):.3f}  p95 {floor95:.3f}  max {float(np.nanmax(null)):.3f}")

    res = res.sort_values("precision", ascending=False)
    top = res.head(12)
    print(f"\n  TOP 12 BY PRECISION (what a naive search would report)")
    print(f"  {'pattern':<34} {'fires':>6} {'sd':>3} {'prec':>6} {'base':>6} {'exc':>7} "
          f"{'EV bp':>8} {'vs hold':>8} {'h1':>6} {'h2':>6}")
    for _, r in top.iterrows():
        print(f"  {r['pattern'][:34]:<34} {r['fires']:>6} {r['side']:>3} {r['precision']:>6.3f} "
              f"{r['base']:>6.3f} {r['excess']:>+7.3f} {r['ev_bp']:>+8.2f} "
              f"{r['excess_vs_hold_bp']:>+8.2f} {r['h1']:>6.3f} {r['h2']:>6.3f}")

    # the pre-registered gate
    surv = res[(res["precision"] > floor95) &
               (res["excess"] > res["se"]) &
               (res["ev_bp"] > 0) &
               (res["excess_vs_hold_bp"] > 0) &
               (((res["h1"] > res["base"]) & (res["h2"] > res["base"])))]
    print(f"\n  SURVIVORS of the pre-registered gate "
          f"(prec > {floor95:.3f} AND excess > 1 SE AND EV > 0 AND beats hold AND both halves):"
          f" {len(surv)} of {K}")
    if len(surv):
        print(surv.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    res.to_csv(OUT / f"scan_{name}.csv", index=False)
    np.save(OUT / f"null_{name}.npy", null)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gold, btc = load("GOLD"), load("BTC")
    common = gold.index.intersection(btc.index)
    ratio = pd.DataFrame({
        "open": btc["open"].reindex(common) / gold["open"].reindex(common),
        "high": btc["high"].reindex(common) / gold["low"].reindex(common),
        "low": btc["low"].reindex(common) / gold["high"].reindex(common),
        "close": btc["close"].reindex(common) / gold["close"].reindex(common),
    }).dropna()

    print("=" * 98)
    print("GATE B -- daily pattern search. Pre-registered; the bar is Gate A's best-of-K p95 floor.")
    print("=" * 98)
    run_asset("GOLD", gold, btc.reindex(gold.index), COST_BP["GOLD"])
    run_asset("BTC", btc, gold.reindex(btc.index), COST_BP["BTC"])
    run_asset("RATIO", ratio, None, COST_BP["RATIO"])


if __name__ == "__main__":
    main()
