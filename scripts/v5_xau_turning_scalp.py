"""XAU bottom-detector "cash out on first reasonable profit" scalp — backtest
BEFORE any live deployment.

User asked to deploy "the 72% accuracy zigzag" on the cent account, parallel to
the existing dual bots, cashing out as soon as a trade is positive (no
holding for a bigger target). Clarified with the user: this means
`v5_xau_turning_ml.py`'s BOTTOM/BUY detector (documented ceiling: 71%
precision @ 22% recall, up to 80% @ 5% recall — the closest number this repo
has to "72%"; SELL/tops never gets close, so this is LONG-ONLY by
construction). That number was measured as pure classification precision on a
static 70/30 split — NEVER before backtested as an actual "enter on flag,
exit at first positive" trade structure. Every P&L structure previously tried
with a related bottom/turning-point detector failed
(`xau-turning-point-detectability`: overlay-on-champion fails; hold-48-bars
merely harvests drift and loses to buy&hold) — but neither of those is THIS
structure, so this gets its own honest test rather than being waved through
by precedent, per this repo's culture.

Pipeline (same rigor as every other live-candidate signal here):
  1. Walk-forward, expanding, refit each calendar year on strictly-past data
     (purge = tol, matching `v5_xau_intermarket_accuracy.py`'s pattern) —
     produces an honest OOS bottom-probability at every H1 bar from 2018 on.
  2. Grid over (probability threshold, take-profit %, stop-loss %, max-hold):
     enter LONG next bar's open when prob >= threshold and flat; exit at the
     first bar whose HIGH clears entry*(1+tp_pct) (SL checked first on a
     same-bar tie, conservative) or whose LOW breaches entry*(1-sl_pct), or
     after max_hold bars at that bar's close.
  3. Live cent-account cost: $0.34 spread (documented cent floor) + $0.10
     slip, round-trip.
  4. Walk-forward SELECTION of the grid cell too (trailing 3y, re-picked each
     year) — not a full-sample "best cell", per this repo's mandatory control.

    python scripts/v5_xau_turning_scalp.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_xau_turning_ml import atr, zigzag_swings, label_near, features  # noqa: E402
from src.evaluation.lookahead_probe import probe_lookahead, print_report, synthetic_ohlc  # noqa: E402
from src.evaluation.dsr_pbo import deflated_sharpe_ratio, pbo_cscv  # noqa: E402
from src.evaluation.walk_forward_grid import walk_forward_select, print_selections  # noqa: E402

TF = "H1"
ORDER, THETA_MULT, TOL = 5, 1.5, 3     # matches the documented 71%@22% config
EVAL_START = "2018-01-01"
YEARS = list(range(2018, 2027))
SELECT_WINDOW_BARS = 3 * 252 * 24      # trailing 3y, H1
SPREAD_USD = 0.34                       # live cent-account floor (xau-live-cent-account)
SLIP_USD = 0.10

THRESHOLDS = (0.5, 0.6, 0.7)
TP_PCTS = (0.0015, 0.0025, 0.004, 0.006)
SL_PCTS = (0.003, 0.005)
MAX_HOLDS = (24, 48)


def load() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / f"XAUUSD_{TF}_long.csv", parse_dates=["time"], index_col="time").sort_index()
    return d[~d.index.duplicated(keep="last")]


def walk_forward_proba(X: pd.DataFrame, y: np.ndarray, purge: int, years: list[int]) -> pd.Series:
    """Expanding-window, refit each calendar year on strictly-past data with a
    `purge`-bar gap before the test year (identical pattern to
    `v5_xau_intermarket_accuracy.py::walk_forward`), returned as a Series
    ALIGNED to X's index (NaN outside any test year, i.e. before the first
    trainable year)."""
    idx = X.index
    valid = np.isfinite(X.values).all(axis=1)
    out = pd.Series(np.nan, index=idx)
    for yr in years:
        test_mask = (idx.year == yr) & valid
        if test_mask.sum() == 0:
            continue
        cutoff = pd.Timestamp(f"{yr}-01-01")
        train_end_pos = np.searchsorted(idx.values, np.datetime64(cutoff)) - purge
        if train_end_pos < 2000:
            continue
        train_mask = np.zeros(len(idx), dtype=bool)
        train_mask[:train_end_pos] = True
        train_mask &= valid
        Xtr, ytr = X.values[train_mask], y[train_mask]
        if len(np.unique(ytr)) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                             l2_regularization=1.0, validation_fraction=0.15, random_state=7)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(X.values[test_mask])[:, 1]
        out.loc[test_mask] = p
    return out


def simulate(df: pd.DataFrame, prob: pd.Series, threshold: float, tp_pct: float,
            sl_pct: float, max_hold: int) -> pd.Series:
    """LONG-ONLY: enter next bar's open when prob>=threshold and flat; exit at
    first TP/SL touch (SL wins a same-bar tie) or max_hold bars at close.
    Returns a per-bar net return Series (0 when flat/no position change)."""
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    n = len(df)
    p = prob.reindex(df.index).values
    cost_frac = np.asarray((SPREAD_USD / 2.0 + SLIP_USD) / df["close"])
    ret = np.zeros(n)
    i = 0
    while i < n - 1:
        if not np.isfinite(p[i]) or p[i] < threshold:
            i += 1
            continue
        entry_i = i + 1
        entry = o[entry_i]
        if not np.isfinite(entry) or entry <= 0:
            i += 1
            continue
        tp = entry * (1 + tp_pct)
        sl = entry * (1 - sl_pct)
        exit_px, exit_i = c[min(entry_i + max_hold, n - 1)], min(entry_i + max_hold, n - 1)
        for k in range(entry_i, min(entry_i + 1 + max_hold, n)):
            if l[k] <= sl:
                exit_px, exit_i = sl, k
                break
            if h[k] >= tp:
                exit_px, exit_i = tp, k
                break
        gross = (exit_px - entry) / entry
        net = gross - cost_frac[entry_i] - cost_frac[exit_i]
        ret[exit_i] += net
        i = exit_i + 1
    return pd.Series(ret, index=df.index)


def sharpe(d: pd.Series, start: str = EVAL_START) -> float:
    dd = d.loc[start:].resample("D").sum()
    dd = dd[dd.index.dayofweek < 5]
    return float(dd.mean() / dd.std() * np.sqrt(252)) if dd.std() > 0 else float("nan")


def dd_pct(d: pd.Series, start: str = EVAL_START) -> float:
    e = (1 + d.loc[start:]).cumprod()
    return float((e / e.cummax() - 1).min() * 100)


def probe_features(n: int = 3000, seed: int = 7):
    """`features()` is reused verbatim from `v5_xau_turning_ml.py` — never
    formally lookahead-probed before (that script only ever ran full-sample
    research passes, not a live deployment). Probe it directly here since
    this is now a live-deployment candidate."""
    # probe_lookahead expects a Series-returning fn; wrap as the row-sum of all features
    def fn_series(d):
        f = features(d)
        return f.sum(axis=1, skipna=True)
    return probe_lookahead(fn_series, n=n, seed=seed, offsets=(200, 120, 60, 30), warmup_buffer=300)


def main():
    print("=== STAGE 0: lookahead probe (features() from v5_xau_turning_ml.py) ===")
    result = probe_features()
    print_report({"turning_ml_features(sum)": result})
    if not result.ok:
        print("!! lookahead probe FAILED — refusing to trust the backtest below.")
        return

    df = load()
    a = atr(df)
    theta = THETA_MULT * a
    _, buys = zigzag_swings(df, ORDER, theta)
    y = label_near(buys, len(df), TOL)
    X = features(df)

    print(f"\n=== STAGE 1: walk-forward OOS bottom-probability, {len(buys)} true buy swings ===")
    prob = walk_forward_proba(X, y, purge=TOL, years=YEARS)
    n_scored = int(prob.notna().sum())
    print(f"  scored {n_scored} bars ({prob.dropna().index[0]}..{prob.dropna().index[-1]})")

    print("\n=== STAGE 2: precompute (threshold, tp, sl, max_hold) grid ===")
    grid_rets = {}
    for thr in THRESHOLDS:
        for tp in TP_PCTS:
            for sl in SL_PCTS:
                for mh in MAX_HOLDS:
                    key = (thr, tp, sl, mh)
                    r = simulate(df, prob, thr, tp, sl, mh)
                    grid_rets[key] = r.loc[EVAL_START:].resample("D").sum()
                    grid_rets[key] = grid_rets[key][grid_rets[key].index.dayofweek < 5]
    print(f"  grid size: {len(grid_rets)}")

    full_sh = {k: (float(v.mean() / v.std() * np.sqrt(252)) if v.std() > 0 else -np.inf)
              for k, v in grid_rets.items()}
    best_full = max(full_sh, key=full_sh.get)
    print(f"(reference only, NOT trusted) full-sample-best: thr={best_full[0]} tp={best_full[1]:.3%} "
          f"sl={best_full[2]:.3%} hold={best_full[3]}  SR={full_sh[best_full]:+.3f}  "
          f"DD={dd_pct(grid_rets[best_full]):+.1f}%")
    print("top 8 by full-sample Sharpe:")
    for k in sorted(full_sh, key=full_sh.get, reverse=True)[:8]:
        n_trades = int((grid_rets[k] != 0).sum())
        print(f"  thr={k[0]} tp={k[1]:.3%} sl={k[2]:.3%} hold={k[3]:>3d}  SR={full_sh[k]:+.3f}  "
              f"DD={dd_pct(grid_rets[k]):+.1f}%  active_days={n_trades}")

    print("\n=== STAGE 3: WALK-FORWARD selection (trailing 3y, re-picked each Jan) — the honest number ===")
    oos, selections = walk_forward_select(grid_rets, YEARS, SELECT_WINDOW_BARS, min_train_bars=100)
    print_selections(selections)
    if oos.empty or oos.std() == 0:
        print("!! no OOS data / zero variance.")
        return
    sr_wf = float(oos.mean() / oos.std() * np.sqrt(252))
    dd_wf = dd_pct(oos, start=oos.index[0].strftime("%Y-%m-%d"))
    print(f"\nWALK-FORWARD scalp bot: SR={sr_wf:+.3f}  DD={dd_wf:+.1f}%")

    bh_ret = df["close"].pct_change().loc[EVAL_START:].resample("D").sum()
    bh_ret = bh_ret[bh_ret.index.dayofweek < 5]
    j = pd.concat([oos.rename("s"), bh_ret.rename("b")], axis=1).dropna()
    delta = j["s"] - j["b"]
    t_bh = float(delta.mean() / delta.std() * np.sqrt(len(delta))) if delta.std() > 0 else float("nan")
    yrp = int((delta.groupby(delta.index.year).mean() > 0).sum())
    yrn = int(delta.groupby(delta.index.year).mean().shape[0])
    print(f"buy&hold(unscaled) SR={float(bh_ret.mean()/bh_ret.std()*np.sqrt(252)):+.3f}   "
          f"paired t vs buy&hold: {t_bh:+.2f}  years better {yrp}/{yrn}")

    trial_sharpes = np.array([s / np.sqrt(252) for s in full_sh.values() if np.isfinite(s)])
    dsr = deflated_sharpe_ratio(oos.values, trial_sharpes)
    common_idx = None
    for v in grid_rets.values():
        common_idx = v.index if common_idx is None else common_idx.intersection(v.index)
    mats = [v.reindex(common_idx).fillna(0.0).values for v in grid_rets.values()]
    pbo_res = pbo_cscv(np.column_stack(mats), n_partitions=10)
    print(f"DSR={dsr['dsr']:.3f} (n_trials={dsr['n_trials']})   PBO={pbo_res.pbo:.3f}")

    print("\nper-year:")
    for yr, g in oos.groupby(oos.index.year):
        sr_y = float(g.mean() / g.std() * np.sqrt(252)) if g.std() > 0 else float("nan")
        n_tr = int((g != 0).sum())
        print(f"  {yr}  SR {sr_y:+.2f}  active_days={n_tr}")

    print("\n=== Verdict ===")
    print(f"walk-forward SR={sr_wf:+.3f} vs buy&hold SR={float(bh_ret.mean()/bh_ret.std()*np.sqrt(252)):+.3f}, "
          f"DSR={dsr['dsr']:.3f}, PBO={pbo_res.pbo:.3f}")
    print("MANDATORY bar for 'deploy this live': DSR>=0.90 AND PBO<0.30 AND walk-forward-positive.")


if __name__ == "__main__":
    main()
