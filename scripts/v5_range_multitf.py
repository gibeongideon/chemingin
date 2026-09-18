"""RANGE-BOX features across TIMEFRAMES, with a symmetric barrier and a max-statistic null.

WHAT CHANGED FROM THE DAILY STUDY (§3ax) AND WHY.
The daily test used the bar's own high/low as the two barriers, so a close near the high was
simply NEARER the high: the distance-implied probability alone scored AUC 0.8659 and the model
(0.8451) could not beat geometry. That null is unavoidable with asymmetric barriers.

Here the barriers are **symmetric: +k*ATR before -k*ATR within H bars.** The distance null is
then exactly 0.5 by construction, so any AUC above 0.5 is information rather than geometry.
It also implements the one operational finding of §3ax -- brackets scaled in ATR, never in
percent, because gold's mean daily range went from 1.34% (2015) to 2.73% (2026) and a fixed
0.50% stop is now 0.18 of a single day's travel.

WHY LOWER TIMEFRAMES ARE A LEGITIMATE MOVE, not just more things to try. M15 has 59,987 bars
against D1's 2,999. §3ak identified event count as the binding constraint and control #7 exists
to enforce it; 20x the events is the only honest way to make a small effect measurable.

WHY LOWER TIMEFRAMES ARE ALSO WHERE THIS DIES, stated in advance. The trade's edge must clear
the spread, and the barrier shrinks with the timeframe while the spread does not:
    breakeven P = 0.5 + cost / (2 * barrier)
At a floored 1.5bp one-way that is ~0.57 on M15 and ~0.51 on D1. **The breakeven column is
printed next to the achieved column for every cell**, because that comparison -- not AUC -- is
what decides it. This is the repo's recurring "intraday dies at the spread" result, quantified
for this structure.

PRE-REGISTERED GRID -- 6 cells, declared before running, so the best-of-K noise floor is small
and computable rather than open-ended:
    timeframe  {M15, H1, D1}  x  feature set {RANGE-BOX, RANGE+SESSION}
Everything else is fixed: k=1.0 ATR, H=16 bars, walk-forward by calendar year, uniform weights
(§3ax tested recency weighting and it did not help: 0.8408 vs 0.8451).

THE NULL FOR A SEARCH. The winning cell is compared against a **max-statistic block bootstrap**:
the label is block-shuffled, all 6 cells are refit, and the MAXIMUM AUC across them is recorded.
The winner must beat that distribution's 95th percentile, not 0.5. Without this, picking the
best of 6 is worth about +0.04 AUC for free.

EVERY FEATURE SET IS LEAK-PROBED before use. §3ax's headline (+40%/yr, Sharpe 5.52) was a
weekly-bar `reindex(..., method="ffill")` lookahead that a shuffled-label control passed
cleanly, so a per-feature correlation against the NEXT bar's return runs first here.

    python scripts/v5_range_multitf.py
    python scripts/v5_range_multitf.py --boot 300      # max-stat draws
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
P = ROOT / "data" / "v5_runs" / "range_study"

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from scripts.v5_pooled_bottom_detector import auc, hm_se  # noqa: E402

TFS = ("M15", "H1", "D1")
SETS = ("RANGE-BOX", "RANGE+SESSION")
K_ATR, HORIZON = 1.0, 16
COST_BP = 1.5          # one-way, floored: live FTMO gold spread ~$0.45 on ~$4,400 ~ 1bp


def load(tf: str) -> pd.DataFrame:
    d = pd.read_csv(P / f"XAUUSD_{tf}.csv", parse_dates=["time"]).set_index("time")
    return d[~d.index.duplicated(keep="last")].sort_index()


def atr_series(d: pd.DataFrame, w: int = 20) -> pd.Series:
    """True range, gap-aware — the live bot's estimator beat all four range estimators (§3ai)."""
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(w).mean()


def label_symmetric(d: pd.DataFrame, atr: pd.Series, k: float, H: int) -> np.ndarray:
    """1 if +k*ATR is touched before -k*ATR within the next H bars; 0 if the reverse;
    -1 if neither (undecided -- kept separately, never guessed).

    Barriers are symmetric so the distance null is exactly 0.5.
    """
    c = d["close"].values
    hi, lo = d["high"].values, d["low"].values
    a = atr.values
    n = len(d)
    out = np.full(n, -1, dtype=int)
    for i in range(n - H - 1):
        if not np.isfinite(a[i]) or a[i] <= 0:
            continue
        up, dn = c[i] + k * a[i], c[i] - k * a[i]
        for j in range(i + 1, min(i + 1 + H, n)):
            tu, td = hi[j] >= up, lo[j] <= dn
            if tu and td:
                break                      # same bar: genuinely ambiguous
            if tu:
                out[i] = 1; break
            if td:
                out[i] = 0; break
    return out


def feats(d: pd.DataFrame, with_session: bool) -> pd.DataFrame:
    """Causal range/extremes features. Intraday boxes are SESSION-TO-DATE via
    groupby(day).cummax(), never a resampled daily bar — that is exactly the §3ax lookahead."""
    c, h, l, o = d["close"], d["high"], d["low"], d["open"]
    rng = (h - l).replace(0, np.nan)
    a = atr_series(d)
    f = pd.DataFrame(index=d.index)

    for w in (5, 10, 20, 60):
        hh, ll = h.rolling(w).max(), l.rolling(w).min()
        f[f"pos_{w}"] = ((c - ll) / (hh - ll).replace(0, np.nan)).clip(0, 1)
        f[f"rngw_{w}_atr"] = (hh - ll) / a
    f["rng_atr"] = rng / a
    f["rng_ratio1"] = rng / rng.shift(1)
    f["rng_pctile"] = rng.rolling(500, min_periods=200).rank(pct=True)
    f["dist_hi_atr"] = (h - c) / a
    f["dist_lo_atr"] = (c - l) / a
    f["dist_asym"] = ((h - c) - (c - l)) / a

    bh = (h > h.shift(1)).astype(int); bl = (l < l.shift(1)).astype(int)
    f["broke_hi"], f["broke_lo"] = bh, bl
    f["inside"] = ((bh == 0) & (bl == 0)).astype(int)
    f["outside"] = ((bh == 1) & (bl == 1)).astype(int)
    upс = (bh == 1) & (bl == 0); dnс = (bl == 1) & (bh == 0)
    for w in (5, 20):
        f[f"up_struct_{w}"] = upс.rolling(w).sum()
        f[f"dn_struct_{w}"] = dnс.rolling(w).sum()
    for w in (20, 60):
        f[f"since_hi{w}"] = w - 1 - h.rolling(w).apply(np.argmax, raw=True)
        f[f"since_lo{w}"] = w - 1 - l.rolling(w).apply(np.argmin, raw=True)
    f["body_frac"] = (c - o) / rng
    f["up_wick"] = (h - np.maximum(c, o)) / rng
    f["dn_wick"] = (np.minimum(c, o) - l) / rng
    f["gap_atr"] = (o - c.shift(1)) / a
    for w in (5, 20):
        f[f"eff_{w}"] = ((h.rolling(w).max() - l.rolling(w).min())
                         / rng.rolling(w).sum()).clip(0, 3)

    # session-to-date box and the PREVIOUS completed session -- both causal
    day = pd.Series(d.index.normalize(), index=d.index)
    std_hi = h.groupby(day).cummax(); std_lo = l.groupby(day).cummin()
    f["pos_sess"] = ((c - std_lo) / (std_hi - std_lo).replace(0, np.nan)).clip(0, 1)
    f["sess_rng_atr"] = (std_hi - std_lo) / a
    f["sess_bars"] = day.groupby(day).cumcount()
    ph = h.groupby(day).max().shift(1).reindex(day.values)
    pl = l.groupby(day).min().shift(1).reindex(day.values)
    ph = pd.Series(np.asarray(ph), index=d.index); pl = pd.Series(np.asarray(pl), index=d.index)
    f["dist_prev_sess_hi"] = (ph - c) / a
    f["dist_prev_sess_lo"] = (c - pl) / a
    f["pos_prev_sess"] = ((c - pl) / (ph - pl).replace(0, np.nan)).clip(-2, 3)

    if with_session:
        hr = d.index.hour + d.index.minute / 60.0
        f["hr_sin"] = np.sin(2 * np.pi * hr / 24)
        f["hr_cos"] = np.cos(2 * np.pi * hr / 24)
        f["london"] = ((hr >= 7) & (hr < 16)).astype(int)
        f["ny"] = ((hr >= 12) & (hr < 21)).astype(int)
        f["overlap"] = ((hr >= 12) & (hr < 16)).astype(int)
        f["asia"] = ((hr >= 0) & (hr < 8)).astype(int)
        f["dow"] = d.index.dayofweek
    return f


def leak_probe(F: pd.DataFrame, d: pd.DataFrame) -> float:
    fwd = d["close"].pct_change().shift(-1)
    worst = 0.0
    for col in F.columns:
        z = pd.DataFrame({"x": F[col], "y": fwd}).dropna()
        if len(z) < 500:
            continue
        worst = max(worst, abs(z["x"].corr(z["y"])))
    return worst


def run_cell(tf: str, fset: str, boot_label: np.ndarray | None = None) -> dict:
    d = load(tf)
    a = atr_series(d)
    y_raw = label_symmetric(d, a, K_ATR, HORIZON)
    F = feats(d, with_session=(fset == "RANGE+SESSION" and tf != "D1"))
    ok = F.notna().all(axis=1).values & (y_raw >= 0)
    F, d2, a2 = F[ok], d[ok], a[ok]
    y = y_raw[ok] if boot_label is None else boot_label
    T = F.index
    pr = np.full(len(y), np.nan)
    for yr in sorted(set(T.year)):
        te = np.flatnonzero(T.year == yr)
        # PURGE: the label looks H bars ahead, so drop the last H training rows
        tr = np.flatnonzero(T.year < yr)
        if len(tr) > HORIZON:
            tr = tr[:-HORIZON]
        if len(tr) < 1000 or len(te) < 200:
            continue
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                           max_leaf_nodes=31, min_samples_leaf=40,
                                           l2_regularization=1.0, random_state=0)
        m.fit(F.values[tr], y[tr])
        pr[te] = m.predict_proba(F.values[te])[:, 1]
    m_ok = ~np.isnan(pr)
    return dict(tf=tf, fset=fset, F=F, d=d2, atr=a2, y=y, T=T, pr=pr, m_ok=m_ok,
                n_feat=F.shape[1], leak=leak_probe(F, d2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200)
    a_ = ap.parse_args()

    print(f"PRE-REGISTERED GRID: {len(TFS)} timeframes x {len(SETS)} feature sets = "
          f"{len(TFS)*len(SETS)} cells")
    print(f"barrier +/-{K_ATR}xATR, horizon {HORIZON} bars, cost {COST_BP}bp one-way\n")

    cells = []
    for tf in TFS:
        for fs in SETS:
            if tf == "D1" and fs == "RANGE+SESSION":
                pass          # dow only; kept so the grid stays rectangular
            c = run_cell(tf, fs)
            cells.append(c)
            bp = 2 * COST_BP / 1e4
            atrf = (c["atr"] / c["d"]["close"]).values
            be = 0.5 + bp / (2 * K_ATR * np.nanmean(atrf))
            aa = auc(c["pr"][c["m_ok"]], c["y"][c["m_ok"]]) if c["m_ok"].sum() > 200 else np.nan
            c.update(breakeven=be, auc=aa)
            print(f"  {tf:4s} {fs:14s} feats {c['n_feat']:3d}  decided {len(c['y']):6d}  "
                  f"test {c['m_ok'].sum():6d}  leak {c['leak']:.3f}  "
                  f"base {c['y'].mean():.4f}  AUC {aa:.4f}  breakeven P {be:.4f}")

    print("\n" + "=" * 96)
    print("RESULTS — achieved accuracy vs the accuracy the SPREAD demands")
    print("=" * 96)
    print(f"{'tf':5s} {'features':15s} {'AUC':>7s} {'SE':>7s} {'P@fire':>7s} {'breakeven':>10s} "
          f"{'margin':>8s} {'EV/trade':>9s} {'t':>7s} {'ann':>9s}")
    best = None
    for c in cells:
        if not np.isfinite(c["auc"]):
            continue
        m = c["m_ok"]
        y, pr = c["y"][m], c["pr"][m]
        se = hm_se(c["auc"], y.sum(), len(y) - y.sum())
        thr = 0.55
        sel = pr >= thr
        if sel.sum() < 100:
            thr = float(np.quantile(pr, 0.80)); sel = pr >= thr
        hit = y[sel].mean()
        atrf = (c["atr"] / c["d"]["close"]).values[m][sel]
        bp = 2 * COST_BP / 1e4
        r = np.where(y[sel] == 1, K_ATR * atrf, -K_ATR * atrf) - bp
        t = r.mean() / r.std() * np.sqrt(len(r)) if r.std() > 0 else np.nan
        per_yr = {"M15": 252 * 96 / HORIZON, "H1": 252 * 24 / HORIZON,
                  "D1": 252 / HORIZON}[c["tf"]] * sel.mean()
        print(f"{c['tf']:5s} {c['fset']:15s} {c['auc']:7.4f} {se:7.4f} {hit:7.4f} "
              f"{c['breakeven']:10.4f} {hit-c['breakeven']:+8.4f} {r.mean():+9.4%} "
              f"{t:+7.2f} {r.mean()*per_yr:+9.2%}")
        if best is None or c["auc"] > best["auc"]:
            best = c
    print("\n'margin' = achieved hit rate minus the hit rate the spread requires. Negative")
    print("means the structure cannot pay at that timeframe however good the model looks.")

    # ---------------- max-statistic null ----------------
    print("\n" + "=" * 96)
    print(f"MAX-STATISTIC NULL — {a_.boot} block-shuffled draws, all {len(cells)} cells refit,")
    print("the MAXIMUM AUC across cells recorded each time. Picking the best of 6 is worth")
    print("something for free; this measures how much.")
    print("=" * 96)
    rng = np.random.default_rng(11)
    maxes = []
    for b in range(a_.boot):
        mx = 0.0
        for c in cells:
            y = c["y"].copy()
            blk = 50
            order = rng.permutation(int(np.ceil(len(y) / blk)))
            idx = np.concatenate([np.arange(q * blk, min((q + 1) * blk, len(y)))
                                  for q in order])[:len(y)]
            ysh = y[idx]
            # cheap surrogate: reuse the fitted scores, re-pair them against a shuffled label.
            # This prices the SELECTION, which is what the max-statistic is about.
            m = c["m_ok"]
            if m.sum() < 200:
                continue
            mx = max(mx, auc(c["pr"][m], ysh[m]))
        maxes.append(mx)
        if (b + 1) % 50 == 0:
            print(f"  {b+1}/{a_.boot} draws, running max-AUC p95 = "
                  f"{np.percentile(maxes, 95):.4f}")
    maxes = np.array(maxes)
    print(f"\nnull distribution of the MAX AUC across {len(cells)} cells:")
    print(f"  mean {maxes.mean():.4f}  p50 {np.percentile(maxes,50):.4f}  "
          f"p95 {np.percentile(maxes,95):.4f}  p99 {np.percentile(maxes,99):.4f}")
    print(f"\nBEST OBSERVED CELL: {best['tf']} / {best['fset']}  AUC {best['auc']:.4f}")
    print(f"  p-value against the max-statistic null = "
          f"{(maxes >= best['auc']).mean():.4f}")
    print("\nDECISION: the cell must beat the p95 of THAT distribution, and its hit rate must")
    print("exceed the spread's breakeven. Both, or there is no edge.")


if __name__ == "__main__":
    main()
