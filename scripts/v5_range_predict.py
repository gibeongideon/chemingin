"""RANGE-BOX prediction on XAUUSD: which extreme does tomorrow touch FIRST?

WHAT THE DESCRIPTIVE STUDY (v5_range_study.py) ESTABLISHED, and why this is the target.
  * Daily RANGE is strongly predictable: lag-1 autocorr 0.500, ma5 -> R^2 0.308.
  * Daily DIRECTION is not: corr(today's return, sign of tomorrow's) = -0.037.
  * The user's literal hypothesis is refuted -- the daily high and low share an hour in only
    0.5% of sessions and sit a MEDIAN 12.5 HOURS apart. The day has directional structure.
  * The close-position U-shape is noise: corr(|pos-0.5|, fwd) = +0.0245, t +1.34, and the
    per-year sign flips (2017 -0.148, 2021 -0.057, 2026 +0.207). One good year, not an effect.
  * Mon-low / Fri-high is pure DRIFT: on UP weeks Mon holds the low 0.508 / Fri the high
    0.521; on DOWN weeks that inverts to 0.088 / 0.042. Tautological, no calendar structure.

So direction is the wrong target and a range forecast alone is direction-free. The target that
uses the min/max framing AND is tradeable is the PATH question:

    does tomorrow touch TODAY'S HIGH before TODAY'S LOW?

It is a real trade -- long at the close with TP = today's high and SL = today's low (or the
mirror) -- and, unlike the zigzag's fixed 0.6%/0.5% bracket, its barriers scale with the range,
which is the defect the study just quantified: at 2026's 2.73% mean daily range an SL of 0.50%
is **0.18 of a single day's travel**, and 96% of sessions have a range exceeding TP+SL combined.

THE NULL THAT MATTERS, and the reason naive AUC here is meaningless. If the close sits near
today's high it is simply CLOSER to the high, so it touches it first more often -- pure
geometry, no skill. The honest baseline is therefore the DISTANCE-IMPLIED probability, and the
model must beat THAT, not 0.5.

RECENCY, as requested: an exponential sample weight with a 250-session half-life (about one
year), tested as its own arm against uniform weighting. Two arms, not a sweep, so there is no
best-of-K noise floor to clear.

LABEL RESOLUTION needs intraday bars: daily bars cannot say which extreme came first. H1 is
used (5.07 years) so the label is resolved, not guessed.

    python scripts/v5_range_predict.py
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


def load(tf: str) -> pd.DataFrame:
    d = pd.read_csv(P / f"XAUUSD_{tf}.csv", parse_dates=["time"]).set_index("time")
    return d[~d.index.duplicated(keep="last")].sort_index()


def build_label(d1: pd.DataFrame, h1: pd.DataFrame) -> pd.DataFrame:
    """For each session, walk the NEXT session's H1 bars and record which of today's extremes
    is touched first. Unresolved sessions (neither touched) are dropped, not guessed."""
    h1 = h1.copy()
    h1["day"] = h1.index.normalize()
    days = sorted(set(h1["day"]))
    pos = {dd: i for i, dd in enumerate(days)}
    by_day = {dd: g for dd, g in h1.groupby("day")}
    out = []
    for t, row in d1.iterrows():
        dd = t.normalize()
        i = pos.get(dd)
        if i is None or i + 1 >= len(days):
            continue
        nxt = by_day[days[i + 1]]
        hi, lo = float(row["high"]), float(row["low"])
        first = 0
        for _, b in nxt.iterrows():
            up = b["high"] >= hi
            dn = b["low"] <= lo
            if up and dn:
                first = 0          # same H1 bar hit both: genuinely ambiguous -> drop
                break
            if up:
                first = 1; break
            if dn:
                first = -1; break
        if first != 0:
            out.append((t, 1 if first == 1 else 0))
    lab = pd.DataFrame(out, columns=["time", "y"]).set_index("time")
    return lab


def features(d1: pd.DataFrame, w1: pd.DataFrame) -> pd.DataFrame:
    """EVERY column is built from extremes, levels or ranges -- never a raw return series.
    That is the point of the exercise: a different data representation, not new models."""
    c, h, l, o = d1["close"], d1["high"], d1["low"], d1["open"]
    rng = (h - l).replace(0, np.nan)
    atr = rng.rolling(20).mean()
    f = pd.DataFrame(index=d1.index)

    # --- where price sits inside boxes of several sizes
    f["pos_day"] = ((c - l) / rng).clip(0, 1)
    for w in (5, 10, 20, 60):
        hh, ll = h.rolling(w).max(), l.rolling(w).min()
        f[f"pos_{w}"] = ((c - ll) / (hh - ll).replace(0, np.nan)).clip(0, 1)
    # WEEKLY BOX -- CAUSAL. The obvious construction is a LOOKAHEAD and it inflated this
    # study's first headline: `w1["high"].reindex(d1.index, method="ffill")` assigns the weekly
    # bar stamped at the week's START to every day Mon-Fri, and that bar's high/low span the
    # WHOLE week. On Tuesday it already knows Friday's high. With it, a direct direction model
    # scored Sharpe 5.52 / paired t +9.19 against buy-and-hold -- impossible on daily gold, and
    # the tell that caught it. Two honest replacements:
    #   week-to-date, expanding within the current week (uses only bars up to t), and
    #   the PREVIOUS COMPLETED week's extremes, which are fully known.
    wk = pd.Series(d1.index.to_period("W"), index=d1.index)
    wtd_hi = h.groupby(wk).cummax()
    wtd_lo = l.groupby(wk).cummin()
    f["pos_wtd"] = ((c - wtd_lo) / (wtd_hi - wtd_lo).replace(0, np.nan)).clip(0, 1)
    f["wtd_rng_atr"] = (wtd_hi - wtd_lo) / atr
    f["wtd_bars"] = wk.groupby(wk).cumcount()
    prev_hi = h.groupby(wk).max().shift(1).reindex(wk.values).values
    prev_lo = l.groupby(wk).min().shift(1).reindex(wk.values).values
    prev_hi = pd.Series(prev_hi, index=d1.index)
    prev_lo = pd.Series(prev_lo, index=d1.index)
    f["pos_prev_week"] = ((c - prev_lo) / (prev_hi - prev_lo).replace(0, np.nan)).clip(-2, 3)
    wk_hi, wk_lo = prev_hi, prev_lo          # everything below now uses the COMPLETED week

    # --- how much range has been spent, and is it expanding
    f["rng_atr"] = rng / atr
    f["rng_ratio1"] = rng / rng.shift(1)
    f["rng_ma5_atr"] = rng.rolling(5).mean() / atr
    f["rng_pctile"] = rng.rolling(250, min_periods=100).rank(pct=True)

    # --- distance to the levels the trade actually brackets, in ATR
    f["dist_hi_atr"] = (h - c) / atr
    f["dist_lo_atr"] = (c - l) / atr
    f["dist_asym"] = ((h - c) - (c - l)) / atr
    f["dist_wk_hi_atr"] = (wk_hi - c) / atr
    f["dist_wk_lo_atr"] = (c - wk_lo) / atr

    # --- structure from extremes: breaks, inside/outside bars, streaks
    bh = (h > h.shift(1)).astype(int)
    bl = (l < l.shift(1)).astype(int)
    f["broke_hi"] = bh
    f["broke_lo"] = bl
    f["inside"] = ((bh == 0) & (bl == 0)).astype(int)
    f["outside"] = ((bh == 1) & (bl == 1)).astype(int)
    up_str = (bh == 1) & (bl == 0)
    dn_str = (bl == 1) & (bh == 0)
    f["up_struct_5"] = up_str.rolling(5).sum()
    f["dn_struct_5"] = dn_str.rolling(5).sum()
    f["up_struct_20"] = up_str.rolling(20).sum()
    f["dn_struct_20"] = dn_str.rolling(20).sum()
    # days since the 20/60-day extreme -- a clock measured on extremes, not a moving average
    for w in (20, 60):
        f[f"since_hi{w}"] = (w - 1 - h.rolling(w).apply(np.argmax, raw=True))
        f[f"since_lo{w}"] = (w - 1 - l.rolling(w).apply(np.argmin, raw=True))

    # --- bar shape
    f["body_frac"] = (c - o) / rng
    f["up_wick"] = (h - np.maximum(c, o)) / rng
    f["dn_wick"] = (np.minimum(c, o) - l) / rng
    f["gap_atr"] = (o - c.shift(1)) / atr

    # --- week travel efficiency: net span / summed daily span, the trend-vs-chop measure
    # travel efficiency on the last 5 COMPLETED sessions, not the in-progress week
    f["eff_5"] = ((h.rolling(5).max() - l.rolling(5).min())
                  / rng.rolling(5).sum()).clip(0, 3)
    f["eff_20"] = ((h.rolling(20).max() - l.rolling(20).min())
                   / rng.rolling(20).sum()).clip(0, 3)
    f["dow"] = d1.index.dayofweek
    return f


def dist_baseline(d1: pd.DataFrame) -> pd.Series:
    """THE NULL: probability implied by relative distance alone, no skill.

    A close sitting near today's high is nearer the high and touches it first more often as a
    matter of geometry. Weighting each side by the inverse of its distance is the simplest
    statement of that, and it is what any honest model must beat."""
    c, h, l = d1["close"], d1["high"], d1["low"]
    du = (h - c).clip(lower=1e-9)
    dd = (c - l).clip(lower=1e-9)
    return dd / (du + dd)          # near the high -> du small -> value near 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--halflife", type=float, default=250.0)
    ap.add_argument("--spread-bp", type=float, default=1.0,
                    help="one-way cost; XAUUSD H1 median spread is ~9pt = $0.09 ~ 0.2bp, "
                         "floored to 1bp because CSV spreads understate 20-50x")
    a = ap.parse_args()

    d1, h1, w1 = load("D1"), load("H1"), load("W1")
    lab = build_label(d1, h1)
    F = features(d1, w1)
    X = F.reindex(lab.index)
    ok = X.notna().all(axis=1).values
    X, y = X[ok], lab["y"].values[ok]
    T = X.index
    base = dist_baseline(d1).reindex(T)
    d1a = d1.reindex(T)
    print(f"resolved sessions: {len(y):,}  ({T[0].date()} -> {T[-1].date()})")
    print(f"base rate P(high touched first) = {y.mean():.4f}")
    print(f"features: {X.shape[1]}, all built from extremes/levels/ranges")

    # ---------------- the null first ----------------
    print("\n" + "=" * 78)
    print("STEP 1 — THE DISTANCE NULL. Any model must beat this, not 0.5.")
    print("=" * 78)
    ab = auc(base.values, y)
    print(f"distance-implied AUC = {ab:.4f}   "
          f"SE {hm_se(ab, y.sum(), len(y)-y.sum()):.4f}")
    print(f"corr(distance-implied prob, label) = {np.corrcoef(base.values, y)[0,1]:+.4f}")
    print("-> geometry alone already predicts the label. This is the bar.")

    # ---------------- walk-forward ----------------
    print("\n" + "=" * 78)
    print("STEP 2 — WALK-FORWARD by calendar year. UNIFORM vs RECENCY-WEIGHTED.")
    print("=" * 78)
    years = sorted(set(T.year))
    arms = ("UNIFORM", f"RECENCY-hl{int(a.halflife)}")
    pr = {k: np.full(len(y), np.nan) for k in arms}
    for yr in years:
        te = np.flatnonzero(T.year == yr)
        tr = np.flatnonzero(T.year < yr)
        if len(tr) < 400 or len(te) < 40:
            continue
        for k in arms:
            if k == "UNIFORM":
                sw = None
            else:
                age = (T[tr[-1]] - T[tr]).days.values.astype(float)
                sw = 0.5 ** (age / a.halflife)
            m = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                min_samples_leaf=40, l2_regularization=1.0, random_state=0)
            m.fit(X.values[tr], y[tr], sample_weight=sw)
            pr[k][te] = m.predict_proba(X.values[te])[:, 1]
        print(f"  {yr}: train {len(tr):5d}  test {len(te):4d}")

    m_ok = ~np.isnan(pr["UNIFORM"])
    print(f"\n{'arm':18s} {'AUC':>8s} {'SE':>7s} {'vs distance null':>17s} {'n':>6s}")
    ab_sub = auc(base.values[m_ok], y[m_ok])
    print(f"{'DISTANCE NULL':18s} {ab_sub:8.4f} "
          f"{hm_se(ab_sub, y[m_ok].sum(), m_ok.sum()-y[m_ok].sum()):7.4f} "
          f"{'—':>17s} {m_ok.sum():6d}")
    for k in arms:
        aa = auc(pr[k][m_ok], y[m_ok])
        print(f"{k:18s} {aa:8.4f} "
              f"{hm_se(aa, y[m_ok].sum(), m_ok.sum()-y[m_ok].sum()):7.4f} "
              f"{aa-ab_sub:+17.4f} {m_ok.sum():6d}")

    # ---------------- economics: control #8 ----------------
    print("\n" + "=" * 78)
    print("STEP 3 — ECONOMICS (control #8). EV per trade, not AUC.")
    print("=" * 78)
    print("Trade: at today's close go LONG if P(high first) is high, SHORT if low; TP and SL")
    print("are today's two extremes, so the bracket SCALES with the range instead of being a")
    print("fixed 0.5%/0.6% that 2026's 2.73% daily range swallows whole.")
    up = (d1a["high"] - d1a["close"]) / d1a["close"]      # gain if high hit first (long)
    dn = (d1a["close"] - d1a["low"]) / d1a["close"]       # loss if low hit first (long)
    cost = 2 * a.spread_bp / 1e4
    print(f"\nmean TP distance {up.mean():.4%}   mean SL distance {dn.mean():.4%}   "
          f"round-trip cost {cost:.4%}")
    print(f"cost as a share of the TP distance: {cost/up.mean():.3%}  "
          f"(the zigzag's 0.60% TP faced ~{cost/0.006:.1%})")
    print(f"\n{'arm':18s} {'thr':>5s} {'trades':>7s} {'hit%':>6s} {'EV/trade':>9s} "
          f"{'net of cost':>12s} {'ann. if 1/day':>14s} {'t':>7s}")
    for k in list(arms) + ["DISTANCE NULL"]:
        p = base.values if k == "DISTANCE NULL" else pr[k]
        for thr in (0.55, 0.60, 0.65):
            sel = m_ok & ~np.isnan(p) & (p >= thr)
            if sel.sum() < 30:
                continue
            r = np.where(y[sel] == 1, up.values[sel], -dn.values[sel]) - cost
            t = r.mean() / r.std() * np.sqrt(len(r)) if r.std() > 0 else np.nan
            print(f"{k:18s} {thr:5.2f} {sel.sum():7d} {y[sel].mean()*100:6.1f} "
                  f"{r.mean()+cost:+9.4%} {r.mean():+12.4%} "
                  f"{r.mean()*252:+14.2%} {t:+7.2f}")

    # ---------------- shuffled-label control ----------------
    print("\n" + "=" * 78)
    print("STEP 4 — SHUFFLED-LABEL CONTROL (must collapse to the null, not to 0.50:")
    print("the features legitimately encode distance, so a leak-free pipeline still scores")
    print("above 0.50 on a shuffled label only if the shuffle preserves that geometry — it")
    print("does not, so ~0.50 is the expectation here)")
    print("=" * 78)
    rng = np.random.default_rng(7)
    ysh = y.copy()
    blk = 20
    order = rng.permutation(int(np.ceil(len(ysh) / blk)))
    idx = np.concatenate([np.arange(b*blk, min((b+1)*blk, len(ysh))) for b in order])[:len(ysh)]
    ysh = ysh[idx]
    psh = np.full(len(y), np.nan)
    for yr in years:
        te = np.flatnonzero(T.year == yr)
        tr = np.flatnonzero(T.year < yr)
        if len(tr) < 400 or len(te) < 40:
            continue
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=1.0, random_state=0).fit(X.values[tr], ysh[tr])
        psh[te] = m.predict_proba(X.values[te])[:, 1]
    ms = ~np.isnan(psh)
    print(f"  shuffled AUC = {auc(psh[ms], ysh[ms]):.4f}")

    print("\n" + "=" * 78)
    print("DECISION RULE: the model is worth something only if it beats the DISTANCE NULL by")
    print("more than 2 x SE **and** its EV/trade net of cost is positive with t >= 2.")
    print("Beating 0.50 proves nothing here — geometry already does that.")
    print("=" * 78)


if __name__ == "__main__":
    main()
