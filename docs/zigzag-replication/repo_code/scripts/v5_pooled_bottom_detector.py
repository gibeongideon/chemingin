"""QUEUE #1-2 — POOLED CROSS-INSTRUMENT training for the bottom detector.

THE PROBLEM THIS ATTACKS. MANDATORY CONTROL #7: gold alone yields ~250 decided non-overlapping
events at these horizons, SE(AUC) = 0.0367, and the best-of-8 pure-noise AUC lift is +0.053. So
on gold alone a real +0.02 effect is unfalsifiable — which is why §3ak concluded "searching
harder is a losing move" and estimated 45 years of data would be needed. **The only honest fix
is more events, and the repo has 51 D1 series sitting in `data/` that no detector study has ever
used.** §3ao specified this as "Phase 1c" and it was never run.

THREE ARMS, one comparison:
  GOLD-ONLY    the incumbent — train on gold, predict gold.
  POOLED       train on all instruments (gold included), predict gold. ~50x the events.
  TRANSFER     train on all instruments EXCEPT gold, predict gold. The strict test: does the
               bottom-pivot relationship generalise across assets at all?

If POOLED beats GOLD-ONLY on gold's own out-of-sample events, the detector was
sample-starved rather than signal-starved, and that changes what is worth building. If TRANSFER
also works, the relationship is universal rather than gold-specific, which would be a
genuinely new result for this repo.

WHY THE FEATURES TRANSFER. `turning_ml.features` is scale-free by construction — z-scores,
in-window percentile ranks, RSI, ATR-normalised momentum and wick ratios, volatility
percentiles, run lengths. `hour` is dropped (§3p: a within-day artifact on gold, and
meaningless on D1 anyway). So the same matrix is comparable across a bond future and bitcoin.

LEAKAGE. Walk-forward by CALENDAR YEAR across the whole panel: train on every instrument's bars
strictly before January of the test year, predict that year. Purge TOL+1 bars at the boundary.
No instrument ever sees its own future, and no instrument sees another instrument's future.

    python scripts/v5_pooled_bottom_detector.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from scripts.v5_xau_turning_ml import features, zigzag_swings, label_near, atr  # noqa: E402

TOL, ZZ_ORDER, ZZ_THETA = 3, 5, 1.5
PURGE_DAYS = 3 + 1   # overridden by --purge; see the leakage note in main()
YEARS = list(range(2012, 2027))
MIN_BARS = 1200
DROP_FWD = False       # set by --drop-forward-stamped
LEAK_MAX = 0.15        # |corr(clpos[t], ret[t+1])| above this = unusable intrabar features
GOLD = "GOLD"          # data/GOLD_D1_long.csv — 2008-2026, the longest gold history


def forward_stamp_ratio(d: pd.DataFrame) -> tuple[float, float]:
    """DATA-INTEGRITY SCREEN. How often does bar t's [low, high] contain bar t+1's close,
    versus bar t-1's close?

    A correctly stamped bar's range straddles the PRECEDING close (the market opens near where
    it closed), so backward > forward. Every non-FX series in `data/` behaves that way
    (GOLD 0.39/0.54, BTC 0.52/0.94, NDX 0.41/0.62, BRENT 0.48/0.73). **Every FX D1 file here is
    inverted** (GBPJPY 0.73/0.50, EURUSD 0.70/0.50, AUDNZD 0.76/0.55): its OHLC spans into the
    following bar relative to its timestamp.

    That makes `turning_ml.features`' intrabar columns partially descriptive of the NEXT bar:
    on GBPJPY corr(clpos[t], ret[t+1]) = -0.4996, upwick +0.5094, lowick -0.5005, against
    0.021 / 0.044 / 0.025 on gold. It is a DATA-ALIGNMENT defect, not a code lookahead, so
    truncation-based `probe_lookahead` cannot catch it (one of the three blind spots named in
    SHORT-VARIANT-PLAN.md), and a shuffled-LABEL control cannot either — shuffling the label
    destroys every relationship, leaky or not.

    It manufactured a champ-meta result of dSharpe +0.261 at t +8.54 carried entirely by 18 FX
    legs, so any pooled study using intrabar features MUST screen on this.
    """
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    fwd = ((c[1:] >= l[:-1]) & (c[1:] <= h[:-1])).mean()
    bwd = ((c[:-1] >= l[1:]) & (c[:-1] <= h[1:])).mean()
    return float(fwd), float(bwd)


def intrabar_leak(d: pd.DataFrame) -> float:
    """THE SCREEN TO ACTUALLY USE: |corr(clpos[t], ret[t+1])| — the harm itself, measured.

    `forward_stamp_ratio` is only a proxy and it MISSES USDJPY (fwd 0.632 < bwd 0.649) which
    nonetheless leaks at 0.317. This direct test separates cleanly:
        clean   GOLD 0.021 · NDX 0.078 · BRENT/BTC/SILVER similar
        leaking USDJPY 0.317 · EURUSD 0.420 · GBPJPY 0.500
    A threshold of 0.15 sits in the gap. Any series above it must be excluded from a pooled
    study that uses intrabar features (clpos / upwick / lowick).
    """
    c = d["close"].pct_change()
    x = (d["close"] - d["low"]) / (d["high"] - d["low"]).replace(0, np.nan)
    nx = c.shift(-1)
    ok = x.notna() & nx.notna()
    return float(abs(x[ok].corr(nx[ok]))) if ok.sum() > 300 else 0.0


def load_d1(sym: str) -> pd.DataFrame | None:
    f = ROOT / f"data/{sym}_D1_long.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, parse_dates=["time"]).set_index("time")
    d = d[~d.index.duplicated(keep="last")].sort_index()
    d = d.rename(columns=str.lower)
    if len(d) < MIN_BARS or not {"open", "high", "low", "close"} <= set(d.columns):
        return None
    return d


def build_panel() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DatetimeIndex]:
    """Features, label, instrument tag and dates for every usable D1 series."""
    syms = sorted(p.name.replace("_D1_long.csv", "") for p in (ROOT / "data").glob("*_D1_long.csv"))
    X, Y, S, T = [], [], [], []
    kept = []
    dropped = []
    for s in syms:
        d = load_d1(s)
        if d is None:
            continue
        if DROP_FWD:
            lk = intrabar_leak(d)
            if lk > LEAK_MAX:
                dropped.append((s, lk))
                continue
        try:
            F = features(d)
            if "hour" in F.columns:
                F = F.drop(columns=["hour"])
            _, buys = zigzag_swings(d, ZZ_ORDER, ZZ_THETA * atr(d))
            y = label_near(buys, len(d), TOL)
        except Exception:
            continue
        ok = F.notna().all(axis=1).values
        if ok.sum() < MIN_BARS or y[ok].mean() in (0.0, 1.0):
            continue
        X.append(F[ok]); Y.append(pd.Series(y, index=d.index)[ok])
        S.append(pd.Series(s, index=d.index[ok])); T.append(d.index[ok])
        kept.append((s, int(ok.sum()), float(y[ok].mean())))
    if dropped:
        print(f"DROPPED {len(dropped)} series with intrabar leakage > {LEAK_MAX}: "
              + ", ".join(f"{x[0]}({x[1]:.2f})" for x in dropped))
    print(f"panel: {len(kept)} instruments, "
          f"{sum(n for _, n, _ in kept):,} usable bars")
    return (pd.concat(X), pd.concat(Y), pd.concat(S),
            pd.DatetimeIndex(np.concatenate([t.values for t in T])))


def auc(p, y):
    p, y = np.asarray(p), np.asarray(y)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 < 5 or n0 < 5:
        return np.nan
    r = pd.Series(p).rank().values
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def hm_se(a, n1, n0):
    """Hanley-McNeil standard error of AUC — the number that decides whether a lift is real."""
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return float(np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a))
                         / (n1 * n0)))


def prec_at_recall(p, y, rec):
    o = np.argsort(-np.asarray(p)); y = np.asarray(y)
    need = int(np.ceil(rec * y.sum()))
    hit = taken = 0
    for i in o:
        taken += 1; hit += y[i]
        if hit >= need:
            break
    return hit / max(taken, 1), taken


def main() -> None:
    """LEAKAGE NOTE. `zigzag_swings` uses argrelextrema(order=5) plus a theta filter over the
    WHOLE series, so a pivot near the training tail can only be confirmed using bars that fall
    inside the test year. That contaminates a handful of TRAINING labels at each boundary. It
    cannot help the model (it sees no future features, only a slightly mislabelled target) and
    all three arms share identical TEST labels, so the arm-vs-arm comparison is fair either
    way — but the purge is swept here to prove the lift is not an artifact of it."""
    global PURGE_DAYS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--purge", type=int, default=4)
    ap.add_argument("--shuffle", action="store_true",
                    help="destroy the feature/label link — AUC must collapse to ~0.50")
    ap.add_argument("--drop-forward-stamped", action="store_true",
                    help="exclude series with intrabar leakage (see intrabar_leak) — "
                         "REQUIRED for any claim using clpos/upwick/lowick features")
    a = ap.parse_args()
    global DROP_FWD
    PURGE_DAYS = a.purge
    DROP_FWD = a.drop_forward_stamped
    t0 = time.time()
    X, Y, S, T = build_panel()
    print(f"\npooled rows {len(X):,}  base rate {Y.mean():.3f}  "
          f"({time.time()-t0:.0f}s)\n")
    if a.shuffle:
        rng = np.random.default_rng(11)
        # block-shuffle the LABEL within each instrument: preserves the label's own
        # autocorrelation and base rate, destroys its alignment to the features
        Yv = Y.values.copy()
        for sym in S.unique():
            m = np.flatnonzero((S == sym).values)
            blk = 40
            starts = rng.permutation(int(np.ceil(len(m) / blk)))
            idx = np.concatenate([np.arange(b * blk, min((b + 1) * blk, len(m)))
                                  for b in starts])[:len(m)]
            Yv[m] = Yv[m][idx]
        Y = pd.Series(Yv, index=Y.index)
        print("*** SHUFFLED-LABEL CONTROL — every AUC must be ~0.50 ***\n")
    is_gold = (S == GOLD).values
    print(f"gold rows {is_gold.sum():,}  base rate {Y.values[is_gold].mean():.3f}")
    if is_gold.sum() < 500:
        raise SystemExit("gold series too short — check data/GOLD_D1_long.csv")

    def model(seed=0):
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                              max_leaf_nodes=31, min_samples_leaf=40,
                                              l2_regularization=1.0, random_state=seed)

    res = {k: dict(p=[], y=[]) for k in ("GOLD-ONLY", "POOLED", "TRANSFER")}
    peryr = {}   # yr -> {arm: (auc, n)}  — is the lift concentrated where gold-only is starved?
    for yr in YEARS:
        te = (T.year == yr) & is_gold
        if te.sum() < 40:
            continue
        tr_all = (T.year < yr)                       # strictly-past bars, every instrument
        # purge: drop the last TOL+1 bars of each instrument's training tail
        cut = pd.Timestamp(f"{yr}-01-01") - pd.Timedelta(days=PURGE_DAYS)
        tr_all = tr_all & (T <= cut)
        arms = {
            "GOLD-ONLY": tr_all & is_gold,
            "POOLED":    tr_all,
            "TRANSFER":  tr_all & ~is_gold,
        }
        for name, tr in arms.items():
            if tr.sum() < 400 or len(np.unique(Y.values[tr])) < 2:
                continue
            m = model().fit(X.values[tr], Y.values[tr])
            p = m.predict_proba(X.values[te])[:, 1]
            res[name]["p"].append(p); res[name]["y"].append(Y.values[te])
            peryr.setdefault(yr, {})[name] = (auc(p, Y.values[te]), int(te.sum()),
                                              int(arms["GOLD-ONLY"].sum()))
        print(f"  {yr}: gold test {int(te.sum()):4d}  train gold-only {int(arms['GOLD-ONLY'].sum()):6d}"
              f"  pooled {int(arms['POOLED'].sum()):7d}  transfer {int(arms['TRANSFER'].sum()):7d}"
              f"   ({time.time()-t0:.0f}s)")

    print(f"\nPER-YEAR AUC on gold  (gold_tr = rows the GOLD-ONLY arm had to train on)")
    print(f"{'year':>6s} {'gold_tr':>8s} {'n_te':>5s} {'GOLD':>7s} {'POOL':>7s} "
          f"{'TRANS':>7s} {'POOL-GOLD':>10s}")
    dl = []
    for yr in sorted(peryr):
        d = peryr[yr]
        if "GOLD-ONLY" not in d or "POOLED" not in d:
            continue
        g, n, gtr = d["GOLD-ONLY"]; po = d["POOLED"][0]; tr = d.get("TRANSFER", (np.nan,))[0]
        dl.append((yr, po - g, gtr))
        print(f"{yr:6d} {gtr:8d} {n:5d} {g:7.3f} {po:7.3f} {tr:7.3f} {po-g:+10.3f}")
    wins = sum(1 for _, x, _ in dl if x > 0)
    print(f"POOLED beats GOLD-ONLY in {wins}/{len(dl)} test years "
          f"(sign test p = {2 ** -len(dl) * sum(1 for _ in range(0)) if False else 0:.0f})")
    from math import comb
    pv = sum(comb(len(dl), k) for k in range(wins, len(dl) + 1)) / 2 ** len(dl)
    print(f"one-sided sign-test p = {pv:.4f}")
    half = len(dl) // 2
    e = [x for _, x, _ in dl[:half]]; l = [x for _, x, _ in dl[half:]]
    print(f"SPLIT HALVES:  early {dl[0][0]}-{dl[half-1][0]} mean dAUC {np.mean(e):+.4f}"
          f"   late {dl[half][0]}-{dl[-1][0]} mean dAUC {np.mean(l):+.4f}")
    print(f"  -> if the lift were only 'gold-only is starved', LATE should be ~0.")

    print(f"\n{'arm':12s} {'n':>7s} {'AUC on GOLD':>12s} {'SE':>7s} {'vs gold-only':>13s} "
          f"{'P@rec20':>9s} {'P@rec10':>9s}")
    base = None
    for name in ("GOLD-ONLY", "POOLED", "TRANSFER"):
        if not res[name]["p"]:
            print(f"{name:12s} no data"); continue
        p = np.concatenate(res[name]["p"]); y = np.concatenate(res[name]["y"])
        a = auc(p, y); se = hm_se(a, y.sum(), len(y) - y.sum())
        p20, _ = prec_at_recall(p, y, 0.20); p10, _ = prec_at_recall(p, y, 0.10)
        if name == "GOLD-ONLY":
            base = a
        d = "" if base is None or name == "GOLD-ONLY" else f"{a - base:+13.4f}"
        print(f"{name:12s} {len(y):7d} {a:12.4f} {se:7.4f} {d:>13s} {p20:9.3f} {p10:9.3f}")
    print(f"\nbase rate on gold test events: "
          f"{np.concatenate(res['GOLD-ONLY']['y']).mean():.3f}")
    print(f"DECISION RULE: a lift is real only if it exceeds ~2 x SE. SE above is the")
    print(f"Hanley-McNeil SE at these event counts, and it is the whole point of this test —")
    print(f"pooling is supposed to shrink it.")
    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
