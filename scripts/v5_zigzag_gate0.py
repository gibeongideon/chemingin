"""GATE 0 for the ZigZag ATR-bracket round (queue #29) — data hygiene, BLOCKING.

Exits non-zero on any failure and writes data/v5_runs/zigzag_atr/gate0.json, which every later
script in this round asserts against. Nothing else runs if this fails.

WHAT IT CHECKS, AND WHY EACH ONE EXISTS (every item is a mistake this repo actually made):

  1. FEED PROVENANCE (SS3bb). 25 of 38 timeframe pairs in data/ are not one series.
     XAUUSD H4 carries FTMO quotes from 2023-06-28 and M30 from 2026; H1 and M15 are believed
     clean. "Believed" is not "verified", and a multi-broker dataset is a leak no timestamp
     control can see, so the agreement is measured here PER YEAR at a RELATIVE tolerance.
     A flat dollar tolerance silently passes cheap assets (SS3bb: $0.10 is 1000 pips on EURUSD).

  2. FORWARD-STAMP SCREEN (SS3av). 18 FX D1 files leak ret[t+1] through clpos/upwick/lowick at
     corr 0.42-0.53, which manufactured a champ-meta result of +0.261 at t +8.54. The truncation
     probe provably CANNOT see this -- it tests WHEN a value was knowable, not WHERE it came
     from. intrabar_leak() measures the harm directly.

  3. LOOKAHEAD PROBE on the feature builder (right-truncation invariance).

  4. PURGE SWEEP. v5_zigzag_signal.py:145 hard-codes purge = TOL + 1 = 4. But label_near(tol=3)
     tags i+/-3 and argrelextrema(order=5) needs 5 bars AFTER i to confirm a pivot, so a label at
     bar j depends on bars out to j+8. The live purge is HALF the label's dependency reach.
     v5_pooled_bottom_detector.py:180-185 documents the contamination and sweeps --purge to prove
     it is harmless; the LIVE path never does. Measured here once.

  5. STALENESS. Clean H1/M15 end 2026-06-15; the only fresher files are the spliced ones.
     Recorded, not fixed -- refresh_xau_h4.py would DEEPEN the splice.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v5_pooled_bottom_detector import intrabar_leak            # noqa: E402
from v5_xau_turning_ml import features, zigzag_swings, label_near, atr  # noqa: E402
from src.evaluation.lookahead_probe import probe_lookahead     # noqa: E402

OUT = ROOT / "data/v5_runs/zigzag_atr"
REL_TOL = 2e-5          # 0.2bp, the v5_feed_audit convention -- relative, never a flat dollar
PASS_FRAC = 0.95        # per-year agreement fraction required
LEAK_MAX = 0.15         # intrabar_leak threshold (gold 0.021, leaking FX 0.32-0.50)
TOL, ZZ_ORDER, ZZ_THETA = 3, 5, 1.5     # the deployed label spec


def load(tf: str) -> pd.DataFrame:
    f = ROOT / f"data/XAUUSD_{tf}_long.csv"
    d = pd.read_csv(f, parse_dates=["time"], index_col="time").sort_index()
    return d[~d.index.duplicated(keep="last")]


def agreement(coarse: pd.DataFrame, fine: pd.DataFrame, minutes: int) -> dict:
    """Does the coarse bar's close equal the fine bar ending at the same instant, per year?

    A coarse bar stamped t spans [t, t+minutes); the fine bar that ENDS it is the last one
    starting before t+minutes. If the two frames are one series those closes are identical.
    """
    ci = coarse.index + pd.Timedelta(minutes=minutes)
    pos = np.searchsorted(fine.index.values, ci.values, side="right") - 1
    ok = pos >= 0
    cc = coarse["close"].values[ok]
    fc = fine["close"].values[pos[ok]]
    # only compare where the fine bar really is the terminal one of that coarse bar
    ft = fine.index.values[pos[ok]]
    inside = (ft >= coarse.index.values[ok]) & (ft < ci.values[ok])
    cc, fc = cc[inside], fc[inside]
    yrs = pd.DatetimeIndex(coarse.index[ok][inside]).year
    rel = np.abs(cc - fc) / np.maximum(np.abs(cc), 1e-12)
    per = {}
    for y in sorted(set(yrs)):
        m = yrs == y
        per[int(y)] = dict(n=int(m.sum()), frac=float((rel[m] <= REL_TOL).mean()),
                           median_abs=float(np.median(np.abs(cc[m] - fc[m]))))
    worst = min((v["frac"] for v in per.values()), default=0.0)
    return dict(per_year=per, worst_year_frac=float(worst), n=int(len(cc)))


def purge_sweep(h1: pd.DataFrame) -> dict:
    """Does the live purge of 4 differ from the label's true dependency reach of TOL+order=8?

    Train on a head slice, score a disjoint tail, varying only the gap between them. If the
    AUC falls as the gap widens, the shipped value is borrowing from the test side.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    F = features(h1).drop(columns=["hour"])
    theta = ZZ_THETA * atr(h1)
    _, buys = zigzag_swings(h1, ZZ_ORDER, theta)
    y = label_near(buys, len(h1), TOL)
    ok = F.notna().all(axis=1).values
    F, y = F[ok], y[ok]
    n = len(F)
    cut = int(n * 0.80)
    out = {}
    for purge in (4, 8, 12, 24):
        hi = cut - purge
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                             max_leaf_nodes=31, random_state=0)
        clf.fit(F.values[:hi], y[:hi])
        p = clf.predict_proba(F.values[cut:])[:, 1]
        yt = y[cut:]
        r = pd.Series(p).rank().values
        n1, n0 = yt.sum(), len(yt) - yt.sum()
        a = float((r[yt == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        out[purge] = dict(auc=round(a, 4), n_train=int(hi), n_test=int(len(yt)))
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rep: dict = dict(utc=datetime.now(timezone.utc).isoformat(), checks={}, failures=[])

    print("=" * 78)
    print("GATE 0 -- ZigZag ATR-bracket round (queue #29)")
    print("=" * 78)

    h1, m15 = load("H1"), load("M15")
    rep["frames"] = {tf: dict(rows=int(len(d)), first=str(d.index[0]), last=str(d.index[-1]))
                     for tf, d in (("H1", h1), ("M15", m15))}
    for tf, d in (("H1", h1), ("M15", m15)):
        print(f"  {tf:4s} {len(d):>7,} rows  {d.index[0].date()} .. {d.index[-1].date()}")

    # ---- 1. feed provenance: is H1 the same series as M15?
    print("\n[1] FEED PROVENANCE -- H1 close vs the M15 bar that ends it, per year")
    ag = agreement(h1, m15, 60)
    rep["checks"]["feed_h1_vs_m15"] = ag
    bad = [y for y, v in ag["per_year"].items() if v["frac"] < PASS_FRAC]
    for y, v in sorted(ag["per_year"].items()):
        flag = "  <-- FAIL" if v["frac"] < PASS_FRAC else ""
        print(f"      {y}  n {v['n']:>6,}  agree {v['frac']:6.3f}  median |d| ${v['median_abs']:.4f}{flag}")
    if bad:
        rep["failures"].append(f"H1/M15 disagree in {bad} -- these are not one series")
        print(f"      *** FAIL: years {bad} below {PASS_FRAC} ***")
    else:
        print(f"      PASS -- one series in all {len(ag['per_year'])} years")

    # ---- 2. forward-stamp screen
    print("\n[2] FORWARD-STAMP SCREEN -- |corr(clpos[t], ret[t+1])|, threshold 0.15")
    leaks = {tf: round(intrabar_leak(d), 4) for tf, d in (("H1", h1), ("M15", m15))}
    rep["checks"]["intrabar_leak"] = leaks
    for tf, v in leaks.items():
        flag = "  <-- FAIL" if v > LEAK_MAX else "  ok"
        print(f"      XAUUSD {tf:4s} {v:.4f}{flag}")
        if v > LEAK_MAX:
            rep["failures"].append(f"XAUUSD {tf} intrabar_leak {v} > {LEAK_MAX}")

    # ---- 3. lookahead probe on the feature builder
    print("\n[3] LOOKAHEAD PROBE -- right-truncation invariance of features()")
    v = probe_lookahead(lambda d: features(d).drop(columns=["hour"]),
                        n=3500, warmup_buffer=2200)
    rep["checks"]["lookahead_probe"] = dict(ok=bool(v.ok), detail=str(v))
    print(f"      {'PASS' if v.ok else 'FAIL'} -- {v}")
    if not v.ok:
        rep["failures"].append("features() fails the lookahead probe")

    # ---- 4. purge sweep
    print("\n[4] PURGE SWEEP -- live hard-codes 4; the label reaches TOL+order = 8")
    ps = purge_sweep(h1)
    rep["checks"]["purge_sweep"] = ps
    for p, v in ps.items():
        print(f"      purge {p:>3} bars  AUC {v['auc']:.4f}")
    drop = ps[4]["auc"] - ps[8]["auc"]
    rep["checks"]["purge_4_minus_8"] = round(drop, 4)
    print(f"      delta(4 -> 8) = {drop:+.4f}  "
          f"{'-- shipped purge is borrowing from the test side' if drop > 0.01 else '-- immaterial'}")

    # ---- 5. staleness
    stale_days = (pd.Timestamp.utcnow().tz_localize(None) - h1.index[-1]).days
    rep["checks"]["staleness_days"] = int(stale_days)
    print(f"\n[5] STALENESS -- clean frames end {h1.index[-1].date()}, {stale_days} days ago")
    print("      backtest window ends there BY CHOICE. refresh_xau_h4.py would deepen the splice.")

    # ---- env fingerprint
    import sklearn
    env = dict(python=sys.version.split()[0], executable=sys.executable,
               sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__)
    for lib in ("xgboost", "lightgbm", "catboost"):
        try:
            env[lib] = __import__(lib).__version__
        except ImportError:
            env[lib] = None
    rep["env"] = env
    print(f"\n[env] {env['executable']}")
    print(f"      sklearn {env['sklearn']} · xgboost {env['xgboost']} · "
          f"lightgbm {env['lightgbm']} · catboost {env['catboost']}")

    rep["pass"] = not rep["failures"]
    (OUT / "gate0.json").write_text(json.dumps(rep, indent=2))
    print("\n" + "=" * 78)
    if rep["failures"]:
        print("GATE 0 FAILED -- the round stops here")
        for f in rep["failures"]:
            print(f"  * {f}")
        return 1
    print("GATE 0 PASSED -- wrote data/v5_runs/zigzag_atr/gate0.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
