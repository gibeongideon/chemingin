#!/usr/bin/env python3
"""OCTOBER A2 — champ-meta at H1. Predict the SIGN OF THE CHAMPION'S OWN P&L.

Pre-registered in data/v5_runs/october/PREREGISTRATION.md. Design in OCTOBER.MD.

WHY THIS TARGET AND NOT A DIRECTION LABEL. MANDATORY CONTROL #8 retires any programme whose
mechanism is "rank the same label better", because a 5.9-sigma AUC gain once produced dSharpe
-0.702. This is the one target where that gap CANNOT open, in the repo's own words:

    "the one target where control #8's valuable-but-unmonetisable gap cannot open, because the
     label IS the economics... monetising it is the definition of the label."

Label: y = sign(champion position x forward return). A model that predicts it is predicting whether
the position the book is ALREADY HOLDING is about to make money, so a correct prediction is a trim
and there is no conversion step left to lose it in.

STATUS BEFORE THIS RUN. Queue #35. Run at H4 (AUC 0.593 -- "the best ever achieved here on any
label") and at D1, where it looked spectacular (+0.261, t +8.54, 14/15 years, block-p 0.0000) and
was then RETRACTED: "all 18 t>+2 legs are FX pairs leaking the next bar" via forward-stamped
clpos/upwick/lowick at corr 0.42-0.53, and the real 4-sleeve book gave -0.009 (t -0.24).
**Never run at H1, and never on clean non-FX data.** This run is GOLD-ONLY, so the FX panel that
caused the retraction is absent by construction.

*** TWO TIMEFRAME CORRECTIONS THIS SCRIPT HAD TO MAKE, both of which would silently leak. ***

1. `f_base`'s DAILY features end in `.shift(6)` -- calibrated for H4's 6 bars/day. `d1.resample("D")
   .last()` is the daily CLOSE, knowable only at end of day; ffilling it onto an intraday index and
   shifting 6 bars is the H4 correction. On H1 (24 bars/day) that leaves up to 18 HOURS of same-day
   information in `d1_slope` and `d1_z`. Corrected to shift(24) here. This repo is three-for-three
   on cross-timeframe resample lookaheads ("a bar stamped t isn't knowable till t+duration", AUC
   0.901 fake), so the shift is made explicit rather than inherited.

2. `champion_recipe`'s lookbacks are in BARS and `champion_signal` hardcodes 6 bars/day = H4
   (rule 1.3.5, the D=6 trap). To hold the CALENDAR span fixed at H1 the scale must be 4.0, not
   1.0 -- EWMAC's 16-bar fast span is 2.67 days at H4 and must stay 2.67 days at H1, i.e. 64 bars.
   A control below asserts the H1 position series actually tracks the H4 one.

ORACLE FIRST (CONTROL #5): if perfect knowledge of y does not move the book by dSharpe >= +0.20 at
t >= 2.5, the detector is not built. The D1 run measured that oracle at +2.534, t +17.16, 15/15.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from v5_xau_champion_lifts import champion_recipe          # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "october"
SPREAD_BP = 0.504
RT_BP = SPREAD_BP * 1.5
H1_PER_DAY, H4_PER_DAY = 24, 6
H1_SCALE = H1_PER_DAY / H4_PER_DAY          # 4.0 — keeps the CALENDAR lookback fixed
HORIZON = 24                                 # bars held = 1 day, the champion's own rebalance scale
N_FOLDS = 8
TARGET_VOL, VOL_HL, MAX_LEV = 0.10, 42, 8.0


def load_h1() -> pd.DataFrame:
    """XAUUSD H1 — single provenance. Gate 0 measured H1-vs-M15 agreement at 1.000 in all 12 years
    and intrabar_leak 0.0191 against a 0.15 bar, so the intrabar features are admissible here."""
    d = pd.read_csv(ROOT / "data/XAUUSD_H1_long.csv", parse_dates=["time"], index_col="time")
    return d[~d.index.duplicated(keep="last")].sort_index()


def intrabar_leak(d: pd.DataFrame) -> float:
    c = d["close"].pct_change()
    x = (d["close"] - d["low"]) / (d["high"] - d["low"]).replace(0, np.nan)
    nx = c.shift(-1)
    ok = x.notna() & nx.notna()
    return float(abs(x[ok].corr(nx[ok])))


def f_base_tf(df: pd.DataFrame, bars_per_day: int) -> pd.DataFrame:
    """§3r's 21-feature REGIME+PRICE block, with the daily shift made TIMEFRAME-AWARE."""
    import scripts.v5_repr_ceiling as rc                       # noqa: PLC0415
    c, h, l = (df[x].astype(float) for x in ("close", "high", "low"))
    ret = c.pct_change()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(halflife=14, min_periods=10).mean()
    F = {}
    for n in (10, 20, 50):
        m, sd = c.rolling(n).mean(), c.rolling(n).std()
        F[f"z{n}"] = (c - m) / sd.replace(0, np.nan)
        F[f"pctl{n}"] = c.rolling(n).rank(pct=True)
    for n in (7, 14):
        dd = c.diff()
        up = dd.clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
        dn = (-dd.clip(upper=0)).ewm(alpha=1 / n, min_periods=n).mean()
        F[f"rsi{n}"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    F["ema_dist"] = (c - c.ewm(span=50, min_periods=25).mean()) / atr14
    F["mom5"] = (c - c.shift(5)) / atr14
    F["accel"] = F["mom5"] - (c.shift(5) - c.shift(10)) / atr14
    o = df["open"].astype(float)
    F["upwick"] = (h - np.maximum(c, o)) / (h - l).replace(0, np.nan)
    F["lowick"] = (np.minimum(c, o) - l) / (h - l).replace(0, np.nan)
    F["clpos"] = (c - l) / (h - l).replace(0, np.nan)
    F["atr_z"] = (atr14 - atr14.rolling(100).mean()) / atr14.rolling(100).std().replace(0, np.nan)
    F["hurst100"] = rc._hurst(np.log(c), 100)
    F["chop14"] = (100 * np.log10(tr.rolling(14).sum() /
                   (h.rolling(14).max() - l.rolling(14).min()).replace(0, np.nan)) / np.log10(14))
    F["vol_ratio"] = atr14 / tr.ewm(halflife=50, min_periods=30).mean()
    # *** THE TIMEFRAME-AWARE SHIFT. f_base hardcodes 6; at H1 that leaks up to 18 hours. ***
    d1 = c.resample("D").last().dropna()
    F["d1_slope"] = np.sign(d1.ewm(span=20, min_periods=10).mean().diff()).reindex(
        c.index, method="ffill").shift(bars_per_day)
    F["d1_z"] = ((d1 - d1.rolling(20).mean()) / d1.rolling(20).std().replace(0, np.nan)
                 ).reindex(c.index, method="ffill").shift(bars_per_day)
    return pd.DataFrame(F).replace([np.inf, -np.inf], np.nan)


def champ_position(close: pd.Series, scale: float, bpy: float) -> pd.Series:
    ret = close.pct_change()
    vol = ret.ewm(halflife=VOL_HL, min_periods=20).std() * np.sqrt(bpy)
    fc = champion_recipe(close, scale, 1.5)
    return (fc.clip(0, 2) * (TARGET_VOL / vol)).clip(0, MAX_LEV).shift(1)


def perf(r, ppy):
    r = r.dropna()
    if len(r) < 50 or r.std() == 0:
        return dict(sharpe=np.nan, cagr=np.nan, maxdd=np.nan)
    eq = (1 + r).cumprod()
    return dict(sharpe=float(r.mean() / r.std() * np.sqrt(ppy)),
                cagr=float(eq.iloc[-1] ** (ppy / len(r)) - 1),
                maxdd=float((eq / eq.cummax() - 1).min()))


def vol_match(a, b):
    sa, sb = a.std(), b.std()
    return a * (sb / sa) if sa > 0 else a


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_h1()
    leak = intrabar_leak(d)
    bpy = len(d) / max((d.index[-1] - d.index[0]).days / 365.25, 1e-9)
    c = d["close"].astype(float)
    ret = c.pct_change()

    print("=" * 104)
    print("OCTOBER A2 — CHAMP-META AT H1 (target = sign of the champion's own P&L)")
    print("=" * 104)
    print(f"  XAUUSD H1  {d.index[0].date()} -> {d.index[-1].date()}  {len(d):,} bars "
          f"({bpy:.0f}/yr)")
    print(f"  intrabar_leak {leak:.4f} (bar 0.15) -> "
          f"{'CLEAN' if leak <= 0.15 else '*** LEAKING ***'}   GOLD-ONLY, no FX panel")
    if leak > 0.15:
        return 1

    pos = champ_position(c, H1_SCALE, bpy)

    # --- CONTROL CELL: the H1 champion must track the H4 champion ---------------------------
    h4 = d.resample("4h").agg({"open": "first", "high": "max", "low": "min",
                               "close": "last"}).dropna()
    pos_h4 = champ_position(h4["close"].astype(float), 1.0, len(h4) /
                            max((h4.index[-1] - h4.index[0]).days / 365.25, 1e-9))
    j = pd.concat([pos.rename("h1"), pos_h4.rename("h4")], axis=1).ffill().dropna()
    print(f"\n  CONTROL — H1 champion (scale {H1_SCALE}) vs H4 champion (scale 1.0): "
          f"position corr {j.h1.corr(j.h4):+.4f}, mean exposure {pos.mean():.3f} vs "
          f"{pos_h4.mean():.3f}")
    print("    (if this correlation is low the scale is wrong — rule 1.3.5, the D=6 trap)")

    # --- the label: did the position the book ALREADY HOLDS make money? ---------------------
    fwd = c.shift(-HORIZON) / c - 1.0
    pnl = pos * fwd
    y = (pnl > 0).astype(float).where(pnl.notna() & (pos > 1e-9))
    base = float(np.nanmean(y))
    print(f"\n  label y = sign(champ position x forward {HORIZON}h return)")
    print(f"  decided bars {int(y.notna().sum()):,}   base P(win) {base:.4f}   "
          f"(non-overlapping ~{int(y.notna().sum() / HORIZON):,})")

    # --- ORACLE FIRST (CONTROL #5) ----------------------------------------------------------
    base_r = (pos * ret).dropna()
    oracle_pos = pos.where(y.reindex(pos.index).fillna(1.0) > 0.5, 0.0)
    or_r = (oracle_pos * ret).dropna()
    idx = base_r.index.intersection(or_r.index)
    a, b = or_r.reindex(idx), base_r.reindex(idx)
    dm = vol_match(a, b) - b
    t_or = float(dm.mean() / (dm.std() / np.sqrt(len(dm))))
    mb, mo = perf(b, bpy), perf(a, bpy)
    print(f"\n  --- ORACLE CEILING (CONTROL #5) ---")
    print(f"  {'arm':<28} {'Sharpe':>9} {'CAGR':>9} {'maxDD':>9}")
    print(f"  {'champion (incumbent)':<28} {mb['sharpe']:>+9.3f} {100*mb['cagr']:>+8.1f}% "
          f"{100*mb['maxdd']:>8.1f}%")
    print(f"  {'ORACLE trim (perfect y)':<28} {mo['sharpe']:>+9.3f} {100*mo['cagr']:>+8.1f}% "
          f"{100*mo['maxdd']:>8.1f}%")
    print(f"  oracle dSharpe {mo['sharpe'] - mb['sharpe']:+.3f}   matched-vol t {t_or:+.2f}")
    gate = (mo["sharpe"] - mb["sharpe"] >= 0.20) and (t_or >= 2.5)
    print(f"  CONTROL #5 gate (dSharpe >= +0.20 AND t >= 2.5): "
          f"{'PASS -> build the detector' if gate else 'FAIL -> do not build'}")
    if not gate:
        print("\n  STOPPING HERE as pre-registered. The prize is not big enough to chase.")
        return 0

    # --- the honest detector ------------------------------------------------------------------
    from sklearn.ensemble import HistGradientBoostingClassifier                  # noqa: PLC0415
    X = f_base_tf(d, H1_PER_DAY)
    Xv = X.to_numpy(np.float32)
    yv = y.to_numpy(float)
    n = len(d)
    edges = np.linspace(int(n * 0.3), n, N_FOLDS + 1).astype(int)
    p = np.full(n, np.nan)
    for i in range(N_FOLDS):
        te0, te1 = edges[i], edges[i + 1]
        tr_end = te0 - HORIZON                      # purge the full label horizon
        if tr_end < 1000:
            continue
        m = slice(0, tr_end)
        ok = ~np.isnan(yv[m]) & np.isfinite(Xv[m]).all(axis=1)
        if ok.sum() < 500:
            continue
        mdl = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4,
                                             random_state=7)
        mdl.fit(Xv[m][ok], yv[m][ok])
        good = np.isfinite(Xv[te0:te1]).all(axis=1)
        pr = np.full(te1 - te0, np.nan)
        if good.any():
            pr[good] = mdl.predict_proba(Xv[te0:te1][good])[:, 1]
        p[te0:te1] = pr

    ev = np.isfinite(p) & ~np.isnan(yv)
    from sklearn.metrics import roc_auc_score                                    # noqa: PLC0415
    auc = float(roc_auc_score(yv[ev], p[ev]))
    acc = float(((p[ev] > 0.5) == (yv[ev] > 0.5)).mean())
    print(f"\n  --- HONEST DETECTOR ({N_FOLDS} purged folds, {ev.sum():,} OOS bars) ---")
    print(f"  AUC {auc:.4f}   accuracy@0.5 {acc:.4f}   base {base:.4f}   "
          f"lift vs base {acc - max(base, 1 - base):+.4f}")

    ps = pd.Series(p, index=d.index)
    print(f"\n  {'trim rule':<26} {'Sharpe':>9} {'CAGR':>9} {'maxDD':>9} {'t@matched':>10} "
          f"{'H1 dSR':>8} {'H2 dSR':>8}")
    print("  " + "-" * 84)
    print(f"  {'champion (incumbent)':<26} {mb['sharpe']:>+9.3f} {100*mb['cagr']:>+8.1f}% "
          f"{100*mb['maxdd']:>8.1f}% {'—':>10} {'—':>8} {'—':>8}")
    rows = []
    for thr in (0.40, 0.45, 0.50, 0.55):
        tp = pos.where(ps.reindex(pos.index) > thr, 0.0)
        r = (tp * ret).dropna()
        ii = b.index.intersection(r.index)
        aa, bb = r.reindex(ii), b.reindex(ii)
        dd = vol_match(aa, bb) - bb
        tt = float(dd.mean() / (dd.std() / np.sqrt(len(dd)))) if dd.std() > 0 else np.nan
        mid = len(ii) // 2
        h1d = perf(aa.iloc[:mid], bpy)["sharpe"] - perf(bb.iloc[:mid], bpy)["sharpe"]
        h2d = perf(aa.iloc[mid:], bpy)["sharpe"] - perf(bb.iloc[mid:], bpy)["sharpe"]
        m_ = perf(aa, bpy)
        rows.append(dict(thr=thr, **m_, t=tt, h1=h1d, h2=h2d))
        print(f"  {'trim below P=' + format(thr, '.2f'):<26} {m_['sharpe']:>+9.3f} "
              f"{100*m_['cagr']:>+8.1f}% {100*m_['maxdd']:>8.1f}% {tt:>+10.2f} "
              f"{h1d:>+8.3f} {h2d:>+8.3f}")
    pd.DataFrame(rows).to_csv(OUT / "a2_champ_meta_h1.csv", index=False)
    print(f"\n  GATE: matched-vol t >= +2.0 AND both halves positive.")
    win = [r for r in rows if r["t"] >= 2.0 and r["h1"] > 0 and r["h2"] > 0]
    print(f"  survivors: {len(win)} of {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
