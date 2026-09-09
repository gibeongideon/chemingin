"""M15 INTRABAR-PATH down-signal as a TRIM overlay on the long champion.

WHY A TRIM AND NOT A SHORT. §3an stopped the conditional-short thesis (no causal regime switch
produced a positive short Sharpe). §3am measured the oracle split: trimming the champion to flat
in known down-windows is worth dSharpe **+2.869** (b=1.0) rising to +4.044 at b=1.5, while going
net short adds only the remaining ~29%. §3ao then found the M15 intrabar path is the one input
that beats a properly-constructed null (dI +0.0142, above the alignment null's p99, 7/9 years,
precision 0.600 at recall 0.20 vs a 0.427 base rate). So the right test is: does that signal,
**at its measured precision and recall rather than an assumed one**, improve the deployed
long-only champion when used to cut exposure?

DESIGN
  * Model trains ONLY on decided first-touch events (purged), but SCORES EVERY BAR — training
    needs labels, inference does not. That yields a continuous probability series over the whole
    H4 index, which an overlay requires.
  * Overlay is `champion_fc * (1 - b * trim)`, so exposure can only be REDUCED, never boosted
    and never inverted (§3p: the champion is already long at bottoms; boosting is the worthless
    side).
  * GATE = matched-vol paired-t (§3ad's method fix). Every cell here multiplies the forecast by
    <= 1, i.e. de-risks, so a raw paired-t would mark a genuine risk-adjusted gain as a loss.
    Bar: t@matched >= +1.50 AND both half-deltas positive.
  * DISCLOSED GRID: 2 horizons x (3 quantile thresholds + 1 continuous) x 2 trim strengths
    = 16 cells, all printed, all counted in DSR/PBO.

    python scripts/v5_m15_trim_overlay.py
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
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402

from scripts.v5_volregime_taper_crossasset import engine_bp, load_sleeve  # noqa: E402
from scripts.v5_xau_champion_lifts import vol_match, sharpe, dd_of  # noqa: E402
from scripts.v5_xau_turn_prob import paired, per_year  # noqa: E402
from scripts.v5_repr_ceiling import first_touch, f_base, f_source, YEARS, PURGE_EXTRA  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402
from src.evaluation.dsr_pbo import deflated_sharpe_ratio, pbo_cscv  # noqa: E402

ANN, HL, COST_BP = 252 * 6, 42, 0.75
SPLITS = [("2018-01-01", "2021-12-31", "2018-21"), ("2022-01-01", "2026-12-31", "2022-26")]


def oos_prob_all_bars(X: pd.DataFrame, y: pd.Series, k: int) -> pd.Series:
    """Walk-forward OOS down-probability at EVERY bar. Trains on decided+purged events only."""
    out = pd.Series(np.nan, index=X.index)
    Xok = X.notna().all(axis=1)
    for yr in YEARS:
        te = (X.index >= f"{yr}-01-01") & (X.index <= f"{yr}-12-31")
        if te.sum() < 30:
            continue
        i0 = X.index.get_indexer([X.index[te][0]])[0]
        cut = i0 - (k + PURGE_EXTRA)
        if cut < 400:
            continue
        tr = np.zeros(len(X), bool); tr[:cut] = True
        m = tr & Xok.values & y.notna().values
        if m.sum() < 300 or y[m].nunique() < 2:
            continue
        Xtr, ytr = X[m].values, y[m].values
        pr = te & Xok.values
        if pr.sum() == 0:
            continue
        sc = StandardScaler().fit(Xtr)
        cands = []
        for mdl, use_sc in ((HistGradientBoostingClassifier(
                max_iter=200, max_leaf_nodes=8, l2_regularization=10.0, learning_rate=0.05,
                min_samples_leaf=40, random_state=0), False),
                (LogisticRegression(C=0.05, max_iter=2000), True)):
            try:
                A = sc.transform(Xtr) if use_sc else Xtr
                mdl.fit(A, ytr)
                # pick by in-train log-loss on a held-back tail (purged) to avoid test peeking
                tail = slice(max(0, len(A) - 400), len(A))
                cands.append((log_loss(ytr[tail], mdl.predict_proba(A[tail])[:, 1]), mdl, use_sc))
            except Exception:
                pass
        if not cands:
            continue
        _, mdl, use_sc = min(cands, key=lambda z: z[0])
        B = sc.transform(X[pr].values) if use_sc else X[pr].values
        out.loc[X.index[pr]] = mdl.predict_proba(B)[:, 1]
    return out


def main() -> None:
    t0 = time.time()
    df = load_sleeve(dict(path="data/XAUUSD_H4_long.csv"))
    m15 = load_sleeve(dict(path="data/XAUUSD_M15_long.csv"))
    close = df["close"]
    base = engine_bp(df, champion_signal(close), COST_BP, ANN, HL)
    hb = [sharpe(base.loc[a:b]) for a, b, _ in SPLITS]
    print(f"CHAMPION baseline: SR {sharpe(base):+.3f}  DD {dd_of(base):+.2f}%  "
          f"halves {hb[0]:+.2f}/{hb[1]:+.2f}")
    print("GATE: t@matched >= +1.50 AND both half-deltas positive "
          "(trim de-risks, so raw paired-t is invalid — §3ad)\n")

    Xs = f_source(df, m15)
    rows, streams = [], {}
    for k in (12, 30):
        y = first_touch(df, k)
        p = oos_prob_all_bars(Xs, y, k)
        cov = p.notna().mean()
        print(f"-- K={k} ({k/6:.1f}d): OOS probability on {cov:.0%} of bars, "
              f"mean {p.mean():.3f}  ({time.time()-t0:.0f}s)")
        pf = p.ffill()
        for tag, trim in (("q90", (pf >= pf.expanding(min_periods=250).quantile(0.90)).astype(float)),
                          ("q80", (pf >= pf.expanding(min_periods=250).quantile(0.80)).astype(float)),
                          ("q70", (pf >= pf.expanding(min_periods=250).quantile(0.70)).astype(float)),
                          ("cont", ((pf - 0.5) * 2).clip(0, 1))):
            for b in (0.5, 1.0):
                fc = (champion_signal(close) * (1.0 - b * trim.fillna(0.0))).clip(0, 2)
                d = engine_bp(df, fc, COST_BP, ANN, HL)
                i = d.index.intersection(base.index)
                dm = vol_match(d.loc[i], base.loc[i])
                _, t, _ = paired(dm, base.loc[i])
                yp, yn = per_year(dm, base.loc[i])
                dh = [sharpe(vol_match(d.loc[a:x], base.loc[a:x])) - sharpe(base.loc[a:x])
                      for a, x, _ in SPLITS]
                name = f"K{k} {tag} b={b}"
                keep = (t >= 1.50 and min(dh) > 0)
                rows.append(dict(cell=name, sr=sharpe(d), dd=dd_of(d), t=t, yy=f"{yp}/{yn}",
                                 d1=dh[0], d2=dh[1], expo=float(1 - b * trim.mean()), keep=keep))
                streams[name] = d
    R = pd.DataFrame(rows).sort_values("t", ascending=False)
    print(f"\n{'cell':18s} {'SR':>7s} {'DD':>8s} {'t@matched':>10s} {'yrs':>6s} "
          f"{'dSR 18-21':>10s} {'dSR 22-26':>10s} {'expo':>6s}")
    for _, r in R.iterrows():
        print(f"{r.cell:18s} {r.sr:+7.3f} {r.dd:7.2f}% {r.t:+10.2f} {r.yy:>6s} "
              f"{r.d1:+10.3f} {r.d2:+10.3f} {r.expo:6.2f}"
              + ("   ** KEEPER **" if r.keep else ""))
    # DSR / PBO over the whole disclosed grid
    per = pd.DataFrame({k: v for k, v in streams.items()}).dropna()
    if per.shape[1] >= 2:
        trials = [sharpe(per[c]) / np.sqrt(252) for c in per.columns]
        best = per[max(per.columns, key=lambda c: sharpe(per[c]))]
        dsr = deflated_sharpe_ratio(best.values, trials)
        pbo = pbo_cscv(per.values)
        print(f"\nDSR {dsr['dsr']:.4f}   PBO {pbo.pbo:.4f}   over {len(trials)} disclosed cells")
    R.to_csv(ROOT / "data/v5_runs/m15_trim_overlay.csv", index=False)
    print(f"keepers {int(R.keep.sum())}/{len(R)}   elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
