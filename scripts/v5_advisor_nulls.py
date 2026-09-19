"""MAX-STATISTIC NULL — price the search that produced the surviving cell.

WHY THIS RUN EXISTS. One cell out of 24 (8 direction + 16 adverse) cleared the pre-registered
tails-only criterion: `k=1.0 ATR / 9h / BASE`, reliability gap +13.20pp. Its Brier skill is
still negative (-0.0072), and it is the best of 24. §3ay measured best-of-6 as worth roughly
+0.03 AUC for free, and §3ak's 21,300-trigger search found the null distribution of the MAXIMUM
|z| had a median of 9.34. A single survivor out of 24 is exactly what a search of that size
produces by chance, so the honest question is not "is this cell good" but "**is it better than
the best cell a no-edge dataset would have produced?**"

THE STATISTIC. The surviving cell passed on the TAILS criterion, so the max-statistic is the
reliability GAP (top-bucket minus bottom-bucket observed frequency) across every cell in the
family. AUC is recorded alongside for comparability with §3ay's published distribution.

THE SURROGATE, AND ITS LIMIT, STATED PLAINLY. The label is block-shuffled and the ALREADY-FITTED
out-of-sample scores are re-paired against it, rather than refitting 16 cells x 200 draws (which
would be hours). This prices the SELECTION — which is what a max-statistic null is for, and the
same surrogate §3ay used. It does NOT price the fitting bias; that is the synthetic control's
job (PREREGISTRATION §9), and the synthetic control is not run here because no cell reached
Tier A, where §7 requires it.

Block shuffling, not permutation: `block=60` preserves the label's own autocorrelation and base
rate while destroying its alignment to the score. A plain permutation would break the
autocorrelation too and give an optimistically narrow null.

    python scripts/v5_advisor_nulls.py --family adv --draws 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "v5_runs" / "xau_advisor"

from src.v5.advisor_calibration import merge_thin_buckets, prob_edges  # noqa: E402


def block_shuffle_labels(y: np.ndarray, block: int, rng) -> np.ndarray:
    """Moving-block shuffle: resample blocks of consecutive labels with replacement."""
    n = len(y)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block, 1), nb)
    return np.concatenate([y[s:s + block] for s in starts])[:n]


def gap_and_auc(y: np.ndarray, p: np.ndarray, edges, min_n: int = 200) -> tuple:
    """Reliability gap (top minus bottom observed frequency, pp) and AUC on the same arrays."""
    e = np.asarray(edges, float)
    idx = np.clip(np.digitize(p, e[1:-1]), 0, len(e) - 2)
    counts = [int((idx == b).sum()) for b in range(len(e) - 1)]
    kept = merge_thin_buckets(counts, e, float(np.mean(y)), min_n)
    obs = []
    for lo, hi in kept:
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= 1.0)
        if m.sum() >= min_n:
            obs.append(float(np.mean(y[m])))
    gap = (obs[-1] - obs[0]) * 100 if len(obs) >= 2 else np.nan
    r = pd.Series(p).rank().values
    n1, n0 = y.sum(), len(y) - y.sum()
    auc = float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan
    return gap, auc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["dir", "adv"], required=True)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--block", type=int, default=60)
    a = ap.parse_args()

    f = OUT / f"oos_{a.family}.parquet"
    if not f.exists():
        raise SystemExit(f"missing {f} — run v5_advisor_measure.py --family {a.family} first")
    df = pd.read_parquet(f)
    kind = "direction" if a.family == "dir" else "adverse"
    edges = prob_edges(kind)
    cells = sorted(df["cell"].unique())
    print(f"family {a.family}: {len(cells)} cells, {len(df):,} OOS rows, "
          f"{a.draws} draws, block {a.block}")

    # observed statistics
    obs = {}
    for c in cells:
        d = df[df["cell"] == c]
        obs[c] = gap_and_auc(d["y"].values, d["p"].values, edges)
    obs_gap = {c: v[0] for c, v in obs.items()}
    obs_auc = {c: v[1] for c, v in obs.items()}
    best_cell = max(obs_gap, key=lambda c: obs_gap[c] if np.isfinite(obs_gap[c]) else -1e9)
    print(f"\nobserved best gap: {best_cell}  {obs_gap[best_cell]:+.2f}pp  "
          f"(AUC {obs_auc[best_cell]:.4f})")
    print(f"observed best AUC: {max(obs_auc, key=lambda c: obs_auc[c])}  "
          f"{max(obs_auc.values()):.4f}")

    # the null of the MAXIMUM across cells
    rng = np.random.default_rng(11)
    arrs = {c: (df[df["cell"] == c]["y"].values, df[df["cell"] == c]["p"].values) for c in cells}
    max_gap, max_auc = np.empty(a.draws), np.empty(a.draws)
    rows = []
    for i in range(a.draws):
        g_i, a_i = -1e9, 0.0
        for c in cells:
            y, p = arrs[c]
            ysh = block_shuffle_labels(y, a.block, rng)
            g, au = gap_and_auc(ysh, p, edges)
            if np.isfinite(g):
                g_i = max(g_i, g)
            if np.isfinite(au):
                a_i = max(a_i, au)
        max_gap[i], max_auc[i] = g_i, a_i
        rows.append(dict(draw=i, max_gap_pp=g_i, max_auc=a_i))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{a.draws}  running p95 gap {np.percentile(max_gap[:i+1], 95):+.2f}pp"
                  f"   p95 AUC {np.percentile(max_auc[:i+1], 95):.4f}")
    pd.DataFrame(rows).to_csv(OUT / f"null_maxstat_{a.family}.csv", index=False)

    q = lambda x, v: float(np.percentile(x, v))
    res = dict(family=a.family, draws=a.draws, block=a.block, cells=len(cells),
               observed_best_cell=best_cell, observed_best_gap_pp=obs_gap[best_cell],
               observed_best_auc=float(max(obs_auc.values())),
               null_gap=dict(mean=float(max_gap.mean()), p50=q(max_gap, 50),
                             p95=q(max_gap, 95), p99=q(max_gap, 99)),
               null_auc=dict(mean=float(max_auc.mean()), p50=q(max_auc, 50),
                             p95=q(max_auc, 95), p99=q(max_auc, 99)),
               p_value_gap=float(np.mean(max_gap >= obs_gap[best_cell])),
               p_value_auc=float(np.mean(max_auc >= max(obs_auc.values()))))
    res["gap_cleared_p95"] = bool(obs_gap[best_cell] > res["null_gap"]["p95"])
    res["auc_cleared_p95"] = bool(max(obs_auc.values()) > res["null_auc"]["p95"])
    (OUT / f"null_maxstat_{a.family}.json").write_text(json.dumps(res, indent=2))

    print(f"\n{'':22s} {'observed':>10s} {'null p50':>10s} {'null p95':>10s} "
          f"{'null p99':>10s} {'p-value':>9s}")
    print(f"{'max reliability gap':22s} {obs_gap[best_cell]:+10.2f} "
          f"{res['null_gap']['p50']:+10.2f} {res['null_gap']['p95']:+10.2f} "
          f"{res['null_gap']['p99']:+10.2f} {res['p_value_gap']:9.4f}")
    print(f"{'max AUC':22s} {max(obs_auc.values()):10.4f} "
          f"{res['null_auc']['p50']:10.4f} {res['null_auc']['p95']:10.4f} "
          f"{res['null_auc']['p99']:10.4f} {res['p_value_auc']:9.4f}")
    print(f"\nGATE 6 (must exceed p95 of the MAX across {len(cells)} cells):")
    print(f"  gap {'CLEARED' if res['gap_cleared_p95'] else '*** NOT CLEARED ***'}   "
          f"AUC {'CLEARED' if res['auc_cleared_p95'] else '*** NOT CLEARED ***'}")
    print(f"\nwrote null_maxstat_{a.family}.{{csv,json}}")


if __name__ == "__main__":
    main()
