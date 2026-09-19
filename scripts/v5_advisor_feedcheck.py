"""Re-run the adverse family on a SINGLE-FEED dataset and compare against §3az.

WHY. `data/XAUUSD_H4_long.csv` and `data/XAUUSD_M15_long.csv` are one feed up to 2022 and two
feeds after that. `refresh_xau_h4.py --bars 5000` reaches back to 2023-06-28, so every H4 bar
from that date on has been overwritten with the FTMO series while the M15 file has never been
refreshed and still carries the original one. Measured at the same instant:

    year   2015-2022   2023    2024    2025    2026
    agree     100.0%   29.9%    2.3%    0.9%    0.5%
    median   $0.0000  $1.24   $3.47   $6.64  $13.61

The adverse label sets its barriers from the H4 close and ATR and then asks which one the M15
path touches first. For 2023-06-28 onward it was pricing barriers in one broker's quotes and
testing touches in another's, drifting apart by up to $13.61 against a 1.0-ATR barrier that is
about $41 wide in 2026 — a third of the half-width. That period is also the SECOND half of the
reliability split, the half that carries the top tail (0.605 vs 0.547), so the contamination sits
exactly where the result lives. This is not a small-print caveat; it has to be re-measured.

THE CLEAN DATASET. The original H4 series is not recoverable from git (the data files are not
tracked), but it does not need to be: H4 is an exact aggregation of M15, and the M15 file is the
original feed throughout. Rebuilding H4 from M15 reproduces the filed H4 TO THE CENT for every
year up to 2022 (100% of bars within $0.10, median difference exactly 0.0000) and diverges only
where FTMO overwrote it — which both validates the reconstruction and localises the damage.

    python scripts/v5_advisor_feedcheck.py --draws 200
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

from scripts.v5_advisor_measure import (  # noqa: E402
    ADV_CELLS, BLOCK, MIN_DECIDED, MIN_NONOVERLAP, features_for, load_frames, nonoverlap,
    score, walk_forward,
)
from scripts.v5_advisor_nulls import block_shuffle_labels, gap_and_auc  # noqa: E402
from scripts.v5_range_multitf import atr_series  # noqa: E402
from scripts.v5_repr_ceiling import PURGE_EXTRA  # noqa: E402
from src.v5.advisor_calibration import (  # noqa: E402
    operating_points, prob_edges, reliability_table,
)
from src.v5.advisor_labels import first_touch_atr, resolution_rate  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "xau_advisor"


def rebuild_h4_from_m15(m15: pd.DataFrame) -> pd.DataFrame:
    """Exact H4 aggregation of the M15 series — one feed by construction.

    Partial bars are KEPT, with `legs` recorded, because the filed H4 file contains them too and
    dropping them here would change the sample rather than only the feed, confounding the very
    comparison this script exists to make.
    """
    d = m15.copy()
    d.index = pd.DatetimeIndex(d.index)
    g = d.groupby(d.index.floor("4h"))
    out = pd.DataFrame(dict(open=g["open"].first(), high=g["high"].max(), low=g["low"].min(),
                            close=g["close"].last(), legs=g["close"].count()))
    for c in ("tick_volume", "spread"):
        if c in d.columns:
            out[c] = g[c].sum() if c == "tick_volume" else g[c].median()
    out["real_volume"] = np.nan
    return out


def feed_divergence(h4: pd.DataFrame, m15: pd.DataFrame) -> pd.DataFrame:
    """The table quoted in this module's docstring, recomputed so it cannot go stale."""
    t = h4.index + pd.Timedelta(minutes=225)
    common = m15.index.intersection(t)
    d = pd.Series(np.abs(m15.loc[common, "close"].values
                         - h4.loc[common - pd.Timedelta(minutes=225), "close"].values),
                  index=common)
    return pd.DataFrame({"n": d.groupby(d.index.year).size(),
                         "median_abs": d.groupby(d.index.year).median(),
                         "frac_within_10c": d.groupby(d.index.year).apply(
                             lambda x: float((x <= 0.10).mean()))})


def run_family(h4: pd.DataFrame, m15: pd.DataFrame, label: str) -> tuple:
    atr = atr_series(h4, 20)
    ov = h4.index[(h4.index >= m15.index[0]) & (h4.index <= m15.index[-1])]
    rows, store = [], {}
    for k, hz in ADV_CELLS:
        nb = hz * 4
        ft = first_touch_atr(ov, h4.loc[ov, "close"].values, atr.loc[ov].values,
                             m15["high"].values, m15["low"].values, m15.index, k, nb,
                             dec_bar_minutes=240)
        rr = resolution_rate(ft)
        for arm in ("BASE", "SOURCE"):
            X = features_for(arm, h4, m15).reindex(ov)
            y = ft["y"].values
            wf = walk_forward(X, y, max(1, nb // 16) + PURGE_EXTRA, ov)
            m = wf["mask"] & np.isfinite(y)
            if m.sum() < MIN_DECIDED:
                continue
            hbars = max(1, nb // 16)
            per = np.full(int(m.sum()), 1.0 if rr["base_rate"] > 0.5 else 0.0)
            s = score(y[m], wf["p_cal"][m], per, wf["drift"][m], wf["clim"][m], ov[m], hbars,
                      kind="adverse")
            s.update(k_atr=k, hours=hz, arm=arm, feed=label,
                     resolved_frac=rr["resolved_frac"],
                     n_nonoverlap=int(len(nonoverlap(np.flatnonzero(m), hbars))))
            rows.append(s)
            store[f"{k}_{hz}_{arm}"] = dict(idx=ov[m], y=y[m], p=wf["p_cal"][m])
            print(f"  [{label}] k={k:<4} {hz}h {arm:6s} n {s['n']:6d} base {s['base_rate']:.4f} "
                  f"AUC {s['auc']:.4f} BSS {s['bss']:+.4f}")
    return pd.DataFrame(rows), store


def maxstat(store: dict, draws: int, block: int = BLOCK) -> dict:
    """The same best-of-16 null §3az used, so the two numbers are comparable."""
    edges = prob_edges("adverse")
    obs = {c: gap_and_auc(d["y"], d["p"], edges) for c, d in store.items()}
    best_gap = max(v[0] for v in obs.values() if np.isfinite(v[0]))
    best_auc = max(v[1] for v in obs.values() if np.isfinite(v[1]))
    rng = np.random.default_rng(11)
    mg, ma = np.empty(draws), np.empty(draws)
    for i in range(draws):
        g_i, a_i = -1e9, 0.0
        for d in store.values():
            g, au = gap_and_auc(block_shuffle_labels(d["y"], block, rng), d["p"], edges)
            if np.isfinite(g):
                g_i = max(g_i, g)
            if np.isfinite(au):
                a_i = max(a_i, au)
        mg[i], ma[i] = g_i, a_i
    q = lambda x, v: float(np.percentile(x, v))
    return dict(best_gap=float(best_gap), best_auc=float(best_auc),
                gap_p50=q(mg, 50), gap_p95=q(mg, 95), gap_p99=q(mg, 99),
                auc_p50=q(ma, 50), auc_p95=q(ma, 95), auc_p99=q(ma, 99),
                p_gap=float(np.mean(mg >= best_gap)), p_auc=float(np.mean(ma >= best_auc)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    a = ap.parse_args()

    h4_filed, m15 = load_frames()
    h4_clean = rebuild_h4_from_m15(m15)

    print("FEED DIVERGENCE between the filed H4 and the M15 it was paired with:")
    print(feed_divergence(h4_filed, m15).to_string(float_format=lambda x: f"{x:.4f}"))

    # the reconstruction's own validation, printed every run rather than asserted once
    ix = h4_clean.index.intersection(h4_filed.index)
    d = (h4_clean.loc[ix, "close"] - h4_filed.loc[ix, "close"]).abs()
    pre = d[ix.year <= 2022]
    print(f"\nreconstruction check (rebuilt vs filed, 2015-2022): n {len(pre):,}  "
          f"median {pre.median():.4f}  within $0.10 {(pre <= 0.10).mean()*100:.1f}%")
    if pre.median() > 0.01 or (pre <= 0.10).mean() < 0.99:
        raise SystemExit("rebuilt H4 does not reproduce the filed H4 in the clean era — stop")

    print("\n=== CLEAN (single-feed) ===")
    clean, store = run_family(h4_clean, m15, "clean")
    clean.to_csv(OUT / "adverse_grid_cleanfeed.csv", index=False)

    flat = [pd.DataFrame(dict(cell=c, time=dd["idx"], y=dd["y"], p=dd["p"]))
            for c, dd in store.items()]
    pd.concat(flat, ignore_index=True).to_parquet(OUT / "oos_adv_cleanfeed.parquet")
    ops = []
    for c, dd in store.items():
        o = operating_points(dd["p"], dd["y"], index=dd["idx"], hysteresis=0.02, n_boot=500)
        o.insert(0, "cell", c)
        ops.append(o)
    pd.concat(ops, ignore_index=True).to_csv(OUT / "operating_points_adv_cleanfeed.csv",
                                             index=False)

    rel = []
    for c, dd in store.items():
        t = reliability_table(dd["y"], dd["p"], edges=prob_edges("adverse"), index=dd["idx"],
                              half_at=pd.Timestamp("2022-01-01"), n_boot=500)
        t.insert(0, "cell", c)
        rel.append(t)
    pd.concat(rel, ignore_index=True).to_csv(OUT / "reliability_adv_cleanfeed.csv", index=False)

    if a.draws > 0:
        print(f"\nrunning the best-of-16 max-stat null, {a.draws} draws ...")
        null = maxstat(store, a.draws)
        (OUT / "null_maxstat_adv_cleanfeed.json").write_text(json.dumps(null, indent=2))
    else:
        null = json.loads((OUT / "null_maxstat_adv_cleanfeed.json").read_text())

    old = json.loads((OUT / "null_maxstat_adv.json").read_text())
    print("\n" + "=" * 92)
    print("GATE 6 — MIXED FEED (§3az, published)  vs  CLEAN FEED (this run)")
    print("=" * 92)
    print(f"{'':16s} {'observed':>10s} {'p50':>9s} {'p95':>9s} {'p99':>9s} {'p-value':>9s}")
    print(f"{'gap  MIXED':16s} {old['observed_best_gap_pp']:+10.2f} "
          f"{old['null_gap']['p50']:+9.2f} {old['null_gap']['p95']:+9.2f} "
          f"{old['null_gap']['p99']:+9.2f} {old['p_value_gap']:9.4f}")
    print(f"{'gap  CLEAN':16s} {null['best_gap']:+10.2f} {null['gap_p50']:+9.2f} "
          f"{null['gap_p95']:+9.2f} {null['gap_p99']:+9.2f} {null['p_gap']:9.4f}")
    print(f"{'AUC  MIXED':16s} {old['observed_best_auc']:10.4f} {old['null_auc']['p50']:9.4f} "
          f"{old['null_auc']['p95']:9.4f} {old['null_auc']['p99']:9.4f} {old['p_value_auc']:9.4f}")
    print(f"{'AUC  CLEAN':16s} {null['best_auc']:10.4f} {null['auc_p50']:9.4f} "
          f"{null['auc_p95']:9.4f} {null['auc_p99']:9.4f} {null['p_auc']:9.4f}")

    mixed = pd.read_csv(OUT / "adverse_grid.csv")
    mixed = mixed[mixed.get("admissible", True).astype(bool)] if "admissible" in mixed else mixed
    j = clean.merge(mixed, on=["k_atr", "hours", "arm"], suffixes=("_clean", "_mixed"))
    print("\nPER-CELL AUC, mixed -> clean:")
    print(f"{'cell':20s} {'mixed':>8s} {'clean':>8s} {'delta':>8s} {'BSS clean':>10s}")
    for _, r in j.sort_values("auc_clean", ascending=False).iterrows():
        print(f"{r['k_atr']}_{int(r['hours'])}_{r['arm']:6s}    {r['auc_mixed']:8.4f} "
              f"{r['auc_clean']:8.4f} {r['auc_clean']-r['auc_mixed']:+8.4f} "
              f"{r['bss_clean']:+10.4f}")
    print(f"\nmean AUC delta {(j['auc_clean']-j['auc_mixed']).mean():+.4f}  "
          f"cells improving {(j['auc_clean']>j['auc_mixed']).sum()}/{len(j)}")
    print(f"\nwrote adverse_grid_cleanfeed.csv, reliability_adv_cleanfeed.csv, "
          f"null_maxstat_adv_cleanfeed.json")


if __name__ == "__main__":
    main()
