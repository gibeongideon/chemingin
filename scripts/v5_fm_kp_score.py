#!/usr/bin/env python3
"""SCORING the Kronos paper-protocol cells and the embedding probe (V5_FINDINGS 3ew, addendum 1,
commit 5ad0167). Features: scripts/v5_fm_kronos_paths.py (blind to outcomes).

  KPd  daily, look-back 90: p^[t+1]/p[t]-1            -> that day's open -> close
  KPn  H1, look-back 80, decided at 08:00 ET: p^[t+8] -> NY session 08:00 -> 16:00 ET
  KPh  H1, look-back 80, every bar from 2024-07: p^[t+12] (Eq. 11) -> realised 12-bar return
  KPd10 (informational) Eq. 13 mean path -> 10-day forward return
  E    ridge probe on the 512-d hidden state, walk-forward inside the POST window

    python scripts/v5_fm_kp_score.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_fm_day_signal import COST, MED_WIN, N_NULL, WINDOWS, sharpe, tstat  # noqa: E402
from scripts.v5_fm_features import NY, OUT, build_frames  # noqa: E402


def cell(f: Path, kind: str, fr: dict) -> pd.DataFrame | None:
    if not f.exists():
        return None
    x = pd.read_parquet(f)
    if kind in ("KPd", "KPd10"):
        b = fr["D1"].reset_index(drop=True)
        pos = pd.Index(pd.DatetimeIndex(b.t_open)).get_indexer(x.index)
        c = b.close.to_numpy()
        if kind == "KPd":
            y, sig = (b.close / b.open - 1).to_numpy()[pos], x.mu1.to_numpy()
        else:
            fwd = np.full(len(b), np.nan)
            fwd[:-9] = c[9:]
            y, sig = fwd[pos] / x.last_close.to_numpy() - 1, x.mu_avg.to_numpy()
    elif kind == "KPn":
        s8 = fr["S8"]
        ny = s8[s8.index.get_level_values("blk") == 1]
        yy = pd.Series((ny.close / ny.open - 1).to_numpy(), index=ny.index.get_level_values("date"))
        et = x.index.tz_localize("UTC").tz_convert(NY).tz_localize(None).normalize()
        y, sig = yy.reindex(et).to_numpy(), x.mu8.to_numpy()
    else:
        b = fr["H1"]
        pos = pd.Index(pd.DatetimeIndex(b.t_open)).get_indexer(x.index)
        c = b.close.to_numpy()
        fwd = np.full(len(b), np.nan)
        fwd[:-11] = c[11:]
        y, sig = fwd[pos] / x.last_close.to_numpy() - 1, x.muH.to_numpy()
    d = pd.DataFrame(dict(t=x.index.values, sig=sig, y=y, px=x.last_close.to_numpy()))
    d["z"] = d.sig - d.sig.rolling(MED_WIN, min_periods=MED_WIN).median().shift(1)
    d["pos"] = np.sign(d.z)
    return d.dropna(subset=["z", "y"]).reset_index(drop=True)


def merge_shards() -> None:
    """The generator runs as single-threaded shards; stitch them back together."""
    groups: dict[str, list[Path]] = {}
    for f in sorted(OUT.glob("kp_*_s?.parquet")):
        groups.setdefault(f.name[:-11], []).append(f)
    for stem, fs in groups.items():
        pd.concat([pd.read_parquet(f) for f in fs]).sort_index().to_parquet(OUT / f"{stem}.parquet")
    es = sorted(OUT.glob("kp_small_oanda_KPh_emb_s?.npz"))
    if es:
        z = [np.load(f) for f in es]
        t, e = np.concatenate([q["t"] for q in z]), np.vstack([q["emb"] for q in z])
        o = np.argsort(t)
        np.savez_compressed(OUT / "kp_small_oanda_KPh_emb.npz", emb=e[o], t=t[o])


def main() -> None:
    merge_shards()
    L = []
    for source in ("oanda", "cme"):
        fr = build_frames(source)
        cells, step = {}, {}
        for size in ("small", "base"):
            for kind, st in (("KPd", 1), ("KPn", 1), ("KPh", 12), ("KPd10", 10)):
                f = OUT / f"kp_{size}_{source}_{'KPd' if kind == 'KPd10' else kind}.parquet"
                d = cell(f, kind, fr)
                if d is not None and len(d) > 200:
                    if kind == "KPh":                             # compute deviation: small sampled every 3rd bar,
                        st = 4 if size == "small" else 2          # base every 6th -> 12 bars between scored decisions
                    cells[f"{size}_{kind}"], step[f"{size}_{kind}"] = d, st
        if not cells:
            continue
        rows = []
        for name, d in cells.items():
            s = d.iloc[::step[name]]
            g = (s.pos * s.y).to_numpy()
            h = len(g) // 2
            r = dict(cell=name, n=len(d), n_indep=len(s), first=str(d.t.iloc[0])[:10], hit=round(100 * (g > 0).mean(), 2),
                     base_up=round(100 * (s.y > 0).mean(), 2), mean_bp=round(1e4 * g.mean(), 2),
                     ic=round(stats.spearmanr(d.sig, d.y)[0], 4), ic_z=round(stats.spearmanr(d.z, d.y)[0], 4), t=round(tstat(g), 2),
                     h1=round(tstat(g[:h]), 2), h2=round(tstat(g[h:]), 2),
                     SR_net=round(sharpe(g - COST / s.px.to_numpy()), 2) if step[name] == 1 else np.nan)
            for w, (a, b) in WINDOWS.items():
                m = np.ones(len(s), bool)
                if a:
                    m &= (s.t >= np.datetime64(a)).to_numpy()
                if b:
                    m &= (s.t < np.datetime64(b)).to_numpy()
                r[w] = f"{tstat(g[m]):+.2f} ({int(m.sum())})" if m.sum() > 40 else "-"
            rows.append(r)
        A = pd.DataFrame(rows)
        fam = [c for c in cells if not c.endswith("KPd10")]
        rng = np.random.default_rng(31)
        mx, cn = np.full(N_NULL, -np.inf), {k: np.empty(N_NULL) for k in cells}
        for j in range(N_NULL):
            u = rng.random()
            for name, d in cells.items():
                s = d.iloc[::step[name]]
                lo = max(2, int(np.ceil(20 * (24 / 12 if "KPh" in name else 1 / step[name]))))
                k = int(lo + u * (len(s) - 2 * lo))
                cn[name][j] = tstat(s.pos.to_numpy() * np.roll(s.y.to_numpy(), k))
            mx[j] = max(cn[n][j] for n in fam)
        A["p_cell"] = [round(float((cn[c] >= t).mean()), 3) for c, t in zip(A.cell, A.t)]
        best = A[A.cell.isin(fam)].sort_values("t").iloc[-1]
        L += [f"KRONOS PAPER PROTOCOL (T 0.6, top-p 0.9, N 10, mean path), source {source.upper()}",
              A.to_string(index=False),
              "   ic = Spearman(raw signal, outcome) on all decisions - the paper's own forex yardstick is 0.040 (small) / 0.055 (base), pooled "
              "over 1,023 pairs; t on non-overlapping decisions; windows = t (n)",
              f"   FAMILY ({len(fam)} cells, KPd10 informational): best {best.cell} t {best.t:+.2f} vs shifted-outcome null max median "
              f"{np.median(mx):+.2f}, 95th {np.percentile(mx, 95):+.2f}, p {(mx >= best.t).mean():.3f}", ""]

    # E: the embedding as a feature
    f = OUT / "kp_small_oanda_KPh_emb.npz"
    if f.exists():
        z = np.load(f)
        E, tt = z["emb"].astype(np.float64), pd.DatetimeIndex(z["t"].astype("datetime64[ns]"))
        b = build_frames("oanda")["H1"]
        pos = pd.Index(pd.DatetimeIndex(b.t_open)).get_indexer(tt)
        c = b.close.to_numpy()
        r1 = pd.Series(c).pct_change()
        vol = (r1.rolling(480).std().shift(1) * np.sqrt(12)).to_numpy()
        fwd = np.full(len(b), np.nan); fwd[:-11] = c[11:]
        y = (fwd[pos] / c[pos - 1] - 1) / vol[pos]
        ok = np.isfinite(y)
        E, y, tt = E[ok], y[ok], tt[ok]
        n, n0 = len(y), int(0.4 * len(y))
        day = tt.normalize()
        udays = day.unique()

        def walk(yv: np.ndarray) -> np.ndarray:
            pred = np.full(n, np.nan)
            start_day = np.searchsorted(udays, day[n0])
            for a in range(start_day, len(udays), 20):
                lo, hi = np.searchsorted(day, udays[a]), np.searchsorted(day, udays[min(a + 20, len(udays) - 1)]) if a + 20 < len(udays) else n
                tr = np.arange(0, max(lo - 4, 1))                 # purge 12 bars = 4 sampled decisions
                m, s = E[tr].mean(0), E[tr].std(0) + 1e-9
                X = (E[tr] - m) / s
                w = np.linalg.solve(X.T @ X + 1000.0 * np.eye(X.shape[1]), X.T @ (yv[tr] - yv[tr].mean()))
                pred[lo:hi] = ((E[lo:hi] - m) / s) @ w
            return pred

        def score(pred, yv):
            m = np.isfinite(pred)
            idx = np.flatnonzero(m)[::4]                         # embeddings exist every 3rd bar
            return stats.spearmanr(pred[m], yv[m])[0], tstat(np.sign(pred[idx]) * yv[idx]), int(m.sum()), len(idx)

        ic, t, nt, ni = score(walk(y), y)
        rng = np.random.default_rng(37)
        nl = []
        for _ in range(200):
            ys = np.roll(y, int(rng.integers(160, n - 160)))
            nl.append(score(walk(ys), ys)[1])
        nl = np.array(nl)
        L += ["E. THE EMBEDDING AS A FEATURE (Kronos-small hidden state, 512-d; ridge alpha 1000; walk-forward inside POST only)",
              f"   test bars {nt:,} ({tt[np.isfinite(walk(y))][0].date()} -> {tt[-1].date()}), independent decisions {ni}: out-of-sample IC {ic:+.4f}, "
              f"t {t:+.2f}; shifted-outcome null median {np.median(nl):+.2f}, 95th {np.percentile(nl, 95):+.2f}, p {(nl >= t).mean():.3f}"]
    txt = "\n".join(L)
    print(txt)
    (OUT / "fm_kp_report.txt").write_text(txt + "\n")


if __name__ == "__main__":
    sys.exit(main())
