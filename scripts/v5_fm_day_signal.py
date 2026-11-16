#!/usr/bin/env python3
"""SCORING the foundation-model features (V5_FINDINGS 3ew). Features come from
scripts/v5_fm_features.py, which never touches an outcome. Bars, cells and pass marks are fixed
in data/v5_runs/intraday_day_signal/PREREGISTRATION.md (commit 264ebb8).

T1  direction as MONEY: pos = sign(mu - median of the previous 250 decisions' mu); long/short,
    sign fixed (follow the model). Signed t of mean(pos x outcome), gross; family = {Kronos,
    Chronos} x {D1, S8, H1}; null = circular shift of the outcome by >= 20 trading days, max
    signed t over the family. Hit rate is always printed beside the base rate (control 8).
T2  range / asymmetry information (descriptive).
T3  as a FEATURE: size dial on ZigZag variant B + breaker fires, event set fixed (3eu protocol).

Windows for Kronos. Its paper (arXiv 2508.02739, App. D) states the pre-training data ends June
2024 and Table 13 lists 1,023 FX pairs from 2020-01-31, so:
  PRE   < 2020-02-01            gold spot not in the corpus
  TRAIN 2020-02-01..2024-06-30  possibly memorised
  POST  >= 2024-07-01           after the paper's data cut-off (the authors' own test period)
  CLEAN > 2025-09-09            after the last change to the released weights (pre-registered)

    python scripts/v5_fm_day_signal.py t1
    python scripts/v5_fm_day_signal.py t3
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
OUT = ROOT / "data" / "v5_runs" / "intraday_day_signal"
MED_WIN, N_NULL, N_RAND, COST = 250, 2000, 500, 0.30
WINDOWS = {"PRE": (None, "2020-02-01"), "TRAIN": ("2020-02-01", "2024-07-01"), "POST": ("2024-07-01", None),
           "CLEAN": ("2025-09-10", None)}


def tstat(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0.0


def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / x.std() * np.sqrt(252)) if len(x) > 2 and x.std() > 0 else 0.0


def load_cell(model: str, source: str, frame: str, frames: dict) -> pd.DataFrame | None:
    f = OUT / f"fm_{model}_{source}_{frame}.parquet"
    if not f.exists():
        return None
    x = pd.read_parquet(f)
    bars = frames[frame].reset_index(drop=True)
    pos_ = x.index.to_numpy()
    b = bars.iloc[pos_]
    assert (pd.DatetimeIndex(b.t_open).values == pd.DatetimeIndex(x.t_open).values).all(), "frame / feature misalignment"
    pre = "k_" if model == "kronos" else "c_"
    d = pd.DataFrame(dict(t=pd.DatetimeIndex(x.t_open).values, mu=x[pre + "mu"].to_numpy(), rng=x[pre + "rng"].to_numpy(),
                          sk=x[pre + "skew"].to_numpy(), y=(b.close / b.open - 1).to_numpy(),
                          rr=((b.high - b.low) / b.open).to_numpy(),
                          clv=(((b.close - b.low) - (b.high - b.close)) / (b.high - b.low).replace(0, np.nan)).to_numpy(),
                          px=b.open.to_numpy()))
    d["day"] = pd.DatetimeIndex(d.t).normalize()
    med = d.mu.rolling(MED_WIN, min_periods=MED_WIN).median().shift(1)
    d["z"] = d.mu - med
    d["pos"] = np.sign(d.z).fillna(0.0)
    return d[d.z.notna()].reset_index(drop=True)


def day_t(d: pd.DataFrame, pnl: np.ndarray) -> float:
    """t of the mean with days as the independent unit (H1 has up to 24 decisions per day)."""
    codes = pd.factorize(d.day.to_numpy())[0]
    return tstat(np.bincount(codes, weights=np.asarray(pnl, float)))


def t1(source: str) -> list[str]:
    from scripts.v5_fm_features import build_frames  # noqa: PLC0415
    frames = build_frames(source)
    cells, L = {}, []
    for model in ("kronos", "chronos"):
        for frame in ("D1", "S8", "H1"):
            d = load_cell(model, source, frame, frames)
            if d is not None and len(d) > 300:
                cells[f"{model}_{frame}"] = d
    rows = []
    for name, d in cells.items():
        g = (d.pos * d.y).to_numpy()
        act = d.pos != 0
        half = len(d) // 2
        per_day = name.endswith("H1")
        daily = pd.Series(g - (COST / d.px.to_numpy()) * act.to_numpy()).groupby(d.day.to_numpy()).sum()
        r = dict(cell=name, n=len(d), first=str(d.t.iloc[0])[:10], hit=round(100 * (g[act] > 0).mean(), 2),
                 base_up=round(100 * (d.y > 0).mean(), 2), mean_bp=round(1e4 * g.mean(), 3),
                 cost_bp=round(1e4 * (COST / d.px).mean(), 2), ic=round(stats.spearmanr(d.z, d.y)[0], 4),
                 t=round(day_t(d, g), 2), h1=round(day_t(d[:half], g[:half]), 2), h2=round(day_t(d[half:], g[half:]), 2),
                 SR_gross=round(sharpe(pd.Series(g).groupby(d.day.to_numpy()).sum()), 2),
                 SR_net=round(sharpe(daily), 2) if not per_day else np.nan)
        for w, (a, b) in WINDOWS.items():
            m = np.ones(len(d), bool)
            if a:
                m &= (d.t >= a).to_numpy()
            if b:
                m &= (d.t < b).to_numpy()
            r[w] = f"{day_t(d[m], g[m]):+.2f} ({int(m.sum())})" if m.sum() > 60 else "-"
        rows.append(r)
    A = pd.DataFrame(rows)
    # family null: one calendar shift per draw, applied to every cell's OUTCOME
    rng = np.random.default_rng(23)
    mx = np.full(N_NULL, -np.inf)
    cell_null = {k: np.empty(N_NULL) for k in cells}
    for j in range(N_NULL):
        u = rng.random()
        for name, d in cells.items():
            per = 24 if name.endswith("H1") else 1
            lo, hi = 20 * per, len(d) - 20 * per
            k = int(lo + u * (hi - lo))
            cell_null[name][j] = day_t(d, d.pos.to_numpy() * np.roll(d.y.to_numpy(), k))
        mx[j] = max(cell_null[n][j] for n in cells)
    A["p_cell"] = [round(float((cell_null[c] >= t).mean()), 3) for c, t in zip(A.cell, A.t)]
    best = A.loc[A.t.idxmax()]
    L += [f"T1 DIRECTION AS MONEY, source {source.upper()}; pos = sign(mu - trailing-250 median), gross, sign fixed",
          A.to_string(index=False),
          "   hit beside base_up (share of up outcomes); mean_bp per decision beside cost_bp (one round trip at $0.30); "
          "t by days; window columns = t (n)",
          f"   FAMILY ({len(cells)} cells): best {best.cell} t {best.t:+.2f} vs shifted-outcome null max median {np.median(mx):+.2f}, "
          f"95th {np.percentile(mx, 95):+.2f}, p {(mx >= best.t).mean():.3f}", ""]
    # long-only variant against a random same-size drop, D1 and S8 only
    L.append("   LONG-ONLY (long when mu above its trailing median, else flat) vs a RANDOM same-size drop and vs always-long:")
    rng = np.random.default_rng(29)
    for name, d in cells.items():
        if name.endswith("H1"):
            continue
        sel = (d.pos > 0).to_numpy()
        y = d.y.to_numpy() - COST / d.px.to_numpy()
        s_sel, s_all = sharpe(np.where(sel, y, 0.0)), sharpe(y)
        rnd = np.array([sharpe(np.where(rng.permutation(sel), y, 0.0)) for _ in range(N_RAND)])
        L.append(f"   {name:11s} long {100 * sel.mean():.0f}% of days: net SR {s_sel:+.2f} vs random drop median {np.median(rnd):+.2f} "
                 f"(pctile {100 * (rnd <= s_sel).mean():.0f}); always-long net SR {s_all:+.2f}; mean bp selected "
                 f"{1e4 * d.y[sel].mean():+.2f} vs skipped {1e4 * d.y[~sel].mean():+.2f}")
    # T2
    L += ["", f"T2 RANGE AND ASYMMETRY, source {source.upper()} (Spearman; partial = after the trailing-20 mean realised range)"]
    for name, d in cells.items():
        base = d.rr.rolling(20).mean().shift(1)
        m = base.notna()
        a, b_, y = d.rng[m].rank(), base[m].rank(), d.rr[m].rank()
        res = lambda u, v: u - np.polyval(np.polyfit(v, u, 1), v)  # noqa: E731
        part = stats.spearmanr(res(a, b_), res(y, b_))[0]
        cm = d.clv.notna()
        L.append(f"   {name:11s} rho(pred range, real) {stats.spearmanr(d.rng[m], d.rr[m])[0]:+.3f}; trailing-20 baseline "
                 f"{stats.spearmanr(base[m], d.rr[m])[0]:+.3f}; PARTIAL {part:+.3f};  rho(pred skew, close location) "
                 f"{stats.spearmanr(d.sk[cm], d.clv[cm])[0]:+.4f}")
    return L


def t3() -> list[str]:
    from scripts.v5_zigzag_fall_gate import circuit, daily  # noqa: PLC0415
    from scripts.v5_zigzag_frequency import Sim  # noqa: PLC0415
    from scripts.v5_zigzag_long_history import OUT as ZZ, RISK, SPREAD, SRC, THR, load  # noqa: PLC0415
    L = ["T3 AS A FEATURE: size dial on ZigZag variant B + breaker fires (event set fixed; m = 0.5 + expanding rank, mean 1)"]
    for src in ("cme", "switch"):
        cfg = SRC[src]
        g = load(src, cfg["gold"])
        p = pd.read_csv(ZZ / src / "probs.csv", index_col=0, parse_dates=True).iloc[:, 0].reindex(g.index).to_numpy()
        sim = Sim(g, pd.Timestamp(cfg["start"]), cfg["swap"])
        fire = np.isfinite(p) & (p >= THR) & ~circuit(g, daily(g), 1.0, "day")
        rows, busy = [], -1
        for i in np.flatnonzero(fire & sim.ok):
            if i >= sim.n - 1 or i < busy:
                continue
            s, j, R = sim.trade(int(i), SPREAD)
            busy = j
            rows.append((g.index[i], sim.t[j].normalize(), p[i], R))
        tr = pd.DataFrame(rows, columns=["t_fire", "day_out", "p", "R"])
        feats = {"kronos k_mu24": pd.read_parquet(OUT / f"fm_kronos_zz_{src}.parquet").k_mu,
                 "chronos c_mu24": pd.read_parquet(OUT / f"fm_chronos_zz_{src}.parquet").c_mu}
        day_idx, wk = sim.days.get_indexer(tr.day_out), (sim.days.dayofweek < 5)

        def sr(mult, sel=None):
            x = np.zeros(len(sim.days))
            r = RISK * mult * tr.R.to_numpy()
            if sel is not None:
                r = np.where(sel, r, 0.0)
            np.add.at(x, day_idx, r)
            x = x[wk]
            return x.mean() / x.std() * np.sqrt(252)

        flat = sr(np.ones(len(tr)))
        L += ["", f"   {src.upper()}: {len(tr)} trades {tr.t_fire.iloc[0].date()} -> {tr.t_fire.iloc[-1].date()}, FLAT Sharpe {flat:+.3f}, "
                  f"expectancy {tr.R.mean():+.3f}R"]
        rng = np.random.default_rng(17)
        perms = [rng.permutation(len(tr)) for _ in range(N_NULL)]
        fam_null, fam_obs = np.full(N_NULL, -np.inf), -np.inf
        half = len(tr) // 2
        for nm, f in feats.items():
            v = f.reindex(tr.t_fire).to_numpy()
            ok = np.isfinite(v)
            rank = np.full(len(tr), 0.5)
            seen: list[float] = []
            for q in range(len(tr)):
                if ok[q]:
                    if len(seen) >= 50:
                        rank[q] = (np.sum(np.array(seen) < v[q]) + 0.5) / (len(seen) + 1)
                    seen.append(v[q])
            m, inv = 0.5 + rank, 1.5 - rank
            m, inv = m / m.mean(), inv / inv.mean()
            s, s_inv = sr(m), sr(inv)
            nl = np.array([sr(m[k]) for k in perms])
            fam_null, fam_obs = np.maximum(fam_null, nl), max(fam_obs, s)
            rho, prho = stats.spearmanr(v[ok], tr.R[ok])
            h = [sr(m, np.arange(len(tr)) < half) - sr(np.ones(len(tr)), np.arange(len(tr)) < half),
                 sr(m, np.arange(len(tr)) >= half) - sr(np.ones(len(tr)), np.arange(len(tr)) >= half)]
            wtxt = []
            for w, (a, b) in WINDOWS.items():
                mk = ok.copy()
                if a:
                    mk &= (tr.t_fire >= a).to_numpy()
                if b:
                    mk &= (tr.t_fire < b).to_numpy()
                if mk.sum() > 40:
                    wtxt.append(f"{w} {stats.spearmanr(v[mk], tr.R[mk])[0]:+.3f} ({int(mk.sum())})")
            L.append(f"   {nm:15s} coverage {100 * ok.mean():.0f}%  Spearman(f, R) {rho:+.4f} (p {prho:.3f})  dial {s:+.3f} "
                     f"(d {s - flat:+.3f}, shuffle p {(nl >= s).mean():.3f})  INV {s_inv:+.3f} (d {s_inv - flat:+.3f})  "
                     f"halves d {h[0]:+.3f} / {h[1]:+.3f}")
            L.append(f"   {'':15s} Spearman by window: " + "  ".join(wtxt))
        L.append(f"   family (2 features): best dial {fam_obs:+.3f} vs shuffled max median {np.median(fam_null):+.3f}, "
                 f"95th {np.percentile(fam_null, 95):+.3f}, p {(fam_null >= fam_obs).mean():.3f}")
    return L


def main() -> None:
    what = sys.argv[1]
    L = []
    if what == "t1":
        for source in (sys.argv[2:] or ("oanda", "cme")):
            L += t1(source) + [""]
        what = "t1_" + "_".join(sys.argv[2:] or ("all",))
    else:
        L = t3()
    txt = "\n".join(L)
    print(txt)
    (OUT / f"fm_{what}_report.txt").write_text(txt + "\n")


if __name__ == "__main__":
    sys.exit(main())
