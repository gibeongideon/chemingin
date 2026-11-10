#!/usr/bin/env python3
"""Can ZigZag stand aside on days gold FALLS? New prediction mechanisms (user request 2026-11-10).

ZigZag variant B (long bottoms, 3 ATR trail, TP 4.3R, 120h) on the §3dx long histories. A GATE
blocks NEW entries while it flags "fall risk"; open trades are left to their stop.

STEP 1 — ORACLE CEILING (control #5, not in the family): block entries on days whose TRUE close-to-close
return is < 0, with the flag correct with probability a in {1.0, .8, .7, .6, .55}. This says how
accurate a fall-day predictor must be before blocking is worth anything.

STEP 2 — PRE-REGISTERED PREDICTORS (10 cells CME, 8 Switch; nothing else is tried):
  CB_day k    intraday circuit breaker: block for the rest of the calendar day once gold is down more
              than k x D1 ATR(14, prior day) from the day's first open (k = 0.5, 1.0)
  CB_24h k    same on the trailing 24 H1 bars (k = 0.5, 1.0)
  HMM q       2-state Gaussian HMM on daily log returns, refit each January on all prior days,
              FORWARD-FILTERED P(low-mean state) at the prior close > q (q = 0.5, 0.7)
  CUSUM h     one-sided downward CUSUM of vol-scaled daily returns, drift 0.5, alarm below -h,
              block while in alarm (h = 3, 5)
  ML x        CME only: HistGB on 8 CME markets' 1/5/20-day returns + gold vol/z-scores, predicting
              a down day; yearly expanding walk-forward; block when P > the training-set quantile
              that blocks x of days (x = 30%, 50%)
Daily predictors use data through the PREVIOUS day's close only.

NULL (the decisive one): the SAME gate mask shifted by a random number of days (200 draws) — same
blocked share, same clustering, wrong days. p = share of shifted gates with Sharpe >= observed.
Family: max over cells vs max over the shifted replays. Also printed per gate: down-day rate on
blocked vs allowed days and mean next-day return (control #8: accuracy is not money).

    python scripts/v5_zigzag_fall_gate.py cme | switch
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_zigzag_long_history import OUT, SPREAD, SRC, THR, load, sharpe  # noqa: E402
from scripts.v5_zigzag_panel_probs import load_cme  # noqa: E402
from scripts.v5_zigzag_regime import bt  # noqa: E402

N_NULL = 200
CROSS = ("GC", "SI", "HG", "6E", "6J", "ZN", "ES", "CL")


# ---------------------------------------------------------------- daily series
def daily(g: pd.DataFrame) -> pd.DataFrame:
    d = g.resample("D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    tr = pd.concat([d.high - d.low, (d.high - d.close.shift()).abs(), (d.low - d.close.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["r"] = np.log(d.close).diff()
    return d


# ---------------------------------------------------------------- 2-state Gaussian HMM (EM + forward filter)
def hmm_fit(x: np.ndarray, iters: int = 60):
    mu = np.array([np.percentile(x, 25), np.percentile(x, 75)])
    sd = np.array([x.std(), x.std()])
    A = np.array([[0.95, 0.05], [0.05, 0.95]])
    pi = np.array([0.5, 0.5])
    for _ in range(iters):
        B = np.exp(-0.5 * ((x[:, None] - mu) / sd) ** 2) / sd + 1e-300
        n = len(x)
        al, c = np.zeros((n, 2)), np.zeros(n)
        al[0] = pi * B[0]
        c[0] = al[0].sum()
        al[0] /= c[0]
        for t in range(1, n):
            al[t] = (al[t - 1] @ A) * B[t]
            c[t] = al[t].sum()
            al[t] /= c[t]
        be = np.ones((n, 2))
        for t in range(n - 2, -1, -1):
            be[t] = (A @ (B[t + 1] * be[t + 1])) / c[t + 1]
        gm = al * be
        gm /= gm.sum(1, keepdims=True)
        xi = (al[:-1, :, None] * A[None] * (B[1:] * be[1:])[:, None, :]) / c[1:, None, None]
        A = xi.sum(0) / gm[:-1].sum(0)[:, None]
        pi = gm[0]
        mu = (gm * x[:, None]).sum(0) / gm.sum(0)
        sd = np.sqrt((gm * (x[:, None] - mu) ** 2).sum(0) / gm.sum(0)) + 1e-8
    return mu, sd, A, pi


def hmm_filter(x, mu, sd, A, pi):
    B = np.exp(-0.5 * ((x[:, None] - mu) / sd) ** 2) / sd + 1e-300
    al = np.zeros((len(x), 2))
    p = pi
    for t in range(len(x)):
        p = (p @ A if t else p) * B[t]
        p = p / p.sum()
        al[t] = p
    return al


def hmm_bear(d: pd.DataFrame, start) -> pd.Series:
    r = d.r.dropna()
    out = pd.Series(np.nan, index=r.index)
    for y in range(pd.Timestamp(start).year, r.index[-1].year + 1):
        tr = r[r.index < f"{y}-01-01"].to_numpy()
        mu, sd, A, pi = hmm_fit(tr)
        bear = int(np.argmin(mu))
        upto = r[r.index < f"{y + 1}-01-01"]
        f = hmm_filter(upto.to_numpy(), mu, sd, A, pi)[:, bear]   # filtered: uses returns <= t only
        sel = upto.index >= f"{y}-01-01"
        out[upto.index[sel]] = f[sel]
    return out


def cusum_alarm(d: pd.DataFrame, h: float, k: float = 0.5) -> pd.Series:
    z = (d.r / d.r.rolling(60).std().shift(1)).fillna(0).to_numpy()
    s, al = 0.0, np.zeros(len(z), bool)
    for t, v in enumerate(z):
        s = min(0.0, s + v + k)
        al[t] = s < -h
    return pd.Series(al, index=d.index)


def ml_fall(start) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415
    cl = pd.DataFrame({r: load_cme(r).close.resample("D").last() for r in CROSS}).dropna()
    lr = np.log(cl).diff()
    F = pd.DataFrame(index=cl.index)
    for r in CROSS:
        for w in (1, 5, 20):
            F[f"{r}_{w}"] = lr[r].rolling(w).sum()
    g = cl["GC"]
    F["gc_vol20"] = lr["GC"].rolling(20).std()
    F["gc_z20"] = (g - g.rolling(20).mean()) / g.rolling(20).std()
    F["gc_z50"] = (g - g.rolling(50).mean()) / g.rolling(50).std()
    y = (lr["GC"].shift(-1) < 0).astype(int)                    # tomorrow is a down day
    ok = F.notna().all(axis=1) & lr["GC"].shift(-1).notna()
    out = pd.DataFrame(index=F.index, columns=["p", "q30", "q50"], dtype=float)
    for yr in range(pd.Timestamp(start).year, F.index[-1].year + 1):
        tr = ok & (F.index < f"{yr}-01-01")
        tr &= np.arange(len(F)) < np.searchsorted(F.index, pd.Timestamp(f"{yr}-01-01")) - 1   # purge 1 day
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03, max_leaf_nodes=15,
                                             min_samples_leaf=40, random_state=0).fit(F[tr], y[tr])
        ptr = clf.predict_proba(F[tr])[:, 1]
        te = F.notna().all(axis=1) & (F.index >= f"{yr}-01-01") & (F.index < f"{yr + 1}-01-01")
        out.loc[te, "p"] = clf.predict_proba(F[te])[:, 1]
        out.loc[te, "q30"] = np.quantile(ptr, 0.70)
        out.loc[te, "q50"] = np.quantile(ptr, 0.50)
    return out                                                   # p at day t = P(day t+1 is down)


# ---------------------------------------------------------------- gates on the H1 index
def day_to_bars(flag_by_day: pd.Series, idx: pd.DatetimeIndex, lag_days: int) -> np.ndarray:
    """flag computed at the close of day d applies to every bar of day d + lag_days."""
    f = flag_by_day.astype(float).copy()
    f.index = f.index + pd.Timedelta(days=lag_days)
    return f.reindex(idx.normalize(), method="ffill").fillna(0).to_numpy() > 0.5


def circuit(g: pd.DataFrame, d: pd.DataFrame, k: float, mode: str) -> np.ndarray:
    atr_prev = d.atr.shift(1).reindex(g.index.normalize()).to_numpy()
    if mode == "day":
        first = g.groupby(g.index.normalize()).open.transform("first").to_numpy()
        move = g.close.to_numpy() - first
    else:
        move = g.close.to_numpy() - g.close.shift(24).to_numpy()
    blocked = move < -k * atr_prev                               # known at this bar's close
    if mode == "day":                                            # stays on for the rest of the day
        blocked = pd.Series(blocked, index=g.index).groupby(g.index.normalize()).cummax().to_numpy()
    return np.nan_to_num(blocked).astype(bool)


def main() -> None:
    src = sys.argv[1]
    cfg = SRC[src]
    start = pd.Timestamp(cfg["start"])
    out = OUT / src
    g = load(src, cfg["gold"])
    p = pd.read_csv(out / "probs.csv", index_col=0, parse_dates=True).iloc[:, 0].reindex(g.index).to_numpy()
    fires = np.isfinite(p) & (p >= THR)
    swl = cfg["swap"]
    z = np.zeros(len(g), bool)
    d = daily(g)
    down_today = (d.r < 0)

    def run(block, spread=SPREAD):
        return bt(g, fires & ~block, z, start, swl, 0.0, spread)

    tr0, base = run(z)
    s_base = sharpe(base)
    lines = [f"ZigZag fall-gate study, {src.upper()} gold H1, test {start.date()} -> {g.index[-1]}",
             f"V0 (no gate): trades {len(tr0)}, Sharpe {s_base:+.2f}; down-day base rate "
             f"{100 * down_today[d.index >= start].mean():.1f}%", "",
             "STEP 1 ORACLE: block entries on TRUE down days, flag correct with probability a"]
    rng = np.random.default_rng(5)
    for a in (1.0, 0.8, 0.7, 0.6, 0.55):
        flip = rng.random(len(d)) > a
        flag = down_today.to_numpy() ^ flip
        blk = day_to_bars(pd.Series(flag, index=d.index), g.index, 0)      # SAME day = foresight
        tr, r = run(blk)
        lines.append(f"  a={a:.2f}: blocked {100 * flag[d.index >= start].mean():.0f}% of days, trades {len(tr)}, "
                     f"Sharpe {sharpe(r):+.2f} (vs {s_base:+.2f})")

    # ---------------- predictors
    gates = {}
    for k in (0.5, 1.0):
        gates[f"CB_day {k}"] = circuit(g, d, k, "day")
        gates[f"CB_24h {k}"] = circuit(g, d, k, "24h")
    hb = hmm_bear(d, start)
    for q in (0.5, 0.7):
        gates[f"HMM {q}"] = day_to_bars(hb > q, g.index, 1)
    for h in (3, 5):
        gates[f"CUSUM {h}"] = day_to_bars(cusum_alarm(d, h), g.index, 1)
    if src == "cme":
        ml = ml_fall(start)
        for x, col in ((30, "q30"), (50, "q50")):
            gates[f"ML {x}%"] = day_to_bars(ml.p > ml[col], g.index, 1)

    test = g.index >= start
    nxt = d.r.shift(-1)
    rows, null_max = [], []
    nulls = {}
    s0 = int((~test).sum())
    shifts = rng.integers(20 * 24, test.sum() - 20 * 24, N_NULL)
    for name, blk in gates.items():
        tr, r = run(blk)
        s = sharpe(r)
        nl = []
        for sh in shifts:                                         # same gate, random days
            b2 = blk.copy()
            b2[s0:] = np.roll(blk[s0:], int(sh))
            nl.append(sharpe(run(b2)[1]))
        nulls[name] = np.array(nl)
        # day-level quality: share of blocked bar-time on down days (same-day for CB, next-day for daily)
        bd = pd.Series(blk[test], index=g.index[test]).groupby(g.index[test].normalize()).mean() > 0.5
        dd = down_today.reindex(bd.index)
        dr = d.r.reindex(bd.index)
        dd_ = (ret := r - base).to_numpy()
        rows.append(dict(gate=name, blocked_days=round(100 * bd.mean(), 0), trades=len(tr),
                         down_rate_blocked=round(100 * dd[bd].mean(), 1), down_rate_allowed=round(100 * dd[~bd].mean(), 1),
                         ret_bp_blocked=round(1e4 * dr[bd].mean(), 1), ret_bp_allowed=round(1e4 * dr[~bd].mean(), 1),
                         SR=round(s, 2), dSR=round(s - s_base, 2),
                         t_vs_V0=round(dd_.mean() / dd_.std() * np.sqrt(len(dd_)), 2) if dd_.std() > 0 else 0,
                         null_med=round(float(np.median(nl)), 2), p=round(float((np.array(nl) >= s).mean()), 3),
                         SR_2x=round(sharpe(run(blk, 2 * SPREAD)[1]), 2),
                         bear_13_15=round(sharpe(r["2013":"2015"]), 2) if src == "cme" else None,
                         y24_26=round(sharpe(r["2024":]), 2)))
        print(rows[-1], flush=True)
    A = pd.DataFrame(rows)
    mx = np.vstack(list(nulls.values())).max(axis=0)
    obs = A.SR.max()
    if src == "cme":
        ok = ml.p.notna() & nxt.reindex(ml.index).notna() & (ml.index >= start)
        from sklearn.metrics import roc_auc_score  # noqa: PLC0415
        auc = roc_auc_score((nxt.reindex(ml.index)[ok] < 0).astype(int), ml.p[ok])
        lines.append(f"\nML fall-day classifier out-of-sample AUC {auc:.3f} (0.5 = no skill)")
    lines += ["", "STEP 2 PREDICTORS (block new entries while flagged; null = same gate on random days)",
              A.to_string(index=False), "",
              f"family max-stat: best {A.gate[A.SR.idxmax()]} {obs:+.2f} vs shifted-gate max median "
              f"{np.median(mx):+.2f}, 95th {np.percentile(mx, 95):+.2f}, p {(mx >= obs).mean():.3f}",
              "down_rate_* = share of down days among blocked/allowed days (CB: same day; daily gates: the day they act on)"]
    txt = "\n".join(lines)
    print(txt)
    (out / "fall_gate_report.txt").write_text(txt + "\n")


if __name__ == "__main__":
    sys.exit(main())
