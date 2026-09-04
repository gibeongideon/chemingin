"""ROUND: advanced portfolio construction + the undeployed fast-reversal sleeve.

Everything else tried on this book has been a SIGNAL overlay, and all of it failed
(§3u/§3y/§3ab/§3ac/§3ad, 40+ configs). The repo's own evidence says the only lever that
ever worked is DIVERSIFICATION, and two things in that direction have never been tested:

  A. HOW THE SLEEVES ARE WEIGHTED. Every book here has been equal-weight or
     equal-class-risk. Grepping V5_FINDINGS + memory returns ZERO hits for hierarchical
     risk parity, random-matrix cleaning, CVaR optimisation or minimum variance. Given
     §3-shootout established that pass rate is decided by the DIAL-SCALED LEFT TAIL versus
     the daily-loss limit (not by Sharpe), a left-tail-aware weighting is directly on
     target rather than a generic optimisation.
  B. THE FAST REVERSAL SLEEVE. §3b found a fast ensemble at ~0.02 correlation to trend and
     measured "TREND+FAST 50/50 = 1.44 vs trend alone 1.17" — then never deployed it. Its
     overnight leg needs a live fill check (close->open gaps on 24h CFDs), but the
     REVERSAL leg was explicitly "cleanly tradeable close->close". Never combined with the
     current 4-sleeve book.

CONSTRAINTS CARRIED IN FROM PRIOR FAILURES (not rediscovered):
  * cross-sectional momentum as an overlay is REFUTED (t -1.51 over 960 trials) - not retried.
  * any cross-sectional/reversal signal must use SYNCHRONOUS-CLOSE instruments only; §3b
    caught a fake Sharpe 3.5 from NIKKEI/ASX closing hours before the US session.
  * weights are ESTIMATED, so every method is re-fit walk-forward on trailing data only.
    An in-sample HRP/min-var comparison would be meaningless.
  * gate = matched-vol paired t vs the equal-weight incumbent, plus split-sample, plus
    pass-rate simulation. DSR is not used as a gate (near-vacuous for variations of an
    already-good base).

    python scripts/v5_portfolio_construction.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "v5_runs" / "challenge-lab"))

from scripts.v5_volregime_taper_crossasset import engine_bp, load_sleeve  # noqa: E402
from scripts.v5_xau_champion_lifts import champion_recipe, sharpe, dd_of  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402
from scripts.v5_xau_turn_prob import paired, per_year  # noqa: E402
from challenge_lab import fp_sim  # noqa: E402

EVAL_START = "2018-01-01"
YEARS = list(range(2019, 2027))          # first year reserved for the initial trailing fit
TRAIL_DAYS = 756                         # 3y trailing window for weight estimation
SPLITS = [("2018-01-01", "2021-12-31", "2018-21"), ("2022-01-01", "2026-12-31", "2022-26")]

# trend sleeves: (path, ann, one-way bp, champion speed scale, vol halflife)
TREND = {
    "XAU":    ("data/XAUUSD_H4_long.csv", 252 * 6, 0.75, 1.0, 42),
    "BTC":    ("data/BTC_D1_long.csv", 252, 1.13, 1 / 6, 7),
    "NDX":    ("data/NDX_D1_long.csv", 252, 0.38, 1 / 6, 7),
    "BRENT":  ("data/BRENT_D1_long.csv", 252, 5.69, 1 / 6, 7),
    # broader trend candidates — under equal weight breadth DILUTED (1.67->1.04); the
    # open question is whether a smarter weighting changes that verdict
    "SILVER": ("data/SILVER_D1_long.csv", 252, 3.0, 1 / 6, 7),
    "DAX":    ("data/DAX_D1_long.csv", 252, 0.5, 1 / 6, 7),
    "ETH":    ("data/ETH_D1_long.csv", 252, 2.03, 1 / 6, 7),
    "COFFEE": ("data/COFFEE_D1_long.csv", 252, 4.0, 1 / 6, 7),
}
CORE = ["XAU", "BTC", "NDX", "BRENT"]      # the deployed book
US_SYNC = ["SPX", "NDX", "DJI"]            # synchronous closes only, per §3b's warning


def trend_sleeves() -> dict[str, pd.Series]:
    out = {}
    for name, (path, ann, bp, scale, hl) in TREND.items():
        df = load_sleeve(dict(path=path))
        if df is None:
            continue
        fc = (champion_signal(df["close"]) if scale == 1.0
              else champion_recipe(df["close"], scale, 1.5))
        r = engine_bp(df, fc, bp, ann, hl).loc[EVAL_START:]
        if len(r) > 200 and r.std() > 0:
            out[name] = r
    return out


def reversal_sleeve(cost_bp: float = 0.5) -> pd.Series | None:
    """§3b's tradeable leg: 1-day cross-sectional reversal on US-SYNCHRONOUS indices
    only (SPX/NDX/DJI all close 21:00 UTC), close->close, so the non-synchronous
    leakage that faked a Sharpe 3.5 cannot occur. Long the biggest 1-day loser,
    short the biggest gainer, vol-scaled, executed next close."""
    px = {}
    for s in US_SYNC:
        d = load_sleeve(dict(path=f"data/{s}_D1_long.csv"))
        if d is not None:
            px[s] = d["close"]
    if len(px) < 3:
        return None
    P = pd.DataFrame(px).dropna()
    R = P.pct_change()
    # rank yesterday's return cross-sectionally; fade it
    z = -R.sub(R.mean(axis=1), axis=0).div(R.std(axis=1).replace(0, np.nan), axis=0)
    w = z.div(z.abs().sum(axis=1).replace(0, np.nan), axis=0).shift(1)   # act next day
    vol = R.ewm(halflife=20, min_periods=20).std()
    w = (w / vol.mean(axis=1).replace(0, np.nan).values[:, None]) * (0.10 / np.sqrt(252))
    w = w.clip(-3, 3)
    gross = (w * R).sum(axis=1)
    turn = w.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turn * cost_bp * 1e-4
    return net.loc[EVAL_START:].dropna()


# ------------------------------------------------------------------ weighting methods
def w_equal(R: pd.DataFrame) -> np.ndarray:
    return np.ones(R.shape[1]) / R.shape[1]


def w_invvol(R: pd.DataFrame) -> np.ndarray:
    v = R.std().replace(0, np.nan)
    w = (1 / v).fillna(0.0).values
    return w / w.sum() if w.sum() > 0 else w_equal(R)


def w_hrp(R: pd.DataFrame) -> np.ndarray:
    """Lopez de Prado hierarchical risk parity: cluster on correlation distance,
    then allocate by recursive bisection with inverse-variance within clusters."""
    C = R.corr().fillna(0.0).values
    n = C.shape[0]
    if n < 2:
        return np.ones(n)
    d = np.sqrt(np.clip((1 - C) / 2, 0, 1))
    np.fill_diagonal(d, 0.0)
    try:
        Z = linkage(squareform(d, checks=False), method="single")
        order = list(leaves_list(Z))
    except Exception:
        return w_invvol(R)
    cov = R.cov().values
    w = np.ones(n)
    clusters = [order]
    while clusters:
        nxt = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            half = len(cl) // 2
            a, b = cl[:half], cl[half:]

            def cvar_of(sub):
                cv = cov[np.ix_(sub, sub)]
                iv = 1 / np.diag(cv)
                iv = iv / iv.sum()
                return float(iv @ cv @ iv)
            va, vb = cvar_of(a), cvar_of(b)
            alpha = 1 - va / (va + vb) if (va + vb) > 0 else 0.5
            w[a] *= alpha
            w[b] *= (1 - alpha)
            nxt += [a, b]
        clusters = nxt
    return w / w.sum()


def w_rmt_minvar(R: pd.DataFrame) -> np.ndarray:
    """Minimum variance on a Marchenko-Pastur CLEANED covariance: eigenvalues
    inside the MP noise band are replaced by their average, which is the standard
    RMT denoising step before any optimisation."""
    X = R.dropna()
    n, p = X.shape
    if n < p * 2 or p < 2:
        return w_invvol(R)
    corr = X.corr().fillna(0.0).values
    ev, EV = np.linalg.eigh(corr)
    q = n / p
    lam_max = (1 + 1 / np.sqrt(q)) ** 2                # MP upper edge (unit variance)
    noise = ev < lam_max
    if noise.any() and not noise.all():
        ev = ev.copy()
        ev[noise] = ev[noise].mean()
    corr_c = EV @ np.diag(ev) @ EV.T
    dd = np.sqrt(np.diag(X.cov().values))
    cov_c = corr_c * np.outer(dd, dd)
    try:
        inv = np.linalg.pinv(cov_c)
        w = inv @ np.ones(p)
        w = np.clip(w, 0, None)                        # long-only in sleeve space
        return w / w.sum() if w.sum() > 0 else w_invvol(R)
    except Exception:
        return w_invvol(R)


def w_mincvar(R: pd.DataFrame, alpha: float = 0.05) -> np.ndarray:
    """Minimise the 5% CVaR of the sleeve mix — the objective that matches how
    prop-firm failure actually happens (a bad left-tail day versus a fixed daily
    limit), rather than minimising variance."""
    X = R.dropna().values
    p = X.shape[1]
    if X.shape[0] < 100 or p < 2:
        return w_invvol(R)

    def obj(w):
        r = X @ w
        k = max(int(len(r) * alpha), 5)
        return -np.mean(np.sort(r)[:k])                # CVaR of the losses

    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
    bnds = [(0.0, 0.6)] * p                            # no sleeve above 60%
    try:
        res = minimize(obj, w_equal(R), method="SLSQP", bounds=bnds, constraints=cons,
                       options=dict(maxiter=200, ftol=1e-9))
        if res.success and res.x.sum() > 0:
            return np.clip(res.x, 0, None) / np.clip(res.x, 0, None).sum()
    except Exception:
        pass
    return w_invvol(R)


METHODS = {"equal (incumbent)": w_equal, "inverse-vol": w_invvol, "HRP": w_hrp,
           "RMT-cleaned min-var": w_rmt_minvar, "min-CVaR 5%": w_mincvar}


def walk_forward_book(R: pd.DataFrame, method) -> tuple[pd.Series, pd.DataFrame]:
    """Re-fit weights each January on the trailing 3 years only, apply OOS."""
    parts, wlog = [], {}
    for yr in YEARS:
        cut = pd.Timestamp(f"{yr}-01-01")
        train = R.loc[:cut].iloc[-TRAIL_DAYS:].dropna(how="all")
        cols = [c for c in R.columns if train[c].notna().sum() > 250 and train[c].std() > 0]
        if len(cols) < 2:
            continue
        w = method(train[cols])
        w = np.nan_to_num(w)
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        wlog[yr] = pd.Series(w, index=cols)
        test = R.loc[R.index.year == yr, cols].fillna(0.0)
        if len(test):
            parts.append(test.values @ w)
            parts[-1] = pd.Series(parts[-1], index=test.index)
    if not parts:
        return pd.Series(dtype=float), pd.DataFrame()
    return pd.concat(parts).sort_index(), pd.DataFrame(wlog).T.fillna(0.0)


def tail_report(b: pd.Series, dial: float, daily_limit: float) -> str:
    """The metric that actually decides pass/fail: worst day scaled to the account
    dial, with the 1.5x intraday-floating proxy, versus the firm's daily limit."""
    k = dial / (b.std() * np.sqrt(252)) if b.std() > 0 else 1.0
    worst = b.min() * k * 1.5
    q01 = b.quantile(0.01) * k * 1.5
    flag = "BREACH" if worst < -daily_limit else "ok"
    return f"worst {worst*100:+.2f}% q01 {q01*100:+.2f}% vs -{daily_limit*100:.0f}% {flag}"


def main() -> None:
    T = trend_sleeves()
    print(f"trend sleeves loaded: {list(T)}")
    rev = reversal_sleeve()
    print(f"reversal sleeve: {'loaded, n=' + str(len(rev)) if rev is not None else 'UNAVAILABLE'}"
          + (f", standalone Sharpe {sharpe(rev):+.3f}, corr to core "
             f"{pd.concat([rev, pd.DataFrame({k: T[k] for k in CORE}).mean(axis=1)], axis=1).dropna().corr().iloc[0,1]:+.3f}"
             if rev is not None else ""))

    core = pd.DataFrame({k: T[k] for k in CORE if k in T}).dropna(how="all")
    wide = pd.DataFrame(T).dropna(how="all")
    print(f"\ncore book {list(core.columns)}  |  wide book {list(wide.columns)}")

    incumbent, _ = walk_forward_book(core, w_equal)
    print(f"\nINCUMBENT (core, equal, walk-forward): Sharpe {sharpe(incumbent):+.3f}  "
          f"DD {dd_of(incumbent):+.1f}%")

    universes = {"core 4": core, "wide 8": wide}
    if rev is not None:
        c2 = core.copy(); c2["REVERSAL"] = rev
        w2 = wide.copy(); w2["REVERSAL"] = rev
        universes["core 4 + reversal"] = c2.dropna(how="all")
        universes["wide 8 + reversal"] = w2.dropna(how="all")

    rows = []
    for uname, U in universes.items():
        print(f"\n{'='*104}\nUNIVERSE: {uname}  ({U.shape[1]} sleeves)\n{'='*104}")
        print(f"{'method':22} {'Sharpe':>8} {'DD':>7} {'Calmar':>7} {'t@matched':>10} "
              f"{'yrs':>5}  {'FTMO tail (dial 9%, limit 5%)':<44}")
        for mname, fn in METHODS.items():
            b, wl = walk_forward_book(U, fn)
            if b.empty or b.std() == 0:
                continue
            j = b.index.intersection(incumbent.index)
            bb, ii = b.reindex(j), incumbent.reindex(j)
            bm = bb * (ii.std() / bb.std()) if bb.std() > 0 else bb
            _, t, _ = paired(bm, ii)
            yp, yn = per_year(bm, ii)
            yrs = (b.index[-1] - b.index[0]).days / 365.25
            cagr = ((1 + b).prod() ** (1 / yrs) - 1) * 100
            rows.append(dict(universe=uname, method=mname, sharpe=sharpe(b), dd=dd_of(b),
                             cagr=cagr, t=t, yrs=f"{yp}/{yn}", series=b, weights=wl))
            print(f"{mname:22} {sharpe(b):>+8.3f} {dd_of(b):>+7.1f} "
                  f"{cagr/abs(dd_of(b)):>7.3f} {t:>+10.2f} {yp}/{yn:<3} "
                  f"{tail_report(b, 0.09, 0.05):<44}")

    R = pd.DataFrame([{k: v for k, v in r.items() if k not in ("series", "weights")}
                      for r in rows])
    print(f"\n{'='*104}\nGATE: matched-vol t vs the equal-weight core incumbent > +1.64\n{'='*104}")
    win = R[R.t > 1.64].sort_values("t", ascending=False)
    print(f"candidates passing: {len(win)}/{len(R)}")
    for _, w in win.iterrows():
        print(f"  {w.universe:20} {w.method:22} Sharpe {w.sharpe:+.3f} t {w.t:+.2f} yrs {w.yrs}")

    print(f"\n{'='*104}\nPASS SIMULATION (FTMO 2-Step @9%, FP Flex @7%)\n{'='*104}")
    print(f"{'universe':20} {'method':22} {'FTMO pass':>10} {'fail_day':>9} "
          f"{'FP pass':>9} {'fail_day':>9}")
    best = R.sort_values("sharpe", ascending=False).head(8)
    for _, r0 in best.iterrows():
        b = [x["series"] for x in rows
             if x["universe"] == r0.universe and x["method"] == r0.method][0]
        out = []
        for dial, rc in ((0.09, dict(p1=.10, p2=.05, dayloss=.05, maxloss=.10)),
                         (0.07, dict(p1=.10, p2=.06, dayloss=.04, maxloss=.12))):
            k = dial / (b.std() * np.sqrt(252))
            s = fp_sim(b.values, k, day_safety=1.5, **rc)
            out += [s["passpct"], s["fail_day"]]
        print(f"{r0.universe:20} {r0.method:22} {out[0]:>10.1f} {out[1]:>9.1f} "
              f"{out[2]:>9.1f} {out[3]:>9.1f}")

    print(f"\n{'='*104}\nSPLIT-SAMPLE for anything that passed the gate\n{'='*104}")
    if len(win) == 0:
        print("  nothing passed — nothing to split-test.")
    for _, w in win.iterrows():
        b = [x["series"] for x in rows
             if x["universe"] == w.universe and x["method"] == w.method][0]
        line = []
        for s0, s1, lbl in SPLITS:
            seg, inc = b.loc[s0:s1], incumbent.loc[s0:s1]
            if len(seg) > 60 and seg.std() > 0:
                sm = seg * (inc.std() / seg.std())
                _, tt, _ = paired(sm, inc)
                line.append(f"{lbl}: SR {sharpe(seg):+.3f} (inc {sharpe(inc):+.3f}) t{tt:+.2f}")
        print(f"  {w.universe} / {w.method}: " + "   |   ".join(line))

    out = ROOT / "data" / "v5_runs" / "portfolio_construction.csv"
    R.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
