"""PRE-REGISTERED: BOOK-LEVEL risk overlays — the explicit open question left by §3ac.

§3ac closed the high-vol-taper line with a recorded methodological lesson: *test overlays at
BOOK level, because diversification already does the smoothing a per-sleeve taper was trying
to do*. Every overlay tried since has still been per-sleeve. This round tests the four
book-level overlays that follow from that lesson, none of which exists anywhere in the repo:

  O1 VOL MANAGEMENT (Moreira-Muir 2017): scale total exposure by target / trailing REALISED
     BOOK vol. Distinct from what already runs: each sleeve is vol-targeted individually, so
     the BOOK's vol still swings with the correlation regime and is not controlled at all.
  O2 PORTFOLIO VOL TARGETING (primary): the same idea done properly - predict book vol from a
     causal EWMA COVARIANCE matrix (w'Sw) instead of from realised book returns, so the
     control reacts to a correlation spike immediately rather than after the damage.
  O3 DRAWDOWN-RESPONSIVE SIZING: de-risk in proportion to the book's own current drawdown.
  O4 CRISIS-CORRELATION GATE: cut size when average pairwise sleeve correlation spikes, i.e.
     precisely when the diversification the whole book depends on has stopped working.

PRE-REGISTRATION (fixed before any result):
  * PRIMARY = O2. O1/O3/O4 are secondary and reported in full; 11 cells total, all printed.
  * GATE = matched vol (§3ad) + the standard 2018-21 / 2022-26 split. KEEPER BAR:
    t@matched >= +1.50 AND both half-deltas positive.
  * Because these overlays exist to change the SHAPE of the equity curve, drawdown and the
    prop-firm pass rate are reported as well - but the keeper decision rests on the
    pre-registered gate, not on whichever secondary metric happens to look best.
  * Baseline is the deployed equal-weight core-4 book, identical construction to §3af.

    python scripts/v5_book_overlays.py
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
sys.path.insert(0, str(ROOT / "data" / "v5_runs" / "challenge-lab"))

from scripts.v5_volregime_taper_crossasset import engine_bp, load_sleeve  # noqa: E402
from scripts.v5_xau_champion_lifts import champion_recipe, vol_match, sharpe, dd_of  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402
from scripts.v5_xau_turn_prob import paired, per_year  # noqa: E402

EVAL_START = "2018-01-01"
SPLITS = [("2018-01-01", "2021-12-31", "2018-21"), ("2022-01-01", "2026-12-31", "2022-26")]
TREND_CORE = {
    "XAU":   ("data/XAUUSD_H4_long.csv", 252 * 6, 0.75, 1.0, 42),
    "BTC":   ("data/BTC_D1_long.csv", 252, 1.13, 1 / 6, 7),
    "NDX":   ("data/NDX_D1_long.csv", 252, 0.38, 1 / 6, 7),
    "BRENT": ("data/BRENT_D1_long.csv", 252, 5.69, 1 / 6, 7),
}


def sleeve_returns() -> pd.DataFrame:
    out = {}
    for name, (path, ann, bp, scale, hl) in TREND_CORE.items():
        df = load_sleeve(dict(path=path))
        fc = champion_signal(df["close"]) if scale == 1.0 else champion_recipe(df["close"], scale, 1.5)
        out[name] = engine_bp(df, fc, bp, ann, hl).loc[EVAL_START:]
    return pd.DataFrame(out).dropna(how="all").fillna(0.0)


# ------------------------------------------------------------------ overlays
def o1_vol_managed(R: pd.DataFrame, hl: int, target: float) -> pd.Series:
    b = R.mean(axis=1)
    rv = b.ewm(halflife=hl, min_periods=40).std() * np.sqrt(252)
    return (target / rv.replace(0, np.nan)).clip(0, 3).shift(1).fillna(1.0)


def o2_portfolio_vt(R: pd.DataFrame, hl: int, target: float) -> pd.Series:
    """Causal EWMA covariance -> predicted book vol -> exposure scalar. Reacts to a
    correlation spike on the day it happens, not after it shows up in realised vol."""
    n = R.shape[1]
    w = np.ones(n) / n
    X = R.values
    lam = 0.5 ** (1.0 / hl)
    S = np.cov(X[:60].T) if len(X) > 60 else np.eye(n) * 1e-4
    pred = np.full(len(X), np.nan)
    for i in range(len(X)):
        x = X[i]
        pred[i] = np.sqrt(max(w @ S @ w, 1e-12) * 252)
        S = lam * S + (1 - lam) * np.outer(x, x)
    s = pd.Series(target / pred, index=R.index).clip(0, 3).shift(1).fillna(1.0)
    return s


def o3_drawdown(R: pd.DataFrame, k: float, floor: float) -> pd.Series:
    b = R.mean(axis=1)
    e = (1 + b).cumprod()
    dd = (e / e.cummax() - 1.0)
    return (1.0 + k * dd).clip(floor, 1.0).shift(1).fillna(1.0)


def o4_corr_gate(R: pd.DataFrame, hl: int, q: float, lo: float) -> pd.Series:
    """Average pairwise correlation from the same causal EWMA covariance, ranked against
    its own trailing 2y history; cut size in the top `1-q` of the distribution."""
    n = R.shape[1]
    X = R.values
    lam = 0.5 ** (1.0 / hl)
    S = np.cov(X[:60].T) if len(X) > 60 else np.eye(n) * 1e-4
    ac = np.full(len(X), np.nan)
    iu = np.triu_indices(n, 1)
    for i in range(len(X)):
        d = np.sqrt(np.clip(np.diag(S), 1e-16, None))
        C = S / np.outer(d, d)
        ac[i] = np.nanmean(np.abs(C[iu]))
        S = lam * S + (1 - lam) * np.outer(X[i], X[i])
    a = pd.Series(ac, index=R.index)
    p = a.rolling(504, min_periods=120).apply(lambda x: (x[-1] > x).mean(), raw=True).fillna(0.5)
    return pd.Series(np.where(p >= q, lo, 1.0), index=R.index).shift(1).fillna(1.0)


# ------------------------------------------------------------------ evaluation
def halves(r: pd.Series) -> list[float]:
    return [sharpe(r.loc[a:b]) for a, b, _ in SPLITS]


def gate(c: pd.Series, b: pd.Series) -> tuple[float, str, list[float]]:
    i = c.index.intersection(b.index)
    c, b = c.loc[i], b.loc[i]
    _, t, _ = paired(vol_match(c, b), b)
    yp, yn = per_year(vol_match(c, b), b)
    dh = [sharpe(vol_match(c.loc[a:x], b.loc[a:x])) - sharpe(b.loc[a:x]) for a, x, _ in SPLITS]
    return t, f"{yp}/{yn}", dh


def main() -> None:
    R = sleeve_returns()
    base = R.mean(axis=1)
    hb = halves(base)
    rv = base.std() * np.sqrt(252)
    print(f"core-4 book {R.index.min().date()}->{R.index.max().date()}  "
          f"realised vol {rv:.2%}")
    print(f"BASELINE: SR {sharpe(base):+.3f}  DD {dd_of(base):+.2f}%  "
          f"halves {hb[0]:+.2f}/{hb[1]:+.2f}")
    print("keeper bar: t@matched>=+1.50 AND both half-deltas positive\n")

    cands: dict[str, pd.Series] = {}
    for hl in (20, 60, 120):
        cands[f"O1 vol-managed hl={hl}d"] = o1_vol_managed(R, hl, rv)
    for hl in (20, 60, 120):
        cands[f"O2 portfolio-VT hl={hl}d (PRIMARY)"] = o2_portfolio_vt(R, hl, rv)
    for k, fl in ((3.0, 0.4), (6.0, 0.4)):
        cands[f"O3 drawdown k={k} floor={fl}"] = o3_drawdown(R, k, fl)
    for q, lo in ((0.80, 0.5), (0.90, 0.5), (0.90, 0.0)):
        cands[f"O4 corr-gate q={q} lo={lo}"] = o4_corr_gate(R, 60, q, lo)

    rows = []
    for tag, s in cands.items():
        r = (base * s).dropna()
        t, yy, dh = gate(r, base)
        h = halves(r)
        keep = t >= 1.50 and min(dh) > 0
        print(f"  {tag:34s} SR {sharpe(r):+.3f}  DD {dd_of(r):+6.2f}%  "
              f"halves {h[0]:+.2f}/{h[1]:+.2f}  t@matched {t:+.2f} [{yy}]  "
              f"dSR {dh[0]:+.2f}/{dh[1]:+.2f}  exposure {s.mean():.2f}x  "
              f"{'** KEEPER **' if keep else ''}")
        rows.append(dict(tag=tag, sr=sharpe(r), dd=dd_of(r), h1=h[0], h2=h[1], t=t,
                         yy=yy, d1=dh[0], d2=dh[1], expo=s.mean(), keep=keep))

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data" / "v5_runs" / "book_overlays.csv", index=False)
    best = out.loc[out.t.idxmax()]
    print(f"\ntrials {len(out)}   keepers {int(out['keep'].sum())}   "
          f"best t {best.tag} ({best.t:+.2f}, SR {best.sr:+.3f} vs {sharpe(base):+.3f})")
    print("-> data/v5_runs/book_overlays.csv")


if __name__ == "__main__":
    main()
