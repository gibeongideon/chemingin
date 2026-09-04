"""PRE-REGISTERED: FINANCING-AWARE BOOK — the lever §3q named and then never pulled.

§3q measured financing properly for the first time and its own closing verdict was explicit:
"Financing is now the single largest lever in the book (-0.29 Sharpe) and the only untested
way to move it is EXECUTION ... not signal research." It then treated the -3.79%/yr as a
POST-HOC correction and kept the book byte-identical. That leaves the obvious thing undone:
every weighting decision this repo has ever made - including §3af's HRP/RMT/CVaR shootout -
was made on GROSS-of-financing sleeve returns, while the four sleeves carry wildly different
carry: XAU -6.6%/yr, BTC -30.4%/yr, NDX -6.9%/yr, BRENT **+9.6%/yr** (live FTMO rates, §3q).

Why this is different from §3af, and why optimisation might win here when it lost there:
§3af's methods all had to ESTIMATE expected returns or a covariance-implied tilt, and
estimation error swamped the gain (the classic 1/N result). Financing is not estimated - it is
a KNOWN, contractual, deterministic component of return, published per symbol. Tilting against
a known drag carries no estimation error at all. So the test is set up to add exactly that one
piece of information and nothing else:

    mu_i = a - f_i          a = one COMMON alpha for every sleeve (the 1/N prior: no
                                sleeve is assumed better than another), f_i = its known
                                financing drag. Sigma_i is re-fit walk-forward. No expected
                                return is ever estimated per sleeve.

Second question, same root cause: `xau-longonly-champion` established "kill the shorts" as THE
lever for XAU - but that was decided GROSS. Maven's live rates are swap_long -45.33 /
swap_short +25.25 per lot per night: a short EARNS carry. The swing between the two legs is
~5.9%/yr on a book targeting ~10% vol, i.e. up to ~0.6 Sharpe units on the short leg's
exposure. Whether carry flips the long-only verdict has never been checked.

PRE-REGISTRATION (fixed before any result):
  * Short-side rates use the ONE live measurement available: Maven XAU short receives 55.7% of
    what long pays (25.25/45.33), applied with the opposite sign to every sleeve. Stated as an
    assumption, not a measurement, for the three non-XAU sleeves.
  * `a` is a free constant, so its full sensitivity is reported: a in {4,6,8,10}%/yr. A result
    that only holds at one value of `a` is reported as fragile, not as a finding.
  * Sleeve weights are long-only, sum to 1, capped at 0.50 so no single sleeve can take over;
    Sigma is re-fit every January on the trailing 756 days (identical protocol to §3af).
  * GATE: matched-vol paired-t vs the NET equal-weight core-4 book + the standard split.
    KEEPER BAR: t@matched >= +1.50 with both half-deltas positive.
  * Financing is charged on the engine's own time-varying notional, per bar, in the sleeve's
    own annualisation - not as a flat end-of-run haircut.

    python scripts/v5_financing_aware_book.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_volregime_taper_crossasset import load_sleeve, TARGET_VOL, MAX_LEV, BUFFER  # noqa: E402
from scripts.v5_xau_champion_lifts import champion_recipe, vol_match, sharpe, dd_of  # noqa: E402
from src.v5.xau_dual_signals import (champion_signal, ewmac_fc, breakout_fc,  # noqa: E402
                                     EWMAC_MID, BKO_FAST)
from scripts.v5_xau_turn_prob import paired, per_year  # noqa: E402

EVAL_START = "2018-01-01"
SPLITS = [("2018-01-01", "2021-12-31", "2018-21"), ("2022-01-01", "2026-12-31", "2022-26")]
YEARS = list(range(2019, 2027))
TRAIL = 756
SHORT_HAIRCUT = 25.25 / 45.33        # a short receives 55.7% of what a long pays (Maven, live)

# name -> (path, ann, one-way bp, champion speed scale, vol halflife, LONG financing %/yr)
SLEEVES = {
    "XAU":   ("data/XAUUSD_H4_long.csv", 252 * 6, 0.75, 1.0, 42, -0.066),
    "BTC":   ("data/BTC_D1_long.csv", 252, 1.13, 1 / 6, 7, -0.304),
    "NDX":   ("data/NDX_D1_long.csv", 252, 0.38, 1 / 6, 7, -0.069),
    "BRENT": ("data/BRENT_D1_long.csv", 252, 5.69, 1 / 6, 7, +0.096),
}


def engine_fin(df: pd.DataFrame, fc: pd.Series, cost_bp: float, ann: int, hl: int,
               rate_long: float, charge_financing: bool = True) -> pd.Series:
    """The standard engine plus a per-bar financing accrual on the ACTUAL notional.
    A long pays `rate_long`/yr; a short receives -rate_long * SHORT_HAIRCUT."""
    close = df["close"]
    ret = close.pct_change()
    vol = ret.ewm(halflife=hl, min_periods=20).std() * np.sqrt(ann)
    pos = (fc.clip(-2.0, 2.0) * (TARGET_VOL / vol)).clip(-MAX_LEV, MAX_LEV)
    band = (BUFFER * (TARGET_VOL / vol).clip(0, MAX_LEV)).values
    p, out, held = pos.values.copy(), np.zeros(len(pos)), 0.0
    for i in range(len(p)):
        if np.isfinite(p[i]):
            b = band[i] if np.isfinite(band[i]) else 0.0
            if abs(p[i] - held) > b:
                held = p[i] - np.sign(p[i] - held) * b
        out[i] = held
    pos = pd.Series(out, index=pos.index).shift(1).fillna(0.0)
    net = (pos * ret - pos.diff().abs().fillna(0.0) * (cost_bp * 1e-4))
    if charge_financing:
        rate_short = -rate_long * SHORT_HAIRCUT
        fin = np.where(pos > 0, pos * rate_long, -pos * rate_short) / ann
        net = net + pd.Series(fin, index=pos.index)
    eq = (1.0 + net.fillna(0.0)).cumprod().loc[EVAL_START:]
    if eq.empty:
        return pd.Series(dtype=float)
    eq = eq / eq.iloc[0]
    return eq.resample("D").last().pct_change(fill_method=None).dropna()


def ls_forecast(px: pd.Series, scale: float) -> pd.Series:
    """Symmetric (long AND short) version of the champion's own components."""
    ewm = tuple((max(2, int(round(f * scale))), max(3, int(round(s * scale))))
                for f, s in EWMAC_MID)
    bko = tuple(max(3, int(round(w * scale))) for w in BKO_FAST)
    return (0.5 * ewmac_fc(px, ewm) + 0.5 * breakout_fc(px, bko)).clip(-2, 2)


def build(charge: bool = True) -> dict[str, pd.Series]:
    out = {}
    for name, (path, ann, bp, scale, hl, rate) in SLEEVES.items():
        df = load_sleeve(dict(path=path))
        fc = champion_signal(df["close"]) if scale == 1.0 else champion_recipe(df["close"], scale, 1.5)
        out[name] = engine_fin(df, fc, bp, ann, hl, rate, charge)
    return out


def halves(r: pd.Series) -> list[float]:
    return [sharpe(r.loc[a:b]) for a, b, _ in SPLITS]


def gate(c: pd.Series, b: pd.Series) -> tuple[float, str, list[float]]:
    i = c.index.intersection(b.index)
    c, b = c.loc[i], b.loc[i]
    _, t, _ = paired(vol_match(c, b), b)
    yp, yn = per_year(vol_match(c, b), b)
    dh = [sharpe(vol_match(c.loc[a:x], b.loc[a:x])) - sharpe(b.loc[a:x]) for a, x, _ in SPLITS]
    return t, f"{yp}/{yn}", dh


def max_sharpe_w(mu: np.ndarray, S: np.ndarray, cap: float = 0.50) -> np.ndarray:
    n = len(mu)
    x0 = np.ones(n) / n

    def neg(w):
        v = np.sqrt(max(w @ S @ w, 1e-16))
        return -(w @ mu) / v

    r = minimize(neg, x0, method="SLSQP", bounds=[(0.0, cap)] * n,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                 options=dict(maxiter=400, ftol=1e-10))
    w = r.x if r.success else x0
    w = np.clip(w, 0, cap)
    return w / w.sum()


def wf_book(R: pd.DataFrame, mu: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Weights re-fit every January from the trailing 756 days of covariance; mu is the
    KNOWN financing tilt and never estimated from returns."""
    segs, wlog = [], {}
    for y in YEARS:
        tr = R.loc[:f"{y - 1}-12-31"].tail(TRAIL)
        te = R.loc[f"{y}-01-01":f"{y}-12-31"]
        if len(tr) < 252 or te.empty:
            continue
        S = tr.cov().values * 252
        w = max_sharpe_w(mu.reindex(R.columns).values, S)
        wlog[y] = pd.Series(w, index=R.columns)
        segs.append(te.values @ w)
    if not segs:
        return pd.Series(dtype=float), pd.DataFrame()
    idx = R.loc[f"{YEARS[0]}-01-01":].index[:sum(len(s) for s in segs)]
    return pd.Series(np.concatenate(segs)[:len(idx)], index=idx), pd.DataFrame(wlog).T


def main() -> None:
    gross = build(charge=False)
    net = build(charge=True)
    Rg = pd.DataFrame(gross).dropna(how="all").fillna(0.0)
    Rn = pd.DataFrame(net).dropna(how="all").fillna(0.0)
    bg, bn = Rg.mean(axis=1), Rn.mean(axis=1)

    print("--- per-sleeve financing impact (charged on the engine's own notional) ---")
    for k in SLEEVES:
        print(f"  {k:6s} rate {SLEEVES[k][5]:+6.1%}/yr   gross SR {sharpe(Rg[k]):+.3f}"
              f" -> net {sharpe(Rn[k]):+.3f}   drag {(Rn[k].mean() - Rg[k].mean()) * 252:+.2%}/yr")
    print(f"\nEQUAL-WEIGHT book: gross SR {sharpe(bg):+.3f} -> NET {sharpe(bn):+.3f}   "
          f"DD {dd_of(bn):+.2f}%   halves {halves(bn)[0]:+.2f}/{halves(bn)[1]:+.2f}")
    print(f"realised drag {(bn.mean() - bg.mean()) * 252:+.2%}/yr "
          f"(§3q reported -3.79%/yr)\n")

    print("--- Q1: financing-aware sleeve weights (mu = common alpha a MINUS known drag) ---")
    print("keeper bar: t@matched>=+1.50 AND both half-deltas positive")
    rows = []
    f = pd.Series({k: -v[5] for k, v in SLEEVES.items()})       # drag as a positive cost
    for a in (0.04, 0.06, 0.08, 0.10):
        mu = a - f
        bk, W = wf_book(Rn, mu)
        if bk.empty:
            continue
        ref = Rn.loc[bk.index].mean(axis=1)
        t, yy, dh = gate(bk, ref)
        keep = t >= 1.50 and min(dh) > 0
        wm = W.mean()
        print(f"  a={a:.0%}  weights " + " ".join(f"{k} {wm[k]:.2f}" for k in Rn.columns) +
              f"   SR {sharpe(bk):+.3f} (equal {sharpe(ref):+.3f})  DD {dd_of(bk):+.2f}%  "
              f"t@matched {t:+.2f} [{yy}]  dSR {dh[0]:+.2f}/{dh[1]:+.2f}  "
              f"{'** KEEPER **' if keep else ''}")
        rows.append(dict(test=f"fin-aware a={a:.0%}", sr=sharpe(bk), t=t, d1=dh[0], d2=dh[1], keep=keep))

    print("\n--- Q1b: simple, deployable weight tilts (no optimiser) ---")
    ref = Rn.mean(axis=1)
    tilts = {
        "drop BTC (worst carry -30.4%)": {"XAU": 1 / 3, "NDX": 1 / 3, "BRENT": 1 / 3, "BTC": 0.0},
        "half BTC, rest equal":          {"XAU": 0.2857, "NDX": 0.2857, "BRENT": 0.2857, "BTC": 0.1429},
        "double BRENT (+9.6% carry)":    {"XAU": 0.2, "BTC": 0.2, "NDX": 0.2, "BRENT": 0.4},
        "inverse-drag weights":          None,
    }
    inv = (1.0 / (f + 0.35)); inv = inv / inv.sum()
    tilts["inverse-drag weights"] = inv.to_dict()
    for tag, w in tilts.items():
        wv = pd.Series(w).reindex(Rn.columns).fillna(0.0)
        bk = (Rn * wv).sum(axis=1)
        t, yy, dh = gate(bk, ref)
        keep = t >= 1.50 and min(dh) > 0
        print(f"  {tag:32s} " + " ".join(f"{k} {wv[k]:.2f}" for k in Rn.columns) +
              f"   SR {sharpe(bk):+.3f}  DD {dd_of(bk):+.2f}%  t@matched {t:+.2f} [{yy}]  "
              f"dSR {dh[0]:+.2f}/{dh[1]:+.2f}  {'** KEEPER **' if keep else ''}")
        rows.append(dict(test=tag, sr=sharpe(bk), t=t, d1=dh[0], d2=dh[1], keep=keep))

    print("\n--- Q2: does the +swap on shorts flip 'kill the shorts' on XAU? (net of swap) ---")
    dx = load_sleeve(dict(path=SLEEVES["XAU"][0]))
    lo_fc = champion_signal(dx["close"])
    ls_fc = ls_forecast(dx["close"], 1.0)
    variants = {"long-only champion (deployed)": lo_fc, "symmetric long/short": ls_fc}
    for k in (0.25, 0.50, 1.00):
        variants[f"long-only + carry shorts k={k:.2f}"] = (lo_fc + k * ls_fc.clip(upper=0.0))
    xref = None
    for tag, fc in variants.items():
        r = engine_fin(dx, fc, 0.75, 252 * 6, 42, SLEEVES["XAU"][5], True)
        g = engine_fin(dx, fc, 0.75, 252 * 6, 42, SLEEVES["XAU"][5], False)
        if xref is None:
            xref = r
        t, yy, dh = gate(r, xref)
        print(f"  {tag:34s} gross SR {sharpe(g):+.3f}  NET SR {sharpe(r):+.3f}  "
              f"DD {dd_of(r):+7.2f}%  halves {halves(r)[0]:+.2f}/{halves(r)[1]:+.2f}  "
              f"t@matched {t:+.2f} [{yy}]")
        rows.append(dict(test="XAU " + tag, sr=sharpe(r), t=t, d1=dh[0], d2=dh[1],
                         keep=bool(t >= 1.50 and min(dh) > 0)))

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data" / "v5_runs" / "financing_aware_book.csv", index=False)
    print(f"\ntrials {len(out)}   keepers {int(out['keep'].sum())}")
    print("-> data/v5_runs/financing_aware_book.csv")


if __name__ == "__main__":
    main()
