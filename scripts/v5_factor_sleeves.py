"""PRE-REGISTERED: the canonical NON-TREND factors this repo has never implemented.

Everything that ever worked on this book was DIVERSIFICATION; everything that failed was a
signal overlay on the same trend driver. §3ae also showed adding more *trend* assets dilutes
(1.67 -> 1.04) and §3af showed no weighting scheme beats equal weight. That leaves exactly one
untried direction: a sleeve with a STRUCTURALLY DIFFERENT return driver.

Repo audit (grep, 2026-09-03): `src/cta/signals.py` implements tsmom, xsmom, ewmac and fx_carry.
It has never implemented any of:
  * VALUE  — long-horizon (5y) mean reversion. The canonical third factor alongside
             momentum and carry (Asness-Moskowitz-Pedersen, "Value and Momentum Everywhere",
             JF 2013); by construction NEGATIVELY correlated with 1y momentum, so it is the
             textbook trend diversifier. Tested both time-series and cross-sectionally.
  * BAB    — betting-against-beta / low-vol (Frazzini-Pedersen 2014), cross-sectional.
  * SEASONALITY — calendar-month effects, expanding-window causal (commodity storage /
             harvest cycles; Levi-Welch). Distinct from the already-disproven XAU weekly cycle
             (§3z), which was one asset and one intra-week schedule.
  * SKEW   — cross-sectional lottery-demand (Bali-Cakici-Whitelaw). §3ad tested a skew
             MULTIPLIER on XAU's own forecast; a cross-sectional skew SLEEVE is a different
             object.

PRE-REGISTRATION (fixed before running; nothing below was tuned):
  * Parameters are FIXED A PRIORI from the literature, not searched: value = 5y (1260d)
    log price ratio; BAB/skew = 252d; seasonality needs >=5 prior years. No grid, therefore
    no selection bias and no DSR/PBO needed - the full-sample number IS the honest number.
    Only ONE secondary robustness variant is reported (3y value) and it cannot change the
    verdict.
  * PRIMARY claim is at BOOK level (the book is what trades), gated at MATCHED VOL (§3ad's
    method fix) with the standard 2018-21 / 2022-26 split.
  * KEEPER CRITERION, all four required:
       standalone Sharpe >= 0.35, BOTH halves > 0, |corr| to the core book < 0.30,
       and book-level t@matched >= +1.50 with both halves positive.
  * Costs in BASIS POINTS per §3ab, per class, deliberately conservative (CSV spreads
    understate 20-50x - `ftmo-xau-diversifier-search`).
  * Cross-sectional signals use only data through t-1 and trade the t->t+1 close-to-close
    return, so the non-synchronous-close leakage that faked a Sharpe 3.5 in §3b cannot occur.

    python scripts/v5_factor_sleeves.py
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

from scripts.v5_volregime_taper_crossasset import engine_bp, load_sleeve  # noqa: E402
from scripts.v5_xau_champion_lifts import champion_recipe, vol_match, sharpe, dd_of  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402
from scripts.v5_xau_turn_prob import paired, per_year  # noqa: E402

EVAL_START = "2018-01-01"
SPLITS = [("2018-01-01", "2021-12-31", "2018-21"), ("2022-01-01", "2026-12-31", "2022-26")]
TARGET_VOL, MAX_ASSET_LEV, BUFFER = 0.10, 3.0, 0.10
VOL_HL = 20            # vol halflife for per-asset risk scaling (days)
MIN_XS = 20            # minimum assets present for a cross-sectional signal to fire

# ---- universe: one-way cost in bp per class, conservative ----
UNIVERSE: dict[str, float] = {}
for _s in ("AUDUSD EURUSD GBPUSD NZDUSD USDCAD USDCHF USDJPY AUDJPY AUDNZD CADJPY "
           "EURAUD EURCHF EURGBP EURJPY GBPAUD GBPCHF GBPJPY NZDJPY").split():
    UNIVERSE[_s] = 1.5
for _s in "ASX DAX DJI FTSE NDX NIKKEI SPX STOXX".split():
    UNIVERSE[_s] = 1.5
for _s in "GOLD SILVER COPPER PLAT PALL".split():
    UNIVERSE[_s] = 3.0
for _s in "BRENT WTI NATGAS GASOIL HEATOIL".split():
    UNIVERSE[_s] = 6.0
for _s in "COFFEE CORN COTTON SOY SUGAR WHEAT".split():
    UNIVERSE[_s] = 6.0
for _s in "UST2Y UST5Y UST10Y UST30Y".split():
    UNIVERSE[_s] = 2.0
for _s in "BTC ETH".split():
    UNIVERSE[_s] = 3.0

# the deployed book, exactly as in §3af
TREND_CORE = {
    "XAU":   ("data/XAUUSD_H4_long.csv", 252 * 6, 0.75, 1.0, 42),
    "BTC":   ("data/BTC_D1_long.csv", 252, 1.13, 1 / 6, 7),
    "NDX":   ("data/NDX_D1_long.csv", 252, 0.38, 1 / 6, 7),
    "BRENT": ("data/BRENT_D1_long.csv", 252, 5.69, 1 / 6, 7),
}


# ---------------------------------------------------------------- data
def price_panel() -> tuple[pd.DataFrame, pd.Series]:
    px, cost = {}, {}
    for s, bp in UNIVERSE.items():
        d = load_sleeve(dict(path=f"data/{s}_D1_long.csv"))
        if d is None or len(d) < 1500:
            continue
        px[s] = d["close"].astype(float)
        cost[s] = bp
    P = pd.DataFrame(px).sort_index()
    P = P[~P.index.duplicated(keep="last")]
    P = P.reindex(pd.date_range(P.index.min(), P.index.max(), freq="B")).ffill(limit=3)
    return P, pd.Series(cost)


# ---------------------------------------------------------------- factor forecasts
def f_ts_value(P: pd.DataFrame, lb: int = 1260) -> pd.DataFrame:
    """Time-series value: how far BELOW its price of `lb` days ago the asset sits, in
    units of that horizon's own volatility. Cheap (fell a lot) -> long."""
    lv = np.log(P)
    raw = lv.shift(lb) - lv                      # >0 == price fell == cheap
    sig = P.pct_change().ewm(halflife=252, min_periods=252).std() * np.sqrt(lb)
    return (raw / sig.replace(0, np.nan)).clip(-2, 2)


def _xs_z(raw: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score, causally lagged one day, fired only when the universe
    is wide enough. Demeaning makes the sleeve market-neutral by construction."""
    ok = raw.notna().sum(axis=1) >= MIN_XS
    z = raw.sub(raw.mean(axis=1), axis=0).div(raw.std(axis=1).replace(0, np.nan), axis=0)
    return z.where(ok, np.nan).clip(-2, 2).shift(1)


def f_xs_value(P: pd.DataFrame, lb: int = 1260) -> pd.DataFrame:
    lv = np.log(P)
    return _xs_z(lv.shift(lb) - lv)


def f_xs_bab(P: pd.DataFrame, lb: int = 252) -> pd.DataFrame:
    """Betting-against-beta / low-vol: long the calm, short the wild (each still
    vol-scaled downstream, so this is not a leverage bet)."""
    v = P.pct_change().rolling(lb, min_periods=180).std()
    return _xs_z(-np.log(v.replace(0, np.nan)))


def f_xs_skew(P: pd.DataFrame, lb: int = 252) -> pd.DataFrame:
    """Lottery demand: short the positively-skewed, long the negatively-skewed."""
    return _xs_z(-P.pct_change().rolling(lb, min_periods=180).skew())


def f_seasonality(P: pd.DataFrame, min_years: int = 5) -> pd.DataFrame:
    """Expanding, strictly-causal calendar-month effect: for each asset and month,
    the t-statistic of that month's mean return over PRIOR years only."""
    R = P.pct_change()
    M = R.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    out = pd.DataFrame(np.nan, index=P.index, columns=P.columns)
    for col in P.columns:
        m = M[col].dropna()
        if len(m) < min_years * 12:
            continue
        by = {}
        for ts, val in m.items():
            k = ts.month
            hist = by.get(k, [])
            if len(hist) >= min_years:
                a = np.asarray(hist, dtype=float)
                sd = a.std(ddof=1)
                score = 0.0 if sd == 0 else a.mean() / (sd / np.sqrt(len(a)))
                # applies to the FOLLOWING month's days, so no in-month lookahead
                nxt = ts + pd.Timedelta(days=1)
                end = nxt + pd.offsets.MonthEnd(1)
                out.loc[nxt:end, col] = np.clip(score, -2, 2)
            by[k] = hist + [val]
    return out


def f_ts_trend(P: pd.DataFrame) -> pd.DataFrame:
    """CONTROL: the champion recipe on the same broad universe, long-only as deployed.
    Present so 'different factor' can be compared against 'more trend'."""
    return pd.DataFrame({c: champion_recipe(P[c].dropna(), 1 / 6, 1.5)
                         for c in P.columns}).reindex(P.index)


# ---------------------------------------------------------------- sleeve engine
def _buffer_col(p: np.ndarray, band: np.ndarray) -> np.ndarray:
    out, held = np.zeros(len(p)), 0.0
    for i in range(len(p)):
        if np.isfinite(p[i]):
            b = band[i] if np.isfinite(band[i]) else 0.0
            if abs(p[i] - held) > b:
                held = p[i] - np.sign(p[i] - held) * b
        out[i] = held
    return out


def factor_sleeve(F: pd.DataFrame, P: pd.DataFrame, cost: pd.Series) -> pd.Series:
    """Equal-risk across whatever the factor has an opinion on, per-asset vol targeted,
    no-trade banded, costed in bp on realised turnover, executed at the NEXT close."""
    R = P.pct_change()
    vol = R.ewm(halflife=VOL_HL, min_periods=60).std() * np.sqrt(252)
    n = F.notna().sum(axis=1).replace(0, np.nan)
    raw = F.div(n, axis=0) * (TARGET_VOL / vol.replace(0, np.nan))
    raw = raw.clip(-MAX_ASSET_LEV, MAX_ASSET_LEV)
    band = (BUFFER * (TARGET_VOL / vol.replace(0, np.nan)).clip(0, MAX_ASSET_LEV)).div(n, axis=0)
    pos = pd.DataFrame({c: _buffer_col(raw[c].values, band[c].values) for c in raw.columns},
                       index=raw.index).shift(1).fillna(0.0)
    gross = (pos * R).sum(axis=1)
    turn = pos.diff().abs().fillna(0.0)
    cst = (turn * cost.reindex(turn.columns).values * 1e-4).sum(axis=1)
    return (gross - cst).loc[EVAL_START:].dropna()


def core_book() -> tuple[dict[str, pd.Series], pd.Series]:
    sl = {}
    for name, (path, ann, bp, scale, hl) in TREND_CORE.items():
        df = load_sleeve(dict(path=path))
        fc = champion_signal(df["close"]) if scale == 1.0 else champion_recipe(df["close"], scale, 1.5)
        sl[name] = engine_bp(df, fc, bp, ann, hl).loc[EVAL_START:]
    return sl, risk_parity(sl)


def risk_parity(sleeves: dict[str, pd.Series]) -> pd.Series:
    """Equal RISK weight (equal weight lost nothing in §3af, but sleeves here have very
    different vols), scaled by a strictly-causal expanding vol estimate."""
    R = pd.DataFrame(sleeves).dropna(how="all").fillna(0.0)
    v = R.expanding(min_periods=252).std().shift(1)
    W = (0.10 / np.sqrt(252)) / v.replace(0, np.nan)
    return (R * W.clip(upper=10)).mean(axis=1).dropna()


def halves(r: pd.Series) -> str:
    return "  ".join(f"{lab} {sharpe(r.loc[a:b]):+.2f}" for a, b, lab in SPLITS)


def gate(cand: pd.Series, base: pd.Series) -> tuple[float, str, str]:
    idx = cand.index.intersection(base.index)
    c, b = cand.loc[idx], base.loc[idx]
    cm = vol_match(c, b)
    _, t, _ = paired(cm, b)
    yp, yn = per_year(cm, b)
    hh = "  ".join(
        f"{lab} {sharpe(vol_match(c.loc[a:b2], b.loc[a:b2])) - sharpe(b.loc[a:b2]):+.2f}"
        for a, b2, lab in SPLITS)
    return t, f"{yp}/{yn}", hh


def main() -> None:
    P, cost = price_panel()
    print(f"universe: {P.shape[1]} instruments, {P.index.min().date()} -> {P.index.max().date()}")
    print(f"pre-registered keeper bar: standalone SR>=0.35, both halves>0, |corr|<0.30, "
          f"book t@matched>=+1.50 both halves +\n")

    sl_core, base = core_book()
    print("--- baseline: deployed core-4 trend book (equal risk) ---")
    print(f"  SR {sharpe(base):+.3f}   maxDD {dd_of(base):+.2f}%   halves: {halves(base)}\n")

    FACTORS = {
        "TS VALUE (5y reversal)":  f_ts_value(P),
        "XS VALUE (5y reversal)":  f_xs_value(P),
        "XS BAB (low-vol)":        f_xs_bab(P),
        "XS SKEW (lottery)":       f_xs_skew(P),
        "SEASONALITY (month)":     f_seasonality(P),
        "TS TREND (control)":      f_ts_trend(P),
        "XS VALUE 3y (secondary)": f_xs_value(P, 756),
    }

    rows = []
    for name, F in FACTORS.items():
        r = factor_sleeve(F, P, cost)
        if len(r) < 250 or r.std() == 0:
            print(f"{name:26s} INSUFFICIENT DATA")
            continue
        idx = r.index.intersection(base.index)
        corr = r.loc[idx].corr(base.loc[idx])
        aug = risk_parity({**sl_core, name: r})
        t, yy, hh = gate(aug, base)
        h = [sharpe(r.loc[a:b]) for a, b, _ in SPLITS]
        keep = (sharpe(r) >= 0.35 and min(h) > 0 and abs(corr) < 0.30 and t >= 1.50)
        rows.append(dict(name=name, sr=sharpe(r), dd=dd_of(r), corr=corr,
                         book_sr=sharpe(aug), book_dd=dd_of(aug), t=t, yy=yy, keep=keep))
        print(f"{name:26s} SR {sharpe(r):+.3f}  DD {dd_of(r):+6.2f}%  corr(core) {corr:+.2f}  "
              f"halves: {halves(r)}")
        print(f"{'':26s} -> book SR {sharpe(aug):+.3f} (base {sharpe(base):+.3f})  "
              f"DD {dd_of(aug):+.2f}%  t@matched {t:+.2f} [{yy}]  dSR by half: {hh}   "
              f"{'** KEEPER **' if keep else 'fail'}")

    out = pd.DataFrame(rows)
    (ROOT / "data" / "v5_runs").mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / "data" / "v5_runs" / "factor_sleeves.csv", index=False)
    print(f"\nkeepers: {int(out['keep'].sum())} / {len(out)}   -> data/v5_runs/factor_sleeves.csv")


if __name__ == "__main__":
    main()
