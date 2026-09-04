"""PRE-REGISTERED: VOLUME — the one data type in this repo that has never been backtested.

`xau-regime-features-fwd-accuracy` closed with an explicit verdict: price-derived features are
exhausted and "a new data type" is needed. Every CSV here carries `tick_volume`, fully
populated and non-degenerate on XAU (M30 median 3,428 ticks/bar, zero-frac 0.000, 2015-2026),
and grep confirms it has never entered a v5 backtest — `src/features/volume_signals.py` exists
but is referenced by nothing, and "volume" appears nowhere in V5_FINDINGS.md.

Four distinct uses of volume, chosen because they are mechanistically different from each
other, not three flavours of one idea:

  1. VOLUME / DOLLAR BARS (Lopez de Prado, "Advances in Financial ML" ch.2). Sample the price
     path on an ACTIVITY clock instead of a wall clock. This does not change the signal at
     all — it changes what "a bar" means, so returns are closer to IID and the same champion
     recipe should estimate trend with less noise. This is the PRIMARY candidate because it is
     a potential REPLACEMENT for the deployed champion, not an overlay.
  2. VOLUME-WEIGHTED PRICE PATH. A synthetic strictly-positive price whose increments are the
     real returns amplified on heavy-volume bars and damped on light ones; trend-follow that
     instead of raw price. Tests the classic claim "trust moves that come with volume".
  3. VOLUME-EXPANSION GATE on the champion's forecast (breakouts on quiet volume are false).
  4. PRICE-VOLUME DIVERGENCE and AMIHUD ILLIQUIDITY as size multipliers.

PRE-REGISTRATION (fixed before any result was seen):
  * Trials disclosed in full: 2 bar kinds x 3 rates = 6, VWPP x 2 halflives = 2, and 3 overlay
    concepts x 3 parameters = 9. Twenty-one cells total, all printed.
  * The D=6 TRAP IS HANDLED: `champion_signal` hardcodes 6 bars/day, so a 3- or 12-bar/day
    clock is run through `champion_recipe(close, scale=6/bpd, 1.5)` to hold the economic
    horizon in DAYS constant. Without this, a faster clock would silently become a faster
    trend follower and the comparison would be about speed, not sampling.
  * The volume-bar threshold is CAUSAL: trailing 252-day mean daily volume / bars-per-day,
    shifted. A full-sample threshold would leak the average activity level.
  * Everything is evaluated on the SAME H4 grid with the SAME engine and the SAME
    live-verified $0.448-equivalent 0.75bp one-way cost as the deployed champion, so only the
    signal differs.
  * GATES: overlays (3,4) at MATCHED VOL per §3ad, since they can only de-risk. Replacements
    (1,2) on raw Sharpe AND matched-vol paired-t, since they change exposure both ways.
  * KEEPER BAR: replacement must beat the champion's Sharpe in BOTH halves and post
    t@matched >= +1.50. Overlay must post t@matched >= +1.50 with both half-deltas positive.

    python scripts/v5_volume_signals_xau.py
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

from scripts.v5_volregime_taper_crossasset import engine_bp  # noqa: E402
from scripts.v5_xau_champion_lifts import champion_recipe, vol_match, sharpe, dd_of  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402
from scripts.v5_xau_turn_prob import paired, per_year  # noqa: E402

EVAL_START = "2018-01-01"
SPLITS = [("2018-01-01", "2021-12-31", "2018-21"), ("2022-01-01", "2026-12-31", "2022-26")]
ANN_H4, HL_H4, COST_BP = 252 * 6, 42, 0.75


def load(tf: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / f"data/XAUUSD_{tf}_long.csv", parse_dates=["time"]).set_index("time")
    d = d[~d.index.duplicated(keep="last")].sort_index()
    return d.astype({c: float for c in ("open", "high", "low", "close", "tick_volume")})


# ------------------------------------------------------------------ 1. activity-clock bars
def volume_bars(df: pd.DataFrame, kind: str, bpd: float, warm: int = 252) -> pd.DataFrame:
    """Cut a new bar every time cumulative activity crosses a CAUSAL threshold
    (trailing 252-day mean daily activity / bars-per-day). `kind='dollar'` weights each
    tick by price, which is the closer analogue of traded value."""
    act = df["tick_volume"] * (df["close"] if kind == "dollar" else 1.0)
    daily = act.resample("D").sum()
    thr_d = daily.rolling(warm, min_periods=40).mean().shift(1)
    thr_d = thr_d.fillna(daily.expanding(min_periods=5).mean().shift(1)).ffill()
    thr = (thr_d / bpd).reindex(df.index, method="ffill")

    o = df["open"].values; h = df["high"].values; lo = df["low"].values
    c = df["close"].values; a = act.values; t = thr.values
    rows, idx = [], []
    cum = 0.0; bo = np.nan; bh = -np.inf; bl = np.inf; bv = 0.0
    for i in range(len(df)):
        if not np.isfinite(t[i]) or t[i] <= 0:
            continue
        if np.isnan(bo):
            bo = o[i]
        bh = max(bh, h[i]); bl = min(bl, lo[i]); bv += a[i]; cum += a[i]
        if cum >= t[i]:
            rows.append((bo, bh, bl, c[i], bv)); idx.append(df.index[i])
            cum = 0.0; bo = np.nan; bh = -np.inf; bl = np.inf; bv = 0.0
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx),
                        columns=["open", "high", "low", "close", "tick_volume"])


# ------------------------------------------------------------------ 2. volume-weighted path
def vw_price_path(df: pd.DataFrame, hl: int, w_clip: float = 4.0) -> pd.Series:
    """Strictly-positive synthetic price: real returns scaled by relative volume.
    Heavy-volume moves count for more, light-volume moves for less."""
    r = df["close"].pct_change().fillna(0.0)
    w = (df["tick_volume"] / df["tick_volume"].ewm(halflife=hl, min_periods=20).mean())
    w = w.clip(1 / w_clip, w_clip).fillna(1.0)
    return np.exp((w * r).cumsum()) * float(df["close"].iloc[0])


# ------------------------------------------------------------------ 3/4. size multipliers
def _pct(s: pd.Series, win: int) -> pd.Series:
    return s.rolling(win, min_periods=win // 4).apply(lambda x: (x[-1] > x).mean(), raw=True)


def m_vol_expansion(df: pd.DataFrame, q: float, lo: float, win: int = 252 * 6) -> pd.Series:
    """Full size only when volume is running hot; `lo` size when it is quiet."""
    p = _pct(df["tick_volume"].ewm(halflife=6, min_periods=6).mean(), win).fillna(0.5)
    return pd.Series(np.where(p >= q, 1.0, lo), index=df.index)


def m_pv_divergence(df: pd.DataFrame, win: int, lo: float) -> pd.Series:
    """Taper when price is trending on FALLING volume (correlation of |ret| with volume
    change turns negative) - the textbook 'weak trend' read."""
    r = df["close"].pct_change().abs()
    dv = df["tick_volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    c = r.rolling(win, min_periods=win // 2).corr(dv).fillna(0.0)
    return (lo + (1.0 - lo) * ((c + 0.3) / 0.6).clip(0, 1)).clip(lo, 1.0)


def m_amihud(df: pd.DataFrame, win: int, lo: float) -> pd.Series:
    """Amihud illiquidity |ret|/volume: size down when the market is thin (moves are
    cheap to cause and less informative)."""
    ill = (df["close"].pct_change().abs() / df["tick_volume"].replace(0, np.nan))
    p = _pct(ill.ewm(halflife=12, min_periods=12).mean(), win).fillna(0.5)
    return (lo + (1.0 - lo) * (1.0 - p)).clip(lo, 1.0)


# ------------------------------------------------------------------ evaluation
def halves(r: pd.Series) -> list[float]:
    return [sharpe(r.loc[a:b]) for a, b, _ in SPLITS]


def gate(cand: pd.Series, base: pd.Series) -> tuple[float, str, list[float]]:
    i = cand.index.intersection(base.index)
    c, b = cand.loc[i], base.loc[i]
    _, t, _ = paired(vol_match(c, b), b)
    yp, yn = per_year(vol_match(c, b), b)
    dh = [sharpe(vol_match(c.loc[a:x], b.loc[a:x])) - sharpe(b.loc[a:x]) for a, x, _ in SPLITS]
    return t, f"{yp}/{yn}", dh


def show(tag: str, r: pd.Series, base: pd.Series, kind: str) -> dict:
    t, yy, dh = gate(r, base)
    h = halves(r)
    hb = halves(base)
    if kind == "repl":
        keep = (h[0] > hb[0] and h[1] > hb[1] and t >= 1.50)
    else:
        keep = (t >= 1.50 and min(dh) > 0)
    print(f"  {tag:34s} SR {sharpe(r):+.3f}  DD {dd_of(r):+6.2f}%  "
          f"halves {h[0]:+.2f}/{h[1]:+.2f}  t@matched {t:+.2f} [{yy}]  "
          f"dSR {dh[0]:+.2f}/{dh[1]:+.2f}  {'** KEEPER **' if keep else ''}")
    return dict(tag=tag, kind=kind, sr=sharpe(r), dd=dd_of(r), h1=h[0], h2=h[1],
                t=t, yy=yy, d1=dh[0], d2=dh[1], keep=keep)


def main() -> None:
    h4, m30 = load("H4"), load("M30")
    base = engine_bp(h4, champion_signal(h4["close"]), COST_BP, ANN_H4, HL_H4)
    hb = halves(base)
    print(f"XAU H4 {h4.index.min().date()}->{h4.index.max().date()}  "
          f"M30 rows {len(m30):,}   eval from {EVAL_START}")
    print(f"\nBASELINE deployed champion: SR {sharpe(base):+.3f}  DD {dd_of(base):+.2f}%  "
          f"halves {hb[0]:+.2f}/{hb[1]:+.2f}")
    print("keeper bar: replacement must beat BOTH halves and t@matched>=+1.50; "
          "overlay t@matched>=+1.50 with both half-deltas +\n")
    rows = []

    print("--- 1. ACTIVITY-CLOCK BARS (primary; signal unchanged, sampling changed) ---")
    for kind in ("tick", "dollar"):
        for bpd in (3, 6, 12):
            vb = volume_bars(m30, kind, bpd)
            if len(vb) < 2000:
                print(f"  {kind}/{bpd}bpd  only {len(vb)} bars - skipped")
                continue
            fc = champion_recipe(vb["close"], 6.0 / bpd, 1.5)
            fch4 = fc.reindex(h4.index, method="ffill")
            r = engine_bp(h4, fch4, COST_BP, ANN_H4, HL_H4)
            rows.append(show(f"{kind} bars @{bpd}/day ({len(vb):,} bars)", r, base, "repl"))

    print("\n--- 2. VOLUME-WEIGHTED PRICE PATH (trend the volume-weighted path) ---")
    for hl in (12, 60):
        p = vw_price_path(h4, hl)
        r = engine_bp(h4, champion_signal(p), COST_BP, ANN_H4, HL_H4)
        rows.append(show(f"VW path hl={hl} bars", r, base, "repl"))

    print("\n--- 3/4. VOLUME OVERLAYS on the deployed champion (matched-vol gated) ---")
    fc0 = champion_signal(h4["close"])
    overlays = {}
    for q, lo in ((0.30, 0.5), (0.50, 0.5), (0.50, 0.0)):
        overlays[f"vol-expansion q>{q:.2f} lo={lo}"] = m_vol_expansion(h4, q, lo)
    for win, lo in ((60, 0.5), (240, 0.5), (240, 0.0)):
        overlays[f"PV-divergence win={win} lo={lo}"] = m_pv_divergence(h4, win, lo)
    for win, lo in ((252 * 6, 0.5), (504 * 6, 0.5), (252 * 6, 0.25)):
        overlays[f"Amihud win={win // 6}d lo={lo}"] = m_amihud(h4, win, lo)
    for tag, m in overlays.items():
        r = engine_bp(h4, (fc0 * m).clip(0, 2), COST_BP, ANN_H4, HL_H4)
        rows.append(show(tag, r, base, "ovl"))

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data" / "v5_runs" / "volume_signals_xau.csv", index=False)
    print(f"\ntrials {len(out)}   keepers {int(out['keep'].sum())}   "
          f"best repl SR {out[out.kind == 'repl'].sr.max():+.3f} vs champion {sharpe(base):+.3f}")
    print("-> data/v5_runs/volume_signals_xau.csv")


if __name__ == "__main__":
    main()
