"""PHASE 1 SCREEN — is there ANY net edge in a 4-15 hour LONG/SHORT XAU trend?

User ask (2026-08-31): a "Trend Zigzag long term follower" on a 6-12h cadence that
"behaves like our champion but both sells and buys", follows 4-15 hour trends, and
"takes profit using discovered signal". Extensive research requested.

This is deliberately the CHEAP step first. Four candidate signal families are screened
through the continuous vol-targeted engine (fast to grid, no per-trade mechanics) at the
REAL measured cost, walk-forward selected. Only what survives here earns the expensive
discrete/exit-rule engine in phase 2. Screening at realistic cost from the start is the
whole point — `xau-nextbar-winrate-martingale` is the cautionary tale in this repo of a
real GROSS edge that was dead net of spread, so gross and net are both printed side by
side and the gap is the finding.

MEASURED COST (Maven live, 2026-08-31, 12 tick samples): XAUUSD spread $0.44, stable.
  swap_long -45.33 / swap_short +25.25 USD per lot per night (shorts EARN carry — a real
  tailwind for a long/short book that the long-only champion cannot use).
Cost levels tested: $0.12 (raw/ECN), $0.34 (HFM cent), $0.44 (Maven/FTMO measured).

PRIOR ART THIS IS SWIMMING AGAINST (both must be beaten, not ignored):
  - `fast-trend-runner-spread-gated`: short-horizon XAU trend is REAL but SR 0.50 at
    $0.34 and only 0.85 at $0.12 — i.e. cost, not signal, is the binding constraint.
  - `xau-longonly-champion`: "kill-the-shorts is THE lever" — on XAU H4 the short side
    bled, and removing it is what took the champion to Sharpe >= 1. A long/short spec
    therefore starts with a documented handicap, partially offset by the +swap above.
  So LONG-ONLY and SHORT-ONLY legs are reported separately for every signal: if the
  short side does not carry its own weight, that is the answer to "it should both sell
  and buy".

SIGNALS (all causal, all long/short, all lookahead-probed before use):
  ewmac_fast   champion's own EWMAC math at H1-fast spans (the "behave like our
               champion" reading)
  breakout_fast Donchian mid/range at 6-24h windows (champion's other half)
  zigzag       the literal "zigzag": CAUSAL confirmed fractal swings, long while
               structure makes higher highs/lows, short on lower highs/lows. NOTE:
               V5_FINDINGS §3o/§3v disproved this family on H4 (walk-forward 0.04-0.54,
               bull-beta) — retested here only because H1 + long/short is a different
               configuration, with priced-in scepticism.
  supertrend   `smc_signals.supertrend` at fast period (§3s found it interchangeable
               with the champ recipe on H4; untested at this horizon)

    python scripts/v5_xau_fast_ls_screen.py
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

from src.features.smc_signals import supertrend as st_signal  # noqa: E402
from src.evaluation.lookahead_probe import probe_lookahead, print_report  # noqa: E402
from src.evaluation.dsr_pbo import deflated_sharpe_ratio, pbo_cscv  # noqa: E402
from src.evaluation.walk_forward_grid import (  # noqa: E402
    walk_forward_select, regime_split_sharpe, print_selections,
)

EVAL_START = "2018-01-01"
YEARS = list(range(2018, 2027))
SELECT_WINDOW_BARS = 3 * 252 * 24     # trailing 3y of H1 bars

ANN_H1 = 252 * 24
VOL_HL_H1 = 42 * 4                    # ~7 trading days, matching the H4 VOL_HL=42
TARGET_VOL = 0.10
BUFFER = 0.1
MAX_LEV = 8.0
SLIP_USD = 0.10

COST_LEVELS = {"raw/ECN $0.12": 0.12, "cent $0.34": 0.34, "Maven/FTMO $0.44": 0.44}

XAU_REGIMES = [
    ("2015-01-01", "2018-12-31", "chop"),
    ("2019-01-01", "2020-12-31", "bull"),
    ("2021-01-01", "2022-12-31", "flat"),
    ("2023-01-01", "2026-12-31", "bull"),
]


def load_h1() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "XAUUSD_H1_long.csv",
                     parse_dates=["time"], index_col="time").sort_index()
    return df[~df.index.duplicated(keep="last")]


# ----------------------------------------------------------------- signals
def _norm_expanding(s: pd.Series) -> pd.Series:
    """Past-only rescale to unit mean |forecast| — same convention as
    src/v5/xau_dual_signals._norm (shift(1), no future information)."""
    return s * (1.0 / s.abs().expanding(min_periods=120).mean().shift(1))


def ewmac_fast(df: pd.DataFrame, speeds=((4, 16), (6, 24), (8, 32)), cap=2.0) -> pd.Series:
    """Champion's EWMAC forecast, spans in H1 bars so the crossover targets a
    4-32 hour trend rather than the champion's 16-256 DAY speeds."""
    close = df["close"]
    ret = close.pct_change()
    price_vol = close * ret.ewm(span=36, min_periods=20).std()
    combined = None
    for fast, slow in speeds:
        raw = (close.ewm(span=fast, min_periods=fast).mean()
               - close.ewm(span=slow, min_periods=slow).mean()) / price_vol
        fc = _norm_expanding(raw).clip(-cap * 2, cap * 2)
        combined = fc if combined is None else combined + fc
    return (combined / len(speeds)).clip(-cap, cap)


def breakout_fast(df: pd.DataFrame, windows=(6, 12, 24), cap=2.0) -> pd.Series:
    close = df["close"]
    combined = None
    for n in windows:
        hi = close.rolling(n, min_periods=max(2, n // 2)).max()
        lo = close.rolling(n, min_periods=max(2, n // 2)).min()
        mid = (hi + lo) / 2.0
        rng = (hi - lo).replace(0.0, np.nan)
        raw = ((close - mid) / rng * 4.0).ewm(span=max(2, n // 4)).mean()
        fc = _norm_expanding(raw).clip(-cap * 2, cap * 2)
        combined = fc if combined is None else combined + fc
    return (combined / len(windows)).clip(-cap, cap)


def zigzag_structure(df: pd.DataFrame, order: int = 3, cap: float = 2.0) -> pd.Series:
    """CAUSAL zigzag/market-structure direction.

    A fractal swing at bar j is only CONFIRMABLE at bar j+order (needs the
    following `order` bars to know nothing beat it), so it is revealed there
    and never earlier. Forecast = +1 while the last two confirmed swing highs
    AND lows are both rising (higher highs / higher lows), -1 while both are
    falling, 0 when structure disagrees with itself. Scaled to unit mean
    |forecast| like every other signal here so the engine sees comparable units.
    """
    h, l = df["high"].values, df["low"].values
    n = len(h)
    sh = np.full(n, np.nan)     # last confirmed swing high
    sh_prev = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    sl_prev = np.full(n, np.nan)
    last_h = prev_h = last_l = prev_l = np.nan
    for i in range(n):
        j = i - order
        if j >= order:
            if h[j] >= h[j - order:i + 1].max():
                prev_h, last_h = last_h, h[j]
            if l[j] <= l[j - order:i + 1].min():
                prev_l, last_l = last_l, l[j]
        sh[i], sh_prev[i], sl[i], sl_prev[i] = last_h, prev_h, last_l, prev_l
    hh = sh > sh_prev
    hl = sl > sl_prev
    lh = sh < sh_prev
    ll = sl < sl_prev
    raw = np.where(hh & hl, 1.0, np.where(lh & ll, -1.0, 0.0))
    return pd.Series(raw, index=df.index).clip(-cap, cap)


def supertrend_fast(df: pd.DataFrame, period: int = 10, mult: float = 3.0,
                    cap: float = 2.0) -> pd.Series:
    d, dist = st_signal(df, period=period, multiplier=mult)
    return (d * np.minimum(1.0 + dist.abs() / 4.0, 2.0)).clip(-cap, cap)


SIGNALS = {
    "ewmac_fast(4-32h)":   lambda d: ewmac_fast(d, ((4, 16), (6, 24), (8, 32))),
    "ewmac_fast(6-48h)":   lambda d: ewmac_fast(d, ((6, 24), (9, 36), (12, 48))),
    "ewmac_fast(12-60h)":  lambda d: ewmac_fast(d, ((12, 48), (15, 60))),
    "breakout(6-24h)":     lambda d: breakout_fast(d, (6, 12, 24)),
    "breakout(12-48h)":    lambda d: breakout_fast(d, (12, 24, 48)),
    "zigzag(order2)":      lambda d: zigzag_structure(d, 2),
    "zigzag(order3)":      lambda d: zigzag_structure(d, 3),
    "zigzag(order5)":      lambda d: zigzag_structure(d, 5),
    "supertrend(10,3)":    lambda d: supertrend_fast(d, 10, 3.0),
    "supertrend(6,2)":     lambda d: supertrend_fast(d, 6, 2.0),
}


# ----------------------------------------------------------------- engine
def engine_h1(df: pd.DataFrame, fc: pd.Series, spread_usd: float,
              *, side: str = "both", buffer_frac: float = BUFFER,
              delay: int = 1, gross: bool = False) -> pd.Series:
    """Continuous vol-targeted H1 engine (same formula as
    v5_xau_turn_prob.engine, H1-scaled constants). `side` restricts the book
    to long-only / short-only so each leg's contribution is measurable.
    Returns DAILY net returns."""
    close = df["close"]
    ret = close.pct_change()
    vol = ret.ewm(halflife=VOL_HL_H1, min_periods=20).std() * np.sqrt(ANN_H1)
    f = fc.copy()
    if side == "long":
        f = f.clip(lower=0.0)
    elif side == "short":
        f = f.clip(upper=0.0)
    pos = (f.clip(-2.0, 2.0) * (TARGET_VOL / vol)).clip(-MAX_LEV, MAX_LEV)

    if buffer_frac > 0:
        band = (buffer_frac * (TARGET_VOL / vol).clip(0, MAX_LEV)).values
        p, out, held = pos.values.copy(), np.zeros(len(pos)), 0.0
        for i in range(len(p)):
            if np.isfinite(p[i]):
                b = band[i] if np.isfinite(band[i]) else 0.0
                if abs(p[i] - held) > b:
                    held = p[i] - np.sign(p[i] - held) * b
            out[i] = held
        pos = pd.Series(out, index=pos.index)

    pos = pos.shift(delay).fillna(0.0)
    cost_frac = 0.0 if gross else (spread_usd / 2.0 + SLIP_USD) / close
    net = (pos * ret - pos.diff().abs().fillna(0.0) * cost_frac).fillna(0.0)
    eq = (1.0 + net).cumprod()
    e = eq.loc[EVAL_START:]
    if e.empty:
        return pd.Series(dtype=float)
    e = e / e.iloc[0]
    return e.resample("D").last().pct_change(fill_method=None).dropna()


def sharpe(d: pd.Series) -> float:
    return float(d.mean() / d.std() * np.sqrt(252)) if len(d) > 20 and d.std() > 0 else np.nan


def turnover_per_year(df: pd.DataFrame, fc: pd.Series) -> float:
    """Rough position turnover — the quantity that decides whether cost kills it."""
    close = df["close"]
    ret = close.pct_change()
    vol = ret.ewm(halflife=VOL_HL_H1, min_periods=20).std() * np.sqrt(ANN_H1)
    pos = (fc.clip(-2, 2) * (TARGET_VOL / vol)).clip(-MAX_LEV, MAX_LEV).shift(1)
    yrs = (df.index[-1] - df.index[0]).days / 365.25
    return float(pos.diff().abs().sum() / yrs)


def main() -> None:
    df = load_h1()
    print(f"XAUUSD H1 {df.index[0].date()}..{df.index[-1].date()}  ({len(df):,} bars)")
    print(f"eval {EVAL_START}+   measured Maven spread $0.44 (live, 2026-08-31)\n")

    print("=== STAGE 0: lookahead probes (every signal, before any backtest) ===")
    # bar DURATION is irrelevant to a truncation-invariance check (the signal
    # functions only see bar ordering), so the probe's default synthetic panel is fine
    probes = {name: probe_lookahead(fn, n=3000, offsets=(200, 120, 60, 30),
                                    warmup_buffer=400)
              for name, fn in SIGNALS.items()}
    print_report(probes)
    if not all(v.ok for v in probes.values()):
        print("!! a signal FAILED the causality probe — refusing to screen it.")
        return

    # ---------------- STAGE 1: gross vs net at every cost level
    print("\n=== STAGE 1: GROSS vs NET (the cost wall, stated explicitly) ===")
    print(f"{'signal':22} {'turn/yr':>8} {'GROSS':>7} " +
          " ".join(f"{k:>18}" for k in COST_LEVELS))
    rows = []
    fcs = {name: fn(df) for name, fn in SIGNALS.items()}
    for name, fc in fcs.items():
        g = sharpe(engine_h1(df, fc, 0.0, gross=True))
        nets = {k: sharpe(engine_h1(df, fc, c)) for k, c in COST_LEVELS.items()}
        turn = turnover_per_year(df, fc)
        rows.append(dict(signal=name, turnover=turn, gross=g, **{k: v for k, v in nets.items()}))
        print(f"{name:22} {turn:>8.0f} {g:>+7.2f} " +
              " ".join(f"{nets[k]:>+18.2f}" for k in COST_LEVELS))
    R = pd.DataFrame(rows)

    best_net = R["Maven/FTMO $0.44"].max()
    print(f"\n  best GROSS Sharpe {R.gross.max():+.2f}   ->   best NET at the real "
          f"$0.44 spread {best_net:+.2f}")
    print(f"  signals with NET Sharpe > 0 at $0.44: "
          f"{int((R['Maven/FTMO $0.44'] > 0).sum())}/{len(R)}")

    # ---------------- STAGE 2: long vs short attribution (the "both buy and sell" question)
    print("\n=== STAGE 2: does the SHORT side carry its own weight? (at $0.44) ===")
    print(f"{'signal':22} {'both':>8} {'long-only':>10} {'short-only':>11} {'verdict':>26}")
    for name, fc in fcs.items():
        b = sharpe(engine_h1(df, fc, 0.44, side="both"))
        lo = sharpe(engine_h1(df, fc, 0.44, side="long"))
        so = sharpe(engine_h1(df, fc, 0.44, side="short"))
        verdict = ("shorts ADD" if (np.isfinite(b) and np.isfinite(lo) and b > lo)
                   else "shorts DILUTE")
        print(f"{name:22} {b:>+8.2f} {lo:>+10.2f} {so:>+11.2f} {verdict:>26}")

    # ---------------- STAGE 3: walk-forward over the signal set at the real cost
    print("\n=== STAGE 3: WALK-FORWARD signal selection at $0.44 (trailing 3y) ===")
    grid = {name: engine_h1(df, fc, 0.44) for name, fc in fcs.items()}
    oos, sel = walk_forward_select(grid, YEARS, SELECT_WINDOW_BARS, min_train_bars=100)
    print_selections(sel)
    if not oos.empty and oos.std() > 0:
        e = (1 + oos).cumprod()
        print(f"\nWALK-FORWARD net SR={sharpe(oos):+.3f}  "
              f"DD={float((e/e.cummax()-1).min()*100):+.1f}%")
        trials = np.array([s / np.sqrt(252) for s in R["Maven/FTMO $0.44"] if np.isfinite(s)])
        dsr = deflated_sharpe_ratio(oos.values, trials)
        common = None
        for v in grid.values():
            common = v.index if common is None else common.intersection(v.index)
        pbo = pbo_cscv(np.column_stack([v.reindex(common).fillna(0.0).values
                                        for v in grid.values()]), n_partitions=10)
        print(f"DSR={dsr['dsr']:.3f} (n_trials={dsr['n_trials']})   PBO={pbo.pbo:.3f}")
        print("\nregime split:")
        for k, v in regime_split_sharpe(oos, XAU_REGIMES).items():
            print(f"  {k:24s} {v:+.2f}")

    out = ROOT / "data" / "v5_runs" / "xau_fast_ls_screen.csv"
    R.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
