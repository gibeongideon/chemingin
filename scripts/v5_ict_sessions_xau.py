"""ICT/SMC concept sweep — session-time concepts. XAU H1 (not H4 — see below).

Plan: `~/.claude/plans/i-wanttt-you-to-playful-widget.md`.

  PO3/AMD (Phase 1, Bucket A) — Accumulation=Asian session range, Manipulation
      =a sweep-and-reclaim of that range during London, Distribution=fade the
      manipulation, held for `hold_bars` bars after it fires. Classic ICT
      teaching maps Asian->accumulation, London->manipulation ("Judas
      Swing"), NY->distribution; this file's PO3/AMD signal literally
      implements that reading.

      RUNS ON H1, NOT H4: an H4-bar first attempt found the London window
      (5h wide) contains at most ONE H4 bar/day, so nearly every (tol, hold)
      cell had zero fires and zero variance (SR=-inf) — a resolution
      mismatch, not a result. The plan explicitly anticipated this ("H4 main,
      plus M15/M30/H1 for session-time concepts"); PO3/AMD is exactly that
      case. Uses a local `engine_h1`/`load_h1` (H1-scaled ANN/vol-halflife,
      identical formula to `v5_xau_turn_prob.engine`) rather than editing
      that shared H4-tuned module. `bh_d`/`ch_d` (buy&hold, champion) stay
      computed on H4 as usual — both sides resample to DAILY before pairing,
      so the underlying bar frequency doesn't need to match for the
      comparison to be valid.

  Judas Swing (Phase 2, Bucket B) — NOT a near-duplicate of PO3/AMD despite
      both being "fade a session fakeout": PO3 anchors to the ASIAN RANGE
      (a pre-formed accumulation box, swept-and-reclaimed during London).
      Judas Swing here anchors to the DAY'S OPEN itself — price displaces
      >= `move_atr` ATR away from the day's open early in London, then
      crosses back THROUGH the open — the classic "initial move traps
      traders, true move is the reversal through the open" read. Different
      reference level, different trigger geometry (a crossing event, not a
      sweep-and-reclaim of a fixed box).

  Silver Bullet, Killzones (Phase 3) — added later.

    python scripts/v5_ict_sessions_xau.py
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

from scripts.v5_xau_turn_prob import load_h4, engine, paired, per_year  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402
from src.features.smc_signals import _atr14, _ohlcv  # noqa: E402
from src.features.ict_primitives import session_windows, daily_open_anchor, REGIME_BOUNDARIES  # noqa: E402
from src.evaluation.lookahead_probe import probe_lookahead, print_report  # noqa: E402
from src.evaluation.dsr_pbo import deflated_sharpe_ratio, pbo_cscv  # noqa: E402
from src.evaluation.walk_forward_grid import run_concept_pipeline, print_summary_table  # noqa: E402

PNL_EVAL_START = "2018-01-01"
YEARS = list(range(2018, 2027))
SELECT_WINDOW_BARS = 3 * 252 * 24    # trailing 3y, H1 bars

TOL_ATRS = (0.1, 0.25, 0.5, 1.0)
HOLD_BARS = (2, 4, 8, 16)            # H1 bars: 2-16h, covers the following NY session

JUDAS_MOVE_ATRS = (0.5, 1.0, 1.5, 2.0)
JUDAS_HOLD_BARS = (2, 4, 8, 16)

ANN_H1 = 252 * 24
VOL_HL_H1 = 42 * 4     # same ~7-trading-day real-time halflife as v5_xau_turn_prob's H4 VOL_HL=42
TARGET_VOL = 0.10
BUFFER = 0.1
MAX_LEV = 8.0
SLIP_USD = 0.10


def load_h1(cost_usd: float | None = None) -> pd.DataFrame:
    """Mirrors `v5_xau_turn_prob.load_h4` exactly, H1 file, honest spread floor."""
    df = pd.read_csv(ROOT / "data" / "XAUUSD_H1_long.csv",
                     parse_dates=["time"], index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    exp_med = df["spread"].expanding(min_periods=20).median().bfill()
    df["spread_px"] = np.maximum(df["spread"], exp_med) * 0.1
    if cost_usd is not None:
        df["spread_px"] = np.maximum(df["spread_px"], float(cost_usd))
    return df


def engine_h1(df: pd.DataFrame, fc: pd.Series, *, buffer_frac: float = BUFFER,
             delay: int = 1, eval_start: str = PNL_EVAL_START) -> tuple[dict, pd.Series]:
    """`v5_xau_turn_prob.engine`'s exact formula, H1-scaled constants."""
    close = df["close"]
    ret = close.pct_change()
    vol = ret.ewm(halflife=VOL_HL_H1, min_periods=20).std() * np.sqrt(ANN_H1)
    pos = (fc.clip(-2.0, 2.0) * (TARGET_VOL / vol)).clip(-MAX_LEV, MAX_LEV)

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
    cost_frac = (df["spread_px"] / 2.0 + SLIP_USD) / close
    net = (pos * ret - pos.diff().abs().fillna(0.0) * cost_frac).fillna(0.0)
    eq = (1.0 + net).cumprod()

    e = eq.loc[eval_start:]
    e = e / e.iloc[0]
    daily = e.resample("D").last().pct_change(fill_method=None).dropna()
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    dd = float((e / e.cummax() - 1).min() * 100)
    m = dict(sharpe=sharpe, dd=dd, years=yrs)
    return m, daily


def asian_range(df: pd.DataFrame, sess: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Asian-session high/low, running max/min WITHIN the session then held
    flat for the rest of that UTC calendar day (causal: `groupby(day).cummax`
    only ever reflects that day's bars up to and including the current one).

    NOTE: `Series.cummax()/cummin()` do NOT forward-fill through NaN in this
    pandas version — they leave NaN at NaN positions (verified directly, not
    assumed) rather than carrying the running extreme forward. Caught this
    the hard way: it silently zeroed out `ah`/`al` for every post-asian bar,
    so `po3_sweep`'s `isfinite` check rejected the entire london window and
    the signal fired 0 times across the whole 67k-bar H1 history. Fixed with
    an explicit per-day `.ffill()` after the cummax/cummin."""
    day = df.index.normalize()
    high_masked = df["high"].where(sess["asian"])
    low_masked = df["low"].where(sess["asian"])
    ah = high_masked.groupby(day).cummax().groupby(day).ffill()
    al = low_masked.groupby(day).cummin().groupby(day).ffill()
    return ah, al


def po3_sweep(df: pd.DataFrame, sess: pd.DataFrame, ah: pd.Series, al: pd.Series, tol_atr: float) -> np.ndarray:
    o, h, l, c, idx = _ohlcv(df)
    a = _atr14(h, l, c)
    tol = tol_atr * a
    london = sess["london"].values
    ah_v, al_v = ah.values, al.values
    sig = np.zeros(len(c))
    for i in range(len(c)):
        if not london[i] or not np.isfinite(ah_v[i]) or not np.isfinite(al_v[i]):
            continue
        if l[i] < al_v[i] - tol[i] and c[i] > al_v[i]:
            sig[i] = 1.0    # bearish sweep of asian low, reclaimed -> fade UP (distribution up)
        elif h[i] > ah_v[i] + tol[i] and c[i] < ah_v[i]:
            sig[i] = -1.0   # bullish sweep of asian high, reclaimed -> fade DOWN
    return sig


def po3_fc(df: pd.DataFrame, tol_atr: float, hold_bars: int) -> pd.Series:
    sess = session_windows(df)
    ah, al = asian_range(df, sess)
    sweep = po3_sweep(df, sess, ah, al, tol_atr)
    n = len(df)
    state = np.zeros(n)
    cur, since = 0.0, 10**9
    for i in range(n):
        if sweep[i] != 0:
            cur, since = sweep[i], 0
        else:
            since += 1
            if since > hold_bars:
                cur = 0.0
        state[i] = cur
    return pd.Series(state, index=df.index, name="po3_fc")


def judas_fc(df: pd.DataFrame, move_atr: float, hold_bars: int) -> pd.Series:
    o, h, l, c, idx = _ohlcv(df)
    a = _atr14(h, l, c)
    day_open, _, _ = daily_open_anchor(df)
    dist = (pd.Series(c, index=idx) - day_open) / pd.Series(a, index=idx)
    day = idx.normalize()
    max_prior = dist.groupby(day).cummax().groupby(day).shift(1)
    min_prior = dist.groupby(day).cummin().groupby(day).shift(1)
    sess = session_windows(df)
    london = sess["london"].values
    d, mx, mn = dist.values, max_prior.values, min_prior.values
    n = len(c)
    sweep = np.zeros(n)
    for i in range(n):
        if not london[i]:
            continue
        if np.isfinite(mx[i]) and mx[i] >= move_atr and d[i] <= 0:
            sweep[i] = -1.0   # was above open by move_atr, crossed back through -> fade DOWN
        elif np.isfinite(mn[i]) and mn[i] <= -move_atr and d[i] >= 0:
            sweep[i] = 1.0    # was below open by move_atr, crossed back through -> fade UP
    state = np.zeros(n)
    cur, since = 0.0, 10**9
    for i in range(n):
        if sweep[i] != 0:
            cur, since = sweep[i], 0
        else:
            since += 1
            if since > hold_bars:
                cur = 0.0
        state[i] = cur
    return pd.Series(state, index=df.index, name="judas_fc")


def main():
    print("=== STAGE 0: lookahead probe ===")
    results = {
        "po3(tol=0.25,hold=4)": probe_lookahead(
            lambda d: po3_fc(d, 0.25, 4), n=1400, offsets=(200, 120, 60, 30), warmup_buffer=600
        ),
        "judas(move=1.0,hold=4)": probe_lookahead(
            lambda d: judas_fc(d, 1.0, 4), n=1400, offsets=(200, 120, 60, 30), warmup_buffer=600
        ),
    }
    print_report(results)
    if not all(v.ok for v in results.values()):
        print("!! lookahead probe FAILED — refusing to trust the backtest below.")
        return

    df4 = load_h4(0.448)
    champ_fc = champion_signal(df4["close"])
    bh_m, bh_d = engine(df4, pd.Series(1.0, index=df4.index), eval_start=PNL_EVAL_START)
    ch_m, ch_d = engine(df4, champ_fc, eval_start=PNL_EVAL_START)

    df = load_h1(0.448)
    n_london_bars_per_day = int(session_windows(df)["london"].groupby(df.index.normalize()).sum().mean().round())
    print(f"\nH1 sanity: ~{n_london_bars_per_day} london-window H1 bars/day (was ~0.2/day at H4)")

    def pipe(name, grid):
        return run_concept_pipeline(
            name, df, grid,
            engine=engine_h1, paired=paired, per_year=per_year,
            deflated_sharpe_ratio=deflated_sharpe_ratio, pbo_cscv=pbo_cscv,
            bh_m=bh_m, bh_d=bh_d, ch_m=ch_m, ch_d=ch_d,
            regime_boundaries=REGIME_BOUNDARIES["XAUUSD"],
            pnl_eval_start=PNL_EVAL_START, years=YEARS, select_window_bars=SELECT_WINDOW_BARS,
        )

    results_all = []
    po3_grid = {(t, h): (lambda t=t, h=h: po3_fc(df, t, h)) for t in TOL_ATRS for h in HOLD_BARS}
    results_all.append(pipe("PO3/AMD (Asian-acc, London-manip, fade) [H1]", po3_grid))

    judas_grid = {(m, h): (lambda m=m, h=h: judas_fc(df, m, h)) for m in JUDAS_MOVE_ATRS for h in JUDAS_HOLD_BARS}
    results_all.append(pipe("Judas Swing (day-open reversal) [H1]", judas_grid))

    print(f"\n{'='*70}\nSUMMARY (sessions)\n{'='*70}")
    print_summary_table(results_all)
    out = ROOT / "data" / "v5_runs" / "ict_sessions_xau_summary.csv"
    pd.DataFrame([{k: v for k, v in r.items() if k != "regimes"} for r in results_all]).to_csv(out, index=False)
    print(f"\nsummary -> {out}")


if __name__ == "__main__":
    main()
