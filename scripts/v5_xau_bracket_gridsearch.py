"""Discrete SL/TP bracket trades off the fwd6/regime predictor — grid search. (2026-08-11)

Different trade structure from the earlier fwd6 P&L tests (which continuously
resized a position every bar and got killed by turnover + sitting flat through
gold's drift). Here: enter a DISCRETE trade in the predictor's direction, then
manage it with a plain bracket —

    TP: close as soon as unrealized profit clears `tp_pct` (NOT a fixed price
        target chosen up front and forced closed at a horizon — the trade just
        waits, open-ended, until it's "positive enough")
    SL: close if unrealized loss reaches `sl_pct` (kept small, per the ask, to
        cap the downside of a signal that's only ~52% accurate)

No time-based exit (beyond a generous 200-bar/~33-day backstop so a trade can't
hang open forever) — holding period is whatever it takes to hit one or the
other. Intrabar TP/SL checked against each bar's HIGH/LOW, not just the close;
if both would trigger within the same bar, SL is assumed to fire first (the
conservative convention this repo already uses — you don't get to assume the
favourable path when you can't see the wick order).

Entry: causal OOS walk-forward probability from the SAME price+regime/HistGBoost
classifier already validated for accuracy (`v5_xau_intermarket_accuracy.py`) —
direction = sign(p-0.5), one bar delay, live cost ($0.448 spread + $0.10 slip)
charged on both entry and exit. Full notional while in a trade, flat otherwise
(no vol-targeting/leverage layered on yet — that's a secondary refinement if a
cell here looks real).

CAVEAT flagged up front, not hidden: the SL/TP grid is selected by looking at
performance over the SAME window it's evaluated on — that is an in-sample
choice of EXIT parameters (the ENTRY signal itself stays genuinely OOS/walk-
forward). The per-year breakdown printed for the winning cells is the check on
whether that choice is a single-window mirage or a real, stable effect.

    python scripts/v5_xau_bracket_gridsearch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_xau_turn_prob import load_h4, SLIP_USD  # noqa: E402
from scripts.v5_xau_fwd6_regime_pnl import oos_probabilities  # noqa: E402
from scripts.v5_xau_intermarket_accuracy import EVAL_START_YEAR  # noqa: E402

PNL_EVAL_START = "2018-01-01"
MAX_HOLD = 200          # bars (~33 days) — backstop only, not the intended exit
SL_GRID = [0.0005, 0.0010, 0.0015, 0.0020, 0.0030, 0.0040, 0.0050, 0.0075]
TP_GRID = [0.0010, 0.0015, 0.0020, 0.0030, 0.0040, 0.0050, 0.0075, 0.0100, 0.0150, 0.0200]


def backtest_bracket(df: pd.DataFrame, sig: np.ndarray, sl_pct: float, tp_pct: float,
                     delay: int = 1, max_hold: int = MAX_HOLD):
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    spread = df["spread_px"].values
    n = len(df)

    ret = np.zeros(n)
    trades = []
    position = 0
    entry_price = entry_bar = 0

    i = 0
    while i < n:
        if position == 0:
            entry_i = i + delay
            if entry_i < n and sig[i] != 0:
                position = int(sig[i])
                entry_price = close[entry_i]
                entry_bar = entry_i
                cost_frac = (spread[entry_i] / 2.0 + SLIP_USD) / entry_price
                ret[entry_i] -= cost_frac
                i = entry_i
                continue
            i += 1
            continue

        if i == entry_bar:
            i += 1
            continue

        prev_close = close[i - 1]
        tp_price = entry_price * (1 + tp_pct) if position == 1 else entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 - sl_pct) if position == 1 else entry_price * (1 + sl_pct)
        hit_tp = (high[i] >= tp_price) if position == 1 else (low[i] <= tp_price)
        hit_sl = (low[i] <= sl_price) if position == 1 else (high[i] >= sl_price)
        hold_bars = i - entry_bar

        if hit_sl:                       # SL wins any same-bar tie — conservative
            exit_price, reason = sl_price, "SL"
        elif hit_tp:
            exit_price, reason = tp_price, "TP"
        elif hold_bars >= max_hold:
            exit_price, reason = close[i], "TIME"
        else:
            ret[i] = position * (close[i] - prev_close) / prev_close
            i += 1
            continue

        realized = position * (exit_price - prev_close) / prev_close
        cost_frac = (spread[i] / 2.0 + SLIP_USD) / exit_price
        ret[i] = realized - cost_frac
        trades.append(dict(entry_time=df.index[entry_bar], exit_time=df.index[i],
                           side=position, hold_bars=hold_bars, reason=reason,
                           trade_ret=(exit_price / entry_price - 1) * position))
        position = 0
        i += 1

    eq = pd.Series((1.0 + ret), index=df.index).cumprod()
    return pd.Series(ret, index=df.index), eq, pd.DataFrame(trades)


def metrics(ret: pd.Series, eq: pd.Series, trades: pd.DataFrame, eval_start: str) -> dict:
    e = eq.loc[eval_start:]
    if len(e) < 2:
        return dict(sharpe=np.nan, cagr=np.nan, dd=np.nan, ntrades=0)
    e = e / e.iloc[0]
    daily = e.resample("D").last().pct_change(fill_method=None).dropna()
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    tr = trades[(trades.exit_time >= pd.Timestamp(eval_start))]
    win = tr[tr.trade_ret > 0]
    n = len(tr)
    return dict(
        sharpe=float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else np.nan,
        cagr=float(e.iloc[-1] ** (1 / yrs) - 1) * 100,
        dd=float((e / e.cummax() - 1.0).min() * 100),
        ntrades=n, trades_per_yr=n / yrs,
        winrate=float((win.shape[0] / n) * 100) if n else np.nan,
        avg_r=float((tr.trade_ret / tr.trade_ret.where(tr.trade_ret < 0).abs().mean()).mean())
        if n and (tr.trade_ret < 0).any() else np.nan,
        profit_factor=float(win.trade_ret.sum() / -tr[tr.trade_ret <= 0].trade_ret.sum())
        if n and (tr.trade_ret <= 0).any() else np.nan,
        avg_hold=float(tr.hold_bars.mean()) if n else np.nan,
        pct_time_hit=float((tr.reason == "TIME").mean() * 100) if n else np.nan,
    )


def per_year_table(ret: pd.Series, eq: pd.Series, eval_start: str) -> pd.DataFrame:
    e = (eq.loc[eval_start:] / eq.loc[eval_start:].iloc[0])
    daily = e.resample("D").last().pct_change(fill_method=None).dropna()
    rows = []
    for yr, g in daily.groupby(daily.index.year):
        sr = float(g.mean() / g.std() * np.sqrt(252)) if g.std() > 0 else np.nan
        tot = float((1 + g).prod() - 1) * 100
        rows.append(dict(year=yr, sharpe=sr, ret_pct=tot))
    return pd.DataFrame(rows)


def main():
    print("Loading XAUUSD H4 (live $0.448 cost floor) + OOS fwd6/regime probabilities...")
    df0 = load_h4(None)
    years = list(range(EVAL_START_YEAR, df0.index.year.max() + 1))
    p = oos_probabilities(df0, years)
    print(f"  OOS signal: {int(p.notna().sum())} bars, "
          f"{p.first_valid_index().date()}..{p.last_valid_index().date()}\n")

    df = load_h4(0.448)
    sig = np.sign(p.reindex(df.index) - 0.5).fillna(0.0).values

    print(f"Grid: {len(SL_GRID)} SL x {len(TP_GRID)} TP = {len(SL_GRID)*len(TP_GRID)} cells "
          f"(SL {SL_GRID[0]*100:.2f}-{SL_GRID[-1]*100:.2f}%, TP {TP_GRID[0]*100:.2f}-{TP_GRID[-1]*100:.2f}%)\n")
    hdr = (f"{'SL%':>6} {'TP%':>6} {'n/yr':>6} {'win%':>6} {'PF':>6} {'avgR':>6} {'|':>1} "
          f"{'SR':>7} {'CAGR':>7} {'DD':>7} {'avgHold':>8} {'%time':>6}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for sl in SL_GRID:
        for tp in TP_GRID:
            ret, eq, trades = backtest_bracket(df, sig, sl, tp)
            m = metrics(ret, eq, trades, PNL_EVAL_START)
            m.update(sl=sl, tp=tp)
            rows.append(m)
            print(f"{sl*100:6.2f} {tp*100:6.2f} {m['trades_per_yr']:6.0f} {m['winrate']:6.1f} "
                  f"{m['profit_factor']:6.2f} {m['avg_r']:6.2f} {'|':>1} "
                  f"{m['sharpe']:+7.3f} {m['cagr']:+6.1f}% {m['dd']:+6.1f}% "
                  f"{m['avg_hold']:8.1f} {m['pct_time_hit']:5.1f}%")

    R = pd.DataFrame(rows)
    out = ROOT / "data" / "v5_runs" / "turning_predict" / "bracket_gridsearch.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(out, index=False)

    top = R.sort_values("sharpe", ascending=False).head(5)
    print("\nTOP 5 CELLS BY SHARPE:")
    print(top[["sl", "tp", "sharpe", "cagr", "dd", "ntrades", "winrate", "profit_factor"]]
         .to_string(index=False))

    print("\nPER-YEAR BREAKDOWN of the #1 cell (robustness check — is this one window or real?):")
    best = top.iloc[0]
    ret, eq, trades = backtest_bracket(df, sig, best.sl, best.tp)
    print(f"  SL={best.sl*100:.2f}%  TP={best.tp*100:.2f}%")
    print(per_year_table(ret, eq, PNL_EVAL_START).to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
