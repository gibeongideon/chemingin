"""Champion, wrapped for Maven's 0% daily-loss rule — day-open entry,
profit-lock exit, forced EOD flatten. XAU H1, first honest backtest.

Maven's daily-loss window resets 00:00 UTC, based on the higher of
equity/balance at that instant (confirmed directly from maventrading.com's
FAQ, 2026-08-28). The user's specific account states a 0% daily-loss
tolerance (stricter than Maven's public 4%-for-Two-Step figure — trusted as
account-specific, not second-guessed here). A literal 0%-loss-EVER guarantee
is not achievable by any strategy carrying real market risk; this backtest's
job is to be honest about how close a disciplined structure gets, not to
pretend the constraint is fully solved.

Structure (does NOT touch the champion's own signal — reuses
`champion_signal` verbatim, wraps it in a new entry/exit discipline):
  1. Once per UTC day: read the champion's H4 forecast as of the last H4 bar
     BEFORE that day starts (causal). Skip the day entirely (flat, zero risk)
     if the forecast is below `fc_threshold` — no reason to risk anything on
     a weak/neutral signal.
  2. Enter LONG at the day's first H1 bar's open.
  3. Tight protective stop `sl_pct` below entry (far tighter than the
     champion's normal 3xATR trail) — caps any single bad day to a small,
     bounded loss instead of a real drawdown.
  4. Take-profit `tp_pct` above entry — closes and locks in the gain the
     moment the day is "meaningfully positive," per the user's own framing.
  5. Forced flatten at the day's last H1 bar (23:00 UTC) if neither TP nor
     SL fired — NEVER carries a position across the 00:00 UTC reset boundary.

Every day therefore resolves to exactly one of: {skipped (flat), stopped out
(small loss), took profit (gain), forced-flat (whatever the close print is)}.
Reports the ACTUAL realized-loss-day frequency/magnitude honestly — this is
the number that matters for a 0%-tolerance account, not a Sharpe ratio.

    python scripts/v5_maven_zero_dd_champion.py
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

from src.v5.xau_dual_signals import champion_signal  # noqa: E402
from src.evaluation.lookahead_probe import probe_lookahead, print_report  # noqa: E402
from src.evaluation.dsr_pbo import deflated_sharpe_ratio, pbo_cscv  # noqa: E402
from src.evaluation.walk_forward_grid import walk_forward_select, print_selections  # noqa: E402

EVAL_START = "2018-01-01"
YEARS = list(range(2018, 2027))
SELECT_WINDOW_BARS = 3 * 365          # trailing 3y, one row/day

# Maven live spread/slippage not yet measured — conservative placeholder
# matching/exceeding FTMO's documented $0.45, flagged explicitly, MUST be
# re-verified against Maven's actual live quote before this goes live.
SPREAD_USD_PLACEHOLDER = 0.50
SLIP_USD = 0.10

FC_THRESHOLDS = (0.15, 0.30, 0.50)
# Wide-to-none stop range added deliberately: a TIGHT stop caps loss SIZE but
# raises loss FREQUENCY (ordinary intraday noise crosses it constantly) —
# backwards for a rule that cares about frequency, not size. 0.05 = 5%, far
# beyond any single day's realistic move, i.e. "no stop" in practice.
SL_PCTS = (0.0015, 0.0025, 0.004, 0.006, 0.010, 0.020, 0.050)
TP_PCTS = (0.0015, 0.0025, 0.004, 0.006, 0.010)


def load_h4() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "XAUUSD_H4_long.csv", parse_dates=["time"], index_col="time").sort_index()
    return df[~df.index.duplicated(keep="last")]


def load_h1() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "XAUUSD_H1_long.csv", parse_dates=["time"], index_col="time").sort_index()
    return df[~df.index.duplicated(keep="last")]


def daily_forecast(h4: pd.DataFrame) -> pd.Series:
    """champion_signal on H4 closes, resampled to ONE causal value per UTC
    calendar day: the forecast as of the last H4 bar strictly BEFORE that
    day (shift(1) at the daily boundary — the day's decision can only use
    information from before the day starts)."""
    fc = champion_signal(h4["close"])
    day = h4.index.normalize()
    last_of_prior_day = fc.groupby(day).last().shift(1)
    return last_of_prior_day  # index = day (normalized), value = that day's forecast


def simulate_day(h1_day: pd.DataFrame, direction: float, sl_pct: float, tp_pct: float,
                 spread_usd: float) -> tuple[float, str]:
    """One day's H1 bars for the whole UTC day. Returns (net_return, outcome)."""
    if len(h1_day) == 0 or direction <= 0:
        return 0.0, "skip"
    o = h1_day["open"].values
    h = h1_day["high"].values
    l = h1_day["low"].values
    c = h1_day["close"].values
    entry = o[0]
    if not np.isfinite(entry) or entry <= 0:
        return 0.0, "skip"
    cost_frac = (spread_usd / 2.0 + SLIP_USD) / entry
    sl = entry * (1 - sl_pct)
    tp = entry * (1 + tp_pct)
    exit_px, outcome = c[-1], "eod_flat"
    for k in range(len(h1_day)):
        if l[k] <= sl:
            exit_px, outcome = sl, "stopped"
            break
        if h[k] >= tp:
            exit_px, outcome = tp, "profit"
            break
    net = (exit_px - entry) / entry - 2 * cost_frac
    return net, outcome


def run_grid(h4: pd.DataFrame, h1: pd.DataFrame) -> dict:
    fc_daily = daily_forecast(h4)
    day_groups = {d: g for d, g in h1.groupby(h1.index.normalize())}
    grid_rets = {}
    grid_outcomes = {}
    for thr in FC_THRESHOLDS:
        for sl in SL_PCTS:
            for tp in TP_PCTS:
                rets, outs, idx = [], [], []
                for d, g in day_groups.items():
                    fc_val = fc_daily.get(d, np.nan)
                    direction = 1.0 if (np.isfinite(fc_val) and fc_val >= thr) else 0.0
                    net, outcome = simulate_day(g, direction, sl, tp, SPREAD_USD_PLACEHOLDER)
                    rets.append(net); outs.append(outcome); idx.append(d)
                s = pd.Series(rets, index=pd.DatetimeIndex(idx))
                grid_rets[(thr, sl, tp)] = s
                grid_outcomes[(thr, sl, tp)] = outs
    return grid_rets, grid_outcomes


def main():
    print("=== STAGE 0: lookahead probe ===")

    def probe_wrapper(d: pd.DataFrame) -> pd.Series:
        """`daily_forecast` returns one row/day; broadcast back onto the
        original per-bar index (same day's decision value on every bar of
        that day) so `probe_lookahead` can compare at the bar-index positions
        it picks."""
        daily = daily_forecast(d)
        vals = daily.reindex(d.index.normalize()).values
        return pd.Series(vals, index=d.index)

    result = probe_lookahead(probe_wrapper, n=3000, offsets=(200, 120, 60, 30), warmup_buffer=300)
    print_report({"daily_forecast(champion)": result})
    if not result.ok:
        print("!! lookahead probe FAILED — refusing to trust the backtest below.")
        return

    h4 = load_h4()
    h1 = load_h1()
    h1 = h1.loc[EVAL_START:]

    print(f"\n=== STAGE 1: precompute grid (spread placeholder ${SPREAD_USD_PLACEHOLDER}, NOT yet Maven-verified) ===")
    grid_rets, grid_outcomes = run_grid(h4, h1)
    print(f"  grid size: {len(grid_rets)}   trading days in window: {len(next(iter(grid_rets.values())))}")

    full_sh = {k: (float(v.mean() / v.std() * np.sqrt(252)) if v.std() > 0 else -np.inf)
              for k, v in grid_rets.items()}
    best_full = max(full_sh, key=full_sh.get)
    print(f"(reference only, NOT trusted) full-sample-best: thr={best_full[0]} sl={best_full[1]:.3%} "
          f"tp={best_full[2]:.3%}  SR={full_sh[best_full]:+.3f}")

    loss_freq = {k: float((v < 0).mean()) for k, v in grid_rets.items()}
    print("\nLOWEST loss-day-frequency cells (the metric that actually matters for a 0% rule):")
    for k in sorted(loss_freq, key=loss_freq.get)[:10]:
        v = grid_rets[k]
        print(f"  thr={k[0]} sl={k[1]:>6.2%} tp={k[2]:>6.2%}  loss_freq={loss_freq[k]*100:5.1f}%  "
              f"SR={full_sh[k]:+.3f}  mean_loss={v[v<0].mean()*100 if (v<0).any() else 0:+.3f}%  "
              f"total_ret={float((1+v).prod()-1)*100:+.1f}%")

    print("\n=== STAGE 2: WALK-FORWARD selection (trailing 3y, re-picked each Jan) ===")
    oos, selections = walk_forward_select(grid_rets, YEARS, SELECT_WINDOW_BARS, min_train_bars=100)
    print_selections(selections)
    if oos.empty:
        print("!! no OOS data.")
        return
    sr_wf = float(oos.mean() / oos.std() * np.sqrt(252)) if oos.std() > 0 else float("nan")
    e = (1 + oos).cumprod()
    dd_wf = float((e / e.cummax() - 1).min() * 100)
    print(f"\nWALK-FORWARD: SR={sr_wf:+.3f}  DD={dd_wf:+.1f}%  total_ret={float(e.iloc[-1]-1)*100:+.1f}%")

    print("\n=== STAGE 3: THE NUMBER THAT MATTERS — realized daily-loss-day frequency/magnitude ===")
    loss_days = oos[oos < 0]
    n_days = len(oos)
    n_loss = len(loss_days)
    n_skip = int((oos == 0).sum())
    n_win = n_days - n_loss - n_skip
    print(f"  total days: {n_days}   skipped (no trade): {n_skip}   winning days: {n_win}   LOSING days: {n_loss}")
    print(f"  P(any given day is a realized loss): {n_loss/n_days*100:.1f}%")
    if n_loss:
        print(f"  worst single-day loss: {loss_days.min()*100:.3f}%   mean loss-day size: {loss_days.mean()*100:.3f}%")
    if n_win:
        win_days = oos[oos > 0]
        print(f"  mean winning-day size: {win_days.mean()*100:.3f}%")

    # "days until first violation" — the practical, honest metric for a 0%-tolerance account
    first_loss_gaps = []
    since_last = 0
    for v in oos.values:
        since_last += 1
        if v < 0:
            first_loss_gaps.append(since_last)
            since_last = 0
    if first_loss_gaps:
        print(f"\n  trading days between realized losses: mean={np.mean(first_loss_gaps):.1f}  "
              f"median={np.median(first_loss_gaps):.0f}  min={min(first_loss_gaps)}  "
              f"(n={len(first_loss_gaps)} loss events over {n_days} days)")
    else:
        print("\n  ZERO realized-loss days in the entire walk-forward window (verify this isn't an artifact).")

    trial_sharpes = np.array([s / np.sqrt(252) for s in full_sh.values() if np.isfinite(s)])
    dsr = deflated_sharpe_ratio(oos.values, trial_sharpes)
    common_idx = None
    for v in grid_rets.values():
        common_idx = v.index if common_idx is None else common_idx.intersection(v.index)
    mats = [v.reindex(common_idx).fillna(0.0).values for v in grid_rets.values()]
    pbo_res = pbo_cscv(np.column_stack(mats), n_partitions=10)
    print(f"\nDSR={dsr['dsr']:.3f} (n_trials={dsr['n_trials']})   PBO={pbo_res.pbo:.3f}")

    print("\nper-year:")
    for yr, g in oos.groupby(oos.index.year):
        n_l = int((g < 0).sum())
        n_t = len(g)
        print(f"  {yr}  days={n_t}  loss_days={n_l} ({n_l/n_t*100:.0f}%)  "
              f"total_ret={float((1+g).prod()-1)*100:+.1f}%")

    out = ROOT / "data" / "v5_runs" / "maven_zero_dd_champion_summary.csv"
    pd.DataFrame({"date": oos.index, "ret": oos.values}).to_csv(out, index=False)
    print(f"\ndaily series -> {out}")


if __name__ == "__main__":
    main()
