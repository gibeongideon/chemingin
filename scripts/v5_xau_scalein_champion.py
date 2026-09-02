"""Champion + adverse-excursion SCALE-IN — backtest of 54 variants vs the
live cent-account champion. XAUUSD H4.

User ask (2026-08-31): "the Champion trend follower that opens position if the
trade goes to a negative enough of what was expected such that when a trade
starts going positive, it complements existing — then backtest several
variants and report results. We already have the trend follower running on
the HFM cent account, work from there."

BASE = exactly what runs live on the cent account (magic 360542):
`xau_trend.run_trades(exit_mode="trail", flip_mode="confidence")` with
`champion_signal` patched in, sl_atr=trail_atr=3.0, conf_risk_scale
{0.5,1.0,1.5}, risk_frac 1%. Documented deploy-variant eval Sharpe 0.968 /
DD -10.0%.

VARIANT = `xau_trend_scalein.run_trades_scalein` — same entries, same trailing
exit, but adds legs when the open trade goes adverse by a fraction of its own
stop distance. Grid (54 cells, ALL counted in the DSR trial set):
    add_trigger_frac  0.20 / 0.35 / 0.50   (of the stop distance, adverse)
    max_adds          1 / 2 / 3
    add_size_mult     0.5 / 1.0            (relative to the first leg)
    stop_policy       risk_constant / original / preallocated

STAGE 0 is a REGRESSION CHECK, not a formality: the variant engine with
max_adds=0 must reproduce the base engine's equity curve and trade list
exactly. If it does not, every number after it is measuring an engine
difference rather than the scale-in idea, so the script refuses to continue.

Reported honestly, including the parts that averaging-down is known to hurt:
max drawdown, worst single trade in R, and the paired daily-return t-stat vs
the base. Prior art in this repo that this must be read against:
`martingale-predictor-bot` (memory) — stake-doubling AFTER realized losses
ruined 71-97% of runs even on a positive-edge signal. This is a different
structure (scaling into an OPEN position on a pullback, inside a signal with
a genuine documented edge) but it lives in the same risk family and gets the
same scepticism.

    python scripts/v5_xau_scalein_champion.py
    python scripts/v5_xau_scalein_champion.py --spread-usd 0.448
"""
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.v5.xau_trend as xt  # noqa: E402
from src.v5.xau_trend_scalein import run_trades_scalein  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402
from src.evaluation.dsr_pbo import deflated_sharpe_ratio, pbo_cscv  # noqa: E402
from src.evaluation.walk_forward_grid import (  # noqa: E402
    walk_forward_select, regime_split_sharpe, print_selections, print_regime_split,
)

EVAL_START = "2018-01-01"
YEARS = list(range(2018, 2027))
SELECT_WINDOW_BARS = 3 * 365          # trailing 3y of DAILY rows
EQUITY0 = 100_000.0                   # scale-free for Sharpe/DD; matches the Maven/FTMO size

BASE_PARAMS = dict(sl_atr=3.0, trail_atr=3.0,
                   conf_risk_scale={"low": 0.5, "med": 1.0, "high": 1.5})

TRIGGERS = (0.20, 0.35, 0.50)
MAX_ADDS = (1, 2, 3)
SIZE_MULTS = (0.5, 1.0)
STOP_POLICIES = ("risk_constant", "original", "preallocated")

# XAU regime boundaries — same splits used throughout this repo's XAU work.
XAU_REGIMES = [
    ("2015-01-01", "2018-12-31", "chop"),
    ("2019-01-01", "2020-12-31", "bull"),
    ("2021-01-01", "2022-12-31", "flat"),
    ("2023-01-01", "2026-12-31", "bull"),
]


def load_h4(spread_usd: float | None) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "XAUUSD_H4_long.csv",
                     parse_dates=["time"], index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if spread_usd is not None:
        # engine reads df["spread"] in pips (pip = 0.1 USD for XAUUSD)
        df["spread"] = np.maximum(df["spread"].values, spread_usd / 0.1)
    return df


def daily_returns(equity: pd.Series) -> pd.Series:
    e = equity.dropna()
    e = e.loc[EVAL_START:]
    if e.empty:
        return pd.Series(dtype=float)
    d = e.resample("D").last().dropna()
    return d.pct_change().dropna()


def metrics(equity: pd.Series, trades: pd.DataFrame) -> dict:
    e = equity.dropna().loc[EVAL_START:]
    r = daily_returns(equity)
    if e.empty or r.empty or r.std() == 0:
        return dict(sharpe=np.nan, dd=np.nan, cagr=np.nan, total=np.nan,
                    n_trades=0, win_pct=np.nan, worst_r=np.nan, mean_legs=np.nan)
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    tr = trades.copy()
    if not tr.empty and "close_time" in tr:
        tr = tr[pd.to_datetime(tr["close_time"]) >= pd.Timestamp(EVAL_START)]
    dd = float((e / e.cummax() - 1).min() * 100)
    cagr = float(((e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1) * 100) if yrs > 0 else np.nan
    return dict(
        sharpe=float(r.mean() / r.std() * np.sqrt(252)),
        dd=dd,
        cagr=cagr,
        # Calmar is the decisive column: adding size on adverse moves raises
        # BOTH return and drawdown. If CAGR/|DD| is unchanged, the variant is
        # leverage, not edge.
        calmar=float(cagr / abs(dd)) if dd else np.nan,
        total=float((e.iloc[-1] / e.iloc[0] - 1) * 100),
        n_trades=int(len(tr)),
        win_pct=float((tr["pnl"] > 0).mean() * 100) if len(tr) else np.nan,
        worst_r=float(tr["r_multiple"].min()) if len(tr) else np.nan,
        mean_legs=float(tr["legs"].mean()) if len(tr) and "legs" in tr else 1.0,
    )


def paired_t(a: pd.Series, b: pd.Series) -> tuple[float, int, int]:
    j = a.index.intersection(b.index)
    d = (a.reindex(j) - b.reindex(j)).dropna()
    if len(d) < 30 or d.std() == 0:
        return 0.0, 0, 0
    t = float(d.mean() / d.std() * np.sqrt(len(d)))
    g = d.groupby(d.index.year).mean()
    return t, int((g > 0).sum()), int(len(g))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spread-usd", type=float, default=None,
                    help="floor the CSV spread at this live round-trip USD "
                         "(0.448 FTMO / 0.34 cent). Default: CSV median floor, "
                         "matching how the deployed champion was validated.")
    args = ap.parse_args()

    df = load_h4(args.spread_usd)
    cost_label = (f"live-floored ${args.spread_usd}" if args.spread_usd
                  else "CSV median floor (as deployed)")
    print(f"XAUUSD H4 {df.index[0].date()}..{df.index[-1].date()}  cost: {cost_label}")
    print(f"eval from {EVAL_START}, equity0 ${EQUITY0:,.0f}\n")

    # ---------------- STAGE 0: regression — variant(max_adds=0) == base
    print("=== STAGE 0: regression check (variant with adds OFF must equal the base) ===")
    xt.xau_signal = champion_signal          # exactly what v5_xau_dual.py does
    base = xt.run_trades(df, equity0=EQUITY0, exit_mode="trail",
                         flip_mode="confidence", params=BASE_PARAMS)
    mirror = run_trades_scalein(df, signal_fn=champion_signal, equity0=EQUITY0,
                                params={**BASE_PARAMS, "max_adds": 0})
    eq_diff = float((base["equity"].dropna() - mirror["equity"].dropna()).abs().max())
    n_base, n_mirror = len(base["trades"]), len(mirror["trades"])
    print(f"  base trades {n_base}  mirror trades {n_mirror}   "
          f"max |equity diff| = {eq_diff:.6f}")
    if n_base != n_mirror or eq_diff > 1e-6:
        print("!! REGRESSION FAILED — the variant engine does not reproduce the base. "
              "Refusing to report scale-in numbers that would be measuring the engine.")
        return
    print("  PASS — engines are identical with adds off.\n")

    base_m = metrics(base["equity"], base["trades"])
    base_daily = daily_returns(base["equity"])
    print(f"BASE champion (live cent config): SR={base_m['sharpe']:+.3f}  "
          f"DD={base_m['dd']:+.1f}%  CAGR={base_m['cagr']:+.2f}%  "
          f"Calmar={base_m['calmar']:.3f}  total={base_m['total']:+.0f}%  "
          f"trades={base_m['n_trades']}  win={base_m['win_pct']:.0f}%  "
          f"worst={base_m['worst_r']:+.1f}R\n")

    # ---- leverage-matched control: the base champion at higher risk_frac.
    # If a scale-in variant only matches THIS, it is reproducing leverage the
    # long way round — obtainable more simply and with far better control by
    # turning one dial, with no averaging-down mechanics to go wrong.
    print("=== leverage-matched control: BASE at higher risk_frac (no scale-in) ===")
    lev_rows = []
    for rf in (0.01, 0.0125, 0.015, 0.0175, 0.02):
        r_lev = xt.run_trades(df, equity0=EQUITY0, exit_mode="trail",
                              flip_mode="confidence",
                              params={**BASE_PARAMS, "risk_frac": rf})
        m = metrics(r_lev["equity"], r_lev["trades"])
        lev_rows.append(dict(risk_frac=rf, **m))
        print(f"  risk_frac {rf:.4f}: SR={m['sharpe']:+.3f}  DD={m['dd']:+.1f}%  "
              f"CAGR={m['cagr']:+.2f}%  Calmar={m['calmar']:.3f}  "
              f"worst={m['worst_r']:+.1f}R")
    LEV = pd.DataFrame(lev_rows)
    print()

    # ---------------- STAGE 1: the 36-cell variant grid
    print("=== STAGE 1: scale-in grid (54 variants) ===")
    grid_rets: dict[tuple, pd.Series] = {}
    rows = []
    for trig, madds, smult, spol in itertools.product(TRIGGERS, MAX_ADDS, SIZE_MULTS, STOP_POLICIES):
        res = run_trades_scalein(
            df, signal_fn=champion_signal, equity0=EQUITY0,
            params={**BASE_PARAMS, "add_trigger_frac": trig, "max_adds": madds,
                    "add_size_mult": smult, "stop_policy": spol})
        m = metrics(res["equity"], res["trades"])
        r = daily_returns(res["equity"])
        key = (trig, madds, smult, spol)
        grid_rets[key] = r
        t, yp, yn = paired_t(r, base_daily)
        rows.append(dict(trigger=trig, max_adds=madds, size_mult=smult, stop_policy=spol,
                         t_vs_base=t, years_better=f"{yp}/{yn}", **m))
    R = pd.DataFrame(rows)

    print(f"{'trig':>5} {'adds':>4} {'sz':>4} {'stop_policy':>14} {'SR':>7} {'DD':>7} "
          f"{'CAGR':>7} {'Calmar':>6} {'win%':>5} {'worstR':>7} {'legs':>5} {'t_vs_base':>9}")
    for _, x in R.sort_values("sharpe", ascending=False).iterrows():
        print(f"{x.trigger:>5.2f} {x.max_adds:>4d} {x.size_mult:>4.1f} {x.stop_policy:>14} "
              f"{x.sharpe:>+7.3f} {x.dd:>+7.1f} {x.cagr:>+7.2f} {x.calmar:>6.3f} "
              f"{x.win_pct:>5.0f} {x.worst_r:>+7.1f} {x.mean_legs:>5.2f} {x.t_vs_base:>+9.2f}")

    n_better = int((R["sharpe"] > base_m["sharpe"]).sum())
    n_sig = int((R["t_vs_base"] > 1.64).sum())
    print(f"\n  variants beating base Sharpe: {n_better}/54   "
          f"paired-t > +1.64 (one-sided 95%): {n_sig}/54")
    print(f"  median variant Sharpe {R.sharpe.median():+.3f} vs base {base_m['sharpe']:+.3f}"
          f"   median DD {R.dd.median():+.1f}% vs base {base_m['dd']:+.1f}%")
    for spol in STOP_POLICIES:
        sub = R[R.stop_policy == spol]
        print(f"  {spol:>14}: median SR {sub.sharpe.median():+.3f}  "
              f"median DD {sub.dd.median():+.1f}%  worst DD {sub.dd.min():+.1f}%  "
              f"worst single trade {sub.worst_r.min():+.1f}R  "
              f"median Calmar {sub.calmar.median():.3f}")

    # ---- the leverage question, answered head-on
    print(f"\n  CALMAR (return per unit of drawdown) — base {base_m['calmar']:.3f}, "
          f"variants median {R.calmar.median():.3f}, best {R.calmar.max():.3f}")
    n_calmar_better = int((R["calmar"] > base_m["calmar"]).sum())
    print(f"  variants with BETTER Calmar than base: {n_calmar_better}/54")
    top_cagr = R.sort_values("cagr", ascending=False).iloc[0]
    lev_match = LEV.iloc[(LEV["cagr"] - top_cagr["cagr"]).abs().argsort()].iloc[0]
    print(f"  highest-CAGR variant: CAGR {top_cagr['cagr']:+.2f}% @ DD {top_cagr['dd']:+.1f}% "
          f"(Calmar {top_cagr['calmar']:.3f}, SR {top_cagr['sharpe']:+.3f})")
    print(f"  base at matched CAGR:  CAGR {lev_match['cagr']:+.2f}% @ DD {lev_match['dd']:+.1f}% "
          f"(Calmar {lev_match['calmar']:.3f}, SR {lev_match['sharpe']:+.3f}, "
          f"risk_frac {lev_match['risk_frac']:.4f})")

    # ---------------- STAGE 2: walk-forward selection over the grid
    print("\n=== STAGE 2: WALK-FORWARD selection over the 54 variants (trailing 3y) ===")
    oos, sel = walk_forward_select(grid_rets, YEARS, SELECT_WINDOW_BARS, min_train_bars=100)
    print_selections(sel)
    if oos.empty or oos.std() == 0:
        print("!! no OOS series")
        return
    sr_wf = float(oos.mean() / oos.std() * np.sqrt(252))
    e_wf = (1 + oos).cumprod()
    dd_wf = float((e_wf / e_wf.cummax() - 1).min() * 100)
    t_wf, yp_wf, yn_wf = paired_t(oos, base_daily)
    print(f"\nWALK-FORWARD scale-in: SR={sr_wf:+.3f}  DD={dd_wf:+.1f}%")
    print(f"BASE champion, same window: SR={base_m['sharpe']:+.3f}  DD={base_m['dd']:+.1f}%")
    print(f"paired t vs base = {t_wf:+.2f}   years better {yp_wf}/{yn_wf}")

    trials = np.array([s / np.sqrt(252) for s in R["sharpe"] if np.isfinite(s)])
    dsr = deflated_sharpe_ratio(oos.values, trials)
    common = None
    for v in grid_rets.values():
        common = v.index if common is None else common.intersection(v.index)
    mats = [v.reindex(common).fillna(0.0).values for v in grid_rets.values()]
    pbo = pbo_cscv(np.column_stack(mats), n_partitions=10)
    print(f"DSR={dsr['dsr']:.3f} (n_trials={dsr['n_trials']})   PBO={pbo.pbo:.3f}")

    print("\nregime split (walk-forward scale-in vs base):")
    reg_v = regime_split_sharpe(oos, XAU_REGIMES)
    reg_b = regime_split_sharpe(base_daily, XAU_REGIMES)
    for k in reg_v:
        print(f"  {k:24s} scale-in {reg_v[k]:+.2f}   base {reg_b.get(k, float('nan')):+.2f}")

    out = ROOT / "data" / "v5_runs" / "xau_scalein_champion_grid.csv"
    R.to_csv(out, index=False)
    print(f"\nfull grid -> {out}")


if __name__ == "__main__":
    main()
