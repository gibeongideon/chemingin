"""risk_frac — what the dial actually is, and what raising it does. XAUUSD H4 champion.

Follow-up to V5_FINDINGS §3y (scale-in is leverage, not edge): if the honest way to
get more return from the champion is "turn risk_frac up deliberately", then quantify
exactly what that buys and what it costs.

WHERE THE DIAL LIVES
  `src/v5/xau_trend.PARAMS["risk_frac"]` = 0.01 default (1% of CURRENT equity risked
  over the 3xATR stop, per trade), overridable per-bot in the config JSON and read by
  the executor at `scripts/v5_xau_dual.py:114`:
      risk = bot_cfg["params"].get("risk_frac", xt.PARAMS["risk_frac"]) \
             * bot_cfg["params"]["conf_risk_scale"][conf]
  So the effective risk per trade is risk_frac x conf_scale (0.5/1.0/1.5), converted
  to lots via `order_calc_profit` on the live symbol. Currently deployed:
      configs/v5_xau_dual.json          (cent)  -> no override => 1.0%
      configs/v5_xau_champ_fundingpips.json     -> 0.5%  ("do NOT raise it")
      configs/v5_xau_maven.json                 -> 0.5%
  Changing it is a one-line config edit; nothing about the SIGNAL or trade timing
  changes, which is exactly why it is a clean dial (see the Sharpe-invariance column).

WHAT IS MEASURED
  1. Historical sweep: Sharpe / CAGR / maxDD / Calmar / worst trade / lots / peak margin.
  2. Maven challenge sim (4% target, 10% max loss) via the repo's own block-bootstrap
     `fp_sim` (data/v5_runs/challenge-lab/challenge_lab.py) — run BOTH with the daily
     rule off (the user closes manually for the 0% rule) and at Maven's public 4% daily.
  3. FUNDED-stage survival over 12 months with no profit target, reported under BOTH
     drawdown conventions (static from starting balance = Maven/FTMO's rule; trailing
     peak-to-trough = the stricter high-water reading). The number that matters once a
     challenge is passed and the account just has to stay alive.

Prior art, related but NOT the same question (`configs/v5_xau_champ_fundingpips.json`):
a single-XAU champion vs a 10% wall breached ~19% at risk_frac 1% and ~2.9% at 0.5% —
measured over a full 8%+5% two-step (median ~20 months of exposure) on the FP micro
config, so it is strictly harsher than the 12-month window here. The ORDERING agrees.

    python scripts/v5_xau_riskfrac_sweep.py
    python scripts/v5_xau_riskfrac_sweep.py --spread-usd 0.448
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "v5_runs" / "challenge-lab"))

import src.v5.xau_trend as xt  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402
from challenge_lab import fp_sim  # noqa: E402

EVAL_START = "2018-01-01"
EQUITY0 = 100_000.0
LEVERAGE = 75.0            # Maven account, verified from account_info 2026-08-28
BASE_PARAMS = dict(sl_atr=3.0, trail_atr=3.0,
                   conf_risk_scale={"low": 0.5, "med": 1.0, "high": 1.5})
RISK_FRACS = (0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03, 0.04)


def load_h4(spread_usd: float | None) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "XAUUSD_H4_long.csv",
                     parse_dates=["time"], index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if spread_usd is not None:
        df["spread"] = np.maximum(df["spread"].values, spread_usd / 0.1)
    return df


def daily_returns(equity: pd.Series) -> pd.Series:
    e = equity.dropna().loc[EVAL_START:]
    if e.empty:
        return pd.Series(dtype=float)
    return e.resample("D").last().dropna().pct_change().dropna()


def survival_breach_prob(r: np.ndarray, horizon: int = 252, maxloss: float = 0.10,
                         nsim: int = 4000, block: int = 20, seed: int = 7) -> tuple[float, float]:
    """Block-bootstrap survival over `horizon` days with no profit target.

    Returns (static_pct, trailing_pct) because the two prop-firm conventions
    give very different answers and conflating them is how a book gets sized
    too big:
      static   — equity falls `maxloss` below its STARTING balance. This is
                 Maven's stated rule ("10% total loss from your starting
                 balance") and FTMO's overall wall.
      trailing — peak-to-trough drawdown reaches `maxloss` at any point. The
                 stricter reading, used by firms with trailing/high-water
                 drawdown, and the honest answer to "how bad will this feel".
    """
    rng = np.random.default_rng(seed)
    n = len(r)
    static = trailing = 0
    for _ in range(nsim):
        idx: list[int] = []
        while len(idx) < horizon:
            s = rng.integers(0, n)
            L = min(rng.geometric(1 / block), horizon - len(idx))
            idx.extend([(s + j) % n for j in range(L)])
        eq = np.cumprod(1.0 + r[np.array(idx)])
        if eq.min() < (1.0 - maxloss):
            static += 1
        if (eq / np.maximum.accumulate(np.concatenate(([1.0], eq)))[1:]).min() < (1.0 - maxloss):
            trailing += 1
    return round(static / nsim * 100, 1), round(trailing / nsim * 100, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spread-usd", type=float, default=None)
    args = ap.parse_args()

    df = load_h4(args.spread_usd)
    xt.xau_signal = champion_signal
    cost = f"live-floored ${args.spread_usd}" if args.spread_usd else "CSV median floor"
    print(f"XAUUSD H4 champion, eval {EVAL_START}+, equity0 ${EQUITY0:,.0f}, cost {cost}")
    print(f"leverage 1:{LEVERAGE:.0f} (Maven), conf_risk_scale {BASE_PARAMS['conf_risk_scale']}\n")

    rows = []
    daily_by_rf = {}
    for rf in RISK_FRACS:
        res = xt.run_trades(df, equity0=EQUITY0, exit_mode="trail",
                            flip_mode="confidence",
                            params={**BASE_PARAMS, "risk_frac": rf})
        e = res["equity"].dropna().loc[EVAL_START:]
        r = daily_returns(res["equity"])
        daily_by_rf[rf] = r
        tr = res["trades"]
        tr = tr[pd.to_datetime(tr["close_time"]) >= pd.Timestamp(EVAL_START)]
        yrs = (e.index[-1] - e.index[0]).days / 365.25
        dd = float((e / e.cummax() - 1).min() * 100)
        cagr = float(((e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1) * 100)
        # margin: lots x contract x price / leverage, as % of equity at that time
        px = df["close"].reindex(pd.to_datetime(tr["open_time"])).values
        notional = tr["lots"].values * 100.0 * px
        margin_pct = np.nanmax(notional / LEVERAGE) / EQUITY0 * 100
        rows.append(dict(
            risk_frac=rf,
            sharpe=float(r.mean() / r.std() * np.sqrt(252)),
            cagr=cagr, dd=dd, calmar=cagr / abs(dd),
            final=float(e.iloc[-1]),
            worst_day_usd=float(r.min() * EQUITY0),
            worst_r=float(tr["r_multiple"].min()),
            mean_lots=float(tr["lots"].mean()), max_lots=float(tr["lots"].max()),
            peak_margin_pct=float(margin_pct),
        ))
    S = pd.DataFrame(rows)

    print("=== 1) HISTORICAL sweep (same signal, same trades — only size changes) ===")
    print(f"{'risk':>6} {'SR':>7} {'CAGR%':>7} {'maxDD%':>7} {'Calmar':>7} "
          f"{'final$':>10} {'worstDay$':>10} {'worstR':>7} {'lots(avg/max)':>14} {'margin%':>8}")
    for _, x in S.iterrows():
        print(f"{x.risk_frac*100:>5.2f}% {x.sharpe:>+7.3f} {x.cagr:>+7.2f} {x.dd:>+7.1f} "
              f"{x.calmar:>7.3f} {x.final:>10,.0f} {x.worst_day_usd:>10,.0f} "
              f"{x.worst_r:>+7.1f} {x.mean_lots:>6.2f}/{x.max_lots:<7.2f} {x.peak_margin_pct:>7.2f}%")

    print("\n  Sharpe is INVARIANT to risk_frac (identical signal/trades, size scales "
          "linearly)\n  -> risk_frac buys return and drawdown in the SAME proportion. "
          "It is a pure risk dial,\n     not an edge lever. Calmar drifts up only from "
          "compounding, not from skill.")

    print("\n=== 2) MAVEN CHALLENGE sim (4% target, 10% max loss; block-bootstrap fp_sim) ===")
    print(f"{'risk':>6} | {'daily rule OFF (manual close)':>32} | {'daily 4% (Maven public)':>30}")
    print(f"{'':>6} | {'pass%':>7} {'fail_dd%':>9} {'med_mo':>7} | {'pass%':>7} "
          f"{'fail_day%':>10} {'fail_dd%':>9}")
    sim_rows = []
    for rf in RISK_FRACS:
        r = daily_by_rf[rf].values
        off = fp_sim(r, 1.0, p1=0.04, p2=0.0, dayloss=0.99, maxloss=0.10, day_safety=1.0)
        on4 = fp_sim(r, 1.0, p1=0.04, p2=0.0, dayloss=0.04, maxloss=0.10, day_safety=1.5)
        sim_rows.append(dict(risk_frac=rf, pass_off=off["passpct"], dd_off=off["fail_dd"],
                             mo_off=off["med_mo"], pass_on4=on4["passpct"],
                             day_on4=on4["fail_day"], dd_on4=on4["fail_dd"]))
        print(f"{rf*100:>5.2f}% | {off['passpct']:>7.1f} {off['fail_dd']:>9.1f} "
              f"{off['med_mo']:>7.1f} | {on4['passpct']:>7.1f} {on4['fail_day']:>10.1f} "
              f"{on4['fail_dd']:>9.1f}")
    SIM = pd.DataFrame(sim_rows)

    print("\n=== 3) FUNDED-STAGE survival over 12 months, no profit target ===")
    print(f"{'risk':>6} {'-10% from START':>16} {'-10% peak-to-trough':>21}")
    surv_s, surv_t = [], []
    for rf in RISK_FRACS:
        bs, bt = survival_breach_prob(daily_by_rf[rf].values)
        surv_s.append(bs); surv_t.append(bt)
        print(f"{rf*100:>5.2f}% {bs:>15.1f}% {bt:>20.1f}%")
    S["breach12mo_static_pct"] = surv_s
    S["breach12mo_trailing_pct"] = surv_t
    print("  static = Maven's stated rule (10% of STARTING balance). trailing = the "
          "stricter\n  high-water convention; also the better read on how painful a "
          "given dial feels.")

    print("\n=== cross-check vs configs/v5_xau_champ_fundingpips.json ===")
    print("  documented there: single-XAU champion vs a 10% wall breached ~19% at "
          "risk_frac 1%, ~2.9% at 0.5%.")
    b1s = S.loc[S.risk_frac == 0.01, "breach12mo_static_pct"].iloc[0]
    b1t = S.loc[S.risk_frac == 0.01, "breach12mo_trailing_pct"].iloc[0]
    b05t = S.loc[S.risk_frac == 0.005, "breach12mo_trailing_pct"].iloc[0]
    print(f"  this run at 1.00%: {b1s:.1f}% static / {b1t:.1f}% trailing;  "
          f"at 0.50%: {b05t:.1f}% trailing.")
    print("  NOT a contradiction — different questions. The documented figure runs until "
          "an 8%+5%\n  TWO-STEP target is met or the wall is hit (median ~20 MONTHS at "
          "rf 0.9%, per\n  configs/v5_xau_challenge.json), so it exposes the account to "
          "far more time than a\n  12-month window, and it is measured on the FP micro "
          "contract/config. Longer exposure\n  and a stricter DD convention both push the "
          "number up; the ordering (1% ~6-7x riskier\n  than 0.5%) is what agrees.")

    out = ROOT / "data" / "v5_runs" / "xau_riskfrac_sweep.csv"
    S.merge(SIM, on="risk_frac").to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
