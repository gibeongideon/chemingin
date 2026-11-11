#!/usr/bin/env python3
"""NOISE-BAND INTRADAY MOMENTUM on gold (user, 2026-11-11: "come up with any strategy of any form
and design that can be profitable as intraday, XAUUSD").

SOURCE (outside this repo). Zarattini, Aziz & Barbon 2024, "Beat the Market: An Effective Intraday
Momentum Strategy for S&P500 ETF (SPY)" (SSRN 4824172; reported Sharpe 1.33, 2007-24), itself
built on Gao, Han, Li & Zhou 2018 "Market intraday momentum" (JFE). NOT the ORB already closed in
3dh/3di: there is no opening range and NO INTRABAR STOP, which is exactly what made gold
unmeasurable on M15 in 3di. Every decision is taken on a bar CLOSE and filled at the next bar.

RULE. For a session opening at time T0 (New York clock):
    sigma(t)  = mean over the previous 14 sessions of |close(t) / session_open - 1|   (past only)
    UB(t)     = max(open, previous session close) * (1 + sigma(t))
    LB(t)     = min(open, previous session close) * (1 - sigma(t))
At each check (every 30 min on M15, every 60 on H1): LONG while close > UB, SHORT while close <
LB, FLAT inside the band. Flat at the session close, nothing held overnight. Sized 1/vol.

PRE-REGISTERED FAMILY, 6 cells: sessions {FULL 18:00->16:30, LDN 03:00->16:00, NY 08:00->16:00}
x {LS long/short, LO long-only}. Fixed before the first run; no parameter was tuned.

CONTROLS.
  * cost: $0.30 round trip (Switch live $0.25, HFM $0.34), charged on every change of position;
    2x stress reported.
  * DAY-SHIFT NULL: the same position matrix applied to OTHER days' returns (rolled >= 20
    sessions). Same time in market, same turnover, same cost, same long bias — so it is the drift
    control and the timing control in one (3r: gold rises; a long-only cell must beat that).
    Family p = max-stat over the 6 cells (3bi).
  * long-only cells are also compared with simply holding gold through the same session.
  * two sources: XAUUSD M15 2015-2026/06 (UTC, one feed) and CME GC H1 2010-2026 (Databento).
    A keeper must clear the family null in BOTH.

    python scripts/v5_noise_band_momentum.py m15
    python scripts/v5_noise_band_momentum.py cme
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
OUT = ROOT / "data" / "v5_runs" / "noise_band_momentum"
LOOKBACK, COST_RT, N_NULL = 14, 0.30, 300
SESSIONS = {"FULL": (18 * 60, 22.5 * 60), "LDN": (3 * 60, 13 * 60), "NY": (8 * 60, 8 * 60)}  # start min ET, length min


def load(src: str) -> tuple[pd.DataFrame, int, int]:
    if src == "m15":
        d = pd.read_csv(ROOT / "data/XAUUSD_M15_long.csv", parse_dates=["time"], index_col="time")
        step, check = 15, 30
    else:
        from scripts.v5_zigzag_panel_probs import load_cme  # noqa: PLC0415
        d, step, check = load_cme("GC"), 60, 60
    d = d[~d.index.duplicated(keep="last")].sort_index()[["open", "high", "low", "close"]]
    d.index = d.index.tz_localize("UTC").tz_convert("America/New_York").tz_localize(None)
    return d, step, check


def matrices(d: pd.DataFrame, step: int, start_min: float, length_min: float):
    """Sessions x slots matrices of open/close. A session starting at 18:00 belongs to the NEXT
    calendar day's trading date, so shift the clock by -start before taking the date."""
    rel = d.index - pd.Timedelta(minutes=start_min)
    sess = rel.normalize()
    slot = ((rel - sess).total_seconds() // 60 // step).astype(int)
    keep = slot < int(length_min // step)
    x = d[keep].assign(sess=sess[keep], slot=slot[keep])
    C = x.pivot_table(index="sess", columns="slot", values="close", aggfunc="last")
    O = x.pivot_table(index="sess", columns="slot", values="open", aggfunc="first")
    C = C.reindex(columns=range(int(length_min // step)))
    O = O.reindex(columns=C.columns)
    ok = O[0].notna() & (C.notna().sum(axis=1) >= 0.8 * C.shape[1])       # drop holidays / half days
    return O[ok], C[ok]


def positions(O: pd.DataFrame, C: pd.DataFrame, step: int, check: int) -> np.ndarray:
    open0 = O[0].to_numpy()
    c = C.to_numpy()
    move = np.abs(c / open0[:, None] - 1)
    sigma = pd.DataFrame(move).rolling(LOOKBACK, min_periods=10).mean().shift(1).to_numpy()
    last_close = C.ffill(axis=1).iloc[:, -1].shift(1).to_numpy()
    ub = np.fmax(open0, last_close)[:, None] * (1 + sigma)
    lb = np.fmin(open0, last_close)[:, None] * (1 - sigma)
    sig = np.where(c > ub, 1.0, np.where(c < lb, -1.0, 0.0))
    sig[~np.isfinite(c) | ~np.isfinite(ub)] = np.nan
    is_check = ((np.arange(c.shape[1]) + 1) * step) % check == 0
    sig[:, ~is_check] = np.nan
    pos = pd.DataFrame(sig).ffill(axis=1).fillna(0.0).to_numpy()          # hold between checks
    pos[:, -1] = 0.0                                                      # flat at the session close
    pos[~np.isfinite(sigma).any(axis=1)] = 0.0                            # warm-up sessions
    return pos


def pnl(pos: np.ndarray, C: pd.DataFrame, cost_rt: float) -> np.ndarray:
    """Position taken at slot k's close earns slot k+1's close-to-close return. Cost = half the
    round trip per unit of position change, including the entry from flat and the final exit."""
    c = C.ffill(axis=1).to_numpy()
    ret = np.zeros_like(c)
    ret[:, 1:] = c[:, 1:] / c[:, :-1] - 1
    ret = np.nan_to_num(ret)
    gross = (pos[:, :-1] * ret[:, 1:]).sum(axis=1)
    dpos = np.abs(np.diff(np.column_stack([np.zeros(len(pos)), pos]), axis=1))
    px = np.nanmedian(c, axis=1)
    return gross - dpos.sum(axis=1) * (cost_rt / 2) / px


def sharpe(x) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / x.std() * np.sqrt(252)) if len(x) > 2 and x.std() > 0 else 0.0


def main() -> None:
    src = sys.argv[1]
    d, step, check = load(src)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    rows, nulls, daily = [], {}, {}
    for sname, (st, ln) in SESSIONS.items():
        O, C = matrices(d, step, st, ln)
        c = C.ffill(axis=1).to_numpy()
        sess_ret = c[:, -1] / O[0].to_numpy() - 1
        vol = pd.Series(sess_ret).ewm(span=20, min_periods=10).std().shift(1).to_numpy()
        w = np.nan_to_num(0.01 / vol, nan=0.0, posinf=0.0)                # 1/vol sizing, past only
        w = np.minimum(w, 4.0)
        base = positions(O, C, step, check)
        hold = np.ones_like(base); hold[:, -1] = 0.0
        hold_r = w * pnl(hold, C, COST_RT)
        yrs = C.index.year.to_numpy()
        for mode in ("LS", "LO"):
            pos = base if mode == "LS" else np.maximum(base, 0.0)
            r = w * pnl(pos, C, COST_RT)
            g = w * pnl(pos, C, 0.0)
            s = sharpe(r)
            shifts = rng.integers(20, len(pos) - 20, N_NULL)
            nl = np.array([sharpe(w * pnl(np.roll(pos, int(k), axis=0), C, COST_RT)) for k in shifts])
            name = f"{sname}_{mode}"
            nulls[name] = nl
            daily[name] = pd.Series(r, index=C.index)
            half = len(r) // 2
            active = np.abs(pos).sum(axis=1) > 0
            trades = (np.abs(np.diff(np.column_stack([np.zeros(len(pos)), pos]), axis=1)) > 0).sum() / 2
            dd = r * (hold_r.std() / r.std()) - hold_r if r.std() > 0 else r - hold_r
            rows.append(dict(
                cell=name, sessions=len(r), active_pct=round(100 * active.mean(), 0),
                trades_wk=round(trades / (len(r) / 5), 1), in_mkt=round(float(np.abs(pos).mean()), 2),
                hit_days=round(100 * (r[active] > 0).mean(), 1),
                SR_gross=round(sharpe(g), 2), SR=round(s, 2), SR_2x=round(sharpe(w * pnl(pos, C, 2 * COST_RT)), 2),
                h1=round(sharpe(r[:half]), 2), h2=round(sharpe(r[half:]), 2),
                yrs_pos=f"{sum(sharpe(r[yrs == y]) > 0 for y in np.unique(yrs))}/{len(np.unique(yrs))}",
                null_med=round(float(np.median(nl)), 2), null95=round(float(np.percentile(nl, 95)), 2),
                p=round(float((nl >= s).mean()), 3),
                SR_hold=round(sharpe(hold_r), 2),
                t_vs_hold=round(float(dd.mean() / dd.std() * np.sqrt(len(dd))), 2) if mode == "LO" else np.nan))
            print(rows[-1], flush=True)
    A = pd.DataFrame(rows)
    mx = np.vstack(list(nulls.values())).max(axis=0)
    best = A.loc[A.SR.idxmax()]
    pd.DataFrame(daily).to_csv(OUT / f"{src}_daily.csv")
    txt = "\n".join([
        f"NOISE-BAND INTRADAY MOMENTUM, gold {src.upper()}, {d.index[0].date()} -> {d.index[-1].date()}, "
        f"bars {step}m, checks every {check}m, cost ${COST_RT} round trip",
        "", A.to_string(index=False), "",
        f"family max-stat (6 cells): best {best.cell} SR {best.SR:+.2f} vs day-shift null max median "
        f"{np.median(mx):+.2f}, 95th {np.percentile(mx, 95):+.2f}, p {(mx >= best.SR).mean():.3f}"])
    print(txt)
    (OUT / f"{src}_report.txt").write_text(txt + "\n")


if __name__ == "__main__":
    sys.exit(main())
