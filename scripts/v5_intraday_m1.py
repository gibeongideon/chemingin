#!/usr/bin/env python3
"""THE INTRADAY FAMILY ON GOLD AT M1 (V5_FINDINGS 3ex). User, 2026-11-16: "i need a intraday
version that is the best ... our aim is to have the best signal for the day".

WHY THIS IS NOT A REPEAT. Gold's intraday momentum is the one intraday effect here that beat its
null in two sources (3ef), and the long-only prior-high break is the one lead (3ch). Both stalled
on the same thing: no M1 path. 3di filed gold's opening breakout as unmeasurable on M15 (+1.17 /
-1.26 by bar-order rule) and 3ci showed 47% of the prior-high result is a FILL assumption. The
repo's own M1 history caps at 73 days. This uses OANDA M1 mid bars 2006-2026 (7.18M bars, one
provenance, alignment checked against three repo feeds) - and 2007-2014 has never been seen by
any study in this repo.

Seven cells, fixed in data/v5_runs/intraday_day_signal/PREREGISTRATION.md (commit 264ebb8):
  N1 noise band NY 08-16 ET, 30-min CLOSE checks, long/short  (3ef verbatim - its own code)
  N2 N1 long-only
  N3 noise band entered by RESTING STOP at the band, exits at the 30-min checks, long/short
  N4 N3 long-only
  P1 prior-day-high buy-stop, long-only, exit at the day's last bar (3ch)
  P2 prior-day high/low, first break wins, exit at the day's last bar (3bj)
  O1 09:30 ET 5-min opening range, both stops, stop 0.10 x ATR14(D1), flat 16:00 (3di)

Fills: a buy-stop at L triggers when the MID high >= L (later and rarer than a real ask
trigger) and fills at max(L, that minute's open); the full round trip ($0.30, stress $0.60) is
then charged. Same-minute double touches are counted and resolved AGAINST the trade.

Statistic: a_d = gross_d - nbar x HOLD_d, nbar = the cell's average signed notional, HOLD = the
raw return of always being long over the cell's window. Family = stationary block bootstrap of
the recentred a_d matrix, signed max studentised mean over the 7 cells (gross).

    python scripts/v5_intraday_m1.py
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
from scripts.v5_noise_band_momentum import LOOKBACK, matrices, pnl, positions  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "intraday_day_signal"
COST, START, NY = 0.30, "2007-01-01", "America/New_York"
N_BOOT, BLOCK = 5000, 20
CELLS = ["N1", "N2", "N3", "N4", "P1", "P2", "O1"]


def sharpe(x) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / x.std() * np.sqrt(252)) if len(x) > 2 and x.std() > 0 else 0.0


def load_m1() -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "data/oanda_hf/XAUUSD_M1.parquet")
    et = pd.DatetimeIndex(pd.to_datetime(d.time, unit="s")).tz_localize("UTC").tz_convert(NY).tz_localize(None)
    d = d[["open", "high", "low", "close"]]
    d.index = et
    return d


def ny_grid(d: pd.DataFrame, dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Minute grids [session, 480] for 08:00-16:00 ET. Missing minutes are NaN (no trigger)."""
    h = d.index.hour
    x = d[(h >= 8) & (h < 16)]
    row = dates.get_indexer(x.index.normalize())
    col = (x.index.hour - 8) * 60 + x.index.minute
    ok = row >= 0
    g = {}
    for k in ("open", "high", "low", "close"):
        a = np.full((len(dates), 480), np.nan)
        a[row[ok], col[ok]] = x[k].to_numpy(dtype=float)[ok]
        g[k] = a
    g["cf"] = pd.DataFrame(g["close"]).ffill(axis=1).to_numpy()        # last known price at each minute
    return g


def vol_weight(r: np.ndarray) -> np.ndarray:
    """1/vol sizing exactly as 3ef: 1% target on the trailing vol of the window return, cap 4."""
    vol = pd.Series(r).ewm(span=20, min_periods=10).std().shift(1).to_numpy()
    return np.minimum(np.nan_to_num(0.01 / vol, nan=0.0, posinf=0.0), 4.0)


# ------------------------------------------------------------------ N cells
def noise_band(d: pd.DataFrame):
    d15 = d.resample("15min").agg(dict(open="first", high="max", low="min", close="last")).dropna()
    O, C = matrices(d15, 15, 8 * 60, 8 * 60)
    dates = pd.DatetimeIndex(C.index)
    c = C.ffill(axis=1).to_numpy()
    open0 = O[0].to_numpy()
    hold = c[:, -1] / open0 - 1
    w = vol_weight(hold)
    base = positions(O, C, 15, 30)                                       # 3ef's own function
    px = np.nanmedian(c, axis=1)
    out = {}
    for name, pos in (("N1", base), ("N2", np.maximum(base, 0.0))):
        gross = pnl(pos, C, 0.0)
        cost1 = gross - pnl(pos, C, 1.0)                                 # cost per $1 of round trip
        ntr = (np.abs(np.diff(np.column_stack([np.zeros(len(pos)), pos]), axis=1)) > 0).sum(axis=1) / 2
        out[name] = pd.DataFrame(dict(gross=w * gross, cost1=w * cost1, expo=w * pos[:, :-1].sum(axis=1) * 15 / 480,
                                      ntr=ntr, hold=hold, w=w, tr_bp=np.where(ntr > 0, 1e4 * gross / np.maximum(ntr, 1), np.nan),
                                      px=px), index=dates)
    # band levels at the 30-min checks, the same arithmetic as positions()
    move = np.abs(c / open0[:, None] - 1)
    sigma = pd.DataFrame(move).rolling(LOOKBACK, min_periods=10).mean().shift(1).to_numpy()
    last_close = C.ffill(axis=1).iloc[:, -1].shift(1).to_numpy()
    ub = np.fmax(open0, last_close)[:, None] * (1 + sigma)
    lb = np.fmin(open0, last_close)[:, None] * (1 - sigma)
    g = ny_grid(d, dates)
    rows, trades = {k: [] for k in ("N3", "N4")}, []
    dbl = 0
    for s in range(len(dates)):
        Hs, Ls, Os, Cf = g["high"][s], g["low"][s], g["open"][s], g["cf"][s]
        for name, shorts in (("N3", True), ("N4", False)):               # each mode is its own state machine
            res = [0.0, 0, 0.0]                                          # gross, round trips, signed minutes
            pos, entry, m_in = 0, np.nan, 0
            for k in range(1, 17) if np.isfinite(sigma[s]).any() else ():  # check k = the close of minute 30k-1
                a, b = 30 * (k - 1), 30 * k
                U, Lw = ub[s, 2 * k - 1], lb[s, 2 * k - 1]
                P = Cf[b - 1]
                band = np.isfinite(U) and np.isfinite(Lw)
                if pos == 0 and band:
                    hu = np.flatnonzero(Hs[a:b] >= U)
                    hl = np.flatnonzero(Ls[a:b] <= Lw) if shorts else ()
                    tu, tl = (hu[0] if len(hu) else 99), (hl[0] if len(hl) else 99)
                    if tu < tl:
                        pos, m_in = 1, a + tu
                        entry = max(U, Os[m_in]) if np.isfinite(Os[m_in]) else U
                    elif tl < tu:
                        pos, m_in = -1, a + tl
                        entry = min(Lw, Os[m_in]) if np.isfinite(Os[m_in]) else Lw
                    elif tu < 99:                                        # both in one minute: take the loser
                        dbl += shorts
                        pos, m_in = (1, a + tu) if (P - U) <= (Lw - P) else (-1, a + tu)
                        entry = U if pos == 1 else Lw
                if pos != 0 and np.isfinite(P) and (k == 16 or (band and ((pos == 1 and not P > U) or (pos == -1 and not P < Lw)))):
                    r = pos * (P / entry - 1)
                    res[0] += r; res[1] += 1; res[2] += pos * (b - m_in)
                    trades.append((name, dates[s], m_in, b, pos, entry, P, 1e4 * r))
                    pos = 0
            rows[name].append(res)
    for k in rows:
        a = np.array(rows[k])
        out[k] = pd.DataFrame(dict(gross=w * a[:, 0], cost1=w * a[:, 1] / px, expo=w * a[:, 2] / 480, ntr=a[:, 1],
                                   hold=hold, w=w, tr_bp=np.where(a[:, 1] > 0, 1e4 * a[:, 0] / np.maximum(a[:, 1], 1), np.nan),
                                   px=px), index=dates)
    tr = pd.DataFrame(trades, columns=["cell", "date", "m_in", "m_out", "side", "entry", "exit", "bp"])
    return out, g, dates, dbl, tr


# ------------------------------------------------------------------ P cells
def prior_day(d: pd.DataFrame):
    key = (d.index + pd.Timedelta(hours=7)).normalize()
    codes, start = np.unique(key.values, return_index=True)
    end = np.append(start[1:], len(d))
    O, H, L, C = (d[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    days = [(pd.Timestamp(codes[i]), start[i], end[i]) for i in range(len(codes)) if end[i] - start[i] >= 600]
    bar = pd.DataFrame([(dt, O[a], H[a:b].max(), L[a:b].min(), C[b - 1]) for dt, a, b in days],
                       columns=["day", "open", "high", "low", "close"]).set_index("day")
    hold = (bar.close / bar.open - 1).to_numpy()
    w = vol_weight(hold)
    r1, r2, dbl = [], [], 0
    for i, (dt, a, b) in enumerate(days):
        x1, x2 = (0.0, 0, 0.0), (0.0, 0, 0.0)
        if i > 0:
            pdh, pdl = bar.high.iloc[i - 1], bar.low.iloc[i - 1]
            hu, hl = np.flatnonzero(H[a:b] >= pdh), np.flatnonzero(L[a:b] <= pdl)
            tu, tl = (hu[0] if len(hu) else 10**9), (hl[0] if len(hl) else 10**9)
            n, last = b - a, C[b - 1]
            if tu < 10**9:
                e = max(pdh, O[a + tu])
                x1 = (last / e - 1, 1, (n - tu) / n)
            if tu < tl:
                e = max(pdh, O[a + tu]); x2 = (last / e - 1, 1, (n - tu) / n)
            elif tl < tu:
                e = min(pdl, O[a + tl]); x2 = (-(last / e - 1), 1, -(n - tl) / n)
            elif tu < 10**9:                                             # both in one minute: take the loser
                dbl += 1
                lg, sh = last / pdh - 1, -(last / pdl - 1)
                x2 = (lg, 1, (n - tu) / n) if lg <= sh else (sh, 1, -(n - tu) / n)
        r1.append(x1); r2.append(x2)
    out = {}
    for name, r in (("P1", np.array(r1)), ("P2", np.array(r2))):
        out[name] = pd.DataFrame(dict(gross=w * r[:, 0], cost1=w * r[:, 1] / bar.close.to_numpy(), expo=w * r[:, 2],
                                      ntr=r[:, 1], hold=hold, w=w, tr_bp=np.where(r[:, 1] > 0, 1e4 * r[:, 0], np.nan),
                                      px=bar.close.to_numpy()), index=bar.index)
    tr_ = np.maximum.reduce([bar.high - bar.low, (bar.high - bar.close.shift()).abs(), (bar.low - bar.close.shift()).abs()])
    atr = pd.Series(tr_, index=bar.index).rolling(14).mean().shift(1)    # known before the day opens
    return out, bar, atr, dbl


# ------------------------------------------------------------------ O1
def opening_range(g: dict, dates: pd.DatetimeIndex, atr: pd.Series, hold: np.ndarray):
    A = atr.reindex(dates).to_numpy()
    rows, same = [], 0
    for s in range(len(dates)):
        Hs, Ls, Os, Cf = g["high"][s], g["low"][s], g["open"][s], g["cf"][s]
        x = (0.0, 0, 0.0, np.nan)
        orh, orl = np.nanmax(Hs[90:95]) if np.isfinite(Hs[90:95]).sum() >= 4 else np.nan, np.nanmin(Ls[90:95]) if np.isfinite(Ls[90:95]).sum() >= 4 else np.nan
        if np.isfinite(orh) and np.isfinite(A[s]) and A[s] > 0 and np.isfinite(Cf[479]):
            risk = 0.10 * A[s]
            hu, hl = np.flatnonzero(Hs[95:] >= orh), np.flatnonzero(Ls[95:] <= orl)
            tu, tl = (hu[0] if len(hu) else 999), (hl[0] if len(hl) else 999)
            side = 0
            if tu < tl:
                side, m = 1, 95 + tu
            elif tl < tu:
                side, m = -1, 95 + tl
            elif tu < 999:                                               # both in one minute: stopped, full loss
                same += 1
                x = (-1.0, 1, 0.0, risk)
            if side:
                e = (max(orh, Os[m]) if side == 1 else min(orl, Os[m])) if np.isfinite(Os[m]) else (orh if side == 1 else orl)
                stop = e - side * risk
                if (side == 1 and Ls[m] <= stop) or (side == -1 and Hs[m] >= stop):   # entry minute also covers the stop
                    same += 1
                    x = (-1.0, 1, 0.0, risk)
                else:
                    hit = np.flatnonzero(Ls[m + 1:] <= stop) if side == 1 else np.flatnonzero(Hs[m + 1:] >= stop)
                    if len(hit):
                        j = m + 1 + hit[0]
                        px_out = (min(stop, Os[j]) if side == 1 else max(stop, Os[j])) if np.isfinite(Os[j]) else stop
                    else:
                        j, px_out = 479, Cf[479]
                    x = (side * (px_out - e) / risk, 1, side * (j - m) / 480 * (0.0025 / (risk / e)), risk)
        rows.append(x)
    r = np.array(rows)
    return pd.DataFrame(dict(gross=0.0025 * r[:, 0], cost1=np.where(r[:, 1] > 0, 0.0025 / r[:, 3], 0.0), expo=r[:, 2],
                             ntr=r[:, 1], hold=hold, w=1.0, tr_bp=np.where(r[:, 1] > 0, r[:, 0], np.nan),
                             px=np.nan), index=dates), same


# ------------------------------------------------------------------ statistics
def reality_check(A: np.ndarray, n: int = N_BOOT, blk: int = BLOCK, seed: int = 11):
    """Stationary bootstrap (mean block `blk`) of the recentred columns; statistic = max over the
    cells of the studentised mean. Signed, one-sided."""
    T = len(A)
    sd = A.std(axis=0)
    t_obs = np.sqrt(T) * A.mean(axis=0) / sd
    Ac = A - A.mean(axis=0)
    rng = np.random.default_rng(seed)
    mx = np.empty(n)
    for b in range(n):
        idx = np.empty(T, dtype=np.int64)
        i = 0
        while i < T:
            s0, ln = rng.integers(T), rng.geometric(1 / blk)
            ln = min(ln, T - i)
            idx[i:i + ln] = (s0 + np.arange(ln)) % T
            i += ln
        x = Ac[idx]
        mx[b] = (np.sqrt(T) * x.mean(axis=0) / sd).max()
    return t_obs, mx


def row(name: str, df: pd.DataFrame, nbar: float, dbl) -> dict:
    net, net2 = df.gross - COST * df.cost1, df.gross - 2 * COST * df.cost1
    a = df.gross - nbar * df.hold
    half = len(df) // 2
    yrs = df.index.year
    early, late = df.index < "2015-01-01", df.index >= "2015-01-01"
    tr = df.tr_bp.dropna()
    be = float(df.gross.sum() / df.cost1.sum()) if df.cost1.sum() > 0 else np.nan
    return dict(cell=name, days=len(df), trades_wk=round(df.ntr.sum() / (len(df) / 5), 2),
                hit=round(100 * (tr > 0).mean(), 1), per_trade=round(float(tr.mean()), 2),
                nbar=round(nbar, 3), SR_gross=round(sharpe(df.gross), 2), SR=round(sharpe(net), 2),
                SR_2x=round(sharpe(net2), 2), h1=round(sharpe(net[:half]), 2), h2=round(sharpe(net[half:]), 2),
                y07_14=round(sharpe(net[early]), 2), y15_26=round(sharpe(net[late]), 2),
                yrs_pos=f"{sum(sharpe(net[yrs == y]) > 0 for y in np.unique(yrs))}/{len(np.unique(yrs))}",
                t_vs_hold=round(float(a.mean() / a.std() * np.sqrt(len(a))), 2),
                breakeven=round(be, 2), dbl=dbl)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_m1()
    ncells, g, dates, dbl_n, ntr = noise_band(d)
    pcells, bar, atr, dbl_p = prior_day(d)
    o1, same = opening_range(g, dates, atr, ncells["N1"].hold.to_numpy())
    cells = {**ncells, **pcells, "O1": o1}
    dbls = dict(N1=0, N2=0, N3=dbl_n, N4=dbl_n, P1=0, P2=dbl_p, O1=same)

    # harness check: the 3ef known answer on the 3ef window
    k = (dates >= "2015-01-01") & (dates <= "2026-06-15")
    chk = {n: (sharpe(ncells[n].gross[k]), sharpe((ncells[n].gross - COST * ncells[n].cost1)[k])) for n in ("N1", "N2")}

    common = None
    for n in CELLS:
        cells[n] = cells[n][cells[n].index >= START]
        common = cells[n].index if common is None else common.intersection(cells[n].index)
    cells = {n: cells[n].loc[common] for n in CELLS}
    nbar = {n: float(cells[n].expo.mean()) for n in CELLS}
    rows = [row(n, cells[n], nbar[n], dbls[n]) for n in CELLS]
    A = np.column_stack([(cells[n].gross - nbar[n] * cells[n].hold).to_numpy() for n in CELLS])
    t_obs, mx = reality_check(A)
    T = pd.DataFrame(rows)
    T["t_a"] = np.round(t_obs, 2)
    T["p_cell"] = [round(float((mx >= t).mean()), 4) for t in t_obs]
    hold_ny = cells["N1"].w * (cells["N1"].hold - COST / cells["N1"].px)
    hold_day = cells["P1"].w * (cells["P1"].hold - COST / cells["P1"].px)
    raw = (1 + cells["P1"].hold).prod() ** (252 / len(common)) - 1
    best = int(np.argmax(t_obs))
    pd.DataFrame({n: cells[n].gross - COST * cells[n].cost1 for n in CELLS}).to_csv(OUT / "m1_family_daily.csv")
    pd.concat({n: cells[n] for n in CELLS}, axis=1).to_parquet(OUT / "m1_family_cells.parquet")
    ntr.to_parquet(OUT / "m1_n3_trades.parquet")
    txt = "\n".join([
        f"GOLD INTRADAY FAMILY AT M1 (OANDA mid), {common[0].date()} -> {common[-1].date()}, {len(common):,} sessions, "
        f"cost ${COST} round trip, stress ${2 * COST}",
        "",
        f"harness check on 3ef's window 2015-01..2026-06 (3ef filed NY_LS gross +0.81 / net +0.05, NY_LO +0.60 / +0.01): "
        f"N1 {chk['N1'][0]:+.2f} / {chk['N1'][1]:+.2f}, N2 {chk['N2'][0]:+.2f} / {chk['N2'][1]:+.2f}",
        "",
        T.to_string(index=False),
        "",
        "per_trade = mean gross bp per round trip (O1: mean R). breakeven = $ round trip at which net = 0. "
        "t_vs_hold / t_a = t of gross - nbar x HOLD.",
        f"dbl = same-minute double touches resolved against the trade.  cost in bp today: {1e4 * COST / 4100:.2f}bp at $4,100, "
        f"{1e4 * COST / 1200:.2f}bp at $1,200",
        f"benchmarks: HOLD NY session 1/vol net SR {sharpe(hold_ny):+.2f}; HOLD day 1/vol net SR {sharpe(hold_day):+.2f}; "
        f"gold itself {100 * raw:+.1f}%/yr over the window",
        "",
        f"FAMILY (7 cells, signed max studentised mean of a_d, stationary bootstrap block {BLOCK}, {N_BOOT} draws): "
        f"best {CELLS[best]} t {t_obs[best]:+.2f} vs null max median {np.median(mx):+.2f}, 95th {np.percentile(mx, 95):+.2f}, "
        f"p {(mx >= t_obs[best]).mean():.4f}"])
    print(txt)
    (OUT / "m1_family_report.txt").write_text(txt + "\n")


if __name__ == "__main__":
    sys.exit(main())
