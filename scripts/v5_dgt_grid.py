#!/usr/bin/env python3
"""Replicate and stress-test "Dynamic Grid Trading" (Chen, Chen & Jang, arXiv 2506.11921, 2025).

THE PAPER. A geometric spot grid (step k, `half` levels each side) buys 1/G of the cash on each
level crossed down and sells 1/G of the coins on each level crossed up (G = levels left on that
side). The classic grid STOPS when price leaves the range; DGT RESETS instead:
  * upside break  -> all coins are sold by construction; the initial capital is reinvested in a
                    new grid centred at the break, the grid profit goes to a `wallet`
  * downside break-> all cash is spent; the strategy KEEPS the coins and the wallet's profit
                    becomes the principal of the new grid
Claimed (BTC/ETH, Binance 1m, Jan 2021 - Jul 2024, fee 0.0008): IRR above buy-and-hold with a
much lower max drawdown (~50% vs ~80%).

TWO READINGS, because §3.1 is ambiguous about what happens to the coins after a downside break:
  LITERAL  coins move to a held `bag` (never sold); only the wallet funds the next grid
  RECENTRE the whole account re-centres (half coins / half cash) on EVERY break

THE CONTROLS THIS REPO REQUIRES (each has killed a published-looking result here before):
  1. EXPOSURE. A grid holds on average ~half its money in coins, so ~half the drawdown of
     holding 100% is arithmetic, not skill. Each cell is compared with a CONSTANT-MIX portfolio
     holding the cell's own average coin fraction, rebalanced daily, paying the same fee.
     Sharpe is printed because it is exposure-invariant.
  2. PERIOD. The paper's window (2021-01..2024-07) is one bull cycle. Also run: the full Binance
     history (2017-08..2026-10), the 2018 and 2022 bears, and everything AFTER the paper.
  3. GRID SEARCH. The paper reports the best of 72-90 cells; we report the median cell too.
  4. YOUR VENUES. `--cfd` runs the same procedure on the Switch CFD feeds (XAU, BTC) with the
     spread as the fee and the long SWAP charged daily on the coin notional.

    python scripts/v5_dgt_grid.py --build      # zips -> npz cache
    python scripts/v5_dgt_grid.py              # Binance replication + controls
    python scripts/v5_dgt_grid.py --cfd        # Switch XAU/BTC
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "binance_1m"
OUT = ROOT / "data" / "v5_runs" / "dgt_grid"
KS = (0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1)
HALVES = (2, 3, 4, 5, 6, 7, 8, 9, 10)
FEE = 0.0008
PERIODS = {
    "PAPER 2021-01..2024-07": ("2021-01-01", "2024-08-01"),
    "FULL 2017-08..2026-10": ("2017-01-01", "2026-12-01"),
    "BEAR 2018": ("2018-01-01", "2019-01-01"),
    "BEAR 2022": ("2022-01-01", "2023-01-01"),
    "AFTER 2024-08..2026-10": ("2024-08-01", "2026-12-01"),
}


# ------------------------------------------------------------------------------------ data ---
def build() -> None:
    for sym in ("BTCUSDT", "ETHUSDT"):
        parts = []
        for z in sorted((RAW / "zips").glob(f"{sym}-1m-*.zip")):
            with zipfile.ZipFile(z) as zf:
                raw = zf.read(zf.namelist()[0])
            d = pd.read_csv(io.BytesIO(raw), header=None, usecols=[0, 1, 2, 3, 4])
            if not str(d.iloc[0, 0]).isdigit():                      # some months have headers
                d = d.iloc[1:].astype(float)
            parts.append(d.to_numpy(float))
        a = np.vstack(parts)
        ts = a[:, 0]
        ts = np.where(ts > 1e14, ts / 1000, ts)                      # 2025+ files are in us
        o = np.argsort(ts, kind="stable")
        ts, a = ts[o], a[o]
        keep = np.r_[True, np.diff(ts) > 0]
        np.savez_compressed(RAW / f"{sym}.npz", t=(ts[keep] / 1000).astype(np.int64),
                            o=a[keep, 1], h=a[keep, 2], l=a[keep, 3], c=a[keep, 4])
        print(sym, keep.sum(), pd.to_datetime(ts[0], unit="ms"), "->",
              pd.to_datetime(ts[-1], unit="ms"))


def load(sym: str) -> pd.DataFrame:
    z = np.load(RAW / f"{sym}.npz")
    return pd.DataFrame({k: z[k] for k in "ohlc"}, index=pd.to_datetime(z["t"], unit="s"))


# ---------------------------------------------------------------------------------- engine ---
@njit(cache=True)
def _run(o, h, l, c, day_end, k, half, fee, recentre, swap_day):
    """Walk every bar open -> near extreme -> far extreme -> close; fill limits at level prices.
    Returns per-day equity and per-day coin fraction."""
    nd = day_end.size
    eq = np.empty(nd)
    ex = np.empty(nd)
    p0 = o[0]
    M = 1.0
    coins = (M * 0.5) * (1 - fee) / p0
    cash = M * 0.5
    wallet = 0.0
    bag = 0.0
    ci = 0
    dead = False
    di = 0
    pts = np.empty(4)
    for i in range(o.size):
        if c[i] >= o[i]:
            pts[0], pts[1], pts[2], pts[3] = o[i], l[i], h[i], c[i]
        else:
            pts[0], pts[1], pts[2], pts[3] = o[i], h[i], l[i], c[i]
        for s in range(3):
            a, b = pts[s], pts[s + 1]
            if dead:
                break
            if b > a:
                while True:
                    nxt = p0 * (1 + k) ** (ci + 1)
                    if nxt > b:
                        break
                    if ci < half:
                        g = half - ci
                        q = coins / g
                        cash += q * nxt * (1 - fee)
                        coins -= q
                        ci += 1
                    else:                                   # upside break -> reset at nxt
                        cash += coins * nxt * (1 - fee)
                        coins = 0.0
                        if recentre:
                            cap = cash + wallet
                            wallet = 0.0
                        else:
                            wallet += cash - M
                            cap = M
                            if wallet < 0:                  # cannot reinvest more than exists
                                cap = M + wallet
                                wallet = 0.0
                        M = cap
                        p0 = nxt
                        ci = 0
                        coins = cap * 0.5 * (1 - fee) / p0
                        cash = cap * 0.5
            elif b < a:
                while True:
                    nxt = p0 * (1 + k) ** (ci - 1)
                    if nxt < b:
                        break
                    if ci > -half:
                        g = ci + half
                        spend = cash / g
                        coins += spend * (1 - fee) / nxt
                        cash -= spend
                        ci -= 1
                    else:                                   # downside break -> reset at nxt
                        if recentre:
                            val = coins * nxt + cash + wallet
                            # sell half the coin value to re-centre (fee on the traded half)
                            cash = val * 0.5 * (1 - fee)
                            coins = val * 0.5 / nxt
                            wallet = 0.0
                            M = val
                        else:
                            bag += coins
                            coins = 0.0
                            cap = cash + wallet
                            wallet = 0.0
                            M = cap
                            if cap <= 1e-9:
                                dead = True
                                cash = 0.0
                                break
                            coins = cap * 0.5 * (1 - fee) / nxt
                            cash = cap * 0.5
                        p0 = nxt
                        ci = 0
        if i == day_end[di]:
            px = c[i]
            if swap_day != 0.0:                              # CFD financing on the long notional
                cash += (coins + bag) * px * swap_day
            e = cash + wallet + (coins + bag) * px
            eq[di] = e
            ex[di] = (coins + bag) * px / e if e > 0 else 0.0
            di += 1
            if di == nd:
                break
    return eq, ex


@njit(cache=True)
def _constant_mix(close_d, w, fee, swap_day):
    """Hold fraction w in the asset, rebalanced at each daily close, same fee and financing."""
    n = close_d.size
    eq = np.empty(n)
    e = 1.0
    units = w * e * (1 - fee) / close_d[0]
    cash = e * (1 - w)
    for i in range(n):
        px = close_d[i]
        cash += units * px * swap_day
        e = cash + units * px
        tgt = w * e / px
        trade = abs(tgt - units) * px
        cash -= (tgt - units) * px + trade * fee
        units = tgt
        eq[i] = cash + units * px
    return eq


def stats(eq: np.ndarray, days_per_year: float = 365.0) -> dict:
    r = np.diff(eq) / eq[:-1]
    yrs = len(eq) / days_per_year
    cagr = (eq[-1] / eq[0]) ** (1 / yrs) - 1 if eq[-1] > 0 else -1.0
    dd = 1 - eq / np.maximum.accumulate(eq)
    sh = r.mean() / r.std() * np.sqrt(days_per_year) if r.std() > 0 else 0.0
    return dict(cagr=100 * cagr, mdd=100 * dd.max(), sharpe=sh)


def evaluate(df: pd.DataFrame, fee: float, swap_yr: float, dpy: float, label: str) -> pd.DataFrame:
    day = df.index.floor("D")
    last = np.flatnonzero(np.r_[day[1:] != day[:-1], True])
    close_d = df.c.to_numpy()[last]
    sd = swap_yr / 365.0
    rows = []
    bh = stats(_constant_mix(close_d, 1.0, fee, sd), dpy)
    for rec in (False, True):
        for k in KS:
            for half in HALVES:
                eq, ex = _run(*(df[x].to_numpy() for x in "ohlc"), last, k, half, fee, rec, sd)
                s = stats(eq, dpy)
                w = float(np.clip(ex.mean(), 0.01, 1.0))
                cm = stats(_constant_mix(close_d, w, fee, sd), dpy)
                rows.append(dict(period=label, mode="RECENTRE" if rec else "LITERAL", k=k,
                                 half=half, cagr=s["cagr"], mdd=s["mdd"], sharpe=s["sharpe"],
                                 avg_exposure=w, mix_cagr=cm["cagr"], mix_mdd=cm["mdd"],
                                 mix_sharpe=cm["sharpe"], bh_cagr=bh["cagr"], bh_mdd=bh["mdd"],
                                 bh_sharpe=bh["sharpe"]))
    return pd.DataFrame(rows)


def digest(res: pd.DataFrame) -> str:
    out = []
    for (per, mode), g in res.groupby(["period", "mode"], sort=False):
        best = g.loc[g.cagr.idxmax()]
        med = g.median(numeric_only=True)
        beat_bh = (g.cagr > g.bh_cagr).mean()
        beat_mix = (g.sharpe > g.mix_sharpe).mean()
        out.append(f"{per:<26} {mode:<8}  B&H {best.bh_cagr:+6.1f}%/yr mdd {best.bh_mdd:4.0f}% "
                   f"SR {best.bh_sharpe:.2f} | best cell k={best.k} h={int(best.half)} "
                   f"{best.cagr:+6.1f}% mdd {best.mdd:3.0f}% SR {best.sharpe:.2f} "
                   f"(mix@{best.avg_exposure:.2f}: {best.mix_cagr:+6.1f}% mdd {best.mix_mdd:3.0f}% "
                   f"SR {best.mix_sharpe:.2f}) | MEDIAN cell {med.cagr:+6.1f}% mdd {med.mdd:3.0f}% "
                   f"SR {med.sharpe:.2f} vs mix SR {med.mix_sharpe:.2f} | cells beating B&H CAGR "
                   f"{100*beat_bh:.0f}%, beating matched-mix Sharpe {100*beat_mix:.0f}%")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--cfd", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.build:
        build()
        return
    if a.cfd:
        # Switch CFD feeds. fee per fill = half the quoted spread in price terms; swap on longs.
        spec = {"XAUUSDs": (0.25 / 2, -0.06, 260), "BTCUSDs": (None, -0.15, 365)}
        frames = []
        for sym, (half_spread, swap, dpy) in spec.items():
            d = pd.read_csv(ROOT / "data" / "sweep_switch" / f"{sym}_M15.csv",
                            parse_dates=["time"]).set_index("time").sort_index()
            d = d[~d.index.duplicated()].rename(columns={"open": "o", "high": "h", "low": "l",
                                                         "close": "c"})
            px = float(d.c.median())
            fee = (half_spread / px) if half_spread else 4.5e-4 / 2   # BTC floor 4.5bp spread
            lab = f"{sym} M15 {d.index[0]:%Y-%m}..{d.index[-1]:%Y-%m}"
            frames.append(evaluate(d[["o", "h", "l", "c"]], fee, swap, dpy, lab))
        res = pd.concat(frames)
        res.to_csv(OUT / "cfd_cells.csv", index=False)
        txt = digest(res)
        print(txt)
        (OUT / "cfd_report.txt").write_text(txt + "\n")
        return
    frames = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        df = load(sym)
        for per, (s, e) in PERIODS.items():
            sub = df[(df.index >= s) & (df.index < e)]
            if len(sub) < 1440 * 60:
                continue
            frames.append(evaluate(sub, FEE, 0.0, 365, f"{sym[:3]} {per}"))
            print(sym, per, "done", flush=True)
    res = pd.concat(frames)
    res.to_csv(OUT / "cells.csv", index=False)
    txt = digest(res)
    print(txt)
    (OUT / "report.txt").write_text(txt + "\n")


if __name__ == "__main__":
    sys.exit(main())
