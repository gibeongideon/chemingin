#!/usr/bin/env python3
"""5-minute ORB on "Stocks in Play" (Zarattini, Barbon & Aziz 2024, SSRN 4729284) on Switch CFDs.

THE PAPER (US stocks 2016-23, IQFeed, $0.0035/share): trade in the direction of the first
5-minute candle (bullish -> buy-stop at its high, bearish -> sell-stop at its low, doji -> no
trade); stop loss 10% of the 14-day ATR from the fill; exit at the close; 1% risk per trade,
4x leverage cap. Filters: open > $5, 14-day average volume >= 1M shares, ATR14 > $0.50, and the
"Stocks in Play" refinement: opening-range RELATIVE VOLUME (first-5-min volume / mean of the
previous 14 days' first-5-min volume) >= 1, top 20 by it. Claimed: Sharpe 2.81, +1,600%.
The base version (no relative-volume filter) made Sharpe 0.48.

WHAT IS DIFFERENT HERE, AND WHY IT IS STILL THE TEST THAT MATTERS
  * venue: Switch US-share CFDs (retail-accessible), M5 BID bars, ~2024-05 .. now. Server time
    is New York + 7h all year, so the 16:30 bar is the 9:30-9:35 ET opening range.
  * cost: the bar's own quoted spread (~$0.31 flat markup, measured 2026-11-05). The feed is
    BID, so the long stop sits at the opening range's ASK high (bid high + spread) and fills on
    the ask; the short stop sits at the bid low and covers on the ask. Gross = spread set to 0.
  * volume: CFD TICK volume (quote updates), not shares. The 1M-share filter is replaced by
    "14-day median opening-range tick volume in the top half of the universe that day".
  * universe: whatever Switch returned history for (the fetch got 313 of 1,302 listed names).

VARIANTS (pre-registered, all printed):
  BASE         all eligible names
  SIP20        the paper's Stocks-in-Play top 20 (RVol >= 1)
  SIP20_CHEAP  SIP20, only names whose stop is >= 6 x the spread (cost <= ~1/3 R round trip)
  each NET and GROSS (spread = 0), to separate "no edge" from "edge eaten by cost".
CONTROL: random DIRECTION — same days, names, levels and stops, long/short by coin flip
(200 draws). Each event is simulated once per side, so a draw is a lookup.

    python scripts/v5_orb_stocks_in_play.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "us_m5"
OUT = ROOT / "data" / "v5_runs" / "orb_sip"
RISK, LEV, TOPN, STOP_ATR = 0.01, 4.0, 20, 0.10
N_NULL = int(os.environ.get("ORB_NULL", 200))
PESS = os.environ.get("ORB_PESS", "0") == "1"   # 1 = same-bar stop always after the fill
SEED = 5
OPEN_M, LAST_M = 16 * 60 + 30, 22 * 60 + 55          # server minutes; server = ET + 7h


def day_events(f: Path, point: float):
    """Per RTH day: opening-range stats + [start, end) slice of the post-OR bars."""
    d = pd.read_csv(f)
    t = pd.to_datetime(d.t)
    mins = (t.dt.hour * 60 + t.dt.minute).to_numpy()
    keep = (mins >= OPEN_M) & (mins <= LAST_M)
    d, t, mins = d[keep].reset_index(drop=True), t[keep].reset_index(drop=True), mins[keep]
    if len(d) < 78 * 40:
        return None
    day = t.dt.normalize().to_numpy()
    o, h, l, c, tv = (d[k].to_numpy(float) for k in ("o", "h", "l", "c", "tv"))
    spr = d.sp.to_numpy(float) * point
    brk = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    end = np.r_[brk[1:], len(d)]
    n = end - brk
    dh = np.maximum.reduceat(h, brk)
    dl = np.minimum.reduceat(l, brk)
    dc = c[end - 1]
    pc = np.r_[np.nan, dc[:-1]]
    tr = np.nanmax(np.c_[dh - dl, np.abs(dh - pc), np.abs(dl - pc)], axis=1)
    atr = pd.Series(tr).rolling(14).mean().shift(1).to_numpy()
    first_ok = mins[brk] == OPEN_M
    or_tv = np.where(first_ok, tv[brk], np.nan)
    s_tv = pd.Series(or_tv)
    rvol = or_tv / s_tv.rolling(14, min_periods=10).mean().shift(1).to_numpy()
    tvm = s_tv.rolling(14, min_periods=10).median().shift(1).to_numpy()
    side = np.sign(c[brk] - o[brk])
    ok = first_ok & (n >= 70) & np.isfinite(atr) & np.isfinite(rvol) & (side != 0)
    i = np.flatnonzero(ok)
    b = brk[i]
    ev = pd.DataFrame(dict(day=day[b], open=o[b], atr=atr[i], rvol=rvol[i], tvm=tvm[i],
                           side=side[i].astype(int), orh=h[b], orl=l[b],
                           s0=b + 1, s1=end[i], spr12=[np.median(spr[x + 1:x + 13]) for x in b]))
    return ev, (o, h, l, c, spr)


@njit(cache=True)
def sim(side, lvl_h, lvl_l, R, s0, s1, o, h, l, c, s, pess):
    """R multiple per event for the given side; nan = never triggered."""
    out = np.full(len(side), np.nan)
    for k in range(len(side)):
        sd, r = side[k], R[k]
        fill = sl = 0.0
        i = -1
        for j in range(s0[k], s1[k]):
            if sd == 1 and h[j] >= lvl_h[k]:              # ask >= ask-high of the OR
                fill = max(lvl_h[k], o[j]) + s[j]
                sl = fill - r
                i = j
                break
            if sd == -1 and l[j] <= lvl_l[k]:             # bid <= bid-low of the OR
                fill = min(lvl_l[k], o[j])
                sl = fill + r
                i = j
                break
        if i < 0:
            continue
        # ENTRY BAR: the 10%-ATR stop is often inside one M5 bar, so the order of high and low
        # matters. pess=True: the stop always comes after the fill (a lower bound). pess=False:
        # the OHLC path convention — an up bar runs O->L->H->C, a down bar O->H->L->C; a gap
        # fill at the open sees the whole bar.
        if sd == 1:
            gap = o[i] >= lvl_h[k]
            after_low = pess or gap or c[i] < o[i]          # does the low come after the fill?
            if (after_low and l[i] <= sl) or c[i] <= sl:
                out[k] = -1.0
                continue
        else:
            gap = o[i] <= lvl_l[k]
            after_high = pess or gap or c[i] >= o[i]
            if (after_high and h[i] + s[i] >= sl) or c[i] + s[i] >= sl:
                out[k] = -1.0
                continue
        res = np.nan
        for j in range(i + 1, s1[k]):
            if sd == 1 and l[j] <= sl:
                res = (min(sl, o[j]) - fill) / r
                break
            if sd == -1 and h[j] + s[j] >= sl:
                res = (fill - max(sl, o[j] + s[j])) / r
                break
        if np.isnan(res):
            e = s1[k] - 1
            res = (c[e] - fill) / r if sd == 1 else (fill - c[e] - s[e]) / r
        out[k] = res
    return out


def load():
    meta = pd.read_csv(DATA / "_meta.csv", index_col=0) if (DATA / "_meta.csv").exists() else None
    evs, arrs, off = [], [[], [], [], [], []], 0
    for f in sorted(DATA.glob("*.csv.gz")):
        sym = f.name[:-7]
        pt = float(meta.loc[sym, "point"]) if meta is not None and sym in meta.index else 0.001
        try:
            r = day_events(f, pt)
        except Exception as e:  # noqa: BLE001
            print("skip", sym, e)
            continue
        if r is None or not len(r[0]):
            continue
        e, a = r
        e["sym"] = sym
        e["s0"] += off
        e["s1"] += off
        off += len(a[0])
        evs.append(e)
        for k in range(5):
            arrs[k].append(a[k])
    return pd.concat(evs, ignore_index=True), [np.concatenate(x) for x in arrs]


def stats(r: np.ndarray) -> tuple[float, float, float]:
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
    eq = np.cumprod(1 + r)
    return 100 * (eq[-1] ** (252 / len(r)) - 1), sh, 100 * (1 - eq / np.maximum.accumulate(eq)).max()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ev, (o, h, l, c, s) = load()
    ev = ev[(ev.open > 5) & (ev.atr > 0.5)]
    ev = ev[ev.tvm >= ev.groupby("day").tvm.transform("median")].reset_index(drop=True)
    ev["R"] = STOP_ATR * ev.atr
    ev["spread_R"] = ev.spr12 / ev.R
    days = np.unique(ev.day.to_numpy())
    z = np.zeros_like(s)
    args = lambda sd: (sd, ev.orh.to_numpy(), ev.orl.to_numpy(), ev.R.to_numpy(),  # noqa: E731
                       ev.s0.to_numpy(), ev.s1.to_numpy(), o, h, l, c)
    n_all = len(ev)
    res = {(sd, cost): sim(np.full(n_all, sd), *args(None)[1:], s if cost else z, PESS)
           for sd in (1, -1) for cost in (True, False)}
    sip = ev.rvol >= 1
    rk = ev[sip].groupby("day").rvol.rank(ascending=False, method="first")
    in_sip = np.zeros(n_all, bool)
    in_sip[rk.index[rk <= TOPN]] = True
    cheap = in_sip & (ev.spread_R.to_numpy() <= 1 / 6)
    dpos = np.searchsorted(days, ev.day.to_numpy())
    lines = [f"entry bar: {'PESSIMISTIC' if PESS else 'OHLC path'}; {ev.sym.nunique()} symbols, {len(days)} days {pd.Timestamp(days[0]):%Y-%m-%d}.."
             f"{pd.Timestamp(days[-1]):%Y-%m-%d}; spread per side in R: median "
             f"{ev.spread_R.median():.2f} all, {ev.spread_R[in_sip].median():.2f} SIP20"]
    rng = np.random.default_rng(SEED)
    rows = []
    for name, m in (("BASE", np.ones(n_all, bool)), ("SIP20", in_sip), ("SIP20_CHEAP", cheap)):
        idx = np.flatnonzero(m)
        nday = np.bincount(dpos[idx], minlength=len(days))
        w = np.minimum(RISK, (LEV / nday[dpos[idx]]) * ev.R.to_numpy()[idx] / ev.open.to_numpy()[idx])
        for cost in (True, False):
            L, S = res[(1, cost)][idx], res[(-1, cost)][idx]

            def daily(sd):
                rm = np.where(sd == 1, L, S)
                ok = np.isfinite(rm)
                return np.bincount(dpos[idx][ok], weights=(rm * w)[ok], minlength=len(days))
            real = ev.side.to_numpy()[idx]
            rr = np.where(real == 1, L, S)
            ann, sh, mdd = stats(daily(real))
            null = np.array([stats(daily(rng.choice([-1, 1], len(idx))))[1] for _ in range(N_NULL)])
            trig = np.isfinite(rr)
            rows.append(dict(variant=name, cost="NET" if cost else "GROSS", cand=len(idx),
                             trades=int(trig.sum()), win=round(100 * (rr[trig] > 0).mean(), 1),
                             avgR=round(float(np.nanmean(rr)), 3), ann=round(ann, 1),
                             sharpe=round(sh, 2), mdd=round(mdd, 1),
                             rand_p50=round(float(np.median(null)), 2),
                             rand_p95=round(float(np.percentile(null, 95)), 2),
                             p=round(float((null >= sh).mean()), 3)))
            print(rows[-1], flush=True)
    # halves for the paper's variant, gross and net
    half = len(days) // 2
    for cost in (True, False):
        idx = np.flatnonzero(in_sip)
        nday = np.bincount(dpos[idx], minlength=len(days))
        w = np.minimum(RISK, (LEV / nday[dpos[idx]]) * ev.R.to_numpy()[idx] / ev.open.to_numpy()[idx])
        rm = np.where(ev.side.to_numpy()[idx] == 1, res[(1, cost)][idx], res[(-1, cost)][idx])
        ok = np.isfinite(rm)
        r = np.bincount(dpos[idx][ok], weights=(rm * w)[ok], minlength=len(days))
        lines.append(f"SIP20 {'NET' if cost else 'GROSS'} halves: Sharpe "
                     f"{stats(r[:half])[1]:+.2f} / {stats(r[half:])[1]:+.2f}")
    txt = "\n".join(lines) + "\n" + pd.DataFrame(rows).to_string(index=False) + (
        "\n\nann/sharpe/mdd at 1% risk per trade (4x cap); avgR per triggered trade; p = share of "
        f"{N_NULL} random-DIRECTION replays (same names, days, levels, stops) with Sharpe >= it.\n")
    print(txt)
    (OUT / ("report_pess.txt" if PESS else "report.txt")).write_text(txt)


if __name__ == "__main__":
    sys.exit(main())
