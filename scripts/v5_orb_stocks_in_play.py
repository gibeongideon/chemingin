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
  * venue: Switch US-share CFDs (retail-accessible), M5 BID bars, 2024-05 .. now. Server time
    is New York + 7h all year, so the 16:30 bar is the 9:30-9:35 ET opening range.
  * cost: the bar's own quoted spread, a ~$0.31 flat markup (measured 2026-11-05) — paid on the
    way in (buy at ask / short at bid) and out. Against a 10%-ATR stop this is large, so the
    report prints the spread in R.
  * volume: CFD TICK volume (quote updates), not shares. The 1M-share filter is therefore
    replaced by "median opening-range tick volume in the top half of the universe that day".
  * relative volume uses the same tick volume (a proxy; tick counts track activity).

VARIANTS (pre-registered, all printed):
  BASE      all eligible names
  SIP20     the paper's Stocks-in-Play top 20 (RVol >= 1)
  SIP20_CHEAP  SIP20, but only names whose stop is >= 6 x the spread (cost <= ~0.33R round trip)
  and each GROSS (spread = 0) to separate "no edge" from "edge eaten by cost".
CONTROL: random DIRECTION — the same days, names, entry levels and stops, but long/short drawn
by coin flip (200 draws). If the first candle's direction carries information, the real
strategy beats this null; if it does not, the paper's edge is the breakout + tight stop + trend
day, not the direction rule.

    python scripts/v5_orb_stocks_in_play.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "us_m5"
OUT = ROOT / "data" / "v5_runs" / "orb_sip"
RISK, LEV, TOPN, STOP_ATR = 0.01, 4.0, 20, 0.10
N_NULL, SEED = 200, 5
OPEN, CLOSE_LAST = "16:30", "22:55"          # server time = ET + 7h


def day_events(f: Path, point: float) -> pd.DataFrame | None:
    """One row per RTH day: opening-range stats and the bar path needed to simulate."""
    d = pd.read_csv(f, parse_dates=["t"]).set_index("t")
    hm = d.index.strftime("%H:%M")
    d = d[(hm >= OPEN) & (hm <= CLOSE_LAST)]
    if len(d) < 78 * 40:
        return None
    d["spr"] = d.sp * point
    day = d.index.normalize()
    g = d.groupby(day)
    daily = pd.DataFrame({"h": g.h.max(), "l": g.l.min(), "c": g.c.last(), "n": g.size()})
    pc = daily.c.shift(1)
    tr = pd.concat([daily.h - daily.l, (daily.h - pc).abs(), (daily.l - pc).abs()], axis=1).max(1)
    daily["atr"] = tr.rolling(14).mean().shift(1)                  # previous 14 days only
    first = d[d.index.strftime("%H:%M") == OPEN]
    first = first.set_index(first.index.normalize())
    daily = daily.join(first[["o", "h", "l", "c", "tv", "spr"]].add_prefix("or_"), how="inner")
    daily["rvol"] = daily.or_tv / daily.or_tv.rolling(14).mean().shift(1)
    daily["tv_med14"] = daily.or_tv.rolling(14).median().shift(1)
    daily = daily[(daily.n >= 70)].dropna(subset=["atr", "rvol"])
    rows = []
    for dt, r in daily.iterrows():
        if r.or_c == r.or_o:
            continue
        side = 1 if r.or_c > r.or_o else -1
        bars = d.loc[(d.index >= dt) & (d.index < dt + pd.Timedelta(days=1))].iloc[1:]
        rows.append(dict(day=dt, open=r.or_o, atr=r.atr, rvol=r.rvol, tvm=r.tv_med14,
                         side=side, orh=r.or_h, orl=r.or_l,
                         path=(bars.o.to_numpy(), bars.h.to_numpy(), bars.l.to_numpy(),
                               bars.c.to_numpy(), bars.spr.to_numpy())))
    return pd.DataFrame(rows)


def trade(ev, side: int, cost: bool) -> tuple[float, float] | None:
    """Simulate one day. Returns (R multiple, R distance in $) or None if never triggered."""
    o, h, l, c, s = ev["path"]
    s = s if cost else np.zeros_like(s)
    R = STOP_ATR * ev["atr"]
    lvl = ev["orh"] if side == 1 else ev["orl"]
    for i in range(len(o)):
        if side == 1 and h[i] + s[i] >= lvl:          # buy-stop on the ASK
            fill = max(lvl, o[i] + s[i])
            sl = fill - R
            if l[i] <= sl:                             # same-bar stop: pessimistic
                return -(fill - sl) / R, R
            break
        if side == -1 and l[i] <= lvl:                 # sell-stop on the BID
            fill = min(lvl, o[i])
            sl = fill + R
            if h[i] + s[i] >= sl:
                return -(sl - fill) / R, R
            break
    else:
        return None
    for j in range(i + 1, len(o)):
        if side == 1 and l[j] <= sl:
            return (min(sl, o[j]) - fill) / R, R
        if side == -1 and h[j] + s[j] >= sl:
            return (fill - max(sl, o[j] + s[j])) / R, R
    px = c[-1] if side == 1 else c[-1] + s[-1]
    return side * (px - fill) / R, R


def portfolio(sel: pd.DataFrame, sides: np.ndarray, cost: bool) -> pd.Series:
    """Daily return: 1% risk per triggered trade, each position's notional capped at 4x/N."""
    n_by_day = sel.groupby("day").size()
    rets = {}
    for k, ev in enumerate(sel.itertuples(index=False)):
        res = trade(ev._asdict(), int(sides[k]), cost)
        if res is None:
            continue
        rm, R = res
        cap_frac = (LEV / n_by_day[ev.day]) * R / ev.open       # risk fraction allowed by cap
        rets.setdefault(ev.day, 0.0)
        rets[ev.day] += rm * min(RISK, cap_frac)
    return pd.Series(rets).sort_index()


def stats(r: pd.Series, all_days: pd.DatetimeIndex) -> dict:
    r = r.reindex(all_days, fill_value=0.0)
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
    eq = (1 + r).cumprod()
    return dict(ann=100 * (eq.iloc[-1] ** (252 / len(r)) - 1), sharpe=sh,
                mdd=100 * (1 - eq / eq.cummax()).max())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(DATA / "_meta.csv", index_col=0) if (DATA / "_meta.csv").exists() else None
    evs = []
    files = sorted(DATA.glob("*.csv.gz"))
    for f in files:
        sym = f.name[:-7]
        pt = float(meta.loc[sym, "point"]) if meta is not None and sym in meta.index else 0.001
        e = day_events(f, pt)
        if e is not None and len(e):
            e["sym"] = sym
            evs.append(e)
    ev = pd.concat(evs, ignore_index=True)
    ev = ev[(ev.open > 5) & (ev.atr > 0.5)]
    # liquidity proxy for the paper's 1M-share filter: top half by 14-day opening tick volume
    ev = ev[ev.tvm >= ev.groupby("day").tvm.transform("median")]
    ev["spread_R"] = [float(np.median(p[4][:12])) / (STOP_ATR * a)
                      for p, a in zip(ev.path, ev.atr)]
    days = pd.DatetimeIndex(sorted(ev.day.unique()))
    sip = ev[ev.rvol >= 1].copy()
    sip["rk"] = sip.groupby("day").rvol.rank(ascending=False, method="first")
    sip = sip[sip.rk <= TOPN]
    cheap = sip[sip.spread_R <= 1 / 6]
    lines = [f"{ev.sym.nunique()} symbols, {len(days)} days {days[0]:%Y-%m-%d}..{days[-1]:%Y-%m-%d}; "
             f"median spread {ev.spread_R.median():.2f}R per side (SIP20 {sip.spread_R.median():.2f}R)"]
    rng = np.random.default_rng(SEED)
    rows = []
    for name, sel in (("BASE", ev), ("SIP20", sip), ("SIP20_CHEAP", cheap)):
        sel = sel.reset_index(drop=True)
        for cost in (True, False):
            st = stats(portfolio(sel, sel.side.to_numpy(), cost), days)
            null = [stats(portfolio(sel, rng.choice([-1, 1], len(sel)), cost), days)["sharpe"]
                    for _ in range(N_NULL if name != "BASE" else 30)]
            rows.append(dict(variant=name, cost="NET" if cost else "GROSS", n_cand=len(sel),
                             **{k: round(v, 2) for k, v in st.items()},
                             rand_dir_p50=round(np.median(null), 2),
                             rand_dir_p95=round(np.percentile(null, 95), 2),
                             p=round(float((np.array(null) >= st["sharpe"]).mean()), 3)))
            print(rows[-1], flush=True)
    txt = "\n".join(lines) + "\n" + pd.DataFrame(rows).to_string(index=False) + (
        "\n\nann/sharpe/mdd at 1% risk per trade; p = share of random-DIRECTION replays "
        "(same names, days, levels, stops) with Sharpe >= the strategy's.\n")
    print(txt)
    (OUT / "report.txt").write_text(txt)


if __name__ == "__main__":
    sys.exit(main())
