#!/usr/bin/env python3
"""The working bit of §3dh (first-candle DIRECTION at the US open) moved to gold and BTC, where
the spread is small against a 10%-ATR stop, then tested INSIDE the champion.

Why this is not §3be again: §3be scored session-open breakouts/fades on fixed forward returns
(and was later voided, §3bx). This is the paper's exact trade: direction of the first candle at
9:30 ET, stop order at the opening-range high/low, stop 10% of ATR14, flat at 16:00 ET, 1% risk.

DATA (single provenance per asset)
  XAU  data/XAUUSD_M15_long.csv (UTC, 2015..2026-06) -> opening range = the 9:30-9:45 ET M15 bar
  BTC  data/binance_1m/BTCUSDT.npz (UTC, 2017-08..) resampled to M5 -> the paper's 9:30-9:35 bar
  Times are converted UTC -> America/New_York, so DST is exact.
COST per side: XAU max($0.25 Switch, bar spread) and a $0.448 stress; BTC 0.05% of price (Switch
  ~$35 at ~$70k). Bars carry no bid/ask split here, so the spread is charged on entry and exit.
CONTROLS: random direction (same days, levels, stops; 300 draws) on GROSS and NET; LONG-ONLY
  (drift assets: shorts have lost on gold in every family tested here); halves.
INTEGRATION (the user's goal: improve the best strategy): XAU champion (H4, `engine_v`, 2018+)
  plus the ORB sleeve at equal risk, judged by `report()` = paired t at MATCHED VOL, years better.

    python scripts/v5_orb_xau_btc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_orb_stocks_in_play import sim  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "orb_xau_btc"
STOP_ATR, RISK, N_NULL, SEED = 0.10, 0.01, 300, 11


def load_xau() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / "XAUUSD_M15_long.csv", parse_dates=["time"], index_col="time")
    d = d[~d.index.duplicated()].sort_index()
    d["spr"] = d.spread * 0.1                       # points -> $ (same scale as the champion)
    return d.rename(columns=dict(open="o", high="h", low="l", close="c"))[["o", "h", "l", "c", "spr"]]


def load_btc() -> pd.DataFrame:
    z = np.load(ROOT / "data" / "binance_1m" / "BTCUSDT.npz")
    d = pd.DataFrame({k: z[k] for k in "ohlc"}, index=pd.to_datetime(z["t"], unit="s"))
    d = d.resample("5min").agg(dict(o="first", h="max", l="min", c="last")).dropna()
    d["spr"] = 0.0
    return d


def events(d: pd.DataFrame, bar_min: int, cost_fn):
    """One row per NY weekday: OR bar at 9:30 ET, path = bars until 16:00 ET."""
    et = d.index.tz_localize("UTC").tz_convert("America/New_York")
    mins = et.hour * 60 + et.minute
    day = et.tz_localize(None).normalize()
    # ATR14 from full UTC days (24h assets), previous days only
    g = d.groupby(d.index.normalize())
    dly = pd.DataFrame(dict(h=g.h.max(), l=g.l.min(), c=g.c.last()))
    pc = dly.c.shift(1)
    tr = pd.concat([dly.h - dly.l, (dly.h - pc).abs(), (dly.l - pc).abs()], axis=1).max(1)
    atr = tr.rolling(14).mean().shift(1)
    sess = (mins >= 570) & (mins < 960) & (et.dayofweek < 5)
    idx = np.flatnonzero(sess)
    o, h, l, c = (d[k].to_numpy(float) for k in "ohlc")
    s = cost_fn(d).to_numpy(float)
    dd = day[idx]
    brk = np.r_[0, np.flatnonzero(dd[1:] != dd[:-1]) + 1]
    end = np.r_[brk[1:], len(idx)]
    rows = []
    for b, e in zip(brk, end):
        i0 = idx[b]
        if mins[i0] != 570 or e - b < (390 // bar_min) * 0.9:
            continue
        if idx[e - 1] - i0 != e - 1 - b:            # gap inside the session
            continue
        a = atr.get(d.index[i0].normalize(), np.nan)
        if not np.isfinite(a) or c[i0] == o[i0]:
            continue
        rows.append((dd[b], o[i0], a, int(np.sign(c[i0] - o[i0])), h[i0], l[i0], i0 + 1, idx[e - 1] + 1))
    ev = pd.DataFrame(rows, columns=["day", "open", "atr", "side", "orh", "orl", "s0", "s1"])
    return ev, (o, h, l, c, s)


def run(ev, arrs, cost: bool):
    o, h, l, c, s = arrs
    R = STOP_ATR * ev.atr.to_numpy()
    lvl_h = ev.orh.to_numpy() + (s[ev.s0 - 1] if cost else 0)   # long stop at the ask-high
    z = s if cost else np.zeros_like(s)
    args = (lvl_h, ev.orl.to_numpy(), R, ev.s0.to_numpy(), ev.s1.to_numpy(), o, h, l, c, z, False)
    n = len(ev)
    L = sim(np.full(n, 1), *args)
    S = sim(np.full(n, -1), *args)
    return L, S


def sharpe(r) -> float:
    r = np.asarray(r, float)
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0


def daily(ev, rm) -> pd.Series:
    r = pd.Series(np.nan_to_num(rm) * RISK, index=pd.DatetimeIndex(ev.day))
    return r.groupby(level=0).sum()


def study(name, ev, arrs, lines, store):
    rng = np.random.default_rng(SEED)
    side = ev.side.to_numpy()
    h = len(ev) // 2
    for cost in (False, True):
        L, S = run(ev, arrs, cost)
        for mode in ("LS", "LONG"):
            real = side if mode == "LS" else np.where(side == 1, 1, 0)
            rm = np.where(real == 1, L, np.where(real == -1, S, np.nan))
            sh = sharpe(daily(ev, rm))
            null = []
            for _ in range(N_NULL):
                rs = rng.choice([-1, 1], len(ev))
                if mode == "LONG":                   # same number of long days, random which
                    rs = np.zeros(len(ev), int)
                    rs[rng.choice(len(ev), int((real == 1).sum()), replace=False)] = 1
                null.append(sharpe(daily(ev, np.where(rs == 1, L, np.where(rs == -1, S, np.nan)))))
            null = np.array(null)
            trig = np.isfinite(rm)
            lines.append(f"{name:4} {mode:4} {'NET  ' if cost else 'GROSS'} trades {trig.sum():5d} "
                         f"win {100*(rm[trig] > 0).mean():4.1f}% avgR {np.nanmean(rm):+.3f}  "
                         f"Sharpe {sh:+.2f} (halves {sharpe(daily(ev.iloc[:h], rm[:h])):+.2f}/"
                         f"{sharpe(daily(ev.iloc[h:], rm[h:])):+.2f})  random p50 {np.median(null):+.2f} "
                         f"p95 {np.percentile(null, 95):+.2f}  p {np.mean(null >= sh):.3f}")
            print(lines[-1], flush=True)
            store[(name, mode, cost)] = daily(ev, rm)


def integrate(store, lines):
    from scripts.v5_xau_champion_lifts import COST_USD, engine_v, report, sharpe as shp, dd_of
    from scripts.v5_xau_turn_prob import load_h4
    from src.v5.xau_dual_signals import champion_signal
    df = load_h4(COST_USD)
    base = engine_v(df, champion_signal(df["close"]))
    lines.append(f"\nINTEGRATION — XAU champion {base.index[0]:%Y-%m-%d}..{base.index[-1]:%Y-%m-%d}: "
                 f"SR {shp(base):+.3f} DD {dd_of(base):+.1f}%")
    for mode in ("LS", "LONG"):
        orb = store[("XAU", mode, True)].reindex(base.index, fill_value=0.0)
        corr = base.corr(orb)
        for w in (0.25, 0.5):                       # sleeve share of RISK
            k = base.std() / orb.std() if orb.std() > 0 else 0
            book = (1 - w) * base + w * k * orb
            r = report(f"champ+ORB_{mode}_{int(w*100)}", book, base)
            lines.append(f"  {r['name']:20} SR {r['sharpe']:+.3f}  DD {r['dd']:+.1f}%  "
                         f"t@matched {r['t_vs_base']:+.2f}  years {r['years_better']}  corr {corr:+.2f}")
            print(lines[-1], flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines, store = [], {}
    x = load_xau()
    ev, arrs = events(x, 15, lambda d: d.spr.clip(lower=0.25))
    lines.append(f"XAU M15 {ev.day.min():%Y-%m-%d}..{ev.day.max():%Y-%m-%d}, {len(ev)} days; "
                 f"median spread/R {np.median(0.25 / (STOP_ATR * ev.atr)):.3f}")
    study("XAU", ev, arrs, lines, store)
    ev448, arrs448 = events(x, 15, lambda d: d.spr.clip(lower=0.448))
    L, S = run(ev448, arrs448, True)
    rm = np.where(ev448.side == 1, L, S)
    lines.append(f"XAU  LS   NET @$0.448 stress: Sharpe {sharpe(daily(ev448, rm)):+.2f}")
    b = load_btc()
    evb, arrb = events(b, 5, lambda d: d.c * 0.0005)
    lines.append(f"BTC M5 {evb.day.min():%Y-%m-%d}..{evb.day.max():%Y-%m-%d}, {len(evb)} days; "
                 f"median spread/R {np.median(evb.open * 0.0005 / (STOP_ATR * evb.atr)):.3f}")
    study("BTC", evb, arrb, lines, store)
    integrate(store, lines)
    pd.DataFrame({f"{a}_{m}_{'net' if c else 'gross'}": v for (a, m, c), v in store.items()}
                 ).to_csv(OUT / "daily_returns.csv")
    txt = "\n".join(lines) + "\n"
    print("\n" + txt)
    (OUT / "report.txt").write_text(txt)


if __name__ == "__main__":
    sys.exit(main())
