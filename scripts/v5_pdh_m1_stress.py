#!/usr/bin/env python3
"""P1 (prior-day-high buy-stop, long-only, flat at the day's end) passed all five pre-registered
bars in scripts/v5_intraday_m1.py. This script tries to BREAK it (V5_FINDINGS 3ex, part 2).

Everything here can only hurt the result or describe it; nothing selects a better cell.
  A. EXECUTION. 3ci found 47% of this lead was a fill assumption on H1. With M1 the assumption can
     be removed outright: enter at the OPEN OF THE MINUTE AFTER the first M1 close above the level
     (a bot polling once a minute, market order, no resting stop at all); also 5 minutes late;
     also on the H1 close (3ci's failing path, as the known answer). Plus a slippage ladder.
  B. EXIT TIME. The day's last bar is the thinnest hour (3cl). Also 16:00 and 16:55 ET.
  C. PLACEBO LEVEL (3cj, control 16): the same rule with the level at another day's
     ATR-distance above the open, 200 circular shifts.
  D. BUY-AND-HOLD in money (control 2): unit-notional year table against gold itself.
  E. RISK (pre-registered 2c): the 3eu downside-vol half-size dial and the 3dz circuit breaker,
     each against a random same-count control. Then the dial table.
  F. PROTECTIVE STOP ladder - EXPLORATORY, not pre-registered: insurance, event set fixed.
  G. SECOND SOURCES: repo M15 2015-26, CME GC H1 2010-26, Switch M15 2024-26.

    python scripts/v5_pdh_m1_stress.py
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
from scripts.v5_intraday_m1 import COST, NY, OUT, START, load_m1, sharpe, vol_weight  # noqa: E402

N_PLACEBO, N_RAND = 200, 500


def to_days(d: pd.DataFrame, min_bars: int):
    """d: OHLC with an ET-naive index. Returns the per-day slices and the daily bar table
    (trading day = 17:00 ET -> 17:00 ET, exactly as scripts/v5_intraday_m1.prior_day)."""
    key = (d.index + pd.Timedelta(hours=7)).normalize()
    codes, start = np.unique(key.values, return_index=True)
    end = np.append(start[1:], len(d))
    keep = (end - start) >= min_bars
    codes, start, end = codes[keep], start[keep], end[keep]
    A = {k: d[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close")}
    mod = (d.index.hour * 60 + d.index.minute).to_numpy()
    bar = pd.DataFrame(dict(open=A["open"][start], high=[A["high"][a:b].max() for a, b in zip(start, end)],
                            low=[A["low"][a:b].min() for a, b in zip(start, end)], close=A["close"][end - 1]),
                       index=pd.DatetimeIndex(codes, name="day"))
    tr = np.maximum.reduce([bar.high - bar.low, (bar.high - bar.close.shift()).abs(), (bar.low - bar.close.shift()).abs()])
    bar["atr"] = pd.Series(tr, index=bar.index).rolling(14).mean().shift(1)
    return A, mod, start, end, bar


def run(A, mod, start, end, bar, level=None, entry="stop", exit_at=None, stop_k=None, block=None):
    """One long per day. level: array per day (default = previous day's high).
    entry: stop | mclose | delay5 | h1close.  exit_at: minutes-of-day ET cut-off (None = last bar).
    stop_k: protective stop at entry - k x ATR14 (None = none).  block: per-day bool = no entry.
    Returns per-day raw return (0 if no trade), traded flag, time-in-trade fraction, entry price."""
    O, H, L, C = A["open"], A["high"], A["low"], A["close"]
    lv = bar.high.shift(1).to_numpy() if level is None else level
    atr = bar.atr.to_numpy()
    n = len(bar)
    ret, on, frac, epx, tin = np.zeros(n), np.zeros(n, bool), np.zeros(n), np.full(n, np.nan), np.full(n, -1)
    for i in range(n):
        if not np.isfinite(lv[i]) or (block is not None and block[i]):
            continue
        a, b = start[i], end[i]
        last = b - 1
        if exit_at is not None:                                   # last bar strictly before the cut-off
            m = mod[a:b]
            pre = np.flatnonzero((m < exit_at) | (m >= 17 * 60))  # evening bars belong to the day's start
            if not len(pre):
                continue
            last = a + pre[-1]
        if entry == "stop":
            hit = np.flatnonzero(H[a:last + 1] >= lv[i])
            if not len(hit):
                continue
            t = a + hit[0]
            e = max(lv[i], O[t])
        else:
            if entry == "h1close":
                cand = np.flatnonzero((mod[a:last + 1] % 60 == 59) & (C[a:last + 1] > lv[i]))
            else:
                cand = np.flatnonzero(C[a:last + 1] > lv[i])
            if not len(cand):
                continue
            t = a + cand[0] + (5 if entry == "delay5" else 1)     # the open of the next (or 5th next) minute
            if t > last:
                continue
            e = O[t]
        px = C[last]
        if stop_k is not None and np.isfinite(atr[i]):
            s = e - stop_k * atr[i]
            lo = np.flatnonzero(L[t + 1:last + 1] <= s)
            if L[t] <= s and entry == "stop":                     # entry minute also covers the stop: assume hit
                px = s
            elif len(lo):
                j = t + 1 + lo[0]
                px, last = min(s, O[j]), j
        ret[i], on[i], frac[i], epx[i], tin[i] = px / e - 1, True, (last - t + 1) / (b - a), e, t - a
    return ret, on, frac, epx, tin


def stats(name, ret, on, frac, w, hold, px, yrs, cost=COST) -> dict:
    gross, cost1, expo = w * ret, w * on / px, w * frac
    net, net2 = gross - cost * cost1, gross - 2 * cost * cost1
    a = gross - expo.mean() * hold
    half = len(ret) // 2
    return dict(variant=name, trades=int(on.sum()), per_wk=round(on.sum() / (len(on) / 5), 2),
                hit=round(100 * (ret[on] > 0).mean(), 1), gross_bp=round(1e4 * ret[on].mean(), 2),
                expo=round(float(frac.mean()), 3), SR_gross=round(sharpe(gross), 2), SR=round(sharpe(net), 2),
                SR_2x=round(sharpe(net2), 2), h1=round(sharpe(net[:half]), 2), h2=round(sharpe(net[half:]), 2),
                yrs_pos=f"{sum(sharpe(net[yrs == y]) > 0 for y in np.unique(yrs))}/{len(np.unique(yrs))}",
                t_vs_hold=round(float(a.mean() / a.std() * np.sqrt(len(a))), 2),
                breakeven=round(float(gross.sum() / cost1.sum()), 2) if cost1.sum() > 0 else np.nan)


def source_frame(name: str) -> tuple[pd.DataFrame, int]:
    """Second sources as ET-naive OHLC + the bar count that makes a valid day."""
    if name == "m15":
        d = pd.read_csv(ROOT / "data/XAUUSD_M15_long.csv", parse_dates=["time"], index_col="time")
        d.index = d.index.tz_localize("UTC").tz_convert(NY).tz_localize(None)
        return d[["open", "high", "low", "close"]], 40
    if name == "cme":
        from scripts.v5_zigzag_panel_probs import load_cme  # noqa: PLC0415
        d = load_cme("GC")
        d.index = d.index.tz_localize("UTC").tz_convert(NY).tz_localize(None)
        return d[["open", "high", "low", "close"]], 10
    d = pd.read_csv(ROOT / "data/sweep_switch/XAUUSDs_M15.csv", parse_dates=["time"]).set_index("time").sort_index()
    d.index = d.index - pd.Timedelta(hours=7)                     # Switch server = New York + 7h (3do)
    return d[["open", "high", "low", "close"]], 40


def main() -> None:
    L = []
    d = load_m1()
    A, mod, start, end, bar = to_days(d, 600)
    hold = (bar.close / bar.open - 1).to_numpy()
    w, px, yrs = vol_weight(hold), bar.close.to_numpy(), bar.index.year.to_numpy()
    k = bar.index >= START
    S = lambda name, r: stats(name, r[0][k], r[1][k], r[2][k], w[k], hold[k], px[k], yrs[k])  # noqa: E731
    base = run(A, mod, start, end, bar)
    L += [f"P1 STRESS, OANDA M1 mid, {bar.index[k][0].date()} -> {bar.index[k][-1].date()}, {int(k.sum()):,} days, "
          f"cost ${COST} round trip", "", "A. EXECUTION and B. EXIT TIME"]
    rows = [S("resting stop at PDH (as pre-registered)", base),
            S("market, open of minute after 1st M1 close > PDH", run(A, mod, start, end, bar, entry="mclose")),
            S("same, 5 minutes late", run(A, mod, start, end, bar, entry="delay5")),
            S("market after the H1 close > PDH (3ci path)", run(A, mod, start, end, bar, entry="h1close")),
            S("resting stop, exit 16:55 ET", run(A, mod, start, end, bar, exit_at=16 * 60 + 55)),
            S("resting stop, exit 16:00 ET", run(A, mod, start, end, bar, exit_at=16 * 60)),
            S("minute-close entry, exit 16:00 ET", run(A, mod, start, end, bar, entry="mclose", exit_at=16 * 60))]
    T = pd.DataFrame(rows)
    L += [T.to_string(index=False), ""]
    r0, on0, fr0, e0, tin0 = (x[k] for x in base)
    g0, c1 = w[k] * r0, w[k] * on0 / px[k]
    L.append("   slippage ladder on the resting stop (extra $ per round trip on top of $0.30): " + "  ".join(
        f"+${s:.2f} -> SR {sharpe(g0 - (COST + s) * c1):+.2f}" for s in (0.10, 0.25, 0.50, 0.75, 1.00, 1.50)))
    mc = run(A, mod, start, end, bar, entry="mclose")
    both = on0 & mc[1][k]
    L.append(f"   entry price, minute-close minus resting stop, same {int(both.sum())} days: mean "
             f"{1e4 * np.nanmean((mc[3][k][both] - e0[both]) / e0[both]):+.2f}bp, median "
             f"{1e4 * np.nanmedian((mc[3][k][both] - e0[both]) / e0[both]):+.2f}bp;  stop-only days (wick, no M1 close above): "
             f"{int((on0 & ~mc[1][k]).sum())}, mean {1e4 * r0[on0 & ~mc[1][k]].mean():+.2f}bp")
    hrs = (tin0[on0] / 60).astype(int)
    L.append("   entry hour after the 17:00 ET open (share of trades): " + "  ".join(
        f"{a}-{b}h {100 * ((hrs >= a) & (hrs < b)).mean():.0f}%" for a, b in ((0, 3), (3, 9), (9, 15), (15, 19), (19, 24))))

    # C. placebo level at a matched ATR distance
    dist = ((bar.high.shift(1) - bar.open) / bar.atr).to_numpy()
    rng = np.random.default_rng(5)
    obs = S("PDH", base)
    pl = []
    for _ in range(N_PLACEBO):
        sh = int(rng.integers(20, len(bar) - 20))
        lvl = bar.open.to_numpy() + np.roll(dist, sh) * bar.atr.to_numpy()
        pl.append(S("placebo", run(A, mod, start, end, bar, level=lvl)))
    P = pd.DataFrame(pl)
    L += ["", "C. PLACEBO LEVEL at another day's ATR-distance above the open (200 circular shifts >= 20 days)"]
    for col in ("SR", "SR_gross", "t_vs_hold", "gross_bp", "expo"):
        L.append(f"   {col:10s} PDH {obs[col]:+.3f}   placebo median {P[col].median():+.3f}  95th {P[col].quantile(.95):+.3f}  "
                 f"p {(P[col] >= obs[col]).mean():.3f}")

    # D. unit notional versus gold itself
    L += ["", "D. IN MONEY, UNIT NOTIONAL (100% of equity long from the fill to the day's end; no vol scaling)"]
    un = pd.Series(r0 - COST * on0 / px[k], index=bar.index[k])
    c2c = bar.close.pct_change().fillna(0)[k]
    cur = lambda x: (1 + x).cumprod()  # noqa: E731
    dd = lambda x: float((cur(x) / cur(x).cummax() - 1).min())  # noqa: E731
    yrs_n = len(un) / 252
    for nm, x in (("P1 net", un), ("HOLD gold (close to close)", c2c)):
        cagr = cur(x).iloc[-1] ** (1 / yrs_n) - 1
        L.append(f"   {nm:27s} CAGR {100 * cagr:+6.2f}%  Sharpe {sharpe(x):+.2f}  maxDD {100 * dd(x):6.1f}%  "
                 f"Calmar {cagr / -dd(x):.2f}  worst day {100 * x.min():+.2f}%  x{cur(x).iloc[-1]:.2f}")
    yt = pd.DataFrame({"P1": un.groupby(un.index.year).apply(lambda x: 100 * ((1 + x).prod() - 1)),
                       "gold": c2c.groupby(c2c.index.year).apply(lambda x: 100 * ((1 + x).prod() - 1))}).round(1)
    L.append("   year: " + "  ".join(f"{y} {a:+.0f}/{b:+.0f}" for y, a, b in zip(yt.index, yt.P1, yt.gold)))
    L.append(f"   P1 positive in {int((yt.P1 > 0).sum())}/{len(yt)} years; beats gold in {int((yt.P1 > yt.gold).sum())}/{len(yt)}; "
             f"P1 in gold's down years: " + "  ".join(f"{y} {a:+.0f} vs {b:+.0f}" for y, a, b in zip(yt.index, yt.P1, yt.gold) if b < 0))

    # E. pre-registered risk overlays, each against a random same-count control
    h1 = d.close.resample("60min").last().dropna()
    r = h1.pct_change()
    dn = r.where(r < 0, 0.0).rolling(500).std()
    calm = (dn < dn.rolling(2000).median()).shift(1)             # state as of the last CLOSED H1 bar
    first_bar = pd.DatetimeIndex(d.index[start])
    risky = ~calm.reindex(first_bar, method="ffill").fillna(True).astype(bool).to_numpy()   # known at the day's open
    net0 = g0 - COST * c1
    half = len(net0) // 2
    L += ["", "E. RISK OVERLAYS (pre-registered 2c), each vs a RANDOM same-count control, event set fixed"]
    rk = risky[k] & on0
    m = np.where(rk, 0.5, 1.0)
    rng = np.random.default_rng(9)
    idx_on = np.flatnonzero(on0)
    rnd = np.empty((N_RAND, 3))
    for j in range(N_RAND):
        mm = np.ones(len(net0)); mm[rng.choice(idx_on, int(rk.sum()), replace=False)] = 0.5
        x = mm * net0
        rnd[j] = sharpe(x), sharpe(x[:half]), sharpe(x[half:])
    x = m * net0
    L.append(f"   (i) 3eu half size in high downside vol: {int(rk.sum())}/{int(on0.sum())} trades halved. "
             f"SR {sharpe(net0):+.3f} -> {sharpe(x):+.3f} (random median {np.median(rnd[:, 0]):+.3f}, pctile "
             f"{100 * (rnd[:, 0] <= sharpe(x)).mean():.0f});  halves {sharpe(x[:half]):+.2f} / {sharpe(x[half:]):+.2f} "
             f"(pctile {100 * (rnd[:, 1] <= sharpe(x[:half])).mean():.0f} / {100 * (rnd[:, 2] <= sharpe(x[half:])).mean():.0f});  "
             f"maxDD {100 * dd(pd.Series(net0)):.1f}% -> {100 * dd(pd.Series(x)):.1f}%;  mean R of halved trades "
             f"{1e4 * r0[rk].mean():+.2f}bp vs kept {1e4 * r0[on0 & ~rk].mean():+.2f}bp")
    # circuit breaker: the day had already traded > 1 ATR below its open before the trigger
    blk = np.zeros(len(bar), bool)
    Lo, O_ = A["low"], bar.open.to_numpy()
    for i in np.flatnonzero(base[1]):
        a = start[i]
        t = int(base[4][i])
        blk[i] = t > 0 and np.isfinite(bar.atr.iloc[i]) and Lo[a:a + t].min() < O_[i] - bar.atr.iloc[i]
    L.append(f"   (ii) 3dz circuit breaker (day already > 1 ATR below its open before the trigger): blocks "
             f"{int(blk[k].sum())} of {int(on0.sum())} trades -> not applicable to this rule (a prior-high break cannot "
             f"follow a 1-ATR fall the same day except {int(blk[k].sum())} times)")

    # F. protective stop ladder - exploratory
    L += ["", "F. PROTECTIVE STOP at entry - k x ATR14(D1): EXPLORATORY (not pre-registered), same entries, exact M1 path"]
    rowsF = []
    for kk in (None, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25):
        rr = run(A, mod, start, end, bar, stop_k=kk)
        s_ = S("no stop" if kk is None else f"stop {kk} ATR", rr)
        un_k = pd.Series(rr[0][k] - COST * rr[1][k] / px[k])
        stopped = 100 * np.mean(rr[2][k][rr[1][k]] < fr0[on0] - 1e-9) if kk is not None else 0.0
        s_.update(stopped_pct=round(stopped, 1), worst_trade=round(100 * rr[0][k].min(), 2),
                  unit_maxDD=round(100 * dd(un_k), 1), unit_CAGR=round(100 * (cur(un_k).iloc[-1] ** (1 / yrs_n) - 1), 2))
        rowsF.append(s_)
    F = pd.DataFrame(rowsF)[["variant", "trades", "hit", "gross_bp", "SR", "SR_2x", "h1", "h2", "stopped_pct",
                             "worst_trade", "unit_maxDD", "unit_CAGR"]]
    L += [F.to_string(index=False)]
    atr_pct = (bar.atr / bar.open)[k]
    L.append(f"   ATR14(D1) as % of price: median {100 * atr_pct.median():.2f}%, 2025-26 {100 * atr_pct[atr_pct.index >= '2025-01-01'].median():.2f}%")

    # dial table on the 1-ATR-stop variant: risk per trade = the stop distance x size
    rr = run(A, mod, start, end, bar, stop_k=1.0)
    Rm = (rr[0][k] - COST * rr[1][k] / px[k]) / (bar.atr.to_numpy()[k] / np.where(np.isfinite(rr[3][k]), rr[3][k], 1.0))
    Rm = np.nan_to_num(Rm)
    L += ["", "   RISK DIAL on the 1-ATR stop variant (R = net P&L / the 1-ATR stop distance; one trade per day at most):",
          f"   R per trade: mean {Rm[rr[1][k]].mean():+.3f}, sd {Rm[rr[1][k]].std():.3f}, worst {Rm.min():+.2f}, best {Rm.max():+.2f}, "
          f"P(R <= -1) {100 * (Rm[rr[1][k]] <= -0.999).mean():.1f}%"]
    for risk in (0.0025, 0.005, 0.01, 0.02):
        x = pd.Series(risk * Rm)
        wk = x.groupby(np.arange(len(x)) // 5).sum()
        L.append(f"   risk {100 * risk:.2f}%/trade: CAGR {100 * (cur(x).iloc[-1] ** (1 / yrs_n) - 1):+5.2f}%  maxDD {100 * dd(x):5.1f}%  "
                 f"worst day {100 * x.min():+.2f}%  worst week {100 * wk.min():+.2f}%  days <= -3%: {int((x <= -0.03).sum())}  "
                 f"Sharpe {sharpe(x):+.2f}")

    # G. second sources
    L += ["", "G. SECOND SOURCES (resting stop at PDH on that feed's own bars; same cost)"]
    for nm in ("m15", "cme", "switch"):
        dd_, mb = source_frame(nm)
        A2, mod2, st2, en2, bar2 = to_days(dd_, mb)
        h2 = (bar2.close / bar2.open - 1).to_numpy()
        w2, yr2 = vol_weight(h2), bar2.index.year.to_numpy()
        r2 = run(A2, mod2, st2, en2, bar2)
        k2 = np.arange(len(bar2)) >= 30
        px2 = bar2.close.to_numpy()
        if nm == "cme":                                           # cost must be charged on the REAL price, not the back-adjusted one
            raw = pd.read_csv(ROOT / "data/databento/ohlcv_1h/GC.csv.gz", parse_dates=["time"], index_col="time").close
            raw.index = raw.index.tz_localize("UTC").tz_convert(NY).tz_localize(None)
            px2 = raw.groupby((raw.index + pd.Timedelta(hours=7)).normalize()).last().reindex(bar2.index).ffill().to_numpy()
        s2 = stats(nm, r2[0][k2], r2[1][k2], r2[2][k2], w2[k2], h2[k2], px2[k2], yr2[k2])
        ko = (bar.index >= bar2.index[k2][0]) & (bar.index <= bar2.index[-1])
        so = stats("oanda", base[0][ko], base[1][ko], base[2][ko], w[ko], hold[ko], px[ko], yrs[ko])
        L.append(f"   {nm:6s} {bar2.index[k2][0].date()} -> {bar2.index[-1].date()}: trades {s2['trades']}, gross {s2['gross_bp']:+.2f}bp, "
                 f"SR gross {s2['SR_gross']:+.2f} net {s2['SR']:+.2f} 2x {s2['SR_2x']:+.2f}, halves {s2['h1']:+.2f}/{s2['h2']:+.2f}, "
                 f"t vs hold {s2['t_vs_hold']:+.2f}   | OANDA M1 same window: net {so['SR']:+.2f}, t {so['t_vs_hold']:+.2f}")
    txt = "\n".join(L)
    print(txt)
    (OUT / "pdh_stress_report.txt").write_text(txt + "\n")
    pd.DataFrame(dict(ret=base[0], on=base[1], frac=base[2], entry=base[3], t_in=base[4], w=w, hold=hold, px=px,
                      atr=bar.atr.to_numpy(), risky=risky), index=bar.index).to_parquet(OUT / "pdh_trades.parquet")


if __name__ == "__main__":
    sys.exit(main())
