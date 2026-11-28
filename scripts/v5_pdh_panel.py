#!/usr/bin/env python3
"""ROUND 2, STUDIES A / B / D (V5_FINDINGS §3ez, §3fa) — the live PDH rule, FROZEN, on 24
instruments; level salience; gold in euros. Pre-registered at 3c4be02
(data/v5_runs/intraday_day_signal/PREREG_ROUND2.md). User, 2026-11-16: "can you try other techniques?"

The rule is §3ey's deployed variant with nothing varied by instrument: a resting stop at the prior
day's extreme (17:00 -> 17:00 New York day), skip a day that opens at/beyond the level, stop
1.0 x ATR14(D1), flat at the close of the last bar before 16:40 New York. LONG leg at the prior
high, SHORT leg the exact mirror at the prior low. OANDA M5 mid bars, one provenance.

It also writes the per-trade EVENT TABLE (features strictly before the fill) that
scripts/v5_pdh_meta.py (Study C) trains on.

    python scripts/v5_pdh_panel.py build        # per-instrument legs + events (cached)
    python scripts/v5_pdh_panel.py report       # Study A, B, D
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_intraday_m1 import reality_check, sharpe, vol_weight  # noqa: E402
from scripts.v5_pdh_m1_stress import run, to_days  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "intraday_day_signal"
CACHE = OUT / "panel"
NY, START, EXIT_AT, STOP_K = "America/New_York", "2007-01-01", 16 * 60 + 40, 1.0
GOLD_COST_USD = 0.30
PANEL = {  # symbol: (class, round-trip cost in bp; None = gold's fixed $0.30)
    "NAS100USD": ("index", 1.0), "US30USD": ("index", 1.0), "DE30EUR": ("index", 1.0), "SPX500USD": ("index", 2.0),
    "JP225USD": ("index", 2.5), "AU200AUD": ("index", 3.5), "US2000USD": ("index", 5.0), "UK100GBP": ("index", 6.0),
    "EURUSD": ("fx", 1.5), "GBPUSD": ("fx", 1.5), "USDJPY": ("fx", 1.5), "USDCAD": ("fx", 1.5), "GBPJPY": ("fx", 1.5),
    "EURJPY": ("fx", 2.0), "AUDJPY": ("fx", 2.0), "EURGBP": ("fx", 2.0), "USDCHF": ("fx", 2.0), "AUDUSD": ("fx", 2.5),
    "NZDUSD": ("fx", 3.0),
    "XAUUSD": ("metal", None), "XAGUSD": ("metal", 5.5),
    "WTICOUSD": ("energy", 4.5), "BCOUSD": ("energy", 5.0),
    "BTCUSD": ("crypto", 4.5),
}
CLASSES = ("index", "fx", "metal", "energy", "crypto")


def load(sym: str) -> pd.DataFrame:
    d = pd.read_parquet(ROOT / f"data/oanda_hf/5/{sym}.parquet")
    idx = pd.DatetimeIndex(pd.to_datetime(d.time, unit="s")).tz_localize("UTC").tz_convert(NY).tz_localize(None)
    d = d[["open", "high", "low", "close"]].astype(float)
    d.index = idx
    return d[~d.index.duplicated(keep="last")]


def _fresh(x: np.ndarray) -> np.ndarray:
    """out[i] = number of consecutive days before i-1 whose value was below x[i-1] (cap 20)."""
    n = len(x)
    out = np.zeros(n)
    for i in range(2, n):
        k, j = 0, i - 2
        while j >= 0 and k < 20 and x[j] < x[i - 1]:
            k, j = k + 1, j - 1
        out[i] = k
    return out


def build(sym: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(daily legs, per-trade events) for one instrument."""
    cls, cost_bp = PANEL[sym]
    d = load(sym)
    key = (d.index + pd.Timedelta(hours=7)).normalize()
    counts = pd.Series(1, index=key).groupby(level=0).sum()
    A, mod, start, end, bar = to_days(d, int(0.5 * counts.median()))
    n = len(bar)
    t_all = d.index.values
    hold = (bar.close / bar.open - 1).to_numpy()
    w = vol_weight(hold)
    px = bar.close.to_numpy()
    cost_frac = (GOLD_COST_USD / px) if cost_bp is None else np.full(n, cost_bp * 1e-4)
    atr = bar.atr.to_numpy()
    O, H, L, C = (bar[k].to_numpy() for k in ("open", "high", "low", "close"))
    dret = np.r_[np.nan, C[1:] / C[:-1] - 1]

    # day x slot grids (slot 0 = 17:00 NY) for the "last 30 minutes vs its 20-day norm" feature
    slot = ((mod - 17 * 60) % 1440) // 5
    row = np.full(len(d), -1)
    for i in range(n):
        row[start[i]:end[i]] = i
    ok = row >= 0
    Hg, Lg = np.full((n, 288), np.nan), np.full((n, 288), np.nan)
    Hg[row[ok], slot[ok]] = A["high"][ok]
    Lg[row[ok], slot[ok]] = A["low"][ok]
    hi6 = np.fmax.reduce([np.roll(Hg, k, axis=1) for k in range(1, 7)])
    lo6 = np.fmin.reduce([np.roll(Lg, k, axis=1) for k in range(1, 7)])
    # roll wraps: the first 6 slots have no full window (a crashing redundant loop was removed
    # 2026-11-28, second-look #6; this line already did the masking, and only Study C uses rng30)
    hi6[:, :6], lo6[:, :6] = np.nan, np.nan
    rng30 = hi6 - lo6
    norm30 = pd.DataFrame(rng30).rolling(20, min_periods=10).mean().shift(1).to_numpy()

    legs, events = {}, []
    for side in (1, -1):
        if side == 1:
            Am, barm = A, bar
        else:                                           # mirror: a sell stop at the prior LOW
            Am = dict(open=-A["open"], high=-A["low"], low=-A["high"], close=-A["close"])
            barm = pd.DataFrame(dict(open=-bar.open, high=-bar.low, low=-bar.high, close=-bar.close, atr=bar.atr))
        lv = barm.high.shift(1).to_numpy()
        gap = np.array([np.isfinite(lv[i]) and Am["open"][start[i]] >= lv[i] for i in range(n)])
        r_m, on, frac, e_m, tin = run(Am, mod, start, end, barm, stop_k=STOP_K, exit_at=EXIT_AT, block=gap)
        ret = r_m if side == 1 else -r_m                # mirrored ratios keep their sign; a short earns the negative
        entry = side * e_m
        legs[side] = pd.DataFrame(dict(gross=w * ret, cost=w * on * cost_frac, expo=side * w * frac, on=on, hold=hold,
                                       ret=ret), index=bar.index)
        fresh = _fresh(barm.high.to_numpy())
        rdir = side * dret
        dn = pd.Series(np.where(rdir < 0, rdir, 0.0)).rolling(20).std()
        dvol = (dn / dn.rolling(250, min_periods=120).median()).shift(1).to_numpy()
        Hm, Lm, Om, Cm = Am["high"], Am["low"], Am["open"], Am["close"]
        for i in np.flatnonzero(on):
            a, t = start[i], start[i] + int(tin[i])
            if not np.isfinite(atr[i]) or atr[i] <= 0 or i < 21:
                continue
            day_lo = Lm[a:t].min() if t > a else Om[a]
            e_frac = atr[i] / abs(entry[i])
            events.append(dict(
                sym=sym, cls=cls, side=side, day=bar.index[i], t_trig=t_all[t], ret=ret[i],
                R=ret[i] / e_frac, R_net=(ret[i] - cost_frac[i]) / e_frac, cost_R=cost_frac[i] / e_frac,
                fresh=fresh[i], dist_open=(lv[i] - Om[a]) / atr[i],
                prev_range=(H[i - 1] - L[i - 1]) / atr[i],
                prev_clv=((C[i - 1] - L[i - 1]) / max(H[i - 1] - L[i - 1], 1e-12)) if side == 1
                else ((H[i - 1] - C[i - 1]) / max(H[i - 1] - L[i - 1], 1e-12)),
                ret5=side * (C[i - 1] / C[i - 6] - 1) / (atr[i] / C[i - 1] * np.sqrt(5)),
                ret20=side * (C[i - 1] / C[i - 21] - 1) / (atr[i] / C[i - 1] * np.sqrt(20)),
                dvol=dvol[i], weekday=bar.index[i].dayofweek, gap=side * (O[i] - C[i - 1]) / atr[i],
                mins_open=(t_all[t] - t_all[a]) / np.timedelta64(1, "m"), ny_hour=int(mod[t] // 60),
                adverse=(Om[a] - day_lo) / atr[i], run_up=(lv[i] - day_lo) / atr[i],
                speed30=(Om[t] - Om[max(a, t - 6)]) / atr[i], speed120=(Om[t] - Om[max(a, t - 24)]) / atr[i],
                rv30=rng30[i, slot[t]] / norm30[i, slot[t]] if np.isfinite(norm30[i, slot[t]]) and norm30[i, slot[t]] > 0 else np.nan,
                touches=int((Hm[a:t] >= lv[i] - 0.05 * atr[i]).sum())))
    L_, S_ = legs[1], legs[-1]
    daily = pd.concat({"L": L_, "S": S_}, axis=1)
    return daily, pd.DataFrame(events)


def cached(sym: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    CACHE.mkdir(parents=True, exist_ok=True)
    fd, fe = CACHE / f"{sym}_daily.parquet", CACHE / f"{sym}_events.parquet"
    if fd.exists() and fe.exists():
        daily = pd.read_parquet(fd)
        daily.columns = pd.MultiIndex.from_tuples([tuple(c.split("|")) for c in daily.columns])
        return daily, pd.read_parquet(fe)
    daily, ev = build(sym)
    flat = daily.copy()
    flat.columns = ["|".join(c) for c in daily.columns]
    flat.to_parquet(fd)
    ev.to_parquet(fe)
    return daily, ev


def cell(name: str, x: pd.DataFrame) -> dict:
    x = x[x.index >= START]
    net, net2 = x.gross - x.cost, x.gross - 2 * x.cost
    a = x.gross - x.expo.mean() * x.hold
    h = len(x) // 2
    early, late = x.index < "2015-01-01", x.index >= "2015-01-01"
    on = x.on.astype(bool)
    return dict(cell=name, days=len(x), first=str(x.index[0].date()), trades_wk=round(on.sum() / (len(x) / 5), 2),
                hit=round(100 * (x.ret[on] > 0).mean(), 1), gross_bp=round(1e4 * x.ret[on].mean(), 2),
                SR_gross=round(sharpe(x.gross), 2), SR=round(sharpe(net), 2), SR_2x=round(sharpe(net2), 2),
                h1=round(sharpe(net[:h]), 2), h2=round(sharpe(net[h:]), 2),
                y07_14=round(sharpe(net[early]), 2) if early.sum() > 250 else np.nan, y15_26=round(sharpe(net[late]), 2),
                t_a=round(float(a.mean() / a.std() * np.sqrt(len(a))), 2),
                breakeven_x=round(float(x.gross.sum() / x.cost.sum()), 2) if x.cost.sum() > 0 else np.nan)


def book_stats(name: str, net: pd.Series, net2: pd.Series, gold: pd.Series, split: str = "2015-01-01") -> dict:
    net, net2 = net.dropna(), net2.dropna()
    h = len(net) // 2
    j = pd.concat([net, gold], axis=1, join="inner").dropna()
    dd = j.iloc[:, 0] * (j.iloc[:, 1].std() / j.iloc[:, 0].std()) - j.iloc[:, 1]
    eq = (1 + net).cumprod()
    return dict(book=name, days=len(net), first=str(net.index[0].date()), SR=round(sharpe(net), 2), SR_2x=round(sharpe(net2), 2),
                h1=round(sharpe(net[:h]), 2), h2=round(sharpe(net[h:]), 2),
                early=round(sharpe(net[net.index < split]), 2), late=round(sharpe(net[net.index >= split]), 2),
                t_vs_gold=round(float(dd.mean() / dd.std() * np.sqrt(len(dd))), 2),
                ann_vol=round(100 * net.std() * np.sqrt(252), 2), maxDD=round(100 * (eq / eq.cummax() - 1).min(), 1),
                yrs_pos=f"{sum(sharpe(net[net.index.year == y]) > 0 for y in sorted(set(net.index.year)))}/{net.index.year.nunique()}")


def report() -> None:
    D, E = {}, []
    for s in PANEL:
        try:
            D[s], ev = cached(s)
            E.append(ev)
        except FileNotFoundError:
            print(f"  ({s}: no data yet)")
    E = pd.concat(E, ignore_index=True)
    E = E[E.day >= START]
    L = [f"ROUND 2 — THE PDH RULE, FROZEN, ON {len(D)} INSTRUMENTS (OANDA M5 mid, from {START}); "
         f"{len(E):,} trades", ""]
    rows = []
    for s, daily in D.items():
        for tag, nm in (("L", "long"), ("S", "short")):
            r = cell(f"{s}_{nm}", daily[tag])
            r["cls"], r["cost_bp"] = PANEL[s][0], PANEL[s][1] if PANEL[s][1] else "$0.30"
            rows.append(r)
    T = pd.DataFrame(rows)
    # family reality check on a_d, missing days = 0
    idx = pd.DatetimeIndex(sorted(set().union(*[d.index[d.index >= START] for d in D.values()])))
    cols, names = [], []
    for s, daily in D.items():
        for tag, nm in (("L", "long"), ("S", "short")):
            x = daily[tag]
            x = x[x.index >= START]
            cols.append((x.gross - x.expo.mean() * x.hold).reindex(idx).fillna(0.0).to_numpy())
            names.append(f"{s}_{nm}")
    Amat = np.column_stack(cols)
    t_obs, mx = reality_check(Amat)
    T["p_fam"] = [round(float((mx >= t).mean()), 3) for t in t_obs]
    order = {c: i for i, c in enumerate(CLASSES)}
    T = T.sort_values(["cell"], key=lambda c: c.map(lambda v: (order[PANEL[v.rsplit("_", 1)[0]][0]], v.endswith("short"), v)))
    L += ["A. EVERY LEG (SR = net at the instrument's own cost; t_a = gross vs holding at matched signed exposure)",
          T[["cell", "cls", "cost_bp", "first", "trades_wk", "hit", "gross_bp", "SR_gross", "SR", "SR_2x", "h1", "h2",
             "y07_14", "y15_26", "t_a", "breakeven_x", "p_fam"]].to_string(index=False), ""]
    best = int(np.argmax(t_obs))
    L.append(f"   FAMILY (48 cells, signed max, block bootstrap): best {names[best]} t {t_obs[best]:+.2f} vs null max median "
             f"{np.median(mx):+.2f}, 95th {np.percentile(mx, 95):+.2f}, p {(mx >= t_obs[best]).mean():.4f};  "
             f"cells with p_fam < 0.05: {int((T.p_fam < 0.05).sum())}")
    longs = T[T.cell.str.endswith("long")]
    nong = longs[~longs.cell.str.startswith("XAUUSD")]
    L.append(f"   LONG legs net-positive: {int((longs.SR > 0).sum())}/{len(longs)} (non-gold {int((nong.SR > 0).sum())}/{len(nong)}); "
             f"SHORT legs net-positive: {int((T[T.cell.str.endswith('short')].SR > 0).sum())}/{len(longs)}")
    by = T.assign(side=T.cell.str.rsplit("_", n=1).str[1]).groupby(["cls", "side"]).agg(
        n=("SR", "size"), SR_gross=("SR_gross", "mean"), SR=("SR", "mean"), t_a=("t_a", "mean")).round(2)
    L += ["   mean by class and side:", by.to_string(), ""]

    # books
    net = {(s, tag): (D[s][tag].gross - D[s][tag].cost) for s in D for tag in ("L", "S")}
    net2 = {(s, tag): (D[s][tag].gross - 2 * D[s][tag].cost) for s in D for tag in ("L", "S")}
    N, N2 = pd.DataFrame(net).reindex(idx), pd.DataFrame(net2).reindex(idx)
    gold = N[("XAUUSD", "L")]
    Lcols = [c for c in N.columns if c[1] == "L"]
    books = [book_stats("XAUUSD long alone", gold, N2[("XAUUSD", "L")], gold),
             book_stats("BOOK_LONG (24 long legs)", N[Lcols].mean(axis=1), N2[Lcols].mean(axis=1), gold),
             book_stats("BOOK_LS (48 legs)", N.mean(axis=1), N2.mean(axis=1), gold)]
    wf, wf2, picks = [], [], {}
    for y in range(2012, idx.year.max() + 1):
        past, cur = N[N.index < f"{y}-01-01"], (N.index.year == y)
        sel = [c for c in N.columns if past[c].notna().sum() >= 756 and sharpe(past[c].dropna()) > 0.30]
        picks[y] = sel
        wf.append(N.loc[cur, sel].mean(axis=1) if sel else pd.Series(0.0, index=N.index[cur]))
        wf2.append(N2.loc[cur, sel].mean(axis=1) if sel else pd.Series(0.0, index=N.index[cur]))
    wf, wf2 = pd.concat(wf).fillna(0.0), pd.concat(wf2).fillna(0.0)
    books.append(book_stats("BOOK_WF (walk-forward pick, 2012+)", wf, wf2, gold, split="2019-01-01"))
    B = pd.DataFrame(books)
    L += ["BOOKS (net; t_vs_gold = paired t against the gold long leg at matched vol; early/late split 2015, WF 2019)",
          B.to_string(index=False), "",
          "   BOOK_WF picks by year: " + "; ".join(f"{y}: {len(v)}" for y, v in picks.items()),
          "   legs picked in >= 8 of the years: " + ", ".join(
              f"{s}_{t}" for (s, t), c in pd.Series([c for v in picks.values() for c in v]).value_counts().items() if c >= 8), ""]
    corr = N[Lcols].corr()
    off = corr.values[np.triu_indices(len(Lcols), 1)]
    L.append(f"   long-leg daily correlation: mean {np.nanmean(off):+.3f}, median {np.nanmedian(off):+.3f}, max {np.nanmax(off):+.3f}")

    # B. salience
    L += ["", "B. LEVEL SALIENCE — return in ATR units by how many days the level had stood (N+1-day extreme)"]
    rng = np.random.default_rng(7)
    for side, nm in ((1, "LONG"), (-1, "SHORT")):
        x = E[E.side == side]
        rho = sps.spearmanr(x.fresh, x.R)[0]
        blk = (x.sym + x.day.dt.year.astype(str)).to_numpy()
        f, r = x.fresh.to_numpy(), x.R.to_numpy()
        groups = [np.flatnonzero(blk == b) for b in np.unique(blk)]
        null = np.empty(2000)
        for q in range(2000):
            fp = f.copy()
            for g in groups:
                fp[g] = f[g][rng.permutation(len(g))]
            null[q] = sps.spearmanr(fp, r)[0]
        h = x.day < x.day.quantile(0.5)
        bk = pd.cut(x.fresh, [-1, 0, 2, 5, 10, 99], labels=["0", "1-2", "3-5", "6-10", ">10"])
        tb = x.groupby(bk, observed=True).R.agg(["size", "mean"])
        L.append(f"   {nm}: n {len(x):,}, Spearman(N, R) {rho:+.4f}, within-instrument-year permutation p {(null >= rho).mean():.4f}; "
                 f"halves {sps.spearmanr(x.fresh[h], x.R[h])[0]:+.4f} / {sps.spearmanr(x.fresh[~h], x.R[~h])[0]:+.4f}")
        L.append("      mean R by N: " + "  ".join(f"{k}: {v['mean']:+.4f} (n {int(v['size'])})" for k, v in tb.iterrows()))
        L.append("      by class: " + "  ".join(
            f"{c} {sps.spearmanr(x.fresh[x.cls == c], x.R[x.cls == c])[0]:+.4f}" for c in CLASSES if (x.cls == c).sum() > 100))

    # D. gold in euros
    fe = ROOT / "data/oanda_hf/5/XAUEUR.parquet"
    if fe.exists() and "XAUUSD" in D and "EURUSD" in D:
        L += ["", "D. GOLD IN EUROS (gold LONG trades): is the break also a break in EUR terms; is the day's move mostly the dollar?"]
        g = E[(E.sym == "XAUUSD") & (E.side == 1)].copy()
        def dayframe(sym):
            d = load(sym)
            key = (d.index + pd.Timedelta(hours=7)).normalize()
            return d.assign(day=key)
        xe, eu, xu = dayframe("XAUEUR"), dayframe("EURUSD"), dayframe("XAUUSD")
        pdh_e = xe.groupby("day").high.max().shift(1)
        op = {k: v.groupby("day").open.first() for k, v in (("xe", xe), ("eu", eu), ("xu", xu))}
        d1, d2 = [], []
        tt = pd.DatetimeIndex(g.t_trig)
        for day, t in zip(g.day, tt):
            pe = xe.open.asof(t) if t >= xe.index[0] else np.nan      # the bar OPEN at the trigger time: known then
            pu, pq = xu.open.asof(t), eu.open.asof(t)
            lvl = pdh_e.get(day, np.nan)
            d1.append(np.nan if not np.isfinite(pe) or not np.isfinite(lvl) else float(pe >= lvl))
            mu = pu / op["xu"].get(day, np.nan) - 1
            me = pq / op["eu"].get(day, np.nan) - 1
            d2.append(np.nan if not np.isfinite(mu) or not np.isfinite(me) or mu <= 0 else float(me > 0.5 * mu))
        g["d1"], g["d2"] = d1, d2
        for col, nm in (("d1", "D1 also at/above the prior-day high in EUR"), ("d2", "D2 dollar-driven (EURUSD move > half of gold's)")):
            x = g.dropna(subset=[col])
            a, b = x.R[x[col] == 1], x.R[x[col] == 0]
            obs = a.mean() - b.mean()
            null = np.array([(lambda p: x.R[p == 1].mean() - x.R[p == 0].mean())(rng.permutation(x[col].to_numpy())) for _ in range(2000)])
            h = x.day < x.day.quantile(0.5)
            hv = [x.R[h & (x[col] == 1)].mean() - x.R[h & (x[col] == 0)].mean(), x.R[~h & (x[col] == 1)].mean() - x.R[~h & (x[col] == 0)].mean()]
            L.append(f"   {nm}: yes n {len(a)} mean {a.mean():+.4f}R | no n {len(b)} mean {b.mean():+.4f}R | diff {obs:+.4f}R, "
                     f"two-sided permutation p {(np.abs(null) >= abs(obs)).mean():.4f}; halves {hv[0]:+.4f} / {hv[1]:+.4f}")
    txt = "\n".join(L)
    print(txt)
    (OUT / "panel_report.txt").write_text(txt + "\n")
    E.to_parquet(OUT / "panel_events.parquet")
    flat = N.copy()
    flat.columns = ["|".join(c) for c in N.columns]
    flat.to_parquet(OUT / "panel_net_daily.parquet")


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "report"
    if what == "build":
        for s in (sys.argv[2:] or PANEL):
            try:
                daily, ev = cached(s)
                x = daily["L"]
                x = x[x.index >= START]
                print(f"{s:10s} days {len(x):5d}  long trades {int(x.on.sum()):5d}  events {len(ev):6d}  "
                      f"long net SR {sharpe(x.gross - x.cost):+.2f}", flush=True)
            except FileNotFoundError:
                print(f"{s}: data not downloaded yet", flush=True)
    else:
        report()


if __name__ == "__main__":
    main()
