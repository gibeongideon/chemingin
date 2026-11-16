#!/usr/bin/env python3
"""THE FILL AUDIT for P1 (V5_FINDINGS 3ex, part 3). scripts/v5_pdh_m1_stress.py showed the
prior-day-high stop nets +0.76 when filled AT the level and +0.42 when entered one minute late:
the first M1 close above the level is already +4.84bp beyond it. 3ci met the same gap on H1
(4.73bp) and the live fill probe (magic 360599) that was meant to settle it was cancelled from
outside the terminal 3 times out of 3 (3cl).

OANDA publishes 5-SECOND bars for the same instrument. Inside each trade's trigger minute this
finds the first 5S bar whose high reaches the level and re-prices the SAME 2,559 entries at:
  level     max(level, that 5S bar's open)      a resting stop, gap-aware at 5 seconds
  5s        that 5S bar's close                 a market order at most 5 seconds late
  10s/30s   the close 1 / 5 bars later          slower bots
and reports the overshoot distribution. Exit and event set are unchanged (control 9).

Also here, exploratory: the day boundary (17:00 ET = the MT5 server day, pre-registered; 00:00
UTC as 3ch's file was stamped) - a sensitivity, not a choice.

    python scripts/v5_pdh_fill_audit.py
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
from scripts.v5_intraday_m1 import COST, NY, OUT, START, sharpe, vol_weight  # noqa: E402
from scripts.v5_pdh_m1_stress import run, stats, to_days  # noqa: E402

S5 = ROOT / "data/oanda_hf/5S"


def main() -> None:
    raw = pd.read_parquet(ROOT / "data/oanda_hf/XAUUSD_M1.parquet")
    utc = pd.DatetimeIndex(pd.to_datetime(raw.time, unit="s"))
    d = raw[["open", "high", "low", "close"]].copy()
    d.index = utc.tz_localize("UTC").tz_convert(NY).tz_localize(None)
    tsec = raw.time.to_numpy()
    A, mod, start, end, bar = to_days(d, 600)
    hold = (bar.close / bar.open - 1).to_numpy()
    w, px, yrs = vol_weight(hold), bar.close.to_numpy(), bar.index.year.to_numpy()
    k = bar.index >= START
    base = run(A, mod, start, end, bar)
    ret, on, frac, epx, tin = base
    lv = bar.high.shift(1).to_numpy()
    last_px = A["close"][end - 1]

    files = sorted(S5.glob("*.parquet"))
    span = [(pd.Timestamp(f.stem.split("_")[-3]), pd.Timestamp(f.stem.split("_")[-1]) + pd.Timedelta(days=1), f) for f in files]
    cache: dict = {}

    def week(ts: pd.Timestamp):
        for a, b, f in span:
            if a <= ts < b:
                if f not in cache:
                    cache.clear()
                    x = pd.read_parquet(f)
                    cache[f] = (x.time.to_numpy(), x.open.to_numpy(), x.high.to_numpy(), x.close.to_numpy())
                return cache[f]
        return None

    idx = np.flatnonzero(on & k)
    rows = []
    for i in idx:
        t0 = int(tsec[start[i] + int(tin[i])])                  # epoch second of the trigger minute's open
        wk = week(pd.Timestamp(t0, unit="s"))
        rec = dict(i=i, level=lv[i], m1=epx[i], lvl5=np.nan, c5=np.nan, c10=np.nan, c30=np.nan, gap=np.nan, n5=0)
        if wk is not None:
            T, O5, H5, C5 = wk
            a, b = np.searchsorted(T, t0), np.searchsorted(T, t0 + 60)
            rec["n5"] = b - a
            hit = np.flatnonzero(H5[a:b] >= lv[i] - 1e-9)
            if len(hit):
                j = a + hit[0]
                rec.update(lvl5=max(lv[i], O5[j]), c5=C5[j], gap=max(0.0, O5[j] - lv[i]),
                           c10=C5[min(j + 1, len(C5) - 1)], c30=C5[min(j + 5, len(C5) - 1)])
        rows.append(rec)
    R = pd.DataFrame(rows).set_index("i")
    ok = R.lvl5.notna()
    L = [f"P1 FILL AUDIT on OANDA 5-second bars, {int(len(R))} trades {bar.index[k][0].date()} -> {bar.index[k][-1].date()}",
         f"trades whose trigger minute has a 5S bar reaching the level: {int(ok.sum())} ({100 * ok.mean():.1f}%); "
         f"the rest keep the M1 fill. median 5S bars in the trigger minute: {int(R.n5.median())}", ""]

    def reprice(col: str, extra_bp: float = 0.0) -> dict:
        e = np.where(np.isfinite(R[col]), R[col], R.m1).astype(float) * (1 + extra_bp * 1e-4)
        r2 = ret.copy()
        r2[R.index] = last_px[R.index] / e - 1
        s = stats(col, r2[k], on[k], frac[k], w[k], hold[k], px[k], yrs[k])
        return s

    over = lambda c: 1e4 * (R[c][ok] / R.level[ok] - 1)  # noqa: E731
    L.append("A. WHERE THE FILL LANDS, in bp above the level (same trades):")
    for c, nm in (("lvl5", "resting stop, gap-aware at 5s"), ("c5", "market, <= 5 s late"), ("c10", "market, ~10 s late"),
                  ("c30", "market, ~30 s late")):
        o = over(c)
        L.append(f"   {nm:30s} mean {o.mean():+6.2f}bp  median {o.median():+6.2f}  75th {o.quantile(.75):+6.2f}  "
                 f"95th {o.quantile(.95):+7.2f}  99th {o.quantile(.99):+7.2f}")
    m1c = 1e4 * (pd.Series(A["close"][start[R.index] + tin[R.index].astype(int)], index=R.index)[ok] / R.level[ok] - 1)
    L.append(f"   {'close of the trigger MINUTE':30s} mean {m1c.mean():+6.2f}bp  median {m1c.median():+6.2f}  75th {m1c.quantile(.75):+6.2f}  "
             f"95th {m1c.quantile(.95):+7.2f}")
    g = 1e4 * R.gap[ok] / R.level[ok]
    L.append(f"   the 5S bar OPENED above the level (a true gap) in {100 * (g > 0).mean():.1f}% of trades; mean gap when it did {g[g > 0].mean():.2f}bp")
    yr = pd.Series(bar.index.year[R.index[ok]], index=R.index[ok])
    L.append("   <= 5 s overshoot by era, mean / median bp: " + "  ".join(
        f"{a}-{b}: {over('c5')[(yr >= a) & (yr <= b)].mean():+.2f} / {over('c5')[(yr >= a) & (yr <= b)].median():+.2f}"
        for a, b in ((2007, 2012), (2013, 2018), (2019, 2023), (2024, 2026))))
    L += ["", "B. THE SAME TRADES RE-PRICED (cost $0.30 round trip on top)"]
    T = pd.DataFrame([reprice("m1"), reprice("lvl5"), reprice("c5"), reprice("c10"), reprice("c30"),
                      reprice("lvl5", 1.0), reprice("lvl5", 2.0), reprice("lvl5", 3.0), reprice("lvl5", 5.0)])
    T.insert(0, "entry", ["M1 fill at the level (pre-registered)", "5S fill at the level, gap-aware", "market <= 5 s late",
                          "market ~10 s late", "market ~30 s late", "5S level + 1bp slip", "5S level + 2bp slip",
                          "5S level + 3bp slip", "5S level + 5bp slip"])
    L += [T.drop(columns=["variant", "trades", "per_wk", "expo"]).to_string(index=False), ""]

    # by time of day and by speed of the break: where is a level fill least believable?
    hrs = (tin[R.index] / 60)
    fast = over("c5").reindex(R.index)
    r_tr = pd.Series(1e4 * ret[R.index], index=R.index)
    L.append("C. WHICH TRADES CARRY IT (gross bp per trade, M1 level fill): " + "  ".join(
        f"overshoot@5s {nm}: n {int(m.sum())} {r_tr[m].mean():+.1f}bp" for nm, m in (
            ("<= 0.5bp", fast <= 0.5), ("0.5-2bp", (fast > 0.5) & (fast <= 2)), ("2-5bp", (fast > 2) & (fast <= 5)), ("> 5bp", fast > 5))))
    L.append("   by entry hour after the 17:00 ET open: " + "  ".join(
        f"{a}-{b}h n {int(((hrs >= a) & (hrs < b)).sum())} {r_tr[(hrs >= a) & (hrs < b)].mean():+.1f}bp" for a, b in ((0, 3), (3, 9), (9, 15), (15, 24))))

    # D. day boundary, exploratory
    d2 = d.copy()
    d2.index = utc - pd.Timedelta(hours=7)                       # +7h inside to_days -> the UTC calendar day
    A2, mod2, st2, en2, bar2 = to_days(d2, 600)
    h2 = (bar2.close / bar2.open - 1).to_numpy()
    k2 = bar2.index >= START
    b2 = run(A2, mod2, st2, en2, bar2)
    s2 = stats("utc", b2[0][k2], b2[1][k2], b2[2][k2], vol_weight(h2)[k2], h2[k2], bar2.close.to_numpy()[k2], bar2.index.year.to_numpy()[k2])
    s1 = stats("et17", ret[k], on[k], frac[k], w[k], hold[k], px[k], yrs[k])
    L += ["", "D. DAY BOUNDARY (exploratory sensitivity; 17:00 ET = the MT5 server day is the pre-registered one)"]
    for nm, s in (("17:00 ET day (pre-registered)", s1), ("00:00 UTC day", s2)):
        L.append(f"   {nm:30s} trades {s['trades']}  gross {s['gross_bp']:+.2f}bp  SR gross {s['SR_gross']:+.2f} net {s['SR']:+.2f} "
                 f"2x {s['SR_2x']:+.2f}  halves {s['h1']:+.2f}/{s['h2']:+.2f}  t vs hold {s['t_vs_hold']:+.2f}")
    txt = "\n".join(L)
    print(txt)
    (OUT / "pdh_fill_audit_report.txt").write_text(txt + "\n")
    R.to_parquet(OUT / "pdh_fill_audit.parquet")


if __name__ == "__main__":
    sys.exit(main())
