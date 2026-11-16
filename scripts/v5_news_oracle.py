#!/usr/bin/env python3
"""T5 (V5_FINDINGS 3ew): the human-label ORACLE for gold headlines, before any sentiment model.
10,570 headlines 2000-2021 annotated by Sinha & Khandait (HF SaguaroCapital/...-gold). The human
label is the ceiling for any model that reads the same headline, so control #5 runs first.
SAME = the trading day the headline is dated (known-answer control: 97% describe the past).
NEXT = 00:00 ET -> 17:00 ET on the next trading date, strictly after the headline's date ended.
    python scripts/v5_news_oracle.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_intraday_m1 import load_m1  # noqa: E402
OUT = ROOT / "data/v5_runs/intraday_day_signal"


def t(x):
    x = np.asarray(x, float)
    return x.mean() / x.std(ddof=1) * np.sqrt(len(x))


def main() -> None:
    n = pd.concat([pd.read_parquet(f) for f in sorted((ROOT / "data/news_gold_hf").glob("*.parquet"))])
    n["date"] = pd.to_datetime(n["Dates"], format="%d-%m-%Y", errors="coerce")
    n = n.dropna(subset=["date"])
    lab = n["Price Sentiment"].map({"positive": 1, "negative": -1}).fillna(0)
    g = pd.DataFrame(dict(pos=(lab > 0), neg=(lab < 0), fut=n["Future Information"] == 1, d=n.date)).groupby("d").sum()
    g = g[(g.pos + g.neg) > 0]
    g["s"] = (g.pos - g.neg) / (g.pos + g.neg)
    d = load_m1()
    day = (d.index + pd.Timedelta(hours=7)).normalize()
    bar = d.groupby(day).agg(open=("open", "first"), close=("close", "last"), n=("close", "size"))
    bar = bar[bar.n >= 600]
    mid = d[d.index.hour < 17].groupby(d.index[d.index.hour < 17].normalize()).open.first()      # first print at/after 00:00 ET
    same = (bar.close / bar.open - 1)
    nxt = (bar.close / mid.reindex(bar.index) - 1)
    dates = bar.index
    rows = []
    for dt, s in g.s.items():
        if dt < dates[0] or dt > dates[-2] or s == 0:
            continue
        j = dates.searchsorted(dt, side="right")                   # first trading date strictly after dt
        rows.append((dt, np.sign(s), same.get(dt, np.nan), nxt.iloc[j]))
    r = pd.DataFrame(rows, columns=["date", "sig", "same", "next"]).dropna(subset=["next"])
    rs = r.dropna(subset=["same"])
    L = [f"T5 HEADLINE ORACLE: {len(n):,} headlines, {len(g):,} dated days with a +/- label, {len(r):,} scored "
         f"{r.date.min().date()} -> {r.date.max().date()} (OANDA M1)",
         f"   SAME day (known answer): long/short by the label earns {1e4 * (rs.sig * rs.same).mean():+.1f}bp, t {t(rs.sig * rs.same):+.1f}, "
         f"hit {100 * (rs.sig * rs.same > 0).mean():.1f}%"]
    x = (r.sig * r.next).to_numpy()
    rng = np.random.default_rng(4)
    nl = np.array([t(r.sig.to_numpy() * np.roll(r.next.to_numpy(), int(k))) for k in rng.integers(20, len(r) - 20, 2000)])
    h = len(x) // 2
    L += [f"   NEXT session: {1e4 * x.mean():+.2f}bp per day, t {t(x):+.2f}, hit {100 * (x > 0).mean():.1f}% vs up-share "
          f"{100 * (r.next > 0).mean():.1f}%; halves t {t(x[:h]):+.2f} / {t(x[h:]):+.2f}; shifted-outcome null 95th {np.percentile(nl, 95):+.2f}, "
          f"p {(nl >= t(x)).mean():.3f}",
          f"   by label: after POSITIVE days next {1e4 * r.next[r.sig > 0].mean():+.2f}bp (n {int((r.sig > 0).sum())}); after NEGATIVE "
          f"{1e4 * r.next[r.sig < 0].mean():+.2f}bp (n {int((r.sig < 0).sum())}); all days {1e4 * nxt.mean():+.2f}bp"]
    txt = "\n".join(L)
    print(txt)
    (OUT / "news_oracle_report.txt").write_text(txt + "\n")


if __name__ == "__main__":
    sys.exit(main())
