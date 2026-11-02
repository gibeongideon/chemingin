#!/usr/bin/env python3
"""If the market behaves a certain way BEFORE a session hand-over, does the next session pay?

THE ASK. "If the market behaves in a certain way just before the transition you can act in a
particular way." That is a conditional trade: read the state of the session about to end, take a
side in the session about to begin.

*** THIS IS A TRIGGER SEARCH, AND TRIGGER SEARCHES ARE HOW THIS REPO GETS FOOLED. ***
§3bi swept 21,300 triggers and the best one was indistinguishable from the best on a dataset
built to contain NO edge. The lesson was not "search less", it was "searching harder raises the
bar faster than it raises the best result". So the only honest statistic here is the MAX over the
whole grid, compared against the distribution of the MAX under a null — never a single cell's
t against zero.

THE NULL. Circular shift of the OUTCOME by at least 20 trading days, then **the entire search is
re-run** on the shifted data and its maximum recorded. That preserves every autocorrelation and
every conditioner's own structure while destroying only the alignment between trigger and
outcome. It is the design from `v5_daily_certainty_scan.py::maxstat_null`, the strongest null in
this repo.

AND THE NULL IS BUILT ON GROSS, NOT NET (§3bx-D). Subtracting a cost from a zero-mean null shifts
the whole distribution negative and manufactures significance for anything above it. Costs are
applied to the REPORTED number, never inside the null.

THE SIDE IS PART OF THE SEARCH. Letting each cell pick long or short after seeing the data is
worth roughly a factor of two in the maximum, so the null picks the side too. A one-sided search
scored against a two-sided null is the most common way to fake a result here.

PRE-REGISTERED GRID, fixed before the first run:
  transitions  ASIA->LONDON, LONDON->OVERLAP, OVERLAP->NY, ASIA->OVERLAP, LONDON->NY
  conditioners ret sign · range tercile · close-in-range tercile · last-hour momentum tercile
               · volume-vs-20d tercile · range-vs-20d tercile
  action       long or short for the whole next block
K is printed with the result.

    python scripts/v5_session_conditional.py --port 18815 --symbol XAUUSDs
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_session_transitions import ORDER, SERVER_UTC_OFFSET, fetch, label  # noqa: E402

OUT = ROOT / "data" / "v5_runs"
PAIRS = [("ASIA", "LONDON"), ("LONDON", "OVERLAP"), ("OVERLAP", "NY"),
         ("ASIA", "OVERLAP"), ("LONDON", "NY")]
MIN_N = 80                      # a bucket below this is not evidence, it is an anecdote
SHIFT_MIN = 20                  # trading days


def session_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per (day, block): return, range, close-in-range, last-hour move, volume — all computed
    from bars INSIDE that block, so every conditioner is known the instant the block ends."""
    b, day = label(df.index)
    g = df.groupby([day, b])
    t = pd.DataFrame(dict(open=g["open"].first(), high=g["high"].max(), low=g["low"].min(),
                          close=g["close"].last(), vol=g["tick_volume"].sum(),
                          bars=g["close"].size()))
    # last hour of the block = its final 4 M15 bars
    rank = df.groupby([day, b]).cumcount(ascending=False)
    lastfour = df[rank < 4].groupby([day[rank < 4], b[rank < 4]])
    t["lh_open"] = lastfour["open"].first()
    t["lh_close"] = lastfour["close"].last()
    t = t[t["bars"] >= 4].reset_index()
    t.columns = ["day", "block"] + list(t.columns[2:])
    t["ret"] = (t["close"] / t["open"] - 1.0) * 1e4
    t["rng"] = (t["high"] - t["low"]) / t["open"] * 1e4
    span = (t["high"] - t["low"]).replace(0, np.nan)
    t["cir"] = (t["close"] - t["low"]) / span            # close-in-range, 0=low 1=high
    t["lh"] = (t["lh_close"] / t["lh_open"] - 1.0) * 1e4
    return t


def conditioners(s: pd.DataFrame) -> dict:
    """{name: integer bucket per day}. Every value is derived from the PRIOR block only."""
    out = {}
    out["ret_sign"] = (s["ret"] > 0).astype(int).to_numpy()
    for col, name in (("rng", "range"), ("cir", "close_in_range"), ("lh", "last_hour")):
        v = s[col].to_numpy(float)
        out[name] = pd.qcut(pd.Series(v), 3, labels=False, duplicates="drop").to_numpy()
    # *** NO VOLUME CONDITIONER. *** Switch's XAUUSDs feed reports tick_volume == 0 on 60.2% of
    # M15 bars, uniformly across every UTC hour, and 48,128 of those 48,133 zero-volume bars have
    # a NON-ZERO high-low range — so price moved and the volume simply was not reported. A
    # "volume tercile" on that column covers 348 of 873 days and its buckets hold 116 rows while
    # a real tercile holds 290. The first run of this study picked exactly such a cell as its
    # best result. Volume from this feed is unusable; do not reintroduce it.
    for col, name in (("rng", "range_vs20"),):
        v = s[col].astype(float)
        rel = (v / v.rolling(20, min_periods=10).mean().shift(1)).to_numpy()
        out[name] = pd.qcut(pd.Series(rel), 3, labels=False, duplicates="drop").to_numpy()
    return out


def search(conds: dict, nxt: np.ndarray, base: float) -> tuple[float, dict, int]:
    """Best |signed edge vs base| over every (conditioner, bucket, side). GROSS."""
    best, where, k = -np.inf, None, 0
    for cname, cv in conds.items():
        for bucket in np.unique(cv[~pd.isna(cv)]):
            m = (cv == bucket) & np.isfinite(nxt)
            n = int(m.sum())
            if n < MIN_N:
                continue
            e = float(nxt[m].mean()) - base
            for side in (1, -1):
                k += 1
                v = side * e if side == 1 else -float(nxt[m].mean()) - base
                # side=-1 means SHORT the next block: its excess over holding is -mean - base
                if v > best:
                    best, where = v, dict(cond=cname, bucket=int(bucket), side=side,
                                          n=n, mean=float(nxt[m].mean()), edge=v)
    return best, where, k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18815)
    ap.add_argument("--symbol", default="XAUUSDs")
    ap.add_argument("--bars", type=int, default=80_000)
    ap.add_argument("--rt-bp", type=float, default=1.08)
    ap.add_argument("--draws", type=int, default=300)
    a = ap.parse_args()

    from mt5linux import MetaTrader5
    m, last = None, None
    for i in range(3):
        if i:
            time.sleep(3 * i)
        m = MetaTrader5(host="localhost", port=a.port)
        if m.initialize():
            break
        last = m.last_error()
    else:
        print("bridge init failed:", last); return 2
    m.symbol_select(a.symbol, True)
    df = fetch(m, a.symbol, a.bars)
    m.shutdown()
    if df.empty:
        print("no bars"); return 1
    print(f"{len(df):,} M15 bars  {df.index[0]} .. {df.index[-1]}  (server UTC+3)\n")

    T = session_table(df)
    rng = np.random.default_rng(0)
    rows, grand_k = [], 0
    print(f"{'transition':<18}{'best cell':<34}{'n':>6}{'gross':>9}{'net':>8}"
          f"{'null p':>9}{'null 95%':>10}{'halves':>17}")

    for a_, b_ in PAIRS:
        pa = T[T.block == a_].set_index("day")
        pb = T[T.block == b_].set_index("day")
        days = pa.index.intersection(pb.index).sort_values()
        if len(days) < 300:
            continue
        pa, pb = pa.loc[days], pb.loc[days]
        conds = conditioners(pa)
        nxt = pb["ret"].to_numpy(float)
        base = float(np.nanmean(nxt))

        # COVERAGE CHECK. A "tercile" that holds 116 of 870 days is not a tercile — it means the
        # conditioner is mostly NaN and the cell is a small, unrepresentative slice wearing the
        # name of a third of the sample. Printed rather than assumed, after a block-width bug in
        # §3cv was caught by exactly this kind of count mismatch.
        cov = {c: int(np.sum(~pd.isna(v))) for c, v in conds.items()}
        thin = {c: n for c, n in cov.items() if n < 0.8 * len(nxt)}
        if thin:
            print(f"   [{a_}->{b_}] thin conditioners (valid days of {len(nxt)}): "
                  + ", ".join(f"{c} {n}" for c, n in thin.items()))

        obs, where, k = search(conds, nxt, base)
        grand_k += k

        # ---- max-stat null: shift the OUTCOME, re-run the WHOLE search, keep its max --------
        null = np.empty(a.draws)
        n = len(nxt)
        for d in range(a.draws):
            sh = rng.integers(SHIFT_MIN, n - SHIFT_MIN)
            null[d] = search(conds, np.roll(nxt, sh), base)[0]
        p = float((null >= obs).mean())
        q95 = float(np.quantile(null, 0.95))

        # halves of the winning cell, on the real data
        cv = conds[where["cond"]]
        msk = (cv == where["bucket"]) & np.isfinite(nxt)
        # excess(LONG) = mean - base and excess(SHORT) = -mean - base (the §3bh identity), so
        # with v = side*ret BOTH sides are mean(v) - base. Subtracting side*base instead would
        # flatter every short cell by 2*base, which in a bull sample is the whole result.
        v = where["side"] * nxt[msk]
        h = len(v) // 2
        h1, h2 = float(v[:h].mean()) - base, float(v[h:].mean()) - base

        tag = (f"{where['cond']}={where['bucket']} "
               f"{'LONG' if where['side'] == 1 else 'SHORT'}")
        rows.append(dict(pair=f"{a_}->{b_}", cell=tag, n=where["n"], gross=obs,
                         net=obs - a.rt_bp, p=p, q95=q95, h1=h1, h2=h2, k=k))
        print(f"{a_+'->'+b_:<18}{tag:<34}{where['n']:>6}{obs:>9.2f}{obs - a.rt_bp:>8.2f}"
              f"{p:>9.3f}{q95:>10.2f}   {h1:>6.2f} /{h2:>6.2f}")

    print(f"\nK = {grand_k} cells searched across {len(rows)} transitions, "
          f"{a.draws} null draws each.")
    print("'null 95%' is the 95th percentile of the BEST CELL under a shifted outcome — the bar")
    print("a real trigger has to clear. 'null p' is how often the null's best beat ours.")
    if rows:
        win = [r for r in rows if r["p"] < 0.05 and r["net"] > 0 and r["h1"] > 0 and r["h2"] > 0]
        print(f"\ncells passing p<0.05 AND net>0 AND both halves positive: {len(win)}")
        for r in win:
            print(f"   {r['pair']}  {r['cell']}  net {r['net']:+.2f}bp  p {r['p']:.3f}")
        if not win:
            b = min(rows, key=lambda r: r["p"])
            print(f"   none. Closest: {b['pair']} {b['cell']} at p {b['p']:.3f}, "
                  f"net {b['net']:+.2f}bp, halves {b['h1']:+.2f}/{b['h2']:+.2f}")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "session_conditional.csv", index=False)
    print(f"\nwrote {OUT / 'session_conditional.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
