"""STAGE 2 — exit mechanism + PHYSICAL CONSISTENCY against our own XAUUSD feed.

Stage 1 established the log is genuine (balance reconciles to the cent) and that the
trade profile is extraordinary: 83.2% win rate WITH a 10.6:1 payoff, zero SL hits in 773
trades, largest loss -$5.40 against a 2.4-wide stop, median hold 14 SECONDS, 97.4% exiting
at neither TP nor SL, fixed 0.10 lots, no overlap.

Before attempting any replication we have to answer two questions with data:

  Q1  WHAT CLOSES THE TRADES? Not TP, not SL. Is the exit rule inferable?
  Q2  ARE THE FILLS PHYSICALLY POSSIBLE? Our M15/M30 XAUUSD history covers the entire
      test window (2025-10-15 .. 2026-04-14) at matching price levels, so for every trade
      we can ask whether the claimed move was actually available in the market within the
      bar it happened in. A claimed profit larger than the containing bar's entire
      high-low range is not a trade, it is an artifact.

Also: entry timestamps (16:34, 16:21, 13:38, 09:52 ...) are NOT aligned to M30/M15 bar
boundaries, which already tells us entries are tick-triggered rather than bar-close
decisions. That is tested here rather than assumed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main() -> None:
    d = pd.read_csv(HERE / "re_parsed_trades.csv", parse_dates=["open_date", "close_date"])
    sgn = np.where(d.action.str.lower().eq("buy"), 1.0, -1.0)
    d["sgn"] = sgn
    d["move"] = (d.close_price - d.open_price) * sgn

    # ---------------------------------------------------------------- Q1 exit mechanism
    print("=== Q1: WHAT CLOSES THE TRADES? ===")
    w, l = d[d.profit > 0], d[d.profit < 0]
    print(f"WINS   n={len(w)}  move: mean {w.move.mean():.3f}  median {w.move.median():.3f}  "
          f"min {w.move.min():.3f}  max {w.move.max():.3f}")
    print(f"LOSSES n={len(l)}  move: mean {l.move.mean():.3f}  median {l.move.median():.3f}  "
          f"min {l.move.min():.3f}  max {l.move.max():.3f}")
    print(f"\nloss moves are capped at {l.move.min():.2f} while the STOP sits "
          f"{d.sl_dist.mean():.2f} away -> the stop is never the binding exit")
    print(f"loss move quantiles: {l.move.quantile([0,.1,.5,.9,1]).round(3).to_dict()}")

    # is there a fixed dollar/point target? look for clustering
    print("\nclustering of the realised move (is there a hidden fixed target?):")
    for lo, hi in [(0, .2), (.2, .5), (.5, 1), (1, 2), (2, 4), (4, 8), (8, 11)]:
        m = (d.move >= lo) & (d.move < hi)
        print(f"  move [{lo:>4},{hi:>4}): n={int(m.sum()):>4} ({m.mean()*100:>5.1f}%)  "
              f"mean dur {d.loc[m,'dur_s'].mean():>5.1f}s")
    print(f"\ncorr(duration, |move|) = {float(d.dur_s.corr(d.move.abs())):+.3f}"
          "   (a time-based exit would show a strong relationship)")

    # ---------------------------------------------------------------- Q2 physical check
    print("\n=== Q2: ARE THE FILLS PHYSICALLY POSSIBLE? (vs our own M15 feed) ===")
    px = pd.read_csv(ROOT / "data" / "XAUUSD_M15_long.csv",
                     parse_dates=["time"], index_col="time").sort_index()
    px = px[~px.index.duplicated(keep="last")]

    # containing M15 bar for each entry, plus the following bar
    idx = px.index
    pos = idx.searchsorted(d.open_date.values, side="right") - 1
    ok = (pos >= 0) & (pos < len(idx) - 1)
    print(f"trades mappable to a bar in our feed: {int(ok.sum())}/{len(d)}")

    bar_lo = np.where(ok, px["low"].values[np.clip(pos, 0, len(idx) - 1)], np.nan)
    bar_hi = np.where(ok, px["high"].values[np.clip(pos, 0, len(idx) - 1)], np.nan)
    nxt_lo = np.where(ok, px["low"].values[np.clip(pos + 1, 0, len(idx) - 1)], np.nan)
    nxt_hi = np.where(ok, px["high"].values[np.clip(pos + 1, 0, len(idx) - 1)], np.nan)
    d["bar_lo"], d["bar_hi"] = bar_lo, bar_hi
    d["win_lo"] = np.minimum(bar_lo, nxt_lo)      # 2-bar window covers boundary trades
    d["win_hi"] = np.maximum(bar_hi, nxt_hi)
    d["bar_range"] = d.bar_hi - d.bar_lo
    d["win_range"] = d.win_hi - d.win_lo

    inside = (d.open_price >= d.win_lo - 0.01) & (d.open_price <= d.win_hi + 0.01)
    print(f"entry price inside the 2-bar window range: {float(inside.mean()*100):.1f}%")
    med_off = ((d.open_price - d.win_lo) / d.win_range).median()
    print(f"median position of entry within that range: {med_off:.2f} (0=low, 1=high)")

    # was the claimed favourable move actually available?
    avail = np.where(d.sgn > 0, d.win_hi - d.open_price, d.open_price - d.win_lo)
    d["avail"] = avail
    impossible = d.move > d.avail + 0.02
    print(f"\nclaimed move EXCEEDS what the 2-bar window allowed: "
          f"{int(impossible.sum())}/{len(d)} ({impossible.mean()*100:.1f}%)")
    print(f"  of the impossible ones, median claimed {d.loc[impossible,'move'].median():.2f} "
          f"vs median available {d.loc[impossible,'avail'].median():.2f}")
    print(f"claimed move as a fraction of available favourable range: "
          f"median {float((d.move/d.avail.replace(0,np.nan)).median()):.2f}")
    print(f"\nbar_range (M15) mean {d.bar_range.mean():.2f}  vs mean |move| "
          f"{d.move.abs().mean():.2f}  -> moves are "
          f"{float(d.move.abs().mean()/d.bar_range.mean()*100):.0f}% of a 15-min bar")

    # ---------------------------------------------------------------- entry timing
    print("\n=== ENTRY TIMING: bar-close decision, or tick-triggered? ===")
    for tf, mod in (("M15", 15), ("M30", 30), ("H1", 60)):
        mm = d.open_date.dt.minute % mod
        first2 = float((mm < 2).mean() * 100)
        print(f"  entries in the first 2 min of a {tf} bar: {first2:.1f}% "
              f"(uniform would be {2/mod*100:.1f}%)")
    print(f"  distinct entry minutes-of-hour used: {d.open_date.dt.minute.nunique()}/60")

    # ---------------------------------------------------------------- entry context
    print("\n=== WHAT DOES PRICE DO BEFORE ENTRY? (spike-following vs mean-reversion) ===")
    c = px["close"]
    cpos = np.clip(pos, 2, len(idx) - 1)
    prev1 = c.values[cpos - 1]
    prev2 = c.values[cpos - 2]
    d["mom_1bar"] = (prev1 - prev2) * d.sgn          # prior 15-min move, in trade direction
    d["entry_vs_prev"] = (d.open_price - prev1) * d.sgn
    print(f"prior-bar momentum in the TRADE's direction: mean {d.mom_1bar.mean():+.3f}  "
          f"share positive {float((d.mom_1bar>0).mean()*100):.1f}%")
    print(f"entry price vs previous close, in trade direction: mean "
          f"{d.entry_vs_prev.mean():+.3f}  share positive "
          f"{float((d.entry_vs_prev>0).mean()*100):.1f}%")
    print("  (>50% positive on both = momentum/breakout entry; <50% = dip/fade entry)")

    d.to_csv(HERE / "re_trades_enriched.csv", index=False)
    print(f"\nenriched -> {HERE/'re_trades_enriched.csv'}")


if __name__ == "__main__":
    main()
