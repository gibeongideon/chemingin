"""STAGE 1 — pure forensics on the Happy Gold - OxS trade log. NO hypotheses yet.

773 XAUUSD trades, 2025-10-15 .. 2026-04-14, from a vendor forward test reporting
+507.88% on a $3,000 deposit (balance $18,251.80), monthly 34.87%, "Drawdown 0.21%"
on the Stats panel while the Forward Test header row says 33%. That contradiction is
itself a measurement to explain, not a typo to ignore.

This stage only DESCRIBES. Every inference in stage 2 has to be forced by numbers here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "happyforex_311_trades_773.csv"


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV, thousands=",")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    for c in ("open_date", "close_date"):
        df[c] = pd.to_datetime(df[c], format="%m.%d.%Y %H:%M")
    for c in ("lots", "sl", "tp", "open_price", "close_price", "pips", "profit"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["change_pct"] = pd.to_numeric(df["change"].astype(str).str.rstrip("%"), errors="coerce")
    # Duration strings like "29s", "1m 5s", "2h 3m"
    def dur_s(x: str) -> float:
        x = str(x).strip()
        tot, num = 0.0, ""
        for ch in x:
            if ch.isdigit():
                num += ch
            elif ch in "dhms" and num:
                tot += int(num) * {"d": 86400, "h": 3600, "m": 60, "s": 1}[ch]
                num = ""
        return tot
    df["dur_s"] = df["duration"].map(dur_s)
    return df.sort_values("open_date").reset_index(drop=True)


def main() -> None:
    d = load()
    print(f"=== SHAPE ===\n{len(d)} trades  {d.open_date.min()} .. {d.open_date.max()}")
    print(f"symbols: {d.symbol.unique().tolist()}   actions: {d.action.value_counts().to_dict()}")
    span_days = (d.open_date.max() - d.open_date.min()).days
    print(f"span {span_days} days -> {len(d)/span_days:.2f} trades/calendar-day")

    print("\n=== P&L ===")
    print(f"total profit ${d.profit.sum():,.2f}   mean ${d.profit.mean():.2f}   "
          f"median ${d.profit.median():.2f}")
    print(f"win rate {float((d.profit>0).mean()*100):.1f}%   "
          f"wins ${d.loc[d.profit>0,'profit'].mean():.2f}  "
          f"losses ${d.loc[d.profit<0,'profit'].mean():.2f}")
    print(f"largest win ${d.profit.max():.2f}   largest loss ${d.profit.min():.2f}")
    print(f"profit quantiles: {d.profit.quantile([.01,.05,.25,.5,.75,.95,.99]).round(2).to_dict()}")
    print(f"payoff ratio {abs(d.loc[d.profit>0,'profit'].mean()/d.loc[d.profit<0,'profit'].mean()):.2f}")

    print("\n=== LOT SIZING (martingale/grid check) ===")
    print(f"lots value_counts:\n{d.lots.value_counts().to_string()}")
    print(f"lots after a LOSS  : mean {d.lots[d.profit.shift(1)<0].mean():.4f}")
    print(f"lots after a WIN   : mean {d.lots[d.profit.shift(1)>0].mean():.4f}")

    print("\n=== DURATION ===")
    print(f"dur_s quantiles: {d.dur_s.quantile([0,.25,.5,.75,.9,.99,1]).to_dict()}")
    print(f"trades closing <60s: {float((d.dur_s<60).mean()*100):.1f}%   "
          f"<10s: {float((d.dur_s<10).mean()*100):.1f}%")

    print("\n=== SL / TP GEOMETRY ===")
    sgn = np.where(d.action.str.lower().eq("buy"), 1.0, -1.0)
    d["sl_dist"] = (d.open_price - d.sl) * sgn
    d["tp_dist"] = (d.tp - d.open_price) * sgn
    for c in ("sl_dist", "tp_dist"):
        print(f"{c}: mean {d[c].mean():.3f}  std {d[c].std():.3f}  "
              f"min {d[c].min():.3f}  max {d[c].max():.3f}  "
              f"q05 {d[c].quantile(.05):.3f} q95 {d[c].quantile(.95):.3f}")
    print(f"implied R:R (tp/sl) mean {(d.tp_dist/d.sl_dist).mean():.2f}")

    print("\n=== HOW DO TRADES ACTUALLY EXIT? (TP/SL hit, or something else?) ===")
    tol = 0.05
    hit_tp = ((d.close_price - d.tp).abs() < tol)
    hit_sl = ((d.close_price - d.sl).abs() < tol)
    print(f"closed AT TP: {int(hit_tp.sum())} ({hit_tp.mean()*100:.1f}%)   "
          f"closed AT SL: {int(hit_sl.sum())} ({hit_sl.mean()*100:.1f}%)   "
          f"NEITHER: {int((~hit_tp & ~hit_sl).sum())} ({(~hit_tp & ~hit_sl).mean()*100:.1f}%)")
    d["move"] = (d.close_price - d.open_price) * sgn
    print(f"realised move: mean {d.move.mean():.3f}  vs tp_dist {d.tp_dist.mean():.3f}  "
          f"sl_dist {d.sl_dist.mean():.3f}")
    print(f"move as fraction of tp_dist: median {float((d.move/d.tp_dist).median()):.3f}")

    print("\n=== SHARED SL/TP LEVELS (basket / same-signal check) ===")
    grp = d.groupby([d.sl.round(2), d.tp.round(2)])
    sizes = grp.size()
    print(f"distinct (SL,TP) level pairs: {len(sizes)} for {len(d)} trades")
    print(f"pairs used by >1 trade: {int((sizes>1).sum())}   max trades on one pair: {sizes.max()}")
    print(f"distribution of trades-per-level-pair:\n{sizes.value_counts().sort_index().to_string()}")

    print("\n=== CONCURRENCY / CLUSTERING ===")
    gap = d.open_date.diff().dt.total_seconds()
    print(f"inter-arrival seconds: median {gap.median():.0f}  "
          f"q10 {gap.quantile(.1):.0f}  q90 {gap.quantile(.9):.0f}")
    print(f"entries within 5min of the previous: {float((gap<300).mean()*100):.1f}%")
    # overlap: does a trade open before the previous closed?
    overl = (d.open_date < d.close_date.shift(1)).mean()
    print(f"opens before previous close (true overlap): {overl*100:.1f}%")

    print("\n=== TIME-OF-DAY / DAY-OF-WEEK ===")
    tod = d.groupby(d.open_date.dt.hour).agg(n=("profit", "size"), pnl=("profit", "sum"))
    print("hour  n   pnl$")
    for h, r in tod.iterrows():
        print(f"{h:>4} {int(r.n):>3}  {r.pnl:>8.2f}")
    dow = d.groupby(d.open_date.dt.dayofweek).agg(n=("profit", "size"), pnl=("profit", "sum"))
    print(f"\nday-of-week (0=Mon): {dow.to_dict()}")

    print("\n=== EQUITY / DRAWDOWN on a $3,000 start (balance-basis) ===")
    eq = 3000 + d.profit.cumsum()
    dd = (eq / eq.cummax() - 1) * 100
    print(f"final balance ${eq.iloc[-1]:,.2f}  (vendor says $18,251.80)")
    print(f"max BALANCE drawdown {dd.min():.2f}%  (vendor Stats panel says 0.21%)")
    print(f"longest losing streak {int((d.profit<0).astype(int).groupby((d.profit>0).cumsum()).sum().max())}")
    d.to_csv(HERE / "re_parsed_trades.csv", index=False)
    print(f"\nparsed -> {HERE/'re_parsed_trades.csv'}")


if __name__ == "__main__":
    main()
