"""STAGE 3 — the decisive decomposition: is the edge in the ENTRY or in the EXIT?

Timestamp alignment SOLVED first: the vendor log runs +2h vs our XAUUSD feed. At a -120min
shift, 85.3% of entry prices land inside the containing M15 bar with a median miss of
$0.00 (at every other offset it is 9-21%). Stage 2's "42.7% physically impossible" was
entirely that artifact. All work below applies OFFSET_MIN.

WHY THIS STAGE IS THE WHOLE BALL GAME. The log shows an 83.2% win rate WITH a 10.6:1
payoff, zero stop-outs in 773 trades, losses capped at -$5.40 against a declared 2.4-wide
stop, and a median hold of 14 SECONDS. Only two things can produce that:

  (A) the ENTRIES carry real directional information, or
  (B) the ENTRIES are ~neutral and everything comes from tick-level EXIT management
      (bank any small favourable tick, cut the moment it turns) — which is either genuine
      microstructure/latency edge or an un-replicable fill assumption.

These have opposite implications for the user's request. If (A), a bar-based replica can
work. If (B), no M15/M30 strategy can ever reproduce it, and saying so is the honest
answer rather than fitting a curve to the equity line.

THE TEST: take the ACTUAL entries — same timestamps, same directions, no signal modelling
at all — and replace the vendor's exit with NEUTRAL, mechanically reproducible exits. If
the entries hold information, they stay profitable under a neutral exit. If they are
noise, they collapse to a coin flip and the edge provably lives in the exit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OFFSET_MIN = -120          # vendor log -> our feed, established in stage 2
CONTRACT = 100.0           # oz per lot (confirmed: profit == move x lots x 100)
SPREAD_USD = 0.44          # live XAUUSD spread, measured on a real account


def load_px(tf: str) -> pd.DataFrame:
    px = pd.read_csv(ROOT / "data" / f"XAUUSD_{tf}_long.csv",
                     parse_dates=["time"], index_col="time").sort_index()
    return px[~px.index.duplicated(keep="last")]


def main() -> None:
    d = pd.read_csv(HERE / "re_parsed_trades.csv", parse_dates=["open_date", "close_date"])
    d["sgn"] = np.where(d.action.str.lower().eq("buy"), 1.0, -1.0)
    d["move"] = (d.close_price - d.open_price) * d.sgn
    d["t"] = d.open_date + pd.Timedelta(minutes=OFFSET_MIN)

    px = load_px("M15")
    idx, o, h, l, c = px.index, px.open.values, px.high.values, px.low.values, px.close.values
    pos = idx.searchsorted(d.t.values, side="right") - 1
    d["pos"] = pos
    valid = (pos >= 2) & (pos < len(idx) - 8)
    d = d[valid].reset_index(drop=True)
    pos = d.pos.values
    print(f"aligned trades usable for forward simulation: {len(d)}\n")

    # ---- sanity: entry price really is inside its bar now
    inside = (d.open_price >= l[pos] - 0.01) & (d.open_price <= h[pos] + 0.01)
    print(f"entry inside containing M15 bar: {float(inside.mean()*100):.1f}%  "
          f"(the rest are boundary/ask-vs-bid cases)\n")

    # ================================================ NEUTRAL EXIT SIMULATIONS
    print("=== THE TEST: same entries, NEUTRAL exits (no vendor exit logic) ===")
    print("entry filled at the vendor's own price; cost = one spread per round trip\n")
    cost = SPREAD_USD
    rows = []

    def add(label, gross_move):
        net = (gross_move - cost) * d.lots.values * CONTRACT
        rows.append(dict(exit=label, n=len(net), win=float((net > 0).mean() * 100),
                         mean=net.mean(), total=net.sum(),
                         sharpe_per_trade=net.mean() / net.std() if net.std() else np.nan))

    # (1) hold to the close of the bar N bars ahead (15/30/60/120 min)
    for k, lbl in ((1, "next M15 close (+15m)"), (2, "+30m"), (4, "+60m"), (8, "+120m")):
        exit_px = c[np.clip(pos + k, 0, len(c) - 1)]
        add(lbl, (exit_px - d.open_price.values) * d.sgn.values)

    # (2) the vendor's OWN declared bracket, checked intrabar over the next 8 bars
    #     (SL checked before TP within a bar = conservative)
    sl_d, tp_d = d.sl_dist.values, d.tp_dist.values
    outcome = np.zeros(len(d))
    for i in range(len(d)):
        e, s = d.open_price.values[i], d.sgn.values[i]
        sl_px, tp_px = e - s * sl_d[i], e + s * tp_d[i]
        res = None
        for k in range(0, 9):
            j = min(pos[i] + k, len(c) - 1)
            hit_sl = (l[j] <= sl_px) if s > 0 else (h[j] >= sl_px)
            hit_tp = (h[j] >= tp_px) if s > 0 else (l[j] <= tp_px)
            if hit_sl:
                res = -sl_d[i]; break
            if hit_tp:
                res = tp_d[i]; break
        outcome[i] = res if res is not None else (c[min(pos[i] + 8, len(c) - 1)] - e) * s
    add("declared bracket (SL2.4/TP10)", outcome)

    R = pd.DataFrame(rows)
    print(f"{'exit rule':32} {'n':>5} {'win%':>7} {'mean$':>9} {'total$':>11} {'SR/trade':>9}")
    for _, r in R.iterrows():
        print(f"{r['exit']:32} {int(r.n):>5} {r.win:>7.1f} {r['mean']:>9.2f} "
              f"{r.total:>11.0f} {r.sharpe_per_trade:>9.3f}")

    print(f"\nVENDOR's actual result on these same trades: win 83.2%, "
          f"mean ${d.profit.mean():.2f}, total ${d.profit.sum():,.0f}")

    # ================================================ is the direction informative at all?
    print("\n=== DIRECTIONAL INFORMATION IN THE ENTRIES ===")
    for k, lbl in ((1, "+15m"), (4, "+60m"), (8, "+120m")):
        fwd = (c[np.clip(pos + k, 0, len(c) - 1)] - d.open_price.values) * d.sgn.values
        print(f"  {lbl:>6}: mean move {fwd.mean():+.3f}  hit {float((fwd>0).mean()*100):.1f}%  "
              f"t-stat {fwd.mean()/(fwd.std()/np.sqrt(len(fwd))):+.2f}")
    print("  (a real entry edge shows hit% > 50 and a t-stat that clears ~2)")

    # ================================================ where does vendor profit come from?
    print("\n=== VENDOR EXIT vs WHAT THE BAR OFFERED ===")
    fav = np.where(d.sgn > 0, h[pos] - d.open_price.values, d.open_price.values - l[pos])
    adv = np.where(d.sgn > 0, d.open_price.values - l[pos], h[pos] - d.open_price.values)
    print(f"within the ENTRY bar alone: favourable excursion mean {fav.mean():.2f}, "
          f"adverse mean {adv.mean():.2f}")
    print(f"vendor captured mean {d.move.mean():.2f} = "
          f"{float(d.move.mean()/fav.mean()*100):.0f}% of the favourable excursion, "
          f"while eating only {float(d.move[d.move<0].abs().mean()/adv.mean()*100):.0f}% "
          f"of the adverse one on its losers")
    print(f"vendor's max loss {d.move.min():.2f} vs mean adverse excursion available "
          f"{adv.mean():.2f} -> it exits adverse moves "
          f"{float(adv.mean()/abs(d.move.min())):.1f}x earlier than the bar's own range")

    d.to_csv(HERE / "re_trades_aligned.csv", index=False)
    print(f"\naligned -> {HERE/'re_trades_aligned.csv'}")


if __name__ == "__main__":
    main()
