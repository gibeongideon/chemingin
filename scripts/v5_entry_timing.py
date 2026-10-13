"""Does WAITING FOR A BOTTOM improve the book's re-entry, or does it select losers?

THE QUESTION, STATED PRECISELY. This is NOT "is the bottom detector a signal" -- SS3p settled
that (oracle +0.017 dSharpe as a boost to this book, -0.675 standalone, because a long-only
near-always-in book is ALREADY long at bottoms) and SS3bl settled it again (skill over a random
entry +0.0023). The open question is narrower: **the book has already decided to buy. Does
delaying that buy until the detector fires get a better fill, and does the better fill survive
to P&L?**

THE CONTROL THAT DECIDES IT (SS3bc, "measure the SKIPPED sample"). A conditional-on-fired return
is a SELECTED sample. SS3bc measured entry timing on the advisor's H4 call and found the oracle
worth +0.2547% at 2h -- 14.6x the spread -- while all 12 honest timing cells were negative,
because base|FILLED was -0.20% against base|SKIPPED +0.22%, t -5.96. Waiting for a pullback
selects the entries you did not want. So every arm here reports, side by side:
    * what the DELAYED entry earned,
    * what MARKET-NOW would have earned on those same events,
    * and what MARKET-NOW earned on the events where waiting changed nothing.

ARMS
  MARKET      buy at the decision bar's close                         (the incumbent)
  DETECTOR    wait up to N hours for P(bottom) >= thr, then buy;
              if it never fires, buy at the deadline                  (the user's proposal)
  ATR-LIMIT   resting buy at close - k*ATR, filled if the low touches
              it within N hours, else buy at the deadline             (SS3bc's flat-offset arm,
                                                                       which is what actually won
                                                                       there: +0.25%/fill)
  ORACLE      buy at the LOWEST low in the window                     (the ceiling, CONTROL #5)

Costs floored at the live FTMO quote (CONTROL #1). Entry improvement is reported in bp of price
AND carried through to the forward return, because an entry gain that does not survive the hold
is bookkeeping, not edge.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from v5_xau_turning_ml import atr                                      # noqa: E402

OUT = ROOT / "data/v5_runs/entry_timing"
COST_USD = 0.448                      # live FTMO round trip, CONTROL #1
DECISION_HOURS = (3, 7, 11, 15, 19, 23)   # the book's 4-hourly passes, server clock
WAIT_HOURS = (4, 8, 12)               # how long we are willing to wait for a better entry
THRESHOLDS = (0.55, 0.60, 0.65)
ATR_OFFSETS = (0.25, 0.50, 1.00)
HOLD_HOURS = (24, 120)                # 1 day and 5 days


def load():
    h1 = pd.read_csv(ROOT / "data/XAUUSD_H1_long.csv", parse_dates=["time"],
                     index_col="time").sort_index()
    h1 = h1[~h1.index.duplicated(keep="last")]
    p = pd.read_csv(OUT.parent / "zigzag_atr/oos_prob_h1.csv", parse_dates=["time"],
                    index_col="time")["p"].reindex(h1.index)
    return h1, p


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    h1, p = load()
    A = atr(h1).values
    close, low, high = h1["close"].values, h1["low"].values, h1["high"].values
    n = len(h1)
    pv = p.values
    scored = np.isfinite(pv)

    # the book's decision bars: 4-hourly passes, only where the detector is already live
    hours = h1.index.hour.values
    ev = np.flatnonzero(np.isin(hours, DECISION_HOURS) & scored & np.isfinite(A))
    ev = ev[ev < n - max(HOLD_HOURS) - max(WAIT_HOURS) - 1]
    print("=" * 100)
    print("ENTRY TIMING -- does waiting for a bottom beat buying now, once the book has decided?")
    print(f"  {len(ev):,} decision events, {h1.index[ev[0]].date()} .. {h1.index[ev[-1]].date()}")
    print(f"  cost floor ${COST_USD}/oz   median price ${np.median(close):,.0f}"
          f"   = {1e4*COST_USD/np.median(close):.2f}bp round trip")
    print("=" * 100)

    rows = []
    for W in WAIT_HOURS:
        for H in HOLD_HOURS:
            # --- MARKET: buy at the decision close
            e_mkt = close[ev]
            # --- ORACLE: the lowest low inside the wait window
            idx = ev[:, None] + np.arange(1, W + 1)[None, :]
            e_orc = low[idx].min(axis=1)
            # exit H hours after the DECISION bar for every arm, so the comparison is
            # like-for-like: a delayed entry must not be handed a longer hold as well.
            exit_px = close[ev + H]

            def pnl(entry):
                return 1e4 * (exit_px - entry - COST_USD) / entry

            base = pnl(e_mkt)
            for name, thr_or_k, kind in (
                    [(f"DETECTOR p>={t}", t, "det") for t in THRESHOLDS] +
                    [(f"ATR-LIMIT -{k}xATR", k, "atr") for k in ATR_OFFSETS]):
                entry = e_mkt.copy()
                filled = np.zeros(len(ev), bool)
                wait_bars = np.zeros(len(ev), int)
                if kind == "det":
                    pw = pv[idx]
                    hit = pw >= thr_or_k
                    any_hit = hit.any(axis=1)
                    first = np.argmax(hit, axis=1)
                    entry[any_hit] = close[ev[any_hit] + 1 + first[any_hit]]
                    filled = any_hit
                    wait_bars[any_hit] = first[any_hit] + 1
                else:
                    lvl = close[ev] - thr_or_k * A[ev]
                    touched = low[idx] <= lvl[:, None]
                    any_hit = touched.any(axis=1)
                    first = np.argmax(touched, axis=1)
                    entry[any_hit] = lvl[any_hit]          # resting limit fills AT the level
                    filled = any_hit
                    wait_bars[any_hit] = first[any_hit] + 1
                # non-fills fall through to a market buy at the deadline
                nofill = ~filled
                entry[nofill] = close[ev[nofill] + W]

                arm = pnl(entry)
                d = arm - base
                t_all = d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else 0.0
                # THE SKIPPED-SAMPLE COLUMNS (SS3bc): what did MARKET earn where we waited?
                bf = base[filled].mean() if filled.any() else np.nan
                bs = base[~filled].mean() if (~filled).any() else np.nan
                entry_bp = 1e4 * (e_mkt - entry) / e_mkt     # +ve = we bought cheaper
                rows.append(dict(wait_h=W, hold_h=H, arm=name,
                                 fill_rate=round(float(filled.mean()), 3),
                                 entry_gain_bp=round(float(entry_bp.mean()), 2),
                                 entry_gain_filled_bp=round(float(entry_bp[filled].mean()), 2)
                                 if filled.any() else np.nan,
                                 pnl_bp=round(float(arm.mean()), 2),
                                 market_bp=round(float(base.mean()), 2),
                                 delta_bp=round(float(d.mean()), 2), t=round(float(t_all), 2),
                                 base_FILLED=round(float(bf), 2),
                                 base_SKIPPED=round(float(bs), 2),
                                 selection_bp=round(float(bf - bs), 2),
                                 med_wait_h=int(np.median(wait_bars[filled])) if filled.any() else 0))
            rows.append(dict(wait_h=W, hold_h=H, arm="ORACLE (lowest low)", fill_rate=1.0,
                             entry_gain_bp=round(float((1e4 * (e_mkt - e_orc) / e_mkt).mean()), 2),
                             entry_gain_filled_bp=np.nan,
                             pnl_bp=round(float(pnl(e_orc).mean()), 2),
                             market_bp=round(float(base.mean()), 2),
                             delta_bp=round(float((pnl(e_orc) - base).mean()), 2),
                             t=np.nan, base_FILLED=np.nan, base_SKIPPED=np.nan,
                             selection_bp=np.nan, med_wait_h=0))

    tb = pd.DataFrame(rows)
    tb.to_csv(OUT / "entry_timing.csv", index=False)
    for H in HOLD_HOURS:
        for W in WAIT_HOURS:
            s = tb[(tb.hold_h == H) & (tb.wait_h == W)]
            print(f"\n--- wait up to {W}h, hold {H}h "
                  f"(market baseline {s.market_bp.iloc[0]:+.2f}bp) ---")
            print(s[["arm", "fill_rate", "entry_gain_bp", "pnl_bp", "delta_bp", "t",
                     "base_FILLED", "base_SKIPPED", "selection_bp", "med_wait_h"]]
                  .to_string(index=False))
    print(f"\n  wrote {OUT}/entry_timing.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
