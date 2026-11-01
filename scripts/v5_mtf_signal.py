#!/usr/bin/env python3
"""Multi-timeframe BUY / FLAT panel with a CALIBRATED 0-100, across M15 M30 H1 H4 H6 H12 D1.

Requested 2026-11-01: one combined signal from the TradingView ratings and the ZigZag detector,
per timeframe, "lower timeframe for entry and bigger for trend", with a 0-100 probability.

*** TWO DELIBERATE DEPARTURES FROM THE REQUEST, BOTH MEASURED. ***

1. LOW PROBABILITY MEANS **FLAT**, NEVER SHORT. The request said a low reading should "mean
   reverse and signal exit or sell". On gold that inference is backwards. §3bh's identity --
   excess(LONG) + excess(SHORT) = -2 x base -- means a short begins twice the drift behind, and
   the short leg has now been refuted six times, most recently §3bn on the trail engine itself
   (long-only 1.091 vs long/short 0.954, BOTH halves negative). §3cm measured it again here: the
   TradingView "strong sell" bucket, traded short, returns **-7.6 to -8.2bp against simply
   HOLDING gold**. So a weak reading closes a long; it does not open a short. The short column is
   printed anyway, so the decision is visible rather than asserted.

2. THE 0-100 IS AN EMPIRICAL FREQUENCY, NOT A CONFIDENCE. Nothing here invents a weighting.
   Every number is P(up over the next H bars | this state), counted from history, printed beside
   **its own base rate** and the **edge** between them. `xau-drift-baseline-inverts-3r`:
   thresholds must sit around the base rate, never 0.5 -- gold rises ~52% of days, so a "70%"
   against a 68% base rate is nothing at all. CONTROL #8 adds the other half: a probability is
   payoff-blind, so EV/fire in bp is printed next to every hit rate.

WHAT THE INGREDIENTS ARE WORTH, stated up front so the panel is not over-read:
  * TradingView rating (§3cm) -- 26 components carrying only **4.8-5.0 independent opinions**,
    **0.80 correlated with a single `close vs SMA50`**, and it LOSES to this repo's champion at
    H4 and D1. It is a disciplined DESCRIPTION of the tape, not an edge.
  * ZigZag bottom detector (§3bv) -- closed on MECHANISM: oracle **0.497** precision at RR=1
    against a required **0.503**. Zero headroom before any model exists. Shown at **H1 only**,
    its design frame; running it on D1 would be off-design.
  **Neither has a measured edge. The panel's value is consistency and calibration -- it replaces
  a feeling with a counted frequency -- and it must not be read as a forecast.**

PROVENANCE AND THE LOOKAHEAD TRAP. Every frame is aggregated from the one clean M15 file, never
the spliced H4 (`multi-broker dataset is a leak`). Bars are stamped at their START by
`index.floor`, so **the forming bar is always dropped**: a bar stamped 12:00 on H4 is not knowable
until 16:00, and reading it live is exactly the error that faked an AUC of 0.901
(`cross-timeframe-resample-is-lookahead`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from v5_tv_ratings import votes, eff_rank, MAS                        # noqa: E402

OUT = ROOT / "data" / "v5_runs"
CALIB = OUT / "mtf_calibration.json"
HORIZON_BARS = 6              # uniform across frames: no per-frame tuning, so no hidden search
RULE = {"M15": "15min", "M30": "30min", "H1": "1h", "H4": "4h",
        "H6": "6h", "H12": "12h", "D1": "1D"}
ENTRY_FRAMES = ["M15", "M30", "H1"]
TREND_FRAMES = ["H4", "H6", "H12", "D1"]
ALL_FRAMES = ENTRY_FRAMES + TREND_FRAMES


def load_m15() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data/XAUUSD_M15_long.csv", parse_dates=["time"], index_col="time")
    return d[~d.index.duplicated(keep="last")].sort_index()


def frame(m15: pd.DataFrame, tf: str, drop_forming: bool = True) -> pd.DataFrame:
    """Exact OHLC aggregation of M15. Bars are START-stamped, so the last one is INCOMPLETE."""
    if tf == "M15":
        out = m15.copy()
    else:
        g = m15.groupby(m15.index.floor(RULE[tf]))
        out = pd.DataFrame(dict(open=g["open"].first(), high=g["high"].max(),
                                low=g["low"].min(), close=g["close"].last(),
                                legs=g["close"].count()))
        if "tick_volume" in m15.columns:
            out["tick_volume"] = g["tick_volume"].sum()
        # completeness by the MODAL leg count rather than a hardcoded number, because the gold
        # day is 23h and D1/H12/H6 therefore do not have a clean power-of-two leg count
        if len(out) > 20:
            modal = out["legs"].mode().iloc[0]
            out = out[out["legs"] >= 0.6 * modal]
    return out.iloc[:-1] if drop_forming and len(out) > 1 else out


def band(r: float) -> str:
    """TradingView's own five labels."""
    return ("strong_sell" if r <= -0.5 else "sell" if r <= -0.1 else
            "neutral" if r < 0.1 else "buy" if r < 0.5 else "strong_buy")


def composite(rating_by_tf: dict, frames: list) -> str:
    """UP / MIXED / DOWN from the mean rating across a group of frames."""
    vals = [rating_by_tf[f] for f in frames if f in rating_by_tf and np.isfinite(rating_by_tf[f])]
    if not vals:
        return "na"
    m = float(np.mean(vals))
    return "UP" if m >= 0.2 else "DOWN" if m <= -0.2 else "MIXED"


def calibrate(m15: pd.DataFrame) -> dict:
    """Count P(up) and EV per state, per frame, and for the entry x trend composite."""
    cal = {"horizon_bars": HORIZON_BARS, "per_frame": {}, "composite": {}}
    rat = {}
    for tf in ALL_FRAMES:
        d = frame(m15, tf)
        W = votes(d)
        if W.empty:
            continue
        c = d["close"].reindex(W.index).astype(float)
        fwd = (c.shift(-HORIZON_BARS) / c - 1.0) * 1e4
        r = W.mean(axis=1)
        rat[tf] = r
        ok = np.isfinite(fwd)
        base_p, base_ev = float((fwd[ok] > 0).mean()), float(fwd[ok].mean())
        rows = {}
        for lab in ("strong_sell", "sell", "neutral", "buy", "strong_buy"):
            msk = ok & (r.map(band) == lab)
            n = int(msk.sum())
            if n < 100:
                rows[lab] = dict(n=n, insufficient=True); continue
            # *** n IS NOT THE SAMPLE SIZE. *** Bars overlap (a 6-bar horizon reuses each
            # move six times) AND the rating persists, so "strong sell" is a handful of long
            # EPISODES, not thousands of independent draws. Counting contiguous runs is the
            # conservative denominator -- the same correction `trigger-search-max-stat-null`
            # made when it reported 19 non-overlapping fires rather than the raw bar count.
            runs = int((msk & ~msk.shift(1, fill_value=False)).sum())
            rows[lab] = dict(n=n, runs=runs, p_up=float((fwd[msk] > 0).mean()),
                             ev_bp=float(fwd[msk].mean()),
                             ev_short_vs_hold=float(-fwd[msk].mean() - base_ev))
        cal["per_frame"][tf] = dict(bars=int(len(W)), base_p=base_p, base_ev_bp=base_ev,
                                    first=str(W.index[0].date()), last=str(W.index[-1].date()),
                                    states=rows)

    # ---- the composite, calibrated on the D1 clock so trend and entry share one horizon -------
    idx = rat["D1"].index if "D1" in rat else None
    if idx is not None:
        al = {tf: rat[tf].reindex(idx, method="ffill") for tf in rat}
        d1 = frame(m15, "D1")
        c = d1["close"].reindex(idx).astype(float)
        fwd = (c.shift(-HORIZON_BARS) / c - 1.0) * 1e4
        ok = np.isfinite(fwd)
        base_p, base_ev = float((fwd[ok] > 0).mean()), float(fwd[ok].mean())
        ent = pd.Series([composite({k: al[k].iloc[i] for k in al}, ENTRY_FRAMES)
                         for i in range(len(idx))], index=idx)
        trd = pd.Series([composite({k: al[k].iloc[i] for k in al}, TREND_FRAMES)
                         for i in range(len(idx))], index=idx)
        cells = {}
        for t in ("UP", "MIXED", "DOWN"):
            for e in ("UP", "MIXED", "DOWN"):
                msk = ok & (trd == t) & (ent == e)
                n = int(msk.sum())
                runs = int((msk & ~msk.shift(1, fill_value=False)).sum())
                cells[f"{t}/{e}"] = (dict(n=n, runs=runs, insufficient=True) if n < 50 else
                                     dict(n=n, runs=runs, p_up=float((fwd[msk] > 0).mean()),
                                          ev_bp=float(fwd[msk].mean())))
        cal["composite"] = dict(base_p=base_p, base_ev_bp=base_ev, n=int(ok.sum()), cells=cells)
    return cal


def fmt_pct(x) -> str:
    return "  —  " if x is None or not np.isfinite(x) else f"{100*x:5.1f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true", help="rebuild the lookup from history")
    ap.add_argument("--port", type=int, default=18815)
    ap.add_argument("--symbol", default="XAUUSDs")
    ap.add_argument("--offline", action="store_true", help="read the CSV instead of the broker")
    a = ap.parse_args()

    m15 = load_m15()
    if a.calibrate or not CALIB.exists():
        print("calibrating from history (this counts, it does not fit)...")
        cal = calibrate(m15)
        OUT.mkdir(parents=True, exist_ok=True)
        CALIB.write_text(json.dumps(cal, indent=2))
        print(f"wrote {CALIB}")
    cal = json.loads(CALIB.read_text())

    # ---------------- the historical table ---------------------------------------------------
    print("=" * 108)
    print("MULTI-TIMEFRAME PANEL — every number counted from history, never invented")
    print("=" * 108)
    print(f"  horizon: {HORIZON_BARS} bars of each frame. P(up) and EV are what FOLLOWED this "
          f"state historically.")
    print(f"\n  {'frame':>5} {'bars':>8} {'base':>6} | {'strong_sell':>12} {'sell':>12} "
          f"{'neutral':>12} {'buy':>12} {'strong_buy':>12}")
    print(f"  {'':>5} {'':>8} {'P(up)':>6} | " + " ".join(f"{'P(up) n':>12}" for _ in range(5)))
    print("  " + "-" * 102)
    for tf in ALL_FRAMES:
        pf = cal["per_frame"].get(tf)
        if not pf:
            continue
        cells = []
        for lab in ("strong_sell", "sell", "neutral", "buy", "strong_buy"):
            s = pf["states"].get(lab, {})
            cells.append("     —      " if s.get("insufficient") or "p_up" not in s
                         else f"{fmt_pct(s['p_up'])} {s['n']:>6}")
        print(f"  {tf:>5} {pf['bars']:>8,} {fmt_pct(pf['base_p']):>6} | " + " ".join(cells))

    print(f"\n  EDGE over each frame's OWN base rate (percentage points) — this is the only "
          f"column that means anything")
    print(f"  {'frame':>5} | {'strong_sell':>12} {'sell':>12} {'neutral':>12} {'buy':>12} "
          f"{'strong_buy':>12}")
    print("  " + "-" * 78)
    for tf in ALL_FRAMES:
        pf = cal["per_frame"].get(tf)
        if not pf:
            continue
        cells = []
        for lab in ("strong_sell", "sell", "neutral", "buy", "strong_buy"):
            s = pf["states"].get(lab, {})
            if "p_up" not in s:
                cells.append("     —      "); continue
            # SE of a proportion at this n. An "edge" smaller than ~2 SE is indistinguishable
            # from the base rate, and with 35 frame-state cells plus 9 composite cells the
            # best-of-44 floor is higher still (§3ak: searching harder raises the bar faster
            # than the statistic). Printing the SE is what stops a +4.8pp cell being believed.
            se = 100 * np.sqrt(0.25 / max(s.get("runs") or s["n"], 1))
            ed = 100 * (s["p_up"] - pf["base_p"])
            cells.append(f"{ed:+6.1f}/{ed/se:+4.1f}se")
        print(f"  {tf:>5} | " + " ".join(cells))
    print("  " + "-" * 78)
    print("  reading: EDGE_pp / standard errors, where SE uses the number of EPISODES "
          "(contiguous runs),")
    print("  not the bar count — bars overlap at a 6-bar horizon and the rating persists in runs.")
    print("  44 cells exist here (7 frames x 5 states + 9 composite), so even |2se| is not a "
          "finding on its own.")

    # ---------------- THE COLUMN THAT REVERSES THE HIT-RATE TABLE ----------------------------
    # CONTROL #8: a probability is payoff-blind. Printing P(up) without EV is how a 5.9-sigma AUC
    # lift once produced a -0.702 Sharpe. Here the two point OPPOSITE ways at the fast frames.
    RT = 0.756
    print(f"\n  EV LADDER — the same states, priced. 'net' is EV minus the {RT:.2f}bp round trip.")
    print(f"  {'frame':>5} {'base EV':>8} | " +
          " ".join(f"{lab[:11]:>13}" for lab in
                   ("strong_sell", "sell", "neutral", "buy", "strong_buy")))
    print("  " + "-" * 92)
    for tf in ALL_FRAMES:
        pf = cal["per_frame"].get(tf)
        if not pf:
            continue
        cells = []
        for lab in ("strong_sell", "sell", "neutral", "buy", "strong_buy"):
            st = pf["states"].get(lab, {})
            if "ev_bp" not in st:
                cells.append(f"{'—':>13}"); continue
            net = st["ev_bp"] - RT
            cells.append(f"{st['ev_bp']:>+7.2f}/{net:>+5.2f}" if net > 0
                         else f"{st['ev_bp']:>+7.2f}/{net:>+5.2f}")
        print(f"  {tf:>5} {pf['base_ev_bp']:>+8.2f} | " + " ".join(cells))
    print("  " + "-" * 92)
    print("  reading: EV_bp / net_bp. Compare EV against that frame's OWN base EV, not against 0 —")
    print("  'beats zero' is not a benchmark when holding gold already pays (CONTROL #2).")

    comp = cal.get("composite", {})
    if comp:
        print(f"\n  COMPOSITE — trend (H4/H6/H12/D1) x entry (M15/M30/H1), on the D1 clock, "
              f"base P(up) {fmt_pct(comp['base_p'])}%")
        print(f"  {'trend/entry':>13} {'n':>6} {'P(up)':>7} {'edge pp':>9} {'vs SE':>7} "
              f"{'runs':>6} {'EV bp':>9}")
        print("  " + "-" * 60)
        for k, v in comp["cells"].items():
            if v.get("insufficient"):
                print(f"  {k:>13} {v['n']:>6}   too few"); continue
            se = 100 * np.sqrt(0.25 / max(v.get("runs") or v["n"], 1))
            ed = 100 * (v["p_up"] - comp["base_p"])
            print(f"  {k:>13} {v['n']:>6} {fmt_pct(v['p_up']):>7} "
                  f"{ed:>+8.1f} {ed/se:>+6.1f}se {v.get('runs','?'):>6} {v['ev_bp']:>+9.2f}")

    # ---------------- the live read ----------------------------------------------------------
    print(f"\n{'='*108}\n  LIVE READ")
    cur, src = {}, "CSV (offline)"
    if not a.offline:
        try:
            from mt5linux import MetaTrader5
            m = MetaTrader5(host="localhost", port=a.port)
            if m.initialize():
                m.symbol_select(a.symbol, True)
                r = m.copy_rates_from_pos(a.symbol, 15, 0, 60000)   # M15
                if r is not None and len(r) > 5000:
                    b = pd.DataFrame([(x[0], x[1], x[2], x[3], x[4], x[5]) for x in r],
                                     columns=["ts", "open", "high", "low", "close", "tick_volume"])
                    b["time"] = pd.to_datetime(b.ts, unit="s")
                    m15 = b.set_index("time").drop(columns=["ts"])
                    src = f"broker {a.symbol} @ port {a.port} (server clock)"
                m.shutdown()
        except Exception as e:
            print(f"  (broker read failed: {str(e)[:60]} — falling back to CSV)")
    print(f"  source: {src}")
    print(f"\n  {'frame':>5} {'rating':>7} {'label':<12} {'b/s/n':>9} {'P(up)':>7} "
          f"{'edge pp':>8} {'EV bp':>8} {'n':>7}")
    print("  " + "-" * 76)
    for tf in ALL_FRAMES:
        d = frame(m15, tf)
        W = votes(d)
        if W.empty:
            print(f"  {tf:>5}   (insufficient history)"); continue
        r = float(W.mean(axis=1).iloc[-1])
        lab = band(r)
        bb, ss = int((W.iloc[-1] == 1).sum()), int((W.iloc[-1] == -1).sum())
        cur[tf] = r
        pf = cal["per_frame"].get(tf, {})
        s = pf.get("states", {}).get(lab, {})
        if "p_up" in s:
            print(f"  {tf:>5} {r:>+7.3f} {lab:<12} {f'{bb}/{ss}/{26-bb-ss}':>9} "
                  f"{fmt_pct(s['p_up']):>7} {100*(s['p_up']-pf['base_p']):>+7.1f} "
                  f"{s['ev_bp']:>+8.2f} {s['n']:>7}")
        else:
            print(f"  {tf:>5} {r:>+7.3f} {lab:<12} {f'{bb}/{ss}/{26-bb-ss}':>9}   (uncalibrated)")

    t, e = composite(cur, TREND_FRAMES), composite(cur, ENTRY_FRAMES)
    cell = comp.get("cells", {}).get(f"{t}/{e}", {})
    print(f"\n  TREND ({'/'.join(TREND_FRAMES)}) = {t}    ENTRY ({'/'.join(ENTRY_FRAMES)}) = {e}")
    if "p_up" in cell:
        edge = 100 * (cell["p_up"] - comp["base_p"])
        score = max(0.0, min(100.0, 50.0 + 2.5 * edge))
        print(f"  CALIBRATED P(up over {HORIZON_BARS} days) = {100*cell['p_up']:.1f}%  "
              f"against a base rate of {100*comp['base_p']:.1f}%  ->  EDGE {edge:+.1f}pp  "
              f"on n={cell['n']}")
        print(f"  SCORE {score:.0f}/100   (50 = the base rate; the score IS the edge, rescaled)")
        verdict = ("LONG" if edge > 2.0 else "FLAT / CLOSE LONGS" if edge < -2.0 else "NO EDGE")
        print(f"  VERDICT: **{verdict}**")
    else:
        print(f"  this trend/entry cell has too few historical observations to quote a number")
    print(f"""
  *** NEVER SHORT ON A WEAK READING. *** §3cm measured the short side of exactly this rating:
  the 'strong sell' bucket, traded short, returns -7.6 to -8.2bp AGAINST SIMPLY HOLDING GOLD,
  because excess(LONG) + excess(SHORT) = -2 x base. A weak reading closes a long. It does not
  open a short. And the ingredients have no measured edge of their own -- the rating is ~5
  opinions dressed as 26 and 0.80 correlated with one moving average (§3cm); the ZigZag's oracle
  is 0.497 against a required 0.503 (§3bv). What this panel buys is CONSISTENCY and CALIBRATION,
  not prediction.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
