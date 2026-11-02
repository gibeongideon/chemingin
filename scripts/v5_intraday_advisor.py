#!/usr/bin/env python3
"""THE INTRADAY ADVISOR — a decision instrument for gold, explicitly not an edge.

Requested 2026-11-01: *"the best trading advisor, not necessarily profitable... remember
intraday."* That framing is the right one. After ~130 pre-registered structures failed, the
honest value this project can deliver to a discretionary intraday trader is not a forecast — it
is **the same six questions answered identically every time, with every number counted rather
than felt, and an explicit "there is nothing here" when that is the truth.**

WHAT IT ANSWERS, in the order a trade is actually decided:
  1. WHERE ARE WE?           session, hour, bars left to the close, liquidity vs the day's profile
  2. WHAT HAS ALREADY HAPPENED?  today's range as a fraction of its own ATR, position vs the
                             prior day's high/low
  3. WHAT IS EXPECTED ANYWAY?  the UNCONDITIONAL drift from here to the session close, by bars
                             left. This is CONTROL #2 made usable: any intraday idea must beat
                             this number, not zero
  4. WHAT DOES THE TAPE SAY?  the calibrated multi-timeframe read, EV beside every probability
  5. WHAT DOES IT COST?      live spread, and the hit rate the geometry needs to break even
  6. WHAT WOULD I DO?        size, stop, trail — or FLAT, which is the usual answer

THE TWO NEW CALIBRATIONS, MEASURED HERE RATHER THAN ASSUMED:
  * DRIFT TO CLOSE BY BARS LEFT. Gold's unconditional return from bar k to the session close.
    §3ci showed this is 2.8 of the 10.2bp that the prior-high break appeared to earn, so an
    intraday trader who does not know it will mistake drift for skill.
  * RANGE ALREADY SPENT. How much of a typical day's ATR today has already covered. ⚠️ Treated as
    CONTEXT and tested against the base rate, never as a signal: `range-features-carry-nothing`
    found the high-low box loses to a plain distance baseline, and `window-extremes-are-tautology`
    found "at the N-day high" inverting 20-1967x purely by window return sign. If it carries
    nothing, the advisor says so on every run.

WHAT IT WILL NOT DO. It will not output a direction it cannot support, it will not suggest a short
on gold (§3bh: excess(LONG)+excess(SHORT) = -2*base, and six refutations), and it holds no order
functions — it cannot place, modify or close anything.
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
from v5_tv_ratings import votes                                       # noqa: E402

OUT = ROOT / "data" / "v5_runs"
CAL = OUT / "intraday_advisor_calibration.json"
MTF = OUT / "mtf_calibration.json"
BARS_PER_DAY = 96                       # M15
RANGE_BUCKETS = [(0.0, 0.5), (0.5, 0.8), (0.8, 1.1), (1.1, 1.5), (1.5, 99.0)]
# Server hours. Measured from 720 broker H1 bars 2026-10-30: XAUUSD trades 01:00-23:59 server,
# hour 00 is the daily break, liquidity peaks at hour 17 (17,878 median ticks) and troughs at
# hour 23 (4,594) -- a quarter of peak, which is where "exit at the session close" lands.
SESSIONS = [(1, 9, "ASIA"), (9, 16, "LONDON"), (16, 22, "NEW YORK"), (22, 24, "LATE / THIN")]

# (name, MT5 timeframe constant, bars to request). TradingView publishes a rating per timeframe
# and traders read the column, so all of them are computed; only M15..D1 have a calibration
# behind them, and the ones that do not are shown WITHOUT a P(up) or an EV rather than given a
# borrowed one. M1/M5 are deliberately absent: §3cp measured the M15 hit rate and EV pointing
# OPPOSITE ways, and going finer makes that worse, not better.
TV_FRAMES = [("M15", 15, 5000), ("M30", 30, 5000), ("H1", 16385, 5000), ("H2", 16386, 5000),
             ("H4", 16388, 5000), ("D1", 16408, 3000), ("W1", 32769, 1500),
             ("MN1", 49153, 600)]
FRAME_HOURS = {"M15": 0.25, "M30": 0.5, "H1": 1, "H2": 2, "H4": 4, "D1": 24,
               "W1": 168, "MN1": 730}


def tv_label(r: float) -> str:
    """TradingView's own cut points on the mean vote. Replicated in `v5_tv_ratings.py` (§3cm)."""
    return ("strong_sell" if r <= -0.5 else "sell" if r <= -0.1 else
            "neutral" if r < 0.1 else "buy" if r < 0.5 else "strong_buy")


def load_m15() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data/XAUUSD_M15_long.csv", parse_dates=["time"], index_col="time")
    return d[~d.index.duplicated(keep="last")].sort_index()


def wilder_atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def calibrate(m15: pd.DataFrame) -> dict:
    """Drift-to-close by bars left, and the range-spent context, both counted from history."""
    day = m15.index.normalize()
    g = m15.groupby(day)
    datr = wilder_atr(g["high"].max(), g["low"].min(), g["close"].last()).shift(1)

    c = m15["close"].to_numpy(float)
    h = m15["high"].to_numpy(float)
    l = m15["low"].to_numpy(float)
    nd = day.to_numpy()
    atr_map = datr.reindex(day).to_numpy(float)

    slices, start = [], 0
    for i in range(1, len(m15) + 1):
        if i == len(m15) or nd[i] != nd[start]:
            slices.append((start, i)); start = i

    bl_r, bl_k = [], []
    rs_b, rs_fwd = [], []
    for a, b in slices:
        n = b - a
        if n < 10 or not np.isfinite(atr_map[a]) or atr_map[a] <= 0:
            continue
        last = c[b - 1]
        run_hi, run_lo = h[a], l[a]
        for j in range(a, b - 1):
            run_hi = max(run_hi, h[j]); run_lo = min(run_lo, l[j])
            bl_k.append(b - 1 - j)
            bl_r.append((last / c[j] - 1.0) * 1e4)
            rs_b.append((run_hi - run_lo) / atr_map[a])
            rs_fwd.append((last / c[j] - 1.0) * 1e4)

    bl = pd.DataFrame(dict(k=bl_k, r=bl_r))
    drift = {int(k): dict(mean_bp=float(v["r"].mean()), n=int(len(v)),
                          p_up=float((v["r"] > 0).mean()))
             for k, v in bl.groupby("k") if len(v) >= 200}

    rs = pd.DataFrame(dict(spent=rs_b, r=rs_fwd))
    base_p, base_ev = float((rs["r"] > 0).mean()), float(rs["r"].mean())
    rows = {}
    for lo, hi in RANGE_BUCKETS:
        msk = (rs.spent >= lo) & (rs.spent < hi)
        s = rs[msk]
        if len(s) < 500:
            continue
        # *** n IS NOT THE SAMPLE SIZE. *** These are overlapping intraday bars and range-spent
        # only ever RISES within a day, so a bucket is a few thousand contiguous episodes, not
        # 46,000 independent draws. §3cp's bar-count SE overstated significance ~1.7x this way.
        runs = int((msk & ~msk.shift(1, fill_value=False)).sum())
        rows[f"{lo:.1f}-{hi:.1f}"] = dict(n=int(len(s)), runs=runs,
                                          p_up=float((s["r"] > 0).mean()),
                                          ev_bp=float(s["r"].mean()))
    return dict(drift_by_bars_left=drift,
                range_spent=dict(base_p=base_p, base_ev_bp=base_ev, buckets=rows),
                note="M15, server clock, drift measured to the SESSION CLOSE of the same day")


def session_of(hour: int) -> str:
    for a, b, nm in SESSIONS:
        if a <= hour < b:
            return nm
    return "BREAK (00:00-01:00)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18815)
    ap.add_argument("--symbol", default="XAUUSDs")
    ap.add_argument("--risk-pct", type=float, default=0.5)
    ap.add_argument("--stop-atr", type=float, default=1.5)
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()

    m15 = load_m15()
    if a.recalibrate or not CAL.exists():
        print("calibrating the intraday curves (counting, not fitting)...")
        OUT.mkdir(parents=True, exist_ok=True)
        CAL.write_text(json.dumps(calibrate(m15), indent=2))
    cal = json.loads(CAL.read_text())
    mtf = json.loads(MTF.read_text()) if MTF.exists() else None

    # ---------------- live state ------------------------------------------------------------
    equity = None
    srv = pd.Timestamp.utcnow().tz_localize(None)
    px = spread = np.nan
    si = None
    native: dict = {}
    if not a.offline:
        try:
            from mt5linux import MetaTrader5
            m = MetaTrader5(host="localhost", port=a.port)
            if m.initialize():
                ai = m.account_info()
                equity = float(ai.equity) if ai else None
                m.symbol_select(a.symbol, True)
                si = m.symbol_info(a.symbol)
                t = m.symbol_info_tick(a.symbol)
                if t and t.ask:
                    px, spread = float(t.bid), float(t.ask - t.bid)
                    srv = pd.to_datetime(int(t.time), unit="s")   # SERVER clock
                r = m.copy_rates_from_pos(a.symbol, 15, 0, 40000)
                if r is not None and len(r) > 2000:
                    b = pd.DataFrame([(x[0], x[1], x[2], x[3], x[4], x[5]) for x in r],
                                     columns=["ts", "open", "high", "low", "close", "tick_volume"])
                    b["time"] = pd.to_datetime(b.ts, unit="s")
                    m15 = b.set_index("time").drop(columns=["ts"])
                # NATIVE BARS PER FRAME, not resampled (user request: all timeframes).
                # 40,000 M15 bars is ~14 months, which cannot produce an SMA200 on W1 or MN1 —
                # resampling would silently give those frames a rating built from far too few
                # bars. Asking the broker for each frame's own history is the only way the long
                # frames mean anything, and it costs one bridge call each on an hourly timer.
                for _nm, _tf, _n in TV_FRAMES:
                    try:
                        rr = m.copy_rates_from_pos(a.symbol, _tf, 0, _n)
                    except Exception:                            # noqa: BLE001
                        continue
                    if rr is None or len(rr) < 220:              # SMA200 needs 200 closed bars
                        continue
                    bb = pd.DataFrame([(x[0], x[1], x[2], x[3], x[4], x[5]) for x in rr],
                                      columns=["ts", "open", "high", "low", "close",
                                               "tick_volume"])
                    bb["time"] = pd.to_datetime(bb.ts, unit="s")
                    native[_nm] = bb.set_index("time").drop(columns=["ts"])
                m.shutdown()
        except Exception as e:
            print(f"(broker read failed: {str(e)[:60]})")
    if not np.isfinite(px):
        px, spread = float(m15["close"].iloc[-1]), 0.28
        srv = m15.index[-1]

    day = m15.index.normalize()
    g = m15.groupby(day)
    datr_s = wilder_atr(g["high"].max(), g["low"].min(), g["close"].last())
    atr = float(datr_s.shift(1).iloc[-1])
    today = m15[day == day[-1]]
    prev = datr_s.index[-2] if len(datr_s) > 1 else None
    pdh = float(g["high"].max().iloc[-2]); pdl = float(g["low"].min().iloc[-2])

    print("=" * 100)
    print(f"  INTRADAY ADVISOR — {a.symbol} @ {px:,.2f}   server {srv}   "
          f"{'equity $%s' % f'{equity:,.0f}' if equity else ''}")
    print("=" * 100)

    # ---- 1. WHERE ARE WE -------------------------------------------------------------------
    hour = int(srv.hour)
    bars_left = max(0, BARS_PER_DAY - len(today))
    print(f"\n  1. WHERE WE ARE")
    print(f"     session {session_of(hour)}  ·  server hour {hour:02d}  ·  "
          f"~{bars_left} M15 bars ({bars_left/4:.1f}h) to the 23:59 close")
    print(f"     gold trades 01:00-23:59 server; hour 00 is the daily break. Liquidity peaks at")
    print(f"     hour 17 and is a QUARTER of that at hour 23 — which is where 'exit at the close' lands.")

    # ---- 2. WHAT HAS HAPPENED ---------------------------------------------------------------
    t_hi, t_lo = float(today["high"].max()), float(today["low"].min())
    spent = (t_hi - t_lo) / atr if atr > 0 else np.nan
    print(f"\n  2. WHAT TODAY HAS ALREADY DONE")
    print(f"     range {t_lo:,.2f} - {t_hi:,.2f} = ${t_hi-t_lo:,.2f} = "
          f"{spent:.2f}x the daily ATR (${atr:,.2f})")
    print(f"     prior day  H {pdh:,.2f} ({100*(pdh/px-1):+.2f}%)   L {pdl:,.2f} "
          f"({100*(pdl/px-1):+.2f}%)")
    rsc = cal["range_spent"]
    key = next((f"{lo:.1f}-{hi:.1f}" for lo, hi in RANGE_BUCKETS if lo <= spent < hi), None)
    bk = rsc["buckets"].get(key or "", {})
    if bk:
        ed = 100 * (bk["p_up"] - rsc["base_p"])
        runs = bk.get("runs", bk["n"])
        se = 100 * np.sqrt(0.25 / max(runs, 1))
        print(f"     days this far through their range ({key} ATR): P(up to close) "
              f"{100*bk['p_up']:.1f}% vs base {100*rsc['base_p']:.1f}% "
              f"-> {ed:+.1f}pp ({ed/se:+.1f} SE on {runs:,} episodes), "
              f"EV {bk['ev_bp']:+.2f}bp vs base {rsc['base_ev_bp']:+.2f}")
        if abs(ed / se) < 2.0:
            print(f"     => INSIDE NOISE. Range-spent is CONTEXT, not a signal — exactly what "
                  f"`range-features-carry-nothing` and `window-extremes-are-tautology` predict.")

    # ---- 3. WHAT IS EXPECTED ANYWAY ---------------------------------------------------------
    dk = cal["drift_by_bars_left"].get(str(bars_left)) or \
        cal["drift_by_bars_left"].get(str(min(int(k) for k in cal["drift_by_bars_left"])))
    print(f"\n  3. WHAT IS EXPECTED ANYWAY — the number any idea must BEAT (CONTROL #2)")
    if dk:
        print(f"     unconditional drift from here to the close, at {bars_left} bars left: "
              f"{dk['mean_bp']:+.2f}bp  (P(up) {100*dk['p_up']:.1f}%, n={dk['n']:,})")
        print(f"     = ${dk['mean_bp']/1e4*px:,.2f} per ounce, for doing nothing but being long.")
        print(f"     §3ci: 2.8 of the 10.2bp that the prior-high break appeared to earn was THIS.")
    else:
        print(f"     (no calibration at {bars_left} bars left)")

    # ---- 4. THE TAPE ------------------------------------------------------------------------
    print(f"\n  4. WHAT THE TAPE SAYS — TradingView technical rating, every frame")
    print(f"     {'frame':>5} {'rating':>7} {'label':<12} {'P(up)':>7} {'edge pp':>8} "
          f"{'EV bp':>8} {'net':>8}")
    ratings = {}
    for tf, _tfc, _n in TV_FRAMES:
        d = native.get(tf)
        if d is None:
            # Fall back to resampling the M15 panel. Valid for frames COARSER than M15 and
            # only those: aggregating finer bars into a closed coarse bar adds no information
            # the bar did not already contain by its own close.
            rule = {"M15": None, "M30": "30min", "H1": "1h", "H2": "2h", "H4": "4h",
                    "D1": "1D"}.get(tf)
            if rule is None and tf != "M15":
                continue                                   # W1/MN1 need the broker's own bars
            if tf == "M15":
                d = m15
            else:
                gg = m15.groupby(m15.index.floor(rule))
                d = pd.DataFrame(dict(open=gg["open"].first(), high=gg["high"].max(),
                                      low=gg["low"].min(), close=gg["close"].last(),
                                      tick_volume=gg["tick_volume"].sum()))
        d = d.iloc[:-1]                                    # never rate a bar still forming
        if len(d) < 220:
            continue
        W = votes(d)
        if W.empty:
            continue
        r = float(W.mean(axis=1).iloc[-1])
        lab = tv_label(r)
        ratings[tf] = (r, lab)

        pf = (mtf or {}).get("per_frame", {}).get(tf)
        s = pf["states"].get(lab, {}) if pf else {}
        if "p_up" in s:
            rt = 0.756
            print(f"     {tf:>5} {r:>+7.3f} {lab:<12} {100*s['p_up']:>6.1f}% "
                  f"{100*(s['p_up']-pf['base_p']):>+7.1f} {s['ev_bp']:>+8.2f} "
                  f"{s['ev_bp']-rt:>+8.2f}")
        else:
            # No calibration for this frame. Printing the rating with EMPTY economics is the
            # point: a rating is an opinion until something measures what it was worth.
            print(f"     {tf:>5} {r:>+7.3f} {lab:<12} {'—':>7} {'—':>8} {'—':>8} {'—':>8}")

    if ratings:
        print("     TV ALL-FRAME  " + " · ".join(f"{k} {v[1]}" for k, v in ratings.items()))
        # SHORT-WEIGHTED, w ∝ 1/sqrt(frame hours). Not a preference — the best of five
        # pre-registered weightings in §3ct, which measured short_1/sqrt > short_1/h > equal >
        # long_sqrt > long_h on both horizons tested. The ORDER is what that run established;
        # see the line below for what the level is worth.
        w = {k: 1.0 / np.sqrt(FRAME_HOURS[k]) for k in ratings}
        tot = sum(w.values())
        gen = sum(w[k] * v[0] for k, v in ratings.items()) / tot
        agree = sum(1 for v in ratings.values() if (v[0] > 0) == (gen > 0))
        print(f"     TV GENERALIZED  {gen:+.3f} {tv_label(gen)}   "
              f"({agree}/{len(ratings)} frames agree on the sign)")
        print("     short-weighted (w ∝ 1/sqrt hours), the best of 5 weightings in §3ct — but "
              "the WHOLE family")
        print("     loses to simply holding once cost is paid: best cell -0.50bp/fire at M15, "
              "+0.95bp at H1")
        print("     against a best-of-K floor of +7.18bp, and the first sample half is negative "
              "everywhere.")
    if mtf:
        print(f"     ⚠️ at M15 the hit rate and the EV point OPPOSITE ways and every bucket loses "
              f"net (§3cp).")

    # ---- 5. COST ----------------------------------------------------------------------------
    sp_bp = spread / px * 1e4
    rt_bp = sp_bp * 1.5
    stop_px = a.stop_atr * atr
    print(f"\n  5. WHAT IT COSTS RIGHT NOW")
    print(f"     spread ${spread:.2f} = {sp_bp:.2f}bp · round trip {rt_bp:.2f}bp = "
          f"${rt_bp/1e4*px:.2f}/oz")
    print(f"     a {a.stop_atr}-ATR stop is ${stop_px:,.2f}; cost is "
          f"{100*rt_bp/(stop_px/px*1e4):.1f}% of it")
    for rr in (1.0, 2.0, 3.0):
        need = (1.0 + rt_bp / (stop_px / px * 1e4)) / (1.0 + rr)
        print(f"     at {rr:.0f}:1 R:R you need {need:.1%} to break even"
              f"{'   <- and gold rises ~52% of days anyway' if rr == 1 else ''}")

    # ---- 6. WHAT I WOULD DO ------------------------------------------------------------------
    print(f"\n  6. WHAT I WOULD DO")
    if equity and si is not None:
        per_lot = stop_px * float(si.trade_contract_size)
        step = float(si.volume_step or 0.01)
        lots = max(float(si.volume_min), round((equity * a.risk_pct / 100 / per_lot) / step) * step)
        print(f"     SIZE  {lots:.2f} lots — {a.risk_pct:.2f}% of ${equity:,.0f} = "
              f"${equity*a.risk_pct/100:,.0f} at a ${stop_px:,.2f} stop")
    else:
        print(f"     SIZE  equity x {a.risk_pct:.2f}% / ${stop_px:,.2f} / contract size")
    print(f"     STOP  entry - ${stop_px:,.2f} ({a.stop_atr} x DAILY ATR), server-side on the order")
    print(f"     EXIT  ratcheting 3.0-ATR trail = ${3.0*atr:,.2f}. NO take-profit (§3bl), "
          f"NO move-to-breakeven (§3co, -6.36bp at t -3.55)")
    print(f"     SIDE  LONG or FLAT. Never short: excess(LONG)+excess(SHORT) = -2 x drift, "
          f"and the short leg is refuted six times.")

    # ---- ACTION ITEM: the levels as PRICES, not as distances ---------------------------------
    # Requested 2026-11-02. A stop quoted as "entry - $117.06" still has to be arithmetic'd on a
    # phone, which is where the mistakes happen. These are the same numbers, resolved.
    #
    # Entry is the ASK, because a long is bought there — quoting the bid as the entry understates
    # the stop distance by the whole spread and overstates size by the same. The stop is measured
    # from the ask for the same reason.
    if np.isfinite(px) and np.isfinite(spread):
        ask = px + spread
        sl = ask - stop_px
        dgt = int(getattr(si, "digits", 2) or 2)
        lot_txt = f"{lots:.2f}" if (equity and si is not None) else "see SIZE"
        print("\n  ACTION ITEM — if you take a long here, these are the levels")
        print(f"     ENTRY  {ask:,.{dgt}f}   market (ask; bid is {px:,.{dgt}f})")
        print(f"     SL     {sl:,.{dgt}f}   = entry - ${stop_px:,.2f}, server-side on the order")
        print(f"     LOTS   {lot_txt}")
        print(f"     TRAIL  ${3.0*atr:,.2f} behind the high-water mark, ratchet only · NO TP")
        # The trail is WIDER than the opening stop, so it does nothing until the high-water mark
        # has risen far enough that (high - 3 ATR) clears the original SL. Stating that level
        # stops the trail reading as if it tightens from the first tick.
        arm = ask + (3.0 * atr - stop_px)
        print(f"     the trail only starts improving the SL once price trades above "
              f"{arm:,.{dgt}f}")
        print("     these are LEVELS, not a recommendation — the advisor's own verdict is on the "
              "SIDE line above,")
        print("     and section 3 is what you give up by not simply holding.")
    print(f"""
  THE HONEST BOTTOM LINE. Nothing above forecasts anything. Section 3 is the number to beat and it
  is usually most of what an intraday idea appears to earn; section 4's ingredients have no
  measured edge (§3cm: ~5 opinions dressed as 26; §3bv: oracle 0.497 against 0.503 required); and
  section 5 is why most intraday ideas on this instrument die. **The usual correct answer is
  FLAT.** What this advisor is for is making that answer visible instead of invisible.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
