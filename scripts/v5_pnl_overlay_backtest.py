"""What a fixed dollar profit-target does to the champion on gold. Measured before deploying.

THE REQUEST. Close ALL positions when combined floating P/L reaches +$200; in the last 5 hours
before the daily close, harvest at +$50; re-enter only on a fresh entry signal.

WHY THIS NEEDS MEASURING RATHER THAN JUST BUILDING. Two closed results in this file are the same
family: §3x backtested "cash out on first reasonable profit" and found ALL 48 cells negative with
walk-forward -0.76 against buy-and-hold +0.79, and §3ab's exit sweep had 0 of 64 cells clearing
+0.50. The mechanism is structural, not incidental -- the champion is a TREND FOLLOWER whose
entire edge is the right tail, and a fixed target truncates exactly that tail while leaving the
left one open. On a $102k account +$200 is **+0.195%**; gold's daily range alone is ~2.7% of
price, so the target sits inside a fifth of one average day.

HOW IT IS MEASURED. The gold sleeve at M15 (clean, provenance-verified), vol-targeted exactly as
the live engine sizes it, which is 50% of the live book by class weight. For each day the engine
holds a position, the intraday path is walked bar by bar:

    * floating P/L crosses +TARGET  -> flatten, stay flat until the re-entry rule allows back in
    * inside the last 5h before the daily close and P/L >= +HARVEST -> flatten for the day
    * otherwise hold, exactly as the champion would

and the result is compared against the SAME engine with no overlay. Equal sizing, equal costs,
so the only difference is the exit discipline.

RE-ENTRY IS MODELLED HONESTLY. "Not immediately -- wait for the right entry" is implemented as
the champion's own forecast having to STRENGTHEN before re-entry (the only measured signal in
this repo), with a same-day lockout so one target does not become ten round trips. A naive
"re-enter next bar" arm is run alongside to show what the waiting is worth.

    python scripts/v5_pnl_overlay_backtest.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "v5_runs" / "pnl_overlay"

EQUITY = 102_413.0
SPREAD_BP, SPREAD_MULT = 0.94, 1.5
TARGET_VOL, MAX_LEV, BUFFER = 0.10, 8.0, 0.20
CLOSE_UTC_H = 21          # gold's daily session break, measured at 21:00-22:00 UTC
HARVEST_WINDOW_H = 5


def engine_positions(d: pd.DataFrame) -> pd.Series:
    """The champion's held leverage on daily bars, buffered exactly as the live engine."""
    from src.v5.xau_dual_signals import champion_signal
    ret = d["close"].pct_change()
    vol = ret.ewm(halflife=20, min_periods=20).std() * np.sqrt(252)
    fc = champion_signal(d["close"])
    tgt = (fc.clip(-2, 2) * (TARGET_VOL / vol)).clip(-MAX_LEV, MAX_LEV)
    band = (BUFFER * (TARGET_VOL / vol).clip(0, MAX_LEV)).values
    p, out, held = tgt.values.copy(), np.zeros(len(tgt)), 0.0
    for i in range(len(p)):
        if np.isfinite(p[i]):
            b = band[i] if np.isfinite(band[i]) else 0.0
            if abs(p[i] - held) > b:
                held = p[i] - np.sign(p[i] - held) * b
        out[i] = held
    return pd.Series(out, index=tgt.index).shift(1).fillna(0.0)


def run(m15: pd.DataFrame, target_usd: float, harvest_usd: float,
        mode: str) -> tuple[pd.Series, dict]:
    """mode: 'none' (champion), 'signal' (wait for a stronger forecast), 'immediate'."""
    from src.v5.xau_dual_signals import champion_signal
    d = m15.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                "close": "last"}).dropna()
    pos = engine_positions(d)
    fc = champion_signal(d["close"])
    ow = SPREAD_BP * SPREAD_MULT * 1e-4
    mi, px = m15.index, m15["close"].values
    daily_ret, n_tp, n_harv, n_days_flat = [], 0, 0, 0
    for i in range(1, len(d)):
        day = d.index[i]
        lev = float(pos.iloc[i])
        if lev == 0 or not np.isfinite(lev):
            daily_ret.append(0.0); continue
        s = int(np.searchsorted(mi.values, np.datetime64(day), side="left"))
        e = int(np.searchsorted(mi.values, np.datetime64(day + pd.Timedelta(days=1)),
                                side="left"))
        if s >= len(mi) or e <= s + 1:
            daily_ret.append(0.0); continue
        entry_px = px[s]
        notional = lev * EQUITY
        # re-entry gate: has the forecast STRENGTHENED versus yesterday?
        stronger = bool(abs(fc.iloc[i]) > abs(fc.iloc[i - 1])) if i > 0 else False
        flat_from = None
        for k in range(s + 1, e):
            pnl = notional * (px[k] - entry_px) / entry_px * np.sign(lev)
            hrs_left = CLOSE_UTC_H - mi[k].hour
            if mode != "none":
                if pnl >= target_usd:
                    flat_from = k; n_tp += 1; break
                if 0 < hrs_left <= HARVEST_WINDOW_H and pnl >= harvest_usd:
                    flat_from = k; n_harv += 1; break
        if flat_from is None:
            r = np.sign(lev) * (px[e - 1] - entry_px) / entry_px * abs(lev)
        else:
            r = np.sign(lev) * (px[flat_from] - entry_px) / entry_px * abs(lev)
            n_days_flat += 1
            if mode == "immediate" and flat_from + 1 < e:
                r += np.sign(lev) * (px[e - 1] - px[flat_from]) / px[flat_from] * abs(lev)
                r -= abs(lev) * ow
            elif mode == "signal" and stronger and flat_from + 1 < e:
                r += np.sign(lev) * (px[e - 1] - px[flat_from]) / px[flat_from] * abs(lev)
                r -= abs(lev) * ow
        turn = abs(pos.iloc[i] - pos.iloc[i - 1]) + (abs(lev) if flat_from is not None else 0)
        daily_ret.append(r - turn * ow)
    s = pd.Series(daily_ret, index=d.index[1:])
    eq = (1 + s).cumprod()
    stats = dict(sharpe=float(s.mean() / s.std() * np.sqrt(252)) if s.std() > 0 else np.nan,
                 cagr=float(eq.iloc[-1] ** (252 / len(s)) - 1) * 100,
                 maxdd=float((eq / eq.cummax() - 1).min() * 100),
                 tp_hits=n_tp, harvests=n_harv, days_cut=n_days_flat,
                 days=len(s))
    return s, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=200.0)
    ap.add_argument("--harvest", type=float, default=50.0)
    a = ap.parse_args()
    from scripts.v5_advisor_measure import load_frames
    from src.v5.mtf_frames import frames
    _, m15 = load_frames()
    M = frames(m15)["M15"]
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 98)
    print(f"FIXED DOLLAR PROFIT-TARGET on the champion's gold sleeve, M15 path")
    print(f"  equity ${EQUITY:,.0f}   target +${a.target:,.0f} (= {a.target/EQUITY*100:.3f}% "
          f"of equity)   harvest +${a.harvest:,.0f} in the last {HARVEST_WINDOW_H}h")
    print("=" * 98)
    print(f"  {'variant':34s} {'Sharpe':>7s} {'CAGR%':>7s} {'maxDD%':>7s} "
          f"{'TP hits':>8s} {'harvest':>8s} {'days cut':>9s}")
    base_s, base = run(M, a.target, a.harvest, "none")
    print(f"  {'champion, no overlay':34s} {base['sharpe']:+7.2f} {base['cagr']:+7.2f} "
          f"{base['maxdd']:7.1f} {'-':>8s} {'-':>8s} {'-':>9s}")
    rows = {}
    for mode, lab in (("signal", "overlay, re-enter on a signal"),
                      ("immediate", "overlay, re-enter immediately")):
        s, st = run(M, a.target, a.harvest, mode)
        rows[mode] = (s, st)
        print(f"  {lab:34s} {st['sharpe']:+7.2f} {st['cagr']:+7.2f} {st['maxdd']:7.1f} "
              f"{st['tp_hits']:8d} {st['harvests']:8d} {st['days_cut']:9d}")
    print()
    for mode, lab in (("signal", "signal re-entry"), ("immediate", "immediate re-entry")):
        st = rows[mode][1]
        print(f"  {lab:20s} vs champion: Sharpe {st['sharpe']-base['sharpe']:+.2f}   "
              f"CAGR {st['cagr']-base['cagr']:+.2f}pp   maxDD {st['maxdd']-base['maxdd']:+.1f}pp")
    print(f"\n  the overlay cut the position on {rows['signal'][1]['days_cut']} of "
          f"{base['days']} days ({rows['signal'][1]['days_cut']/base['days']*100:.0f}%)")
    pd.DataFrame({"champion": base_s, "overlay_signal": rows["signal"][0],
                  "overlay_now": rows["immediate"][0]}).to_csv(OUT / "overlay.csv")
    print(f"\n  wrote {(OUT/'overlay.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
