#!/usr/bin/env python3
"""A PRE-TRADE HELPER for discretionary gold trades — built to remove emotion, not to predict.

Requested 2026-10-30: *"for the OANDA, usually we speculate blindly, i need a helper so that we
dont add emotions."*

*** WHAT THIS IS NOT. *** It is not a signal. It will never tell you to buy or sell. §3cm measured
the TradingView rating this prints and found it is **~5 independent opinions dressed as 26**, about
**0.80 correlated with a single `close vs SMA50`**, and it **loses to this repo's own champion** on
the frame the champion is built for. Printing a number that cannot predict, as though it could,
would add a new emotion rather than remove one.

*** WHAT IT IS. *** The four things that actually decide whether a discretionary trade is survivable,
computed instead of felt:

  1. THE TAPE, described identically every time. Same indicators, same thresholds, no
     interpretation -- and the vote count reported beside its EFFECTIVE opinion count, so a
     "22 of 26 agree" can never be read as 22 pieces of evidence.
  2. THE COST FLOOR. The live spread, and the hit rate this specific trade geometry must achieve
     to break even. `intraday-breakeven-table`: the spread demands 0.587 at M15 against 0.510 at
     D1 -- the single most common way a discretionary intraday trade is dead before it is opened.
  3. THE SIZE, from equity x risk fraction / (stop distance x contract size). Never from
     conviction. `xau-riskfrac-sizing-table`: Sharpe is INVARIANT across 0.25-4% risk, so size is
     a pure risk dial and never a view -- but 1% -> 2% multiplies breach risk 24x.
  4. THE WARNINGS THIS ACCOUNT HAS EARNED. Shorts refuted six times, brackets in ATR not percent,
     no take-profit, and the user's own -37% manual drawdown that latched a permanent bot halt.

Read-only by construction: it holds no order functions and cannot place, modify or close anything.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from v5_tv_ratings import votes, eff_rank, MAS, OSC                   # noqa: E402
from v5_xau_champion_lifts import champion_recipe                     # noqa: E402

TF_CODE = {"M15": 15, "M30": 30, "H1": 16385, "H4": 16388, "D1": 16408}
VALIDATED_TRAIL_ATR = 3.0          # §3bn: peaks at 3.0, smooth decay both sides, ratchet +0.077 SR

# *** THE D=6 TRAP. *** `champion_recipe` is defined on H4 at scale 1.0, where it spans 96-1536
# BARS. The scale is therefore per-timeframe -- H4 1.0, H1 4.0, D1 1/6 -- and the bar count needed
# scales with it. Feeding it 600 bars at scale 1.0 silently returns NaN for every row, which is
# exactly what the first run of this helper printed. M15 is deliberately absent: the equivalent
# scale is 16.0, i.e. 24,576 bars ~ 256 days, so the champion is simply not an M15 model.
CHAMP = {"H1": (4.0, 7000), "H4": (1.0, 1900), "D1": (1 / 6, 500)}


def fetch(m, sym: str, tf: str, n: int = 600) -> pd.DataFrame:
    r = m.copy_rates_from_pos(sym, TF_CODE[tf], 0, n)
    if r is None or len(r) < 250:
        return pd.DataFrame()
    d = pd.DataFrame([(x[0], x[1], x[2], x[3], x[4], x[5]) for x in r],
                     columns=["ts", "open", "high", "low", "close", "tick_volume"])
    d["time"] = pd.to_datetime(d.ts, unit="s")      # SERVER clock throughout
    return d.set_index("time").drop(columns=["ts"])


def wilder_atr(d: pd.DataFrame, n: int = 14) -> float:
    h, l, c = d["high"], d["low"], d["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / n, adjust=False).mean().iloc[-1])


def label(r: float) -> str:
    return ("STRONG SELL" if r <= -0.5 else "SELL" if r <= -0.1 else
            "NEUTRAL" if r < 0.1 else "BUY" if r < 0.5 else "STRONG BUY")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18815, help="MT5 bridge (18815 = Switch demo)")
    ap.add_argument("--symbol", default="XAUUSDs")
    ap.add_argument("--side", choices=["long", "short"], help="the trade you are considering")
    ap.add_argument("--risk-pct", type=float, default=0.5, help="%% of equity at risk, default 0.5")
    ap.add_argument("--stop-atr", type=float, default=1.5, help="initial stop in ATR, default 1.5")
    a = ap.parse_args()

    from mt5linux import MetaTrader5
    m = MetaTrader5(host="localhost", port=a.port)
    if not m.initialize():
        print("bridge init failed:", m.last_error()); return 2
    ai = m.account_info()
    m.symbol_select(a.symbol, True)
    si, tick = m.symbol_info(a.symbol), m.symbol_info_tick(a.symbol)
    if si is None or tick is None:
        print(f"{a.symbol} does not resolve on this bridge "
              f"(Switch symbols carry an 's' suffix: XAUUSDs)"); m.shutdown(); return 1

    px, spread = float(tick.bid), float(tick.ask - tick.bid)
    spread_bp = spread / px * 1e4
    print("=" * 92)
    print(f"  PRE-TRADE HELPER — {a.symbol} @ {px:,.2f}   "
          f"spread ${spread:.2f} = {spread_bp:.2f}bp   "
          f"account {ai.login} ({'DEMO' if ai.trade_mode == 0 else 'REAL'}) "
          f"equity ${ai.equity:,.2f}")
    print("=" * 92)

    # ---------------- 1. THE TAPE, described the same way every time -----------------------
    print(f"\n  1. THE TAPE — identical rules every time, no interpretation")
    print(f"     {'frame':>6} {'rating':>7}  {'label':<12} {'buy/sell/neut':>14} "
          f"{'ATR(14)':>10} {'champion':>9}")
    frames = {}
    for tf in ("M15", "H1", "H4", "D1"):
        d = fetch(m, a.symbol, tf)
        if d.empty:
            print(f"     {tf:>6}  (insufficient history)"); continue
        W = votes(d)
        if W.empty:
            continue
        r = float(W.mean(axis=1).iloc[-1])
        b, s = int((W.iloc[-1] == 1).sum()), int((W.iloc[-1] == -1).sum())
        atr = wilder_atr(d)
        fcv = np.nan
        if tf in CHAMP:
            scale, need = CHAMP[tf]
            dd = fetch(m, a.symbol, tf, need)
            if not dd.empty and len(dd) >= need * 0.6:
                fcv = float(champion_recipe(dd["close"].astype(float), scale, 1.5).iloc[-1])
        fcs = f"{fcv:>9.2f}" if np.isfinite(fcv) else f"{'—':>9}"
        print(f"     {tf:>6} {r:>+7.3f}  {label(r):<12} {f'{b}/{s}/{26-b-s}':>14} "
              f"{atr:>9.2f} {fcs}")
        frames[tf] = dict(rating=r, atr=atr, W=W, fc=fcv)
    if not frames:
        print("     no usable history on this bridge"); m.shutdown(); return 1

    ref = frames.get("H4") or next(iter(frames.values()))
    pr, _, _ = eff_rank(ref["W"])
    ma_block = ref["W"][MAS].mean(axis=1).iloc[-1]
    print(f"\n     *** the 26 votes carry ~{pr:.1f} INDEPENDENT opinions (§3cm). A 22-of-26 "
          f"consensus is about a 4-of-5. ***")
    print(f"     the 15 moving averages read {ma_block:+.2f} — they share ONE input (price vs a "
          f"trailing average) and are ~2.7 opinions, not 15.")
    print(f"     champion column is this repo's own forecast, 0..2 (0 = flat, 2 = max long), run "
          f"at each frame's OWN scale (H4 1.0, H1 4.0, D1 1/6 — the D=6 trap).")
    print(f"     It BEAT the TradingView rating at H4 and D1 (§3cm). M15 has no champion: the "
          f"equivalent scale spans 24,576 bars, so it is not an M15 model.")

    # ---------------- 2. THE COST FLOOR -----------------------------------------------------
    rt_bp = spread_bp * 1.5
    print(f"\n  2. THE COST FLOOR — what this trade must clear before it is worth opening")
    print(f"     round trip ≈ {rt_bp:.2f}bp (one-way = 0.75 x spread). "
          f"On ${px:,.0f} gold that is ${rt_bp/1e4*px:.2f} per ounce.")
    atr = ref["atr"]
    stop_px = a.stop_atr * atr
    print(f"     a {a.stop_atr:.1f}-ATR stop is ${stop_px:.2f} = {stop_px/px*1e4:.0f}bp. "
          f"Cost is {100*rt_bp/(stop_px/px*1e4):.1f}% of your stop distance.")
    for rr in (1.0, 2.0, 3.0):
        need = (1.0 + rt_bp / (stop_px / px * 1e4)) / (1.0 + rr)
        print(f"     at {rr:.0f}:1 reward:risk you need a {need:.1%} hit rate to break even "
              f"{'<- and gold rises ~52% of days anyway' if rr == 1.0 else ''}")

    # ---------------- 3. THE SIZE ------------------------------------------------------------
    risk_usd = ai.equity * a.risk_pct / 100.0
    per_lot = stop_px * float(si.trade_contract_size)
    lots = risk_usd / per_lot if per_lot > 0 else 0.0
    step = float(si.volume_step or 0.01)
    lots_r = max(float(si.volume_min), round(lots / step) * step)
    print(f"\n  3. THE SIZE — computed, never felt")
    print(f"     risking {a.risk_pct:.2f}% of ${ai.equity:,.2f} = ${risk_usd:,.2f}")
    print(f"     stop ${stop_px:.2f} x contract {si.trade_contract_size:g} = "
          f"${per_lot:,.2f} risked per lot")
    print(f"     -> {lots_r:.2f} lots   (raw {lots:.3f}, min {si.volume_min:g}, step {step:g})")
    if lots_r > lots * 1.5 and lots > 0:
        print(f"     ⚠️  the minimum lot forces {lots_r/lots:.1f}x your intended risk — "
              f"${lots_r*per_lot:,.2f}, i.e. {100*lots_r*per_lot/ai.equity:.2f}% of equity")
    print(f"     `xau-riskfrac-sizing-table`: Sharpe is INVARIANT from 0.25% to 4% — size is a "
          f"pure risk dial, never a view. Under a 10% wall use 0.5% and never above 1.5%.")

    # ---------------- 4. THE EXIT, at validated settings --------------------------------------
    print(f"\n  4. THE EXIT — §3bn's measured shape, not a preference")
    print(f"     NO take-profit. §3bl: a fixed bracket costs even a PERFECT-FORESIGHT oracle "
          f"-0.94 to -1.43bp vs holding, because barriers truncate the moves that pay for losers.")
    print(f"     trail at {VALIDATED_TRAIL_ATR:.1f} ATR = ${VALIDATED_TRAIL_ATR*atr:.2f}, "
          f"RATCHETING (never loosening). Worth +0.077 Sharpe over a fixed SL/TP, "
          f"+0.263 over signal-flip. The curve peaks AT 3.0 and decays smoothly both sides.")
    print(f"     put the stop SERVER-SIDE on the order so a dead bot or a closed laptop "
          f"cannot orphan the position.")

    # ---------------- 5. THE WARNINGS THIS ACCOUNT HAS EARNED ----------------------------------
    print(f"\n  5. BEFORE YOU CLICK")
    warn = []
    if a.side == "short":
        warn.append("YOU ARE CONSIDERING A SHORT. The short leg has been refuted SIX times here, "
                    "most recently §3bn on the trail engine itself (long-only 1.091 vs "
                    "long/short 0.954, BOTH halves negative). Gold's drift is the reason: "
                    "excess(LONG) + excess(SHORT) = -2 x base.")
    if a.side == "long" and ref["rating"] <= -0.1:
        warn.append(f"you are considering a LONG while the tape reads {label(ref['rating'])} "
                    f"({ref['rating']:+.3f}). That is a discretionary override — fine, but name "
                    f"it as one.")
    if a.side == "short" and ref["rating"] >= 0.1:
        warn.append(f"you are considering a SHORT while the tape reads {label(ref['rating'])}.")
    warn.append("brackets in ATR, NEVER percent. 2026 daily range is ~$125 (2.73%), 8x 2015 — "
                "a 0.50% stop is 0.18 of ONE day's range and will be taken out by noise.")
    warn.append("hand-trading one of these accounts once drove equity -37%, which made the bot "
                "guard FLATTEN the book and latch a PERMANENT halt while it sat at +25.76%. "
                "If a bot runs on this login, your manual trade can kill it.")
    for i, w in enumerate(warn, 1):
        print(f"     {i}. {w}")
    print(f"\n     This tool describes and sizes. It does not forecast, and it has placed "
          f"no orders — it holds no order functions at all.")
    m.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
