"""EMAIL ALERT for manual execution — mails you ONLY when a hand-trade is actually due.

Companion to `v5_manual_ticket.py`, meant to run on the VPS on a daily timer where the Maven
bridge and `.env.mail` both live. It reads prices and positions, computes what the champion
wants, and emails an instruction for a HUMAN to place. It has no order-sending code path.

THREE DESIGN POINTS WORTH KNOWING:

1. SIGNAL FROM XAUUSD, TRADE ON GoldEternal. GoldEternal launched ~2026-03-13 and has only a
   few hundred bars — far short of the champion's slowest lookback (EWMAC 64/256 x 6 = up to
   1,536 H4 bars). XAUUSD has years. The two correlate at 0.998 daily with a flat basis
   (V5_FINDINGS §3ag), so the signal is computed on XAUUSD and executed on the swap-free
   instrument. If that basis ever drifts, this assumption breaks — it is re-checked monthly.

2. THE SWAP IS RE-CHECKED EVERY RUN, because the whole reason to trade GoldEternal instead of
   XAUUSD is that it charges no financing — worth +0.20 Sharpe and +3.2pp CAGR (§3bd). That rests
   on one broker field, `swap_mode = DISABLED`, which a broker can change without notice. §3bd's
   breakeven is -3.62%/yr: the edge survives almost any hidden basis drift but NOT the swap being
   switched on. So every ticket carries the live reading, and if it ever changes the alert says
   STOP rather than quietly continuing to trade a thesis that has expired.

3. RECONCILE AGAINST WHAT YOU ACTUALLY HOLD, not against a simulated path. The backtest walks a
   buffered path bar by bar; live, the honest equivalent is to compare the raw target to your
   REAL position and act only if the gap exceeds the band. That is what the automated executor
   does, it is path-independent (so a different history length cannot change today's answer),
   and it self-corrects if you skip a day or fill at a different size than instructed.

    python scripts/v5_manual_alert.py --dry              # print, never email
    python scripts/v5_manual_alert.py --dial 0.20        # email only if action due
    python scripts/v5_manual_alert.py --always           # email even when nothing to do
"""
from __future__ import annotations

import argparse
import smtplib
import ssl
import sys
import time
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v5.xau_dual_signals import champion_signal  # noqa: E402  (canonical, do not vendor)
from scripts.v5_notify import notify, channel, env as notify_env  # noqa: E402

TF_H4 = 16388
TARGET_VOL, MAX_LEV = 0.10, 8.0
ANN, VOL_HL = 252 * 6, 42
BUFFER = 0.20             # widened for manual cadence — V5_FINDINGS §3ai
CONTRACT, LOT_STEP = 100.0, 0.01
SIGNAL_SYMBOL = "XAUUSD"          # long history
TRADE_SYMBOL = "GoldEternal"      # swap-free, what you actually buy
MAX_STALE_H = 12


def fetch(mt5, symbol: str, n: int = 6000) -> pd.DataFrame:
    """H4 bars, CLOSED ONLY. `copy_rates_from_pos(..., 0, n)` returns the bar currently in
    progress at position 0; feeding that to the signal would evaluate a forecast on a bar that
    can still change, which the backtest never does. Anything whose 4h window has not elapsed
    is dropped."""
    mt5.symbol_select(symbol, True)
    for _ in range(6):
        r = mt5.copy_rates_from_pos(symbol, TF_H4, 0, n)
        if r is not None and len(r) > 500:
            a = np.array([(x[0], x[1], x[2], x[3], x[4]) for x in r], dtype=float)
            df = pd.DataFrame(a[:, 1:], columns=["open", "high", "low", "close"],
                              index=pd.to_datetime(a[:, 0], unit="s"))
            now = pd.Timestamp.now(tz="UTC").tz_localize(None)
            open_bars = df.index[df.index + pd.Timedelta(hours=4) > now]
            if len(open_bars):
                df = df.drop(index=open_bars)
            return df
        time.sleep(2)
    raise SystemExit(f"could not fetch {symbol} H4: {mt5.last_error()}")


def send(subj: str, body: str) -> bool:
    """Delegates to the multi-channel notifier, which picks a transport that actually works
    from this host (the VPS has outbound SMTP blocked on every usual port — see
    scripts/v5_notify.py). Failure is reported, never raised: a notifier that crashes takes
    its own timer down with it."""
    if notify(subj, body):
        return True
    print(f"[alert] delivery failed via channel '{channel(notify_env())}' — the ticket JSON "
          f"records delivered=false, so the laptop watchdog will send it instead.")
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18812)
    ap.add_argument("--dial", type=float, default=0.10, help="vol dial, e.g. 0.20")
    ap.add_argument("--balance", type=float, default=None, help="override live equity")
    ap.add_argument("--held", type=float, default=None, help="override live position")
    ap.add_argument("--dry", action="store_true", help="print only, never email")
    ap.add_argument("--always", action="store_true", help="email even when no action due")
    ap.add_argument("--json-out", default=None,
                    help="write the ticket here for a relay host to mail (see deploy/)")
    args = ap.parse_args()

    from mt5linux import MetaTrader5
    mt5 = MetaTrader5(host="localhost", port=args.port)
    if not mt5.initialize():
        raise SystemExit(f"bridge {args.port} init failed: {mt5.last_error()}")
    ai = mt5.account_info()
    eq = args.balance if args.balance is not None else float(ai.equity)

    df = fetch(mt5, SIGNAL_SYMBOL)
    mt5.symbol_select(TRADE_SYMBOL, True)
    time.sleep(1)
    ti = mt5.symbol_info(TRADE_SYMBOL)
    tick = mt5.symbol_info_tick(TRADE_SYMBOL)
    # `symbol_info` can transiently return None over the bridge, and `ti.bid` then raises and
    # takes the whole ticket down. Every read of it is guarded.
    trade_px = (getattr(tick, "bid", 0.0) or getattr(ti, "bid", 0.0)
                or float(df["close"].iloc[-1]))
    # *** THE MARKET-OPEN CHECK MUST READ THE SIGNAL SYMBOL, NOT THE TRADED ONE. ***
    # GoldEternal is a true perpetual: measured 2026-09-20, it printed 648 Saturday and 637
    # Sunday H1 bars against XAUUSD's 0 and 191, and was ticking 0.4 minutes ago on a Sunday
    # while XAUUSD had been shut for 40 hours. Reading the traded symbol therefore reports
    # "market open" all weekend and then complains that the signal is 37h stale, which is what
    # this code did on its first run. What can go stale is the SIGNAL, and the signal comes
    # from XAUUSD.
    sig_tick = mt5.symbol_info_tick(SIGNAL_SYMBOL)
    sig_age_min = ((time.time() - float(getattr(sig_tick, "time", 0) or 0)) / 60.0
                   if sig_tick is not None else float("inf"))
    market_closed = sig_age_min > 10
    trade_tick_age_min = ((time.time() - float(getattr(tick, "time", 0) or 0)) / 60.0
                          if tick is not None else float("inf"))
    tradeable_now = trade_tick_age_min <= 10
    # The swap thesis, re-read live every run (see design point 2).
    swap_ok = (getattr(ti, "swap_mode", None) == 0)
    swap_note = (f"swap_mode={getattr(ti, 'swap_mode', '?')} "
                 f"long={getattr(ti, 'swap_long', '?')} short={getattr(ti, 'swap_short', '?')}")
    if args.held is not None:
        held = args.held
    else:
        ps = mt5.positions_get(symbol=TRADE_SYMBOL) or []
        held = float(sum((p.volume if p.type == 0 else -p.volume) for p in ps))
    mt5.shutdown()

    close = df["close"]
    ret = close.pct_change()
    vol = ret.ewm(halflife=VOL_HL, min_periods=20).std() * np.sqrt(ANN)
    fc = champion_signal(close)
    f_now, v_now = float(fc.iloc[-1]), float(vol.iloc[-1])
    lev = float(np.clip(f_now, -2, 2) * (TARGET_VOL / v_now))
    lev = float(np.clip(lev, -MAX_LEV, MAX_LEV)) * (args.dial / TARGET_VOL)
    band = BUFFER * min(TARGET_VOL / v_now, MAX_LEV) * (args.dial / TARGET_VOL)

    per_lot = trade_px * CONTRACT
    held_lev = held * per_lot / eq
    gap = lev - held_lev
    if abs(gap) > band:
        tgt_lev = lev - np.sign(gap) * band
    else:
        tgt_lev = held_lev
    tgt_lots = round(tgt_lev * eq / per_lot / LOT_STEP) * LOT_STEP
    delta = round((tgt_lots - held) / LOT_STEP) * LOT_STEP
    due = abs(delta) >= LOT_STEP

    last = df.index[-1]
    stale = (datetime.now(timezone.utc)
             - (last + pd.Timedelta(hours=4)).tz_localize("UTC")).total_seconds() / 3600
    verb = "BUY" if delta > 0 else "SELL"
    side_word = "LONG" if tgt_lots > 0 else ("SHORT" if tgt_lots < 0 else "FLAT")
    L = [
        f"account        {ai.login} @ {ai.server}   equity ${eq:,.2f}",
        f"signal from    {SIGNAL_SYMBOL} H4, {len(df)} closed bars, last close "
        f"{last + pd.Timedelta(hours=4):%Y-%m-%d %H:%M} UTC ({stale:.1f}h ago)",
        f"trading        {TRADE_SYMBOL} @ ${trade_px:,.2f}   (1 lot = ${per_lot:,.0f})"
        + ("   [quoting now]" if tradeable_now else "   [NOT quoting]"),
        "",
        f"forecast       {f_now:.3f}   (0 = flat, 2 = max long)",
        f"ann vol        {v_now:.1%}   dial {args.dial:.0%}",
        f"target lev     {lev:.3f}   no-trade band +/-{band:.3f}",
        f"held           {held:+.2f} lots  ({held_lev:.3f} lev)",
        "",
        (f"ACTION         {verb} {abs(delta):.2f} lots   ({held:+.2f} -> {tgt_lots:+.2f})"
         if due else
         f"ACTION         none. gap {gap:+.3f} lev is inside the band."),
        "",
    ]
    if due:
        L += [
            "-" * 64,
            "EXACTLY WHAT TO PLACE — copy these fields into the MT5 order window:",
            "-" * 64,
            f"   Symbol       {TRADE_SYMBOL}",
            f"   Type         Market Execution",
            f"   Side         {verb}",
            f"   Volume       {abs(delta):.2f}",
            f"   Stop Loss    (leave blank)",
            f"   Take Profit  (leave blank)",
            "",
            f"   after filling you should be {side_word} {abs(tgt_lots):.2f} lots in total",
            f"   (you are {held:+.2f} now; this order moves you by {delta:+.2f})",
            "",
            "   No SL/TP by design: the champion exits by SIZING DOWN as its forecast decays,",
            "   not by being stopped out. A stop would convert a trend follower into a",
            "   different strategy that has never been measured here.",
            "",
        ]
    L += [
        "This mail places no orders. You place them.",
        "Missed a day? Harmless - this is a multi-week trend follower. Do NOT trade twice",
        "to catch up; the next run reconciles against whatever you actually hold.",
    ]
    # Staleness only means something while the market is OPEN. Gold prints no bars between
    # Friday ~21:00 and Sunday ~22:00 UTC, so a flat wall-clock rule cries stale every weekend
    # and a genuinely dead feed then looks identical to a normal Saturday. Same false alarm that
    # had book-ftmo.service failing every weekend pass until 2026-09-19.
    if market_closed:
        L.insert(0, f"({SIGNAL_SYMBOL} is CLOSED - last tick {sig_age_min/60:.0f}h ago, so the "
                    f"signal cannot change until it reopens."
                    + (f" {TRADE_SYMBOL} IS still quoting, so you CAN place this whenever suits "
                       f"you." if tradeable_now else
                       f" {TRADE_SYMBOL} is not quoting either.")
                    + " This is the standing instruction, not a fresh one.)")
    elif stale > MAX_STALE_H:
        L.insert(0, f"** WARNING: price data is {stale:.0f}h stale while the market is OPEN - "
                    f"do not act on this **")
    if not swap_ok:
        L.insert(0, "")
        L.insert(0, f"    live reading: {swap_note}   (expected swap_mode=0 / DISABLED)")
        L.insert(0, "    The ENTIRE reason to hold GoldEternal rather than XAUUSD is that it "
                    "charges no")
        L.insert(0, "    financing. Breakeven is -3.62%/yr (V5_FINDINGS 3bd). Verify with the "
                    "broker before")
        L.insert(0, "    placing anything, and switch the sleeve back to XAUUSD if the swap is "
                    "real.")
        L.insert(0, "*** STOP: GoldEternal NO LONGER LOOKS SWAP-FREE ***")
    else:
        L.append(f"swap check     OK - {TRADE_SYMBOL} {swap_note}")
        L.append("               (the edge dies if this ever stops saying swap_mode=0; "
                 "3bd breakeven -3.62%/yr)")
    body = "\n".join(L)
    print(body)

    delivered = None
    if not args.dry and (due or args.always or not swap_ok):
        if not swap_ok:
            subj = f"[XAU manual] STOP - {TRADE_SYMBOL} may no longer be swap-free"
        elif due:
            subj = f"[XAU manual] {verb} {abs(delta):.2f} lots {TRADE_SYMBOL} -> {side_word} {abs(tgt_lots):.2f}"
        else:
            subj = f"[XAU manual] no action (fc {f_now:.2f}, hold {held:+.2f})"
        delivered = send(subj, body)

    if args.json_out:
        import json
        Path(args.json_out).write_text(json.dumps(dict(
            computed_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            login=ai.login, server=ai.server, equity=eq,
            signal_symbol=SIGNAL_SYMBOL, trade_symbol=TRADE_SYMBOL,
            bars=len(df), last_close=str(last + pd.Timedelta(hours=4)),
            stale_hours=round(stale, 2), forecast=round(f_now, 4),
            ann_vol=round(v_now, 4), dial=args.dial, target_lev=round(lev, 4),
            band=round(band, 4), held_lots=held, target_lots=tgt_lots,
            delta_lots=delta, action=("BUY" if delta > 0 else "SELL") if due else "NONE",
            due=bool(due), delivered=delivered, body=body), indent=2))
        print(f"\nwrote {args.json_out}")

    if args.dry:
        return
    if delivered is True:
        print("\ndelivered from this host")
    elif delivered is False:
        print("\nNOT delivered from this host — laptop watchdog will relay it")
    else:
        print("\n(no action due -> nothing sent)")


if __name__ == "__main__":
    main()
