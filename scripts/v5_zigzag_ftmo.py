"""ZIGZAG bottom-detector OBSERVATION HARNESS — FTMO demo, bridge 18814.

*** THIS STRATEGY HAS NEGATIVE EXPECTANCY AND IS NOT A RECOMMENDATION. ***
V5_FINDINGS §3x: all 48 grid cells negative full-sample (best SR -0.44), walk-forward
**SR -0.76 / DD -35.7%** against buy-and-hold's +0.79, paired t **-3.72**, **0 of 8 years
better**, DSR **0.000**. Deployed at the user's explicit request to watch a documented
detector operate live on a DEMO account ("I just want to see it in action, no profit needed").
Sized small for that reason. Do not point this at a funded account.

WHAT IT TRADES. The walk-forward-selected cell from §3x, not the full-sample best:
    threshold 0.60  ·  TP +0.60%  ·  SL -0.50%  ·  max hold 48 H1 bars
(2023/24/25 all selected 0.6/0.6%/0.5%; 2026 selected the 48-bar hold. The full-sample best
was 0.5/0.6%/0.5%/48 at SR -0.441 and is deliberately not used.)

DESIGN CHOICES THAT MATTER
  * Rates come from the BROKER'S OWN FEED via the bridge, never from the CSVs. The H1 CSV is
    currently 2,063 hours stale, and two of this repo's data files are known to be different
    feeds from the same instrument (differing by $405 and $177 at the extremes). One feed, no
    splice.
  * TP and SL are attached to the order SERVER-SIDE, so the two profitable exits do not depend
    on this process being alive. Only the max-hold exit needs the bot, which is why the timer
    runs hourly.
  * Double safety lock: `--live --execute` both required to send anything. Default is a dry
    plan. `--execute` is ignored on a non-demo account unless `--live` is also passed.
  * Writes state to `data/v5_runs/zigzag_ftmo_state.json` for the notifier, so the manual
    channel and the executor never disagree about what fired.

    python scripts/v5_zigzag_ftmo.py                      # dry plan
    python scripts/v5_zigzag_ftmo.py --live --execute      # actually trade the demo
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_zigzag_signal import bottom_probability, TOL  # noqa: E402

CONFIG = ROOT / "configs" / "v5_zigzag_ftmo.json"
STATE = ROOT / "data" / "v5_runs" / "zigzag_ftmo_state.json"
TF_H1 = 16385
FETCH_BARS = 30_000


def fetch_h1_closed(mt5, symbol: str) -> pd.DataFrame:
    """Broker's own H1 history, CLOSED BARS ONLY. `copy_rates_from_pos(..,0,n)` returns the
    in-progress bar at position 0; scoring an unfinished bar would evaluate a signal that can
    still change (the same bug caught in the manual-alert build)."""
    mt5.symbol_select(symbol, True)
    r = None
    for _ in range(6):
        r = mt5.copy_rates_from_pos(symbol, TF_H1, 0, FETCH_BARS)
        if r is not None and len(r) > 5000:
            break
        time.sleep(2)
    if r is None or len(r) < 5000:
        raise SystemExit(f"could not fetch enough {symbol} H1 history: {mt5.last_error()}")
    a = np.array([(x[0], x[1], x[2], x[3], x[4]) for x in r], dtype=float)
    df = pd.DataFrame(a[:, 1:], columns=["open", "high", "low", "close"],
                      index=pd.to_datetime(a[:, 0], unit="s"))
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    open_bars = df.index[df.index + pd.Timedelta(hours=1) > now]
    return df.drop(index=open_bars) if len(open_bars) else df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    port = args.port or cfg["port"]
    symbol, magic = cfg["symbol"], cfg["magic"]
    thr, tp_pct, sl_pct = cfg["threshold"], cfg["tp_pct"], cfg["sl_pct"]
    max_hold, risk_frac = cfg["max_hold_bars"], cfg["risk_frac"]

    from mt5linux import MetaTrader5
    mt5 = MetaTrader5(host="localhost", port=port)
    if not mt5.initialize():
        raise SystemExit(f"bridge {port} init failed: {mt5.last_error()}")
    ai = mt5.account_info()
    is_demo = "demo" in str(getattr(ai, "server", "")).lower() or getattr(ai, "trade_mode", 0) == 0
    print(f"account {ai.login} @ {ai.server}   equity ${ai.equity:,.2f}   "
          f"{'DEMO' if is_demo else '*** NOT FLAGGED DEMO ***'}")

    df = fetch_h1_closed(mt5, symbol)
    st = bottom_probability(df, thr, tp_pct, sl_pct, max_hold)
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    ask = getattr(tick, "ask", 0.0) or info.ask
    contract = info.trade_contract_size
    print(f"feed: {len(df):,} closed H1 bars, last {st.asof} ({st.stale_hours:.1f}h ago)")
    print(f"P(bottom) {st.prob:.4f}  threshold {thr:.2f}  -> "
          f"{'FIRES' if st.fires else 'no fire'}")

    held = [p for p in (mt5.positions_get(symbol=symbol) or []) if p.magic == magic]
    print(f"open under magic {magic}: {len(held)}")
    actions = []

    # ---- 1. max-hold exits (the only exit this process is responsible for) ----
    now = datetime.now(timezone.utc).timestamp()
    for p in held:
        age_h = (now - p.time) / 3600.0
        if age_h >= max_hold:
            actions.append(("CLOSE", p.ticket, p.volume,
                            f"max-hold {age_h:.0f}h >= {max_hold}h",
                            dict(entry=float(p.price_open), sl=None, tp=None,
                                 close_by=None, age_h=round(age_h, 1))))
        else:
            print(f"  ticket {p.ticket}: {p.volume} lots, age {age_h:.1f}h "
                  f"(closes at {max_hold}h; TP/SL are server-side)")

    # ---- 2. entry ----
    if st.fires and not held:
        if st.stale_hours > 3:
            print(f"  SKIP entry: feed is {st.stale_hours:.1f}h stale")
        else:
            risk_usd = ai.equity * risk_frac
            lots = risk_usd / (sl_pct * ask * contract)
            step = info.volume_step or 0.01
            lots = max(info.volume_min, round(lots / step) * step)
            sl = round(ask * (1 - sl_pct), info.digits)
            tp = round(ask * (1 + tp_pct), info.digits)
            close_by = (datetime.now(timezone.utc)
                        + pd.Timedelta(hours=max_hold)).strftime("%Y-%m-%d %H:%M UTC")
            actions.append(("BUY", None, round(lots, 2),
                            f"prob {st.prob:.3f} >= {thr}",
                            dict(entry=round(float(ask), info.digits), sl=sl, tp=tp,
                                 close_by=close_by, age_h=None)))
    elif st.fires and held:
        print("  fires, but already in a position — no pyramiding")

    if not actions:
        print("  ACTION: none")
    for kind, ticket, vol, why, px in actions:
        extra = (f" @ {px['entry']}  SL {px['sl']}  TP {px['tp']}"
                 if kind == "BUY" else "")
        print(f"  ACTION: {kind} {vol} lots{extra}  ({why})")

    sent = []
    if args.live and args.execute and actions:
        if not is_demo:
            print("  ** REFUSING: account is not flagged demo. "
                  "This harness is demo-only by design. **")
        else:
            for kind, ticket, vol, why, px in actions:
                if kind == "CLOSE":
                    pos = next((p for p in held if p.ticket == ticket), None)
                    req = dict(action=mt5.TRADE_ACTION_DEAL, symbol=symbol, volume=vol,
                               type=mt5.ORDER_TYPE_SELL, position=ticket,
                               price=getattr(tick, "bid", 0.0) or info.bid,
                               magic=magic, comment=cfg["run_id"][:31],
                               type_filling=mt5.ORDER_FILLING_IOC)
                else:
                    sl = round(ask * (1 - sl_pct), info.digits)
                    tp = round(ask * (1 + tp_pct), info.digits)
                    req = dict(action=mt5.TRADE_ACTION_DEAL, symbol=symbol, volume=vol,
                               type=mt5.ORDER_TYPE_BUY, price=ask, sl=sl, tp=tp,
                               magic=magic, comment=cfg["run_id"][:31],
                               type_filling=mt5.ORDER_FILLING_IOC)
                r = mt5.order_send(req)
                rc = getattr(r, "retcode", r)
                print(f"    sent {kind}: retcode {rc} "
                      f"{'OK' if rc == 10009 else getattr(r, 'comment', '')}")
                sent.append(dict(kind=kind, vol=vol, retcode=int(rc) if rc else None))
    elif actions:
        print("  (dry plan — rerun with --live --execute to send)")

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(dict(
        computed_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        login=ai.login, server=ai.server, equity=float(ai.equity), symbol=symbol,
        bars=len(df), last_bar=st.asof, stale_hours=st.stale_hours,
        prob=st.prob, threshold=thr, fires=st.fires,
        open_positions=len(held), actions=[dict(kind=k, vol=v, why=w, **px)
                                           for k, _, v, w, px in actions],
        sent=sent, executed=bool(args.live and args.execute and is_demo),
        warning="NEGATIVE EXPECTANCY — walk-forward SR -0.76, 0/8 years, DSR 0.000 (V5_FINDINGS 3x)"
    ), indent=2))
    print(f"\nstate -> {STATE.relative_to(ROOT)}")
    mt5.shutdown()


if __name__ == "__main__":
    main()
