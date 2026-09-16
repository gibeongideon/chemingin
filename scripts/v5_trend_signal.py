"""TREND-FOLLOWER signal — the CHAMPION book, emailed when the target position MOVES.

WHY THIS IS THE ONE WORTH READING. Unlike the ZigZag harness (§3x: walk-forward SR -0.76,
negative expectancy, demo-only), the champion trend follower is this project's actual result:
long-only EWMAC/breakout blend, net book Sharpe ~1.21 on the gold-tilted core-4, and the only
family here that has survived every control. `v5_zigzag_*` is an observation harness; THIS is
the strategy.

WHAT IT SENDS AND WHEN. The champion is a CONTINUOUS-SIZING follower, not an entry/exit system,
so "a signal" is not a fire — it is a change in TARGET position. This mails when any sleeve's
target moves materially, and stays quiet otherwise:

    |new target - last notified target| > max(MIN_LOTS, BAND x last target)

Dedupe is against the LAST NOTIFIED TARGET, not against what is held. That matters: nothing
places these orders automatically on this account, so comparing against holdings would re-send
the same "you are not in position" message every run forever.

NO DUPLICATION. Sleeve definitions, bar clocks, speed scales, the vol dial and the sizing
formula are all imported from `v5_signals_digest`, so the daily digest and this alert can never
disagree about what the champion wants — the same reason both zigzag senders share
`v5_zigzag_message`.

HONEST CAVEATS THAT BELONG IN THE EMAIL, NOT A FOOTNOTE:
  * Nothing is auto-traded on this account. These are targets for a HUMAN to place.
  * Gold's carry at FTMO is roughly -3.7%/yr, which is why the live book prefers a swap-free
    instrument (§3ag). The forecast does not know about financing.
  * A missed run is harmless: this is a multi-week follower and the next run reconciles to the
    new target. Do NOT double up to "catch up".

    python scripts/v5_trend_signal.py --dry        # print, never send
    python scripts/v5_trend_signal.py             # send only if a target moved
    python scripts/v5_trend_signal.py --always     # send a snapshot regardless
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_notify import channel, env, notify  # noqa: E402
from scripts.v5_signals_digest import (  # noqa: E402
    BOOK, TILT, pct_yr_carry, rates, target_leverage,
)
from src.v5.xau_dual_signals import champion_signal  # noqa: E402
from scripts.v5_xau_champion_lifts import champion_recipe  # noqa: E402

SEEN = ROOT / "data" / "v5_runs" / "trend_signal_seen.json"
STATE = ROOT / "data" / "v5_runs" / "trend_signal_state.json"
BAND = 0.10        # 10% of the last notified target, matching the digest's BUFFER
MIN_LOTS = 0.02    # and never chatter below a broker lot step
STALE_H = 30       # XAU is H4 and the others D1, so a day-plus of silence is the red line


def compute(mt5, eq: float) -> dict:
    """Per-sleeve champion forecast, target leverage and target lots. Same maths as the digest."""
    out = {}
    for name, (sym, tf, scale, hl, ann) in BOOK.items():
        df = rates(mt5, sym, tf, 6000 if tf != "D1" else 3000)
        if df.empty:
            out[name] = dict(symbol=sym, error="no data")
            continue
        c = df["close"]
        fc = champion_signal(c) if scale == 1.0 else champion_recipe(c, scale, 1.5)
        lev = target_leverage(c, fc, ann, hl)
        info = mt5.symbol_info(sym)
        px = float(c.iloc[-1])
        step = info.volume_step or 0.01
        lots = lev * TILT[name] * eq / (px * info.trade_contract_size)
        hours = {"M15": 0.25, "H1": 1, "H4": 4, "D1": 24}[tf]
        import pandas as pd
        stale = ((pd.Timestamp.now(tz="UTC").tz_localize(None) - df.index[-1])
                 .total_seconds() / 3600 - hours)
        held = sum(p.volume * (1 if p.type == 0 else -1)
                   for p in (mt5.positions_get(symbol=sym) or []))
        out[name] = dict(symbol=sym, tf=tf, forecast=round(float(fc.iloc[-1]), 3),
                         target_lev=round(lev, 3), weight=TILT[name],
                         target_lots=round(round(lots / step) * step, 2),
                         held_lots=round(float(held), 2),
                         price=round(px, 2), carry_pct_yr=round(pct_yr_carry(info, px), 2),
                         last_bar=str(df.index[-1]), stale_hours=round(max(stale, 0.0), 1))
    return out


def moved(cur: dict, prev: dict) -> list:
    """Sleeves whose TARGET has moved materially since the last notification."""
    out = []
    for name, d in cur.items():
        if d.get("error"):
            continue
        new = d["target_lots"]
        old = prev.get(name, {}).get("target_lots")
        if old is None:
            out.append((name, None, new)); continue
        thr = max(MIN_LOTS, BAND * abs(old))
        if abs(new - old) > thr:
            out.append((name, old, new))
    return out


def body_for(cur: dict, changes: list, eq: float, login, server) -> tuple:
    L = [f"CHAMPION TREND FOLLOWER — target positions",
         f"account {login} @ {server}   equity ${eq:,.2f}", ""]
    L.append(f"{'sleeve':7s} {'symbol':12s} {'fc':>6s} {'target':>8s} {'held':>7s} "
             f"{'to do':>8s} {'carry%/yr':>10s}")
    for name, d in cur.items():
        if d.get("error"):
            L.append(f"{name:7s} {d['symbol']:12s}  {d['error']}")
            continue
        todo = round(d["target_lots"] - d["held_lots"], 2)
        L.append(f"{name:7s} {d['symbol']:12s} {d['forecast']:6.2f} "
                 f"{d['target_lots']:8.2f} {d['held_lots']:7.2f} {todo:+8.2f} "
                 f"{d['carry_pct_yr']:+10.2f}")
    if changes:
        L += ["", "TARGET MOVED:"]
        for name, old, new in changes:
            was = "first run" if old is None else f"{old:+.2f}"
            L.append(f"  {name}: {was} -> {new:+.2f} lots")
    stale = [f"{n} {d['stale_hours']:.0f}h" for n, d in cur.items()
             if not d.get("error") and d["stale_hours"] > STALE_H]
    if stale:
        L += ["", f"** STALE FEED: {', '.join(stale)} — do not act on those **"]
    L += ["", "Nothing is auto-traded on this account. These are targets for you to place.",
          "'fc' is 0 = flat to 2 = max long. The forecast ignores financing; gold's carry",
          "here is shown above and is why the live book prefers a swap-free instrument.",
          "Missed a run? Harmless — this is a multi-week follower and the next run reconciles",
          "to the new target. Do NOT double up to catch up.",
          "",
          "Champion trend is the TRADABLE strategy (net book Sharpe ~1.21), unlike the ZigZag",
          "harness, which is a disproven observation-only detector."]
    if changes:
        n0, o0, nn = changes[0]
        subj = (f"[trend] {n0} target {nn:+.2f} lots"
                + (f" (was {o0:+.2f})" if o0 is not None else "")
                + (f" +{len(changes)-1} more" if len(changes) > 1 else ""))
    else:
        fcs = " ".join(f"{n} {d['forecast']:.2f}" for n, d in cur.items()
                       if not d.get("error"))
        subj = f"[trend] no change — {fcs}"
    return subj, "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18814)
    ap.add_argument("--always", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    print(f"channel: {channel(env())}")
    from mt5linux import MetaTrader5
    mt5 = MetaTrader5(host="localhost", port=a.port)
    if not mt5.initialize():
        raise SystemExit(f"bridge {a.port} init failed: {mt5.last_error()}")
    ai = mt5.account_info()
    eq = float(ai.equity)
    cur = compute(mt5, eq)
    mt5.shutdown()

    prev = {}
    try:
        prev = json.loads(SEEN.read_text()).get("sleeves", {})
    except Exception:
        pass
    changes = moved(cur, prev)
    subj, body = body_for(cur, changes, eq, ai.login, ai.server)
    print(f"\n{subj}\n{'-'*64}\n{body}")

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(dict(
        computed_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        login=ai.login, server=ai.server, equity=eq, sleeves=cur,
        changes=[dict(sleeve=n, was=o, now=v) for n, o, v in changes],
        subject=subj, body=body), indent=2))

    if a.dry:
        return
    if changes or a.always:
        if notify(subj, body):
            # only advance the baseline on a SUCCESSFUL send, so a failed delivery
            # re-sends next run instead of silently swallowing the change
            SEEN.write_text(json.dumps(dict(
                at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                sleeves=cur)))
        else:
            print("send FAILED — baseline NOT advanced, will retry next run")
            sys.exit(1)
    else:
        print("\n(no target moved -> nothing sent; --always for a snapshot)")


if __name__ == "__main__":
    main()
