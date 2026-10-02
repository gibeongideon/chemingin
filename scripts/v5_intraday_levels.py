"""Short-form intraday action email: yesterday's levels, as a two-sided OCO, for XAUUSD.

WHAT THIS IS, STATED PLAINLY AT THE TOP BECAUSE IT MATTERS. §3bj measured this exact mechanic
honestly on M15, where the first break is OBSERVABLE rather than assumed:

    enter at the prior day's high or low, whichever breaks first, exit at the day's close
    2,430 trades, +2.06bp net each, hit 48.7%, **Sharpe +0.47**, maxDD -17.7%

That is a real but THIN edge, and it is well below the champion book's **1.18**. It is a
measurement, not a recommendation. The NR7 ("narrowest range in 7 days") filter does NOT improve
it — 315 NR7 trades at Sharpe +0.43 against the non-NR7 control's +0.47, difference t +0.76 — so
the card flags NR7 days as a note only, never as a reason to size up.

The same §3bj run also showed the daily-bar version of this inflating to Sharpe +8.23 through
three separate bugs, so the numbers quoted here are the M15 ones and the card says which.

WHY A TWO-SIDED OCO AND NOT A DIRECTION. §3bi closed the directional claim: across 51 assets the
break-AGAINST-trend variant (+25.31bp) beat break-WITH-trend (+21.66bp), so nothing in this family
picks the side. The honest instruction is therefore both orders with the loser cancelled, which is
what was measured.

NO ORDER PATH. This file prints and emails. `tests/test_v5_xau_advisor.py`-style AST checks are not
duplicated here, but the module imports nothing that can trade.

    python scripts/v5_intraday_levels.py --dry
    python scripts/v5_intraday_levels.py --port 18814
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_notify import channel, env as notify_env, notify  # noqa: E402

SEEN = ROOT / "data" / "v5_runs" / "intraday_levels_seen.json"
SYMBOL, TF_D1 = "XAUUSD", 16408
# measured in §3bj on M15, first break observed:
MEASURED = dict(n=2430, net_bp=2.06, hit=48.7, sharpe=0.47, maxdd=-17.7,
                nr7_n=315, nr7_sharpe=0.43, nr7_t=0.76)


def fetch_daily(mt5, n: int = 40) -> pd.DataFrame:
    mt5.symbol_select(SYMBOL, True)
    r = mt5.copy_rates_from_pos(SYMBOL, TF_D1, 0, n)
    if r is None or len(r) < 10:
        raise SystemExit(f"no D1 bars for {SYMBOL}: {mt5.last_error()}")
    d = pd.DataFrame([(x[0], x[1], x[2], x[3], x[4]) for x in r],
                     columns=["ts", "open", "high", "low", "close"])
    d["time"] = pd.to_datetime(d["ts"], unit="s")
    return d.drop(columns=["ts"]).set_index("time").sort_index()


def build(d: pd.DataFrame, tick, now: datetime) -> dict:
    """The card. `d` ends with the most recent CLOSED daily bar."""
    prev = d.iloc[-1]
    hi, lo = float(prev["high"]), float(prev["low"])
    rng = hi - lo
    last7 = d["high"].tail(7) - d["low"].tail(7)
    is_nr7 = bool(rng <= last7.min() + 1e-9)
    px = float(getattr(tick, "bid", 0.0) or prev["close"])
    inside = lo < px < hi
    M = MEASURED
    L = [
        f"XAUUSD — levels from {d.index[-1]:%a %d %b}",
        "",
        f"  BUY STOP   {hi:,.2f}",
        f"  SELL STOP  {lo:,.2f}",
        f"  range      {rng:,.2f}   now {px:,.2f}"
        + ("  (inside)" if inside else "  (ALREADY OUTSIDE — see note)"),
        "",
        "  First fill wins. CANCEL THE OTHER immediately.",
        "  Exit at 23:00 UTC regardless. No stop, no target.",
    ]
    if not inside:
        L += ["",
              "  NOTE: price is already beyond a level, so today's first break has happened",
              "  or is not available. Skip today rather than chase — the measurement enters",
              "  AT the level, and entering late is a different trade."]
    if is_nr7:
        L += ["",
              f"  (yesterday was an NR7 day — narrowest range in 7. This does NOT improve the",
              f"   edge: {M['nr7_n']} NR7 trades at Sharpe {M['nr7_sharpe']:+.2f} vs the non-NR7",
              f"   control's {M['sharpe']:+.2f}, difference t {M['nr7_t']:+.2f}. Do not size up.)"]
    L += [
        "",
        "-" * 58,
        f"MEASURED (V5_FINDINGS 3bj, M15, first break observed, not assumed):",
        f"  {M['n']:,} trades   {M['net_bp']:+.2f}bp net each   hit {M['hit']:.1f}%",
        f"  Sharpe {M['sharpe']:+.2f}   maxDD {M['maxdd']:.1f}%",
        "",
        "  This is THIN and is below the champion book's 1.18. It is a measurement,",
        "  not a recommendation, and it is NOT what the automated book trades.",
        "  This mail places no orders.",
    ]
    subj = (f"[xau-intraday] BUY>{hi:,.0f} / SELL<{lo:,.0f}"
            + ("  (NR7)" if is_nr7 else "")
            + ("" if inside else "  [already outside — skip]"))
    return dict(subject=subj, body="\n".join(L), hi=hi, lo=lo, nr7=is_nr7,
                inside=inside, day=str(d.index[-1].date()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18814)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--always", action="store_true")
    a = ap.parse_args()

    from mt5linux import MetaTrader5
    mt5 = MetaTrader5(host="localhost", port=a.port)
    if not mt5.initialize():
        raise SystemExit(f"bridge {a.port} init failed: {mt5.last_error()}")
    d = fetch_daily(mt5)
    mt5.symbol_select(SYMBOL, True)
    tick = mt5.symbol_info_tick(SYMBOL)
    mt5.shutdown()

    now = datetime.now(timezone.utc)
    # the newest D1 bar is TODAY's in progress; the levels come from the last CLOSED one
    if d.index[-1].date() >= now.date():
        d = d.iloc[:-1]
    card = build(d, tick, now)
    print(card["body"])

    try:
        seen = json.loads(SEEN.read_text())
    except Exception:
        seen = {}
    due = seen.get("day") != card["day"]
    print(f"\ntrigger: {'send' if (due or a.always) else 'already sent for ' + card['day']}")
    if a.dry:
        return
    if due or a.always:
        print(f"channel: {channel(notify_env())}")
        if notify(card["subject"], card["body"]) is not False:
            SEEN.parent.mkdir(parents=True, exist_ok=True)
            SEEN.write_text(json.dumps({**seen, "day": card["day"],
                                        "hi": card["hi"], "lo": card["lo"]}, indent=2))


if __name__ == "__main__":
    main()
