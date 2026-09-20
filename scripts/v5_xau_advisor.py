"""Compute the XAUUSD advisor reading and advance its state. Writes state; sends nothing.

SEEDED FROM THE BROKER, NOT FROM A CSV. `data/XAUUSD_H4_long.csv` splices feeds (§3ba) and is
refreshed on someone else's schedule, so the service reads H4 straight from the bridge. The
shipped cell is the BASE arm, so H4 is all it needs — no M15, which removes the entire
feed-mismatch and M15-staleness class of failure from the live path.

A 5-MINUTE CADENCE CANNOT MOVE A 9-HOUR PROBABILITY, AND THE CODE SAYS SO. The model is called
once per closed H4 bar — 6 times a day, not 288 — and the reading is cached by
`(last_closed_h4, model_id)`. A 5-minute increment carries a tiny fraction of the variance that
resolves a 9-hour call, while the reliability bucket's own error bar is +/-0.03. The cadence buys
LATENCY OF DETECTION (dead feed, dead bridge, a state flip arriving within minutes of the close),
and every field in the message carries its own age so the cadence is never mistaken for freshness.

THE STATE IS STEPPED ONLY ON A NEW CLOSED BAR. `xau_advisor_state.step` counts transitions for
the dwell timer and the thrash guard, so re-stepping the same bar 288 times a day would let both
count events that did not happen.

    python scripts/v5_xau_advisor.py --port 18814
    python scripts/v5_xau_advisor.py --port 18814 --dry --print
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v5.xau_advisor_model import AdvisorUnavailable, advise, health_of, load  # noqa: E402
from src.v5.xau_advisor_state import (  # noqa: E402
    AdvisorState, Thresholds, describe, from_dict, observed_for, step, to_dict, utcnow,
)

STATE = ROOT / "data" / "v5_runs" / "xau_advisor_state.json"
CACHE = ROOT / "data" / "v5_runs" / "xau_advisor_cache.json"
TF_H4, TF_M5 = 16388, 5


def _atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def fetch_h4(mt5, symbol: str, n: int = 4000) -> pd.DataFrame:
    """Closed H4 bars from the broker. The in-progress bar is dropped, not merely ignored."""
    mt5.symbol_select(symbol, True)
    r = None
    for _ in range(6):
        r = mt5.copy_rates_from_pos(symbol, TF_H4, 0, n)
        if r is not None and len(r) >= 500:
            break
        time.sleep(2)
    if r is None or len(r) < 500:
        raise AdvisorUnavailable(f"bridge returned no usable H4 for {symbol}: {mt5.last_error()}")
    rows = [(x[0], x[1], x[2], x[3], x[4], x[5], x[6]) for x in r]
    d = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "tick_volume",
                                    "spread"])
    d["time"] = pd.to_datetime(d["ts"], unit="s")
    d = d.drop(columns=["ts"]).set_index("time").sort_index()
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return d.drop(index=d.index[d.index + pd.Timedelta(hours=4) > now], errors="ignore")


def market_panel(mt5, symbol: str) -> dict:
    """The only block that legitimately changes every 5 minutes."""
    t = mt5.symbol_info_tick(symbol)
    if t is None:
        return {"quote": "no tick", "closed": True}
    age = time.time() - float(t.time)
    # 10 minutes: gold ticks several times a second when open, so a gap this long means closed.
    # Asked of the tick stream rather than of a session calendar, which cannot see holidays or
    # broker-specific hours.
    closed = age > 600
    return {"bid": f"{t.bid:.2f}", "ask": f"{t.ask:.2f}",
            "spread": f"${t.ask - t.bid:.2f}",
            "tick age": f"{age/60:.1f} min" + ("  (MARKET CLOSED)" if closed else ""),
            "checked": utcnow().strftime("%Y-%m-%d %H:%M:%SZ"), "closed": closed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18814)
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--dry", action="store_true", help="never write state")
    ap.add_argument("--print", dest="show", action="store_true")
    a = ap.parse_args()

    art = load()
    meta = art["meta"]
    adv = meta["adverse"]
    th = Thresholds(down_at=float(adv["down_at"]), down_release=float(adv["down_release"]),
                    up_at=float(adv["up_at"]), up_release=float(adv["up_release"]),
                    base_rate=float(adv["measured"]["base_rate"]),
                    min_dwell_h=float(adv.get("hours", 9)))
    th.validate()

    state = from_dict(json.loads(STATE.read_text())) if STATE.exists() else AdvisorState()
    now = utcnow()
    out = {"computed_utc": now.isoformat(timespec="seconds"), "model_id": meta.get("model_id"),
           "cell": adv["cell"], "port": a.port}

    reading, broken, market = None, None, None
    try:
        from mt5linux import MetaTrader5
        mt5 = MetaTrader5(host="localhost", port=a.port)
        if not mt5.initialize():
            raise AdvisorUnavailable(f"bridge {a.port} init failed: {mt5.last_error()}")
        h4 = fetch_h4(mt5, a.symbol)
        market = market_panel(mt5, a.symbol)
        mt5.shutdown()

        cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
        key = f"{h4.index[-1].isoformat()}|{meta.get('model_id')}"
        if cache.get("key") == key:
            reading = cache["reading"]
            reading["asof"] = pd.Timestamp(reading["asof"])
            reading["decision_time"] = pd.Timestamp(reading["decision_time"])
            out["cache"] = "hit"
        else:
            if meta.get("serve_needs_m15"):
                # Refusing loudly beats fetching M15 quietly: the SOURCE arm's apparent edge was
                # a two-broker artifact (§3ba), so an artifact asking for M15 here is a sign it
                # was trained against the withdrawn result, not a reason to go and get M15.
                raise AdvisorUnavailable(
                    f"artifact {meta.get('model_id')} is a SOURCE-arm model and wants M15 at "
                    "serve time. This service serves the BASE arm only (§3ba). Retrain with "
                    "ADV['arm']='BASE' in scripts/v5_train_advisor.py.")
            # BASE needs H4 alone. H4 is passed in the M15 slot so that `advise`'s own frame
            # checks still execute against a real frame instead of being special-cased away.
            reading = advise(h4, h4)
            out["cache"] = "miss"
            if not a.dry:
                _atomic(CACHE, {"key": key, "reading": reading})
    except AdvisorUnavailable as e:
        broken = f"{type(e).__name__}: {e}"
    except Exception as e:                        # a notifier that crashes takes its timer down
        broken = f"unexpected {type(e).__name__}: {e}"

    closed = bool((market or {}).get("closed"))
    health = (health_of(reading, pd.Timestamp(now).tz_localize(None), market_closed=closed)
              if reading else "model_unavailable")
    p = (reading or {}).get("adverse", {}).get("p")
    asof = (reading or {}).get("decision_time")
    asof = pd.Timestamp(asof).tz_localize(timezone.utc) if asof is not None else None

    # step ONLY on a new closed bar
    stepped = False
    if reading is not None and (state.asof is None or asof > state.asof):
        state = step(state, p, asof, th, now, health=health)
        stepped = True
    elif health != "ok":
        state = step(state, None, None, th, now, health=health)

    # The displayed figure is oriented to the STATE, not to the label: an UP state must quote
    # P(up-first) = 1 - observed, or the card shows 40% where the measurement says 60%.
    od = (observed_for(state.name, p, adv["measured"]["reliability"],
                       float(adv["measured"]["base_rate"]))
          if (reading is not None and p is not None) else {})
    obs = od.get("pct")
    out.update(state=to_dict(state), health=health, stepped=stepped, broken=broken,
               market=market, market_closed=closed, p=p, observed=obs,
               observed_n=od.get("n"), observed_base=od.get("base"),
               observed_halves=list(od.get("halves", (None, None))),
               observed_bucket=list(od.get("bucket") or []),
               headline=describe(state, th, now, observed=obs))
    if not a.dry:
        _atomic(STATE, out)

    print(f"model {meta.get('model_id')}  cell {adv['cell']}  cache {out.get('cache', '-')}")
    if broken:
        print(f"BROKEN: {broken}")
    else:
        # Built with plain concatenation, NOT nested f-strings: quote-reuse inside an
        # f-string is PEP 701 and only parses on Python 3.12+. The serving host runs 3.10,
        # so it was a SyntaxError there while the local ast.parse check passed happily.
        shown = "n/a" if obs is None else f"{obs:.4f}"
        ob = od.get("base")
        shown_base = "n/a" if ob is None else f"{ob:.3f}"
        print(f"p {p:.4f}  state {state.name}  displayed {shown} (base {shown_base})  "
              f"health {health}  stepped {stepped}")
        print(f"STATE: {out['headline']}")
        if state.last_transition:
            print(f"  transition {state.last_transition[0]} -> {state.last_transition[1]}")
    if a.show and market:
        for k, v in market.items():
            print(f"  {k}: {v}")
    if a.dry:
        print("(dry — no state written)")


if __name__ == "__main__":
    main()
