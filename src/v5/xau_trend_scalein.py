"""XAUUSD H4 champion + ADVERSE-EXCURSION SCALE-IN (research variant).

The live cent-account champion (`src/v5/xau_trend.run_trades` with
`champion_signal`, exit_mode="trail") holds ONE leg: enter long when the
forecast clears 0.50, exit on a 3xATR trailing stop. This module adds the
variant the user asked for (2026-08-31): when an open trade goes ADVERSE by
"enough of what was expected", add another leg, so that when price turns back
up the added leg complements the original.

`src/v5/xau_trend.py` is deliberately NOT modified — it is live-deployed code
on a real account (magic 360542). This is a separate engine that mirrors its
structure leg-for-leg so the two are directly comparable.

WHAT "NEGATIVE ENOUGH OF WHAT WAS EXPECTED" MEANS HERE
  The trade's own declared risk is the stop distance, sl_atr x ATR-at-entry.
  An add trigger is therefore expressed as a FRACTION of that stop distance,
  which makes it invariant to the sl_atr setting: `add_trigger_frac=0.35`
  means "add once we are 35% of the way from entry to the stop, against us".
  Each further add needs another `add_trigger_frac` of adverse move from the
  LAST fill (standard downward pyramiding, not all-at-one-level).

STOP POLICY IS THE DECISIVE PARAMETER, so both readings are tested:
  "original"      — the stop stays where the first leg put it. Total risk
                    GROWS with every add (2 legs at the same stop = ~2x the
                    loss). This is the naive/martingale-flavoured reading and
                    is included because it is what "just add more" literally
                    means, not because it is expected to be safe.
  "risk_constant" — after each add the common stop is TIGHTENED to the level
                    that keeps total loss-if-stopped equal to the ORIGINAL
                    one-leg risk budget:
                        S = avg_entry - risk_budget / (total_lots * contract)
                    Risk-neutral by construction: it buys a better average
                    entry and pays for it with less room before the stop.
  "preallocated"  — the cleanest test of whether the IDEA has merit as
                    opposed to merely adding leverage. Same tightening stop as
                    risk_constant, but the FIRST leg is sized DOWN by
                    1/(1+max_adds) so that a fully scaled-in stack carries
                    roughly the ORIGINAL one-leg risk. Average exposure is
                    therefore NOT inflated, and the only thing the variant can
                    win with is a better average entry price. The honest cost
                    is explicit: trades that never pull back stay under-sized.

FILLS ARE CONSERVATIVE ON PURPOSE. An add could be modelled as a resting buy
limit filling intrabar the moment price touches the trigger (a favourable
fill). Instead this engine requires the adverse excursion to be confirmed by
a BAR CLOSE and then fills the add at the NEXT bar's open plus half-spread +
slippage — strictly worse than the limit-order version, and identical in
discipline to how the base engine handles its own entries. No intrabar
optimism anywhere.

Note that deep triggers self-limit: with add_trigger_frac=0.5 the second add
would sit at -1.0x the stop distance, i.e. at/through the stop itself, so the
stop fires before that add can ever happen. The grid does not need to special
-case this; the simulation just never reaches those adds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.v5.xau_trend import (  # noqa: F401  (SPECS/PARAMS reused verbatim)
    PARAMS, SPECS, VOL_MIN, VOL_STEP, MAX_LOT, confidence_bucket, wilder_atr, _round_lot,
)

SCALEIN_DEFAULTS = dict(
    add_trigger_frac=0.35,   # fraction of the stop distance, adverse, per add
    max_adds=2,              # additional legs beyond the first
    add_size_mult=1.0,       # each add's lots relative to the FIRST leg's lots
    stop_policy="risk_constant",   # "risk_constant" | "original"
)


def run_trades_scalein(df: pd.DataFrame, *, signal_fn, equity0: float = 3000.0,
                       params: dict | None = None, symbol: str = "XAUUSD") -> dict:
    """Long/short discrete engine with adverse-excursion scale-ins.

    Mirrors `xau_trend.run_trades(exit_mode="trail", flip_mode="confidence")`
    when `max_adds=0`, which is asserted by the backtest script as a
    regression check (the variant must reproduce the base exactly with adds
    switched off, otherwise any difference later is the engine, not the idea).

    `signal_fn`: close -> forecast Series (pass `champion_signal` to match the
    live cent bot; the base module's monkey-patched global is not used here).
    """
    p = {**PARAMS, **SCALEIN_DEFAULTS, **(params or {})}
    if p["stop_policy"] not in ("risk_constant", "original", "preallocated"):
        raise ValueError(f"unknown stop_policy {p['stop_policy']!r}")
    spec = SPECS[symbol]
    pip, contract, quote_jpy = spec["pip"], spec["contract"], spec["quote_jpy"]

    sig = signal_fn(df["close"]).values
    atr = wilder_atr(df, p["atr_period"]).values
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    spread_px = np.maximum(df["spread"].values,
                           np.nanmedian(df["spread"].values)) * pip * p["spread_cost_mult"]
    half_cost = spread_px / 2.0 + p["slippage_pips"] * pip
    idx = df.index

    eq = equity0
    equity = np.full(len(df), np.nan)
    trades: list[dict] = []
    pos = None       # dict(dir, legs[list], sl, peak, trail_on, atr_at_entry,
                     #      risk_usd, opened_t, conf, sl_dist, n_adds, last_fill)
    pending = None   # dict(dir, strength, wait) — a NEW position
    pending_add = None  # dict(wait) — an additional leg for the open position

    def fill_price(t, direction, side):
        return o[t] + side * direction * half_cost[t]

    def total_lots():
        return sum(leg["lots"] for leg in pos["legs"])

    def avg_entry():
        L = total_lots()
        return sum(leg["entry"] * leg["lots"] for leg in pos["legs"]) / L if L else np.nan

    def recompute_stop():
        """Common stop for the whole stack under the chosen policy."""
        if p["stop_policy"] == "original":
            return pos["legs"][0]["entry"] - pos["dir"] * pos["sl_dist"]
        # "preallocated" uses the same risk-holding stop as "risk_constant";
        # its difference is in the FIRST leg's size, set at entry below.
        L = total_lots()
        per_price = L * contract
        if quote_jpy:
            per_price = L * contract / max(avg_entry(), 1e-9)
        return avg_entry() - pos["dir"] * (pos["risk_usd"] / per_price)

    def close_position(t, price, reason):
        nonlocal eq, pos
        pnl = 0.0
        for leg in pos["legs"]:
            leg_pnl = (price - leg["entry"]) * pos["dir"] * leg["lots"] * contract
            if quote_jpy:
                leg_pnl /= price
            pnl += leg_pnl
        eq += pnl
        trades.append(dict(
            open_time=pos["opened_t"], close_time=idx[t],
            direction="buy" if pos["dir"] > 0 else "sell",
            legs=len(pos["legs"]), lots=round(total_lots(), 2),
            entry=round(pos["legs"][0]["entry"], 2), avg_entry=round(avg_entry(), 2),
            exit=round(price, 2), pnl=round(pnl, 2),
            r_multiple=round(pnl / pos["risk_usd"], 2) if pos["risk_usd"] else np.nan,
            confidence=pos["conf"], exit_reason=reason))
        pos = None

    for t in range(len(df)):
        # ---- 1) fill a pending ADD at this bar's open (checked before a new
        #         position fill so an add can never be mistaken for an entry)
        if pending_add is not None and pos is not None:
            if pending_add["wait"] > 0:
                pending_add["wait"] -= 1
            else:
                d = pos["dir"]
                add_lots = _round_lot(pos["legs"][0]["lots"] * p["add_size_mult"])
                if add_lots > 0:
                    entry = fill_price(t, d, +1)
                    pos["legs"].append(dict(entry=entry, lots=add_lots, opened_t=idx[t]))
                    pos["n_adds"] += 1
                    pos["last_fill"] = entry
                    new_sl = recompute_stop()
                    # under risk_constant this TIGHTENS; under original it is a no-op
                    if pos["sl"] is None or (new_sl - pos["sl"]) * d > 0:
                        pos["sl"] = new_sl
                pending_add = None
        elif pending_add is not None:
            pending_add = None  # position vanished (stopped) before the add filled

        # ---- 2) fill a pending NEW position
        if pending is not None and pending["wait"] > 0:
            pending["wait"] -= 1
        elif pending is not None:
            d = pending["dir"]
            if pos is not None:
                close_position(t, fill_price(t, pos["dir"], -1), "flip")
            if d != 0 and np.isfinite(atr[t - 1]):
                sl_dist = p["sl_atr"] * atr[t - 1]
                entry = fill_price(t, d, +1)
                risk = p["risk_frac"]
                if p["conf_risk_scale"]:
                    risk *= p["conf_risk_scale"][confidence_bucket(pending["strength"])]
                loss_per_lot = sl_dist * contract
                if quote_jpy:
                    loss_per_lot /= entry
                if p["stop_policy"] == "preallocated" and p["max_adds"] > 0:
                    # first leg sized so a fully scaled-in stack carries the
                    # ORIGINAL one-leg risk, not a multiple of it
                    risk = risk / (1.0 + p["max_adds"] * p["add_size_mult"])
                lots = _round_lot((risk * eq) / loss_per_lot)
                if lots > 0:
                    risk_usd = (sl_dist * contract * lots / entry
                                if quote_jpy else sl_dist * contract * lots)
                    if p["stop_policy"] == "preallocated" and p["max_adds"] > 0:
                        # budget the FULL stack, so recompute_stop() targets the
                        # same risk the base champion would have taken and
                        # r_multiple stays comparable across policies
                        risk_usd *= (1.0 + p["max_adds"] * p["add_size_mult"])
                    pos = dict(
                        dir=d, legs=[dict(entry=entry, lots=lots, opened_t=idx[t])],
                        sl=entry - d * sl_dist, sl_dist=sl_dist,
                        peak=entry, trail_on=False, atr_at_entry=atr[t - 1],
                        risk_usd=risk_usd, opened_t=idx[t], n_adds=0,
                        last_fill=entry, conf=confidence_bucket(pending["strength"]))
            pending = None
            pending_add = None

        # ---- 3) intrabar stop check (conservative: stop before anything else)
        if pos is not None and pos["sl"] is not None:
            hit_sl = (l[t] <= pos["sl"]) if pos["dir"] > 0 else (h[t] >= pos["sl"])
            if hit_sl:
                close_position(t, pos["sl"] - pos["dir"] * half_cost[t],
                               "trail_stop" if pos["trail_on"] else "stop_loss")
                pending_add = None

        # ---- 4) trailing stop from the completed bar's extreme (stack-wide)
        if pos is not None:
            ext = h[t] if pos["dir"] > 0 else l[t]
            pos["peak"] = (max(pos["peak"], ext) if pos["dir"] > 0
                           else min(pos["peak"], ext))
            gain = (pos["peak"] - avg_entry()) * pos["dir"]
            if gain >= p["trail_activation_atr"] * pos["atr_at_entry"]:
                pos["trail_on"] = True
                new_sl = pos["peak"] - pos["dir"] * p["trail_atr"] * pos["atr_at_entry"]
                if (new_sl - pos["sl"]) * pos["dir"] > 0:
                    pos["sl"] = new_sl

        # ---- 5) ADD decision on this completed bar -> filled next open
        if (pos is not None and pending_add is None and pending is None
                and p["max_adds"] > 0 and pos["n_adds"] < p["max_adds"]):
            adverse = (pos["last_fill"] - c[t]) * pos["dir"]   # >0 = against us
            if adverse >= p["add_trigger_frac"] * pos["sl_dist"]:
                pending_add = dict(wait=p["entry_delay_bars"] - 1)

        # ---- 6) signal decision on this completed bar -> filled next open
        s = sig[t]
        if np.isfinite(s) and np.isfinite(atr[t]):
            d = 1 if s > 0 else -1
            strong = abs(s) >= p["enter_thresh"]
            if pos is None and pending is None and strong:
                pending = dict(dir=d, strength=s, wait=p["entry_delay_bars"] - 1)
            elif pos is not None and d != pos["dir"] and strong:
                if abs(s) >= p["flip_thresh"]:
                    pending = dict(dir=d, strength=s, wait=p["entry_delay_bars"] - 1)

        # ---- 7) mark equity (closed + floating across all legs)
        floating = 0.0
        if pos is not None:
            for leg in pos["legs"]:
                lp = (c[t] - leg["entry"]) * pos["dir"] * leg["lots"] * contract
                if quote_jpy:
                    lp /= c[t]
                floating += lp
        equity[t] = eq + floating

    open_position = dict(pos) if pos is not None else None
    if pos is not None:
        close_position(len(df) - 1, c[-1], "eod_mark")
        equity[-1] = eq

    return dict(trades=pd.DataFrame(trades),
                equity=pd.Series(equity, index=idx, name="equity"),
                signal=pd.Series(sig, index=idx, name="forecast"),
                open_position=open_position, pending=pending)
