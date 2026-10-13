"""v5_basket_challenge_exec.py — FundingPips DIVERSIFIED BASKET challenge executor.

Guard-first, multi-symbol sibling of v5_xau_challenge.py. Trades the champion
long-only recipe across the tradeable drift classes (indices + crypto + XAU +
silver), 2-Step Standard @ 7% vol (configs/v5_basket_challenge.json). Every pass:

  1. GUARD on live account equity (src.v5.challenge_guards.decide):
       halt / day_lock / locked / complete  -> ensure FLAT (all symbols), exit
       realize_target                        -> close everything (bank it), exit
       trade                                 -> reconcile each symbol to target
  2. TARGET per symbol = scripts.v5_basket_challenge.target_leverage(model)
     -> account-leverage; converted to lots at execute time from live
     symbol_info/tick; buffered no-trade band cuts churn.
  3. State persisted (day anchor 00:00 UTC+3, phase, locks). Phase promotion
     is manual: --advance-phase.

Safety: real accounts need --live; orders need --execute; a symbol only trades
if its broker name is mapped in config (fp_symbol) AND resolves on the terminal.
Until the FundingPips account exists, every fp_symbol is null -> DRY-RUN only:
it prints the guard status + per-symbol target plan and logs a CSV row, never
sending an order. This is the pre-live verification mode.

    # dry-run one pass (read-only, safe on any account):
    conda run -n envmt5 python scripts/v5_basket_challenge_exec.py \
        --state data/v5_runs/basket_challenge_dry_state.json \
        --paper-csv data/v5_runs/basket_challenge_dry_log.csv
    # live (after account purchase + fp_symbol mapping):
    ... --live --execute
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.v5.challenge_guards as cg  # noqa: E402
from scripts.v5_basket_challenge import target_leverage, MODELS  # noqa: E402
from src.core.mt5_connector import MT5Connector  # noqa: E402

CONFIG_FILE = ROOT / "configs" / "v5_basket_challenge.json"
STATE_DEFAULT = ROOT / "data" / "v5_runs" / "basket_challenge_state.json"

# MT5 SYMBOL_TRADE_MODE_*. Anything but FULL means the broker restricts orders
# on that symbol, and the restriction can appear WITHOUT WARNING mid-challenge:
# FundingPips flipped XAUUSDmicro to CLOSEONLY between 2026-07-20 and 07-23, so
# after a manual close the XAU sleeve could never reopen. Every hourly pass just
# printed a one-line "ORDER REJECTED: retcode=10044" and carried on looking
# healthy, and the book silently ran 2 of 3 sleeves for ~14h. Check the mode
# BEFORE sending and say so loudly. (2026-07-24)
TRADE_MODES = {0: "DISABLED", 1: "LONGONLY", 2: "SHORTONLY", 3: "CLOSEONLY", 4: "FULL"}


def open_restriction(info, want_long: bool) -> str | None:
    """Broker-side reason this symbol cannot be OPENED/ADDED in this direction.

    Reducing is always permitted, so callers only consult this when increasing
    exposure. LONGONLY/SHORTONLY still allow the matching side.
    """
    if info is None:
        return None                       # unavailable is reported separately
    mode = getattr(info, "trade_mode", 4)
    if mode == 4 or (mode == 1 and want_long) or (mode == 2 and not want_long):
        return None
    return TRADE_MODES.get(mode, str(mode))


# ---------------------------------------------------------------- evening session gate
# "Do not open new positions after 5PM if I close manually, but reopen as the market opens
# at night." Blocks every INCREASE in exposure inside the evening window; reductions, closes
# and compliance-SL maintenance are always permitted, so the gate is a ONE-WAY VALVE and can
# never trap the book in a position it wants out of.
#
# WHY THE TIMES ARE WHAT THEY ARE (measured off this broker's own M1 gaps, 2026-10-08):
#     XAUUSD     break 23:49 -> 01:05   (12 of 12 days)
#     US100.cash break 23:49 -> 01:05   (12 of 12 days)
#     UKOIL.cash break 23:49 -> 03:05   (13 of 13 days)
#     BTCUSD     break 23:54 -> 00:05   (12 of 12 days)
# The window therefore ENDS at 01:05, when gold reopens. The book's own 4-hourly timer next
# fires at 03:40 server, by which point UKOIL has reopened too, so the whole book comes back
# in one pass rather than running a narrow 3-sleeve book overnight (BRENT is the sleeve that
# took daily-limit breaches from 13.9% to 0%, so a partial book is a real risk change).
NO_INCREASE_FROM = (17, 0)        # 17:00 server == 17:00 EAT, the user's 5PM
NO_INCREASE_UNTIL = (1, 5)        # gold's measured reopen


def server_now(conn, ref_symbol: str = "XAUUSD"):
    """The BROKER's wall clock, read from its own tick, never a hard-coded UTC offset.

    FTMO-Demo runs UTC+3 today and shifts with US DST. A fixed offset would silently move the
    5PM cutoff by an hour twice a year with nothing failing visibly -- the same class of error
    as the session_windows 2h-DST claim that 3be had to correct to 1h. Returns None if no tick
    is available, and the caller then FAILS CLOSED (treats the window as active).
    """
    from datetime import datetime as _dt
    try:
        # MT5Connector exposes get_tick(); the raw module's symbol_info_tick() is NOT on it,
        # and calling the wrong name raised, hit the except, and failed the gate CLOSED --
        # which blocked every sleeve all day while printing a confident reason. Same class of
        # error as conn.history_deals_get in 687ea6f: the connector is a wrapper, not the
        # module. Fall back to the raw module rather than assuming either exists.
        tick = None
        if hasattr(conn, "get_tick"):
            tick = conn.get_tick(ref_symbol)
        elif hasattr(conn, "_mt5"):
            tick = conn._mt5.symbol_info_tick(ref_symbol)
        else:
            print("  session gate: connector exposes neither get_tick nor _mt5")
            return None
        ts = int(getattr(tick, "time", 0) or 0)
    except Exception as exc:                      # noqa: BLE001 -- named, never swallowed
        print(f"  session gate: could not read {ref_symbol} tick ({exc})")
        return None
    if ts <= 0:
        print(f"  session gate: {ref_symbol} tick has no timestamp")
        return None
    return _dt.utcfromtimestamp(ts)               # the epoch IS server-stamped


def no_increase_reason(srv, frm=NO_INCREASE_FROM, until=NO_INCREASE_UNTIL) -> str | None:
    """Reason increases are blocked right now, or None. The window WRAPS MIDNIGHT."""
    if srv is None:
        return (f"server time unknown — failing closed, no increases "
                f"(window {frm[0]:02d}:{frm[1]:02d}-{until[0]:02d}:{until[1]:02d} server)")
    hm = srv.hour * 60 + srv.minute
    a = frm[0] * 60 + frm[1]
    b = until[0] * 60 + until[1]
    inside = (hm >= a or hm < b) if a > b else (a <= hm < b)
    if not inside:
        return None
    return (f"{srv:%H:%M} server is inside the evening no-increase window "
            f"{frm[0]:02d}:{frm[1]:02d}-{until[0]:02d}:{until[1]:02d}")


# ------------------------------------------------- re-entry gated on the bottom detector
# USER REQUEST, 2026-10-13: "when i close a position, when opening use bottom detector."
#
# *** THIS IS MEASURED AS HARMFUL AND IS SHIPPED BECAUSE IT WAS ASKED FOR TWICE. ***
# SS3bm, 9,000+ decision events on XAU H1: waiting for the detector makes the ENTRY PRICE WORSE
# by 1.1-3.6bp against a 2.59bp round trip, and 18 of 18 cells are negative. The mechanism is
# adverse selection, not calibration -- at p>=0.60 with an 8h wait, buying at market earned
# -22.80bp on the events where the detector FIRED and +17.85bp where it did NOT, a 40.65bp gap
# at t -5.23. A "bottom" is a bar where price has already fallen, and on a positive-drift asset
# this book is long, those bars have worse forward returns. SS3bc found the same thing
# independently (t -5.96).
#
# It is therefore OFF BY DEFAULT and switched on per-book in config, so turning it off is a
# one-line change rather than a code edit.
#
# BOUNDED ON PURPOSE. The book only looks every 4h and the detector fires on roughly 8% of bars
# at 0.60, so an ungated wait could leave a sleeve flat for days. Being flat costs far more than
# 2.4bp of entry, so `max_wait_hours` forces a market entry once it expires. Without that this
# feature's worst case is much worse than the thing it is trying to improve.
BOTTOM_GATE_DEFAULTS = dict(enabled=False, threshold=0.60, max_wait_hours=24,
                            train_bars=20000, h1_bars=24000)


def _h1_frame(conn, symbol: str, bars: int):
    """Closed H1 bars from the BROKER's own feed -- never the CSVs, which are a different feed
    and months stale (SS3ba: a dataset from two brokers is a leak no timestamp control sees)."""
    try:
        r = conn.get_rates(symbol, "H1", bars)
    except Exception as exc:                       # noqa: BLE001 -- named, never swallowed
        print(f"    bottom gate: get_rates({symbol}) failed: {exc}")
        return None
    if r is None or len(r) < 5000:
        print(f"    bottom gate: {symbol} returned "
              f"{0 if r is None else len(r)} H1 bars, need >= 5000")
        return None
    d = pd.DataFrame(r)
    if "time" in d.columns:
        d["time"] = pd.to_datetime(d["time"], unit="s")
        d = d.set_index("time")
    return d.iloc[:-1]                             # drop the bar still forming


def bottom_probability_for(conn, symbol: str, cfg_gate: dict):
    """P(this bar is within TOL of a swing low) for ONE symbol, on its own H1 series.

    Uses the deployed detector verbatim (scripts/v5_zigzag_signal.bottom_probability). SS3as
    measured the bottom-pivot relationship as UNIVERSAL across instruments -- the TRANSFER arm,
    trained without ever seeing gold, scored +0.0570 against POOLED's +0.0576 -- so scoring a
    non-gold sleeve on its own bars is supported rather than improvised. Returns None if the
    detector cannot be evaluated, and the caller then opens at market rather than stalling.
    """
    df = _h1_frame(conn, symbol, int(cfg_gate.get("h1_bars", 24000)))
    if df is None:
        return None
    try:
        from scripts.v5_zigzag_signal import bottom_probability
        st = bottom_probability(df, float(cfg_gate.get("threshold", 0.60)),
                                0.006, 0.005, 48,
                                train_bars=int(cfg_gate.get("train_bars", 20000)))
        return float(st.prob)
    except Exception as exc:                       # noqa: BLE001
        print(f"    bottom gate: scoring {symbol} failed: {exc}")
        return None


def apply_model_to_guards(cfg) -> None:
    """Override challenge_guards module constants from config (model-correct)."""
    g = cfg["guards"]
    cg.DAILY_GUARD_FRAC = float(g["daily_guard_frac"])
    cg.OVERALL_HALT_FRAC = float(g["overall_halt_frac"])
    cg.PHASE_TARGETS = {int(k): float(v) for k, v in g["phase_targets"].items()}
    cg.set_reset_tz(cfg.get("reset_tz"))   # FundingPips UTC+3 / FTMO CE(S)T
    # Per-config vol-dial override (default 7% lives in MODELS). Lets one config
    # pick a faster/riskier operating point on the pass%-vs-speed frontier
    # without editing the shared MODELS table. (2026-07-24)
    if cfg.get("vol") is not None:
        MODELS[cfg["model"]]["vol"] = float(cfg["vol"])


def log_row(path, row) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    exists = p.exists()
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def account_start_balance(conn, acct) -> float:
    """The account's OWN starting balance, from its DEAL_TYPE_BALANCE credit.

    Better than today's balance because an account may already have been traded before the bot
    was pointed at it. Falls back to the live balance if the history is unavailable.
    """
    # MT5Connector has no history_deals_get; the raw module is on `_mt5`. Reaching for a
    # method the wrapper does not have, under a bare `except Exception`, is exactly how an
    # AttributeError once got swallowed here and printed a confident wrong answer -- so the
    # failure is NAMED rather than silently falling through.
    raw = getattr(conn, "_mt5", None)
    if raw is None or not hasattr(raw, "history_deals_get"):
        print(f"  (no deal history available; anchoring to the live balance "
              f"{float(acct.balance):,.2f})")
        return float(acct.balance)
    try:
        from datetime import datetime, timedelta, timezone
        deals = raw.history_deals_get(datetime(2015, 1, 1, tzinfo=timezone.utc),
                                      datetime.now(timezone.utc) + timedelta(days=1))
        credits = [d.profit for d in (deals or []) if getattr(d, "type", None) == 2]
        if credits:
            print(f"  anchor from the account's own balance deal: {float(credits[0]):,.2f}")
            return float(credits[0])
        print("  (no DEAL_TYPE_BALANCE credit found; anchoring to the live balance)")
    except Exception as e:
        print(f"  (deal history unreadable: {type(e).__name__}: {e}; "
              f"anchoring to the live balance)")
    return float(acct.balance)


def load_state(path: Path, acct, conn=None) -> dict:
    """Load the guard state, RE-ANCHORING if the account has changed.

    *** WHY THE ACCOUNT CHECK EXISTS. *** The guards (daily lock, overall halt, phase target)
    are all fractions of `initial_balance`. Pointing the bot at a new account without clearing
    the state leaves them anchored to the OLD account's balance, silently. Measured 2026-10-02:
    after switching to FTMO 1514735296 the state still carried `initial_balance 99564.82` from
    the expired account, which put the 8% halt at 91,600 instead of 92,000 -- i.e. it permitted
    ~$400 MORE loss than intended, eating into FTMO's own max-loss buffer. Nothing in the log
    looked wrong; the only tell was a "+2.86% progress" on an account that had made +2.41%.
    """
    if path.exists():
        st = json.loads(path.read_text())
        prev = st.get("account")
        if prev is not None and int(prev) != int(acct.login):
            start = account_start_balance(conn, acct) if conn is not None else float(acct.balance)
            st = cg.init_state(start, float(acct.equity))
            st["account"] = int(acct.login)
            print(f"  *** ACCOUNT CHANGED {prev} -> {acct.login}: guard state RE-ANCHORED to "
                  f"{start:,.2f} (was carrying the previous account's baseline) ***")
            return st
        st.setdefault("account", int(acct.login))
        return st
    start = account_start_balance(conn, acct) if conn is not None else float(acct.balance)
    st = cg.init_state(start, float(acct.equity))
    st["account"] = int(acct.login)
    print(f"  state INIT: account {acct.login}  initial_balance {st['initial_balance']:,.2f}  "
          f"phase 1 target +{cg.PHASE_TARGETS[1]*100:.0f}%")
    return st


def save_state(path: Path, st) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(st, indent=1))


def held_lots(conn, symbol, magic) -> float:
    """Signed net lots we hold on `symbol` (our magic only)."""
    net = 0.0
    for p in (conn.get_positions(magic=magic) or []):
        if p.symbol == symbol:
            net += p.volume if p.type == 0 else -p.volume
    return net


def close_volume(conn, symbol, magic, lots, run_id) -> list:
    """Reduce our exposure on `symbol` by `lots` by CLOSING our own tickets
    (fully, then partially for the remainder), oldest first.

    Required on HEDGING accounts (FTMO is margin_mode=2): sending an opposing
    market order there does NOT net the position down — it opens a second,
    opposite position, doubling margin and cost. Closing by ticket is correct
    on both hedging and netting accounts.
    """
    mt5 = conn._mt5
    try:
        conn.symbol_select(symbol, True)   # tick/info are None if not in Market Watch
    except Exception:  # noqa: BLE001
        pass
    remaining = round(float(lots), 2)
    done = []
    ps = [p for p in (conn.get_positions(magic=magic) or []) if p.symbol == symbol]
    ps.sort(key=lambda p: getattr(p, "time", 0))          # FIFO
    for p in ps:
        if remaining <= 1e-9:
            break
        vol = round(min(float(p.volume), remaining), 2)
        if vol <= 0:
            continue
        tick = conn.get_tick(symbol)
        is_long = (p.type == 0)
        req = dict(action=mt5.TRADE_ACTION_DEAL, symbol=symbol, volume=vol,
                   type=mt5.ORDER_TYPE_SELL if is_long else mt5.ORDER_TYPE_BUY,
                   position=p.ticket,
                   price=(tick.bid if is_long else tick.ask),
                   deviation=50, magic=magic, comment=run_id,
                   type_time=mt5.ORDER_TIME_GTC,
                   type_filling=conn._fill_type(symbol))
        r = mt5.order_send(req)
        rc = getattr(r, "retcode", None)
        done.append((p.ticket, vol, rc))
        if rc == mt5.TRADE_RETCODE_DONE:
            remaining = round(remaining - vol, 2)
    return done


def apply_compliance_sl(conn, symbol, magic, equity, cap_pct, send,
                        min_atr_mult=2.0) -> str:
    """Cap each symbol's worst-case loss at `cap_pct` of ACCOUNT equity via a broker SL.

    Prop-firm compliance only — the book's real protection is the trend filter shrinking
    exposure, not this stop. Measured on 2018-2026 it sits **2.3-4.2x daily ATR** away at
    the tightest historical position sizes and was **never hit by a single session** in
    8+ years (worst days: BTC -38.6%, BRENT -31.5%, NDX -12.5%, XAU -11.6%), so it does
    not interfere with the strategy.

    MUST run every pass: vol-targeting changes position size continuously, so a stop that
    was 3% yesterday is not 3% today. All tickets on a symbol are treated as ONE economic
    position and given the SAME stop price, derived from the volume-weighted average
    entry, so the AGGREGATE loss is capped (per-ticket stops would not bound the total).
    """
    mine = [p for p in (conn.get_positions(magic=magic) or []) if p.symbol == symbol]
    if not mine:
        return ""
    mt5 = conn._mt5
    si = conn.symbol_info(symbol)
    tick = conn.get_tick(symbol)
    vol = sum(float(p.volume) for p in mine)
    if vol <= 0:
        return ""
    d = 1 if mine[0].type == 0 else -1
    avg_entry = sum(float(p.price_open) * float(p.volume) for p in mine) / vol
    budget = float(equity) * float(cap_pct)

    # $ lost per 1.0 price unit of adverse move, from the broker's own calc
    otype = mt5.ORDER_TYPE_BUY if d > 0 else mt5.ORDER_TYPE_SELL
    per_unit = abs(mt5.order_calc_profit(otype, symbol, float(vol),
                                         float(avg_entry), float(avg_entry - d * 1.0)))
    if not per_unit:
        return f"{symbol}: cannot size SL (order_calc_profit=0)"
    dist = budget / per_unit
    px = float(tick.bid) if d > 0 else float(tick.ask)
    sl = avg_entry - d * dist

    # THE CAP CAN BE MATHEMATICALLY UNREACHABLE ON A SMALL LONG. `dist = budget / per_unit`
    # takes no account of the fact that a long's worst case is price -> 0. On 2026-09-16 the
    # 0.03-lot BTCUSD sleeve gave budget $2,987 / per_unit 0.03 = dist 99,566 against an entry
    # of 75,874, i.e. a stop at -23,692, which the broker refused with retcode 10016 ("invalid
    # stops"). The position could only ever lose its $2,276 of notional, so there was nothing
    # to cap -- but the log said "REJECTED" and would have buried a genuine rejection in noise.
    # Report the arithmetic instead of sending an impossible order.
    if d > 0:
        max_possible_loss = per_unit * float(avg_entry)      # price -> 0
        if max_possible_loss <= budget:
            return (f"{symbol}: no compliant SL needed — worst case (price -> 0) loses "
                    f"${max_possible_loss:,.0f}, under the {cap_pct:.1%} cap "
                    f"(${budget:,.0f}). Position is too small to breach it.")

    # broker minimum stop distance
    point = float(si.point) or 0.01
    min_gap = float(getattr(si, "trade_stops_level", 0) or 0) * point
    if d > 0 and sl > px - min_gap:
        sl = px - max(min_gap, point)
    if d < 0 and sl < px + min_gap:
        sl = px + max(min_gap, point)

    # Already losing more than the cap -> the stop cannot be placed compliantly.
    # Closing IS the compliant action; say so loudly rather than silently skipping.
    open_loss = sum(float(p.profit) for p in mine)
    if open_loss < -budget:
        return (f"{symbol}: open loss {open_loss:+.0f} already exceeds the "
                f"{cap_pct:.1%} cap ({budget:.0f}) — REDUCE OR CLOSE (no compliant SL)")

    digits = int(getattr(si, "digits", 2) or 2)
    sl = round(float(sl), digits)
    changed = 0
    for p in mine:
        if abs(float(p.sl or 0.0) - sl) <= point:
            continue
        changed += 1
        if send:
            try:
                conn.modify_position(p.ticket, sl=sl, tp=float(p.tp or 0.0))
            except Exception as exc:  # noqa: BLE001
                # 10018 = TRADE_RETCODE_MARKET_CLOSED. Expected at weekends for
                # everything except crypto; the stop lands on the next pass after
                # the open. Not an error, and must not read like one in the logs.
                if "10018" in str(exc):
                    return (f"{symbol}: market closed — compliance SL "
                            f"{sl:.{digits}f} queued for the next pass after open")
                return f"{symbol}: SL modify REJECTED on #{p.ticket}: {exc}"
    pct = dist / avg_entry * 100
    note = (f"{symbol}: SL {sl:.{digits}f}  ({dist:.2f} away = {pct:.1f}% of price, "
            f"caps loss at {cap_pct:.1%} = ${budget:,.0f})"
            + (f"  [{changed} ticket(s) updated]" if changed else "  [already set]"))
    return note


def target_lots(conn, symbol, lev, equity) -> float | None:
    """lots = lev * equity / (contract_size * price). None if symbol unavailable.

    MUST symbol_select() first: a symbol that is not in Market Watch returns
    symbol_info()=None. After any terminal restart nothing is selected, so
    without this the bot reports "in sync — nothing to do" and SILENTLY NEVER
    TRADES while looking healthy in the logs.
    """
    try:
        conn.symbol_select(symbol, True)
    except Exception:  # noqa: BLE001
        pass
    info = conn.symbol_info(symbol)
    tick = conn.get_tick(symbol)
    if info is None or tick is None:
        return None
    price = float(getattr(tick, "ask", 0) or getattr(tick, "last", 0) or 0)
    csize = float(getattr(info, "trade_contract_size", 0) or 0)
    step = float(getattr(info, "volume_step", 0.01) or 0.01)
    if price <= 0 or csize <= 0:
        return None
    raw = lev * equity / (csize * price)
    return round(raw / step) * step


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--state", default=str(STATE_DEFAULT))
    ap.add_argument("--paper-csv", default=None)
    ap.add_argument("--advance-phase", action="store_true")
    ap.add_argument("--config", default=str(CONFIG_FILE),
                    help="challenge config JSON (default: FundingPips basket)")
    ap.add_argument("--port", type=int, default=None,
                    help="mt5linux bridge port; falls back to the config's 'port', then 18812. "
                         "Needed because one host runs several terminals (18812 Maven, "
                         "18813 cent, 18814 FTMO) and MT5Connector defaults to Maven.")
    ap.add_argument("--guard-only", action="store_true",
                    help="fast real-time protector: evaluate guards + flatten on "
                         "breach, but DO NOT reconcile/open (for a 1-2min timer)")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    apply_model_to_guards(cfg)
    model, magic, run_id = cfg["model"], cfg["magic"], cfg["run_id"]
    buf = float(cfg.get("reconcile_buffer", 0.15))
    symmap = {k: v for k, v in cfg["symbols"].items() if not k.startswith("_")}
    state_path = Path(args.state)

    port = args.port or int(cfg.get("port") or 18812)
    conn = MT5Connector(port=port)
    print(f"  bridge port {port}")
    conn.connect()
    try:
        acct = conn.account_info()
        equity, balance = float(acct.equity), float(acct.balance)
        # real-account order lock (mirror of dual/challenge bots)
        is_demo = "demo" in str(getattr(acct, "server", "")).lower()
        send = args.execute and (is_demo or args.live)

        state = load_state(state_path, acct, conn)
        if args.advance_phase:
            state = cg.advance_phase(state, balance)
            save_state(state_path, state)
            print(f"  advanced to PHASE 2, start {state['phase_start']:,.2f}, "
                  f"target +{cg.PHASE_TARGETS[2]*100:.0f}%")
            return

        state, action = cg.decide(state, balance, equity)
        save_state(state_path, state)
        day_dd = (equity / state["day_anchor"] - 1) * 100
        tot_dd = (equity / state["initial_balance"] - 1) * 100
        prog = (equity / state["phase_start"] - 1) * 100
        print(f"[basket challenge] model={model.upper()} acct={acct.login} "
              f"bal={balance:,.2f} eq={equity:,.2f} {acct.currency}")
        print(f"  guards: action={action}  phase {state['phase']} "
              f"progress {prog:+.2f}% (target +{state['phase_target_frac']*100:.0f}%)  "
              f"day {day_dd:+.2f}% (lock -{cg.DAILY_GUARD_FRAC*100:.1f}%)  "
              f"total {tot_dd:+.2f}% (halt -{cg.OVERALL_HALT_FRAC*100:.0f}%)")

        row = dict(time_utc=datetime.now(timezone.utc).strftime("%F %T"),
                   account=acct.login, balance=round(balance, 2),
                   equity=round(equity, 2), action=action, phase=state["phase"],
                   progress_pct=round(prog, 3), day_dd_pct=round(day_dd, 3),
                   total_dd_pct=round(tot_dd, 3), n_symbols=len(symmap),
                   n_mapped=0, plan="", sent=int(send), blocked="")

        # ---- guard actions: ensure flat, no reconcile ----
        if action in ("halt", "day_lock", "locked", "complete", "realize_target"):
            flats = []
            for esym, meta in symmap.items():
                bsym = meta.get("fp_symbol")
                if bsym and held_lots(conn, bsym, magic) != 0.0:
                    flats.append(bsym)
                    print(f"  GUARD[{action}]: flatten {bsym}")
                    if send:
                        for p in (conn.get_positions(magic=magic) or []):
                            if p.symbol == bsym:
                                conn.close_position(p)
            # A guard flatten is the BOOK closing, not the user. Clearing book_lots here stops
            # the next pass mistaking it for a manual close and waiting for a bottom to undo a
            # risk action -- which would be the single worst failure this feature could have.
            state["book_lots"] = {}
            state["awaiting_bottom"] = {}
            save_state(state_path, state)
            if action == "halt":
                print("  *** PERMANENT HALT — risk line hit ***")
            elif action == "complete":
                print("  *** PHASE TARGET REALIZED — await promotion, run --advance-phase ***")
            elif not flats:
                print(f"  GUARD[{action}]: already flat")
            row["plan"] = f"{action}:flatten({len(flats)})"
            if args.paper_csv:
                log_row(args.paper_csv, row)
            return

        # ---- guard-only mode: no reconcile, just confirm we're safe ----
        if args.guard_only:
            print("  guard-only: safe (action=trade) — no reconcile")
            row["plan"] = "guard_only:safe"
            if args.paper_csv:
                log_row(args.paper_csv, row)
            return

        # ---- normal reconcile: move each symbol toward its target ----
        # cfg["classes"] lets one engine serve several books (100K vs 10K).
        # cfg["class_weights"] optionally tilts them away from equal class risk;
        # absent, the engine keeps its historical equal-risk behaviour.
        targets = target_leverage(model, cfg.get("classes"), cfg.get("class_weights"))
        plans, n_mapped, blocked = [], 0, []

        # Evening gate, evaluated ONCE so every sleeve in this pass sees one clock.
        sess = cfg.get("session", {})
        _frm = tuple(sess.get("no_increase_from", NO_INCREASE_FROM))
        _unt = tuple(sess.get("no_increase_until", NO_INCREASE_UNTIL))
        srv = server_now(conn, cfg.get("session_ref_symbol", "XAUUSD"))
        sess_blk = None if sess.get("disabled") else no_increase_reason(srv, _frm, _unt)
        if sess_blk:
            print(f"  EVENING GATE ACTIVE — {sess_blk}. No sleeve may ADD exposure; "
                  f"reductions, closes and SL maintenance still run.")
        elif srv is not None:
            print(f"  session: {srv:%H:%M} server — outside the no-increase window "
                  f"{_frm[0]:02d}:{_frm[1]:02d}-{_unt[0]:02d}:{_unt[1]:02d}, increases allowed")
        sess_skipped = []

        # ---- re-entry gated on the bottom detector (OFF unless config asks for it; §3bm
        # measures it as harmful, so it must never switch itself on).
        bg = dict(BOTTOM_GATE_DEFAULTS); bg.update(cfg.get("bottom_gate", {}))
        bg_on = bool(bg.get("enabled"))
        book_lots = dict(state.get("book_lots") or {})
        awaiting = dict(state.get("awaiting_bottom") or {})
        now_iso = datetime.now(timezone.utc)
        if bg_on:
            print(f"  BOTTOM GATE on: after a manual close a sleeve waits for "
                  f"P(bottom) >= {bg['threshold']} (max {bg['max_wait_hours']}h, then market). "
                  f"§3bm measured this as -1.1 to -3.6bp of entry — enabled by request.")
        bg_waiting, bg_fired, bg_timeout = [], [], []
        print(f"  {'symbol':9s} {'tgt lev':>8s} {'held lot':>9s} {'tgt lot':>9s}  action")
        for esym, lev in sorted(targets.items()):
            meta = symmap.get(esym, {})
            bsym = meta.get("fp_symbol")
            if not bsym:                            # unmapped -> can't trade yet
                print(f"  {esym:9s} {lev:8.3f} {'—':>9s} {'—':>9s}  UNMAPPED (dry)")
                continue
            n_mapped += 1
            tl = target_lots(conn, bsym, lev, equity)
            hl = held_lots(conn, bsym, magic)
            if tl is None:
                print(f"  {esym:9s} {lev:8.3f} {hl:9.2f} {'n/a':>9s}  SYMBOL UNAVAILABLE")
                continue
            band = buf * max(abs(tl), 1e-9)
            act = "hold"
            if abs(tl - hl) > band:
                act = "adjust" if hl != 0 else "open"
                plans.append((bsym, hl, tl))
            print(f"  {esym:9s} {lev:8.3f} {hl:9.2f} {tl:9.2f}  {act}")

            # Broker restriction check BEFORE sending: only increases can be
            # blocked, and a blocked sleeve must be loud — it silently halves
            # the book's diversification otherwise.
            # The evening gate sits BESIDE the broker restriction because it has identical
            # semantics: only increases can be blocked, and a blocked sleeve must be loud.
            if act != "hold" and tl > hl and sess_blk:
                sess_skipped.append(bsym)
                kind = "re-open" if hl == 0 else f"add {tl - hl:+.2f} lot to"
                print(f"    -- EVENING GATE: will not {kind} {bsym}. Stays as-is until the "
                      f"market reopens; the {_unt[0]:02d}:{_unt[1]:02d} lift means the next "
                      f"scheduled pass restores the book.")
                continue

            # --- bottom-detector re-entry gate (only a brand-new position after a close the
            # BOOK did not make; adds and reductions are untouched).
            if bg_on and act != "hold" and tl > hl and hl == 0:
                prev = float(book_lots.get(bsym) or 0.0)
                if prev > 0 and bsym not in awaiting:
                    awaiting[bsym] = now_iso.isoformat()
                    print(f"    MANUAL CLOSE detected on {bsym} (book left {prev:.2f} lot, "
                          f"holds 0.00) — re-entry now waits for a bottom")
                if bsym in awaiting:
                    since = datetime.fromisoformat(awaiting[bsym])
                    waited = (now_iso - since).total_seconds() / 3600.0
                    prob = bottom_probability_for(conn, bsym, bg)
                    if prob is None:
                        print(f"    bottom gate: {bsym} unscorable — opening at market "
                              f"rather than stalling the sleeve")
                        awaiting.pop(bsym, None)
                    elif prob >= float(bg["threshold"]):
                        print(f"    BOTTOM FIRED on {bsym}: P={prob:.3f} "
                              f">= {bg['threshold']} after {waited:.1f}h — re-entering")
                        bg_fired.append(bsym)
                        awaiting.pop(bsym, None)
                    elif waited >= float(bg["max_wait_hours"]):
                        print(f"    bottom gate TIMEOUT on {bsym}: P={prob:.3f} after "
                              f"{waited:.1f}h >= {bg['max_wait_hours']}h — opening at market. "
                              f"Being flat costs more than the entry does.")
                        bg_timeout.append(bsym)
                        awaiting.pop(bsym, None)
                    else:
                        print(f"    -- BOTTOM GATE: {bsym} P={prob:.3f} < {bg['threshold']}, "
                              f"waited {waited:.1f}h of {bg['max_wait_hours']}h — staying flat")
                        bg_waiting.append(bsym)
                        continue

            if act != "hold" and tl > hl:
                restr = open_restriction(conn.symbol_info(bsym), want_long=True)
                if restr:
                    blocked.append(f"{bsym}:{restr}")
                    print(f"    !! BROKER-BLOCKED: {bsym} trade_mode={restr} — "
                          f"cannot add {tl - hl:+.2f} lot. This sleeve stays flat "
                          f"until the broker restores full trading.")
                    continue

            if send and act != "hold":
                delta = round(tl - hl, 2)
                try:
                    if delta > 0:                      # increase -> open/add
                        r = conn.open_position(bsym, "buy", abs(delta),
                                               magic=magic, comment=run_id)
                        print(f"    EXECUTED buy {abs(delta):.2f}: "
                              f"{r.get('retcode', r) if isinstance(r, dict) else r}")
                    else:                              # decrease -> CLOSE tickets
                        res = close_volume(conn, bsym, magic, abs(delta), run_id)
                        for tk, vol, rc in res:
                            print(f"    CLOSED {vol:.2f} of #{tk}: retcode={rc}")
                        if not res:
                            print("    ! nothing to close (no tickets found)")
                except Exception as exc:  # noqa: BLE001
                    print(f"    ORDER REJECTED: {exc}")
                    # Backstop: the broker can flip a symbol between the check
                    # above and the send. 10044 = TRADE_RETCODE_CLOSE_ONLY.
                    if "10044" in str(exc):
                        blocked.append(f"{bsym}:CLOSEONLY")
                        print(f"    !! BROKER-BLOCKED: {bsym} is CLOSE-ONLY (10044)")

        # ---- prop-firm compliance: cap each symbol's worst case at N% of equity ----
        # Runs AFTER the reconcile so the stop matches the position we now hold, and on
        # EVERY pass because vol-targeting keeps changing that size.
        sl_cap = cfg.get("stop_loss_pct_per_position")
        if not sl_cap:
            # SL DISABLED IN CONFIG -> actively CLEAR any stop this bot previously set.
            # Merely skipping `apply_compliance_sl` would leave the old broker-side stops in
            # place forever, so "remove the stop loss" would silently not remove anything.
            # Only magic-matched tickets are touched: a hand-placed position is never modified.
            cleared, failed = 0, []
            for esym in sorted(targets):
                bsym = symmap.get(esym, {}).get("fp_symbol")
                if not bsym:
                    continue
                # MT5Connector exposes get_positions(symbol, magic) -- NOT positions_get.
                # The first version of this block called the wrong name inside a bare
                # `except Exception: held = []`, so it printed "none outstanding" while four
                # positions still carried broker-side stops. Narrow the except to the calls
                # that can legitimately fail and let a typo raise.
                held = [q for q in (conn.get_positions(symbol=bsym, magic=magic) or [])
                        if float(q.sl or 0.0) != 0.0]
                for q in held:
                    if not send:
                        print(f"  would CLEAR SL on {bsym} #{q.ticket} "
                              f"(currently {q.sl})")
                        continue
                    try:
                        conn.modify_position(q.ticket, sl=0.0, tp=float(q.tp or 0.0))
                        cleared += 1
                    except Exception as exc:  # noqa: BLE001
                        failed.append(f"{bsym}#{q.ticket}: {exc}")
            print(f"  compliance SL: DISABLED in config"
                  + (f" — cleared {cleared} broker-side stop(s)" if cleared else "")
                  + (" — none outstanding" if not cleared and not failed else ""))
            for msg in failed:
                print(f"  !! SL clear FAILED {msg}")
        if sl_cap:
            print(f"  compliance SL: cap {float(sl_cap):.1%} of equity per symbol")
            for esym in sorted(targets):
                bsym = symmap.get(esym, {}).get("fp_symbol")
                if not bsym:
                    continue
                note = apply_compliance_sl(conn, bsym, magic, equity, float(sl_cap), send)
                if note:
                    flag = "  !! " if ("REJECT" in note or "REDUCE" in note
                                       or "cannot" in note) else "    "
                    print(f"{flag}{note}")

        if bg_on:
            for esym, meta in symmap.items():
                b = meta.get("fp_symbol")
                if b:
                    book_lots[b] = round(held_lots(conn, b, magic), 2) if send else \
                        book_lots.get(b, 0.0)
            state["book_lots"] = book_lots
            state["awaiting_bottom"] = awaiting
            save_state(state_path, state)
            if bg_waiting:
                print(f"  BOTTOM GATE holding {len(bg_waiting)} sleeve(s) flat: "
                      f"{', '.join(bg_waiting)}")
            if bg_fired:
                print(f"  BOTTOM GATE released on a fire: {', '.join(bg_fired)}")
            if bg_timeout:
                print(f"  BOTTOM GATE timed out (market entry): {', '.join(bg_timeout)}")
        if sess_skipped:
            print(f"  EVENING GATE held {len(sess_skipped)} sleeve(s) flat/unchanged: "
                  f"{', '.join(sess_skipped)} — by design, lifts at "
                  f"{_unt[0]:02d}:{_unt[1]:02d} server")
        if blocked:
            print(f"  *** {len(blocked)} SLEEVE(S) BROKER-BLOCKED: "
                  f"{', '.join(blocked)} — book is running incomplete ***")

        if n_mapped == 0:
            print("  PLAN: DRY-RUN — no fp_symbol mapped yet (targets shown above; "
                  "fill configs/v5_basket_challenge.json symbols after account purchase)")
        elif not plans:
            print("  PLAN: in sync — nothing to do")
        row.update(n_mapped=n_mapped,
                   plan="; ".join(f"{s}:{h:.2f}->{t:.2f}" for s, h, t in plans) or "in_sync",
                   blocked="; ".join(blocked + [f"{s}:EVENING_GATE" for s in sess_skipped]))
        if not send and plans:
            print("  (dry plan — rerun with --live --execute to send)")
        if args.paper_csv:
            log_row(args.paper_csv, row)
    finally:
        conn.disconnect()


if __name__ == "__main__":
    main()
