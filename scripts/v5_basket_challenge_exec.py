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


def load_state(path: Path, acct) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    st = cg.init_state(float(acct.balance), float(acct.equity))
    print(f"  state INIT: initial_balance {st['initial_balance']:,.2f}  "
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

    conn = MT5Connector()
    conn.connect()
    try:
        acct = conn.account_info()
        equity, balance = float(acct.equity), float(acct.balance)
        # real-account order lock (mirror of dual/challenge bots)
        is_demo = "demo" in str(getattr(acct, "server", "")).lower()
        send = args.execute and (is_demo or args.live)

        state = load_state(state_path, acct)
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
                   blocked="; ".join(blocked))
        if not send and plans:
            print("  (dry plan — rerun with --live --execute to send)")
        if args.paper_csv:
            log_row(args.paper_csv, row)
    finally:
        conn.disconnect()


if __name__ == "__main__":
    main()
