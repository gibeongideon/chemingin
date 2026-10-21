#!/usr/bin/env python3
"""Futures roll manager for the swap-free sleeves found in 3by. Read-only unless --execute.

WHY A ROLL MANAGER IS NEEDED AT ALL. 3by found GC.Z6s -- swap-free gold tracking XAUUSDs at
return-correlation 0.9990, worth +0.05 to +0.11 book Sharpe. But a future EXPIRES, so holding the
exposure means rolling, and a roll that is missed turns a position into a forced close at whatever
the broker decides.

*** TWO THINGS MT5 DOES NOT TELL YOU HERE, BOTH MEASURED 2026-10-21. ***

1. `expiration_time` IS ZERO on every Switch futures symbol (checked GC/NQ100/US30: all 0, with
   expiration_mode 15). So the expiry date CANNOT be read from the terminal and must be derived
   from the contract code: GC.Z6s = Z (December) + 6 (2026).

2. SWITCH LISTS ONLY ONE GOLD CONTRACT. The census of all 2,067 symbols found exactly `GC.Z6s`
   and no successor -- same for NQ100, US30, ES500, SI, DE40, UK100. Only BRN and CL carry several
   months (BRN: Q6/V6/X6 close-only, Z6/F7 full). **So the successor contract does not exist yet
   and the roll cannot be pre-wired to a known target.** It has to DISCOVER the next contract when
   the broker lists it, and the real operational risk is that one never appears -- in which case
   the sleeve must fall back to spot XAUUSDs rather than silently go flat.

THE ROLL SIGNAL IS `trade_mode`, NOT A DATE. The census shows expiring contracts drop from
trade_mode 4 (FULL) to 3 (CLOSE_ONLY) while remaining listed -- BRN.Q6s/V6s/X6s are all sitting at
3 right now. That transition is the broker's own statement that the contract is retiring, it needs
no expiry calendar, and it cannot drift out of sync with the broker's schedule the way a hard-coded
date can. A date is still derived from the code, but only to ORDER the contracts and to warn early.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v5_runs" / "futures_roll"

# CME/ICE delivery month codes
MONTH = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
         "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}
FULL, CLOSE_ONLY = 4, 3


def parse_contract(name: str):
    """'GC.Z6s' -> ('GC', 2026, 12). Returns None if the name is not a futures code."""
    if "." not in name:
        return None
    root, tail = name.split(".", 1)
    tail = tail[:-1] if tail.endswith("s") else tail
    if len(tail) < 2 or tail[0] not in MONTH or not tail[1:].isdigit():
        return None
    yr = int(tail[1:])
    # single digit -> current decade; this needs revisiting in 2030 and the assert says so
    year = 2020 + yr if yr < 10 else 2000 + yr
    assert 2020 <= year <= 2039, "contract year %s out of supported range" % year
    return root, year, MONTH[tail[0]]


def chain(m, root: str) -> list[dict]:
    """Every listed contract for a root, ordered by delivery, with its tradability."""
    rows = []
    for s in m.symbols_get():
        p = parse_contract(s.name)
        if not p or p[0] != root:
            continue
        si = m.symbol_info(s.name)
        if si is None:
            continue
        rows.append(dict(name=s.name, root=p[0], year=p[1], month=p[2],
                         delivery="%04d-%02d" % (p[1], p[2]),
                         trade_mode=int(si.trade_mode),
                         contract_size=float(si.trade_contract_size),
                         volume_min=float(si.volume_min),
                         swap_mode=int(getattr(si, "swap_mode", -1))))
    return sorted(rows, key=lambda r: (r["year"], r["month"]))


def front(ch: list[dict]):
    """Nearest contract the broker still lets you OPEN."""
    return next((c for c in ch if c["trade_mode"] == FULL), None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18815)
    ap.add_argument("--roots", default="GC,BRN,CL,NQ100,US30,ES500,SI")
    ap.add_argument("--held-magic", type=int, default=360596)
    ap.add_argument("--spot-ref", default="XAUUSDs", help="basis reference for the GC sleeve")
    ap.add_argument("--execute", action="store_true", help="actually roll (default: report only)")
    a = ap.parse_args()

    from mt5linux import MetaTrader5
    m = MetaTrader5(host="localhost", port=a.port)
    if not m.initialize():
        print("init failed: %s" % (m.last_error(),)); return 2
    ai = m.account_info()
    print("account %s on %s (trade_mode %s)" % (ai.login, ai.server, ai.trade_mode))

    report = {"utc": datetime.now(timezone.utc).isoformat(), "roots": {}}
    print("\n%-8s %-11s %-10s %-11s %s" % ("root", "front", "delivery", "successor", "status"))
    print("-" * 74)
    for root in [r.strip() for r in a.roots.split(",") if r.strip()]:
        ch = chain(m, root)
        if not ch:
            print("%-8s %-11s %-10s %-11s %s" % (root, "-", "-", "-", "NOT LISTED"))
            continue
        f = front(ch)
        later = [c for c in ch if f and (c["year"], c["month"]) > (f["year"], f["month"])
                 and c["trade_mode"] == FULL]
        succ = later[0]["name"] if later else None
        if f is None:
            status = "*** NO OPENABLE CONTRACT -- every month is CLOSE_ONLY ***"
        elif succ is None:
            status = "front only; NO SUCCESSOR LISTED YET (roll target unknown)"
        else:
            status = "successor available"
        print("%-8s %-11s %-10s %-11s %s" % (root, f["name"] if f else "-",
                                             f["delivery"] if f else "-", succ or "-", status))
        report["roots"][root] = dict(chain=ch, front=f["name"] if f else None,
                                     successor=succ, status=status)

    # --- held futures positions and whether any needs rolling -------------------------------
    held = [p for p in (m.positions_get() or []) if parse_contract(p.symbol)]
    print("\nHELD FUTURES POSITIONS")
    if not held:
        print("  none")
    actions = []
    for p in held:
        root = parse_contract(p.symbol)[0]
        ch = chain(m, root)
        mine = next((c for c in ch if c["name"] == p.symbol), None)
        f = front(ch)
        tm = mine["trade_mode"] if mine else -1
        need = (tm == CLOSE_ONLY)
        tgt = f["name"] if (f and f["name"] != p.symbol) else None
        print("  %-10s %.2f @ %9.2f  swap %+7.2f  trade_mode %s  magic %s  -> %s" % (
            p.symbol, p.volume, p.price_open, p.swap, tm, p.magic,
            ("ROLL NEEDED to %s" % tgt) if (need and tgt) else
            ("ROLL NEEDED but NO TARGET LISTED" if need else "ok, still openable")))
        if need:
            actions.append((p, tgt))

    # --- basis, re-measured every run (the plan requires it at each roll) --------------------
    gc = report["roots"].get("GC", {}).get("front")
    if gc:
        m.symbol_select(gc, True); m.symbol_select(a.spot_ref, True); time.sleep(4)
        tf, ts = m.symbol_info_tick(gc), m.symbol_info_tick(a.spot_ref)
        if tf and ts and tf.ask and ts.ask:
            fm, sm = (tf.bid + tf.ask) / 2, (ts.bid + ts.ask) / 2
            basis = (fm / sm - 1) * 100
            print("\nBASIS  %s %.2f vs %s %.2f = %+.3f%%" % (gc, fm, a.spot_ref, sm, basis))
            print("  (3by measured +1.532%% -> +0.491%% over 92 days = -4.13%%/yr realised;")
            print("   spot swap is -5.76%%/yr, so the future is cheaper while basis/yr < 5.76%%)")
            report["basis_pct"] = basis
            report["basis_pair"] = [gc, a.spot_ref]

    if actions and a.execute:
        print("\nEXECUTING ROLL(S)")
        for p, tgt in actions:
            if not tgt:
                print("  %s: no target, SKIPPED (fall back to spot manually)" % p.symbol); continue
            si_new = m.symbol_info(tgt)
            fmask = int(si_new.filling_mode)
            fill = (m.ORDER_FILLING_FOK if fmask & 1 else
                    m.ORDER_FILLING_IOC if fmask & 2 else m.ORDER_FILLING_RETURN)
            # close the expiring leg
            tk = m.symbol_info_tick(p.symbol)
            close_req = dict(action=m.TRADE_ACTION_DEAL, symbol=p.symbol, volume=p.volume,
                             type=m.ORDER_TYPE_SELL if p.type == 0 else m.ORDER_TYPE_BUY,
                             position=p.ticket, price=tk.bid if p.type == 0 else tk.ask,
                             deviation=30, magic=p.magic, comment="roll-out",
                             type_filling=fill)
            rc = m.order_send(close_req)
            print("  close %s: retcode %s" % (p.symbol, rc.retcode))
            if rc.retcode != 10009:
                print("    FAILED, not opening the new leg:", getattr(rc, "comment", "")); continue
            # re-open matched NOTIONAL, not matched lots -- contract sizes can differ between months
            tn = m.symbol_info_tick(tgt)
            old_notional = p.volume * (m.symbol_info(p.symbol).trade_contract_size) * rc.price
            vol = round(old_notional / (si_new.trade_contract_size * tn.ask) / si_new.volume_step)\
                * si_new.volume_step
            vol = max(vol, si_new.volume_min)
            open_req = dict(action=m.TRADE_ACTION_DEAL, symbol=tgt, volume=vol,
                            type=m.ORDER_TYPE_BUY if p.type == 0 else m.ORDER_TYPE_SELL,
                            price=tn.ask if p.type == 0 else tn.bid, deviation=30,
                            magic=p.magic, comment="roll-in", type_filling=fill)
            ro = m.order_send(open_req)
            print("  open  %s %.2f: retcode %s @ %.2f" % (tgt, vol, ro.retcode,
                                                          getattr(ro, "price", 0)))
    elif actions:
        print("\n  %d roll(s) needed -- rerun with --execute to send" % len(actions))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "roll_status.json").write_text(json.dumps(report, indent=1, default=str))
    print("\n-> %s" % (OUT / "roll_status.json"))
    m.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
