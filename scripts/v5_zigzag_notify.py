"""ZigZag harness notifier that runs ON THE VPS — no desktop involved.

WHY A SECOND SENDER EXISTS. `v5_zigzag_mail_relay.py` runs on the desktop because **the VPS
cannot reach smtp.gmail.com on any port.** Re-verified 2026-09-10 from the droplet:

    smtp.gmail.com:587      BLOCKED        api.telegram.org:443    OPEN
    smtp.gmail.com:465      BLOCKED        api.resend.com:443      OPEN
    smtp.gmail.com:25       BLOCKED        api.brevo.com:443       OPEN
    smtp-relay.brevo.com:587 BLOCKED       smtp.sendgrid.net:2525  OPEN

So the VPS can absolutely send — it just cannot use a Gmail SMTP password, because the ports
that needs are blocked. It has to go out over HTTPS (or SendGrid's 2525). `v5_notify.py` already
speaks all four of those channels and picks the first one configured, so this file is thin: read
the executor's own state file from disk, build the message with the SHARED builder, hand it to
`notify()`.

SETUP — ONE LINE. Put a credential in `/home/trader/MT5/.env.notify` (chmod 600) and this starts
sending. Any ONE of:

    TELEGRAM_TOKEN=...            # plus TELEGRAM_CHAT_ID=...   free, instant, not email
    RESEND_API_KEY=...            # plus MAIL_TO=you@example.com  free tier, real email
    BREVO_API_KEY=...             # plus MAIL_TO=...              free 300/day, real email
    SENDGRID_API_KEY=...          # plus MAIL_TO=...              uses the open 2525 port

Until then it runs harmlessly and prints the message it WOULD have sent, so the pipeline can be
verified before any account exists.

WHY NOT JUST FORWARD OVER SSH. The desktop relay already does that and it works, but it needs
the desktop powered on. This path is independent of it.

SILENCE IS NEVER AMBIGUOUS. It notifies on a fire, on entering the 0.55-0.60 watch band, and on
a broken or stale state file. A month of report emails was once lost to a blocked SMTP port with
nobody noticing.

*** NEGATIVE EXPECTANCY — §3x walk-forward SR -0.76, DSR 0.000; §3as EV/fire below the drift it
displaces. Demo observation harness only. ***

RESULTS, NOT JUST SIGNALS (2026-11-01). Every message now carries what the deployed magic has
actually DONE — each closed trade marked WIN or LOSS with its net, and the running record — read
from the broker's deal history, never from the executor's state file. A closed trade also sends
on its own, so an outcome is never waiting behind a signal that may not come for days. The
forecast and the outcome come from different sources on purpose; a state file records the order
the bot *sent*, which is not evidence it filled or that it paid what the backtest charged.

    python scripts/v5_zigzag_notify.py            # send if a band was entered or a trade closed
    python scripts/v5_zigzag_notify.py --always   # send regardless (delivery test)
    python scripts/v5_zigzag_notify.py --dry      # print only, never send
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
from scripts.v5_zigzag_message import build, load_band, save_band  # noqa: E402
from scripts.v5_zigzag_results import closed_trades, render  # noqa: E402

RUNS = ROOT / "data" / "v5_runs"
STALE_ALERT_H = 6


def results_block(port: int, magic: int, seen_path: Path, dry: bool) -> tuple[str, list]:
    """(rendered results, newly-closed trades). Never fatal — a dead bridge must not silence
    the signal half of the message, so a failure degrades to a one-line note."""
    try:
        from src.core.mt5_connector import get_mt5          # noqa: PLC0415 — slow, keep lazy
        m = get_mt5("localhost", port)
        if not m.initialize():
            return f"RESULTS: bridge {port} not connected — outcomes unavailable.", []
        acc = m.account_info()
        trades = closed_trades(m, magic)
        block = render(trades, equity=float(acc.equity) if acc else None)
    except Exception as exc:                                # noqa: BLE001
        return f"RESULTS: unavailable ({type(exc).__name__}: {exc}).", []

    try:
        seen = set(json.loads(seen_path.read_text()).get("positions", []))
    except Exception:                                       # noqa: BLE001
        seen = set()
    fresh = [t for t in trades if t["position"] not in seen]
    if fresh and not dry:
        seen_path.parent.mkdir(parents=True, exist_ok=True)
        seen_path.write_text(json.dumps({"positions": sorted(t["position"] for t in trades)}))
    return block, fresh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--always", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--state", default="zigzag_trail_switch_state.json",
                    help="executor state file in data/v5_runs/ (which deployment to report)")
    ap.add_argument("--port", type=int, default=18815, help="bridge for the results half")
    ap.add_argument("--magic", type=int, default=360601, help="magic whose trades to report")
    ap.add_argument("--no-results", action="store_true", help="signal only, skip the bridge")
    args = ap.parse_args()

    STATE = RUNS / args.state
    SEEN = RUNS / f"{Path(args.state).stem}_notify_seen.json"
    RSEEN = RUNS / f"{Path(args.state).stem}_results_seen.json"

    ch = channel(env())
    print(f"channel: {ch}")
    if ch == "smtp":
        print("  WARNING: only SMTP is configured, and this host cannot reach SMTP. "
              "Add an HTTPS provider key to .env.notify (see the module docstring).")

    send = (lambda subj, body: print(f"[dry] {subj}\n\n{body}")) if args.dry else notify

    # --- outcomes first. A trade that closed is news whether or not the detector has anything
    # to say today, and the detector may stay quiet for weeks.
    results, fresh = ("RESULTS: skipped (--no-results).", [])
    if not args.no_results:
        results, fresh = results_block(args.port, args.magic, RSEEN, args.dry)
    print(results)

    for t in fresh:
        verdict = "WIN" if t["net"] > 0 else ("LOSS" if t["net"] < 0 else "FLAT")
        send(f"[zigzag] {verdict} {t['net']:+.2f} — {t['side']} {t['volume']:g} {t['symbol']}",
             f"Closed {t['close_time']:%Y-%m-%d %H:%M} srv after {t['held_h']:.0f}h.\n"
             f"  {t['side']} {t['volume']:g} {t['symbol']}  "
             f"{t['open_price']:.2f} -> {t['close_price']:.2f}\n"
             f"  gross {t['profit']:+.2f}   swap {t['swap']:+.2f}   "
             f"commission {t['commission']:+.2f}\n"
             f"  NET {t['net']:+.2f}"
             + ("" if t["closed_by_us"] else "\n  (closed from outside the bot)")
             + f"\n\n{results}\n\nmagic {args.magic}, bridge {args.port}")

    if not STATE.exists():
        send("[zigzag] PIPELINE BROKEN",
             f"The executor has written no state file at {STATE}.\n"
             f"Nothing was computed, so there is no instruction.\n\n"
             f"  systemctl --user status zigzag-trail-switch.timer\n"
             f"  tail -30 {RUNS}/zigzag-trail-switch.log\n\n{results}")
        sys.exit("no state file — warning sent")

    try:
        s = json.loads(STATE.read_text())
    except Exception as e:
        send("[zigzag] PIPELINE BROKEN",
             f"State file at {STATE} is unreadable: {type(e).__name__}: {e}\n\n{results}")
        sys.exit(f"state unreadable ({e}) — warning sent")

    age = (datetime.now(timezone.utc)
           - datetime.fromisoformat(s["computed_utc"])).total_seconds() / 3600
    m = build(s, load_band(SEEN))
    body = f"{m['body']}\n\n{results}"
    print(m["body"])

    if age > STALE_ALERT_H:
        send("[zigzag] PIPELINE STALE",
             f"Newest harness state is {age:.0f}h old (computed {s['computed_utc']}).\n"
             f"'no fire' can no longer be trusted to mean the market is quiet.\n\n"
             f"  systemctl --user status zigzag-trail-switch.timer\n\n"
             f"Last known state:\n\n{body}")
        sys.exit(f"state {age:.0f}h stale — warning sent")

    if m["should_send"] or args.always:
        ok = send(m["subject"], body)
        if ok is False:
            sys.exit("send FAILED — see the channel error above")
    else:
        print(f"\n(nothing sent — band {m['band']}; --always to send anyway)")
    if not args.dry:
        save_band(SEEN, m["band"], s.get("prob"))


if __name__ == "__main__":
    main()
