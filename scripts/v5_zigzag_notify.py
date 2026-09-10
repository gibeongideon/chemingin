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

    python scripts/v5_zigzag_notify.py            # send if a band was entered
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

STATE = ROOT / "data" / "v5_runs" / "zigzag_ftmo_state.json"
SEEN = ROOT / "data" / "v5_runs" / "zigzag_notify_seen.json"   # separate from the desktop's
STALE_ALERT_H = 6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--always", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    ch = channel(env())
    print(f"channel: {ch}")
    if ch == "smtp":
        print("  WARNING: only SMTP is configured, and this host cannot reach SMTP. "
              "Add an HTTPS provider key to .env.notify (see the module docstring).")

    send = (lambda subj, body: print(f"[dry] {subj}\n\n{body}")) if args.dry else notify

    if not STATE.exists():
        send("[zigzag] PIPELINE BROKEN",
             f"The executor has written no state file at {STATE}.\n"
             f"Nothing was computed, so there is no instruction.\n\n"
             f"  systemctl --user status zigzag-obs.timer\n"
             f"  tail -30 {ROOT}/data/v5_runs/zigzag-obs.log")
        sys.exit("no state file — warning sent")

    try:
        s = json.loads(STATE.read_text())
    except Exception as e:
        send("[zigzag] PIPELINE BROKEN",
             f"State file at {STATE} is unreadable: {type(e).__name__}: {e}")
        sys.exit(f"state unreadable ({e}) — warning sent")

    age = (datetime.now(timezone.utc)
           - datetime.fromisoformat(s["computed_utc"])).total_seconds() / 3600
    m = build(s, load_band(SEEN))
    print(m["body"])

    if age > STALE_ALERT_H:
        send("[zigzag] PIPELINE STALE",
             f"Newest harness state is {age:.0f}h old (computed {s['computed_utc']}).\n"
             f"'no fire' can no longer be trusted to mean the market is quiet.\n\n"
             f"  systemctl --user status zigzag-obs.timer\n\n"
             f"Last known state:\n\n{m['body']}")
        sys.exit(f"state {age:.0f}h stale — warning sent")

    if m["should_send"] or args.always:
        ok = send(m["subject"], m["body"])
        if ok is False:
            sys.exit("send FAILED — see the channel error above")
    else:
        print(f"\n(nothing sent — band {m['band']}; --always to send anyway)")
    if not args.dry:
        save_band(SEEN, m["band"], s.get("prob"))


if __name__ == "__main__":
    main()
