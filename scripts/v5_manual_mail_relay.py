"""RELAY MAILER — fetches the VPS's manual-trade ticket over SSH and emails it from a host
that can actually reach an SMTP server.

Why this split exists (measured 2026-09-07): the VPS cannot reach smtp.gmail.com on ANY port
(25 / 465 / 587 / 2525 all blocked; api.telegram.org:443 IS open, so it is a destination block,
not general egress). The repo's existing daily report timers had therefore been failing silently
with "Network is unreachable" since roughly 2026-08-05 — 49 successful sends, then 14 failures.
This desktop reaches 587 and 465 fine. So: the VPS computes (it owns the MT5 bridge), this host
mails (it owns working SMTP).

The VPS side writes data/v5_runs/manual_alert.json on its own timer. This script reads that
file, and mails only when an action is actually due. Nothing here can place an order.

Deliberately degrades safely: if this machine is off for days, the next run mails the CURRENT
ticket, not a backlog of stale ones — the champion is a multi-week trend follower and the VPS
side reconciles against whatever you actually hold.

    python scripts/v5_manual_mail_relay.py            # mail only if a trade is due
    python scripts/v5_manual_mail_relay.py --always   # mail regardless (delivery test)
"""
from __future__ import annotations

import argparse
import json
import smtplib
import ssl
import subprocess
import sys
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VPS = "trader@68.183.91.240"
REMOTE_JSON = "/home/trader/MT5/data/v5_runs/manual_alert.json"
MAX_TICKET_AGE_H = 30          # a ticket older than this is not worth acting on
STALE_ALERT_H = 30             # past this, mail a HEALTH WARNING even if no trade is due


def creds() -> dict:
    e, mf = {}, ROOT / ".env.mail"
    if not mf.exists():
        sys.exit(f"missing {mf} — needed to send. Copy it from the VPS (chmod 600).")
    for ln in mf.read_text().splitlines():
        if "=" in ln and not ln.strip().startswith("#"):
            k, v = ln.split("=", 1)
            e[k.strip()] = v.strip()
    return e


def mail(subj: str, body: str) -> None:
    e = creds()
    user, pw = e.get("SMTP_USER"), e.get("SMTP_PASS")
    to = e.get("REPORT_TO", user)
    if not (e.get("SMTP_HOST") and user and pw):
        sys.exit("incomplete creds in .env.mail (need SMTP_HOST, SMTP_USER, SMTP_PASS)")
    m = MIMEText(body)
    m["Subject"], m["From"], m["To"] = subj, user, to
    with smtplib.SMTP(e["SMTP_HOST"], int(e.get("SMTP_PORT", "587")), timeout=30) as srv:
        srv.starttls(context=ssl.create_default_context())
        srv.login(user, pw)
        srv.send_message(m)
    print(f"emailed {to}: {subj}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--always", action="store_true", help="send even if no action is due")
    args = ap.parse_args()

    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                        VPS, f"cat {REMOTE_JSON}"], capture_output=True, text=True, timeout=60)

    # SILENCE MUST NEVER BE AMBIGUOUS. This design mails nothing when no trade is due, so a
    # dead pipeline looks exactly like a quiet market — which is precisely how a month of
    # FTMO/FundingPips report emails were lost to a blocked SMTP port without anyone noticing
    # (V5_FINDINGS §3ai). If the ticket cannot be read, or is stale, that is mailed as a
    # health warning rather than swallowed.
    if r.returncode != 0 or not r.stdout.strip():
        why = r.stderr.strip() or "empty file"
        mail("[XAU manual] ALERT PIPELINE BROKEN",
             f"Could not read the ticket from the VPS.\n\n  {why}\n\n"
             f"Nothing was computed, so there is no instruction today. Check:\n"
             f"  ssh {VPS} 'systemctl --user status xau-manual-alert.timer'\n"
             f"  ssh {VPS} 'tail -30 /home/trader/MT5/data/v5_runs/xau-manual-alert.log'\n\n"
             f"A common cause: logging into the same MT5 account from another device can\n"
             f"disconnect the VPS terminal that feeds this alert.")
        sys.exit(f"ticket unreadable ({why}) — health warning mailed")
    t = json.loads(r.stdout)

    age = (datetime.now(timezone.utc)
           - datetime.fromisoformat(t["computed_utc"])).total_seconds() / 3600
    if age > STALE_ALERT_H and not t.get("due"):
        mail("[XAU manual] ALERT PIPELINE STALE",
             f"The newest ticket is {age:.0f}h old (computed {t['computed_utc']}).\n\n"
             f"The VPS side has stopped producing fresh tickets, so 'no action due' can no\n"
             f"longer be trusted to mean the market is quiet.\n\n"
             f"  ssh {VPS} 'systemctl --user status xau-manual-alert.timer'\n"
             f"  ssh {VPS} 'tail -30 /home/trader/MT5/data/v5_runs/xau-manual-alert.log'\n\n"
             f"Last known state:\n\n{t['body']}")
        sys.exit(f"ticket {age:.0f}h stale — health warning mailed")

    if t.get("delivered") is True and not args.always:
        # The VPS delivered it itself (a channel is configured there), so sending again
        # from here would duplicate every alert. The hosts coordinate through the ticket
        # rather than through a flag someone has to remember to flip.
        print(f"already delivered by the VPS — not duplicating (action {t.get('action')})")
        return

    if not t.get("due") and not args.always:
        print(f"no action due (forecast {t['forecast']:.3f}, ticket {age:.1f}h old) "
              f"— nothing sent")
        return

    warn = ""
    if age > MAX_TICKET_AGE_H:
        warn = (f"** This ticket was computed {age:.0f}h ago and may be out of date. "
                f"Check the VPS timer before acting. **\n\n")
    if t.get("due"):
        subj = f"[XAU manual] {t['action']} {abs(t['delta_lots']):.2f} lots {t['trade_symbol']}"
    else:
        subj = f"[XAU manual] no action (forecast {t['forecast']:.2f})"
    body = warn + t["body"] + f"\n\nticket computed {t['computed_utc']} ({age:.1f}h ago)"

    mail(subj, body)


if __name__ == "__main__":
    main()
