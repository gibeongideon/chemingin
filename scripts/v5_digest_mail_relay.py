"""EMAIL RELAY for the five-finding daily digest — VPS computes, this host mails.

Same split as the zigzag relay and for the same reason: the VPS cannot reach smtp.gmail.com on
any port (25/465/587/2525 all blocked — a port block, not a destination block), while this
desktop reaches 587 fine. The VPS writes `data/v5_runs/signals_digest.json` daily and this
mails its pre-formatted body.

SILENCE IS NEVER AMBIGUOUS: it mails the digest every day it runs, and mails a warning if the
state is unreadable or stale. A month of report emails was once lost to a blocked SMTP port
with nobody noticing.

Only the ZigZag harness trades. Findings 1, 2, 4 and 5 are informational, and two of them are
measurements that deliberately do NOT convert to P&L.

    python scripts/v5_digest_mail_relay.py          # fetch and mail
    python scripts/v5_digest_mail_relay.py --dry    # print only
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
REMOTE_JSON = "/home/trader/MT5/data/v5_runs/signals_digest.json"
STALE_ALERT_H = 30


def mail(subj: str, body: str) -> None:
    e, mf = {}, ROOT / ".env.mail"
    if not mf.exists():
        sys.exit(f"missing {mf}")
    for ln in mf.read_text().splitlines():
        if "=" in ln and not ln.strip().startswith("#"):
            k, v = ln.split("=", 1)
            e[k.strip()] = v.strip()
    user, pw = e.get("SMTP_USER"), e.get("SMTP_PASS")
    to = e.get("REPORT_TO", user)
    if not (e.get("SMTP_HOST") and user and pw):
        sys.exit("incomplete creds in .env.mail")
    m = MIMEText(body)
    m["Subject"], m["From"], m["To"] = subj, user, to
    with smtplib.SMTP(e["SMTP_HOST"], int(e.get("SMTP_PORT", "587")), timeout=30) as srv:
        srv.starttls(context=ssl.create_default_context())
        srv.login(user, pw)
        srv.send_message(m)
    print(f"emailed {to}: {subj}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                        VPS, f"cat {REMOTE_JSON}"], capture_output=True, text=True, timeout=90)
    if r.returncode != 0 or not r.stdout.strip():
        why = r.stderr.strip() or "empty file"
        msg = (f"Could not read the signal digest from the VPS.\n\n  {why}\n\n"
               f"  ssh {VPS} 'systemctl --user status signals-digest.timer'\n"
               f"  ssh {VPS} 'tail -40 /home/trader/MT5/data/v5_runs/signals-digest.log'")
        print(msg)
        if not args.dry:
            mail("[signals] DIGEST PIPELINE BROKEN", msg)
        sys.exit(1)

    s = json.loads(r.stdout)
    age = (datetime.now(timezone.utc)
           - datetime.fromisoformat(s["computed_utc"])).total_seconds() / 3600
    body = s.get("body", "(no body in state)")
    body += f"\n\ndigest computed {s['computed_utc']} ({age:.1f}h ago)"
    print(body)
    if args.dry:
        return

    if age > STALE_ALERT_H:
        mail("[signals] DIGEST STALE",
             f"Newest digest is {age:.0f}h old (computed {s['computed_utc']}).\n\n"
             f"  ssh {VPS} 'systemctl --user status signals-digest.timer'\n\n{body}")
        sys.exit(1)

    z = s.get("zigzag", {})
    xau = s.get("book", {}).get("XAU", {})
    subj = ("[signals] daily — zigzag P {:.2f}{}, XAU fc {:.2f}".format(
        z.get("prob", float("nan")),
        " FIRES" if z.get("fires") else "",
        xau.get("forecast", float("nan"))))
    mail(subj, body)


if __name__ == "__main__":
    main()
