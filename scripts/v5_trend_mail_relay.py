"""GMAIL relay for the trend-follower alert — VPS computes, this desktop mails.

WHY BOTH PATHS EXIST. `v5_trend_signal.py` already sends from the VPS over HTTPS (Resend),
because the VPS cannot reach any SMTP port. But Resend's free tier sends from the SHARED test
domain `onboarding@resend.dev`, which Gmail routinely files as spam — so VPS delivery is
"accepted by Resend" (HTTP 200), not "in your inbox". Gmail SMTP from this host is the path
with proven inbox delivery, so it runs as well. Same belt-and-braces split as the zigzag
harness, and it earned its keep there: the first live TRADABLE fire arrived via the desktop
while the VPS path was still broken.

It reads the VPS's own `trend_signal_state.json`, so the two senders can never disagree about
what the champion wants — the body is the one the VPS already composed, not a re-derivation.

DEDUPE IS INDEPENDENT of the VPS sender's, deliberately: if the Resend copy is silently going to
spam, this path must still deliver, so it keeps its own baseline of last-notified targets.

    python scripts/v5_trend_mail_relay.py           # mail only if a target moved
    python scripts/v5_trend_mail_relay.py --always  # mail regardless (delivery test)
    python scripts/v5_trend_mail_relay.py --dry     # print only
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
sys.path.insert(0, str(ROOT))
VPS = "trader@68.183.91.240"
REMOTE_JSON = "/home/trader/MT5/data/v5_runs/trend_signal_state.json"
SEEN = ROOT / "data" / "v5_runs" / "trend_relay_seen.json"
STALE_ALERT_H = 30          # XAU H4 / others D1, so a day-plus of silence is the red line
BAND, MIN_LOTS = 0.10, 0.02


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
    ap.add_argument("--always", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                        VPS, f"cat {REMOTE_JSON}"], capture_output=True, text=True, timeout=90)
    if r.returncode != 0 or not r.stdout.strip():
        why = r.stderr.strip() or "empty file"
        msg = (f"Could not read the trend-signal state from the VPS.\n\n  {why}\n\n"
               f"  ssh {VPS} 'systemctl --user status trend-signal.timer'\n"
               f"  ssh {VPS} 'tail -40 /home/trader/MT5/data/v5_runs/trend-signal.log'")
        print(msg)
        if not args.dry:
            mail("[trend] PIPELINE BROKEN", msg)
        sys.exit(1)

    s = json.loads(r.stdout)
    age = (datetime.now(timezone.utc)
           - datetime.fromisoformat(s["computed_utc"])).total_seconds() / 3600
    body = s.get("body", "(no body in state)") + f"\n\ncomputed {s['computed_utc']} ({age:.1f}h ago)"
    print(body)
    if args.dry:
        return

    if age > STALE_ALERT_H:
        mail("[trend] PIPELINE STALE",
             f"Newest trend state is {age:.0f}h old (computed {s['computed_utc']}).\n"
             f"'no change' can no longer be trusted.\n\n"
             f"  ssh {VPS} 'systemctl --user status trend-signal.timer'\n\n{body}")
        sys.exit(1)

    # independent dedupe: compare targets against what THIS path last delivered
    prev = {}
    try:
        prev = json.loads(SEEN.read_text()).get("sleeves", {})
    except Exception:
        pass
    cur = s.get("sleeves", {})
    changes = []
    for name, d in cur.items():
        if d.get("error"):
            continue
        new = d["target_lots"]
        old = prev.get(name, {}).get("target_lots")
        if old is None or abs(new - old) > max(MIN_LOTS, BAND * abs(old)):
            changes.append(name)

    if changes or args.always:
        mail(s.get("subject", "[trend] update"), body)
        SEEN.parent.mkdir(parents=True, exist_ok=True)
        SEEN.write_text(json.dumps(dict(
            at=datetime.now(timezone.utc).isoformat(timespec="seconds"), sleeves=cur)))
    else:
        print("\n(no target moved -> nothing sent; --always to mail anyway)")


if __name__ == "__main__":
    main()
