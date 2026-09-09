"""EMAIL RELAY for the ZigZag observation harness — VPS computes, this host mails.

The VPS cannot reach smtp.gmail.com on ANY port (25/465/587/2525 all blocked; verified
2026-09-07 against four providers — it is a port block, not a destination block). This desktop
reaches 587 and 465 fine. So the executor on the VPS writes
`data/v5_runs/zigzag_ftmo_state.json` every hour and this script mails it.

*** NEGATIVE EXPECTANCY. *** V5_FINDINGS §3x: walk-forward SR -0.76, 0 of 8 years better than
buy-and-hold, DSR 0.000. Every message carries that line. A notifier that quietly hands over
trade instructions from a disproven model is how a research artifact becomes a live loss.

SILENCE IS NEVER AMBIGUOUS. It mails when a trade fires AND when the pipeline is unreadable or
stale. A month of FTMO/FundingPips report emails was once lost to a blocked SMTP port with
nobody noticing, so "no email" must never be indistinguishable from "nothing happened".

    python scripts/v5_zigzag_mail_relay.py            # mail only if a trade fired
    python scripts/v5_zigzag_mail_relay.py --always   # mail regardless (delivery test)
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
REMOTE_JSON = "/home/trader/MT5/data/v5_runs/zigzag_ftmo_state.json"
STALE_ALERT_H = 6
DISCLAIMER = ("NEGATIVE EXPECTANCY — walk-forward SR -0.76, 0/8 years better than buy&hold, "
              "DSR 0.000 (V5_FINDINGS 3x). Observation harness, not a recommendation.")


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
    args = ap.parse_args()

    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                        VPS, f"cat {REMOTE_JSON}"], capture_output=True, text=True, timeout=60)
    if r.returncode != 0 or not r.stdout.strip():
        why = r.stderr.strip() or "empty file"
        mail("[zigzag] PIPELINE BROKEN",
             f"Could not read the harness state from the VPS.\n\n  {why}\n\n"
             f"Nothing was computed, so there is no instruction.\n\n"
             f"  ssh {VPS} 'systemctl --user status zigzag-obs.timer'\n"
             f"  ssh {VPS} 'tail -30 /home/trader/MT5/data/v5_runs/zigzag-obs.log'\n\n"
             f"{DISCLAIMER}")
        sys.exit(f"state unreadable ({why}) — warning mailed")

    s = json.loads(r.stdout)
    age = (datetime.now(timezone.utc)
           - datetime.fromisoformat(s["computed_utc"])).total_seconds() / 3600

    lines = [
        f"account   {s['login']} @ {s['server']}   equity ${s['equity']:,.2f}",
        f"symbol    {s['symbol']}   feed {s['bars']:,} closed H1 bars",
        f"last bar  {s['last_bar']}   ({s['stale_hours']:.1f}h ago)",
        "",
        f"P(bottom) {s['prob']:.4f}   threshold {s['threshold']:.2f}   "
        f"-> {'FIRES' if s['fires'] else 'no fire'}",
        f"open      {s['open_positions']} position(s) under the harness magic",
        "",
    ]
    if s["actions"]:
        for a in s["actions"]:
            lines.append(f"ACTION    {a['kind']} {a['vol']} lots   ({a['why']})")
        lines.append("")
        lines.append("The demo bot has already sent these. To mirror it by hand elsewhere:")
        for a in s["actions"]:
            if a["kind"] == "BUY":
                lines.append(f"  BUY {a['vol']} lots XAUUSD, then attach TP +0.60% and SL -0.50%")
                lines.append(f"  close it after 48 hours if neither has been hit")
            else:
                lines.append(f"  CLOSE {a['vol']} lots XAUUSD ({a['why']})")
    else:
        lines.append("ACTION    none")
    lines += ["", f"ticket computed {s['computed_utc']} ({age:.1f}h ago)", "", DISCLAIMER]
    body = "\n".join(lines)
    print(body)

    if age > STALE_ALERT_H:
        mail("[zigzag] PIPELINE STALE",
             f"Newest harness state is {age:.0f}h old (computed {s['computed_utc']}).\n"
             f"'no fire' can no longer be trusted to mean the market is quiet.\n\n"
             f"  ssh {VPS} 'systemctl --user status zigzag-obs.timer'\n\n"
             f"Last known state:\n\n{body}")
        sys.exit(f"state {age:.0f}h stale — warning mailed")

    if s["actions"] or args.always:
        act = s["actions"][0]["kind"] if s["actions"] else "no action"
        subj = (f"[zigzag] {act} {s['symbol']} — P(bottom) {s['prob']:.2f}" if s["actions"]
                else f"[zigzag] no action (P {s['prob']:.2f})")
        mail(subj, body)
    else:
        print(f"\n(no fire -> nothing sent; --always to mail anyway)")


if __name__ == "__main__":
    main()
