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
DISCLAIMER = "Demo test only — this strategy is expected to lose money (walk-forward -0.76)."


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

    # --- SIMPLE, ACTIONABLE BODY. The order details first; everything else is one line. ---
    if s["actions"]:
        L = []
        for a in s["actions"]:
            if a["kind"] == "BUY":
                L += [
                    f"BUY {s['symbol']}   {a['vol']} lots",
                    "",
                    f"  Entry   {a['entry']}      (market)",
                    f"  SL      {a['sl']}      (-0.50%)",
                    f"  TP      {a['tp']}      (+0.60%)",
                    f"  Close   {a['close_by']} if neither is hit",
                ]
            else:
                L += [
                    f"CLOSE {s['symbol']}   {a['vol']} lots",
                    "",
                    f"  Opened at {a['entry']}, held {a.get('age_h')}h — 48h limit reached.",
                ]
            L.append("")
        L.append("The demo bot has already placed this. Mirror it by hand if you want to.")
        if s.get("arm"):
            L.append(f"Model: {s['arm']}"
                     + (f" (+{', '.join(s.get('aux_used', []))})"
                        if s.get("aux_used") else ""))
    else:
        L = [f"{s['symbol']}   no trade.   P(bottom) {s['prob']:.2f}  vs  "
             f"{s['threshold']:.2f} threshold."]
        if s.get("arm"):
            L.append(f"Model: {s['arm']}"
                     + (f" (+{', '.join(s.get('aux_used', []))})"
                        if s.get("aux_used") else ""))
        if s["open_positions"]:
            L.append(f"Holding {s['open_positions']} open position(s); TP/SL are on the broker.")

    if s.get("aux_rejected") or s.get("aux_missing"):
        bad = [f"{a}" for a, _ in (s.get("aux_rejected") or [])] + \
              [f"{a}" for a, _ in (s.get("aux_missing") or [])]
        L.append(f"(reduced model — unavailable: {', '.join(bad)})")
    L += ["", f"{DISCLAIMER}"]
    body = "\n".join(L)
    print(body)

    if age > STALE_ALERT_H:
        mail("[zigzag] PIPELINE STALE",
             f"Newest harness state is {age:.0f}h old (computed {s['computed_utc']}).\n"
             f"'no fire' can no longer be trusted to mean the market is quiet.\n\n"
             f"  ssh {VPS} 'systemctl --user status zigzag-obs.timer'\n\n"
             f"Last known state:\n\n{body}")
        sys.exit(f"state {age:.0f}h stale — warning mailed")

    if s["actions"] or args.always:
        if s["actions"]:
            a = s["actions"][0]
            # subject carries the order itself, so it is actionable from a phone lock screen
            subj = (f"[zigzag] BUY {a['vol']} lots {s['symbol']} @ {a['entry']}"
                    if a["kind"] == "BUY"
                    else f"[zigzag] CLOSE {a['vol']} lots {s['symbol']}")
        else:
            subj = (f"[zigzag] {s['symbol']} no trade "
                    f"(P {s['prob']:.2f}/{s['threshold']:.2f})")
        mail(subj, body)
    else:
        print(f"\n(no fire -> nothing sent; --always to mail anyway)")


if __name__ == "__main__":
    main()
