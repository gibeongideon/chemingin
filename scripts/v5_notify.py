"""Multi-channel notifier that works FROM THE VPS. Stdlib only — no new dependencies.

WHY THIS EXISTS. The VPS cannot send mail with `smtplib` on the usual ports: DigitalOcean
blocks outbound SMTP, verified 2026-09-07 against four different providers (587 blocked to
smtp.gmail.com, smtp-relay.brevo.com, smtp.sendgrid.net and smtp.mailgun.org alike). It is a
PORT block, not a destination block — an earlier note in this repo said otherwise, based on a
bad test against smtp.gmail.com:443, a port Gmail does not serve at all.

What IS reachable from the VPS (all verified open 2026-09-07):
    * HTTPS 443 to api.resend.com, api.brevo.com, api.mailgun.net, api.sendgrid.com,
      api.postmarkapp.com, gmail.googleapis.com and api.telegram.org
    * SMTP on port 2525 to smtp.sendgrid.net  (2525 is not in the block list)

So each channel below is chosen because its transport is known to work from this host. The
first channel whose credentials are present in `.env.notify` (falling back to `.env.mail`) is
used, in this order:

    1. telegram  TELEGRAM_TOKEN + TELEGRAM_CHAT_ID   -> HTTPS 443. Zero signup beyond
                 BotFather, ~2 minutes, phone push. The quickest way to make the VPS
                 self-contained.
    2. resend    RESEND_API_KEY + MAIL_TO            -> HTTPS 443, real email. Send from
                 onboarding@resend.dev without owning a domain.
    3. brevo     BREVO_API_KEY + MAIL_TO             -> HTTPS 443, real email, verified
                 sender address rather than a whole domain.
    4. sendgrid  SENDGRID_API_KEY + MAIL_TO          -> SMTP on 2525, the one open SMTP port.
    5. smtp      SMTP_HOST/USER/PASS + REPORT_TO     -> plain smtplib. Works from the laptop,
                 NOT from the VPS. Kept as the fallback and for local runs.

Add exactly one of the above to .env.notify and the VPS needs nothing else.

    python scripts/v5_notify.py --test        # show which channel is configured, send a ping
"""
from __future__ import annotations

import json
import smtplib
import ssl
import sys
import urllib.error
import urllib.request
from email.mime.text import MIMEText
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = 30


def env() -> dict:
    e = {}
    for name in (".env.notify", ".env.mail"):        # .env.notify wins
        f = ROOT / name
        if not f.exists():
            continue
        for ln in f.read_text().splitlines():
            if "=" in ln and not ln.strip().startswith("#"):
                k, v = ln.split("=", 1)
                e.setdefault(k.strip(), v.strip())
    return e


def _post(url: str, payload: dict, headers: dict) -> tuple[int, str]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status, r.read().decode()[:400]
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()[:400]


def channel(e: dict) -> str:
    if e.get("TELEGRAM_TOKEN") and e.get("TELEGRAM_CHAT_ID"):
        return "telegram"
    if e.get("RESEND_API_KEY") and e.get("MAIL_TO"):
        return "resend"
    if e.get("BREVO_API_KEY") and e.get("MAIL_TO"):
        return "brevo"
    if e.get("SENDGRID_API_KEY") and e.get("MAIL_TO"):
        return "sendgrid"
    if e.get("SMTP_HOST") and e.get("SMTP_USER") and e.get("SMTP_PASS"):
        return "smtp"
    return "none"


def notify(subj: str, body: str) -> bool:
    """Send via the first configured channel. Returns True on success. Never raises —
    a notifier that crashes takes its own timer down with it."""
    e = env()
    ch = channel(e)
    try:
        if ch == "telegram":
            code, resp = _post(
                f"https://api.telegram.org/bot{e['TELEGRAM_TOKEN']}/sendMessage",
                {"chat_id": e["TELEGRAM_CHAT_ID"],
                 "text": f"*{subj}*\n```\n{body}\n```", "parse_mode": "Markdown"}, {})
        elif ch == "resend":
            code, resp = _post(
                "https://api.resend.com/emails",
                {"from": e.get("MAIL_FROM", "onboarding@resend.dev"),
                 "to": [e["MAIL_TO"]], "subject": subj, "text": body},
                {"Authorization": f"Bearer {e['RESEND_API_KEY']}"})
        elif ch == "brevo":
            code, resp = _post(
                "https://api.brevo.com/v3/smtp/email",
                {"sender": {"email": e.get("MAIL_FROM", e["MAIL_TO"])},
                 "to": [{"email": e["MAIL_TO"]}], "subject": subj, "textContent": body},
                {"api-key": e["BREVO_API_KEY"]})
        elif ch in ("sendgrid", "smtp"):
            if ch == "sendgrid":
                host, port = "smtp.sendgrid.net", 2525
                user, pw = "apikey", e["SENDGRID_API_KEY"]
                to = e["MAIL_TO"]
                frm = e.get("MAIL_FROM", to)
            else:
                host, port = e["SMTP_HOST"], int(e.get("SMTP_PORT", "587"))
                user, pw = e["SMTP_USER"], e["SMTP_PASS"]
                to = e.get("REPORT_TO", user)
                frm = user
            m = MIMEText(body)
            m["Subject"], m["From"], m["To"] = subj, frm, to
            with smtplib.SMTP(host, port, timeout=TIMEOUT) as srv:
                srv.starttls(context=ssl.create_default_context())
                srv.login(user, pw)
                srv.send_message(m)
            print(f"[notify:{ch}] sent to {to}: {subj}")
            return True
        else:
            print(f"[notify] NO CHANNEL CONFIGURED — would have sent:\n\n{subj}\n\n{body}")
            return False
    except Exception as exc:                       # noqa: BLE001 — must never propagate
        print(f"[notify:{ch}] FAILED: {type(exc).__name__}: {exc}")
        return False

    ok = 200 <= code < 300
    print(f"[notify:{ch}] HTTP {code} {'OK' if ok else resp}")
    return ok


def main() -> None:
    e = env()
    ch = channel(e)
    print(f"configured channel: {ch}")
    present = [k for k in ("TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID", "RESEND_API_KEY",
                           "BREVO_API_KEY", "SENDGRID_API_KEY", "MAIL_TO",
                           "SMTP_HOST", "SMTP_USER", "SMTP_PASS", "REPORT_TO") if e.get(k)]
    print(f"credentials found:  {', '.join(present) or 'none'}")
    if "--test" in sys.argv:
        ok = notify("[XAU manual] notifier test",
                    "If you are reading this, the VPS can reach you without the laptop.")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
