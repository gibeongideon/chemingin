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


# Cloudflare sits in front of api.resend.com (and others) and rejects Python's default
# `Python-urllib/3.x` User-Agent on client fingerprint, returning a bare `HTTP 403 error code:
# 1010` with no JSON body — which looks exactly like a bad API key and is not one. Verified
# 2026-09-10 from the VPS: default UA 403, browser UA 200, curl UA 200, same key and payload.
# Any ordinary UA passes, so every request sends one.
UA = "Mozilla/5.0 (X11; Linux x86_64) v5-notify/1.0"


def _post(url: str, payload: dict, headers: dict) -> tuple[int, str]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": UA, **headers},
        method="POST")
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


def channels_all(e: dict) -> list:
    """Every configured channel, in priority order."""
    out = []
    if e.get("TELEGRAM_TOKEN") and e.get("TELEGRAM_CHAT_ID"):
        out.append("telegram")
    if e.get("RESEND_API_KEY") and e.get("MAIL_TO"):
        out.append("resend")
    if e.get("BREVO_API_KEY") and e.get("MAIL_TO"):
        out.append("brevo")
    if e.get("SENDGRID_API_KEY") and e.get("MAIL_TO"):
        out.append("sendgrid")
    if e.get("SMTP_HOST") and e.get("SMTP_USER") and e.get("SMTP_PASS"):
        out.append("smtp")
    return out


def notify(subj: str, body: str, short: str | None = None) -> bool:
    """Send via the first configured channel, or ALL of them if NOTIFY_ALL is set.

    *** WHY NOTIFY_ALL EXISTS. *** `channel()` is first-match-wins and telegram outranks every
    mail channel, so simply adding TELEGRAM_TOKEN to a working .env.notify SILENTLY STOPS THE
    EMAIL. That is a surprise worth preventing rather than documenting: set NOTIFY_ALL=1 and
    every configured channel receives the message. Returns True if ANY channel succeeded.

    *** WHY `short` EXISTS (user request, 2026-11-01). *** The two channels are read in
    different postures. Email is read at a desk and wants the caveats — this project's messages
    carry them deliberately, because a number without its provenance is how a research artifact
    turns into a live loss. Telegram is read on a phone, where a 4,000-character caveat block
    buries the one line that matters and trains the reader to ignore the alert.

    So `short` is the PHONE body and `body` stays the FULL one. Telegram takes `short` when it
    is given; every mail channel always takes `body`. Both are built from the same state by the
    same builder, so they cannot disagree about what fired — which is the property the shared
    message modules exist to guarantee.
    """
    e = env()
    if str(e.get("NOTIFY_ALL", "")).strip().lower() in ("1", "true", "yes"):
        chans = channels_all(e)
        if len(chans) > 1:
            # *** A LIST, NOT A GENERATOR. *** `any(gen)` SHORT-CIRCUITS on the first True, so
            # with a generator here the second channel is never evaluated and never sent. That
            # is exactly backwards for NOTIFY_ALL, whose whole purpose is that adding telegram
            # does not stop the email — and the bug hid perfectly, because it only appeared once
            # telegram started SUCCEEDING. Found 2026-11-01: while telegram was returning 400
            # on long messages the email went out fine, and silently stopped the moment the
            # length fix landed. Build the list first, then reduce.
            sent = [_send_one(c, e, subj, _body_for(c, body, short)) for c in chans]
            return any(sent)
    ch = channel(e)
    return _send_one(ch, e, subj, _body_for(ch, body, short))


def _body_for(ch: str, body: str, short: str | None) -> str:
    return short if (short and ch == "telegram") else body


# Telegram hard-caps `text` at 4096 CHARACTERS and returns 400 "message is too long" — it does
# not truncate. Discovered 2026-11-01 when the xau-advisor's full reading (its caveat block alone
# runs past 4k) was silently lost to Telegram while the same message mailed fine. A notifier that
# drops the LONGEST messages is worse than one that drops none: length correlates with how much
# the model had to say. So long bodies are split and sent as numbered parts.
TG_LIMIT = 4096
TG_BODY = 3600                 # leaves room for the subject line and the code fence


def _tg_chunks(body: str) -> list[str]:
    """Split on line boundaries where possible; hard-split any single line over the limit."""
    out, cur = [], ""
    for ln in body.splitlines(keepends=True):
        while len(ln) > TG_BODY:                    # a single monstrous line
            if cur:
                out.append(cur); cur = ""
            out.append(ln[:TG_BODY]); ln = ln[TG_BODY:]
        if len(cur) + len(ln) > TG_BODY:
            out.append(cur); cur = ""
        cur += ln
    if cur or not out:
        out.append(cur)
    return out


def _telegram(e: dict, subj: str, body: str) -> tuple[int, str]:
    """Send, splitting if needed. Returns the LAST failure, or the last success if all passed."""
    url = f"https://api.telegram.org/bot{e['TELEGRAM_TOKEN']}/sendMessage"
    parts = _tg_chunks(body)
    code, resp = 200, "OK"
    for i, part in enumerate(parts, 1):
        head = subj if len(parts) == 1 else f"{subj}  ({i}/{len(parts)})"
        payload = {"chat_id": e["TELEGRAM_CHAT_ID"],
                   "text": f"*{head}*\n```\n{part}\n```", "parse_mode": "Markdown"}
        c, r = _post(url, payload, {})
        if c == 400:
            # An unbalanced `*`, `_` or backtick in the body is a parse error, not a content
            # error, and it fails the whole send. Plain text always goes through.
            payload.pop("parse_mode")
            payload["text"] = f"{head}\n\n{part}"
            c, r = _post(url, payload, {})
        if not (200 <= c < 300):
            code, resp = c, r
    return code, resp


def _send_one(ch: str, e: dict, subj: str, body: str) -> bool:
    """One channel. Never raises — a notifier that crashes takes its own timer down with it."""
    try:
        if ch == "telegram":
            code, resp = _telegram(e, subj, body)
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
