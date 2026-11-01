#!/usr/bin/env python3
"""One-command Telegram setup: discover the chat id, write .env.notify, send a test.

Telegram is already the FIRST-priority channel in `scripts/v5_notify.py` and api.telegram.org is
reachable from the VPS on 443, so nothing needs building — only two credentials need to exist in
`.env.notify`. This finds the second one for you and verifies the whole path end to end.

USAGE (the token is read from a FILE, never from the command line, so it does not land in shell
history or in the process table where `ps` would show it):

    1. In Telegram, message @BotFather -> /newbot -> pick a name and a username ending in `bot`.
       It replies with a token like 1234567890:AAE...
    2. Put ONLY that token in a file:   printf '%s' '<token>' > /tmp/tgtoken
    3. Send your new bot any message at all (say "hi"). A bot cannot start a conversation, so
       this step is what creates the chat and makes the chat id discoverable.
    4. python scripts/v5_telegram_setup.py --token-file /tmp/tgtoken
    5. shred -u /tmp/tgtoken

By default it sets NOTIFY_ALL=1 so Telegram is ADDED alongside the existing email rather than
silently replacing it — `channel()` is first-match-wins and telegram outranks resend. Pass
--telegram-only to let it take over instead.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENVF = ROOT / ".env.notify"
API = "https://api.telegram.org/bot%s/%s"


def get(token: str, method: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(API % (token, method), timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-file", required=True,
                    help="file containing ONLY the BotFather token")
    ap.add_argument("--chat-id", help="skip discovery and use this chat id")
    ap.add_argument("--telegram-only", action="store_true",
                    help="let telegram REPLACE email instead of adding to it")
    ap.add_argument("--dry", action="store_true", help="do not write .env.notify")
    a = ap.parse_args()

    tok = Path(a.token_file).read_text().strip()
    if ":" not in tok or len(tok) < 20:
        print("that does not look like a BotFather token"); return 1

    me = get(tok, "getMe")
    if not me.get("ok"):
        print(f"token rejected by Telegram: {me}"); return 1
    bot = me["result"]
    print(f"bot verified: @{bot.get('username')} ({bot.get('first_name')})")

    chat_id = a.chat_id
    if not chat_id:
        upd = get(tok, "getUpdates")
        if not upd.get("ok"):
            print(f"getUpdates failed: {upd}"); return 1
        chats = {}
        for u in upd.get("result", []):
            msg = u.get("message") or u.get("channel_post") or {}
            ch = msg.get("chat") or {}
            if ch.get("id"):
                who = ch.get("username") or ch.get("title") or ch.get("first_name") or "?"
                chats[str(ch["id"])] = f"{who} ({ch.get('type')})"
        if not chats:
            print("\nNo messages found. A bot cannot open a conversation — you must message it "
                  f"first.\nOpen Telegram, find @{bot.get('username')}, send any text, then "
                  "re-run this.")
            return 1
        if len(chats) > 1:
            print("several chats found; pass --chat-id to choose:")
            for k, v in chats.items():
                print(f"   {k}  {v}")
            return 1
        chat_id, who = next(iter(chats.items()))
        print(f"chat discovered: {chat_id}  {who}")

    lines, seen = [], set()
    if ENVF.exists():
        for ln in ENVF.read_text().splitlines():
            k = ln.split("=", 1)[0].strip()
            if k in ("TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID", "NOTIFY_ALL"):
                continue
            lines.append(ln); seen.add(k)
    lines += [f"TELEGRAM_TOKEN={tok}", f"TELEGRAM_CHAT_ID={chat_id}"]
    if not a.telegram_only:
        lines.append("NOTIFY_ALL=1")
    had_mail = any(x.startswith(("RESEND_API_KEY", "BREVO_API_KEY", "SENDGRID_API_KEY"))
                   for x in lines)

    if a.dry:
        print("\n[dry] would write .env.notify with TELEGRAM_TOKEN, TELEGRAM_CHAT_ID"
              f"{' and NOTIFY_ALL=1' if not a.telegram_only else ''}")
    else:
        ENVF.write_text("\n".join(lines) + "\n")
        ENVF.chmod(0o600)
        print(f"wrote {ENVF} (mode 600)")
        if had_mail and not a.telegram_only:
            print("  NOTIFY_ALL=1 set -> telegram is ADDED, your email keeps working")
        elif had_mail:
            print("  --telegram-only -> telegram now REPLACES the email channel")

    sys.path.insert(0, str(ROOT))
    from scripts.v5_notify import notify, channel, env, channels_all   # noqa: E402
    e = env()
    print(f"channels now configured: {channels_all(e)}  (primary: {channel(e)})")
    ok = notify("[advisor] Telegram wired up",
                "If you can read this, the VPS can reach you on Telegram.\n"
                "The intraday advisor will arrive here hourly-on-change with a 4h heartbeat.\n"
                "This path is REPORT ONLY — it holds no order functions.")
    print("test send:", "OK" if ok else "FAILED")
    print("\nNow shred the token file:  shred -u", a.token_file)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
