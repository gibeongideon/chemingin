#!/usr/bin/env python3
"""One-command Telegram setup: resolve the destination, write .env.notify, send a test.

Telegram is already the FIRST-priority channel in `scripts/v5_notify.py` and api.telegram.org is
reachable from the VPS on 443, so nothing needs building — only two credentials need to exist in
`.env.notify`. This finds the second one for you and verifies the whole path end to end.

*** A CHANNEL IS NOT A CHAT, AND THE DIFFERENCE IS THE WHOLE SETUP. ***
A bot can message a private chat as soon as you message it first. A CHANNEL ignores it entirely
until the bot is an ADMINISTRATOR of that channel with "Post Messages" on. Until then every send
returns `403 Bot is not a member of the channel chat` — which looks exactly like a bad token and
is not one. So this script verifies membership and post rights BEFORE writing anything, and says
which of the two is missing.

Destination forms, in order of preference:
    --channel @yourchannel     public channel — resolved by username, no discovery needed
    --channel -1001234567890   private channel — numeric id (note the -100 prefix)
    --chat-id <id>             any chat id you already know
    (nothing)                  discovery: scans recent updates for a channel post, a forwarded
                               channel message, or a private chat

USAGE (the token is read from a FILE, never from the command line, so it does not land in shell
history or in the process table where `ps` would show it):

    1. In Telegram, message @BotFather -> /newbot -> pick a name and a username ending in `bot`.
       It replies with a token like 1234567890:AAE...
    2. Put ONLY that token in a file:   printf '%s' '<token>' > /tmp/tgtoken
    3. Add the bot to your channel as an ADMINISTRATOR with "Post Messages" enabled.
    4. python scripts/v5_telegram_setup.py --token-file /tmp/tgtoken --channel @yourchannel
    5. shred -u /tmp/tgtoken

By default it sets NOTIFY_ALL=1 so Telegram is ADDED alongside the existing email rather than
silently replacing it — `channel()` is first-match-wins and telegram outranks resend. Pass
--telegram-only to let it take over instead.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENVF = ROOT / ".env.notify"
API = "https://api.telegram.org/bot%s/%s"


def get(token: str, method: str, params: dict | None = None, timeout: int = 30) -> dict:
    url = API % (token, method)
    if params:
        url += "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:          # Telegram puts the reason in the BODY
        try:
            return json.loads(exc.read().decode())
        except Exception:                           # noqa: BLE001
            return {"ok": False, "description": f"HTTP {exc.code}"}


def discover(token: str, bot_user: str, prefer: str | None = None) -> str | None:
    """Scan recent updates for anything that names a destination.

    Three shapes arrive here, and a channel only ever produces the first two:
      * `channel_post`        — a post made in a channel the bot administrates
      * `forward_from_chat`   — a channel message forwarded into the bot's private chat
      * `message.chat`        — the private chat itself

    `prefer` resolves the common ambiguity without another round trip: once you have messaged
    the bot privately AND given it a channel, both are in the update list forever, so plain
    discovery would stop and ask every time.
    """
    upd = get(token, "getUpdates")
    if not upd.get("ok"):
        print(f"getUpdates failed: {upd.get('description')}")
        return None
    found, kinds = {}, {}
    for u in upd.get("result", []):
        msg = u.get("channel_post") or u.get("edited_channel_post") or u.get("message") or {}
        for ch in (msg.get("forward_from_chat"), msg.get("chat")):
            if ch and ch.get("id"):
                who = ch.get("title") or ch.get("username") or ch.get("first_name") or "?"
                found[str(ch["id"])] = f"{who} ({ch.get('type')})"
                kinds[str(ch["id"])] = ch.get("type")
    if prefer and found:
        want = ("channel", "supergroup", "group") if prefer == "channel" else ("private",)
        hits = [k for k, t in kinds.items() if t in want]
        if len(hits) == 1:
            print(f"destination discovered: {hits[0]}  {found[hits[0]]}  (--prefer {prefer})")
            return hits[0]
        if len(hits) > 1:
            print(f"several {prefer} destinations found; re-run with --chat-id to choose:")
            for k in hits:
                print(f"   {k}  {found[k]}")
            return None
        print(f"no {prefer} destination in recent updates "
              f"(found: {', '.join(found.values()) or 'nothing'})")
        found = {}
    if not found:
        print("\nNo destination found in recent updates. Pick whichever applies:\n"
              f"  CHANNEL : add @{bot_user} to the channel as an ADMINISTRATOR with "
              "\"Post Messages\" on,\n            then post any message there and re-run — or "
              "just pass --channel @yourchannel\n"
              f"  PRIVATE : open @{bot_user} in Telegram and send it any text, then re-run")
        return None
    if len(found) > 1:
        print("several destinations found; re-run with --chat-id to choose:")
        for k, v in found.items():
            print(f"   {k}  {v}")
        return None
    cid, who = next(iter(found.items()))
    print(f"destination discovered: {cid}  {who}")
    return cid


def verify(token: str, chat_id: str, bot_id: int, bot_user: str) -> bool:
    """Confirm the chat exists AND that the bot may post into it, before writing credentials.

    A channel that has not been given the bot as an admin accepts `getChat` perfectly well and
    then refuses every `sendMessage`. Checking membership here turns that into one clear line
    instead of a silent dead notifier discovered weeks later.
    """
    chat = get(token, "getChat", {"chat_id": chat_id})
    if not chat.get("ok"):
        print(f"cannot reach {chat_id}: {chat.get('description')}")
        if str(chat_id).startswith("@"):
            print("  a @username only resolves for PUBLIC channels — a private one needs its "
                  "numeric -100... id")
        return False
    c = chat["result"]
    kind = c.get("type")
    print(f"destination: {c.get('title') or c.get('username') or c.get('first_name')} "
          f"[{kind}] id={c['id']}")

    if kind == "private":
        return True

    mem = get(token, "getChatMember", {"chat_id": chat_id, "user_id": bot_id})
    if not mem.get("ok"):
        print(f"  cannot read the bot's membership: {mem.get('description')}")
        return False
    st = mem["result"].get("status")
    if st not in ("administrator", "creator"):
        print(f"  *** @{bot_user} is '{st}' in this {kind}, not an administrator. ***\n"
              f"  Telegram will refuse every send. Open the {kind} -> Administrators -> "
              f"Add Admin -> @{bot_user}, and enable \"Post Messages\".")
        return False
    if mem["result"].get("can_post_messages") is False:
        print(f"  *** @{bot_user} is an admin but \"Post Messages\" is OFF. *** Turn it on.")
        return False
    print(f"  @{bot_user} is an administrator and may post")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-file",
                    help="file containing ONLY the BotFather token; omit to reuse the token "
                         "already in .env.notify")
    ap.add_argument("--channel", help="@username of a public channel, or its numeric -100... id")
    ap.add_argument("--chat-id", help="any chat id you already know (same as --channel)")
    ap.add_argument("--prefer", choices=("channel", "private"),
                    help="when discovery finds several, take this kind")
    ap.add_argument("--telegram-only", action="store_true",
                    help="let telegram REPLACE email instead of adding to it")
    ap.add_argument("--dry", action="store_true", help="do not write .env.notify")
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from scripts.v5_notify import env as notify_env                   # noqa: E402

    # Changing WHERE messages go should not need the token again. It is already in .env.notify
    # at mode 600, and re-asking for it means it gets pasted somewhere a second time.
    if a.token_file:
        tok = Path(a.token_file).read_text().strip()
    else:
        tok = notify_env().get("TELEGRAM_TOKEN", "")
        if not tok:
            print("no --token-file and no TELEGRAM_TOKEN in .env.notify"); return 1
        print("reusing the token already in .env.notify")
    if ":" not in tok or len(tok) < 20:
        print("that does not look like a BotFather token"); return 1

    me = get(tok, "getMe")
    if not me.get("ok"):
        print(f"token rejected by Telegram: {me.get('description')}"); return 1
    bot = me["result"]
    bot_user = bot.get("username")
    print(f"bot verified: @{bot_user} ({bot.get('first_name')})")

    chat_id = a.channel or a.chat_id or discover(tok, bot_user, a.prefer)
    if not chat_id:
        return 1
    if not verify(tok, chat_id, bot["id"], bot_user):
        return 1

    lines, _ = [], None
    if ENVF.exists():
        for ln in ENVF.read_text().splitlines():
            k = ln.split("=", 1)[0].strip()
            if k in ("TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID", "NOTIFY_ALL"):
                continue
            lines.append(ln)
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

    from scripts.v5_notify import notify, channel, env, channels_all   # noqa: E402
    e = env()
    print(f"channels now configured: {channels_all(e)}  (primary: {channel(e)})")
    ok = notify("[advisor] Telegram wired up",
                "If you can read this, the VPS can reach this channel.\n"
                "What will arrive here:\n"
                "  * intraday advisor — hourly on change, 4h heartbeat\n"
                "  * zigzag detector  — on a fire, on the 0.55-0.60 watch band, on a stale feed\n"
                "This path is REPORT ONLY — it holds no order functions.")
    print("test send:", "OK" if ok else "FAILED")
    print("\nNow shred the token file:  shred -u", a.token_file)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
