#!/bin/bash
# Wire the VPS notifier to a Telegram channel. RUN THIS FROM THE LAPTOP.
#
#   deploy/telegram-setup.sh /tmp/tgtoken @yourchannel      # public channel
#   deploy/telegram-setup.sh /tmp/tgtoken -1001234567890    # private channel, numeric id
#   deploy/telegram-setup.sh /tmp/tgtoken                   # discover from recent updates
#
# WHY A FILE AND NOT AN ARGUMENT. A token passed on a command line lands in shell history and is
# visible to every user on the box through `ps`. It is written to a file, copied, used, and then
# SHREDDED AT BOTH ENDS — including when the setup fails, which is what the trap is for.
#
# The notifier itself lives on the VPS, so the credential must too: `.env.notify` there is what
# the advisor and zigzag timers read. Nothing is stored on the laptop.
set -uo pipefail

TOK="${1:-}"
DEST="${2:-}"
VPS="trader@68.183.91.240"
REMOTE="/home/trader/MT5"
PY="/home/trader/miniconda3/envs/envmt5/bin/python"

[ -n "$TOK" ] && [ -f "$TOK" ] || {
  echo "usage: $0 <token-file> [@channel | -100numeric]"
  echo "  put ONLY the BotFather token in the file:  printf '%s' '<token>' > /tmp/tgtoken"
  exit 1
}

RT=$(ssh -o BatchMode=yes "$VPS" 'f=$(mktemp /tmp/tg.XXXXXXXX); chmod 600 "$f"; echo "$f"') || {
  echo "cannot reach the VPS"; exit 1; }
trap 'ssh -o BatchMode=yes "$VPS" "shred -u $RT 2>/dev/null"; shred -u "$TOK" 2>/dev/null; \
      echo "token shredded on both ends"' EXIT

scp -q -o BatchMode=yes "$TOK" "$VPS:$RT" || { echo "copy failed"; exit 1; }

ARGS=""
[ -n "$DEST" ] && ARGS="--channel $DEST"
# shellcheck disable=SC2029  # $RT/$ARGS are meant to expand locally
ssh -o BatchMode=yes "$VPS" "cd $REMOTE && $PY scripts/v5_telegram_setup.py --token-file $RT $ARGS"
rc=$?

if [ "$rc" -eq 0 ]; then
  echo
  echo "Telegram is live. Both senders pick it up on their next run, no restart needed:"
  echo "  advisor-notify.timer   hourly at :07   (intraday advisor)"
  echo "  zigzag-notify.timer    hourly at :10   (detector + WIN/LOSS results)"
fi
exit "$rc"
