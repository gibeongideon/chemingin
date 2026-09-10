#!/bin/bash
# Configure the VPS-side zigzag notifier to send via Resend over HTTPS.
#
# WHY THIS SCRIPT RATHER THAN AN INLINE COMMAND: the key is read with `read -rs`, so it never
# appears on screen, in `ps`, or in bash history. .env.notify is created with umask 177 so it is
# mode 600 from the instant it exists, never briefly world-readable.
#
# Resend free tier note: without a verified domain it delivers only to the address that OWNS the
# Resend account. Sign up with the address you want alerts at, or verify it in their dashboard.
set -eu
cd /home/trader/MT5

read -rsp "Resend API key (starts re_, input hidden): " KEY; echo
[ -n "$KEY" ] || { echo "no key entered, aborting"; exit 1; }
read -rp  "Deliver alerts to [kipngenol@gmail.com]: " TO
TO=${TO:-kipngenol@gmail.com}

umask 177
printf 'RESEND_API_KEY=%s\nMAIL_TO=%s\n' "$KEY" "$TO" > .env.notify
unset KEY
echo "wrote .env.notify ($(stat -c %a .env.notify)) for $TO"

echo "--- channel now selected ---"
/home/trader/miniconda3/envs/envmt5/bin/python -c "
import sys; sys.path.insert(0,'.')
from scripts.v5_notify import channel, env
print(' ', channel(env()))
"
echo "--- test send (forced, so it proves delivery even with no signal) ---"
/home/trader/miniconda3/envs/envmt5/bin/python scripts/v5_zigzag_notify.py --always
