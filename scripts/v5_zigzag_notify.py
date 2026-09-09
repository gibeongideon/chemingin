"""NOTIFIER for the ZigZag observation harness — so the same signal can be traded by hand.

The user wants to test the detector manually elsewhere in parallel with the demo bot. This reads
the state the executor writes and notifies through whichever channel is configured
(`scripts/v5_notify.py`: Telegram / Resend / Brevo / SendGrid-2525 / SMTP), so the automated
and manual sides can never disagree about what fired.

*** NEGATIVE EXPECTANCY. *** V5_FINDINGS §3x: walk-forward SR -0.76, 0 of 8 years better than
buy-and-hold, DSR 0.000. Every message this sends carries that line, deliberately — a notifier
that quietly hands you trade instructions from a disproven model is how a research artifact
turns into a live loss.

SILENCE IS NEVER AMBIGUOUS. It notifies when a trade is due AND when the pipeline is broken or
stale. A month of FTMO/FundingPips report emails was once lost to a blocked SMTP port with
nobody noticing, so "no message" must never be the same as "nothing happened".

    python scripts/v5_zigzag_notify.py --dry        # print, never send
    python scripts/v5_zigzag_notify.py              # send only if a trade is due
    python scripts/v5_zigzag_notify.py --always     # send regardless (delivery test)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_notify import notify, channel, env as notify_env  # noqa: E402

STATE = ROOT / "data" / "v5_runs" / "zigzag_ftmo_state.json"
STALE_ALERT_H = 6           # H1 bars: anything older than 6h means the pipeline stalled
DISCLAIMER = ("NEGATIVE EXPECTANCY — walk-forward SR -0.76, 0/8 years better than buy&hold, "
              "DSR 0.000 (V5_FINDINGS 3x). Observation harness, not a recommendation.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--always", action="store_true")
    ap.add_argument("--state", default=str(STATE))
    args = ap.parse_args()

    f = Path(args.state)
    if not f.exists():
        subj = "[zigzag] PIPELINE BROKEN — no state file"
        body = (f"{f} does not exist, so the executor has not run.\n\n"
                f"  ssh trader@68.183.91.240 "
                f"'systemctl --user status zigzag-obs.timer'\n"
                f"  ssh trader@68.183.91.240 "
                f"'tail -30 /home/trader/MT5/data/v5_runs/zigzag-obs.log'\n\n{DISCLAIMER}")
        print(subj + "\n\n" + body)
        if not args.dry:
            notify(subj, body)
        sys.exit(1)

    s = json.loads(f.read_text())
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
        if s.get("executed"):
            lines.append("")
            lines.append("(the demo bot has already sent these; this is for your manual copy)")
        else:
            lines.append("")
            lines.append("(dry on the bot side — nothing was sent automatically)")
    else:
        lines.append("ACTION    none")
    lines += ["",
              "TP and SL are attached server-side; only the 48h max-hold needs the bot.",
              "", DISCLAIMER]
    body = "\n".join(lines)
    print(body)

    if args.dry:
        return

    # health first — a stalled pipeline must not look like a quiet market
    if age > STALE_ALERT_H:
        notify("[zigzag] PIPELINE STALE",
               f"Newest state is {age:.0f}h old (computed {s['computed_utc']}).\n"
               f"'no action' can no longer be trusted to mean 'nothing fired'.\n\n"
               f"  ssh trader@68.183.91.240 "
               f"'systemctl --user status zigzag-obs.timer'\n\n"
               f"Last known state:\n\n{body}")
        sys.exit(1)

    due = bool(s["actions"])
    if due or args.always:
        act = s["actions"][0]["kind"] if due else "no action"
        subj = (f"[zigzag] {act} {s['symbol']} — P(bottom) {s['prob']:.2f}" if due
                else f"[zigzag] no action (P {s['prob']:.2f})")
        ok = notify(subj, body)
        if not ok:
            print(f"[notify] delivery failed on channel '{channel(notify_env())}'")
    else:
        print(f"\n(no action due -> nothing sent; --always to send anyway)")


if __name__ == "__main__":
    main()
