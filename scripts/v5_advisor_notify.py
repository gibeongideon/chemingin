#!/usr/bin/env python3
"""Email the intraday advisor. Runs hourly; sends on CHANGE, with a heartbeat.

Requested 2026-11-01 ("let it run continuously"). The cadence is deliberate rather than literal:

  * it RUNS every hour and always writes to the log, so nothing is lost;
  * it SENDS when the tape state changes, or when 4 hours have passed since the last send.

Hourly email would be 23 messages a day, which is how a feed becomes something you stop opening —
and an advisor whose usual correct answer is FLAT (§3cp) has nothing new to say most hours. Change
plus a heartbeat keeps it continuous without that. `--always` overrides, `--dry` prints instead of
sending, and `--force` ignores the dedupe once.

It sends over HTTPS via `scripts/v5_notify.notify` because SMTP is blocked on this VPS, and it
shells out to the advisor rather than importing it so that a crash in the advisor produces a
"PIPELINE BROKEN" email instead of silence — the failure mode `zigzag-notify` was built to avoid.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_notify import notify, channel, env                   # noqa: E402

STATE = ROOT / "data" / "v5_runs" / "advisor_notify_state.json"
HEARTBEAT_HOURS = 4


def signature(text: str) -> str:
    """The tape states and the session — what a reader would call 'the situation'.

    Deliberately NOT the prices: a 50-cent move is not news, and including it would make every
    hour look like a change and defeat the dedupe.
    """
    labs = re.findall(r"^\s+(M15|M30|H1|H4|D1)\s+[-+0-9.]+\s+(\w+)", text, re.M)
    sess = re.search(r"session ([A-Z /]+)\s+·", text)
    return "|".join(f"{a}:{b}" for a, b in labs) + "||" + (sess.group(1).strip() if sess else "?")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18815)
    ap.add_argument("--symbol", default="XAUUSDs")
    ap.add_argument("--risk-pct", type=float, default=0.5)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--always", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    send = (lambda s, b: print(f"[dry] {s}\n\n{b}")) if a.dry else notify
    py = sys.executable
    cmd = [py, str(ROOT / "scripts" / "v5_intraday_advisor.py"),
           "--port", str(a.port), "--symbol", a.symbol, "--risk-pct", str(a.risk_pct)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(ROOT))
    except subprocess.TimeoutExpired:
        send("[advisor] PIPELINE BROKEN — timeout",
             "v5_intraday_advisor.py did not finish in 600s.")
        return 1
    out = p.stdout or ""
    if p.returncode != 0 or "INTRADAY ADVISOR" not in out:
        send("[advisor] PIPELINE BROKEN",
             f"exit {p.returncode}\n\nstdout tail:\n{out[-1500:]}\n\nstderr tail:\n"
             f"{(p.stderr or '')[-1500:]}")
        return 1

    try:
        st = json.loads(STATE.read_text())
    except Exception:
        st = {}
    sig = signature(out)
    now = datetime.now(timezone.utc)
    last = st.get("last_sent")
    hours = 999.0
    if last:
        try:
            hours = (now - datetime.fromisoformat(last)).total_seconds() / 3600.0
        except Exception:
            pass

    changed = sig != st.get("sig")
    due = hours >= HEARTBEAT_HOURS
    if not (a.always or a.force or changed or due):
        print(f"no change and {hours:.1f}h since last send (<{HEARTBEAT_HOURS}h) — not sending")
        return 0

    why = "state changed" if changed else ("heartbeat" if due else "forced")
    m = re.search(r"session ([A-Z /]+)\s+·", out)
    sess = m.group(1).strip() if m else "?"
    verdict = "FLAT is the default"
    subj = f"[advisor] {a.symbol} · {sess} · {why}"
    body = (f"{out}\n\n--\nsent because: {why}"
            f"\nchannel: {channel(env())}"
            f"\nthis is a decision instrument, not a forecast — {verdict}\n")
    ok = send(subj, body)
    if ok is not False and not a.dry:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(dict(sig=sig, last_sent=now.isoformat(), why=why), indent=2))
    print(f"sent ({why}): {subj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
