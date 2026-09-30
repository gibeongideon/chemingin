"""Build the advisor message. The ONLY place advisor text is formatted.

Both senders import `build()`; neither formats anything itself. That is not tidiness — the
message carries the honesty of the whole product, and two formatters drift.

THE HEADLINE IS A DIRECTION: DOWN, NEUTRAL or UP, with the measured rate for THAT side.
`p` is P(gold touches -1 ATR before +1 ATR within 9h), so its complement is the UP call. That
is a statement about the PATH, and it is not the close-to-close DRIFT question §3az closed at
Tier B — two different labels, one with measured skill and one without. Both appear on the card,
labelled, so the distinction is visible rather than asserted.

WHAT THE CARD REFUSES TO DO, AND WHY EACH REFUSAL IS A MEASUREMENT.
  * The close-to-close DIRECTION panel is printed but labelled NOT USABLE. §3az: all 8 cells have
    negative Brier skill and the best fails its own best-of-8 max-statistic null (p 0.175/0.100).
  * It never proposes a short. §3ab: the short leg lost on 10/10 signals; trimming to flat
    captured ~71% of the oracle prize. An UP call says a long is on the measured side; a DOWN
    call says trim, never reverse.
  * It quotes the MEASURED bucket frequency, not the model's probability. §3ba: 0 of 16 cells
    have positive Brier skill, so the ranking informs and the per-bar number does not.
  * It prints the BASE RATE beside every frequency (§3ay's rule), and the resolution rate, so a
    conditional number is never read as an unconditional one.
  * It prints the 71.9% feed-transfer figure, because the model was fitted on one broker's
    quotes and is served on another's, and that is the single largest known tax on it.

TRIGGERS, AND THE CADENCE THE USER ASKED FOR. A message goes out when the DIRECTION CHANGES —
that is the point of the product, and it is what they asked for ("only when direction change").
It also goes out when the pipeline is broken or unhealthy, because silence must never be
ambiguous, and ONCE A DAY otherwise.

THE DAILY HEARTBEAT IS KEPT ON PURPOSE, against a literal reading of "only on change". A month
of reports was once lost to a blocked SMTP port with nobody noticing, and a change-only channel
cannot tell "nothing changed" from "the sender died three weeks ago". One message a day is the
cheapest thing that keeps silence meaningful, and every card states the rule so the guarantee is
legible rather than implied. `seen` advances ONLY on a successful send
(`v5_trend_signal.py:185-193`), so a failed delivery retries next run.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

BANNED = ("go short", "sell short", "guaranteed", "risk-free")
QUIET_HOURS = 26                 # a daily heartbeat plus slack for a missed slot

# A PERSISTENT FAULT MUST ESCALATE ONCE, THEN BACK OFF. Measured failure, 2026-09-30: the FTMO
# bridge lost its authorisation on 2026-09-26 and this notifier sent **47 identical "NO READING"
# emails**, hourly, for four days. That is not alerting, it is training the reader to filter the
# channel -- the same mistake as the weekend false alarm in `book-ftmo.sh`, one level up. An
# unchanging fault carries no new information after the first message; what DOES carry
# information is how long it has been broken, so the age goes in the subject line and the repeat
# rate decays.
#
#   first 3 hours   hourly   -- you may be mid-deploy and want to see it clear
#   to 24 hours     every 6h -- it is real; you know
#   beyond 24h      daily    -- it needs a human and nagging will not summon one
BROKEN_BACKOFF = ((3, 1), (24, 6), (10**6, 24))


def load_seen(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


def save_seen(path: Path, seen: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(seen, indent=2))
    tmp.replace(p)


def _hour_key(now: datetime) -> str:
    return now.strftime("%Y-%m-%dT%H")


def _fmt_ci(ci) -> str:
    if not ci or ci[0] is None or ci[1] != ci[1]:
        return "CI n/a (bucket too thin to state one)"
    return f"CI [{float(ci[0]):.3f}, {float(ci[1]):.3f}]"


def build(state: dict, reading: dict | None, meta: dict, seen: dict,
          now: datetime | None = None, market: dict | None = None,
          broken: str | None = None, market_closed: bool = False) -> dict:
    """Return {subject, body, should_send, trigger, seen}. Pure — no I/O, no sending."""
    now = now or datetime.now(timezone.utc)
    hk = _hour_key(now)
    a = (meta or {}).get("adverse", {})
    d = (meta or {}).get("direction", {})
    m = (reading or {}).get("adverse", {})

    # ------------------------------------------------------------------ pipeline is broken
    if broken:
        # `broken_since` is the FIRST time this fault was seen, so the age is real rather than
        # "since the last time I happened to email you".
        since = seen.get("broken_since")
        since_dt = None
        if since:
            try:
                since_dt = datetime.fromisoformat(since)
            except ValueError:
                since_dt = None
        if since_dt is None:
            since_dt, since = now, now.isoformat()
        age_h = (now - since_dt).total_seconds() / 3600
        every = next(h for lim, h in BROKEN_BACKOFF if age_h < lim)
        last = seen.get("broken_sent")
        last_dt = None
        if last:
            try:
                last_dt = datetime.fromisoformat(last)
            except ValueError:
                last_dt = None
        due = last_dt is None or (now - last_dt).total_seconds() / 3600 >= every
        age_s = (f"{age_h*60:.0f} min" if age_h < 1 else
                 f"{age_h:.0f}h" if age_h < 48 else f"{age_h/24:.1f} days")

        # Name the remedy for THIS fault rather than printing a generic runbook. A bridge
        # authorisation failure is an infrastructure problem with a specific fix, and it is not
        # the same as a model that will not load.
        bl = str(broken).lower()
        if "authorization" in bl or "init failed" in bl:
            cause = ("The MT5 bridge will not AUTHORISE. The terminal process is usually running "
                     "and the port listening, so this is a credential problem, not a crash: a "
                     "demo account that has expired, a reset password, or a changed server.")
            remedy = ["  1. open the FTMO terminal and check the account is still valid",
                      "  2. if the demo expired, create a new one and update "
                      "~/.mt5c/drive_c/mt5_login.ini",
                      "  3. systemctl --user restart mt5-terminal-ftmo.service   "
                      "(prefix-scoped, safe)",
                      "  NOTE: never restart the bare mt5-terminal.service -- its ExecStop "
                      "kills every terminal."]
        else:
            cause = "The advisor could not produce a reading."
            remedy = ["  systemctl --user status xau-advisor.timer",
                      "  tail -40 ~/MT5/data/v5_runs/xau-advisor.log"]

        body = "\n".join([
            f"NO READING for {age_s} — since {since_dt:%Y-%m-%d %H:%M} UTC.",
            "",
            cause,
            "",
            f"  reported fault : {broken}",
            f"  last good state: {state.get('name', 'UNKNOWN')} (asof {state.get('asof')})",
            "",
            "No reading means NO INSTRUCTION. This is not a quiet market; it is an absent model,",
            "and the state above is frozen at its own age.",
            "",
            "TO FIX:",
            *remedy,
            "",
            f"This alert now repeats every {every}h, not hourly — a fault that has not changed",
            f"carries no new information. It has sent {int(seen.get('broken_count', 0)) + 1} "
            f"message(s) for this outage.",
        ])
        return dict(subject=f"[xau-advisor] NO READING for {age_s} — bridge/model down",
                    body=body, should_send=due, trigger="broken",
                    seen={**seen, "broken_since": since,
                          "broken_sent": now.isoformat() if due else last,
                          "broken_count": int(seen.get("broken_count", 0)) + (1 if due else 0)})

    name = state.get("name", "UNKNOWN")
    frozen = bool(state.get("frozen"))
    health = state.get("health", "ok")
    observed = m.get("observed")
    obs_pct = f"{100*observed:.0f}%" if observed is not None else "n/a"
    base = m.get("base_rate", a.get("measured", {}).get("base_rate"))
    prev = seen.get("state")

    # ------------------------------------------------------------------ subject
    flipped = bool(prev and prev != name and not frozen)
    rate = f" P={obs_pct}" if observed is not None else ""
    if frozen or health != "ok":
        subject = f"[xau-advisor] {name} HELD — {health} (no current reading)"
    elif flipped:
        # The flip is the headline, in the user's own idiom: "now UP (P=60%), was DOWN".
        subject = f"[xau-advisor] {name}{rate} — was {prev}"
        if name == "DOWN":
            subject += " — consider trimming longs"
    elif market_closed:
        subject = f"[xau-advisor] {name}{rate} — market closed, no new bars due"
    else:
        subject = f"[xau-advisor] {name}{rate} (unchanged)"

    # ------------------------------------------------------------------ body
    L = []
    L.append("=" * 72)
    L.append("XAUUSD ADVISOR — advisory only. It places no orders and has no order path.")
    L.append("=" * 72)
    L.append("")
    arrow = {"DOWN": "v  DOWN", "UP": "^  UP", "NEUTRAL": "-  NEUTRAL"}.get(name, f"?  {name}")
    L.append(f"DIRECTION    {arrow}" + ("   (HELD — see HEALTH)" if frozen else ""))
    if flipped:
        L.append(f"             *** CHANGED from {prev} ***")
    if not frozen and observed is not None:
        what = ("fall 1 ATR before rising 1 ATR" if name == "DOWN"
                else "rise 1 ATR before falling 1 ATR" if name == "UP" else "resolve this way")
        L.append(f"             measured: bars reading like this went on to {what}")
        L.append(f"             {obs_pct} of the time  (n {m.get('observed_n')}, "
                 f"base rate {100*base:.1f}%)")
        h = m.get("observed_halves") or [None, None]
        if h[0] is not None and h[0] == h[0]:
            L.append(f"             both halves: {100*float(h[0]):.1f}% in 2018-21, "
                     f"{100*float(h[1]):.1f}% in 2022-26")
    L.append(f"ADVICE       {m.get('advice', 'no reading')}")
    L.append(f"HELD SINCE   {state.get('since')}")
    L.append("")

    L.append("-" * 72)
    L.append(f"HOW THE CALL IS MADE   ({a.get('cell', '?')}, tier {a.get('tier', '?')})")
    L.append("-" * 72)
    L.append(f"  question   : {m.get('label', a.get('k_atr'))}")
    L.append(f"  ranking    : {m.get('p', float('nan')):.3f}   "
             f"DOWN at >={a.get('down_at')} (hold >={a.get('down_release')})   "
             f"UP at <={a.get('up_at')} (hold <={a.get('up_release')})")
    L.append(f"  a reading between {a.get('up_at')} and {a.get('down_at')} is NEUTRAL: no "
             f"measured edge either way")
    L.append(f"  bar        : {reading.get('asof')} closed "
             f"{reading.get('decision_time')}" if reading else "  bar        : none")
    L.append(f"  resolution : {100*float(a.get('resolved_frac', float('nan'))):.0f}% of "
             f"{a.get('hours')}h windows resolve +/-{a.get('k_atr')} ATR; the rest end inside "
             f"the band and are NOT forecast")
    L.append("")
    L.append("  reliability table this reading is quoted from (the whole basis, auditable):")
    L.append(f"    {'bucket':>12}  {'n':>6}  {'observed':>9}  {'2018-21':>8}  {'2022-26':>8}")
    for row in a.get("measured", {}).get("reliability", []):
        h1 = row.get("obs_first_half")
        h2 = row.get("obs_second_half")
        f1 = f"{float(h1):.3f}" if h1 is not None and h1 == h1 else "   —"
        f2 = f"{float(h2):.3f}" if h2 is not None and h2 == h2 else "   —"
        L.append(f"    {float(row['lo']):.2f}-{float(row['hi']):.2f}  {int(row['n']):>6}  "
                 f"{float(row['observed']):>9.3f}  {f1:>8}  {f2:>8}")
    L.append("")

    L.append("-" * 72)
    L.append(f"DIRECTION PANEL      ({d.get('horizon_hours', '?')}h)  —  NOT USABLE")
    L.append("-" * 72)
    pu = (reading or {}).get("direction", {}).get("p_up")
    L.append(f"  model says : {pu:.3f}" if pu is not None else "  model says : unavailable")
    L.append(f"  base rate  : {100*float(d.get('measured', {}).get('base_rate', float('nan'))):.1f}%"
             "   <-- this is the number to use")
    for line in (d.get("edge_note", ""), d.get("drift_note", "")):
        for chunk in _wrap(line, 68):
            L.append(f"  {chunk}")
    L.append("")

    if market:
        L.append("-" * 72)
        L.append("MARKET")
        L.append("-" * 72)
        for k, v in market.items():
            L.append(f"  {k:<11}: {v}")
        L.append("")

    L.append("-" * 72)
    L.append("HEALTH")
    L.append("-" * 72)
    L.append(f"  status     : {health}")
    if market_closed:
        L.append("  Gold is not quoting. No H4 bar can close and no reading can change until")
        L.append("  the market reopens, so the state below is held for an EXPECTED reason.")
        L.append("  quiet rule : one message a day while closed; a direction change still")
        L.append("               sends immediately.")
    elif frozen:
        L.append("  The state above is FROZEN at the last good reading. It is not current.")
        L.append(f"  quiet rule : more than {QUIET_HOURS}h without any message means this")
        L.append("               pipeline is down, NOT that the direction is steady.")
    else:
        L.append("  You are emailed when the DIRECTION CHANGES, plus once a day either way.")
        L.append(f"  quiet rule : more than {QUIET_HOURS}h without any message means this")
        L.append("               pipeline is down, NOT that the direction is steady.")
    L.append(f"  held since : {state.get('since')}")
    L.append("")

    L.append("-" * 72)
    L.append("LIMITS — measured, not boilerplate")
    L.append("-" * 72)
    for lim in a.get("limits", []):
        for chunk in _wrap(lim, 68):
            L.append(f"  {chunk}")
    L.append("  this model was fitted on one broker's quotes and is served on another's; the")
    L.append("  warning RATE transfers (11.2% vs 11.1%) but only 71.9% of individual warnings")
    L.append("  survive the change (V5_FINDINGS 3ba)")
    L.append("  a DOWN call is a reason to trim, never a reason to reverse: the short leg lost")
    L.append("  on 10 of 10 signals (§3ab) while trimming to flat captured ~71% of the prize.")
    L.append(f"  model {meta.get('model_id')} trained {meta.get('trained_on')} — "
             f"{meta.get('findings')}")
    body = "\n".join(L)

    # ------------------------------------------------------------------ trigger
    day = now.strftime("%Y-%m-%d")
    # A real reading ENDS the outage. Without this the age would be measured from the first
    # fault ever seen, so a later unrelated outage would report itself as days old on its
    # first message -- a false alarm that reads exactly like a real one.
    seen = {k: v for k, v in seen.items()
            if k not in ("broken_since", "broken_sent", "broken_count")}
    if frozen or health != "ok":
        trigger = "health"
        should = seen.get("health_day") != day or seen.get("health") != health
        new_seen = {**seen, "health_day": day, "health": health}
    elif flipped:
        # The whole point of the product. Sent immediately, including with the market shut: the
        # reading is real and a position may well be held over the close.
        trigger, should = "change", True
        new_seen = {**seen, "state": name, "day": day, "health": health}
    else:
        trigger = "daily"
        should = seen.get("day") != day
        new_seen = {**seen, "state": name, "day": day, "health": health}
    new_seen["observed"] = observed
    new_seen["p"] = m.get("p")

    # A first reading resolving UNKNOWN is initialisation, not a direction change.
    if trigger == "change" and prev is None:
        trigger, should = "daily", seen.get("day") != day

    assert not any(b in subject.lower() for b in BANNED), f"banned phrase in subject: {subject}"
    return dict(subject=subject, body=body, should_send=bool(should), trigger=trigger,
                seen=new_seen)


def _wrap(text: str, width: int) -> list:
    out, line = [], ""
    for w in str(text).split():
        if len(line) + len(w) + 1 > width:
            out.append(line); line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out or [""]
