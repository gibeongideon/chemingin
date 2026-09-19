"""Build the advisor message. The ONLY place advisor text is formatted.

Both senders import `build()`; neither formats anything itself. That is not tidiness — the
message carries the honesty of the whole product, and two formatters drift.

WHAT THE CARD REFUSES TO DO, AND WHY EACH REFUSAL IS A MEASUREMENT.
  * It never prints a direction call as actionable. §3az: all 8 direction cells have negative
    Brier skill and the best one fails its own best-of-8 max-statistic null (p 0.175 / 0.100).
  * It never says "low risk" or "safe". The BOTTOM reliability bucket is not above chance in
    both halves, so a low reading means "no signal", not "safe".
  * It never says short or sell. §3ab: the short leg lost on 10/10 signals; trimming to flat
    captured ~71% of the oracle prize.
  * It quotes the MEASURED bucket frequency, not the model's probability. §3ba: 0 of 16 cells
    have positive Brier skill, so the ranking informs and the per-bar number does not.
  * It prints the BASE RATE beside every frequency (§3ay's rule), and the resolution rate, so a
    conditional number is never read as an unconditional one.
  * It prints the 71.9% feed-transfer figure, because the model was fitted on one broker's
    quotes and is served on another's, and that is the single largest known tax on it.

TRIGGERS. `broken` / `stale` / `change` / `health` / `heartbeat`. Any successful send advances the
hour key, so a state change absorbs that hour's heartbeat. `seen` advances ONLY on a successful
send (`v5_trend_signal.py:185-193`), so a failed delivery retries on the next run and the
mechanism is self-limiting rather than silently lossy.

THE HEARTBEAT IS THE CONTRACT. Every message states that more than 70 minutes of silence means
the pipeline is down, not that the market is quiet. A month of reports was once lost to a blocked
SMTP port with nobody noticing.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

BANNED = ("short", "sell", "low risk", "safe", "go long", "buy")
QUIET_MINUTES = 70


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
        body = (f"THE ADVISOR PRODUCED NO READING.\n\n{broken}\n\n"
                "No reading means NO INSTRUCTION. This is not a quiet market; it is an absent\n"
                "model, and the last state below is frozen at its own age.\n\n"
                f"  last state : {state.get('name', 'UNKNOWN')} "
                f"(asof {state.get('asof')})\n"
                f"  health     : {state.get('health')}\n\n"
                "  systemctl --user status xau-advisor.timer\n"
                "  tail -40 ~/MT5/data/v5_runs/xau-advisor.log\n")
        return dict(subject="[xau-advisor] NO READING — pipeline broken", body=body,
                    should_send=seen.get("broken_hour") != hk, trigger="broken",
                    seen={**seen, "broken_hour": hk})

    name = state.get("name", "UNKNOWN")
    frozen = bool(state.get("frozen"))
    health = state.get("health", "ok")
    observed = m.get("observed")
    obs_pct = f"{100*observed:.0f}%" if observed is not None else "n/a"
    base = m.get("base_rate", a.get("measured", {}).get("base_rate"))
    prev = seen.get("state")

    # ------------------------------------------------------------------ subject
    if market_closed and health == "ok" and not frozen:
        subject = (f"[xau-advisor] {name} — market closed, no new bars due"
                   + (f" ({obs_pct} vs base {100*base:.0f}%)" if observed is not None else ""))
    elif frozen or health != "ok":
        subject = f"[xau-advisor] {name} HELD — {health} (no current reading)"
    elif name == "ELEVATED":
        subject = (f"[xau-advisor] ELEVATED adverse risk {obs_pct} "
                   f"(base {100*base:.0f}%) — consider trimming longs")
    else:
        subject = f"[xau-advisor] CLEAR — no elevated adverse risk measured"
    if prev and prev != name and not frozen:
        subject += f"  [was {prev}]"

    # ------------------------------------------------------------------ body
    L = []
    L.append("=" * 72)
    L.append("XAUUSD ADVISOR — advisory only. It places no orders and has no order path.")
    L.append("=" * 72)
    L.append("")
    L.append(f"STATE        {name}" + ("  (HELD — see HEALTH)" if frozen else ""))
    if not frozen and observed is not None:
        L.append(f"             bars scoring this high went on to move adversely "
                 f"{obs_pct} of the time")
        L.append(f"             ({m.get('observed_n')} such bars, "
                 f"{_fmt_ci(m.get('observed_ci'))}) against a base rate of {100*base:.1f}%")
    L.append(f"ADVICE       {m.get('advice', 'no reading')}")
    L.append("")

    L.append("-" * 72)
    L.append(f"ADVERSE-MOVE PANEL   ({a.get('cell', '?')}, tier {a.get('tier', '?')})")
    L.append("-" * 72)
    L.append(f"  question   : {m.get('label', a.get('k_atr'))}")
    L.append(f"  ranking    : {m.get('p', float('nan')):.3f}  "
             f"(warn at {a.get('warn_at')}, release below {a.get('clear_at')})")
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
        L.append(f"  quiet rule : while closed this card drops to ONE message a day. The")
        L.append(f"               {QUIET_MINUTES}-minute rule resumes at the reopen.")
    elif frozen:
        L.append("  The state above is FROZEN at the last good reading. It is not current.")
        L.append(f"  quiet rule : more than {QUIET_MINUTES} minutes without a message means this")
        L.append("               pipeline is down, NOT that the market is calm.")
    else:
        L.append(f"  quiet rule : more than {QUIET_MINUTES} minutes without a message means this")
        L.append("               pipeline is down, NOT that the market is calm.")
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
    L.append("  it may warn about downside. It will never tell you to go the other way, and a")
    L.append("  low reading means NO SIGNAL, not an all-clear.")
    L.append(f"  model {meta.get('model_id')} trained {meta.get('trained_on')} — "
             f"{meta.get('findings')}")
    body = "\n".join(L)

    # ------------------------------------------------------------------ trigger
    if market_closed and health == "ok":
        # Once a day, not once an hour. A weekend at hourly cadence is ~48 identical emails,
        # which is exactly how a real alert gets trained into background noise -- the same
        # mistake `book-ftmo.service` was making by failing every weekend pass.
        trigger = "closed"
        day = now.strftime("%Y-%m-%d")
        # A flip INTO the warning still goes out immediately even with the market shut: the
        # reading is real, and the user may well hold a position over the close.
        should = seen.get("closed_day") != day or seen.get("state") != name
        new_seen = {**seen, "closed_day": day, "health": health, "state": name}
    elif frozen or health != "ok":
        trigger = "health"
        should = seen.get("health_hour") != hk or seen.get("health") != health
        new_seen = {**seen, "health_hour": hk, "health": health}
    elif prev != name:
        trigger, should = "change", True
        new_seen = {**seen, "state": name, "hour": hk, "health": health}
    else:
        trigger = "heartbeat"
        should = seen.get("hour") != hk
        new_seen = {**seen, "state": name, "hour": hk, "health": health}
    new_seen["observed"] = observed
    new_seen["p"] = m.get("p")

    # A transition out of UNKNOWN is initialisation, not news.
    if trigger == "change" and prev is None and name == "CLEAR":
        trigger, should = "heartbeat", seen.get("hour") != hk

    assert not any(b in subject.lower() for b in BANNED), f"banned word in subject: {subject}"
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
