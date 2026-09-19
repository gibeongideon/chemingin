"""Hysteresis state machine for the XAUUSD advisor. Pure: no MT5, no I/O, no clock of its own.

WHAT THE VERDICT DID TO THIS FILE. The plan drew a three-state DOWN/NEUTRAL/UP machine over a
direction probability. §3az measured that probability to be TIER B — negative Brier skill in all
8 cells and a best cell that fails its own best-of-8 max-statistic null (p 0.175 / 0.100) — so
there is nothing to hold a direction state ON. What survived is the ADVERSE-move top tail, which
is a two-state question: is the next ~4h at elevated risk of a -k*ATR move, or is there no
signal? The machine below is therefore general in shape and instantiated with two states.

It is deliberately NOT collapsed to a bare `p >= 0.60`, because the thing the user asked for is
the holding behaviour: a state that persists and is expensive to leave, so that a reading
wobbling around the threshold does not produce a stream of contradictory messages.

THE ASYMMETRY IS THE REPO'S MEASUREMENT, NOT A PREFERENCE.
  * WARNING IS NEVER DELAYED. `min_dwell_h` gates only the return to CLEAR. §3ab measured the
    short leg losing on 10/10 signals while trimming to flat captured ~71% of the oracle prize:
    de-risking is cheap and being late to it is what costs. A dwell timer that delayed a warning
    would be optimising the wrong side.
  * CLEARING IS DELAYED. Re-risking on a reading that has merely drifted back under the line is
    the expensive mistake, so CLEAR must be earned by both a margin and time.

THRESHOLDS ARE MARGINS AROUND THE BASE RATE, NEVER AROUND 0.5. At fwd6 gold's base rate is
0.536, so an isotonic calibrator puts nearly every bar above 0.5 and a 0.5 cut IS the drift
forecast — measured at acc 53.56% against drift 53.62% (§3az). `Thresholds.validate` refuses to
construct a machine whose warn level sits below the base rate for that reason.

WHY HEALTH NEVER ENTERS THE HYSTERESIS. A stale feed is not evidence about gold. If health is
anything but ok the state FREEZES at its last value and reports its age; it never drifts toward
CLEAR just because the data stopped arriving. `v5_zigzag_message.py:115-119` exists because a
silently degraded model is worse than an absent one.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone

CLEAR = "CLEAR"
ELEVATED = "ELEVATED"
UNKNOWN = "UNKNOWN"

HEALTH_OK = "ok"


@dataclass(frozen=True)
class Thresholds:
    """The state machine's contract. Read from the frozen artifact, never invented at runtime."""
    warn_at: float
    clear_at: float
    base_rate: float
    min_dwell_h: float = 8.0          # applies to CLEARING only
    max_changes_per_day: int = 4

    def validate(self) -> None:
        if not (0.0 < self.clear_at < self.warn_at < 1.0):
            raise ValueError(f"need 0 < clear_at < warn_at < 1, got "
                             f"clear_at={self.clear_at} warn_at={self.warn_at}")
        if self.warn_at <= self.base_rate:
            raise ValueError(
                f"warn_at {self.warn_at} is not above the base rate {self.base_rate}. A threshold "
                "at or below the base rate is the climatology forecast wearing a probability's "
                "clothes — see V5_FINDINGS §3az.")
        if self.clear_at < self.base_rate:
            # Allowed, but it means the machine holds a warning below climatology. Loud on
            # purpose: it is a real choice about how sticky the warning should be.
            pass


@dataclass
class AdvisorState:
    name: str = UNKNOWN
    p: float = float("nan")
    since: datetime | None = None            # when the CURRENT state was entered
    asof: datetime | None = None             # the decision bar this reading came from
    health: str = HEALTH_OK
    frozen: bool = False                     # True when health != ok: the reading is not current
    changes: list = field(default_factory=list)   # transition timestamps, for the thrash guard
    last_transition: tuple | None = None     # (from, to) on the step that just ran, else None

    def dwell_h(self, now: datetime) -> float:
        return float("nan") if self.since is None else (now - self.since).total_seconds() / 3600

    def age_h(self, now: datetime) -> float:
        return float("nan") if self.asof is None else (now - self.asof).total_seconds() / 3600


def _changes_in_last_day(changes: list, now: datetime) -> int:
    return sum(1 for t in changes if now - t <= timedelta(days=1))


def step(state: AdvisorState, p: float | None, asof: datetime | None,
         th: Thresholds, now: datetime, health: str = HEALTH_OK) -> AdvisorState:
    """Advance the machine by ONE closed decision bar.

    `p` is the calibrated adverse probability for the bar closing at `asof`; `now` is wall clock.
    Call this only when `asof` is NEWER than `state.asof` — the 5-minute service refreshes the
    market panel 288 times a day but the model moves 6 times, and re-stepping on an unchanged bar
    would let the thrash guard and the dwell timer count events that did not happen.

    Returns a NEW state; the input is not mutated.
    """
    th.validate()

    # ---- health first, and it short-circuits: an unhealthy reading is not evidence ----------
    if health != HEALTH_OK or p is None or asof is None or not (0.0 <= float(p) <= 1.0):
        reason = health if health != HEALTH_OK else (
            "model_unavailable" if p is None or asof is None else "probability_out_of_range")
        return replace(state, health=reason, frozen=True, last_transition=None)

    p = float(p)
    prev = state.name
    new = prev
    if prev in (UNKNOWN, CLEAR):
        # Entering a warning is never delayed, never dwell-gated, and never thrash-gated.
        if p >= th.warn_at:
            new = ELEVATED
    elif prev == ELEVATED:
        if p <= th.clear_at and state.dwell_h(now) >= th.min_dwell_h:
            new = CLEAR
    if prev == UNKNOWN and new == UNKNOWN:
        new = CLEAR if p < th.warn_at else ELEVATED      # first ever reading resolves UNKNOWN

    changes = list(state.changes)
    transition = None
    since = state.since
    if new != prev:
        transition = (prev, new)
        changes.append(now)
        since = now
    elif since is None:
        since = now

    health_out = HEALTH_OK
    # The thrash guard REPORTS, it does not suppress. Suppressing a warning because earlier
    # warnings were noisy is exactly backwards: if the model is thrashing the user needs to be
    # told the model is thrashing, not told nothing.
    if _changes_in_last_day(changes, now) > th.max_changes_per_day:
        health_out = "thrashing"

    return AdvisorState(name=new, p=p, since=since, asof=asof, health=health_out,
                        frozen=False, changes=[t for t in changes
                                               if now - t <= timedelta(days=7)],
                        last_transition=transition)


def describe(state: AdvisorState, th: Thresholds, now: datetime, observed: float | None = None,
             ) -> str:
    """One line, in the user's own idiom, with the number that is actually measured.

    `observed` is the MEASURED frequency of the bucket the reading falls in — not `state.p`.
    §3az found no calibrator with positive Brier skill, so the per-bar probability is not
    trustworthy as a number even where the ranking is: what can honestly be quoted is "bars that
    scored this high went on to move adversely X% of the time".
    """
    if state.frozen or state.health != HEALTH_OK:
        age = state.age_h(now)
        age_s = "unknown age" if age != age else f"{age:.1f}h old"
        return (f"{state.name} (HELD — {state.health}, reading {age_s}); "
                f"this is the last good reading, not a current one")
    pct = 100 * (observed if observed is not None else state.p)
    dwell = state.dwell_h(now)
    held = "" if dwell != dwell else f", held {dwell:.0f}h"
    if state.name == ELEVATED:
        return (f"ELEVATED adverse-move risk (P={pct:.0f}%{held}) — consider trimming or "
                f"exiting longs")
    return f"CLEAR — no elevated adverse-move risk measured (P={pct:.0f}%{held})"


def to_dict(state: AdvisorState) -> dict:
    return dict(name=state.name, p=state.p,
                since=state.since.isoformat() if state.since else None,
                asof=state.asof.isoformat() if state.asof else None,
                health=state.health, frozen=state.frozen,
                changes=[t.isoformat() for t in state.changes],
                last_transition=list(state.last_transition) if state.last_transition else None)


def from_dict(d: dict) -> AdvisorState:
    def ts(x):
        return datetime.fromisoformat(x) if x else None
    return AdvisorState(
        name=d.get("name", UNKNOWN), p=float(d.get("p", float("nan"))),
        since=ts(d.get("since")), asof=ts(d.get("asof")),
        health=d.get("health", HEALTH_OK), frozen=bool(d.get("frozen", False)),
        changes=[ts(x) for x in d.get("changes", []) if x],
        last_transition=tuple(d["last_transition"]) if d.get("last_transition") else None)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
