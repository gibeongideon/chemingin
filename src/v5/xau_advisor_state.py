"""Three-state hysteresis machine for the XAUUSD advisor: DOWN / NEUTRAL / UP. Pure, no I/O.

WHY THIS IS A DIRECTION CALL AFTER ALL, AND WHY IT DOES NOT CONTRADICT §3az.
Two different questions get confused as one:

  * "will the close in 8h be higher than now?" — the DIRECTION panel, a question about DRIFT.
    §3az measured it at TIER B: negative Brier skill in all 8 cells and a best cell that fails
    its own best-of-8 max-statistic null (p 0.175 / 0.100). There is no edge and it ships
    `usable: false`.
  * "which does gold touch first, -1.0xATR or +1.0xATR, within 9h?" — the ADVERSE panel, a
    question about PATH. That one has measured skill (§3ba: the family clears its best-of-16 null
    at p 0.010 on a clean single feed).

The second question is directional: `p` is P(down-barrier first), so `1-p` is P(up-barrier
first). Naming the states DOWN and UP is therefore not a softening of §3az — it is the honest
reading of a different label. What it is NOT is a trend call: the horizon is nine hours and the
claim is about which side gets touched first, not about where price ends up.

BOTH TAILS ARE VALIDATED, WHICH IS WHY TWO SIDES ARE ALLOWED. `1.0 ATR / 9h / BASE` is the only
one of 16 clean-feed cells whose TOP and BOTTOM buckets both sit past the base rate in BOTH
halves of the sample:

    p <= 0.40  n 363   P(up-first)   0.603   halves 0.677 / 0.562   vs base 0.510
    p >= 0.60  n 404   P(down-first) 0.562   halves 0.596 / 0.536   vs base 0.490

THE THRESHOLDS ARE THE BUCKET EDGES, AND THAT IS A CORRECTION. An earlier version warned at
p >= 0.55, taken from the operating-point curve. That was wrong: the [0.55, 0.60) bucket has NO
first-half observations, so a 0.55 threshold leans part of its headline on a band that cannot be
checked across halves. Anchoring each state to a bucket whose frequency was measured in both
halves means the number displayed to the user IS that bucket's observed rate — not an
interpolation, and not a model probability (§3ba: 0 of 16 cells have positive Brier skill).

THE ASYMMETRY IS MEASURED, NOT PREFERRED. §3ab: the short leg lost on 10/10 signals while
trimming to flat captured ~71% of the oracle prize. So DOWN is entered immediately and never
dwell-gated, while UP — the re-risking direction — must serve `min_dwell_h` when coming from
DOWN. Being late to de-risk is the expensive error; being early to re-risk is the other one.

HEALTH NEVER ENTERS THE HYSTERESIS. A stale feed is not evidence about gold, so an unhealthy
reading FREEZES the state at its last value with its age visible, rather than letting it drift
toward NEUTRAL because data stopped arriving.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone

DOWN = "DOWN"
NEUTRAL = "NEUTRAL"
UP = "UP"
UNKNOWN = "UNKNOWN"
HEALTH_OK = "ok"

RANK = {DOWN: 0, NEUTRAL: 1, UP: 2, UNKNOWN: 1}


@dataclass(frozen=True)
class Thresholds:
    """Read from the frozen artifact, never invented at runtime."""
    down_at: float           # p >= this  -> DOWN   (the top reliability bucket's lower edge)
    down_release: float      # hold DOWN while p >= this
    up_at: float             # p <= this  -> UP     (the bottom bucket's upper edge)
    up_release: float        # hold UP while p <= this
    base_rate: float
    min_dwell_h: float = 9.0      # applies to RE-RISKING (moving toward UP) only
    max_changes_per_day: int = 4

    def validate(self) -> None:
        if not (0.0 < self.up_at < self.up_release < self.down_release < self.down_at < 1.0):
            raise ValueError(
                "need 0 < up_at < up_release < down_release < down_at < 1, got "
                f"up_at={self.up_at} up_release={self.up_release} "
                f"down_release={self.down_release} down_at={self.down_at}")
        if self.down_at <= self.base_rate:
            raise ValueError(
                f"down_at {self.down_at} is not above the base rate {self.base_rate}. A threshold "
                "at or below the base rate is climatology wearing a probability's clothes — a 0.5 "
                "cut on a calibrated probability IS the drift forecast (§3az: acc 53.56% vs "
                "drift 53.62%).")
        if self.up_at >= self.base_rate:
            raise ValueError(
                f"up_at {self.up_at} is not below the base rate {self.base_rate}; same reason.")


@dataclass
class AdvisorState:
    name: str = UNKNOWN
    p: float = float("nan")
    since: datetime | None = None
    asof: datetime | None = None
    health: str = HEALTH_OK
    frozen: bool = False
    changes: list = field(default_factory=list)
    last_transition: tuple | None = None

    def dwell_h(self, now: datetime) -> float:
        return float("nan") if self.since is None else (now - self.since).total_seconds() / 3600

    def age_h(self, now: datetime) -> float:
        return float("nan") if self.asof is None else (now - self.asof).total_seconds() / 3600


def _changes_in_last_day(changes: list, now: datetime) -> int:
    return sum(1 for t in changes if now - t <= timedelta(days=1))


def _target(p: float, prev: str, th: Thresholds) -> str:
    """Where the reading alone would put the state, before dwell and thrash rules."""
    if prev == DOWN:
        return DOWN if p >= th.down_release else (UP if p <= th.up_at else NEUTRAL)
    if prev == UP:
        return UP if p <= th.up_release else (DOWN if p >= th.down_at else NEUTRAL)
    # from NEUTRAL or UNKNOWN
    if p >= th.down_at:
        return DOWN
    if p <= th.up_at:
        return UP
    return NEUTRAL


def step(state: AdvisorState, p: float | None, asof: datetime | None,
         th: Thresholds, now: datetime, health: str = HEALTH_OK) -> AdvisorState:
    """Advance the machine by ONE closed decision bar. Returns a NEW state.

    Call only when `asof` is NEWER than `state.asof`. The 5-minute service refreshes the market
    panel 288 times a day while the model moves 6 times, and re-stepping an unchanged bar would
    let the dwell timer and the thrash guard count events that did not happen.
    """
    th.validate()

    if health != HEALTH_OK or p is None or asof is None or not (0.0 <= float(p) <= 1.0):
        reason = health if health != HEALTH_OK else (
            "model_unavailable" if p is None or asof is None else "probability_out_of_range")
        return replace(state, health=reason, frozen=True, last_transition=None)

    p = float(p)
    prev = state.name
    new = _target(p, prev, th)

    # Dwell gates RE-RISKING only: moving up the DOWN -> NEUTRAL -> UP ladder. De-risking is
    # never delayed (§3ab). UNKNOWN is rank-neutral so a first reading is never dwell-blocked.
    if prev != UNKNOWN and RANK[new] > RANK[prev]:
        if state.dwell_h(now) < th.min_dwell_h:
            new = prev

    changes = list(state.changes)
    transition, since = None, state.since
    if new != prev:
        transition = (prev, new)
        changes.append(now)
        since = now
    elif since is None:
        since = now

    health_out = HEALTH_OK
    # The thrash guard REPORTS; it does not suppress. Hiding a warning because earlier warnings
    # were noisy is backwards — if the model is thrashing, that is the thing to say.
    if _changes_in_last_day(changes, now) > th.max_changes_per_day:
        health_out = "thrashing"

    return AdvisorState(name=new, p=p, since=since, asof=asof, health=health_out, frozen=False,
                        changes=[t for t in changes if now - t <= timedelta(days=7)],
                        last_transition=transition)


def observed_for(state_name: str, p: float, reliability: list, base_rate: float) -> dict:
    """The MEASURED frequency to display, oriented to the state's own direction.

    §3ba found no calibrator with positive Brier skill, so the per-bar probability is not a
    quotable number. What can be quoted is the frequency of the bucket the reading falls in —
    a fact about the sample. For UP that frequency is 1 - observed, because the label counts
    DOWN-first as 1.
    """
    row = None
    for r in reliability:
        lo, hi = float(r["lo"]), float(r["hi"])
        if (lo <= p < hi) or (hi >= 1.0 and p >= lo):
            row = r
            break
    if row is None:
        return dict(pct=None, n=None, base=None, halves=(None, None), bucket=None)
    flip = state_name == UP
    conv = (lambda x: (1.0 - float(x)) if x is not None and x == x else None)
    obs = conv(row["observed"]) if flip else (float(row["observed"]))
    h1 = conv(row.get("obs_first_half")) if flip else row.get("obs_first_half")
    h2 = conv(row.get("obs_second_half")) if flip else row.get("obs_second_half")
    base = (1.0 - base_rate) if flip else base_rate
    return dict(pct=obs, n=int(row["n"]), base=base,
                halves=(h1, h2), bucket=(float(row["lo"]), float(row["hi"])))


def describe(state: AdvisorState, th: Thresholds, now: datetime,
             observed: float | None = None) -> str:
    """One line, in the user's own idiom: the direction, its measured rate, and how long held."""
    if state.frozen or state.health != HEALTH_OK:
        age = state.age_h(now)
        age_s = "unknown age" if age != age else f"{age:.1f}h old"
        return (f"{state.name} (HELD — {state.health}, reading {age_s}); "
                f"this is the last good reading, not a current one")
    dwell = state.dwell_h(now)
    held = "" if dwell != dwell else f", held {dwell:.0f}h"
    pct = observed if observed is not None else state.p
    p_s = f" (P={100*pct:.0f}%{held})" if pct is not None and pct == pct else f"({held})"
    if state.name == DOWN:
        return f"DOWN{p_s} — gold is more likely to fall 1 ATR before rising 1 ATR in ~9h"
    if state.name == UP:
        return f"UP{p_s} — gold is more likely to rise 1 ATR before falling 1 ATR in ~9h"
    if state.name == UNKNOWN:
        return "UNKNOWN — no reading has been taken yet"
    return f"NEUTRAL{p_s} — no directional edge measured at this reading"


def to_dict(state: AdvisorState) -> dict:
    return dict(name=state.name, p=state.p,
                since=state.since.isoformat() if state.since else None,
                asof=state.asof.isoformat() if state.asof else None,
                health=state.health, frozen=state.frozen,
                changes=[t.isoformat() for t in state.changes],
                last_transition=list(state.last_transition) if state.last_transition else None)


def from_dict(d: dict) -> AdvisorState:
    """Rebuild from the persisted state.

    EVERY FIELD MUST TOLERATE null. A run that could not produce a reading persists `p: null`,
    and `float(None)` raises TypeError — which meant that once an outage wrote a broken state,
    the advisor could never start again EVEN AFTER the fault was fixed. The failure state
    poisoned its own recovery path, and it stayed wedged for the whole 2026-09-26 to 10-02
    bridge outage. A deserialiser for crash state has to read what a crash actually writes.
    """
    def ts(x):
        try:
            return datetime.fromisoformat(x) if x else None
        except (TypeError, ValueError):
            return None

    def num(x):
        try:
            return float(x) if x is not None else float("nan")
        except (TypeError, ValueError):
            return float("nan")

    lt = d.get("last_transition")
    return AdvisorState(
        name=d.get("name") or UNKNOWN, p=num(d.get("p")),
        since=ts(d.get("since")), asof=ts(d.get("asof")),
        health=d.get("health") or HEALTH_OK, frozen=bool(d.get("frozen", False)),
        changes=[t for t in (ts(x) for x in (d.get("changes") or [])) if t is not None],
        last_transition=tuple(lt) if lt else None)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
