"""State-machine tests. These run before the advisor is wired to MT5, by design.

The flip-flop case is the one that matters: a probability wobbling either side of a bare
threshold produces a stream of contradictory messages, which is precisely the failure the user
described wanting to avoid ("it keeps saying DOWN ... then it may say UPTREND"). It is asserted
numerically rather than argued for.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v5.xau_advisor_state import (  # noqa: E402
    CLEAR, ELEVATED, UNKNOWN, AdvisorState, Thresholds, describe, from_dict, step, to_dict,
)

T0 = datetime(2026, 9, 14, 0, 0, tzinfo=timezone.utc)
TH = Thresholds(warn_at=0.60, clear_at=0.55, base_rate=0.494, min_dwell_h=8.0)


def run(seq, th=TH, start=None, bar_h=4):
    """Feed a sequence of probabilities, one per closed H4 bar."""
    s = start or AdvisorState()
    out = []
    for i, p in enumerate(seq):
        t = T0 + timedelta(hours=bar_h * (i + 1))
        s = step(s, p, asof=t, th=th, now=t)
        out.append(s)
    return s, out


# --------------------------------------------------------------------------- thresholds
def test_threshold_below_base_rate_is_refused():
    """A 0.5 cut on a calibrated probability IS the drift forecast (§3az). The machine must
    refuse to be built that way rather than silently ship climatology as a signal."""
    with pytest.raises(ValueError, match="base rate"):
        Thresholds(warn_at=0.49, clear_at=0.45, base_rate=0.494).validate()
    with pytest.raises(ValueError, match="clear_at"):
        Thresholds(warn_at=0.55, clear_at=0.60, base_rate=0.494).validate()
    Thresholds(warn_at=0.60, clear_at=0.55, base_rate=0.494).validate()   # the shipped one


# --------------------------------------------------------------------------- the flip-flop
def test_dead_band_kills_the_flip_flop():
    """The plan's worked example, in adverse-probability space. A path that crosses a bare
    threshold five times must produce ZERO transitions here."""
    path = [0.59, 0.61, 0.58, 0.62, 0.57]
    naive = sum(1 for a, b in zip(path, path[1:]) if (a >= 0.60) != (b >= 0.60))
    assert naive == 4, "the path must actually cross a bare threshold, or the test proves nothing"

    s, hist = run(path)
    transitions = [h.last_transition for h in hist if h.last_transition]
    # The first reading resolving UNKNOWN is initialisation, not a flip-flop; what must be
    # absent is any CLEAR<->ELEVATED oscillation after it. It enters ELEVATED once (at 0.61)
    # and never leaves, because nothing in the path reaches clear_at 0.55.
    assert transitions == [(UNKNOWN, CLEAR), (CLEAR, ELEVATED)], transitions
    settled = [t for t in transitions if t[0] != UNKNOWN]
    assert len(settled) == 1, f"{naive} bare-threshold crossings must collapse to 1, got {settled}"
    assert s.name == ELEVATED


def test_clearing_needs_both_margin_and_time():
    s, _ = run([0.65])
    assert s.name == ELEVATED
    t1 = T0 + timedelta(hours=8)

    # margin met, dwell NOT met -> holds
    s2 = step(s, 0.50, asof=t1, th=TH, now=T0 + timedelta(hours=6))
    assert s2.name == ELEVATED, "cleared before serving min_dwell_h"

    # dwell met, margin NOT met -> holds
    s3 = step(s, 0.58, asof=t1, th=TH, now=T0 + timedelta(hours=40))
    assert s3.name == ELEVATED, "cleared on a reading inside the dead band"

    # both met -> clears
    s4 = step(s, 0.50, asof=t1, th=TH, now=T0 + timedelta(hours=40))
    assert s4.name == CLEAR
    assert s4.last_transition == (ELEVATED, CLEAR)


def test_warning_is_never_delayed():
    """De-risking is never dwell-gated. §3ab: the short leg lost on 10/10 signals while trimming
    to flat captured ~71% of the oracle prize, so being late to a warning is the costly error."""
    s, _ = run([0.30])
    assert s.name == CLEAR
    # one minute later, still far inside any dwell window, a warning must fire immediately
    s2 = step(s, 0.95, asof=T0 + timedelta(hours=8), th=TH,
              now=T0 + timedelta(hours=4, minutes=1))
    assert s2.name == ELEVATED
    assert s2.last_transition == (CLEAR, ELEVATED)


# --------------------------------------------------------------------------- health
@pytest.mark.parametrize("health,p,asof", [
    ("stale_feed", 0.95, T0), ("model_unavailable", 0.95, T0),
    ("ok", None, T0), ("ok", 0.95, None), ("ok", 1.4, T0), ("ok", -0.2, T0),
])
def test_unhealthy_readings_freeze_rather_than_move(health, p, asof):
    s, _ = run([0.65])
    assert s.name == ELEVATED
    s2 = step(s, p, asof=asof, th=TH, now=T0 + timedelta(hours=40), health=health)
    assert s2.frozen is True
    assert s2.name == ELEVATED, "an unhealthy reading must not move the state"
    assert s2.health != "ok"
    assert s2.last_transition is None
    assert "HELD" in describe(s2, TH, T0 + timedelta(hours=40))


def test_freeze_does_not_drift_toward_clear_while_data_is_missing():
    """The dangerous direction: a warning quietly lapsing because the feed died."""
    s, _ = run([0.75])
    now = T0 + timedelta(hours=4)
    for _ in range(50):
        now += timedelta(hours=4)
        s = step(s, None, asof=None, th=TH, now=now, health="stale_feed")
    assert s.name == ELEVATED and s.frozen
    assert s.age_h(now) > 190, "the message must be able to show how old the held reading is"


# --------------------------------------------------------------------------- thrash guard
def test_thrash_guard_reports_and_does_not_suppress():
    th = Thresholds(warn_at=0.60, clear_at=0.55, base_rate=0.494,
                    min_dwell_h=0.0, max_changes_per_day=2)
    s, hist = run([0.65, 0.50, 0.65, 0.50, 0.65, 0.50], th=th)
    assert sum(1 for h in hist if h.last_transition) >= 5
    assert s.health == "thrashing"
    assert s.name in (CLEAR, ELEVATED), "the guard must not blank the state"
    assert s.frozen is False, "thrashing is a health note on a real reading, not a freeze"


# --------------------------------------------------------------------------- wording
def test_wording_never_offers_a_short_or_a_safe_reading():
    """§3ab (short leg lost 10/10) and §3az (the bottom tail is not trustworthy) are both
    wording constraints, so they are asserted on the string the user actually sees."""
    now = T0 + timedelta(hours=12)
    for seq in ([0.95], [0.10], [0.65, 0.20]):
        s, _ = run(seq)
        for text in (describe(s, TH, now), describe(s, TH, now, observed=0.646)):
            low = text.lower()
            for banned in ("short", "sell", "low risk", "safe", "go long"):
                assert banned not in low, f"{banned!r} in {text!r}"


def test_describe_quotes_the_measured_frequency_not_the_model_probability():
    s, _ = run([0.83])
    text = describe(s, TH, T0 + timedelta(hours=8), observed=0.646)
    assert "65%" in text, text
    assert "83%" not in text, "the per-bar probability has no positive Brier skill; don't quote it"


# --------------------------------------------------------------------------- persistence
def test_round_trip_through_json_preserves_behaviour():
    s, _ = run([0.65, 0.62])
    s2 = from_dict(to_dict(s))
    assert (s2.name, s2.p, s2.since, s2.asof, s2.health) == (s.name, s.p, s.since, s.asof,
                                                             s.health)
    now = T0 + timedelta(hours=40)
    assert step(s, 0.50, T0 + timedelta(hours=12), TH, now).name == \
           step(s2, 0.50, T0 + timedelta(hours=12), TH, now).name


def test_first_reading_resolves_unknown_without_a_spurious_transition():
    s = AdvisorState()
    assert s.name == UNKNOWN
    s = step(s, 0.30, asof=T0, th=TH, now=T0)
    assert s.name == CLEAR
    assert s.last_transition == (UNKNOWN, CLEAR)
