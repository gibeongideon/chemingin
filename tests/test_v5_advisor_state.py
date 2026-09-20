"""State-machine tests for the three-state DOWN / NEUTRAL / UP advisor.

The flip-flop case is the one that matters: a probability wobbling either side of a bare
threshold produces a stream of contradictory messages, which is exactly the failure the user
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
    DOWN, NEUTRAL, UNKNOWN, UP, AdvisorState, Thresholds, describe, from_dict,
    observed_for, step, to_dict,
)

T0 = datetime(2026, 9, 14, 0, 0, tzinfo=timezone.utc)
# the shipped contract: bucket edges 0.40 / 0.60, 5pp release bands, base 0.490
TH = Thresholds(down_at=0.60, down_release=0.55, up_at=0.40, up_release=0.45,
                base_rate=0.490, min_dwell_h=9.0)

RELIABILITY = [
    {"lo": 0.00, "hi": 0.40, "n": 363, "observed": 0.397, "obs_first_half": 0.323,
     "obs_second_half": 0.438},
    {"lo": 0.40, "hi": 0.45, "n": 1415, "observed": 0.467, "obs_first_half": 0.473,
     "obs_second_half": 0.461},
    {"lo": 0.45, "hi": 0.50, "n": 3772, "observed": 0.485, "obs_first_half": 0.508,
     "obs_second_half": 0.461},
    {"lo": 0.50, "hi": 0.55, "n": 1493, "observed": 0.506, "obs_first_half": 0.475,
     "obs_second_half": 0.530},
    {"lo": 0.55, "hi": 0.60, "n": 322, "observed": 0.578, "obs_first_half": float("nan"),
     "obs_second_half": 0.578},
    {"lo": 0.60, "hi": 1.00, "n": 404, "observed": 0.562, "obs_first_half": 0.596,
     "obs_second_half": 0.536},
]


def run(seq, th=TH, start=None, bar_h=4, dwell_ok=True):
    """Feed one probability per closed bar. `dwell_ok` spaces bars far enough apart that the
    re-risk dwell is always served, so tests that are not about dwell are not gated by it."""
    s = start or AdvisorState()
    gap = max(bar_h, th.min_dwell_h + 1) if dwell_ok else bar_h
    out = []
    for i, p in enumerate(seq):
        t = T0 + timedelta(hours=gap * (i + 1))
        s = step(s, p, asof=t, th=th, now=t)
        out.append(s)
    return s, out


# --------------------------------------------------------------------------- the contract
def test_thresholds_must_straddle_the_base_rate():
    Thresholds(down_at=0.60, down_release=0.55, up_at=0.40, up_release=0.45,
               base_rate=0.490).validate()
    with pytest.raises(ValueError, match="base rate"):
        Thresholds(down_at=0.48, down_release=0.46, up_at=0.40, up_release=0.44,
                   base_rate=0.490).validate()
    with pytest.raises(ValueError, match="same reason"):
        Thresholds(down_at=0.60, down_release=0.55, up_at=0.50, up_release=0.52,
                   base_rate=0.490).validate()
    with pytest.raises(ValueError, match="need 0 <"):
        Thresholds(down_at=0.40, down_release=0.55, up_at=0.60, up_release=0.45,
                   base_rate=0.490).validate()


# --------------------------------------------------------------------------- direction calls
def test_it_actually_says_up_and_down():
    """The user's ask: the message must carry a direction, not only a risk flag."""
    s, _ = run([0.72])
    assert s.name == DOWN
    assert "DOWN" in describe(s, TH, T0 + timedelta(hours=10))
    s, _ = run([0.20])
    assert s.name == UP
    assert "UP" in describe(s, TH, T0 + timedelta(hours=10))
    s, _ = run([0.50])
    assert s.name == NEUTRAL


def test_the_users_worked_sequence():
    """Their words: keeps saying DOWN, holds, then flips UP and they exit."""
    s, hist = run([0.68, 0.62, 0.57, 0.35])
    names = [h.name for h in hist]
    assert names == [DOWN, DOWN, DOWN, UP], names
    assert hist[-1].last_transition == (DOWN, UP)
    assert hist[1].last_transition is None, "holding must not emit a transition"


def test_up_reads_the_complementary_frequency():
    """P(up-first) is 1 - observed, because the label counts DOWN-first as 1. Getting this
    backwards would show the user 40% when the measurement says 60%."""
    up = observed_for(UP, 0.20, RELIABILITY, 0.490)
    assert up["pct"] == pytest.approx(1 - 0.397, abs=1e-9)
    assert up["base"] == pytest.approx(0.510, abs=1e-9)
    assert up["halves"][0] == pytest.approx(1 - 0.323, abs=1e-9)
    assert up["n"] == 363
    dn = observed_for(DOWN, 0.72, RELIABILITY, 0.490)
    assert dn["pct"] == pytest.approx(0.562, abs=1e-9)
    assert dn["base"] == pytest.approx(0.490, abs=1e-9)
    assert dn["n"] == 404


def test_displayed_number_is_above_its_own_base_rate_on_both_sides():
    for name, p in ((UP, 0.20), (DOWN, 0.72)):
        o = observed_for(name, p, RELIABILITY, 0.490)
        assert o["pct"] > o["base"], f"{name} would display a number below its base rate"
        assert all(h > o["base"] for h in o["halves"]), f"{name} tail fails in a half"


# --------------------------------------------------------------------------- the flip-flop
def test_dead_band_kills_the_flip_flop():
    path = [0.59, 0.61, 0.58, 0.62, 0.57]
    naive = sum(1 for a, b in zip(path, path[1:]) if (a >= 0.60) != (b >= 0.60))
    assert naive == 4, "the path must really cross a bare threshold, or this proves nothing"
    s, hist = run(path)
    settled = [h.last_transition for h in hist
               if h.last_transition and h.last_transition[0] != UNKNOWN]
    assert len(settled) == 1, f"{naive} bare crossings must collapse to 1, got {settled}"
    assert s.name == DOWN


def test_neutral_is_not_a_revolving_door():
    """Readings drifting around the middle must not produce a message each bar."""
    s, hist = run([0.50, 0.52, 0.48, 0.53, 0.47, 0.51])
    settled = [h.last_transition for h in hist
               if h.last_transition and h.last_transition[0] != UNKNOWN]
    assert settled == [], settled
    assert s.name == NEUTRAL


# --------------------------------------------------------------------------- asymmetry
def test_down_is_never_delayed_but_up_is():
    """§3ab: the short leg lost 10/10 while trimming to flat captured ~71% of the oracle prize,
    so being late to de-risk is the costly error and being early to re-risk is the other one."""
    s, _ = run([0.20])                       # UP
    assert s.name == UP
    s2 = step(s, 0.95, asof=T0 + timedelta(hours=30), th=TH,
              now=T0 + timedelta(hours=10, minutes=1))    # minutes after entering UP
    assert s2.name == DOWN, "de-risking must never be dwell-gated"

    s3, _ = run([0.95])                      # DOWN
    assert s3.name == DOWN
    early = step(s3, 0.10, asof=T0 + timedelta(hours=30), th=TH,
                 now=T0 + timedelta(hours=10, minutes=1))
    assert early.name == DOWN, "re-risking before min_dwell_h must be blocked"
    late = step(s3, 0.10, asof=T0 + timedelta(hours=30), th=TH,
                now=T0 + timedelta(hours=40))
    assert late.name == UP, "re-risking after min_dwell_h must be allowed"


def test_release_needs_a_margin_not_just_a_crossing():
    s, _ = run([0.72])
    assert s.name == DOWN
    t = T0 + timedelta(hours=30)
    assert step(s, 0.57, t, TH, T0 + timedelta(hours=40)).name == DOWN, "released inside the band"
    assert step(s, 0.52, t, TH, T0 + timedelta(hours=40)).name == NEUTRAL


# --------------------------------------------------------------------------- health
@pytest.mark.parametrize("health,p,asof", [
    ("stale_feed", 0.95, T0), ("model_unavailable", 0.95, T0),
    ("ok", None, T0), ("ok", 0.95, None), ("ok", 1.4, T0), ("ok", -0.2, T0),
])
def test_unhealthy_readings_freeze_rather_than_move(health, p, asof):
    s, _ = run([0.72])
    s2 = step(s, p, asof=asof, th=TH, now=T0 + timedelta(hours=40), health=health)
    assert s2.frozen is True and s2.name == DOWN and s2.health != "ok"
    assert s2.last_transition is None
    assert "HELD" in describe(s2, TH, T0 + timedelta(hours=40))


def test_a_dead_feed_never_quietly_becomes_neutral():
    s, _ = run([0.75])
    now = T0 + timedelta(hours=10)
    for _ in range(50):
        now += timedelta(hours=4)
        s = step(s, None, asof=None, th=TH, now=now, health="stale_feed")
    assert s.name == DOWN and s.frozen
    assert s.age_h(now) > 190


def test_thrash_guard_reports_and_does_not_suppress():
    th = Thresholds(down_at=0.60, down_release=0.55, up_at=0.40, up_release=0.45,
                    base_rate=0.490, min_dwell_h=0.0, max_changes_per_day=2)
    s, hist = run([0.72, 0.20, 0.72, 0.20, 0.72], th=th, bar_h=1, dwell_ok=False)
    assert sum(1 for h in hist if h.last_transition) >= 4
    assert s.health == "thrashing"
    assert s.name in (DOWN, UP) and s.frozen is False


# --------------------------------------------------------------------------- wording
def test_wording_never_offers_a_short_or_an_all_clear():
    now = T0 + timedelta(hours=12)
    for seq in ([0.95], [0.10], [0.50], [0.72, 0.20]):
        s, _ = run(seq)
        for text in (describe(s, TH, now), describe(s, TH, now, observed=0.562)):
            low = text.lower()
            for banned in ("short", "sell", "low risk", "safe"):
                assert banned not in low, f"{banned!r} in {text!r}"


def test_describe_quotes_the_measured_frequency_not_the_model_probability():
    s, _ = run([0.83])
    text = describe(s, TH, T0 + timedelta(hours=12), observed=0.562)
    assert "56%" in text and "83%" not in text


# --------------------------------------------------------------------------- persistence
def test_round_trip_through_json_preserves_behaviour():
    s, _ = run([0.72, 0.62])
    s2 = from_dict(to_dict(s))
    assert (s2.name, s2.p, s2.since, s2.asof, s2.health) == \
           (s.name, s.p, s.since, s.asof, s.health)
    now, bar = T0 + timedelta(hours=60), T0 + timedelta(hours=50)
    assert step(s, 0.30, bar, TH, now).name == step(s2, 0.30, bar, TH, now).name


def test_first_reading_resolves_unknown_and_is_not_dwell_blocked():
    s = step(AdvisorState(), 0.20, asof=T0, th=TH, now=T0)
    assert s.name == UP, "a first reading must not be blocked by an unserved dwell"
    assert s.last_transition == (UNKNOWN, UP)
