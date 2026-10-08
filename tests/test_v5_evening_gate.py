"""The evening no-increase gate: window arithmetic, the midnight wrap, and failing closed.

The user's requirement: "not to open new positions after 5PM if i close manually, but should
reopen as the market opens at night." Three things make this worth asserting rather than
eyeballing:

  * THE WINDOW WRAPS MIDNIGHT (17:00 -> 01:05). A naive `a <= hm < b` is False for the whole
    window, which would disable the gate completely while still printing a reassuring config.
  * 01:05 IS A MEASURED REOPEN, not a guess (M1-gap study 2026-10-08: XAUUSD and US100.cash
    break 23:49->01:05, UKOIL.cash 23:49->03:05, BTCUSD 23:54->00:05).
  * AN UNKNOWN CLOCK MUST FAIL CLOSED. If the broker tick cannot be read the gate has no idea
    what time it is, and the safe default for "may I add exposure?" is no.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_basket_challenge_exec import (  # noqa: E402
    NO_INCREASE_FROM, NO_INCREASE_UNTIL, no_increase_reason,
)


def at(h, m=0):
    return datetime(2026, 10, 8, h, m)


@pytest.mark.parametrize("h,m,blocked", [
    (16, 59, False),   # just before 5PM — still free to trade
    (17, 0, True),     # the cutoff itself is INSIDE the window
    (19, 40, True),    # the book's 19:40 server pass — the first one the user cared about
    (23, 40, True),    # the 23:40 pass, still before the 23:49 break
    (0, 30, True),     # wrapped past midnight, market in its break
    (1, 4, True),      # one minute before gold reopens
    (1, 5, False),     # gold's measured reopen — gate lifts
    (3, 40, False),    # the pass that actually restores the book (UKOIL open by 03:05)
    (11, 40, False),
])
def test_window_including_the_midnight_wrap(h, m, blocked):
    r = no_increase_reason(at(h, m), NO_INCREASE_FROM, NO_INCREASE_UNTIL)
    assert (r is not None) is blocked, f"{h:02d}:{m:02d} -> {r!r}"


def test_the_two_passes_the_user_asked_about_are_blocked_and_the_third_is_not():
    """Stated as the behaviour rather than the arithmetic, so the intent survives a refactor."""
    blocked = [t for t in (19 * 60 + 40, 23 * 60 + 40)
               if no_increase_reason(at(t // 60, t % 60)) is not None]
    assert len(blocked) == 2, "the evening passes must not add exposure"
    assert no_increase_reason(at(3, 40)) is None, "the 03:40 pass must restore the book"


def test_unknown_server_clock_fails_closed():
    r = no_increase_reason(None)
    assert r is not None and "failing closed" in r


def test_a_non_wrapping_window_still_works():
    """A future edit could set a same-day window; the branch must not silently invert."""
    frm, until = (10, 0), (12, 0)
    assert no_increase_reason(at(9, 59), frm, until) is None
    assert no_increase_reason(at(10, 0), frm, until) is not None
    assert no_increase_reason(at(11, 59), frm, until) is not None
    assert no_increase_reason(at(12, 0), frm, until) is None


def test_reason_names_the_clock_and_the_window():
    r = no_increase_reason(at(19, 40))
    assert "19:40" in r and "17:00" in r and "01:05" in r


def test_shipped_defaults_match_the_measured_reopen():
    assert NO_INCREASE_FROM == (17, 0)
    assert NO_INCREASE_UNTIL == (1, 5), "01:05 is XAUUSD's measured reopen"
