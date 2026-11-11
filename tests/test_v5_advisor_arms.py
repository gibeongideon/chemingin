"""The pure parts of the two FTMO advisor arms.

Arms LS (360611, long_short) and L (360612, long_only) must differ in EXACTLY ONE thing: whether
a SELL call opens a short. If they ever differ in what they CLOSE, the live A/B stops measuring
the short leg and starts measuring an exit rule as well (the §3co event-set rule).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_xau_advisor_trader import (  # noqa: E402
    BUY, SELL, Pos, decide, lots_for, paths_for, rescue_tp, tighter, trail_target,
)

EP, PREV = "2026-11-10T16:03:31+00:00", "2026-11-09T08:00:00+00:00"


def pos(ticket, typ, pnl, vol=0.06, open_px=4100.0, sl=0.0):
    return Pos(ticket, typ, vol, open_px, sl, pnl)


# ----------------------------------------------------------------- the flip rule, as asked ---

def test_flip_closes_the_winner_keeps_the_loser_and_opens_the_opposite():
    d = decide("DOWN", EP, PREV, [pos(1, BUY, +80.0), pos(2, BUY, -40.0)])
    assert d["close"] == [1]
    assert d["hedge"] == [2]
    assert d["open"] == SELL


def test_profit_is_profit_plus_swap_and_zero_is_not_profit():
    assert decide("DOWN", EP, PREV, [pos(1, BUY, 0.0)])["hedge"] == [1]
    assert decide("DOWN", EP, PREV, [pos(1, BUY, 0.01)])["close"] == [1]


def test_same_direction_opens_nothing_but_still_resolves_the_other_side():
    d = decide("UP", EP, PREV, [pos(1, BUY, -5.0), pos(2, SELL, +9.0)])
    assert d["open"] is None          # already long
    assert d["close"] == [2]          # the short was in profit -> taken
    assert d["hedge"] == []


def test_neutral_and_unknown_never_trade():
    for name in ("NEUTRAL", "UNKNOWN", "FROZEN"):
        d = decide(name, EP, PREV, [pos(1, BUY, -40.0)])
        assert d == {"open": None, "close": [], "hedge": [], "why": d["why"]}


# ------------------------------------------------- one action per episode (anti-martingale) ---

def test_the_flip_fires_once_per_episode_then_trails_only():
    first = decide("DOWN", EP, PREV, [pos(1, BUY, +80.0)])
    assert first["open"] == SELL and first["close"] == [1]
    again = decide("DOWN", EP, EP, [pos(1, BUY, +80.0)])
    assert again["open"] is None and again["close"] == [] and again["hedge"] == []


def test_a_stop_out_inside_the_same_episode_does_not_re_enter():
    assert decide("UP", EP, EP, [])["open"] is None


# ------------------------------------------------------- LS vs L differ in ONE thing only ---

@pytest.mark.parametrize("positions", [
    [], [pos(1, BUY, +80.0)], [pos(1, BUY, -40.0)],
    [pos(1, BUY, +80.0), pos(2, BUY, -40.0)],
])
def test_the_two_arms_close_and_keep_identically_on_a_sell_call(positions):
    ls = decide("DOWN", EP, PREV, positions, side_mode="long_short")
    lo = decide("DOWN", EP, PREV, positions, side_mode="long_only")
    assert ls["close"] == lo["close"]
    assert ls["hedge"] == lo["hedge"]
    assert ls["open"] == SELL and lo["open"] is None      # the one permitted difference


def test_the_two_arms_are_identical_on_a_buy_call():
    for positions in ([], [pos(1, SELL, +80.0)], [pos(1, SELL, -40.0)]):
        ls = decide("UP", EP, PREV, positions, side_mode="long_short")
        lo = decide("UP", EP, PREV, positions, side_mode="long_only")
        assert (ls["open"], ls["close"], ls["hedge"]) == (lo["open"], lo["close"], lo["hedge"])
        assert ls["open"] == BUY


# ---------------------------------------------------------------------- the account guard ---

def test_a_tripped_guard_blocks_the_entry_but_still_takes_the_winner():
    d = decide("DOWN", EP, PREV, [pos(1, BUY, +80.0), pos(2, BUY, -40.0)], halted=True)
    assert d["open"] is None          # no new risk
    assert d["close"] == [1]          # but reducing risk is still allowed
    assert d["hedge"] == [2]


# ------------------------------------------------------- the +$50 rescue target (user ask) ---

def test_rescue_tp_is_the_price_where_profit_plus_swap_equals_the_target():
    # 0.06 lots x 100 oz = $6 per $1 of gold, so +$50 needs $8.33 of price
    tp = rescue_tp(BUY, 4100.0, 0.06, 100.0, 0.0, 50.0, 2)
    assert tp == pytest.approx(4108.33, abs=0.01)
    assert (tp - 4100.0) * 0.06 * 100.0 == pytest.approx(50.0, abs=0.6)


def test_rescue_tp_pays_for_accrued_swap_and_mirrors_for_a_short():
    # gold is -92 pts/lot/day long on FTMO, so a leg held days must earn the swap back too
    assert rescue_tp(BUY, 4100.0, 0.06, 100.0, -20.0, 50.0, 2) > \
        rescue_tp(BUY, 4100.0, 0.06, 100.0, 0.0, 50.0, 2)
    assert rescue_tp(SELL, 4100.0, 0.06, 100.0, 0.0, 50.0, 2) == pytest.approx(4091.67, abs=0.01)


def test_the_rescue_target_sits_the_right_side_of_a_losing_leg():
    # a kept BUY is under water, so its +$50 target is above the market: a legal long TP
    assert rescue_tp(BUY, 4100.0, 0.06, 100.0, -5.0, 50.0, 2) > 4100.0
    assert rescue_tp(SELL, 4100.0, 0.06, 100.0, -5.0, 50.0, 2) < 4100.0


# ------------------------------------------------------------------------- the plumbing ---

def test_the_stop_only_ever_ratchets():
    assert tighter(BUY, 4100.0, 0.0) is True              # unset
    assert tighter(BUY, 4101.0, 4100.0) is True
    assert tighter(BUY, 4099.0, 4100.0) is False          # never loosens
    assert tighter(SELL, 4099.0, 4100.0) is True
    assert tighter(SELL, 4101.0, 4100.0) is False


def test_trail_sits_behind_the_best_price_on_both_sides():
    assert trail_target(BUY, 4150.0, 40.0, 1.0) == 4110.0
    assert trail_target(SELL, 4050.0, 40.0, 1.0) == 4090.0


def test_lots_come_from_risk_over_stop_distance():
    assert lots_for(100_000, 0.0025, 39.57, 100.0, 0.01, 100.0, 0.01) == 0.06
    assert lots_for(100_000, 0.005, 39.57, 100.0, 0.01, 100.0, 0.01) == 0.13


def test_each_arm_gets_its_own_state_and_log():
    a_state, a_log = paths_for("ftmo_ls")
    b_state, b_log = paths_for("ftmo_lo")
    default_state, _ = paths_for("")
    assert a_state != b_state and a_log != b_log
    assert default_state.name == "xau_advisor_trader_state.json"   # Switch arm untouched
