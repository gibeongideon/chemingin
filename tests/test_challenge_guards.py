"""Guard-logic tests for the FundingPips challenge bot: each guard must trip
in the SAME evaluation that breaches its line, and locks must persist/reset
exactly at the platform-day boundary."""
from datetime import datetime, timezone

from src.v5.challenge_guards import (advance_phase, decide, init_state)

T0 = datetime(2026, 7, 14, 8, 0, tzinfo=timezone.utc)   # 11:00 UTC+3
T1 = datetime(2026, 7, 14, 20, 59, tzinfo=timezone.utc)  # 23:59 UTC+3
T2 = datetime(2026, 7, 14, 21, 1, tzinfo=timezone.utc)   # 00:01 UTC+3 NEXT day


def test_normal_pass_trades():
    s = init_state(100_000, 100_000, T0)
    s, a = decide(s, 100_000, 100_200, T0)
    assert a == "trade"


def test_daily_guard_trips_same_pass_and_locks_till_reset():
    s = init_state(100_000, 100_000, T0)
    s, a = decide(s, 100_000, 96_400, T0)          # -3.6% < -3.5%
    assert a == "day_lock" and s["day_locked"]
    s, a = decide(s, 96_400, 97_500, T1)           # recovered, same day
    assert a == "locked"                           # still locked
    s, a = decide(s, 96_400, 97_500, T2)           # new platform day
    assert a == "trade" and not s["day_locked"]
    assert s["day_anchor"] == 97_500               # anchor = max(bal, eq)


def test_daily_guard_uses_day_anchor_not_initial():
    s = init_state(100_000, 100_000, T0)
    s, _ = decide(s, 104_000, 104_000, T2)         # new day, anchor 104k
    s, a = decide(s, 104_000, 100_300, T2)         # -3.56% from anchor
    assert a == "day_lock"


def test_overall_halt_trips_and_is_permanent():
    s = init_state(100_000, 100_000, T0)
    s, a = decide(s, 93_000, 91_900, T0)           # -8.1% < -8%
    assert a == "halt" and s["halted"]
    s, a = decide(s, 95_000, 99_000, T2)           # recovery cannot unhalt
    assert a == "halt"


def test_overall_halt_beats_daily_guard():
    s = init_state(100_000, 100_000, T0)
    s, a = decide(s, 95_000, 91_000, T0)           # breaches both
    assert a == "halt"


def test_target_realize_then_complete():
    s = init_state(100_000, 100_000, T0)
    s, a = decide(s, 100_000, 108_200, T0)         # +8.2% floating
    assert a == "realize_target"                   # close to bank it
    s, a = decide(s, 108_100, 108_100, T0)         # now realized, flat
    assert a == "complete" and s["phase_complete"]
    s, a = decide(s, 108_100, 108_100, T2)
    assert a == "complete"                         # stays complete


def test_floating_above_target_but_balance_below_keeps_trading_state():
    s = init_state(100_000, 100_000, T0)
    s, a = decide(s, 107_000, 108_500, T0)         # balance below target
    assert a == "realize_target"


def test_phase2_fresh_budget_and_target():
    s = init_state(100_000, 100_000, T0)
    s["phase_complete"] = True
    s = advance_phase(s, 108_000)
    assert s["phase"] == 2 and s["phase_target_frac"] == 0.05
    s, a = decide(s, 108_000, 113_500, T0)         # +5.09% on phase start
    assert a == "realize_target"
    # overall halt still anchored to INITIAL balance
    s2, a2 = decide(s, 108_000, 91_900, T0)
    assert a2 == "halt"


def test_cushion_multiplier_static_floor():
    from src.v5.challenge_guards import cushion_multiplier
    st = {"initial_balance": 100_000.0}
    assert cushion_multiplier(st, 100_000, 0.03) == 1.0          # start: full cushion = 1x
    assert abs(cushion_multiplier(st, 101_500, 0.03) - 1.5) < 1e-9
    assert cushion_multiplier(st, 110_000, 0.03) == 2.0          # capped
    assert abs(cushion_multiplier(st, 98_500, 0.03) - 0.5) < 1e-9
    assert cushion_multiplier(st, 96_000, 0.03) == 0.0           # at/below the floor -> flat


def test_ny_plus_7_server_day_follows_us_dst():
    from src.v5 import challenge_guards as cg
    tz = cg.NYPlus(7)
    # winter (EST): 21:59 UTC is still the old server day, 22:00 UTC starts the next one (UTC+2)
    assert datetime(2026, 11, 27, 21, 59, tzinfo=timezone.utc).astimezone(tz).strftime("%F %H") == "2026-11-27 23"
    assert datetime(2026, 11, 27, 22, 0, tzinfo=timezone.utc).astimezone(tz).strftime("%F %H") == "2026-11-28 00"
    # summer (EDT): the boundary moves to 21:00 UTC (UTC+3)
    assert datetime(2026, 7, 15, 21, 0, tzinfo=timezone.utc).astimezone(tz).strftime("%F %H") == "2026-07-16 00"
    cg.set_reset_tz("NY+7")
    assert cg.platform_date(datetime(2026, 11, 27, 22, 30, tzinfo=timezone.utc)) == "2026-11-28"
    cg.set_reset_tz("UTC+3")
