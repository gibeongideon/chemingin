"""Decision rules for the advisor executor (scripts/v5_xau_advisor_trader.py). Pure, no MT5."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_xau_advisor_trader import (  # noqa: E402
    BUY, SELL, Pos, decide, lots_for, tighter, trail_target,
)


def P(t, side, pnl, sl=0.0):
    return Pos(ticket=t, type=side, volume=0.1, price_open=4000.0, sl=sl, pnl=pnl)


def test_flat_up_opens_buy():
    d = decide("UP", "e1", None, [])
    assert d["open"] == BUY and d["close"] == [] and d["hedge"] == []


def test_flat_down_opens_sell():
    assert decide("DOWN", "e1", None, [])["open"] == SELL


def test_same_direction_no_order():
    d = decide("UP", "e2", None, [P(1, BUY, -50)])
    assert d["open"] is None and d["close"] == [] and d["hedge"] == []


def test_flip_closes_winner():
    d = decide("DOWN", "e2", "e1", [P(1, BUY, +12.5)])
    assert d["open"] == SELL and d["close"] == [1] and d["hedge"] == []


def test_flip_keeps_loser_as_hedge():
    d = decide("DOWN", "e2", "e1", [P(1, BUY, -30.0)])
    assert d["open"] == SELL and d["close"] == [] and d["hedge"] == [1]


def test_breakeven_is_not_profit():
    assert decide("UP", "e2", "e1", [P(1, SELL, 0.0)])["hedge"] == [1]


def test_hedge_then_flip_back_opens_nothing():
    # losing BUY kept + SELL opened; advisor returns to UP -> a BUY already exists
    d = decide("UP", "e3", "e2", [P(1, BUY, -40), P(2, SELL, +10)])
    assert d["open"] is None and d["close"] == []


def test_neutral_and_unknown_never_enter():
    for name in ("NEUTRAL", "UNKNOWN"):
        d = decide(name, "e1", None, [P(1, BUY, 50)])
        assert d["open"] is None and d["close"] == []


def test_no_reentry_in_same_episode_after_stopout():
    assert decide("UP", "e1", "e1", [])["open"] is None
    assert decide("UP", "e2", "e1", [])["open"] == BUY


def test_trail_ratchets_only():
    assert trail_target(BUY, 4050.0, 20.0) == 4030.0
    assert trail_target(SELL, 3950.0, 20.0) == 3970.0
    assert tighter(BUY, 4030.0, 4020.0) and not tighter(BUY, 4010.0, 4020.0)
    assert tighter(SELL, 3970.0, 3980.0) and not tighter(SELL, 3990.0, 3980.0)
    assert tighter(BUY, 1.0, 0.0) and tighter(SELL, 1.0, 0.0)


def test_lots_risk_half_percent():
    # $100k * 0.5% = $500 over a $40 stop on 100oz -> 0.125 -> 0.12 or 0.13 by step
    lots = lots_for(100_000, 0.005, 40.0, 100.0, 0.01, 100.0, 0.01)
    assert abs(lots - 0.12) < 0.011
    assert lots_for(100, 0.005, 40.0, 100.0, 0.01, 100.0, 0.01) == 0.01   # floors at vmin


def test_executor_trades_the_bias_not_the_three_state(tmp_path, monkeypatch):
    import json
    from datetime import datetime, timezone
    import scripts.v5_xau_advisor_trader as T
    now = datetime(2026, 11, 5, 17, 0, tzinfo=timezone.utc)
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"computed_utc": now.isoformat(), "health": "ok",
                             "state": {"name": "NEUTRAL"},
                             "bias": {"side": "SELL", "since": "x1", "p_raw": 0.53}}))
    monkeypatch.setattr(T, "ADVISOR_STATE", f)
    monkeypatch.setattr(T, "RULE", "BIAS")
    assert T.read_advisor(now)[:2] == ("DOWN", "x1")
    monkeypatch.setattr(T, "RULE", "STRONG")
    assert T.read_advisor(now)[0] == "NEUTRAL", "STRONG rule ignores the bias"
