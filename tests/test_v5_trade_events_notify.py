from scripts.v5_trade_events_notify import diff

P = dict(magic=360595, symbol="XAUUSDs", type=0, volume=0.02, price=4123.47, sl=2075.45, profit=0.0)
O = dict(magic=360600, symbol="XAUUSDs", type=4, volume=0.05, price=4146.9, sl=4060.0, tp=0.0)


def test_new_order_is_tradable_new_position_is_traded():
    ev = diff({"orders": {}, "positions": {}}, {"1": O}, {"2": P})
    assert ("TRADABLE", "1", O) in ev and ("TRADED", "2", P) in ev


def test_vanished_position_is_closed_and_unchanged_state_is_silent():
    prev = {"orders": {"1": O}, "positions": {"2": P}}
    assert diff(prev, {"1": O}, {"2": P}) == []
    assert diff(prev, {}, {}) == [("CLOSED", "2", P)]       # a filled/cancelled order alone is silent


def test_volume_change_is_reported_as_resized():
    prev = {"orders": {}, "positions": {"2": P}}
    ev = diff(prev, {}, {"2": dict(P, volume=0.03)})
    assert ev[0][0] == "RESIZED" and ev[0][2]["old_volume"] == 0.02
