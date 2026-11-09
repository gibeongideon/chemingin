from scripts.v5_trade_events_notify import diff, milestones

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


def test_zigzag_profit_milestones_fire_once_highest_level_only():
    z = dict(magic=360601, symbol="XAUUSDs", type=0, volume=0.05, price=4000.0, sl=3950.0, profit=60.0)
    ev = milestones({"positions": {}}, {"9": dict(z)})
    assert [(k, x["level"]) for k, _, x in ev] == [("PROFIT", 50.0)]
    prev = {"positions": {"9": dict(z, hits=[50.0])}}
    assert milestones(prev, {"9": dict(z, profit=120.0)}) == []                  # no repeat
    ev = milestones(prev, {"9": dict(z, profit=310.0)})                          # jump past 200 and 300
    assert len(ev) == 1 and ev[0][2]["level"] == 300.0 and ev[0][2]["hits"] == [50.0, 200.0, 300.0]
    assert milestones({"positions": {}}, {"2": dict(P, profit=500.0)}) == []     # Book is not ZigZag


def test_zigzag_watch_and_signal_alerts_once_per_bar():
    from scripts.v5_trade_events_notify import zz_alert
    zz = dict(prob=0.5713, threshold=0.6, last_bar="2026-11-09 10:00:00", open_positions=0, actions=[],
              prospective=dict(symbol="XAUUSDs", price=4095.33))
    head, body = zz_alert(zz, None)
    assert "WATCH" in head and "0.57" in head
    assert zz_alert(zz, "2026-11-09 10:00:00") is None                       # same bar: silent
    assert zz_alert(dict(zz, prob=0.54), None) is None                       # below 0.55: silent
    head, body = zz_alert(dict(zz, prob=0.63, actions=[{"x": 1}]), None)
    assert "SIGNAL" in head and "order placed" in body
    head, body = zz_alert(dict(zz, prob=0.63, open_positions=1), None)
    assert "already in a ZigZag trade" in body
