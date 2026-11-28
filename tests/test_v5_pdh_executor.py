"""PDH day-trade executor (scripts/v5_pdh_executor.py): pure logic, then the whole live code path
through the simulated broker in scripts/v5_pdh_replay.py, one scenario per branch."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.v5_pdh_executor as X  # noqa: E402
from scripts.v5_pdh_replay import FakeMT5, replay, run_times  # noqa: E402

DAY = pd.Timestamp("2026-06-17")            # a Wednesday, not a holiday


# ------------------------------------------------------------------ synthetic bars (server clock)
def session(day: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(day + pd.Timedelta(hours=1, minutes=1), day + pd.Timedelta(hours=23, minutes=58), freq="1min")


def history_days(last: pd.Timestamp, n: int = 22) -> list[pd.Timestamp]:
    days, d = [], last - pd.Timedelta(days=1)
    while len(days) < n:
        if d.dayofweek < 5:
            days.append(d)
        d -= pd.Timedelta(days=1)
    return sorted(days)


def bars_for(day: pd.Timestamp, waypoints: list[tuple[str, float]]) -> pd.DataFrame:
    """Piecewise-linear price through (HH:MM, price) waypoints; high/low 0.05 either side."""
    idx = session(day)
    x = [((pd.Timestamp(f"{day.date()} {hm}") - idx[0]) / pd.Timedelta(minutes=1)) for hm, _ in waypoints]
    px = np.interp(np.arange(len(idx)), x, [p for _, p in waypoints])
    o = np.r_[px[0], px[:-1]]
    return pd.DataFrame(dict(open=o, high=np.maximum(o, px) + 0.05, low=np.minimum(o, px) - 0.05, close=px), index=idx)


def history(last: pd.Timestamp) -> pd.DataFrame:
    """22 identical days ranging 1990..2010: prior-day high 2010.05, true range 20.10 -> ATR14 20.10."""
    return pd.concat([bars_for(d, [("01:01", 2000), ("06:00", 2010), ("12:00", 1990), ("23:58", 2000)])
                      for d in history_days(last)])


def scenario(waypoints, day=DAY, extra_days=()):
    parts = [history(day), bars_for(day, waypoints)] + [bars_for(d, w) for d, w in extra_days]
    return pd.concat(parts)


def times(day, start="00:00", end="23:59"):
    lo, hi = pd.Timestamp(f"{day.date()} {start}"), pd.Timestamp(f"{day.date()} {end}")
    return [t for t in run_times(day, day) if lo <= pd.Timestamp(t, unit="s") <= hi]


def events(rows):
    return [r["event"] for r in rows]


@pytest.fixture(autouse=True)
def no_real_side_effects(monkeypatch):
    sent = []
    monkeypatch.setattr(X, "alert", lambda s, b: sent.append(s))
    monkeypatch.setattr(X.time, "sleep", lambda s: None)
    yield sent


# ------------------------------------------------------------------ pure logic
def test_module_parses_as_python_310():
    for f in ("scripts/v5_pdh_executor.py", "scripts/v5_pdh_replay.py"):
        ast.parse((ROOT / f).read_text(), feature_version=(3, 10))


def test_levels_use_completed_days_only():
    rows = [(int((d - pd.Timestamp(0)).total_seconds()), 100.0 + i, 110.0 + i, 90.0 + i, 105.0 + i)
            for i, d in enumerate(pd.date_range("2026-01-01", periods=20, freq="D"))]
    d1 = X.daily_table(rows)
    today = pd.Timestamp("2026-01-20")                       # the 20th row is today's forming bar
    pdh, atr, bar = X.levels(d1, today)
    assert pdh == 110.0 + 18                                 # yesterday, never today
    # true range = max(H-L=20, |H-Cprev|=6, |L-Cprev|=14) = 20 on every day
    assert atr == pytest.approx(20.0)
    assert float(bar.high) == 110.0 + 19
    with pytest.raises(ValueError):
        X.levels(d1.iloc[:10], pd.Timestamp("2026-01-11"))


def test_size_lots_floors_and_caps():
    lots, how = X.size_lots(140_000, 95.0, 100.0, 0.01, 0.01, 35.0, 3900.0)
    assert lots == 0.07 and how == "risk budget"             # 700 / 9500 = 0.0737 -> floor 0.07
    lots, how = X.size_lots(10_000, 95.0, 100.0, 0.01, 0.01, 35.0, 3900.0)
    assert lots == 0.0 and "minimum lot" in how              # 0.01 lot risks 95 > 1.5 x 50
    lots, _ = X.size_lots(10_000_000, 1.0, 100.0, 0.01, 0.01, 35.0, 3900.0)
    assert lots == X.HARD_MAX_LOTS


def test_plan_entry_branches():
    now = DAY + pd.Timedelta(hours=10)
    kw = dict(srv_now=now, level=2010.0, ask=2000.0, spread=0.2, today_high_bid=2005.0, open_ask=2000.0,
              minutes_open=500.0, halted=None)
    assert X.plan_entry(**kw)[0] == "place"
    assert X.plan_entry(**{**kw, "halted": "x"})[0] == "halt"
    assert X.plan_entry(**{**kw, "open_ask": 2010.0})[0] == "skip"                       # gap
    assert X.plan_entry(**{**kw, "today_high_bid": 2009.9})[0] == "missed"              # ask-high touched
    assert X.plan_entry(**{**kw, "ask": 2011.0, "minutes_open": 4.0})[0] == "market"    # early break
    assert X.plan_entry(**{**kw, "srv_now": DAY + pd.Timedelta(hours=23, minutes=41)})[0] == "skip"
    assert X.plan_entry(**{**kw, "srv_now": pd.Timestamp("2026-11-26 10:00")})[0] == "skip"   # Thanksgiving


def test_exit_action_and_expiration():
    opened = DAY + pd.Timedelta(hours=9)
    assert X.exit_action(DAY + pd.Timedelta(hours=23, minutes=39), opened) is None
    assert X.exit_action(DAY + pd.Timedelta(hours=23, minutes=40), opened) == "exit"
    assert X.exit_action(DAY + pd.Timedelta(days=1, hours=1, minutes=2), opened) == "stale"
    exp = X.expiration_epoch(DAY + pd.Timedelta(hours=1, minutes=1, seconds=5))
    assert pd.Timestamp(exp, unit="s") == DAY + pd.Timedelta(hours=23, minutes=40)


def test_kill_rules():
    fills = [dict(event="fill", entry="stop", slip_bp="3.0")] * 39
    assert X.kill_status(fills)[0] is None                   # 39 fills: not yet evaluable
    assert "KILL RULE 1" in X.kill_status(fills + [dict(event="fill", entry="stop", slip_bp="3.0")])[0]
    late = [dict(event="fill", entry="market", slip_bp="9.0")] * 50
    assert X.kill_status(late)[0] is None                    # market entries are not slippage
    dd = [dict(event="closed", R="1.0")] * 5 + [dict(event="closed", R="-1.0")] * 24
    assert X.kill_status(dd)[0] is None                      # peak +5, trough -19: drawdown 24R
    assert "KILL RULE 2" in X.kill_status(dd + [dict(event="closed", R="-1.0")])[0]


# ------------------------------------------------------------------ the live path, simulated
def test_normal_day_fills_and_exits_at_cutoff(tmp_path):
    bars = scenario([("01:01", 2000), ("09:00", 2005), ("12:00", 2020), ("23:58", 2018)])
    rows, fake, alerts = replay(bars, None, None, workdir=tmp_path, times=times(DAY))
    ev = events(rows)
    assert ev[:2] == ["place", "fill"] and ev[-2:] == ["exit", "closed"], ev
    place = rows[0]
    assert float(place["level"]) == pytest.approx(2010.05)
    assert float(place["sl"]) == pytest.approx(2010.05 - 20.10)
    first_order = next(r for r in fake.sent if r["action"] == fake.TRADE_ACTION_PENDING)
    assert pd.Timestamp(first_order["expiration"], unit="s") == DAY + pd.Timedelta(hours=23, minutes=40)
    assert pd.Timestamp(place["srv"]) == DAY + pd.Timedelta(hours=1, minutes=1, seconds=5)   # the open burst
    fill = rows[1]
    assert float(fill["slip_bp"]) == pytest.approx(0.0, abs=1e-6)
    closed = rows[-1]
    assert closed["note"].startswith("exit") and float(closed["R"]) > 0
    assert pd.Timestamp(rows[-2]["srv"]) == DAY + pd.Timedelta(hours=23, minutes=44, seconds=5)
    assert not fake.positions and not fake.orders
    assert alerts == []


def test_stop_out_is_one_R(tmp_path):
    bars = scenario([("01:01", 2000), ("08:00", 2015), ("14:00", 1980), ("23:58", 1985)])
    rows, _, _ = replay(bars, None, None, workdir=tmp_path, times=times(DAY))
    closed = [r for r in rows if r["event"] == "closed"]
    assert len(closed) == 1 and closed[0]["note"].startswith("stop")
    assert float(closed[0]["R"]) == pytest.approx(-1.0, abs=0.02)


def test_gap_open_is_skipped(tmp_path):
    bars = scenario([("01:01", 2015), ("23:58", 2030)])
    rows, fake, _ = replay(bars, None, None, workdir=tmp_path, times=times(DAY))
    assert events(rows) == ["skip"] and "opened at/above" in rows[0]["note"]
    assert not fake.sent


def test_unbroken_day_expires_server_side(tmp_path):
    bars = scenario([("01:01", 2000), ("12:00", 2008), ("23:58", 2001)])
    rows, fake, _ = replay(bars, None, None, workdir=tmp_path, times=times(DAY))
    assert events(rows) == ["place", "expired"]
    assert not fake.orders


def test_vanished_order_is_recorded_not_replaced(tmp_path):
    bars = scenario([("01:01", 2000), ("23:58", 2001)])
    fake = FakeMT5(bars)
    t = times(DAY)
    split = next(i for i, x in enumerate(t) if pd.Timestamp(x, unit="s") >= DAY + pd.Timedelta(hours=1, minutes=5))
    replay(bars, None, None, workdir=tmp_path, times=t[:split], fake=fake)   # through the open burst
    o = next(iter(fake.orders.values()))
    fake._finish(o, 2)                                       # someone deletes it by hand
    rows, fake, alerts = replay(bars, None, None, workdir=tmp_path, times=t[split:], fake=fake)
    assert events(rows) == ["place", "vanished"]
    assert sum(1 for r in fake.sent if r["action"] == fake.TRADE_ACTION_PENDING) == 1
    assert "order vanished" in alerts


def test_early_break_bought_at_market_late_one_missed(tmp_path):
    bars = scenario([("01:01", 2008), ("01:04", 2012), ("23:58", 2020)])
    rows, _, _ = replay(bars, None, None, workdir=tmp_path / "a", times=times(DAY, start="01:08"))
    assert events(rows)[:2] == ["enter_market", "fill"] and rows[1]["entry"] == "market"
    rows, fake, _ = replay(bars, None, None, workdir=tmp_path / "b", times=times(DAY, start="06:00"))
    assert events(rows) == ["missed"] and not fake.sent


def test_holiday_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "HOLIDAYS", frozenset({str(DAY.date())}))
    bars = scenario([("01:01", 2000), ("12:00", 2020), ("23:58", 2018)])
    rows, fake, _ = replay(bars, None, None, workdir=tmp_path, times=times(DAY))
    assert events(rows) == ["skip"] and "holiday" in rows[0]["note"] and not fake.sent


def test_position_that_survives_the_day_is_closed_at_the_next_open(tmp_path):
    nxt = DAY + pd.Timedelta(days=1)
    bars = scenario([("01:01", 2000), ("12:00", 2020), ("23:58", 2018)], extra_days=[(nxt, [("01:01", 2019), ("23:58", 2019)])])
    t = times(DAY, end="23:30") + times(nxt, end="01:03")    # the bot is down through the cut-off window
    rows, fake, alerts = replay(bars, None, None, workdir=tmp_path, times=t)
    ev = events(rows)
    assert ev[:4] == ["place", "fill", "stale", "closed"], ev       # flat at the first run after the open
    assert rows[3]["note"].startswith("stale_exit")
    assert ev[4:] == ["place"]                                       # and the new day gets its own order
    assert not fake.positions
    assert "stale position closed" in alerts


def test_kill_rule_blocks_new_entries(tmp_path):
    files = X.Files("replay", base=tmp_path)
    for _ in range(40):
        files.append(dict(event="fill", entry="stop", slip_bp=3.0))
    bars = scenario([("01:01", 2000), ("12:00", 2020), ("23:58", 2018)])
    rows, fake, alerts = replay(bars, None, None, workdir=tmp_path, times=times(DAY))
    assert events(rows)[-1] == "halt" and not fake.sent
    assert "KILL RULE TRIPPED — no new entries" in alerts


def test_dry_run_writes_nothing(tmp_path):
    bars = scenario([("01:01", 2000), ("12:00", 2020), ("23:58", 2018)])
    fake = FakeMT5(bars)
    fake.set_time(times(DAY)[0])
    files = X.Files("dry", base=tmp_path)
    from types import SimpleNamespace as NS
    r = X.Runner(fake, NS(execute=False, symbol="XAUUSDs"), files, {}, [])
    assert r.run() == 0
    assert not fake.sent and not fake.orders
    assert not files.log.exists() and not files.state.exists()


def test_plan_entry_crypto_flags():
    kw = dict(srv_now=pd.Timestamp("2026-11-28 10:00"), level=2010.0, ask=2000.0, spread=0.3,      # a Saturday
              today_high_bid=2005.0, open_ask=1999.0, minutes_open=300.0, halted=None)
    assert X.plan_entry(**kw)[0] == "place"                                    # default: weekends are traded
    assert X.plan_entry(**kw, weekdays_only=True)[0] == "skip"
    th = {**kw, "srv_now": pd.Timestamp("2026-11-26 10:00")}                    # Thanksgiving, a Thursday
    assert X.plan_entry(**th)[0] == "skip"
    assert X.plan_entry(**th, ignore_holidays=True)[0] == "place"
