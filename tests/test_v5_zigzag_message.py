"""A notifier may only claim a fill the broker acknowledged.

REAL INCIDENT, 2026-11-10. `zigzag-obs` runs `v5_zigzag_ftmo.py` WITHOUT `--live --execute`
(trading disabled 2026-09-16: walk-forward SR -0.76, 0/8 years, DSR 0.000). It fired at
P 0.6149 >= 0.60, wrote `actions:[BUY 0.12 @ 4027.47]`, `executed:false`, `sent:[]` — and the
email said "The demo bot has already placed this." No such order existed on any account. The
user went looking for it across three brokers.

`actions` is the PLAN. `sent` is what reached a broker. These tests keep them apart.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_zigzag_message import build  # noqa: E402

ACTION = {"kind": "BUY", "vol": 0.12, "why": "prob 0.615 >= 0.6", "entry": 4027.47,
          "sl": 4007.33, "tp": 4051.63, "close_by": "2026-11-12 19:04 UTC", "age_h": None}


def state(**kw):
    s = {"symbol": "XAUUSD", "prob": 0.6149, "threshold": 0.60, "fires": True,
         "actions": [ACTION], "sent": [], "executed": False, "open_positions": 0,
         "prospective": {"lots": 0.12, "entry": 4027.47, "sl": 4007.33, "tp": 4051.63,
                         "sl_pct": 0.005, "tp_pct": 0.006}}
    s.update(kw)
    return s


def test_a_signal_only_job_never_claims_an_order_exists():
    m = build(state())                       # executed False, sent [] — the obs job
    assert "already placed" not in m["body"]
    assert "NOTHING HAS BEEN PLACED" in m["body"]
    assert "SIGNAL (no order)" in m["subject"]
    assert "TRADABLE —" not in m["subject"]


def test_an_executed_fire_with_a_broker_ack_does_claim_it():
    m = build(state(executed=True, sent=[{"kind": "BUY", "retcode": 10009, "order": 123}]))
    assert "already placed" in m["body"]
    assert m["subject"].startswith("[zigzag] TRADABLE — BUY 0.12 lots XAUUSD @ 4027.47")


def test_a_rejected_order_is_not_a_placed_order():
    # 10006 = request rejected. The plan existed, the order did not.
    m = build(state(executed=True, sent=[{"kind": "BUY", "retcode": 10006}]))
    assert "already placed" not in m["body"]
    assert "SIGNAL (no order)" in m["subject"]


def test_execute_flag_without_any_sent_entry_is_not_a_placed_order():
    m = build(state(executed=True, sent=[]))
    assert "already placed" not in m["body"]


def test_10008_placed_counts_as_acknowledged():
    m = build(state(executed=True, sent=[{"kind": "BUY", "retcode": 10008}]))
    assert "already placed" in m["body"]


def test_the_order_levels_are_reported_either_way():
    for s in (state(), state(executed=True, sent=[{"kind": "BUY", "retcode": 10009}])):
        b = build(s)["body"]
        assert "4027.47" in b and "4007.33" in b and "4051.63" in b
        assert "0.12 lots" in b
        assert build(s)["should_send"] is True


def test_the_watch_band_still_says_no_order_exists():
    m = build(state(prob=0.57, actions=[], fires=False), prev_band="none")
    assert "No order has been placed" in m["body"]
    assert "NOT TRADABLE" in m["subject"]
