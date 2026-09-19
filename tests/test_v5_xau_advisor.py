"""Service-level tests for the advisor: the message contract and the absence of an order path.

The no-order-path test is not decoration. Everything else in `scripts/` that touches a bridge
can trade, and this one must not be able to, so it is asserted by walking the actual import
closure rather than trusting a docstring.
"""
from __future__ import annotations

import ast
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_xau_advisor_message import BANNED, build  # noqa: E402
from src.v5.xau_advisor_state import AdvisorState, Thresholds  # noqa: E402

ART = ROOT / "data" / "models" / "xau_advisor.json"
NOW = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)

ENTRY = ["scripts/v5_xau_advisor.py", "scripts/v5_xau_advisor_notify.py",
         "scripts/v5_xau_advisor_message.py", "src/v5/xau_advisor_model.py",
         "src/v5/xau_advisor_state.py"]
FORBIDDEN = ("order_send", "TRADE_ACTION", "ORDER_TYPE_BUY", "ORDER_TYPE_SELL",
             "position_close", "Close_by")


def _closure(entries: list) -> dict:
    """Local repo modules reachable from `entries`, as {relpath: source}."""
    seen, queue = {}, list(entries)
    while queue:
        rel = queue.pop()
        p = ROOT / rel
        if rel in seen or not p.exists():
            continue
        src = p.read_text()
        seen[rel] = src
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        mods = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods += [al.name for al in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods.append(n.module)
        for m in mods:
            if m.split(".")[0] in ("scripts", "src"):
                cand = ROOT / (m.replace(".", "/") + ".py")
                if cand.exists():
                    queue.append(str(cand.relative_to(ROOT)))
    return seen


def test_no_order_path_anywhere_in_the_import_closure():
    closure = _closure(ENTRY)
    assert len(closure) >= len(ENTRY), f"closure looks truncated: {sorted(closure)}"
    hits = []
    for rel, src in closure.items():
        # Strip comments and docstrings: the modules legitimately DISCUSS the absence of an
        # order path, and a naive grep would fail on its own explanation.
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        code = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                code.append(node.attr)
            elif isinstance(node, ast.Name):
                code.append(node.id)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                code.append(node.value)
        blob = "\n".join(code)
        for bad in FORBIDDEN:
            if bad in blob:
                hits.append(f"{rel}: {bad}")
    assert not hits, f"advisor import closure can trade: {hits}"


def test_the_closure_does_not_reach_an_executor():
    closure = _closure(ENTRY)
    leaked = [r for r in closure
              if any(k in r for k in ("exec", "basket_challenge", "zigzag_ftmo", "trade"))]
    assert not leaked, f"advisor imports execution code: {leaked}"


# --------------------------------------------------------------------------- message contract
@pytest.fixture
def meta():
    if not ART.exists():
        pytest.skip("artifact not built — run scripts/v5_train_advisor.py")
    return json.loads(ART.read_text())


def _reading(p, meta, observed=0.57):
    a = meta["adverse"]
    return {"asof": "2026-09-18T20:00:00", "decision_time": "2026-09-19T00:00:00",
            "adverse": {"p": p, "observed": observed, "observed_n": 820,
                        "observed_ci": [0.530, 0.596],
                        "base_rate": a["measured"]["base_rate"], "label": "lbl",
                        "advice": a["advice_warn"] if p >= a["warn_at"] else a["advice_clear"],
                        "limits": a["limits"]},
            "direction": {"p_up": 0.52}}


def test_no_banned_word_in_subject_or_advice(meta):
    for p in (0.05, 0.30, 0.50, 0.56, 0.75, 0.99):
        for nm in ("CLEAR", "ELEVATED", "UNKNOWN"):
            m = build({"name": nm, "health": "ok", "since": str(NOW)},
                      _reading(p, meta), meta, {}, now=NOW)
            low = m["subject"].lower()
            for b in BANNED:
                assert b not in low, f"{b!r} in subject {m['subject']!r}"
            advice = [ln for ln in m["body"].splitlines() if ln.startswith("ADVICE")]
            for ln in advice:
                for b in ("short", "sell", "low risk", "safe"):
                    assert b not in ln.lower(), f"{b!r} in {ln!r}"


def test_body_always_states_the_base_rate_and_the_resolution_rate(meta):
    m = build({"name": "ELEVATED", "health": "ok", "since": str(NOW)},
              _reading(0.60, meta), meta, {}, now=NOW)
    assert "base rate" in m["body"].lower()
    assert "resolve" in m["body"].lower()
    assert "NOT USABLE" in m["body"], "the direction panel must be labelled unusable"
    assert "71.9%" in m["body"], "the feed-transfer tax must be stated"


def test_frozen_state_is_never_presented_as_current(meta):
    m = build({"name": "ELEVATED", "health": "stale_feed", "frozen": True, "since": str(NOW),
               "asof": "2026-09-10T00:00:00"}, None, meta, {}, now=NOW)
    assert "HELD" in m["subject"]
    assert "not current" in m["body"].lower() or "FROZEN" in m["body"]


def test_broken_pipeline_sends_once_per_hour(meta):
    seen = {}
    m1 = build({}, None, meta, seen, now=NOW, broken="bridge down")
    assert m1["should_send"] and m1["trigger"] == "broken"
    m2 = build({}, None, meta, m1["seen"], now=NOW, broken="bridge down")
    assert not m2["should_send"], "duplicate broken alert in the same hour"
    m3 = build({}, None, meta, m1["seen"], now=NOW + timedelta(hours=1), broken="bridge down")
    assert m3["should_send"], "broken alert must repeat the next hour"


def test_state_change_absorbs_that_hours_heartbeat(meta):
    seen = {"state": "CLEAR", "hour": "2026-09-19T11"}
    m = build({"name": "ELEVATED", "health": "ok", "since": str(NOW)},
              _reading(0.60, meta), meta, seen, now=NOW)
    assert m["trigger"] == "change" and m["should_send"]
    assert m["seen"]["hour"] == "2026-09-19T12"
    again = build({"name": "ELEVATED", "health": "ok", "since": str(NOW)},
                  _reading(0.60, meta), meta, m["seen"], now=NOW)
    assert not again["should_send"], "heartbeat fired in the same hour as the change alert"


def test_heartbeat_fires_once_an_hour_when_nothing_changes(meta):
    seen = {"state": "CLEAR", "hour": "2026-09-19T11"}
    m = build({"name": "CLEAR", "health": "ok", "since": str(NOW)},
              _reading(0.20, meta), meta, seen, now=NOW)
    assert m["trigger"] == "heartbeat" and m["should_send"]
    m2 = build({"name": "CLEAR", "health": "ok", "since": str(NOW)},
               _reading(0.20, meta), meta, m["seen"], now=NOW)
    assert not m2["should_send"]


def test_first_run_does_not_announce_a_state_change(meta):
    m = build({"name": "CLEAR", "health": "ok", "since": str(NOW)},
              _reading(0.20, meta), meta, {}, now=NOW)
    assert m["trigger"] == "heartbeat", "UNKNOWN->CLEAR is initialisation, not news"
    assert "[was" not in m["subject"]


# --------------------------------------------------------------------------- artifact contract
def test_artifact_is_honest_about_what_it_is(meta):
    assert meta["direction"]["usable"] is False
    assert meta["direction"]["tier"] == "B"
    assert meta["adverse"]["usable"] == "tails_only"
    assert meta["adverse"]["arm"] == "BASE", "SOURCE was withdrawn in §3ba"
    assert meta["serve_needs_m15"] is False
    assert "CLEAN" in meta["feed"]
    assert len(meta["adverse"]["limits"]) >= 4
    for banned in ("low risk", "go short"):
        assert banned in meta["adverse"]["must_never_say"]


def test_thresholds_in_the_artifact_are_margins_around_the_base_rate(meta):
    a = meta["adverse"]
    th = Thresholds(warn_at=a["warn_at"], clear_at=a["clear_at"],
                    base_rate=a["measured"]["base_rate"])
    th.validate()
    assert a["warn_at"] > a["measured"]["base_rate"]
