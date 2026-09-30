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
# The VPS that serves this runs 3.10 while the desktop runs 3.13; see the grammar test below.
SERVING_PYTHON = (3, 10)
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


def _state_for(p, meta):
    a = meta["adverse"]
    return "DOWN" if p >= a["down_at"] else "UP" if p <= a["up_at"] else "NEUTRAL"


def _reading(p, meta, observed=0.562):
    a = meta["adverse"]
    nm = _state_for(p, meta)
    return {"asof": "2026-09-18T20:00:00", "decision_time": "2026-09-19T00:00:00",
            "adverse": {"p": p, "observed": observed, "observed_n": 404,
                        "observed_halves": [0.596, 0.536], "observed_ci": None,
                        "base_rate": a["measured"]["base_rate"], "label": "lbl",
                        "advice": (a["advice_down"] if nm == "DOWN"
                                   else a["advice_up"] if nm == "UP" else a["advice_neutral"]),
                        "limits": a["limits"]},
            "direction": {"p_up": 0.52}}


def test_no_banned_word_in_subject_or_advice(meta):
    for p in (0.05, 0.30, 0.50, 0.56, 0.75, 0.99):
        for nm in ("DOWN", "NEUTRAL", "UP", "UNKNOWN"):
            m = build({"name": nm, "health": "ok", "since": str(NOW)},
                      _reading(p, meta), meta, {}, now=NOW)
            low = m["subject"].lower()
            for b in BANNED:
                assert b not in low, f"{b!r} in subject {m['subject']!r}"
            advice = [ln for ln in m["body"].splitlines() if ln.startswith("ADVICE")]
            for ln in advice:
                for b in ("go short", "sell short", "reverse"):
                    assert b not in ln.lower(), f"{b!r} in {ln!r}"


def test_body_always_states_the_base_rate_and_the_resolution_rate(meta):
    m = build({"name": "DOWN", "health": "ok", "since": str(NOW)},
              _reading(0.72, meta), meta, {}, now=NOW)
    assert "base rate" in m["body"].lower()
    assert "resolve" in m["body"].lower()
    assert "NOT USABLE" in m["body"], "the direction panel must be labelled unusable"
    assert "71.9%" in m["body"], "the feed-transfer tax must be stated"


def test_frozen_state_is_never_presented_as_current(meta):
    m = build({"name": "DOWN", "health": "stale_feed", "frozen": True, "since": str(NOW),
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


def test_both_tails_are_declared_validated(meta):
    """Two-sided calls are only legitimate because 1.0/9/BASE is the one cell of 16 whose top
    AND bottom buckets clear the base rate in both halves."""
    assert meta["adverse"]["both_tails_validated"] is True
    assert meta["adverse"]["up_at"] < meta["adverse"]["measured"]["base_rate"] \
        < meta["adverse"]["down_at"]


def test_direction_change_sends_immediately_and_names_both_states(meta):
    """The user's ask: tell me when the direction changes."""
    seen = {"state": "UP", "day": "2026-09-19"}     # today's message already went
    m = build({"name": "DOWN", "health": "ok", "since": str(NOW)},
              _reading(0.72, meta), meta, seen, now=NOW)
    assert m["trigger"] == "change" and m["should_send"], "a flip must send even after the daily"
    assert "DOWN" in m["subject"] and "was UP" in m["subject"]
    assert "CHANGED from UP" in m["body"]
    again = build({"name": "DOWN", "health": "ok", "since": str(NOW)},
                  _reading(0.72, meta), meta, m["seen"], now=NOW)
    assert not again["should_send"], "the same direction must not resend"


def test_a_flip_sends_even_with_the_market_closed(meta):
    seen = {"state": "NEUTRAL", "day": "2026-09-19"}
    m = build({"name": "DOWN", "health": "ok", "since": str(NOW)},
              _reading(0.72, meta), meta, seen, now=NOW, market_closed=True)
    assert m["should_send"] and m["trigger"] == "change"


def test_unchanged_direction_sends_once_a_day(meta):
    seen = {"state": "UP", "day": "2026-09-18"}
    m = build({"name": "UP", "health": "ok", "since": str(NOW)},
              _reading(0.20, meta), meta, seen, now=NOW)
    assert m["trigger"] == "daily" and m["should_send"]
    m2 = build({"name": "UP", "health": "ok", "since": str(NOW)},
               _reading(0.20, meta), meta, m["seen"], now=NOW)
    assert not m2["should_send"]
    m3 = build({"name": "UP", "health": "ok", "since": str(NOW)},
               _reading(0.20, meta), meta, m["seen"], now=NOW + timedelta(days=1))
    assert m3["should_send"], "the daily liveness message must resume the next day"


def test_first_run_does_not_announce_a_direction_change(meta):
    m = build({"name": "NEUTRAL", "health": "ok", "since": str(NOW)},
              _reading(0.50, meta), meta, {}, now=NOW)
    assert m["trigger"] == "daily", "UNKNOWN->first state is initialisation, not news"
    assert "was" not in m["subject"]


def test_up_state_shows_the_up_oriented_number(meta):
    """An UP card quoting 40% when the measurement says 60% would be the single worst bug
    available here, so it is asserted on the rendered body."""
    m = build({"name": "UP", "health": "ok", "since": str(NOW)},
              _reading(0.20, meta, observed=0.603), meta, {}, now=NOW)
    assert "60%" in m["body"] or "60.3" in m["body"]
    assert "rise 1 ATR before falling 1 ATR" in m["body"]


# --------------------------------------------------------------------------- artifact contract
def test_artifact_is_honest_about_what_it_is(meta):
    assert meta["direction"]["usable"] is False
    assert meta["direction"]["tier"] == "B"
    assert meta["adverse"]["usable"] == "tails_only"
    assert meta["adverse"]["arm"] == "BASE", "SOURCE was withdrawn in §3ba"
    assert meta["serve_needs_m15"] is False
    assert "CLEAN" in meta["feed"]
    assert len(meta["adverse"]["limits"]) >= 4
    for banned in ("go short", "safe", "guaranteed"):
        assert banned in meta["adverse"]["must_never_say"]
    # "low risk" is deliberately NOT banned any more: the low end is now the UP state and the
    # better-measured of the two sides (0.603 vs 0.562), so calling it is honest. What stays
    # banned is wording that turns a 6-in-10 directional reading into a safety guarantee.
    assert "low risk" not in meta["adverse"]["must_never_say"]


def test_thresholds_in_the_artifact_are_margins_around_the_base_rate(meta):
    a = meta["adverse"]
    Thresholds(down_at=a["down_at"], down_release=a["down_release"], up_at=a["up_at"],
               up_release=a["up_release"], base_rate=a["measured"]["base_rate"]).validate()


# --------------------------------------------------------------------------- target grammar
def test_every_advisor_module_parses_under_the_serving_pythons_grammar():
    """The VPS runs Python 3.10; this desktop runs 3.13.

    Quote reuse inside an f-string (`f"{d["k"]}"`) is PEP 701 and parses only from 3.12, so a
    local `ast.parse` check passes while the serving host raises SyntaxError. That is exactly
    what happened: the compute step died on the VPS and the notifier correctly emailed
    "NO READING — pipeline broken". The alerting worked; the deploy check did not. Pinning
    `feature_version` here makes the desktop refuse code the server cannot run.
    """
    targets = (sorted(ROOT.glob("scripts/v5_xau_advisor*.py"))
               + sorted(ROOT.glob("src/v5/xau_advisor*.py"))
               + sorted(ROOT.glob("src/v5/advisor_*.py"))
               + [ROOT / "scripts/v5_train_advisor.py", ROOT / "scripts/v5_feed_audit.py",
                  ROOT / "scripts/v5_advisor_feedcheck.py", ROOT / "scripts/v5_advisor_measure.py"])
    bad = []
    for f in targets:
        if not f.exists():
            continue
        try:
            ast.parse(f.read_text(), feature_version=SERVING_PYTHON)
        except SyntaxError as e:
            bad.append(f"{f.relative_to(ROOT)}:{e.lineno} {e.msg}")
    assert not bad, ("these modules will not parse on the serving host "
                     f"(Python {'.'.join(map(str, SERVING_PYTHON))}):\n  " + "\n  ".join(bad))


# --------------------------------------------------------------------------- outage back-off
def test_a_persistent_fault_backs_off_instead_of_nagging(meta):
    """Measured 2026-09-30: the FTMO bridge lost authorisation on 09-26 and the notifier sent
    47 IDENTICAL hourly emails over four days. That trains the reader to filter the channel.
    An unchanging fault carries no new information; its AGE does."""
    st = {"name": "NEUTRAL", "health": "model_unavailable", "asof": "2026-09-26T10:00:00"}
    fault = "AdvisorUnavailable: bridge 18814 init failed: (-6, 'Terminal: Authorization failed')"
    t0, seen, sent = NOW, {}, 0
    for i in range(4 * 24 * 12):                      # four days at the real 5-minute cadence
        m = build(st, None, meta, seen, now=t0 + timedelta(minutes=5 * i), broken=fault)
        seen = m["seen"]
        sent += bool(m["should_send"])
    assert sent <= 12, f"still nagging: {sent} emails over four days"
    assert sent >= 4, f"backed off too far, a 4-day outage must still be audible: {sent}"


def test_the_outage_email_states_its_age_and_the_actual_remedy(meta):
    st = {"name": "NEUTRAL", "health": "model_unavailable", "asof": "2026-09-26T10:00:00"}
    fault = "bridge 18814 init failed: (-6, 'Terminal: Authorization failed')"
    seen = {"broken_since": (NOW - timedelta(days=4)).isoformat()}
    m = build(st, None, meta, seen, now=NOW, broken=fault)
    assert "4.0 days" in m["subject"], m["subject"]
    # an auth fault has a specific fix; a generic runbook would be useless here
    assert "AUTHORISE" in m["body"] or "authoris" in m["body"].lower()
    assert "mt5-terminal-ftmo.service" in m["body"]
    assert "never restart the bare mt5-terminal.service" in m["body"], \
        "the destructive-ExecStop warning must travel with the remedy"


def test_a_recovered_reading_clears_the_outage_age(meta):
    """Otherwise the NEXT unrelated outage reports itself as days old on its first message —
    a false alarm indistinguishable from a real one."""
    fault = "bridge 18814 init failed"
    st = {"name": "NEUTRAL", "health": "model_unavailable", "asof": "2026-09-26T10:00:00"}
    seen = build(st, None, meta, {}, now=NOW - timedelta(days=3), broken=fault)["seen"]
    assert "broken_since" in seen
    good = {"name": "NEUTRAL", "health": "ok", "since": str(NOW), "asof": str(NOW)}
    r = build(good, _reading(0.50, meta), meta, seen, now=NOW)
    assert not [k for k in r["seen"] if "broken" in k], r["seen"]
    # and a fresh fault afterwards must start its clock from zero, not from three days ago
    again = build(st, None, meta, r["seen"], now=NOW + timedelta(minutes=5), broken=fault)
    assert "0 min" in again["subject"] or "min" in again["subject"], again["subject"]
