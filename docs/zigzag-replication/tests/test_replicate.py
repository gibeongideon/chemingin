"""Tests for replicate.py.  Run from the package root:   pytest -q tests/

  1. matches the original project code (features, ATR, labels, leak screen)  — test_matches_repo_*
  2. no look-ahead: features and predictions for bar t ignore every bar after t — test_*causal*
  3. every exit rule on hand-built bars (stop, gap, TP, same-bar order, trail, time)
  4. end-to-end: the backtest on expected_probs.csv reproduces all 510 trades and Sharpe 1.58
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "repo_code"))
import replicate as R  # noqa: E402


@pytest.fixture(scope="module")
def gold():
    return R.load("XAUUSDs")


# ---------------------------------------------------------------- 1. same as the original code
def test_matches_repo_features_atr_labels(gold):
    from scripts import v5_xau_turning_ml as orig          # the project's original detector code
    from scripts.v5_zigzag_signal import _frame              # the live bot's feature/label builder
    d = gold.iloc[-8000:]
    F_new, y_new, ok_new = R.frame(d)
    F_old, y_old, ok_old = _frame(d)
    assert list(F_new.columns) == list(F_old.columns)
    pd.testing.assert_frame_equal(F_new, F_old)
    assert (y_new == y_old).all() and (ok_new == ok_old).all()
    pd.testing.assert_series_equal(R.atr(d), orig.atr(d))


def test_matches_repo_leak_screen():
    from scripts.v5_pooled_bottom_detector import intrabar_leak
    for s in ("XAUUSDs", "EURUSDs", "GBPUSDs", "USDJPYs"):
        d = R.load(s)
        assert R.intrabar_leak(d) == pytest.approx(intrabar_leak(d), abs=1e-12)
        assert R.intrabar_leak(d) <= R.LEAK_MAX


# ---------------------------------------------------------------- 2. no look-ahead
def test_features_are_causal(gold):
    d = gold.iloc[-3000:]
    cut = 2500
    full = R.features(d).iloc[:cut]
    trunc = R.features(d.iloc[:cut])
    pd.testing.assert_frame_equal(full, trunc)               # appending future bars changes nothing


def test_labels_need_the_future_so_training_is_purged(gold):
    """The label DOES depend on later bars — that is why the walk-forward truncates and purges."""
    d = gold.iloc[-3000:]
    _, y_full, _ = R.frame(d)
    changed = 0
    for cut in range(2400, 2600, 7):
        _, y_cut, _ = R.frame(d.iloc[:cut])
        changed += int((y_full[:cut] != y_cut).any())        # labels near the cut change with more data
    assert changed >= 5


def test_walk_forward_predictions_are_causal():
    """Probabilities for 2024-01-02..12, recomputed with ALL data after 2024-01-13 deleted, are identical."""
    cut = pd.Timestamp("2024-01-13")
    gold = R.load("XAUUSDs")
    aux = {s: R.load(s) for s in R.AUX}
    p = R.walk_forward(gold[gold.index < cut], {k: v[v.index < cut] for k, v in aux.items()})
    ref = pd.read_csv(PKG / "expected_probs.csv", index_col=0, parse_dates=True).iloc[:, 0]
    both = pd.concat([p.dropna(), ref], axis=1, join="inner")
    assert len(both) > 100
    np.testing.assert_allclose(both.iloc[:, 0], both.iloc[:, 1], atol=1e-9)


# ---------------------------------------------------------------- 3. exit rules on synthetic bars
def flat(n=200):
    """Flat market: open=close=100, high 100.5, low 99.5 -> true range 1, ATR exactly 1."""
    t = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0}, index=t)


def one(g, at=20, **kw):
    f = np.zeros(len(g), bool)
    f[at] = True
    tr, _ = R.backtest(g, f, **kw)
    assert len(tr) == 1
    return tr.iloc[0]


def test_initial_risk_and_trail_before_stop():
    g = flat()
    g.loc[g.index[25], "low"] = 96.0
    t = one(g)
    # entry 100.25, R = 3 x ATR = 3; first bar's high 100.5 lifts the stop to 97.5 -> exit there
    assert t.entry == 100.25 and t.risk == 3.0 and t.exit_reason == "stop"
    assert t.exit == 97.5 and t.R == pytest.approx((97.5 - 100.25) / 3, abs=1e-3)


def test_gap_through_stop_fills_at_open():
    g = flat()
    g.loc[g.index[25], ["open", "low"]] = [95.0, 94.0]
    t = one(g)
    assert t.exit == 95.0 and t.R == pytest.approx(-1.75, abs=1e-3)


def test_take_profit_at_4_3_R():
    g = flat()
    g.loc[g.index[25], "high"] = 114.0
    t = one(g)
    assert t.exit_reason == "tp" and t.exit == pytest.approx(100.25 + 4.3 * 3, abs=0.01)
    assert t.R == pytest.approx(4.3, abs=0.01)


def test_stop_checked_before_tp_on_the_same_bar():
    g = flat()
    g.loc[g.index[25], ["high", "low"]] = [114.0, 96.0]
    assert one(g).exit_reason == "stop"


def test_trailing_stop_follows_highest_high():
    g = flat(300)
    g.loc[g.index[22:30], ["open", "close"]] = 104.0         # market steps up, ATR stays ~1
    g.loc[g.index[22:30], "high"] = 104.5
    g.loc[g.index[22:30], "low"] = 103.5
    g.loc[g.index[30], ["open", "close", "high", "low"]] = [104.0, 99.0, 104.0, 99.0]
    t = one(g)
    a = R.atr(g).to_numpy()
    expected_stop = max(104.5 - 3 * a[29], 100.25 - 3)       # set on bar 29, used on bar 30
    assert t.exit_reason == "stop" and t.exit == pytest.approx(expected_stop, abs=0.01)  # csv rounds to cents
    assert t.R > 0                                           # a "stop" exit that made money


def test_time_exit_after_120_bars():
    t = one(flat(300))
    assert t.exit_reason == "time" and t.bars == 120 and t.exit == 100.0


def test_no_entry_while_a_position_is_open():
    g = flat(300)
    f = np.zeros(len(g), bool)
    f[[20, 30, 40]] = True                                   # 30 and 40 fire while the first is held
    tr, _ = R.backtest(g, f)
    assert len(tr) == 1


# ---------------------------------------------------------------- 4. end to end
def test_reproduces_expected_trades_and_sharpe(gold):
    p = pd.read_csv(PKG / "expected_probs.csv", index_col=0, parse_dates=True).iloc[:, 0]
    pr = p.reindex(gold.index).to_numpy()
    tr, ret = R.backtest(gold, np.isfinite(pr) & (pr >= R.THR))
    exp = pd.read_csv(PKG / "expected_trades.csv", parse_dates=["t_in", "t_out"])
    assert len(tr) == len(exp) == 510
    assert (tr.t_in.values == exp.t_in.values).all() and (tr.exit_reason.values == exp.exit_reason.values).all()
    np.testing.assert_allclose(tr.R.values, exp.R.values, atol=1e-9)
    assert round(R.sharpe(ret), 2) == 1.58
    assert round(len(tr) / ((gold.index[-1] - R.START).days / 7), 2) == 3.44
