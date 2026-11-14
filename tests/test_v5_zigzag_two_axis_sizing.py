from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.v5_zigzag_two_axis_sizing import (
    causal_multiplier,
    daily_returns,
    matched_t,
    probability_multiplier,
    shuffle_sharpes,
    volatility_multiplier,
    ledger_entry,
)


def test_probability_multiplier_uses_pre_registered_linear_knots():
    p = np.array([0.40, 0.60, 0.65, 0.70, 0.80, 0.95])
    np.testing.assert_allclose(probability_multiplier(p), [0.5, 0.5, 0.75, 1.0, 1.5, 1.5])


def test_volatility_multiplier_only_halves_high_downside_volatility_entries():
    np.testing.assert_allclose(volatility_multiplier([False, True, False, True]), [1, 0.5, 1, 0.5])


def test_causal_multiplier_uses_prior_entries_and_respects_bounds():
    raw, denominator, final = causal_multiplier(
        np.array([0.70, 0.60, 0.80]), np.array([False, False, True])
    )
    np.testing.assert_allclose(raw, [1.0, 0.5, 0.75])
    np.testing.assert_allclose(denominator, [1.0, 1.0, 0.75])
    np.testing.assert_allclose(final, [1.0, 0.5, 1.0])
    assert np.all((final >= 0.25) & (final <= 2.0))


def test_appending_future_entries_cannot_change_prior_multipliers():
    p, high = np.array([0.60, 0.70, 0.80]), np.array([False, True, False])
    before = causal_multiplier(p, high)[2]
    after = causal_multiplier(np.append(p, [0.60, 0.95]), np.append(high, [True, False]))[2]
    np.testing.assert_allclose(after[:len(before)], before)


def test_daily_returns_keep_trade_outcomes_when_only_size_changes():
    trades = pd.DataFrame({
        "t_in": pd.to_datetime(["2026-01-05 01:00", "2026-01-06 01:00"]),
        "t_out": pd.to_datetime(["2026-01-05 15:00", "2026-01-06 15:00"]),
        "R": [1.0, -0.5],
    })
    days = pd.date_range("2026-01-05", "2026-01-09", freq="D")
    flat = daily_returns(trades, np.ones(2), days, risk=0.0025)
    sized = daily_returns(trades, np.array([2.0, 0.5]), days, risk=0.0025)
    assert flat.loc["2026-01-05"] == 0.0025
    assert flat.loc["2026-01-06"] == -0.00125
    assert sized.loc["2026-01-05"] == 0.005
    assert sized.loc["2026-01-06"] == -0.000625
    assert list(flat.index) == list(sized.index)


def test_matched_t_is_zero_for_identical_streams():
    r = pd.Series([0.01, -0.01, 0.02, -0.01])
    assert matched_t(r, r) == 0.0


def test_shuffle_sharpes_is_repeatable_from_its_seed():
    trades = pd.DataFrame({
        "t_in": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
        "t_out": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
        "R": [1.0, -0.5, 0.25],
    })
    days = pd.date_range("2026-01-05", "2026-01-09", freq="D")
    m = np.array([0.5, 1.0, 1.5])
    first = shuffle_sharpes(trades, m, days, draws=20, seed=20261114)
    second = shuffle_sharpes(trades, m, days, draws=20, seed=20261114)
    np.testing.assert_allclose(first, second)
    assert len(first) == 20


def test_ledger_entry_preserves_split_randomisation_and_risk_metrics():
    metrics = {
        "candidate_sharpe": 0.8,
        "control_sharpe": 0.7,
        "matched_t": 0.9,
        "shuffle_p95": 0.75,
        "shuffle_pvalue": 0.02,
        "candidate_sharpe_2x_cost": 0.6,
        "half_1_dsharpe": 0.1,
        "half_2_dsharpe": 0.01,
        "control_cagr": 0.02,
        "candidate_cagr_matched_risk": 0.03,
        "control_maxdd": -0.04,
        "candidate_maxdd_matched_risk": -0.03,
        "promotion_pass": False,
    }
    text = ledger_entry({"cme": metrics, "switch": metrics})
    assert "shuffle p 0.020" in text
    assert "half changes +0.100 / +0.010" in text
    assert "CAGR +2.00% → +3.00%" in text
    assert "maxDD -4.00% → -3.00%" in text
