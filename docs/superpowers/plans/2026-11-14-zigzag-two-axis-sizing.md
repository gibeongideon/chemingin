# ZigZag Two-Axis Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate one pre-registered, offline-only ZigZag size multiplier that combines model probability and downside volatility without changing the Arm V event set or exit path.

**Architecture:** Add one research script that consumes the cached, walk-forward ZigZag probabilities and existing long-history simulators. It will reconstruct the Arm V accepted trades once, calculate causal trade-size multipliers, aggregate fixed-event daily returns, and report the result against Arm V and a conditional size-shuffle null. Pure multiplier logic lives in importable functions and is covered by focused pytest tests.

**Tech Stack:** Python 3, NumPy, pandas, SciPy, pytest; existing `scripts/v5_zigzag_long_history.py`, `scripts/v5_zigzag_frequency.py`, `scripts/v5_zigzag_fall_gate.py`, and `scripts/v5_zigzag_downtrend_guards.py`.

## Global Constraints

- The study is offline only: no MT5 bridge, execution, config, timer, or relay-state access.
- The control is Arm V: long-only H1 ZigZag, 3.0 ATR trail, 4.3R target, 120-bar maximum hold, one position at a time, and the 1.0 daily-ATR circuit breaker.
- CME and Switch use their existing cached walk-forward probabilities, single-source history, recorded costs, and source-specific swap assumptions.
- The only new candidate is `m_probability × m_volatility` with the exact formula in the approved design; do not tune or sweep any values.
- Every comparison must use the same accepted entry and exit sequence as Arm V. The result must assert this before scoring.
- Use 2,000 deterministic conditional shuffles, seed `20261114`, and report full sample, chronological halves, and two-times-cost results.
- A result is a research report only. Do not modify a live JSON config even if all pass criteria are met.

## File Structure

- Create: `scripts/v5_zigzag_two_axis_sizing.py` — pure multiplier functions, Arm V trade reconstruction, daily return aggregation, randomisation control, metrics, report, and CLI.
- Create: `tests/test_v5_zigzag_two_axis_sizing.py` — formula, cap, causality, and trade-size invariants for the new script.
- Create at runtime: `data/v5_runs/zigzag_two_axis_sizing/{cme,switch}/trades.csv` — auditable per-trade factors and returns.
- Create at runtime: `data/v5_runs/zigzag_two_axis_sizing/{cme,switch}/metrics.json` — machine-readable full, half, cost, and shuffle results.
- Create at runtime: `data/v5_runs/zigzag_two_axis_sizing/{cme,switch}/report.md` — human-readable source-level outcome and decision metrics.
- Create after both source runs: `data/v5_runs/zigzag_two_axis_sizing/REPORT.md` — combined promotion decision and the single ledger entry generated from both source metrics.
- Modify after the outcome is reviewed: `V5_FINDINGS.md` — append the report-generated ledger entry verbatim, including a pass or closure verdict.

---

### Task 1: Lock down causal multiplier arithmetic

**Files:**

- Create: `tests/test_v5_zigzag_two_axis_sizing.py`
- Create: `scripts/v5_zigzag_two_axis_sizing.py`

**Interfaces:**

- Consumes: NumPy arrays of probability and high-downside-volatility flags.
- Produces: `probability_multiplier(probability: np.ndarray) -> np.ndarray`, `volatility_multiplier(high_vol: np.ndarray) -> np.ndarray`, and `causal_multiplier(probability: np.ndarray, high_vol: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]`.
- Guarantees: all output multipliers lie in `[0.25, 2.00]`; multiplier `i` uses only raw multipliers from accepted trades `< i`.

- [ ] **Step 1: Write the failing unit tests**

Create `tests/test_v5_zigzag_two_axis_sizing.py`:

```python
from __future__ import annotations

import numpy as np

from scripts.v5_zigzag_two_axis_sizing import (
    causal_multiplier,
    probability_multiplier,
    volatility_multiplier,
)


def test_probability_multiplier_uses_the_pre_registered_linear_knots():
    p = np.array([0.40, 0.60, 0.65, 0.70, 0.80, 0.95])
    np.testing.assert_allclose(
        probability_multiplier(p),
        np.array([0.50, 0.50, 0.75, 1.00, 1.50, 1.50]),
    )


def test_volatility_multiplier_only_halves_high_downside_volatility_entries():
    high_vol = np.array([False, True, False, True])
    np.testing.assert_allclose(volatility_multiplier(high_vol), [1.0, 0.5, 1.0, 0.5])


def test_causal_multiplier_uses_prior_entries_and_respects_bounds():
    p = np.array([0.70, 0.60, 0.80])
    high_vol = np.array([False, False, True])
    raw, denominator, final = causal_multiplier(p, high_vol)

    np.testing.assert_allclose(raw, [1.0, 0.5, 0.75])
    np.testing.assert_allclose(denominator, [1.0, 1.0, 0.75])
    np.testing.assert_allclose(final, [1.0, 0.5, 1.0])
    assert np.all((final >= 0.25) & (final <= 2.0))


def test_appending_future_entries_cannot_change_prior_multipliers():
    p = np.array([0.60, 0.70, 0.80])
    high_vol = np.array([False, True, False])
    before = causal_multiplier(p, high_vol)[2]
    after = causal_multiplier(
        np.append(p, [0.60, 0.95]),
        np.append(high_vol, [True, False]),
    )[2]
    np.testing.assert_allclose(after[: len(before)], before)
```

- [ ] **Step 2: Run the tests to verify the module does not exist yet**

Run:

```bash
pytest -q tests/test_v5_zigzag_two_axis_sizing.py
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'scripts.v5_zigzag_two_axis_sizing'`.

- [ ] **Step 3: Implement the pure functions with the fixed formula**

Create `scripts/v5_zigzag_two_axis_sizing.py` with these constants and functions before adding the backtest runner:

```python
from __future__ import annotations

import numpy as np

PROBABILITY_FLOOR = 0.60
PROBABILITY_CEILING = 0.80
MIN_MULTIPLIER = 0.25
MAX_MULTIPLIER = 2.00
HIGH_VOL_MULTIPLIER = 0.50


def probability_multiplier(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    return 0.5 + np.clip(
        (probability - PROBABILITY_FLOOR)
        / (PROBABILITY_CEILING - PROBABILITY_FLOOR),
        0.0,
        1.0,
    )


def volatility_multiplier(high_vol: np.ndarray) -> np.ndarray:
    high_vol = np.asarray(high_vol, dtype=bool)
    return np.where(high_vol, HIGH_VOL_MULTIPLIER, 1.0)


def causal_multiplier(
    probability: np.ndarray,
    high_vol: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = probability_multiplier(probability) * volatility_multiplier(high_vol)
    denominator = np.ones(len(raw), dtype=float)
    if len(raw) > 1:
        denominator[1:] = np.cumsum(raw)[:-1] / np.arange(1, len(raw))
    final = np.clip(raw / denominator, MIN_MULTIPLIER, MAX_MULTIPLIER)
    return raw, denominator, final
```

- [ ] **Step 4: Run the focused tests to verify the formula and causality**

Run:

```bash
pytest -q tests/test_v5_zigzag_two_axis_sizing.py
```

Expected: PASS with four tests.

- [ ] **Step 5: Commit the pure sizing implementation and tests**

```bash
git add scripts/v5_zigzag_two_axis_sizing.py tests/test_v5_zigzag_two_axis_sizing.py
git commit -m "Add causal ZigZag two-axis sizing formula"
```

### Task 2: Reconstruct Arm V once and produce comparable candidate returns

**Files:**

- Modify: `scripts/v5_zigzag_two_axis_sizing.py`
- Modify: `tests/test_v5_zigzag_two_axis_sizing.py`

**Interfaces:**

- Consumes: `SRC`, `THR`, `RISK`, `SPREAD`, `load`, and `sharpe` from `scripts.v5_zigzag_long_history`; `Sim` from `scripts.v5_zigzag_frequency`; circuit-breaker logic from `scripts.v5_zigzag_fall_gate`; and the downside-vol flag from `scripts.v5_zigzag_downtrend_guards.guards`.
- Produces: `accepted_trades(...) -> pd.DataFrame` with columns `t_in`, `t_out`, `p`, and `R`; and `daily_returns(...) -> pd.Series` indexed by weekday business dates.
- Guarantees: the candidate has exactly the same `t_in`, `t_out`, and `R` values as the Arm V control; only `risk_multiplier` differs.

- [ ] **Step 1: Add a failing fixed-event-set test**

Append to `tests/test_v5_zigzag_two_axis_sizing.py`:

```python
import pandas as pd

from scripts.v5_zigzag_two_axis_sizing import daily_returns


def test_daily_returns_keep_the_same_trade_outcomes_when_only_size_changes():
    trades = pd.DataFrame(
        {
            "t_in": pd.to_datetime(["2026-01-05 01:00", "2026-01-06 01:00"]),
            "t_out": pd.to_datetime(["2026-01-05 15:00", "2026-01-06 15:00"]),
            "R": [1.0, -0.5],
        }
    )
    days = pd.date_range("2026-01-05", "2026-01-09", freq="D")
    flat = daily_returns(trades, np.ones(2), days, risk=0.0025)
    sized = daily_returns(trades, np.array([2.0, 0.5]), days, risk=0.0025)

    assert flat.loc["2026-01-05"] == 0.0025
    assert flat.loc["2026-01-06"] == -0.00125
    assert sized.loc["2026-01-05"] == 0.005
    assert sized.loc["2026-01-06"] == -0.000625
    assert list(flat.index) == list(sized.index)
```

- [ ] **Step 2: Run the test to verify the return aggregator is absent**

Run:

```bash
pytest -q tests/test_v5_zigzag_two_axis_sizing.py::test_daily_returns_keep_the_same_trade_outcomes_when_only_size_changes
```

Expected: FAIL with `ImportError` for `daily_returns`.

- [ ] **Step 3: Implement trade reconstruction and daily aggregation**

Add these imports and functions to `scripts/v5_zigzag_two_axis_sizing.py`:

```python
from pathlib import Path

import pandas as pd

from scripts.v5_zigzag_downtrend_guards import guards
from scripts.v5_zigzag_fall_gate import circuit, daily
from scripts.v5_zigzag_frequency import Sim
from scripts.v5_zigzag_long_history import OUT as LONG_HISTORY_OUT
from scripts.v5_zigzag_long_history import RISK, SPREAD, SRC, THR, load, sharpe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v5_runs" / "zigzag_two_axis_sizing"


def accepted_trades(sim: Sim, fire: np.ndarray, probability: np.ndarray, spread: float) -> pd.DataFrame:
    rows: list[tuple[pd.Timestamp, pd.Timestamp, float, float]] = []
    busy_until = -1
    for entry_bar in np.flatnonzero(fire & sim.ok):
        if entry_bar >= sim.n - 1 or entry_bar < busy_until:
            continue
        _, exit_bar, r_multiple = sim.trade(int(entry_bar), spread)
        busy_until = exit_bar
        rows.append((sim.t[entry_bar + 1], sim.t[exit_bar], float(probability[entry_bar]), float(r_multiple)))
    return pd.DataFrame(rows, columns=["t_in", "t_out", "p", "R"])


def daily_returns(
    trades: pd.DataFrame,
    multiplier: np.ndarray,
    days: pd.DatetimeIndex,
    risk: float = RISK,
) -> pd.Series:
    ret = pd.Series(0.0, index=days)
    exit_days = pd.DatetimeIndex(trades["t_out"]).normalize()
    np.add.at(ret.values, days.get_indexer(exit_days), risk * multiplier * trades["R"].to_numpy())
    return ret[ret.index.dayofweek < 5]


def source_trades(source: str, spread: float) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    cfg = SRC[source]
    gold = load(source, cfg["gold"])
    probability = pd.read_csv(
        LONG_HISTORY_OUT / source / "probs.csv", index_col=0, parse_dates=True,
    ).iloc[:, 0].reindex(gold.index).to_numpy()
    fire = (
        np.isfinite(probability)
        & (probability >= THR)
        & ~circuit(gold, daily(gold), 1.0, "day")
    )
    trades = accepted_trades(Sim(gold, pd.Timestamp(cfg["start"]), cfg["swap"]), fire, probability, spread)
    trade_entry_times = pd.DatetimeIndex(trades["t_in"])
    calm = guards(gold)["downside vol below median"].reindex(trade_entry_times).to_numpy(dtype=bool)
    if len(trades) == 0 or np.isnan(trades["p"]).any() or pd.isna(calm).any():
        raise ValueError(f"{source}: missing an accepted-trade probability or volatility flag")
    trades["high_downside_vol"] = ~calm
    return trades, pd.date_range(pd.Timestamp(cfg["start"]).normalize(), gold.index[-1].normalize(), freq="D")
```

In `main`, reconstruct the Arm V trade table once per source. Compute `raw`, `denominator`, and
`candidate_multiplier` with `causal_multiplier(trades.p.to_numpy(), trades.high_downside_vol.to_numpy())`.
Use `np.where(trades.high_downside_vol, 0.5, 1.0)` for Arm V's current vol-taper multiplier.
Assert that the `t_in`, `t_out`, and `R` columns are unchanged before aggregating either return series.

- [ ] **Step 4: Run all focused tests**

Run:

```bash
pytest -q tests/test_v5_zigzag_two_axis_sizing.py
```

Expected: PASS with five tests.

- [ ] **Step 5: Commit fixed-event reconstruction**

```bash
git add scripts/v5_zigzag_two_axis_sizing.py tests/test_v5_zigzag_two_axis_sizing.py
git commit -m "Reconstruct ZigZag two-axis sizing trades"
```

### Task 3: Add statistical gates, artifacts, and a read-only CLI

**Files:**

- Modify: `scripts/v5_zigzag_two_axis_sizing.py`
- Modify: `tests/test_v5_zigzag_two_axis_sizing.py`

**Interfaces:**

- Consumes: fixed-event control and candidate daily return series from Task 2.
- Produces: `matched_t(control: pd.Series, candidate: pd.Series) -> float`, `shuffle_sharpes(...) -> np.ndarray`, and source report artifacts.
- Produces: `metrics.json` fields for full and half-sample Sharpe, CAGR, max drawdown, matched-volatility t-statistic, shuffle 95th percentile, shuffle p-value, two-times-cost Sharpe, and `promotion_pass`.

- [ ] **Step 1: Add failing tests for matched-volatility comparison and deterministic shuffle**

Append to `tests/test_v5_zigzag_two_axis_sizing.py`:

```python
from scripts.v5_zigzag_two_axis_sizing import matched_t, shuffle_sharpes


def test_matched_t_is_zero_for_identical_return_streams():
    r = pd.Series([0.01, -0.01, 0.02, -0.01])
    assert matched_t(r, r) == 0.0


def test_shuffle_sharpes_is_repeatable_from_its_seed():
    trades = pd.DataFrame(
        {
            "t_in": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
            "t_out": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
            "R": [1.0, -0.5, 0.25],
        }
    )
    days = pd.date_range("2026-01-05", "2026-01-09", freq="D")
    m = np.array([0.5, 1.0, 1.5])
    first = shuffle_sharpes(trades, m, days, draws=20, seed=20261114)
    second = shuffle_sharpes(trades, m, days, draws=20, seed=20261114)
    np.testing.assert_allclose(first, second)
    assert len(first) == 20
```

- [ ] **Step 2: Run the new tests to verify the statistics functions are absent**

Run:

```bash
pytest -q tests/test_v5_zigzag_two_axis_sizing.py -k 'matched_t or shuffle_sharpes'
```

Expected: FAIL during collection with `ImportError` for `matched_t` and `shuffle_sharpes`.

- [ ] **Step 3: Implement scoring, promotion gates, and reports**

Add the following functions to `scripts/v5_zigzag_two_axis_sizing.py`:

```python
import json

N_SHUFFLES = 2_000
SHUFFLE_SEED = 20261114


def max_drawdown(ret: pd.Series) -> float:
    equity = (1.0 + ret).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def cagr(ret: pd.Series) -> float:
    equity = (1.0 + ret).cumprod()
    years = len(ret) / 252.0
    return float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0


def matched_t(control: pd.Series, candidate: pd.Series) -> float:
    aligned = pd.concat([control.rename("control"), candidate.rename("candidate")], axis=1).dropna()
    if len(aligned) < 3 or aligned["candidate"].std(ddof=1) == 0:
        return 0.0
    scaled_candidate = aligned["candidate"] * aligned["control"].std(ddof=1) / aligned["candidate"].std(ddof=1)
    difference = scaled_candidate - aligned["control"]
    if difference.std(ddof=1) == 0:
        return 0.0
    return float(difference.mean() / (difference.std(ddof=1) / np.sqrt(len(difference))))


def shuffle_sharpes(
    trades: pd.DataFrame,
    multiplier: np.ndarray,
    days: pd.DatetimeIndex,
    draws: int = N_SHUFFLES,
    seed: int = SHUFFLE_SEED,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([
        sharpe(daily_returns(trades, rng.permutation(multiplier), days))
        for _ in range(draws)
    ])
```

For each source, calculate control and candidate returns at recorded costs and at twice recorded
costs. Split on the chronological midpoint of the daily index. The source passes only when all
of these booleans are true:

```python
source_pass = (
    metrics["matched_t"] >= 2.0
    and metrics["candidate_sharpe"] > metrics["shuffle_p95"]
    and metrics["half_1_dsharpe"] > 0.0
    and metrics["half_2_dsharpe"] > 0.0
    and metrics["candidate_cagr_matched_risk"] >= metrics["control_cagr"]
    and metrics["candidate_maxdd_matched_risk"] >= metrics["control_maxdd"]
)
```

Write a source `trades.csv` with `t_in`, `t_out`, `p`, `R`, `high_downside_vol`,
`m_probability`, `m_volatility`, `m_raw`, `normalizer`, and `m_final`. Write `metrics.json`
using plain Python `float` and `bool` values. Write `report.md` with the full decision table,
the exact pass conditions, and a ledger entry generated only from the metrics dictionary:

```python
def ledger_entry(all_metrics: dict[str, dict[str, float | bool]]) -> str:
    cme, switch = all_metrics["cme"], all_metrics["switch"]
    verdict = "PASS" if cme["promotion_pass"] and switch["promotion_pass"] else "CLOSED"
    return "\n".join([
        "### §3ev — ZigZag probability × downside-vol sizing (2026-11-14)",
        "",
        "Control: Arm V with the existing daily circuit breaker and downside-vol dial.",
        "Candidate: fixed-event causal probability × downside-vol multiplier, capped at 0.25–2.00.",
        "",
        f"CME: Sharpe {cme['candidate_sharpe']:+.3f} vs {cme['control_sharpe']:+.3f}; "
        f"matched t {cme['matched_t']:+.2f}; shuffle p95 {cme['shuffle_p95']:+.3f}; "
        f"2x-cost Sharpe {cme['candidate_sharpe_2x_cost']:+.3f}.",
        f"Switch: Sharpe {switch['candidate_sharpe']:+.3f} vs {switch['control_sharpe']:+.3f}; "
        f"matched t {switch['matched_t']:+.2f}; shuffle p95 {switch['shuffle_p95']:+.3f}; "
        f"2x-cost Sharpe {switch['candidate_sharpe_2x_cost']:+.3f}.",
        "",
        f"Verdict: {verdict}. No live configuration was modified.",
    ])
```

After each source run writes its own `metrics.json`, call `write_combined_report()` from `main`.
That function returns without writing when either source metrics file is absent. Once both files
exist, it loads them, calls `ledger_entry({"cme": cme_metrics, "switch": switch_metrics})`, and
writes the result under a `## Ledger draft` heading in
`data/v5_runs/zigzag_two_axis_sizing/REPORT.md`.
```

The brackets above are generated by the script from the metrics dictionary; they must never be
hand-filled or selected from an alternate result.

Add an `argparse` source argument constrained to `cme` or `switch`, then run only the selected
source. The script must print `OFFLINE RESEARCH — no live configuration modified` before loading data.

- [ ] **Step 4: Run the unit tests**

Run:

```bash
pytest -q tests/test_v5_zigzag_two_axis_sizing.py
```

Expected: PASS with seven tests.

- [ ] **Step 5: Run the pre-registered research study**

Run:

```bash
python scripts/v5_zigzag_two_axis_sizing.py cme
python scripts/v5_zigzag_two_axis_sizing.py switch
```

Expected: each command creates its source directory under
`data/v5_runs/zigzag_two_axis_sizing/`, prints the fixed-event assertion, all metrics, shuffle
result, and `PROMOTION: PASS` or `PROMOTION: FAIL — Arm V retained`. Neither command connects to MT5.

- [ ] **Step 6: Inspect artifacts and commit the study implementation**

Run:

```bash
git diff --check
pytest -q tests/test_v5_zigzag_two_axis_sizing.py
git add scripts/v5_zigzag_two_axis_sizing.py tests/test_v5_zigzag_two_axis_sizing.py data/v5_runs/zigzag_two_axis_sizing
git commit -m "Evaluate ZigZag two-axis sizing"
```

Expected: the committed results include both source reports and no live executor, deployment, or configuration file.

### Task 4: Record the unambiguous result in the research ledger

**Files:**

- Modify: `V5_FINDINGS.md`
- Read: `data/v5_runs/zigzag_two_axis_sizing/REPORT.md`

**Interfaces:**

- Consumes: the report-generated ledger draft from Task 3.
- Produces: one permanent factual `§3ev` result entry.

- [ ] **Step 1: Copy the generated ledger entry without changing figures or verdict**

Append the `§3ev` section emitted by the script to `V5_FINDINGS.md`. Preserve the reported CME
and Switch values, cost sensitivity, half-sample values, conditional-shuffle percentiles, and the
script-selected `PASS` or `CLOSED` verdict. Add one sentence stating that no live configuration
changed during this study.

- [ ] **Step 2: Verify the ledger matches both source reports**

Run:

```bash
rg -n -A 18 '§3ev' V5_FINDINGS.md
cat data/v5_runs/zigzag_two_axis_sizing/REPORT.md
```

Expected: every quoted figure and the verdict agree exactly with the generated combined report.

- [ ] **Step 3: Commit the ledger result**

```bash
git add V5_FINDINGS.md
git commit -m "Record ZigZag two-axis sizing result"
```

## Plan Self-Review

- Spec coverage: Task 1 implements the exact fixed multiplier formula and causal normalisation. Task 2 preserves Arm V’s event set and cost model. Task 3 supplies both required-source metrics, randomized control, split halves, cost stress, pass gate, and reviewable artifacts. Task 4 makes the outcome permanent without rewriting it.
- Placeholder scan: no implementation parameter, path, formula, command, or acceptance criterion is deferred; the ledger figures are intentionally generated after the pre-registered calculation completes.
- Type consistency: multiplier arrays align one-for-one with accepted-trade rows; daily returns are indexed by weekdays for every metric, shuffle, and paired comparison.
