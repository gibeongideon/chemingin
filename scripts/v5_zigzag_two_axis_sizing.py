#!/usr/bin/env python3
"""Offline evaluation of the pre-registered ZigZag two-axis size dial.

The candidate combines the existing probability LIN multiplier and downside-volatility
half-size dial.  It never changes Arm V's signals, exits, or a live configuration.

    python scripts/v5_zigzag_two_axis_sizing.py cme
    python scripts/v5_zigzag_two_axis_sizing.py switch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "v5_runs" / "zigzag_two_axis_sizing"
PROBABILITY_FLOOR = 0.60
PROBABILITY_CEILING = 0.80
MIN_MULTIPLIER = 0.25
MAX_MULTIPLIER = 2.00
HIGH_VOL_MULTIPLIER = 0.50
RISK = 0.0025
N_SHUFFLES = 2_000
SHUFFLE_SEED = 20261114


def probability_multiplier(probability: np.ndarray) -> np.ndarray:
    """Pre-registered LIN multiplier, before causal normalisation."""
    probability = np.asarray(probability, dtype=float)
    return 0.5 + np.clip(
        (probability - PROBABILITY_FLOOR) / (PROBABILITY_CEILING - PROBABILITY_FLOOR),
        0.0,
        1.0,
    )


def volatility_multiplier(high_vol: np.ndarray) -> np.ndarray:
    """The deployed Arm V downside-volatility dial."""
    return np.where(np.asarray(high_vol, dtype=bool), HIGH_VOL_MULTIPLIER, 1.0)


def causal_multiplier(
    probability: np.ndarray,
    high_vol: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw, prior-trade normaliser, and capped causal size multiplier."""
    raw = probability_multiplier(probability) * volatility_multiplier(high_vol)
    denominator = np.ones(len(raw), dtype=float)
    if len(raw) > 1:
        denominator[1:] = np.cumsum(raw)[:-1] / np.arange(1, len(raw))
    final = np.clip(raw / denominator, MIN_MULTIPLIER, MAX_MULTIPLIER)
    return raw, denominator, final


def daily_returns(
    trades: pd.DataFrame,
    multiplier: np.ndarray,
    days: pd.DatetimeIndex,
    risk: float = RISK,
) -> pd.Series:
    """Aggregate exact, size-independent trade R outcomes on exit days."""
    multiplier = np.asarray(multiplier, dtype=float)
    if len(trades) != len(multiplier):
        raise ValueError("trade and multiplier lengths differ")
    ret = pd.Series(0.0, index=days)
    exit_days = pd.DatetimeIndex(trades["t_out"]).normalize()
    positions = days.get_indexer(exit_days)
    if (positions < 0).any():
        raise ValueError("trade exit lies outside the return index")
    np.add.at(ret.values, positions, risk * multiplier * trades["R"].to_numpy())
    return ret[ret.index.dayofweek < 5]


def sharpe(ret: pd.Series) -> float:
    std = ret.std()
    return float(ret.mean() / std * np.sqrt(252)) if len(ret) > 2 and std > 0 else 0.0


def max_drawdown(ret: pd.Series) -> float:
    equity = (1.0 + ret).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def cagr(ret: pd.Series) -> float:
    if len(ret) == 0:
        return 0.0
    equity = float((1.0 + ret).prod())
    years = len(ret) / 252.0
    return float(equity ** (1.0 / years) - 1.0) if years > 0 and equity > 0 else -1.0


def match_volatility(control: pd.Series, candidate: pd.Series) -> pd.Series:
    """Scale candidate daily returns to the control's realised volatility."""
    candidate_std = candidate.std(ddof=1)
    control_std = control.std(ddof=1)
    if candidate_std == 0 or not np.isfinite(candidate_std) or not np.isfinite(control_std):
        return candidate.copy()
    return candidate * control_std / candidate_std


def matched_t(control: pd.Series, candidate: pd.Series) -> float:
    """Paired t-statistic after matching candidate daily volatility to the control."""
    aligned = pd.concat([control.rename("control"), candidate.rename("candidate")], axis=1).dropna()
    if len(aligned) < 3:
        return 0.0
    scaled_candidate = match_volatility(aligned["control"], aligned["candidate"])
    difference = scaled_candidate - aligned["control"]
    std = difference.std(ddof=1)
    return float(difference.mean() / (std / np.sqrt(len(difference)))) if std > 0 else 0.0


def shuffle_sharpes(
    trades: pd.DataFrame,
    multiplier: np.ndarray,
    days: pd.DatetimeIndex,
    draws: int = N_SHUFFLES,
    seed: int = SHUFFLE_SEED,
) -> np.ndarray:
    """Conditional null: assign the complete candidate size vector to trades at random."""
    rng = np.random.default_rng(seed)
    return np.array([
        sharpe(daily_returns(trades, rng.permutation(multiplier), days))
        for _ in range(draws)
    ])


def accepted_trades(sim, fire: np.ndarray, probability: np.ndarray, spread: float) -> pd.DataFrame:
    """Reconstruct Arm V's one-position-at-a-time accepted entries exactly once."""
    rows: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, float, float]] = []
    busy_until = -1
    for signal_bar in np.flatnonzero(fire & sim.ok):
        if signal_bar >= sim.n - 1 or signal_bar < busy_until:
            continue
        entry_bar, exit_bar, r_multiple = sim.trade(int(signal_bar), spread)
        busy_until = exit_bar
        rows.append((
            sim.t[signal_bar], sim.t[entry_bar], sim.t[exit_bar],
            float(probability[signal_bar]), float(r_multiple),
        ))
    return pd.DataFrame(rows, columns=["t_signal", "t_in", "t_out", "p", "R"])


def source_trades(source: str, spread: float) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Load cached probabilities and reconstruct the Arm V control event set."""
    from scripts.v5_zigzag_downtrend_guards import guards
    from scripts.v5_zigzag_fall_gate import circuit, daily
    from scripts.v5_zigzag_frequency import Sim
    from scripts.v5_zigzag_long_history import OUT as LONG_HISTORY_OUT
    from scripts.v5_zigzag_long_history import SRC, THR, load

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
    sim = Sim(gold, pd.Timestamp(cfg["start"]), cfg["swap"])
    trades = accepted_trades(sim, fire, probability, spread)
    if trades.empty or trades["p"].isna().any():
        raise ValueError(f"{source}: missing accepted trades or entry probabilities")

    # The H1 signal is decided at its close and entered at the next bar's open.
    # Use that signal bar for the volatility condition; using t_in would read one
    # future H1 close and introduce look-ahead.
    signal_positions = gold.index.get_indexer(pd.DatetimeIndex(trades["t_signal"]))
    if (signal_positions < 0).any():
        raise ValueError(f"{source}: accepted signal is absent from the price index")
    calm = guards(gold)["downside vol below median"][signal_positions]
    trades["high_downside_vol"] = ~np.asarray(calm, dtype=bool)
    days = pd.date_range(
        pd.Timestamp(cfg["start"]).normalize(), gold.index[-1].normalize(), freq="D",
    )
    return trades, days


def assert_fixed_event_set(control: pd.DataFrame, candidate: pd.DataFrame) -> None:
    columns = ["t_in", "t_out", "R"]
    if not control[columns].equals(candidate[columns]):
        raise AssertionError("candidate changed the Arm V accepted entries or trade outcomes")


def candidate_factors(trades: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probability = trades["p"].to_numpy()
    high_vol = trades["high_downside_vol"].to_numpy()
    raw, normalizer, final = causal_multiplier(probability, high_vol)
    return probability_multiplier(probability), volatility_multiplier(high_vol), raw, normalizer, final


def evaluate_source(source: str) -> dict[str, float | int | bool]:
    """Run the one pre-registered source experiment and write auditable artifacts."""
    from scripts.v5_zigzag_long_history import SPREAD

    trades, days = source_trades(source, SPREAD)
    candidate = trades.copy()
    assert_fixed_event_set(trades, candidate)
    m_probability, m_volatility, raw, normalizer, m_final = candidate_factors(candidate)
    control_multiplier = volatility_multiplier(candidate["high_downside_vol"].to_numpy())
    control = daily_returns(trades, control_multiplier, days)
    candidate_returns = daily_returns(candidate, m_final, days)
    candidate_matched = match_volatility(control, candidate_returns)
    shuffled = shuffle_sharpes(candidate, m_final, days)

    midpoint = len(control) // 2
    h1_control, h2_control = control.iloc[:midpoint], control.iloc[midpoint:]
    h1_candidate, h2_candidate = candidate_returns.iloc[:midpoint], candidate_returns.iloc[midpoint:]

    twice_cost_trades, twice_cost_days = source_trades(source, 2.0 * SPREAD)
    twice_cost_candidate = twice_cost_trades.copy()
    assert_fixed_event_set(twice_cost_trades, twice_cost_candidate)
    _, _, _, _, twice_cost_multiplier = candidate_factors(twice_cost_candidate)
    control_2x = daily_returns(
        twice_cost_trades,
        volatility_multiplier(twice_cost_trades["high_downside_vol"].to_numpy()),
        twice_cost_days,
    )
    candidate_2x = daily_returns(twice_cost_candidate, twice_cost_multiplier, twice_cost_days)

    metrics: dict[str, float | int | bool] = {
        "trades": int(len(candidate)),
        "annual_trades": float(len(candidate) / (len(control) / 252.0)),
        "mean_multiplier": float(m_final.mean()),
        "control_sharpe": sharpe(control),
        "candidate_sharpe": sharpe(candidate_returns),
        "control_cagr": cagr(control),
        "candidate_cagr_matched_risk": cagr(candidate_matched),
        "control_maxdd": max_drawdown(control),
        "candidate_maxdd_matched_risk": max_drawdown(candidate_matched),
        "matched_t": matched_t(control, candidate_returns),
        "half_1_control_sharpe": sharpe(h1_control),
        "half_1_candidate_sharpe": sharpe(h1_candidate),
        "half_1_dsharpe": sharpe(h1_candidate) - sharpe(h1_control),
        "half_2_control_sharpe": sharpe(h2_control),
        "half_2_candidate_sharpe": sharpe(h2_candidate),
        "half_2_dsharpe": sharpe(h2_candidate) - sharpe(h2_control),
        "control_sharpe_2x_cost": sharpe(control_2x),
        "candidate_sharpe_2x_cost": sharpe(candidate_2x),
        "shuffle_p95": float(np.percentile(shuffled, 95)),
        "shuffle_pvalue": float((shuffled >= sharpe(candidate_returns)).mean()),
    }
    metrics["promotion_pass"] = bool(
        metrics["matched_t"] >= 2.0
        and metrics["candidate_sharpe"] > metrics["shuffle_p95"]
        and metrics["half_1_dsharpe"] > 0.0
        and metrics["half_2_dsharpe"] > 0.0
        and metrics["candidate_cagr_matched_risk"] >= metrics["control_cagr"]
        and metrics["candidate_maxdd_matched_risk"] >= metrics["control_maxdd"]
    )

    candidate["m_probability"] = m_probability
    candidate["m_volatility"] = m_volatility
    candidate["m_raw"] = raw
    candidate["normalizer"] = normalizer
    candidate["m_final"] = m_final
    source_out = OUT / source
    source_out.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(source_out / "trades.csv", index=False)
    (source_out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    (source_out / "report.md").write_text(source_report(source, metrics) + "\n")
    write_combined_report()
    return metrics


def source_report(source: str, metrics: dict[str, float | int | bool]) -> str:
    return "\n".join([
        f"# ZigZag probability × downside-vol sizing — {source.upper()}",
        "",
        "Control: Arm V with its existing daily circuit breaker and downside-vol dial.",
        "Candidate: fixed-event causal probability × downside-vol multiplier, capped at 0.25–2.00.",
        "",
        "| Metric | Control | Candidate |",
        "| --- | ---: | ---: |",
        f"| Sharpe | {metrics['control_sharpe']:+.3f} | {metrics['candidate_sharpe']:+.3f} |",
        f"| CAGR at matched risk | {metrics['control_cagr']:+.2%} | {metrics['candidate_cagr_matched_risk']:+.2%} |",
        f"| Max drawdown at matched risk | {metrics['control_maxdd']:+.2%} | {metrics['candidate_maxdd_matched_risk']:+.2%} |",
        f"| 2x-cost Sharpe | {metrics['control_sharpe_2x_cost']:+.3f} | {metrics['candidate_sharpe_2x_cost']:+.3f} |",
        "",
        f"Trades: {metrics['trades']} ({metrics['annual_trades']:.1f}/year); mean candidate multiplier: {metrics['mean_multiplier']:.3f}.",
        f"Matched-volatility paired t: {metrics['matched_t']:+.2f}. Conditional-shuffle p95: {metrics['shuffle_p95']:+.3f}; p-value: {metrics['shuffle_pvalue']:.3f}.",
        f"Chronological-half Sharpe changes: {metrics['half_1_dsharpe']:+.3f}, {metrics['half_2_dsharpe']:+.3f}.",
        "",
        "Promotion requires t ≥ +2.0, Sharpe above shuffle p95, both half changes positive, CAGR no lower, and maximum drawdown no worse.",
        f"PROMOTION: {'PASS' if metrics['promotion_pass'] else 'FAIL — Arm V retained'}.",
        "Fixed-event assertion passed: entry times, exit times, and R outcomes are shared by control and candidate.",
        "OFFLINE RESEARCH — no live configuration modified.",
    ])


def ledger_entry(all_metrics: dict[str, dict[str, float | int | bool]]) -> str:
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
        f"shuffle p {cme['shuffle_pvalue']:.3f}; 2x-cost Sharpe {cme['candidate_sharpe_2x_cost']:+.3f}; "
        f"half changes {cme['half_1_dsharpe']:+.3f} / {cme['half_2_dsharpe']:+.3f}; "
        f"CAGR {cme['control_cagr']:+.2%} → {cme['candidate_cagr_matched_risk']:+.2%}; "
        f"maxDD {cme['control_maxdd']:+.2%} → {cme['candidate_maxdd_matched_risk']:+.2%}.",
        f"Switch: Sharpe {switch['candidate_sharpe']:+.3f} vs {switch['control_sharpe']:+.3f}; "
        f"matched t {switch['matched_t']:+.2f}; shuffle p95 {switch['shuffle_p95']:+.3f}; "
        f"shuffle p {switch['shuffle_pvalue']:.3f}; 2x-cost Sharpe {switch['candidate_sharpe_2x_cost']:+.3f}; "
        f"half changes {switch['half_1_dsharpe']:+.3f} / {switch['half_2_dsharpe']:+.3f}; "
        f"CAGR {switch['control_cagr']:+.2%} → {switch['candidate_cagr_matched_risk']:+.2%}; "
        f"maxDD {switch['control_maxdd']:+.2%} → {switch['candidate_maxdd_matched_risk']:+.2%}.",
        "",
        f"Verdict: {verdict}. No live configuration was modified.",
    ])


def write_combined_report() -> None:
    paths = {source: OUT / source / "metrics.json" for source in ("cme", "switch")}
    if not all(path.exists() for path in paths.values()):
        return
    metrics = {source: json.loads(path.read_text()) for source, path in paths.items()}
    (OUT / "REPORT.md").write_text(
        "# ZigZag two-axis sizing study\n\n## Ledger draft\n\n" + ledger_entry(metrics) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=("cme", "switch"))
    args = parser.parse_args()
    print("OFFLINE RESEARCH — no live configuration modified")
    metrics = evaluate_source(args.source)
    print(source_report(args.source, metrics))


if __name__ == "__main__":
    main()
