"""Walk-forward grid selection + regime split — the ICT/SMC concept-sweep's shared
harness (see `~/.claude/plans/i-wanttt-you-to-playful-widget.md`).

This is the exact pattern `scripts/v5_xau_regime_gate.py` established and
MANDATORY CONTROL #6 (`V5_FINDINGS.md`) now requires of every new signal:
re-select the best grid cell EVERY YEAR using only a trailing window of
strictly-past data, never a full-sample pick "validated" afterward. Extracted
here because the ICT sweep re-runs this exact loop for every concept —
copy-pasting it ten times would be the wrong kind of repetition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def walk_forward_select(
    grid_rets: dict, years: list[int], select_window_bars: int, min_train_bars: int = 100
) -> tuple[pd.Series, dict]:
    """`grid_rets`: {param_key: daily_return_series}, all sharing the same index.
    For each year, pick the cell with the best Sharpe over the trailing
    `select_window_bars` of data strictly before that year, apply it
    out-of-sample for that year only, concatenate across years.

    Returns (oos_series, selections) where `selections[year] = (best_key,
    trailing_train_sharpe)` — always print this, it's the fragility tell (an
    erratically-jumping selection across years is itself a red flag, as it was
    for the SPX/NDX divergence walk-forward)."""
    idx = next(iter(grid_rets.values())).index
    oos_parts = []
    selections: dict[int, tuple] = {}
    for yr in years:
        cutoff = pd.Timestamp(f"{yr}-01-01")
        train_end_pos = int(np.searchsorted(idx.values, np.datetime64(cutoff)))
        train_start_pos = max(0, train_end_pos - select_window_bars)
        if train_end_pos - train_start_pos < min_train_bars:
            continue
        train_sh = {}
        for k, ret in grid_rets.items():
            r = ret.iloc[train_start_pos:train_end_pos]
            train_sh[k] = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else -np.inf
        best_k = max(train_sh, key=train_sh.get)
        selections[yr] = (best_k, train_sh[best_k])
        test_mask = idx.year == yr
        if test_mask.sum() == 0:
            continue
        oos_parts.append(grid_rets[best_k][test_mask])
    if not oos_parts:
        return pd.Series(dtype=float), selections
    return pd.concat(oos_parts).sort_index(), selections


def regime_split_sharpe(daily: pd.Series, boundaries: list[tuple[str, str, str]]) -> dict[str, float]:
    """`boundaries`: list of (start_date, end_date, label) — e.g.
    `src.features.ict_primitives.REGIME_BOUNDARIES["XAUUSD"]`. Returns
    {"YYYY-YYYY label": sharpe} per segment; segments with <20 observations or
    zero variance are skipped (too short to mean anything)."""
    out = {}
    for start, end, label in boundaries:
        seg = daily.loc[start:end].dropna()
        if len(seg) > 20 and seg.std() > 0:
            key = f"{pd.Timestamp(start).year}-{pd.Timestamp(end).year} {label}"
            out[key] = float(seg.mean() / seg.std() * np.sqrt(252))
    return out


def print_selections(selections: dict) -> None:
    for yr, (key, train_sr) in selections.items():
        print(f"  {yr}: {key}  (trailing train SR={train_sr:+.2f})")


def print_regime_split(regimes: dict) -> None:
    for label, sr in regimes.items():
        print(f"  {label:24s} SR {sr:+.2f}")


def run_concept_pipeline(
    name: str, df: pd.DataFrame, grid_fns: dict, *,
    engine, paired, per_year, deflated_sharpe_ratio, pbo_cscv,
    bh_m: dict, bh_d: pd.Series, ch_m: dict, ch_d: pd.Series,
    regime_boundaries: list[tuple[str, str, str]],
    pnl_eval_start: str, years: list[int], select_window_bars: int,
) -> dict:
    """The full mandatory pipeline (steps 2-7 of the ICT-sweep plan) for ONE
    concept's parameter grid: precompute -> full-sample reference (untrusted)
    -> walk-forward selection -> paired-t vs buy&hold/champion -> DSR/PBO on
    the full grid -> regime split. Shared by every `scripts/v5_ict_*.py`
    concept script so the pipeline is identical (and its bugs get fixed once,
    not per-script) — `engine`/`paired`/`per_year` come from
    `scripts.v5_xau_turn_prob`, `deflated_sharpe_ratio`/`pbo_cscv` from
    `src.evaluation.dsr_pbo`, passed in rather than imported here to avoid a
    scripts->src reverse dependency.

    `grid_fns`: {param_key: zero-arg callable returning a forecast Series}.
    """
    print(f"\n{'='*70}\n{name}\n{'='*70}")

    print("precomputing grid...")
    grid_rets = {}
    for key, fc_fn in grid_fns.items():
        fc = fc_fn()
        m, d = engine(df, fc, eval_start="2015-01-01")
        grid_rets[key] = d
    print(f"  grid size: {len(grid_rets)}")

    full_sh = {k: float(v.loc[pnl_eval_start:].mean() / v.loc[pnl_eval_start:].std() * np.sqrt(252))
              if v.loc[pnl_eval_start:].std() > 0 else -np.inf
              for k, v in grid_rets.items()}
    best_full = max(full_sh, key=full_sh.get)
    print(f"(reference only, NOT trusted) full-sample-best: {best_full}  SR={full_sh[best_full]:+.3f}")

    oos, selections = walk_forward_select(grid_rets, years, select_window_bars)
    print("walk-forward selections:")
    print_selections(selections)
    if oos.empty or oos.std() == 0:
        print("!! no OOS data / zero variance — skipping.")
        return dict(name=name, sr_wf=float("nan"))

    sr_wf = float(oos.mean() / oos.std() * np.sqrt(252))
    eq = (1 + oos).cumprod()
    dd_wf = float((eq / eq.cummax() - 1).min() * 100)
    print(f"\nWALK-FORWARD {name}: SR={sr_wf:+.3f}  DD={dd_wf:+.1f}%")
    print(f"buy&hold-vt SR={bh_m['sharpe']:+.3f}   champion SR={ch_m['sharpe']:+.3f}")

    j_bh = pd.concat([oos.rename("s"), bh_d.rename("b")], axis=1).dropna()
    j_ch = pd.concat([oos.rename("s"), ch_d.rename("c")], axis=1).dropna()
    mean_d_bh, t_bh, n_bh = paired(j_bh["s"], j_bh["b"])
    yrp_bh, yrn_bh = per_year(j_bh["s"], j_bh["b"])
    mean_d_ch, t_ch, n_ch = paired(j_ch["s"], j_ch["c"])
    yrp_ch, yrn_ch = per_year(j_ch["s"], j_ch["c"])
    print(f"paired t vs buy&hold:  t={t_bh:+.2f}  years better {yrp_bh}/{yrn_bh}")
    print(f"paired t vs champion:  t={t_ch:+.2f}  years better {yrp_ch}/{yrn_ch}")

    trial_sharpes = np.array([full_sh[k] / np.sqrt(252) for k in full_sh])  # de-annualize
    dsr = deflated_sharpe_ratio(oos.values, trial_sharpes)
    common_idx = None
    for k, v in grid_rets.items():
        s = v.loc[pnl_eval_start:]
        common_idx = s.index if common_idx is None else common_idx.intersection(s.index)
    mats = [v.reindex(common_idx).fillna(0.0).values for v in grid_rets.values()]
    perf_matrix = np.column_stack(mats)
    pbo_res = pbo_cscv(perf_matrix, n_partitions=10)
    print(f"\nDSR={dsr['dsr']:.3f} (n_trials={dsr['n_trials']}, benchmark={dsr['sr_benchmark']:.3f})   "
          f"PBO={pbo_res.pbo:.3f} (n_splits={pbo_res.n_splits})")

    print("\nregime split:")
    regimes = regime_split_sharpe(oos, regime_boundaries)
    print_regime_split(regimes)

    return dict(name=name, sr_wf=sr_wf, dd_wf=dd_wf, t_bh=t_bh, t_ch=t_ch,
               yrp_bh=yrp_bh, yrn_bh=yrn_bh, dsr=dsr["dsr"], pbo=pbo_res.pbo, regimes=regimes)


def print_summary_table(all_results: list[dict]) -> None:
    print(f"{'concept':32s} {'SR_wf':>7} {'DD':>7} {'t_bh':>7} {'t_ch':>7} {'DSR':>6} {'PBO':>6}")
    for r in all_results:
        if r.get("sr_wf") is None or np.isnan(r.get("sr_wf", float("nan"))):
            print(f"{r['name']:32s}  (no data)")
            continue
        print(f"{r['name']:32s} {r['sr_wf']:>+7.2f} {r['dd_wf']:>+7.1f} {r['t_bh']:>+7.2f} "
              f"{r['t_ch']:>+7.2f} {r['dsr']:>6.3f} {r['pbo']:>6.3f}")
