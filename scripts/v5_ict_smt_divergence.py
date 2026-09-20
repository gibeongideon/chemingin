"""ICT/SMC concept sweep, Phase 3: SMT Divergence. D1, two pairs.

Plan: `~/.claude/plans/i-wanttt-you-to-playful-widget.md`. Structurally close
to the existing `v5_cross_asset_divergence.py` family (whose SPX/NDX result
was the closest-to-real thing that line found: DSR 0.523, not disproven, just
sub-threshold) — but SMT Divergence is a MECHANISTICALLY DIFFERENT signal, not
a re-run: the earlier work fades a rolling Z-SCORE of relative RETURNS
(continuous, distributional); SMT fades a discrete STRUCTURAL disagreement —
asset A sweeps to a new confirmed swing extreme that asset B does NOT confirm
(P1+P10). Same underlying intuition (two correlated assets, one lying), a
genuinely different detector.

  SPX/NDX first (existing infra reuse, corr ~0.86, the prior line's
  best-scoring pair).
  XAU/SILVER second, EXPLICITLY checked against the already-disproven
  GOLD/SILVER z-score-divergence precedent (`gold-silver-spread-disproven`:
  corr 0.79, edge pre-2015-only, dead OOS 2017+) — must clear the same bar on
  its own merits, not be waved through by analogy.

    python scripts/v5_ict_smt_divergence.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.v5_basket_challenge as vbc  # noqa: E402
from src.features.ict_primitives import swing_levels, smt_divergence, build_smt_probe_frame  # noqa: E402
from src.evaluation.lookahead_probe import probe_lookahead, print_report  # noqa: E402
from src.evaluation.dsr_pbo import deflated_sharpe_ratio, pbo_cscv  # noqa: E402
from src.evaluation.fitness import rank_candidates  # noqa: E402
from src.evaluation.walk_forward_grid import walk_forward_select, print_selections  # noqa: E402

EVAL_START = "2018-01-01"
YEARS = list(range(2018, 2027))
SELECT_WINDOW_BARS = 3 * 252   # trailing 3y, D1
ORDERS = (3, 5, 8)
HOLD_BARS = (3, 5, 10, 20)   # D1 bars = trading days

PAIRS = (("SPX", "NDX"), ("GOLD", "SILVER"))


def load_ohlc(sym: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / f"{sym}_D1_long.csv", parse_dates=["time"], index_col="time").sort_index()
    return df[~df.index.duplicated(keep="last")]


def smt_fc(df_a: pd.DataFrame, df_b: pd.DataFrame, order: int, hold_bars: int) -> pd.Series:
    j = df_a.index.intersection(df_b.index)
    a, b = df_a.loc[j], df_b.loc[j]
    sh_a, sl_a = swing_levels(a, order=order)
    sh_b, sl_b = swing_levels(b, order=order)
    raw = smt_divergence(sh_a, sl_a, a["close"], sh_b, sl_b, b["close"]).values
    n = len(j)
    state = np.zeros(n)
    cur, since = 0.0, 10**9
    for i in range(n):
        if raw[i] != 0:
            cur, since = raw[i], 0
        else:
            since += 1
            if since > hold_bars:
                cur = 0.0
        state[i] = cur
    return pd.Series(state, index=j, name="smt_fc")


def net_returns(sym_a: str, sym_b: str, order: int, hold_bars: int) -> pd.Series:
    """Trade sym_a on the SMT signal, vol-targeted/buffered — same D1 engine
    as `v5_cross_asset_divergence.py::net_returns_for_pair`."""
    df_a, df_b = load_ohlc(sym_a), load_ohlc(sym_b)
    fc = smt_fc(df_a, df_b, order, hold_bars).reindex(df_a.index).fillna(0.0)
    close = df_a["close"]
    spread_px = df_a["spread"].clip(lower=df_a["spread"].median())
    ret = close.pct_change()
    vol = ret.ewm(halflife=42, min_periods=20).std() * np.sqrt(252)
    pos = vbc._buffered_pos(fc, vol, spread_px, close, 252).shift(1).fillna(0.0)
    cost = pos.diff().abs().fillna(0.0) * (spread_px / close)
    net = (pos * ret - cost).fillna(0.0).resample("D").sum()
    return net[net.index.dayofweek < 5]


def sharpe(d: pd.Series, start: str = EVAL_START) -> float:
    d = d.loc[start:].dropna()
    return float(d.mean() / d.std() * np.sqrt(252)) if len(d) > 20 and d.std() > 0 else float("nan")


def dd(d: pd.Series, start: str = EVAL_START) -> float:
    e = (1 + d.loc[start:]).cumprod()
    return float((e / e.cummax() - 1).min() * 100) if len(e) else float("nan")


def main():
    print("=== STAGE 0: lookahead probe (two-asset full-OHLC synthetic frame) ===")
    probes = {}
    for o in (5,):
        for h in (5,):
            def fn(d, o=o, h=h):
                a = d[["open_a", "high_a", "low_a", "close_a"]].rename(
                    columns={"open_a": "open", "high_a": "high", "low_a": "low", "close_a": "close"})
                b = d[["open_b", "high_b", "low_b", "close_b"]].rename(
                    columns={"open_b": "open", "high_b": "high", "low_b": "low", "close_b": "close"})
                return smt_fc(a, b, o, h)
            probes[f"smt_fc(order={o},hold={h})"] = probe_lookahead(
                fn, n=900, offsets=(60, 40, 20, 10), warmup_buffer=400, frame_builder=build_smt_probe_frame
            )
    print_report(probes)
    if not all(v.ok for v in probes.values()):
        print("!! lookahead probe FAILED — refusing to trust the backtest below.")
        return

    all_rows = []
    for sym_a, sym_b in PAIRS:
        control = "  <-- checked against gold-silver-spread-disproven precedent" if (sym_a, sym_b) == ("GOLD", "SILVER") else ""
        print(f"\n{'='*70}\nSMT Divergence: {sym_a}/{sym_b}{control}\n{'='*70}")
        grid = {}
        for o in ORDERS:
            for h in HOLD_BARS:
                grid[(o, h)] = net_returns(sym_a, sym_b, o, h)
        full_sh = {k: sharpe(v) for k, v in grid.items()}
        full_sh = {k: v for k, v in full_sh.items() if np.isfinite(v)}
        if not full_sh:
            print("  no valid cells — skipping pair.")
            continue
        best_k = max(full_sh, key=full_sh.get)
        best_ret = grid[best_k]
        print(f"(reference only, NOT trusted) full-sample-best: order={best_k[0]} hold={best_k[1]}  "
              f"SR={full_sh[best_k]:+.3f}  DD={dd(best_ret):+.1f}%")
        print("full grid:")
        for k in sorted(full_sh, key=full_sh.get, reverse=True):
            print(f"  order={k[0]} hold={k[1]:>2d}  SR={full_sh[k]:+.3f}")

        trial_sharpes = np.array([s / np.sqrt(252) for s in full_sh.values()])
        obs = best_ret.loc[EVAL_START:].dropna().values
        dsr = deflated_sharpe_ratio(obs, trial_sharpes)
        common_idx = None
        for v in grid.values():
            s = v.loc[EVAL_START:]
            common_idx = s.index if common_idx is None else common_idx.intersection(s.index)
        mats = [v.reindex(common_idx).fillna(0.0).values for v in grid.values()]
        pbo_res = pbo_cscv(np.column_stack(mats), n_partitions=10)
        print(f"\nDSR (best cell)={dsr['dsr']:.3f} (n_trials={dsr['n_trials']})   PBO={pbo_res.pbo:.3f}")

        by_year = best_ret.loc[EVAL_START:].groupby(best_ret.loc[EVAL_START:].index.year)
        print("per-year (full-sample-best cell, reference only):")
        for yr, g in by_year:
            sr_y = float(g.mean() / g.std() * np.sqrt(252)) if g.std() > 0 else float("nan")
            print(f"  {yr}  SR {sr_y:+.2f}")

        print("\nWALK-FORWARD selection (trailing 3y, re-picked each Jan) — the honest number:")
        oos, selections = walk_forward_select(grid, YEARS, SELECT_WINDOW_BARS)
        print_selections(selections)
        if oos.empty or oos.std() == 0:
            sr_wf, dd_wf = float("nan"), float("nan")
            print("  no OOS data / zero variance.")
        else:
            sr_wf = float(oos.mean() / oos.std() * np.sqrt(252))
            e = (1 + oos).cumprod()
            dd_wf = float((e / e.cummax() - 1).min() * 100)
            print(f"  WALK-FORWARD {sym_a}/{sym_b} SMT: SR={sr_wf:+.3f}  DD={dd_wf:+.1f}%  "
                  f"(vs full-sample-best-cell reference SR={full_sh[best_k]:+.3f} — NOT trusted)")
            for yr, g in oos.groupby(oos.index.year):
                sr_y = float(g.mean() / g.std() * np.sqrt(252)) if g.std() > 0 else float("nan")
                print(f"    {yr}  SR {sr_y:+.2f}")

        all_rows.append(dict(pair=f"{sym_a}/{sym_b}", order=best_k[0], hold=best_k[1],
                             sharpe=sr_wf if np.isfinite(sr_wf) else full_sh[best_k],
                             max_drawdown_pct=abs(dd_wf if np.isfinite(dd_wf) else dd(best_ret)),
                             dsr=dsr["dsr"], pbo=pbo_res.pbo))

    if all_rows:
        print(f"\n{'='*70}\nSMT DIVERGENCE SUMMARY (walk-forward SR/DD where available)\n{'='*70}")
        ranked = rank_candidates(all_rows)
        for r in ranked:
            print(f"  {r['pair']:14s} SR={r['sharpe']:+.3f}  DD={-r['max_drawdown_pct']:+.1f}%  "
                  f"DSR={r['dsr']:.3f}  PBO={r['pbo']:.3f}  fitness={r['fitness']:.1f}")
        out = ROOT / "data" / "v5_runs" / "ict_smt_divergence_summary.csv"
        pd.DataFrame(ranked).to_csv(out, index=False)
        print(f"\nsummary -> {out}")


if __name__ == "__main__":
    main()
