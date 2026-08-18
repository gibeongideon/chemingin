"""Rolling hedge-ratio spread mean-reversion — the properly-built version of §3t.

§3t's simple 1:1 return-divergence z-score was a cost mirage on GOLD/SILVER and
PLAT/PALL (real Sharpe evaporates at documented live cost). This rebuilds the
signal closer to the ORIGINAL gold-silver study's construction
([[gold-silver-spread-disproven]]: spread = logA - beta*logB, beta a rolling-OLS
hedge ratio shifted 1 bar, z-scored, traded with a discrete entry/exit/stop
state machine) rather than the simpler unweighted version — and, having learned
the lesson, goes straight to HONEST/DOCUMENTED live cost instead of screening on
thin CSV cost first.

Tested on GOLD/SILVER and PLAT/PALL (where the simple version died on cost) and
on SPX/NDX (where the simple version survived cost cleanly at DSR 0.68 — this
checks whether the more careful construction improves on that baseline, or is
just a different way to lose the same way).

    python scripts/v5_hedge_ratio_divergence.py
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
from src.evaluation.lookahead_probe import probe_lookahead, print_report  # noqa: E402
from src.evaluation.dsr_pbo import deflated_sharpe_ratio, pbo_cscv  # noqa: E402
from src.evaluation.fitness import rank_candidates  # noqa: E402
from scripts.v5_cross_asset_divergence import build_divergence_probe_frame  # noqa: E402

EVAL_START = "2018-01-01"
BETA_WINDOWS = (40, 60, 90)
Z_WINDOWS = (15, 20, 40)
ENTRIES = (1.5, 2.0, 2.5)
EXIT_Z = 0.5
STOP_Z = 4.0

# (sym_a, sym_b, honest_bp_floor_for_a) — bp=None means no correction needed
# (already on MANDATORY CONTROLS' "cost-verified conservative" list)
PAIRS = (
    ("GOLD", "SILVER", 8.5),
    ("PLAT", "PALL", 56.4),
    ("SPX", "NDX", None),
)


def rolling_beta(log_a: pd.Series, log_b: pd.Series, window: int) -> pd.Series:
    """Rolling-OLS hedge ratio, SHIFTED 1 bar: the beta applied to bar t's
    spread is estimated from the window ENDING at bar t-1, so nothing at bar t
    leaks into its own hedge ratio."""
    cov = log_a.rolling(window).cov(log_b)
    var = log_b.rolling(window).var()
    return (cov / var).shift(1)


def hedge_spread_z(close_a: pd.Series, close_b: pd.Series, beta_window: int, z_window: int) -> pd.Series:
    log_a, log_b = np.log(close_a), np.log(close_b)
    beta = rolling_beta(log_a, log_b, beta_window)
    spread = log_a - beta * log_b
    m = spread.rolling(z_window).mean()
    sd = spread.rolling(z_window).std()
    return (spread - m) / sd


def hedge_fc_from_frame(df: pd.DataFrame, beta_window: int, z_window: int,
                        entry: float = 2.0, exit_z: float = EXIT_Z, stop_z: float = STOP_Z) -> pd.Series:
    """Discrete entry/exit/stop state machine on the hedge-ratio spread z-score.
    +1 = long spread (long A), -1 = short spread (short A), 0 = flat. Causal:
    state at bar i depends only on z[i] (itself using beta shifted 1 bar) and
    state at bar i-1."""
    z = hedge_spread_z(df["close_a"], df["close_b"], beta_window, z_window).values
    n = len(z)
    state = np.zeros(n)
    cur = 0
    for i in range(n):
        zi = z[i]
        if not np.isfinite(zi):
            state[i] = cur
            continue
        if cur == 0:
            if zi > entry:
                cur = -1
            elif zi < -entry:
                cur = 1
        elif cur == 1:
            if zi >= -exit_z or zi <= -stop_z:
                cur = 0
        elif cur == -1:
            if zi <= exit_z or zi >= stop_z:
                cur = 0
        state[i] = cur
    return pd.Series(state, index=df.index)


def net_returns(sym_a: str, sym_b: str, beta_window: int, z_window: int, entry: float,
                bp_floor: float | None) -> pd.Series:
    df_a = pd.read_csv(ROOT / "data" / f"{sym_a}_D1_long.csv", parse_dates=["time"], index_col="time").sort_index()
    df_a = df_a[~df_a.index.duplicated(keep="last")]
    df_b = pd.read_csv(ROOT / "data" / f"{sym_b}_D1_long.csv", parse_dates=["time"], index_col="time").sort_index()
    df_b = df_b[~df_b.index.duplicated(keep="last")]
    j = pd.concat([df_a["close"].rename("close_a"), df_b["close"].rename("close_b")], axis=1).dropna()
    fc = hedge_fc_from_frame(j, beta_window, z_window, entry).reindex(df_a.index).fillna(0.0)

    close = df_a["close"]
    base_floor = df_a["spread"].clip(lower=df_a["spread"].median())
    spread_px = np.maximum(base_floor, close * bp_floor / 1e4) if bp_floor else base_floor
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
    print("=== STAGE 0: lookahead probe (rolling hedge-ratio spread, discrete state machine) ===")
    probes = {
        f"hedge_fc(bw={bw},zw={zw},entry={e})": (lambda d, bw=bw, zw=zw, e=e: hedge_fc_from_frame(d, bw, zw, e))
        for bw in (60,) for zw in (20,) for e in (2.0,)
    }
    results = {
        name: probe_lookahead(fn, n=900, offsets=(150, 100, 50, 20), frame_builder=build_divergence_probe_frame)
        for name, fn in probes.items()
    }
    print_report(results)
    if not all(v.ok for v in results.values()):
        print("!! lookahead probe FAILED — refusing to trust the backtest below.")
        return

    all_rows = []
    for sym_a, sym_b, bp in PAIRS:
        print(f"\n=== {sym_a}/{sym_b} — honest cost ({'documented ' + str(bp) + 'bp floor' if bp else 'already conservative'}) ===")
        grid_rets = {}
        for bw in BETA_WINDOWS:
            for zw in Z_WINDOWS:
                for e in ENTRIES:
                    grid_rets[(bw, zw, e)] = net_returns(sym_a, sym_b, bw, zw, e, bp)
        M = pd.DataFrame(grid_rets).dropna(how="all").fillna(0.0).loc[EVAL_START:]
        sh = {k: sharpe(M[k], start=M.index[0].strftime("%Y-%m-%d")) for k in M.columns}
        best_key = max(sh, key=lambda k: sh[k] if np.isfinite(sh[k]) else -1e9)
        n_pos = sum(1 for v in sh.values() if np.isfinite(v) and v > 0)
        trial_sh_daily = np.array([sh[k] / np.sqrt(252) for k in sh])
        dsr = deflated_sharpe_ratio(M[best_key].values, trial_sh_daily)
        pbo = pbo_cscv(M.values, n_partitions=10)
        best_ret = M[best_key]
        print(f"  best (beta_w={best_key[0]}, z_w={best_key[1]}, entry={best_key[2]}): "
              f"SR={sh[best_key]:+.3f}  DD={dd(best_ret):+.1f}%  pos={n_pos}/{len(sh)}")
        print(f"  DSR (within-pair, {len(sh)} cells) = {dsr['dsr']:.3f}   PBO = {pbo.pbo:.3f}")
        print("  per-year:")
        for yr, g in best_ret.groupby(best_ret.index.year):
            sr_y = float(g.mean() / g.std() * np.sqrt(252)) if g.std() > 0 else float("nan")
            print(f"    {yr}  SR {sr_y:+.2f}")
        all_rows.append(dict(pair=f"{sym_a}/{sym_b}", sharpe=sh[best_key], max_drawdown_pct=dd(best_ret),
                             dsr=dsr["dsr"], pbo=pbo.pbo, n_pos=n_pos, n_cells=len(sh)))

    print("\n=== Fitness-ranked summary (hedge-ratio version, honest cost throughout) ===")
    for r in rank_candidates(all_rows):
        print(f"  fitness={r['fitness']:5.1f}  {r['pair']:12s}  SR={r['sharpe']:+.3f}  "
              f"DD={r['max_drawdown_pct']:+.1f}%  DSR={r['dsr']:.3f}  PBO={r['pbo']:.3f}")


if __name__ == "__main__":
    main()
