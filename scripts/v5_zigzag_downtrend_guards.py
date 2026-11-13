#!/usr/bin/env python3
"""A FAMILY of downtrend guards on the live ZigZag, all judged against the same honest control.

USER, 2026-11-13: "any other way to try to guard when in downtrend, explore".

TWO GUARDS ARE ALREADY LIVE, so this starts by auditing them rather than inventing a third:
  * §3dz intraday circuit breaker, k=1.0 — on ALL six ZigZag configs since 2026-11-11
  * the champion trend gate (`trend_gate`, min_forecast 0.5) — on Switch 360601 / 360601-c only
Neither has ever been measured under the control below, which is new (§3er).

*** THE CONTROL. *** A guard removes trades, and removing trades moves Sharpe and drawdown by
itself — §3er's plain SMA50 filter looked like +0.244 Sharpe on CME and turned out to be a coin
flip on Switch. So every guard is scored against a RANDOM veto that drops the SAME NUMBER of
fires, resampled N_RANDOM times, and the reported number is its percentile in that distribution.
Beating the unguarded rule means nothing; beating the coin is the test.

*** THE BAR, SET BEFORE LOOKING. *** A guard must reach the 95th percentile in BOTH sources.
A useless guard clears one source 5% of the time, so it clears both 0.25% of the time; across the
~11 guards tried that is a family-wise error of ~2.7%. One source is not a result — §3er's two
best candidates each passed one source and failed the other, in opposite directions.

Everything is causal: each guard is evaluated on the CLOSE of the fire bar and the entry is the
next bar's open. The H4 champion forecast is resampled with the bar stamped at its CLOSE and then
lagged one more bar, because an H4 bar stamped t is not knowable until t+4h
(`cross-timeframe-resample-is-lookahead`).

    python scripts/v5_zigzag_downtrend_guards.py
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
from scripts.v5_elliott_wave import pivot_sequence  # noqa: E402
from scripts.v5_xau_turning_ml import atr  # noqa: E402
from scripts.v5_zigzag_elliott_veto import stats  # noqa: E402
from scripts.v5_zigzag_long_history import OUT as ZZ, SRC, THR, backtest, boot, load, sharpe  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "elliott"
N_RANDOM, PASS_PCT = 200, 0.95
RNG = np.random.default_rng(20261113)


def champion_ok(g: pd.DataFrame, min_fc: float = 0.5) -> np.ndarray:
    """The LIVE Switch gate: champion H4 forecast >= min_fc, with no cross-timeframe lookahead."""
    h4 = g.close.resample("4h", label="right", closed="right").last().dropna()
    fc = champion_signal(h4).shift(1)            # stamped at the H4 CLOSE, then lagged one bar
    return (fc.reindex(g.index, method="ffill").to_numpy() >= min_fc)


def circuit_ok(g: pd.DataFrame, k: float) -> np.ndarray:
    """§3dz, as deployed: block if the broker day is already down > k x YESTERDAY's D1 ATR(14)."""
    day = g.index.normalize()
    d1 = g.groupby(day).agg(high=("high", "max"), low=("low", "min"),
                            close=("close", "last"), open=("open", "first"))
    pc = d1.close.shift(1)
    tr = pd.concat([d1.high - d1.low, (d1.high - pc).abs(), (d1.low - pc).abs()], axis=1).max(axis=1)
    atr_d1 = tr.ewm(alpha=1 / 14, adjust=False).mean().shift(1)
    first_open = g.open.groupby(day).transform("first")
    run_low = g.low.groupby(day).cummin()
    thr = -k * atr_d1.reindex(day).to_numpy()
    move = (run_low - first_open).to_numpy()
    return ~(np.isfinite(thr) & (move < thr))


def structure_ok(g: pd.DataFrame, order: int = 10, theta: float = 2.5) -> np.ndarray:
    """Structural uptrend: the last CONFIRMED swing high is above the one before it."""
    pv = pivot_sequence(g, order, theta)
    hs = pv[pv.kind == "H"].sort_values("conf")
    conf, px = hs.conf.to_numpy(), hs.price.to_numpy()
    idx = np.searchsorted(conf, np.arange(len(g)), side="right") - 1
    out = np.zeros(len(g), dtype=bool)
    ok = idx >= 1
    out[ok] = px[idx[ok]] > px[idx[ok] - 1]
    return out


def guards(g: pd.DataFrame) -> dict:
    c = g.close
    hi500 = g.high.rolling(500).max()
    r = c.pct_change()
    dn = r.where(r < 0, 0.0).rolling(500).std()
    return {
        "close > SMA50":              (c > c.rolling(50).mean()).to_numpy(),
        "close > SMA200":             (c > c.rolling(200).mean()).to_numpy(),
        "EWMAC 16/64 > 0":            (c.ewm(span=16).mean() > c.ewm(span=64).mean()).to_numpy(),
        "EWMAC 32/128 > 0":           (c.ewm(span=32).mean() > c.ewm(span=128).mean()).to_numpy(),
        "120-bar return > 0":         (c.pct_change(120) > 0).to_numpy(),
        "within 5% of 500-bar high":  (c > hi500 * 0.95).to_numpy(),
        "within 10% of 500-bar high": (c > hi500 * 0.90).to_numpy(),
        "circuit breaker k=1.0 LIVE": circuit_ok(g, 1.0),
        "circuit breaker k=1.5":      circuit_ok(g, 1.5),
        "champion fc >= 0.5  LIVE":   champion_ok(g, 0.5),
        "higher confirmed swing high": structure_ok(g),
        "downside vol below median":  (dn < dn.rolling(2000).median()).to_numpy(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, verdict = [], {}
    for src, cfg in SRC.items():
        g = load(src, cfg["gold"])
        p = (pd.read_csv(ZZ / src / "probs.csv", index_col=0, parse_dates=True)
             .iloc[:, 0].reindex(g.index).to_numpy())
        start, swap = cfg["start"], cfg["swap"]
        yrs = (g.index[-1] - pd.Timestamp(start)).days / 365.25
        fire = np.isfinite(p) & (p >= THR)
        fire_idx = np.flatnonzero(fire & (g.index >= pd.Timestamp(start)))

        tr0, ret0 = backtest(g, fire, start, swap)
        s0 = stats(tr0, ret0, yrs)
        lo, hi = boot(ret0)
        print("=" * 120)
        print(f"{src.upper()}  {cfg['gold']}  {start} -> {g.index[-1]:%Y-%m-%d} ({yrs:.1f}y)   "
              f"BASE: n {s0['n']} ({s0['per_yr']:.0f}/yr)  Sharpe {s0['sharpe']:+.3f} "
              f"[boot {lo:+.2f},{hi:+.2f}]  CAGR {100 * s0['cagr']:+.2f}%  "
              f"maxDD {100 * s0['maxdd']:+.2f}%")
        print("=" * 120)
        print(f"  {'guard (fires BLOCKED when false)':<30}{'kept':>6}{'/yr':>6}"
              f"{'Sharpe':>8}{'dSh':>7}{'CAGR':>8}{'maxDD':>8}{'EV(R)':>8}"
              f"{'drop':>6}{'rand med':>9}{'pctile':>8}")

        for name, ok in guards(g).items():
            f = fire & ok
            ndrop = int((fire & ~ok)[fire_idx - fire_idx.min() + fire_idx.min()].sum()
                        if False else (fire & ~ok & (g.index >= pd.Timestamp(start))).sum())
            tr, ret = backtest(g, f, start, swap)
            if len(tr) < 20 or ndrop <= 0 or ndrop >= len(fire_idx):
                print(f"  {name:<30} unusable (kept {len(tr)}, dropped {ndrop})")
                continue
            s = stats(tr, ret, yrs)
            sh = np.empty(N_RANDOM)
            for i in range(N_RANDOM):
                ff = fire.copy()
                ff[RNG.choice(fire_idx, size=ndrop, replace=False)] = False
                sh[i] = sharpe(backtest(g, ff, start, swap)[1])
            pct = float((sh < s["sharpe"]).mean())
            print(f"  {name:<30}{s['n']:>6}{s['per_yr']:>6.0f}{s['sharpe']:>8.3f}"
                  f"{s['sharpe'] - s0['sharpe']:>+7.3f}{100 * s['cagr']:>7.2f}%"
                  f"{100 * s['maxdd']:>7.2f}%{s['evR']:>8.3f}{ndrop:>6}"
                  f"{np.median(sh):>9.3f}{100 * pct:>7.1f}%")
            rows.append(dict(src=src, guard=name, **s, ndrop=ndrop,
                             rnd_med=float(np.median(sh)), pctile=pct))
            verdict.setdefault(name, {})[src] = pct

    print("\n" + "=" * 120)
    print("BOTH-SOURCES BAR — a guard must reach the 95th percentile vs a random same-size drop "
          "in CME *and* Switch")
    print("=" * 120)
    print(f"  {'guard':<30}{'CME pctile':>12}{'Switch pctile':>15}   verdict")
    for name, d in verdict.items():
        a, b = d.get("cme"), d.get("switch")
        if a is None or b is None:
            continue
        ok = a >= PASS_PCT and b >= PASS_PCT
        print(f"  {name:<30}{100 * a:>11.1f}%{100 * b:>14.1f}%   "
              f"{'*** PASSES BOTH ***' if ok else 'fails'}")
    if not any(d.get("cme", 0) >= PASS_PCT and d.get("switch", 0) >= PASS_PCT
               for d in verdict.values()):
        print("\n  NOTHING PASSES. Every guard that looks good in one source is a coin in the "
              "other —\n  the same split §3er found, and the reason the bar is both.")
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "downtrend_guards.csv", index=False)
        print(f"\nwrote {OUT / 'downtrend_guards.csv'}")


if __name__ == "__main__":
    main()
