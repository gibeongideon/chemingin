#!/usr/bin/env python3
"""§3eq's one surviving fact, applied to the LIVE ZigZag rule: what does the veto actually cost/buy?

USER, 2026-11-13: "veto on existing ZigZag fires, what is the impact for this if we add?"

THE VETO. §3eq found exactly one cross-source, trend-independent, mechanically sensible result,
and it is a NEGATIVE one: in an uptrend, a pullback that has already broken the prior swing low is
a bad buy (CME -0.107%, t -2.00; Switch -0.295%, t -4.28). As a live gate at fire bar t:

    broke_prior_low = low[t] < price of the most recently CONFIRMED swing low
    trend_up        = close[t] > SMA50[t]
    VETO            = trend_up and broke_prior_low          (the measured cell)

Both inputs are known at the close of bar t and the entry is the open of t+1, so nothing here
looks forward. "Most recently confirmed" means conf <= t, never bar <= t — a pivot is not knowable
on its own bar (§3eq: by confirmation, 46-52% of wave 1 is already given back).

*** THE CONTROL THAT DECIDES IT. *** A veto removes trades, and removing trades changes exposure,
which moves Sharpe and drawdown on its own. So the benchmark is NOT the unvetoed rule — it is a
RANDOM veto that drops the SAME NUMBER of fires, resampled many times. If the real veto sits
inside that distribution it is buying nothing that a coin could not. This is `zigzag-hit-rate-is-
geometry` and §3cx's best-of-K discipline applied to a filter instead of an exit.

The rule backtested is the DEPLOYED one, imported not reimplemented: P>=0.60, stop 3 ATR,
trail 3 ATR, TP 4.3R, max hold 120, one position at a time, 0.25% risk, spread $0.25, Switch
swap -6%/yr. Note the sequencing effect: because the harness holds one trade at a time, vetoing a
fire can ALLOW a later fire that was previously blocked. The veto is not simply "fewer trades".

    python scripts/v5_zigzag_elliott_veto.py
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
from scripts.v5_zigzag_long_history import OUT as ZZ, SRC, THR, backtest, boot, load, sharpe  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "elliott"
SMA_N, N_RANDOM = 50, 250
RNG = np.random.default_rng(20261113)


def prior_confirmed_low(g: pd.DataFrame, order: int = 10, theta: float = 2.5) -> np.ndarray:
    """Price of the most recently CONFIRMED swing low, per bar. NaN until the first one exists."""
    pv = pivot_sequence(g, order, theta)
    lows = pv[pv.kind == "L"].sort_values("conf")
    conf, px = lows.conf.to_numpy(), lows.price.to_numpy()
    idx = np.searchsorted(conf, np.arange(len(g)), side="right") - 1
    out = np.full(len(g), np.nan)
    ok = idx >= 0
    out[ok] = px[idx[ok]]
    return out


def stats(tr: pd.DataFrame, ret: pd.Series, yrs: float) -> dict:
    eq = (1 + ret).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    return dict(n=len(tr), per_yr=len(tr) / yrs, sharpe=sharpe(ret),
                cagr=float(eq.iloc[-1] ** (1 / yrs) - 1) if len(eq) else 0.0,
                maxdd=dd, totR=float(tr.R.sum()) if len(tr) else 0.0,
                win=float((tr.R > 0).mean()) if len(tr) else 0.0,
                evR=float(tr.R.mean()) if len(tr) else 0.0)


def line(tag: str, s: dict, ref: dict | None = None) -> None:
    d = ""
    if ref:
        d = (f"   dSharpe {s['sharpe'] - ref['sharpe']:+.3f}  "
             f"dCAGR {100 * (s['cagr'] - ref['cagr']):+.2f}pp  "
             f"dDD {100 * (s['maxdd'] - ref['maxdd']):+.2f}pp")
    print(f"  {tag:<34} n {s['n']:>4} ({s['per_yr']:>5.1f}/yr)  Sharpe {s['sharpe']:+.3f}  "
          f"CAGR {100 * s['cagr']:+6.2f}%  maxDD {100 * s['maxdd']:+6.2f}%  "
          f"EV {s['evR']:+.3f}R  win {s['win']:.3f}{d}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for src, cfg in SRC.items():
        g = load(src, cfg["gold"])
        p = (pd.read_csv(ZZ / src / "probs.csv", index_col=0, parse_dates=True)
             .iloc[:, 0].reindex(g.index).to_numpy())
        start, swap = cfg["start"], cfg["swap"]
        yrs = (g.index[-1] - pd.Timestamp(start)).days / 365.25

        close, low = g.close.to_numpy(), g.low.to_numpy()
        sma = pd.Series(close).rolling(SMA_N).mean().to_numpy()
        prev_low = prior_confirmed_low(g)
        trend_up = close > sma
        broke = np.isfinite(prev_low) & (low < prev_low)

        fire = np.isfinite(p) & (p >= THR)
        veto_cell = trend_up & broke                    # the §3eq cell, as measured
        veto_any = broke                                # unconditional, for comparison

        print("=" * 118)
        print(f"{src.upper()}  {cfg['gold']}  {start} -> {g.index[-1]:%Y-%m-%d}  ({yrs:.1f}y)   "
              f"deployed rule: P>={THR}, 3 ATR trail, TP 4.3R, hold 120, 0.25% risk, "
              f"swap {swap:+.0%}/yr")
        print("=" * 118)

        tr0, ret0 = backtest(g, fire, start, swap)
        s0 = stats(tr0, ret0, yrs)
        line("BASE — deployed, no veto", s0)

        results = {}
        # THE ALTERNATIVE EXPLANATION. `broke low` clusters in downtrends, and the random
        # control drops fires UNIFORMLY, so a win there could just be "do not buy in a
        # downtrend" — a regime filter, which `regime-gate-champion-disproven` already killed on
        # the champion (SR 1.08->1.01, drawdown WORSE, paired-t -2.88). If a plain SMA filter
        # matches it, the Elliott framing is decoration.
        veto_trend_only = ~trend_up                     # don't buy below SMA50
        for tag, mask in (("VETO trend-up & broke low", veto_cell),
                          ("VETO broke low (any trend)", veto_any),
                          ("VETO close < SMA50 (plain trend)", veto_trend_only),
                          ("INVERSE — keep ONLY vetoed", ~veto_cell)):
            f = fire & ~mask
            tr, ret = backtest(g, f, start, swap)
            if len(tr) < 10:
                print(f"  {tag:<34} only {len(tr)} trades — skipped")
                continue
            s = stats(tr, ret, yrs)
            line(tag, s, s0)
            results[tag] = (s, int((fire & mask).sum()))
            rows.append(dict(src=src, variant=tag, **s))

        # ------------------------------------------------ the control that decides it ----
        print(f"\n  RANDOM-VETO CONTROL — drop the same NUMBER of fires at random, "
              f"{N_RANDOM} draws each.")
        print("  A filter must beat this, not the unfiltered rule: fewer trades moves Sharpe "
              "and drawdown by itself.")
        fire_idx = np.flatnonzero(fire)
        for tag, (s, ndrop) in results.items():
            if tag.startswith("INVERSE") or ndrop <= 0 or ndrop >= len(fire_idx):
                continue
            sh, cg = np.empty(N_RANDOM), np.empty(N_RANDOM)
            for i in range(N_RANDOM):
                drop = RNG.choice(fire_idx, size=ndrop, replace=False)
                f = fire.copy()
                f[drop] = False
                tr, ret = backtest(g, f, start, swap)
                ss = stats(tr, ret, yrs)
                sh[i], cg[i] = ss["sharpe"], ss["cagr"]
            pct = float((sh < s["sharpe"]).mean())
            print(f"    {tag:<32} dropped {ndrop:>3} fires  ->  Sharpe {s['sharpe']:+.3f}  "
                  f"vs random median {np.median(sh):+.3f} "
                  f"[p5 {np.percentile(sh, 5):+.3f}, p95 {np.percentile(sh, 95):+.3f}]")
            print(f"    {'':<32} percentile {100 * pct:5.1f}%  -> "
                  f"{'BEATS the coin' if pct >= 0.95 else 'indistinguishable from a coin'}")
            rows.append(dict(src=src, variant=tag + " [random-control]",
                             sharpe=s["sharpe"], rnd_med=float(np.median(sh)),
                             rnd_p95=float(np.percentile(sh, 95)), pctile=pct, n=ndrop))

        lo, hi = boot(ret0)
        print(f"\n  BASE Sharpe 95% block-bootstrap CI [{lo:+.3f}, {hi:+.3f}] — any delta "
              f"smaller than this band is noise.")

    if rows:
        pd.DataFrame(rows).to_csv(OUT / "veto_impact.csv", index=False)
        print(f"\nwrote {OUT / 'veto_impact.csv'}")


if __name__ == "__main__":
    main()
