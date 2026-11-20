#!/usr/bin/env python3
"""ZigZag NEAR-SL SIGNAL ADD (user 2026-11-20): "if a signal comes in the middle of another order, and the
earlier order is almost at its SL, place another BIGGER order aiming to recover".

Live today a fire during an open position is SKIPPED (no pyramiding). Variant: take that fire at m x size when
the open trade is "almost at SL": close - current_stop <= z x ATR at the fire bar. One add per base trade (the
first qualifying fire). The base event set is FIXED; the add is additive and blocks nothing.

BASE = variant B + breaker (P >= 0.60, 3 ATR stop/trail, TP 4.3R, hold 120, 0.25% risk). CME 2012+, Switch 2018+.
GRID: z in {0.5, 1.0, 1.5, inf (any overlap fire)} x add exit {NORMAL = same as a ZigZag trade, TIGHT k=0.5 /
k=1.0 = k ATR stop + k ATR trail} x size m = 2 (exact scaling; Sharpe is leverage-invariant).
CONTROLS:
  * RANDOM: same trade shape at random hours, matched count (gold drift)
  * STATE-ONLY: same shape at random NON-fire bars that are equally near the stop inside a base trade
    -> does the SIGNAL add anything once you are near the stop?
  * book dSharpe, both halves, vol-matched paired t, maxDD, worst combined loss.
GATE: own-R beats both nulls at 95%, dSharpe > 0 both halves, t_vm >= 2, both sources.

    python scripts/v5_zigzag_nearsl_add.py
"""
from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_zigzag_downtrend_guards import circuit_ok  # noqa: E402
from scripts.v5_zigzag_long_history import (  # noqa: E402
    MAX_HOLD, OUT as ZZ, RISK, SPREAD, SRC, THR, TP_R, TRAIL, backtest, load, sharpe)
from scripts.v5_zigzag_signal import atr  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "zigzag_nearsl_add"
M = 2.0
ZS = (0.5, 1.0, 1.5, np.inf)
EXITS = {"NORMAL": (TRAIL, TP_R), "TIGHT0.5": (0.5, None), "TIGHT1.0": (1.0, None)}
N_NULL = 2000
RNG = np.random.default_rng(20261120)


def trade(o, h, l, a, t, i, k, tp_r, swap):
    """Buy at the open after signal bar i. Stop/trail k x ATR, optional TP at tp_r x risk.
    Returns (t_in, t_out, pnl price incl swap, risk price)."""
    n = len(o)
    if i + 1 >= n or not np.isfinite(a[i]):
        return None
    s, e = i + 1, o[i + 1] + SPREAD
    risk = k * a[i]
    sl, best = e - risk, e
    tp = e + tp_r * risk if tp_r else np.inf
    j, x = n - 1, o[-1]
    for j in range(s, n):
        if l[j] <= sl:
            x = min(sl, o[j]); break
        if h[j] >= tp:
            x = max(tp, o[j]); break
        if j - s >= MAX_HOLD:
            x = o[j]; break
        best = max(best, h[j])
        if np.isfinite(a[j]):
            sl = max(sl, best - k * a[j])
    days = (t[j] - t[s]).total_seconds() / 86400
    return t[s], t[j], x - e + e * swap * days / 365, risk


def base_paths(o, h, l, c, a, s_in, j_out):
    """Re-trace each base trade's stop: distance (close - stop)/ATR at the close of every bar it is open
    (stop as updated at that bar's close, i.e. what the bot sees when it scores the fire)."""
    out = []
    for s, jo in zip(s_in, j_out):
        e = o[s] + SPREAD
        sl, best = e - TRAIL * a[s - 1], e
        d = {}
        for j in range(s, jo):                    # bars strictly before the exit bar
            best = max(best, h[j])
            sl = max(sl, best - TRAIL * a[j])
            d[j] = (c[j] - sl) / a[j]
        out.append(d)
    return out


def to_daily(x: pd.Series, start, last):
    days = pd.date_range(pd.Timestamp(start).normalize(), last.normalize(), freq="D")
    r = x.groupby(x.index.normalize()).sum().reindex(days, fill_value=0.0)
    return r[r.index.dayofweek < 5]


def shape(r, yrs):
    eq = (1 + r).cumprod()
    return sharpe(r), float(eq.iloc[-1] ** (1 / yrs) - 1), float((eq / eq.cummax() - 1).min())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, lines = [], []

    def say(s=""):
        print(s, flush=True); lines.append(s)

    for src, cfg in SRC.items():
        g = load(src, cfg["gold"])
        o, h, l, c = (g[x].to_numpy() for x in ("open", "high", "low", "close"))
        a, t = atr(g).to_numpy(), g.index
        start, swap = pd.Timestamp(cfg["start"]), cfg["swap"]
        yrs = (t[-1] - start).days / 365.25
        p = pd.read_csv(ZZ / src / "probs.csv", index_col=0, parse_dates=True).iloc[:, 0].reindex(t).to_numpy()
        fire = np.isfinite(p) & (p >= THR) & circuit_ok(g, 1.0) & (t >= start) & np.isfinite(a)
        tr, base = backtest(g, fire, start, swap)
        bar = pd.Series(np.arange(len(g)), index=t)
        s_in, j_out = bar.reindex(tr.t_in).to_numpy(), bar.reindex(tr.t_out).to_numpy()
        paths = base_paths(o, h, l, c, a, s_in, j_out)
        mid = start + (t[-1] - start) / 2
        b_sh, b_cg, b_dd = shape(base, yrs)
        b_h1, b_h2 = sharpe(base[base.index < mid]), sharpe(base[base.index >= mid])
        n_overlap = sum(any(fire[j] for j in d) for d in paths)
        say("=" * 122)
        say(f"{src.upper()} {start:%Y-%m-%d} -> {t[-1]:%Y-%m-%d} ({yrs:.1f}y)  base {len(tr)} trades, Sharpe {b_sh:+.3f}, "
            f"CAGR {100 * b_cg:+.2f}%, maxDD {100 * b_dd:.2f}%, halves {b_h1:+.3f}/{b_h2:+.3f}")
        say(f"  base trades with >=1 skipped fire while open: {n_overlap}")
        say("=" * 122)
        say(f"  {'z':>4} {'exit':<9}{'n':>5}{'win%':>6}{'ownR':>8}{'origR':>7}{'p_rand':>8}{'p_state':>8}"
            f"{'Sharpe':>8}{'dSh':>8}{'dSh h1':>8}{'dSh h2':>8}{'t_vm':>7}{'maxDD':>8}{'worst':>7}{'PASS':>6}")

        win = np.where(np.isfinite(a) & (t >= start))[0]
        win = win[(win > 1) & (win < len(g) - 2)]
        for z, (ename, (k, tpr)) in itertools.product(ZS, EXITS.items()):
            adds, state_bars = [], []
            for b, d in enumerate(paths):
                near = [j for j, dist in d.items() if dist <= z]
                hit = [j for j in near if fire[j]]
                state_bars += [j for j in near if not fire[j]]
                if not hit:
                    continue
                r = trade(o, h, l, a, t, hit[0], k, tpr, swap)
                if r:
                    brisk = TRAIL * a[s_in[b] - 1]
                    adds.append((r[0], r[1], r[2] / r[3], M * r[2] / brisk, tr.R.iat[b]))
            if len(adds) < 15:
                say(f"  {z:>4} {ename:<9}{len(adds):>5}  too few"); continue
            ad = pd.DataFrame(adds, columns=["t_in", "t_out", "ownR", "baseR", "origR"])

            def null(pool_bars):
                pool = RNG.choice(pool_bars, size=min(4000, len(pool_bars)), replace=False)
                v = np.array([(lambda r: r[2] / r[3] if r else np.nan)(trade(o, h, l, a, t, i, k, tpr, swap))
                              for i in pool])
                v = v[np.isfinite(v)]
                ms = np.array([RNG.choice(v, len(ad)).mean() for _ in range(N_NULL)])
                return float((ms >= ad.ownR.mean()).mean())
            p_rand = null(win)
            p_state = null(np.array(state_bars)) if len(state_bars) > 50 else np.nan

            book = base.add(to_daily(pd.Series(RISK * ad.baseR.to_numpy(), index=ad.t_out), start, t[-1]),
                            fill_value=0.0)
            sh, cg, dd = shape(book, yrs)
            h1, h2 = sharpe(book[book.index < mid]), sharpe(book[book.index >= mid])
            dl = book * (base.std() / book.std()) - base
            t_vm = float(dl.mean() / dl.std() * np.sqrt(len(dl)))
            worst = float((ad.origR + ad.baseR).min())
            ok = p_rand <= 0.05 and (p_state <= 0.05) and h1 > b_h1 and h2 > b_h2 and t_vm >= 2
            say(f"  {z:>4} {ename:<9}{len(ad):>5}{100 * (ad.ownR > 0).mean():>6.1f}{ad.ownR.mean():>+8.3f}"
                f"{ad.origR.mean():>+7.2f}{p_rand:>8.3f}{p_state:>8.3f}{sh:>+8.3f}{sh - b_sh:>+8.3f}"
                f"{h1 - b_h1:>+8.3f}{h2 - b_h2:>+8.3f}{t_vm:>+7.2f}{100 * dd:>7.2f}%{worst:>+7.2f}"
                f"{('YES' if ok else 'no'):>6}")
            rows.append(dict(src=src, z=z, exit=ename, n=len(ad), win=(ad.ownR > 0).mean(), own_R=ad.ownR.mean(),
                             orig_R=ad.origR.mean(), p_rand=p_rand, p_state=p_state, sharpe=sh, base_sharpe=b_sh,
                             d_h1=h1 - b_h1, d_h2=h2 - b_h2, t_vm=t_vm, maxdd=dd, base_dd=b_dd, worst=worst,
                             passed=ok))
        say()
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "cells.csv", index=False)
    pv = d.pivot_table(index=["z", "exit"], columns="src", values="passed", aggfunc="first").fillna(False)
    say(f"CELLS PASSING IN BOTH SOURCES: {int(pv.all(axis=1).sum())} / {len(pv)}")
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
