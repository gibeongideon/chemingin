#!/usr/bin/env python3
"""ZigZag SL RECOVERY (user request 2026-11-19): "if the buy reaches the SL, add a DOUBLE-lot buy AT the SL
with a tight SL and a trailing stop to recover". Follow-up: "if it goes well, no order to place".

DIFFERENT FROM §3ds. §3ds added BEFORE the stop (d = 1-2 ATR, stop at 3). This order is placed only AFTER
the original is stopped out AT A LOSS, as a fresh 2x-lot buy at the stop price. Winners (TP, time exit,
a trail that locked profit) never trigger it.

BASE = live variant B (FTMO 360608): P >= 0.60, 3.0 ATR initial stop + trail, TP 4.3R, hold 120 H1 bars,
circuit breaker k=1.0, 0.25% risk. Both sources: CME GC 2012+ (no swap), Switch XAUUSDs 2018+ (swap -6%/yr).
The base event set is FIXED: the recovery trade does not block later ZigZag fires (live it would need its
own magic or the no-pyramid check would stop new fires), so its P&L is purely additive.

RECOVERY TRADE GRID (24 cells/source):
  k     tight stop = k x ATR(14) below its entry, and the trail runs at the same k x ATR from the best high
        k in {0.5, 1.0, 1.5}  (the base stop is 3.0)
  mode  TRAIL      exit only on the trailing stop / max hold
        TRAIL+TP   also take profit once the 2x position has earned back the original loss
  fill  A  at the stop price + spread, ON the stop bar. Pessimistic: that bar's low can stop it out, its
           high is NOT used for TP/trail (H1 can't say whether the high came after the stop-out)
        B  at the NEXT bar's open + spread (an hourly bot that notices on its next pass)
  cb    recovery also obeys the circuit breaker (no new buy on a day already down > 1 D1 ATR)
Size m = 2 (the spec). R is size-independent, so the m-scaling is exact.

CONTROLS
  * RANDOM TIMING: the same trade shape (k stop, k trail, same TP-in-ATR distribution, same hold) entered at
    random bars. Gold drifts up, so ANY long earns something; the question is whether "just got stopped
    out" is a better entry than a random hour. Null = 2000 resamples of the same count from a 6000-bar pool.
  * Sharpe is leverage-invariant, so dSharpe (book vs base) is the edge measure. Doubling only scales.
  * Vol-matched paired t: combined book rescaled to the base's daily vol, t of the daily difference.
  * Both halves, worst combined trade (base loss + recovery loss, base R), maxDD.
GATE (all): random-timing pctile >= 95, dSharpe > 0 in both halves, vol-matched t >= 2, in BOTH sources.

    python scripts/v5_zigzag_sl_recovery.py
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
    MAX_HOLD, OUT as ZZ, RISK, SPREAD, SRC, THR, TRAIL, backtest, load, sharpe)
from scripts.v5_zigzag_signal import atr  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "zigzag_sl_recovery"
M = 2.0
KS, MODES, FILLS, CBS = (0.5, 1.0, 1.5), ("TRAIL", "TRAIL+TP"), ("A", "B"), (False, True)
N_POOL, N_NULL = 6000, 2000
RNG = np.random.default_rng(20261119)


def rec_trade(o, h, l, a, t, j, px, fill, k, tp_atr, swap):
    """One recovery buy after a base stop-out on bar j at price px.
    Returns (t_in, t_out, pnl per unit in price incl swap, own risk in price, exit)."""
    n = len(o)
    if fill == "A":
        s, e, ar = j, px + SPREAD, a[j - 1]
    else:
        if j + 1 >= n:
            return None
        s, e, ar = j + 1, o[j + 1] + SPREAD, a[j]
    if not np.isfinite(ar) or ar <= 0:
        return None
    risk = k * ar
    sl, best = e - risk, e
    tp = e + tp_atr * ar if tp_atr is not None else np.inf
    jj, x, why = n - 1, o[-1], "open"
    for jj in range(s, n):
        if fill == "A" and jj == s:           # entered mid-bar: only the remaining low counts
            if l[jj] <= sl:
                x, why = sl, "stop"
                break
            continue
        if l[jj] <= sl:
            x, why = min(sl, o[jj]), "stop"
            break
        if h[jj] >= tp:
            x, why = max(tp, o[jj]), "tp"
            break
        if jj - s >= MAX_HOLD:
            x, why = o[jj], "time"
            break
        best = max(best, h[jj])
        if np.isfinite(a[jj]):
            sl = max(sl, best - k * a[jj])
    days = (t[jj] - t[s]).total_seconds() / 86400
    return t[s], t[jj], x - e + e * swap * days / 365, risk, why


def to_daily(pnl_by_day: pd.Series, start, last) -> pd.Series:
    days = pd.date_range(pd.Timestamp(start).normalize(), last.normalize(), freq="D")
    r = pnl_by_day.groupby(pnl_by_day.index.normalize()).sum().reindex(days, fill_value=0.0)
    return r[r.index.dayofweek < 5]


def shape(r: pd.Series, yrs: float):
    eq = (1 + r).cumprod()
    return sharpe(r), float(eq.iloc[-1] ** (1 / yrs) - 1), float((eq / eq.cummax() - 1).min())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, lines = [], []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    for src, cfg in SRC.items():
        g = load(src, cfg["gold"])
        o, h, l = (g[c].to_numpy() for c in ("open", "high", "low"))
        a, t = atr(g).to_numpy(), g.index
        start, swap = pd.Timestamp(cfg["start"]), cfg["swap"]
        yrs = (t[-1] - start).days / 365.25
        p = pd.read_csv(ZZ / src / "probs.csv", index_col=0, parse_dates=True).iloc[:, 0].reindex(t).to_numpy()
        cb = circuit_ok(g, 1.0)
        fire = np.isfinite(p) & (p >= THR) & cb
        tr, base = backtest(g, fire, start, swap)
        bar = pd.Series(np.arange(len(g)), index=t)
        s_in = bar.reindex(tr.t_in).to_numpy()
        j_out = bar.reindex(tr.t_out).to_numpy()
        brisk = TRAIL * a[s_in - 1]                               # base risk in price
        e_in = o[s_in] + SPREAD
        hold_d = (tr.t_out - tr.t_in).dt.total_seconds().to_numpy() / 86400
        stop_px = tr.R.to_numpy() * brisk + e_in - e_in * swap * hold_d / 365   # undo the swap term
        # sanity: the recovered stop price must sit inside the stop bar's range
        assert np.all(stop_px[tr.exit == "stop"] <= h[j_out[tr.exit == "stop"]] + 1e-6)
        assert np.all(stop_px[tr.exit == "stop"] >= l[j_out[tr.exit == "stop"]] - 1e-6)
        losers = np.where((tr.exit == "stop").to_numpy() & (tr.R < 0).to_numpy())[0]
        mid = start + (t[-1] - start) / 2
        b_sh, b_cg, b_dd = shape(base, yrs)
        b_h1, b_h2 = sharpe(base[base.index < mid]), sharpe(base[base.index >= mid])

        say("=" * 124)
        say(f"{src.upper()}  {start:%Y-%m-%d} -> {t[-1]:%Y-%m-%d} ({yrs:.1f}y)  BASE variant B + breaker: "
            f"{len(tr)} trades, {len(losers)} losing stop-outs ({100 * len(losers) / len(tr):.0f}%) trigger a recovery")
        say(f"  base: Sharpe {b_sh:+.3f}  CAGR {100 * b_cg:+.2f}%  maxDD {100 * b_dd:.2f}%  "
            f"halves {b_h1:+.3f} / {b_h2:+.3f}   expectancy {tr.R.mean():+.3f}R")
        say("=" * 124)
        say(f"  {'k':>4} {'mode':<9}{'fill':>4}{'cb':>4}{'n':>5}{'win%':>6}{'ownR':>8}{'null p':>8}"
            f"{'Sharpe':>8}{'dSh':>8}{'dSh h1':>8}{'dSh h2':>8}{'t_vm':>7}{'CAGR':>8}{'maxDD':>8}"
            f"{'worst':>7}{'PASS':>6}")

        # random-entry pool, shared by all cells of this source
        win = np.where((t >= start) & np.isfinite(a))[0]
        win = win[(win > 1) & (win < len(g) - 2)]
        pool = RNG.choice(win, size=N_POOL, replace=False)

        for k, mode, fill, use_cb in itertools.product(KS, MODES, FILLS, CBS):
            recs = []
            for i in losers:
                j = j_out[i]
                dec = j - 1 if fill == "A" else j                # bar whose close is known at decision
                if use_cb and not cb[dec]:
                    continue
                px = stop_px[i]                                   # the price the original was stopped at
                loss = -tr.R.iat[i] * brisk[i]                    # original loss in price (per base unit)
                ar = a[j - 1] if fill == "A" else a[j]
                tp_atr = (loss / M) / ar if mode == "TRAIL+TP" else None
                r = rec_trade(o, h, l, a, t, j, px, fill, k, tp_atr, swap)
                if r is None:
                    continue
                ti, to, pnl, rrisk, why = r
                recs.append((ti, to, pnl / rrisk, M * pnl / brisk[i], tr.R.iat[i], why, tp_atr))
            if len(recs) < 20:
                continue
            rc = pd.DataFrame(recs, columns=["t_in", "t_out", "ownR", "baseR", "origR", "exit", "tp_atr"])

            # random-timing control: same shape at random bars (fill-B semantics, TP from the real distribution)
            tps = rc.tp_atr.to_numpy()
            null_r = np.empty(N_POOL)
            for q, b in enumerate(pool):
                tpa = tps[q % len(tps)] if mode == "TRAIL+TP" else None
                rr = rec_trade(o, h, l, a, t, b - 1, None, "B", k, tpa, swap)
                null_r[q] = rr[2] / rr[3] if rr else np.nan
            null_r = null_r[np.isfinite(null_r)]
            means = np.array([RNG.choice(null_r, len(rc)).mean() for _ in range(N_NULL)])
            pct = float((means < rc.ownR.mean()).mean())

            book = base.add(to_daily(pd.Series(RISK * rc.baseR.to_numpy(), index=rc.t_out), start, t[-1]),
                            fill_value=0.0)
            sh, cg, dd = shape(book, yrs)
            h1, h2 = sharpe(book[book.index < mid]), sharpe(book[book.index >= mid])
            vm = book * (base.std() / book.std())
            dlt = vm - base
            t_vm = float(dlt.mean() / dlt.std() * np.sqrt(len(dlt))) if dlt.std() > 0 else 0.0
            worst = float((rc.origR + rc.baseR).min())
            ok = pct >= 0.95 and h1 > b_h1 and h2 > b_h2 and t_vm >= 2
            say(f"  {k:>4.1f} {mode:<9}{fill:>4}{('on' if use_cb else '-'):>4}{len(rc):>5}"
                f"{100 * (rc.ownR > 0).mean():>6.1f}{rc.ownR.mean():>+8.3f}{1 - pct:>8.3f}"
                f"{sh:>+8.3f}{sh - b_sh:>+8.3f}{h1 - b_h1:>+8.3f}{h2 - b_h2:>+8.3f}{t_vm:>+7.2f}"
                f"{100 * cg:>+7.2f}%{100 * dd:>7.2f}%{worst:>+7.2f}{('YES' if ok else 'no'):>6}")
            rows.append(dict(src=src, k=k, mode=mode, fill=fill, cb=use_cb, n=len(rc),
                             win=(rc.ownR > 0).mean(), own_R=rc.ownR.mean(), null_mean=null_r.mean(),
                             null_p=1 - pct, sharpe=sh, base_sharpe=b_sh, d_sh_h1=h1 - b_h1,
                             d_sh_h2=h2 - b_h2, t_vm=t_vm, cagr=cg, base_cagr=b_cg, maxdd=dd, base_dd=b_dd,
                             worst_pair_R=worst, passed=ok,
                             exits=str(rc.exit.value_counts().to_dict())))
        say(f"  random-entry pool own-R mean (k-shape, no TP) is in the CSV column null_mean")
        say()

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "cells.csv", index=False)
    both = d.pivot_table(index=["k", "mode", "fill", "cb"], columns="src", values="passed").fillna(False)
    n_pass = int((both.all(axis=1)).sum()) if len(both.columns) == 2 else 0
    say(f"CELLS PASSING THE GATE IN BOTH SOURCES: {n_pass} / {len(both)}")
    say(f"wrote {OUT / 'cells.csv'}")
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
