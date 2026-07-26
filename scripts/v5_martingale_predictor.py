"""GAMBLING BOT — capped loss-recovery martingale on the turning-point predictor.

Objective is NOT Sharpe. Maximise return on a $100,000 account subject to a HARD
10% drawdown wall ($90,000 floor, never breached). The bot bets from the smallest
lot, escalates on losses to recover them, resets to the smallest lot on a win, and
takes DIRECTION from the existing bottom detector (scripts/v5_xau_turning_ml.py).

Governing law (V5_FINDINGS §2, memory xau-nextbar-winrate-martingale, proven three
times): a martingale is an edge LEVER, not an edge CREATOR. On a positive-edge base
it multiplied returns ~25x with bounded DD and 0% bust; on a negative-edge base it
busted 100% of paths. So the whole outcome reduces to ONE measurable quantity —
does the detector's directional bet clear its breakeven win rate AFTER real spread?
That is Stage A (the gate); Stage B only matters if the gate opens.

  Stage A  predictor -> directional long bets (ATR stop/target) -> net-return
           stream r, win rate p, payoff b, breakeven p*=1/(1+b), loss streaks.
  Stage B  feed r to the capped martingale (reuse v5_xau_fade_martingale.simulate),
           equity0=100k, floor=90k, block-bootstrap -> P(ruin at -10%).
  Stage C  frontier over (engine x base x K): P(ruin), median return, maxDD.

Costs floored at the live FTMO gold quote (never the 10x-understated CSV column) —
see the MANDATORY CONTROLS block in V5_FINDINGS.md.

    python scripts/v5_martingale_predictor.py --tf H1
    python scripts/v5_martingale_predictor.py --tf H4 --rr 2.0
    python scripts/v5_martingale_predictor.py --sanity      # engine wiring check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# --- reuse the martingale engine + ruin model verbatim (do not re-implement) ---
from v5_xau_fade_martingale import simulate, stats, show      # noqa: E402
# --- reuse the predictor's causal features + ZigZag ground truth ---------------
from v5_xau_turning_ml import (load, atr, rsi, zigzag_swings,  # noqa: E402,F401
                               label_near, features)

# Live FTMO gold spread, measured 2026-07-24 (V5_FINDINGS MANDATORY CONTROLS §1).
XAU_LIVE_BP = 1.11
SLIP_BP = 0.5


# ---------------------------------------------------------------------------
# STAGE A — turn the predictor into a sequential directional-bet return stream
# ---------------------------------------------------------------------------
def predictor_flags(d, order, theta_mult, tol):
    """OOS long-flag probabilities from the bottom detector (last 30%, causal).

    Reproduces v5_xau_turning_ml.supervised() but RETURNS the aligned proba so we
    can trade it, instead of only printing precision. Train first 70% time-ordered.
    """
    theta = theta_mult * atr(d)
    _sells, buys = zigzag_swings(d, order, theta)
    y = label_near(buys, len(d), tol)               # 1 = bar within +/-tol of a bottom
    f = features(d)
    X = f.values
    ok = ~np.isnan(X).any(axis=1)
    Xok, yok, idx = X[ok], y[ok], np.where(ok)[0]
    split = int(len(Xok) * 0.7)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         max_depth=4, l2_regularization=1.0,
                                         validation_fraction=0.15, random_state=7)
    clf.fit(Xok[:split], yok[:split])
    proba = np.full(len(d), np.nan)
    proba[idx[split:]] = clf.predict_proba(Xok[split:])[:, 1]
    oos_start = idx[split]
    return proba, oos_start, len(buys)


def bet_stream(d, proba, oos_start, thr, stop_atr, rr, max_hold):
    """Sequential LONG bets: enter next open on a flag, ATR stop / R*stop target /
    timeout. NON-OVERLAPPING (a martingale trades one bet at a time), so the next
    flag is only taken after the current bet has closed. Returns net-return array.
    """
    a = atr(d).values
    op, hi, lo, cl = d.open.values, d.high.values, d.low.values, d.close.values
    cost = (XAU_LIVE_BP / 2 + SLIP_BP) / 1e4          # one-way fraction; charged x2
    n = len(d)
    r, hold_bars, wins_px = [], [], []
    i = oos_start
    while i < n - 1:
        if np.isnan(proba[i]) or proba[i] < thr or not np.isfinite(a[i]) or a[i] <= 0:
            i += 1
            continue
        entry = op[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            i += 1
            continue
        stop_d = stop_atr * a[i]
        stop_px, tgt_px = entry - stop_d, entry + rr * stop_d
        exit_px, j = cl[min(i + max_hold, n - 1)], min(i + max_hold, n - 1)
        for k in range(i + 1, min(i + 1 + max_hold, n)):
            if lo[k] <= stop_px:                      # stop checked first (conservative)
                exit_px, j = stop_px, k
                break
            if hi[k] >= tgt_px:
                exit_px, j = tgt_px, k
                break
            exit_px, j = cl[k], k
        net = (exit_px - entry) / entry - 2 * cost    # long, round-trip cost
        r.append(net)
        hold_bars.append(j - i)
        if net > 0:
            wins_px.append(net)
        i = j + 1                                     # sequential: resume after close
    return np.array(r), np.array(hold_bars), np.array(wins_px)


def consec_losses(r):
    """Distribution of consecutive-loss run lengths — what actually kills a
    martingale (gold's losses cluster, so this is far worse than iid)."""
    runs, c = [], 0
    for x in r:
        if x <= 0:
            c += 1
        elif c:
            runs.append(c)
            c = 0
    if c:
        runs.append(c)
    runs = np.array(runs) if runs else np.array([0])
    return {k: int((runs >= k).sum()) for k in (2, 4, 6, 8, 10, 12)}


def buy_hold(d, oos_start):
    px = d.close.values[oos_start:]
    yrs = len(px) / (252 * (24 if len(d) > 40000 else 6))   # rough bars/yr by tf
    return px[-1] / px[0] - 1.0, yrs


def gate_report(r, wins_px, streaks, bh):
    p = float((r > 0).mean())
    win = r[r > 0].mean() if (r > 0).any() else 0.0
    loss = -r[r <= 0].mean() if (r <= 0).any() else 1e-9
    b = win / loss if loss > 0 else np.inf
    pstar = 1.0 / (1.0 + b)
    se = np.sqrt(p * (1 - p) / len(r)) if len(r) else np.nan
    print("=" * 84)
    print("STAGE A — THE GATE: does the predictor clear breakeven after real cost?")
    print("=" * 84)
    print(f"  bets {len(r)}   net win rate p = {p*100:.1f}% (+-{se*100:.1f})   "
          f"avg win {win*100:+.2f}%  avg loss {-loss*100:+.2f}%")
    print(f"  payoff b = {b:.2f}   breakeven p* = 1/(1+b) = {pstar*100:.1f}%   "
          f"=> EDGE {'POSITIVE' if p > pstar else 'NEGATIVE/BREAKEVEN'} "
          f"(p - p* = {(p-pstar)*100:+.1f} pts)")
    print(f"  total flat P&L (sum r) = {r.sum()*100:+.1f}%   "
          f"buy&hold gold same window = {bh*100:+.1f}%")
    print(f"  consecutive-loss streaks >=k : {streaks}")
    return p, b, float(win) if win > 0 else 0.01, pstar


# ---------------------------------------------------------------------------
# STAGE B/C — capped martingale against the hard $90k floor
# ---------------------------------------------------------------------------
def frontier(r, w, equity0, floor, engines, bases, Ks, cap, n_mc, block, seed=7):
    rng = np.random.default_rng(seed)
    N = len(r)
    # pre-build block-bootstrap index sets once, shared across cells
    seqs = []
    for _ in range(n_mc):
        idx = []
        while len(idx) < N:
            s0 = rng.integers(0, N)
            idx.extend(range(s0, s0 + block))
        seqs.append(r[np.array(idx[:N]) % N])
    print("\n" + "=" * 84)
    print(f"STAGE C — RUIN FRONTIER  ({n_mc} block-bootstrap paths, {block}-bet blocks, "
          f"hard floor ${floor:,.0f})")
    print("=" * 84)
    print(f"{'engine':9s} {'base$':>8s} {'K':>3s} {'P(ruin)':>8s} {'medRet%':>8s} "
          f"{'p05Ret%':>8s} {'p95Ret%':>8s} {'medMaxDD%':>10s} {'medSurv':>8s}")
    for eng in engines:
        for base in bases:
            for K in Ks:
                ends, dds, survs, busts = [], [], [], 0
                for seq in seqs:
                    c, _stk, bust = simulate(seq, eng, base, equity0, floor, K, cap, w)
                    ends.append(c[-1] / equity0 - 1.0)
                    eqs = pd.Series(c)
                    dds.append(float((eqs / eqs.cummax() - 1).min() * 100))
                    survs.append(bust if bust >= 0 else N)
                    busts += bust >= 0
                ends = np.array(ends) * 100
                print(f"{eng:9s} {base:8,.0f} {K:3d} {busts/n_mc*100:7.1f}% "
                      f"{np.median(ends):8.1f} {np.percentile(ends,5):8.1f} "
                      f"{np.percentile(ends,95):8.1f} {np.median(dds):10.2f} "
                      f"{int(np.median(survs)):8d}")
        print("-" * 84)


def sanity():
    """Engine wiring check: a +EV base must survive, a -EV base must ruin."""
    print("=" * 84 + "\nSANITY — engine wiring (synthetic iid streams)\n" + "=" * 84)
    rng = np.random.default_rng(1)
    for p, tag in ((0.55, "positive-edge p=0.55 b=1  -> expect ~0% ruin, +return"),
                   (0.40, "negative-edge p=0.40 b=1  -> expect ~100% ruin")):
        r = np.where(rng.random(4000) < p, 0.01, -0.01)
        print(f"\n{tag}")
        for eng in ("flat", "double4", "recover4"):
            c, stk, bust = simulate(r, eng, 20_000.0, 100_000.0, 90_000.0, 5, 50.0, 0.01)
            s = stats(c, stk, r, 20_000.0, 100_000.0, f"{eng}", bust)
            print(f"  {eng:9s} end ${s['end']:>9,.0f}  ret {s['ret']*100:+6.1f}%  "
                  f"maxDD {s['maxdd']:5.1f}%  {'BUST@'+str(bust) if bust>=0 else 'survived'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf", default="H1")
    ap.add_argument("--order", type=int, default=5)
    ap.add_argument("--theta", type=float, default=1.5)
    ap.add_argument("--tol", type=int, default=3)
    ap.add_argument("--thr", type=float, default=0.5, help="proba flag threshold")
    ap.add_argument("--stop-atr", type=float, default=1.5)
    ap.add_argument("--rr", type=float, default=1.5, help="target = rr * stop")
    ap.add_argument("--max-hold", type=int, default=24)
    ap.add_argument("--mc", type=int, default=500)
    ap.add_argument("--block", type=int, default=100)
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()

    if args.sanity:
        sanity()
        return

    d = load(args.tf)
    proba, oos_start, n_bottoms = predictor_flags(d, args.order, args.theta, args.tol)
    print(f"[martingale-predictor] tf={args.tf} bars={len(d)}  bottoms={n_bottoms}  "
          f"OOS from {d.index[oos_start].date()} ({len(d)-oos_start} bars)")
    r, holds, wins_px = bet_stream(d, proba, oos_start, args.thr,
                                   args.stop_atr, args.rr, args.max_hold)
    if len(r) < 30:
        print(f"  too few bets ({len(r)}) at thr={args.thr}; lower --thr.")
        return
    bh, _yrs = buy_hold(d, oos_start)
    streaks = consec_losses(r)
    print(f"  stop {args.stop_atr}xATR  target {args.rr}xstop  hold<={args.max_hold} "
          f"bars  avg hold {holds.mean():.1f}\n")
    p, b, w, pstar = gate_report(r, wins_px, streaks, bh)

    # Stage B headline + Stage C frontier. base = per-bet NOTIONAL $; a base bet
    # loses ~base*avg_loss on a stop. Bases chosen so the worst K-ladder stays
    # inside the $10k budget (doubling: base*(2^K-1)).
    equity0, floor = 100_000.0, 90_000.0
    print("\n" + "=" * 84)
    print("STAGE B — headline engines at a mid base (single historical path)")
    print("=" * 84)
    for eng in ("flat", "double4", "recover4"):
        c, stk, bust = simulate(r, eng, 40_000.0, equity0, floor, 5, 50.0, w)
        show(stats(c, stk, r, 40_000.0, equity0, f"{eng}  base$40k K5", bust))

    frontier(r, w, equity0, floor,
             engines=("flat", "double4", "recover4"),
             bases=(20_000.0, 40_000.0, 80_000.0),
             Ks=(3, 4, 5, 6), cap=50.0, n_mc=args.mc, block=args.block)

    print("\nGATE RULE: if p <= p* the base bet is net-negative and every martingale "
          "ruins — read Stage A first. Frontier 'survivable' cell needs P(ruin)~0 "
          "AND +median return AND medMaxDD < 10%.")


if __name__ == "__main__":
    main()
