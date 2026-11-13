#!/usr/bin/env python3
"""Elliott Wave structure over the repo's own ZigZag pivots — does the RULE SET add anything?

USER, 2026-11-13: "Something to try Elliot Waves to improve Zigzag, do research and backtest",
and then: borrow ideas from arXiv 2102.01499 (Qi/Khushi/Poon, "Event-Driven LSTM For Forex Price
Prediction") without replicating it.

WHAT ELLIOTT ACTUALLY ASSERTS, reduced to something testable. A trend unfolds as a 5-wave
impulse. Three rules are inviolable (Elliott 1938; every modern source agrees):
    R1  wave 2 never retraces more than 100% of wave 1
    R2  wave 3 is never the shortest of waves 1, 3, 5
    R3  wave 4 does not overlap wave 1's price territory (diagonals excepted)
Only R1 is knowable at the moment you would trade (R2 and R3 need waves 4 and 5 to exist), so R1
plus the Fibonacci retracement guideline (wave 2 holds 0.5-0.618 of wave 1) IS the tradeable
claim: after a 1-2, wave 3 is the big one.

WHY THIS IS NOT `pivot-detection-closed-on-mechanism` REOPENED. That closure killed DETECTING a
pivot. Elliott is not a pivot detector — it is a rule over pivots that are already confirmed. The
question here is strictly: given three confirmed pivots, does the Elliott rule separate the next
leg from the same leg without the rule?

WHAT IS BORROWED FROM 2102.01499, AND WHAT IS NOT.
  BORROWED - its event spine e1 -> e2 -> e3: a ZigZag pivot (e1) is unusable until something
             causal confirms it (e2), and only then do you look for an entry (e3). The paper is
             candid that its own "ZigZag normally requires approximately an additional 40 bars to
             confirm a transition point". That lag is the whole ballgame, so this script MEASURES
             it on gold instead of assuming it.
  BORROWED - event-anchored sample selection: use only windows anchored on a confirmed event
             rather than every bar.
  REJECTED  - its evaluation. The paper scores MSE/RMSE/MAE/MAPE on the PRICE LEVEL and never
             trades it, never charges a spread, and never reports the naive "tomorrow = today"
             baseline that a near-random-walk makes almost unbeatable on MAPE. That is control #8
             (AUC/accuracy is payoff-blind) in its purest form, and this repo has killed that
             pattern repeatedly. Every number below is EV per fire against the base rate and the
             cost, never an error metric.

CONTROL #5 RUNS FIRST. Before any detector: with HINDSIGHT pivots, does an Elliott-valid 1-2 lead
to a bigger wave 3 than a 1-2 that violates R1? If the oracle cannot separate them, no detector
can, and the study stops there.

TWO TAUTOLOGY TRAPS THIS SCRIPT REFUSES TO FALL INTO.
  * entering AT the pivot bar is buying the exact low (`zigzag-oracle-must-enter-at-confirmation`:
    0.97 -> 0.19-0.67 once entry moves to confirmation). Every cell is reported at BOTH.
  * a rule that merely selects "lows in an uptrend" beats random bars on drift alone. So the
    control is not a random bar — it is the SAME event type (a confirmed L-H-L triple) that
    FAILS the Elliott rule.

    python scripts/v5_elliott_wave.py            # both sources
    python scripts/v5_elliott_wave.py cme
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
from scripts.v5_xau_turning_ml import atr, zigzag_swings  # noqa: E402
from scripts.v5_zigzag_long_history import SRC, load  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "elliott"
ZZ_ORDER, ZZ_THETA = 5, 1.5          # the live detector's own pivot geometry
SPREAD = 0.25                        # $/oz, both venues (v5_zigzag_long_history.SPREAD)
RNG = np.random.default_rng(20261113)
N_NULL = 2000


# ----------------------------------------------------------------- pivots and confirmation ---

def pivot_sequence(d: pd.DataFrame, order: int = ZZ_ORDER, theta: float = ZZ_THETA) -> pd.DataFrame:
    """Alternating H/L pivots, oldest first, with the bar each one could FIRST be known on.

    `zigzag_swings` runs argrelextrema over the whole array, so its pivots are HINDSIGHT. A pivot
    at bar i is not knowable until (a) `order` bars have printed to its right without breaking it,
    and (b) price has moved `theta` away from it — which is the same two conditions the extractor
    itself applies. `conf` is the first bar satisfying both; everything causal keys off `conf`,
    never off `bar`.
    """
    a = theta * atr(d)
    sells, buys = zigzag_swings(d, order, a)
    piv = sorted([(int(i), "H") for i in sells] + [(int(i), "L") for i in buys])
    hi, lo, n = d.high.values, d.low.values, len(d)
    av = a.values

    rows = []
    for i, k in piv:
        p = hi[i] if k == "H" else lo[i]
        j = i + order                                  # (a) the extremum window has closed
        need = av[i]
        while j < n:                                   # (b) price has travelled theta away
            moved = (p - lo[j]) if k == "H" else (hi[j] - p)
            if moved >= need:
                break
            j += 1
        if j < n:
            rows.append((i, k, p, j))
    return pd.DataFrame(rows, columns=["bar", "kind", "price", "conf"])


def triples(pv: pd.DataFrame) -> pd.DataFrame:
    """Every consecutive L-H-L triple: a candidate wave 1 (L0->H1) and wave 2 (H1->L2).

    This is the UNIVERSE. The Elliott rule partitions it; it does not create it. Keeping the
    universe fixed is what makes the comparison a test of the rule rather than of exposure.
    """
    k = pv.kind.values
    out = []
    for t in range(len(pv) - 3):
        if not (k[t] == "L" and k[t + 1] == "H" and k[t + 2] == "L" and k[t + 3] == "H"):
            continue
        l0, h1, l2, h3 = (pv.iloc[t], pv.iloc[t + 1], pv.iloc[t + 2], pv.iloc[t + 3])
        w1 = h1.price - l0.price
        if w1 <= 0:
            continue
        out.append(dict(
            l0_bar=int(l0.bar), l0_p=l0.price,
            h1_bar=int(h1.bar), h1_p=h1.price,
            l2_bar=int(l2.bar), l2_p=l2.price, l2_conf=int(l2.conf),
            h3_bar=int(h3.bar), h3_p=h3.price, h3_conf=int(h3.conf),
            w1=w1,
            retr=(h1.price - l2.price) / w1,           # wave-2 retracement of wave 1
            w3=h3.price - l2.price,
            r1_ok=bool(l2.price > l0.price),           # RULE 1 — the only rule knowable at entry
        ))
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------ the oracle ---

def base_curve(d: pd.DataFrame) -> callable:
    """base(h) = the mean h-bar return of simply HOLDING gold, over every start in the sample.

    CONTROL #2. A wave-3 leg IS a long gold position, so "beat zero" is meaningless on an asset
    that drifts up: every leg of every kind is positive on average. The only honest question is
    whether an Elliott leg beats a RANDOM window of the same length. Computed exactly (not as
    h x mean-bar-drift) from cumulative log prices, so it costs O(1) per distinct h.
    """
    lc = np.log(d.close.values)
    cs = np.concatenate([[0.0], np.cumsum(lc)])
    n = len(lc)

    def base(h: int) -> float:
        if h <= 0 or h >= n:
            return 0.0
        # mean over i of (lc[i+h] - lc[i]) == (sum lc[h:] - sum lc[:-h]) / (n-h)
        m = ((cs[n] - cs[h]) - (cs[n - h] - cs[0])) / (n - h)
        return float(np.expm1(m))
    return base


def leg_returns(d: pd.DataFrame, tr: pd.DataFrame, entry: str, exit_: str,
                spread: float, swap_yr: float, base=None) -> np.ndarray:
    """Return of the wave-3 leg in %, net of spread and carry.

    entry/exit 'pivot' = at the pivot's own extreme price (TAUTOLOGY: it buys the exact low and
    sells the exact high). entry/exit 'conf' = at the open of the confirmation bar, which is the
    first bar a live system could have acted on.
    """
    o = d.open.values
    n = len(d)
    r, hold, keep = [], [], []
    for pos, (_, t) in enumerate(tr.iterrows()):
        if entry == "pivot":
            ei, ep = int(t.l2_bar), t.l2_p + spread
        else:
            ei = int(t.l2_conf)
            if ei >= n:
                continue
            ep = o[ei] + spread
        if exit_ == "pivot":
            xi, xp = int(t.h3_bar), t.h3_p
        else:
            xi = int(t.h3_conf)
            if xi >= n:
                continue
            xp = o[xi]
        if xi <= ei:
            continue
        h = xi - ei
        ret = (xp / ep - 1.0) + swap_yr * (h / (24 * 365.0))
        if base is not None:                 # EXCESS over holding gold the same number of bars
            ret -= base(h)
        r.append(ret)
        hold.append(h)
        keep.append(pos)                     # WHICH triples survived — masks must use this
    return np.array(r), np.array(hold), np.array(keep, dtype=int)


def tstat(x: np.ndarray) -> float:
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else float("nan")


def describe(tag: str, xh: tuple, allmean: float) -> dict:
    x, h = xh[0], xh[1]
    if len(x) == 0:
        print(f"    {tag:<36} (no events)")
        return {}
    row = dict(cell=tag, n=len(x), ev_pct=100 * x.mean(), med=100 * np.median(x),
               win=float((x > 0).mean()), t=tstat(x), hold=float(np.median(h)),
               vs_all=100 * (x.mean() - allmean))
    print(f"    {tag:<36} n {row['n']:>5}  EXCESS {row['ev_pct']:+7.3f}%  "
          f"med {row['med']:+7.3f}%  win {row['win']:.3f}  t {row['t']:+6.2f}  "
          f"hold {row['hold']:>5.0f}b  vs-all {row['vs_all']:+6.3f}%")
    return row


# ------------------------------------------------------------------------ the search ---
# Every cell searched, declared up front so the null can re-run ALL of it (control #7).
SCALES = ((5, 1.5), (10, 2.5), (20, 4.0))       # small / medium / large wave geometry
MODES = (("pivot", "pivot"), ("conf", "pivot"), ("conf", "conf"))
PARTS = (
    ("R1 holds (Elliott-valid 1-2)", lambda t: t.r1_ok.values),
    ("R1 + wave2 retr [.50,.618]", lambda t: (t.r1_ok & t.retr.between(0.5, 0.618)).values),
    ("R1 + wave2 retr [.382,.786]", lambda t: (t.r1_ok & t.retr.between(0.382, 0.786)).values),
)
MIN_SIDE = 25


def sep_t(y: np.ndarray, m: np.ndarray) -> float:
    a, b = y[m], y[~m]
    if len(a) < MIN_SIDE or len(b) < MIN_SIDE:
        return 0.0
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else 0.0


def maxstat(cells: list, n_null: int = N_NULL) -> tuple:
    """Re-run the WHOLE search under a circular shift of the OUTCOME; take the max |t| each draw.

    `cells` is [(returns, [masks...])] — one entry per (scale x mode) evaluated, each carrying
    every partition tried on it. The rule's labels stay exactly where they are and only the
    outcome moves, which is the null "the structure carries nothing" and nothing else. The side
    is chosen INSIDE the null because the search was free to choose it (|t|, not t). Computed on
    the excess series as searched -- no cost offset is injected into a zero-mean null (3bx-D).
    """
    def best(shift: int | None) -> float:
        out = 0.0
        for y, masks in cells:
            yy = y if shift is None else np.roll(y, shift)
            for m in masks:
                out = max(out, abs(sep_t(yy, m)))
        return out

    obs = best(None)
    nmin = min(len(y) for y, _ in cells)
    if nmin < 60:
        return obs, float("nan"), float("nan")
    null = np.array([best(int(RNG.integers(20, nmin - 20))) for _ in range(n_null)])
    return obs, float(np.percentile(null, 95)), float((null >= obs).mean())


def trend_control(g, tr, ka, xa, sma_n: int = 50) -> None:
    """IS ELLIOTT'S RULE 1 ANYTHING MORE THAN A TREND FILTER?

    R1 says the wave-2 low must hold above the wave-1 start. In plain language that is a HIGHER
    LOW, which is the oldest trend definition there is — and this repo's champion is a trend
    follower that already earns ~1.0 Sharpe doing exactly that. So a positive R1 result proves
    nothing on its own: the question is whether R1 separates INSIDE a trend state it did not
    choose.

    This is `longonly-prior-high-break-lead`'s lesson applied: a matched placebo earned ~60% of
    that study's excess, and the part that survived was the part the placebo could not explain.
    The 2x2 below is the same idea. If R1's edge lives only in the trend-positive column, Elliott
    is a trend filter wearing a costume.
    """
    c = g.close.values
    sma = pd.Series(c).rolling(sma_n).mean().values
    kt = tr.iloc[ka]
    ei = kt.l2_conf.values.clip(max=len(g) - 1) - 1          # known at the entry OPEN
    ok = ei >= sma_n
    trend = np.zeros(len(kt), dtype=bool)
    trend[ok] = c[ei[ok]] > sma[ei[ok]]
    r1 = kt.r1_ok.values

    print(f"\n    TREND CONTROL — is R1 more than 'close > SMA{sma_n}'?")
    print(f"      {'cell':<28}{'n':>6}{'EXCESS':>10}{'t':>8}")
    for tag, m in (("trend UP  & R1 holds", trend & r1),
                   ("trend UP  & R1 violated", trend & ~r1),
                   ("trend DOWN & R1 holds", ~trend & r1),
                   ("trend DOWN & R1 violated", ~trend & ~r1)):
        if m.sum() < MIN_SIDE:
            print(f"      {tag:<28}{m.sum():>6}   (too few)")
            continue
        y = xa[m]
        print(f"      {tag:<28}{m.sum():>6}{100 * y.mean():>9.3f}%{tstat(y):>8.2f}")

    for name, cond in (("inside trend UP", trend), ("inside trend DOWN", ~trend)):
        a, b = xa[cond & r1], xa[cond & ~r1]
        if len(a) >= MIN_SIDE and len(b) >= MIN_SIDE:
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            t_ = (a.mean() - b.mean()) / se if se > 0 else float("nan")
            print(f"      R1 effect {name:<20} {100 * (a.mean() - b.mean()):+7.3f}%  t {t_:+6.2f}")
    # and the mirror question: what does TREND do, ignoring Elliott entirely?
    a, b = xa[trend], xa[~trend]
    if len(a) >= MIN_SIDE and len(b) >= MIN_SIDE:
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        print(f"      TREND effect, Elliott ignored  {100 * (a.mean() - b.mean()):+7.3f}%  "
              f"t {(a.mean() - b.mean()) / se:+6.2f}")


# ------------------------------------------------------- the tradeable version ---
# Everything above is the ORACLE CEILING: the exit "sell at the next pivot high" uses a pivot
# chosen with hindsight. No live system knows which bar will turn out to be H3. This section
# replaces that exit with a bracket a broker can hold, charges the spread and the carry, and
# benchmarks against the SAME bracket on random bars (`zigzag-hit-rate-is-geometry`: a small
# target with a wide stop wins often for ANY entry on a rising asset).
TPS, SLS, HOLDS = (1.0, 2.0, 3.0), (1.0, 2.0, 3.0), (24, 72, 168)


def bracket(o, h, l, a, idx, tp, sl, hold, spread, swap_bar):
    """Long at the next open + spread; first touch of TP/SL wins; else exit at `hold`. Returns R."""
    n = len(o)
    out = np.empty(len(idx))
    for k in range(len(idx)):
        i = int(idx[k])
        st = i + 1
        if st >= n - 1 or not np.isfinite(a[i]) or a[i] <= 0:
            out[k] = np.nan
            continue
        e = o[st] + spread
        up, dn = e + tp * a[i], e - sl * a[i]
        end = min(st + hold, n - 1)
        px, held = o[end], end - st
        for j in range(st, end):
            if l[j] <= dn:
                px, held = min(dn, o[j]), j - st
                break
            if h[j] >= up:
                px, held = max(up, o[j]), j - st
                break
        out[k] = (px - e + swap_bar * held * e) / (sl * a[i])
    return out


def tradeable(g, tr, ka, src, swap, order, allrows) -> list:
    """R1 vs not-R1 under a real bracket, against the same bracket on random bars."""
    o, h, l = g.open.values, g.high.values, g.low.values
    av = atr(g).values
    swap_bar = swap / (24 * 365.0)
    kt = tr.iloc[ka]
    ei = kt.l2_conf.values                                  # entry decided at confirmation
    ok = ei < len(g) - 2
    ei, r1 = ei[ok], kt.r1_ok.values[ok]
    rnd = RNG.choice(np.arange(200, len(g) - 200), size=min(4000, len(g) - 400), replace=False)

    print(f"\n    TRADEABLE — bracket exit, spread ${SPREAD}, swap {swap:+.0%}/yr, "
          f"{len(ei)} entries ({100 * r1.mean():.0f}% R1-valid)")
    print(f"      {'tp/sl/hold':<14}{'R1 EV(R)':>10}{'~R1 EV(R)':>11}{'sep t':>8}"
          f"{'random':>9}{'R1-rand':>9}")
    out = []
    for tp in TPS:
        for sl in SLS:
            for hold in HOLDS:
                y = bracket(o, h, l, av, ei, tp, sl, hold, SPREAD, swap_bar)
                rr = bracket(o, h, l, av, rnd, tp, sl, hold, SPREAD, swap_bar)
                m = np.isfinite(y)
                ya, yb = y[m & r1], y[m & ~r1]
                rr = rr[np.isfinite(rr)]
                if len(ya) < MIN_SIDE or len(yb) < MIN_SIDE:
                    continue
                se = np.sqrt(ya.var(ddof=1) / len(ya) + yb.var(ddof=1) / len(yb))
                t_ = (ya.mean() - yb.mean()) / se if se > 0 else 0.0
                print(f"      {tp:.0f}/{sl:.0f}/{hold:<9d}{ya.mean():>10.4f}{yb.mean():>11.4f}"
                      f"{t_:>8.2f}{rr.mean():>9.4f}{ya.mean() - rr.mean():>9.4f}")
                out.append((y[m], r1[m], rr.mean()))
                allrows.append(dict(src=src, order=order, cell="TRADE", tp=tp, sl=sl,
                                    hold=hold, n=int(m.sum()), ev_r1=ya.mean(),
                                    ev_not=yb.mean(), sep_t=t_, ev_random=rr.mean(),
                                    vs_random=ya.mean() - rr.mean()))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    srcs = [a for a in sys.argv[1:] if a in SRC] or list(SRC)
    allrows = []

    for src in srcs:
        cfg = SRC[src]
        g = load(src, cfg["gold"])
        g = g[g.index >= cfg["start"]]
        swap = cfg["swap"]
        base = base_curve(g)
        print("=" * 104)
        print(f"{src.upper()}  {cfg['gold']}  {g.index[0]:%Y-%m-%d} -> {g.index[-1]:%Y-%m-%d}  "
              f"{len(g):,} H1 bars   spread ${SPREAD}  swap {swap:+.0%}/yr")
        print("=" * 104)

        cells, printed = [], []
        for order, theta in SCALES:
            pv = pivot_sequence(g, order, theta)
            tr = triples(pv)
            if len(tr) < 100:
                print(f"\n  scale order={order} theta={theta}: only {len(tr)} triples — skipped")
                continue
            lag = (pv.conf - pv.bar).values
            gap = np.median((g.open.values[tr.l2_conf.values.clip(max=len(g) - 1)] - tr.l2_p)
                            / tr.w1)
            print(f"\n  {'-' * 100}")
            print(f"  SCALE order={order} theta={theta}xATR   {len(pv):,} pivots, "
                  f"{len(tr):,} L-H-L triples   R1 holds on {100 * tr.r1_ok.mean():.0f}%")
            print(f"  confirmation lag: median {np.median(lag):.0f} bars, p90 "
                  f"{np.percentile(lag, 90):.0f}  ->  by then price has already retraced "
                  f"{100 * gap:.0f}% of wave 1 back up (median)")

            for entry, exit_ in MODES:
                xa, ha, ka = leg_returns(g, tr, entry, exit_, SPREAD, swap, base)
                if len(xa) < 100:
                    continue
                kt = tr.iloc[ka]                      # ALIGNED: only the triples that produced a leg
                masks = [f(kt) for _, f in PARTS]
                masks = [m for m in masks if MIN_SIDE <= m.sum() <= len(m) - MIN_SIDE]
                if not masks:
                    continue
                cells.append((xa, masks))
                am = xa.mean()
                run_control = (entry, exit_) == ("conf", "conf")
                print(f"\n    entry {entry:<5} exit {exit_:<5} | ALL {len(xa):>5} legs  "
                      f"EXCESS {100 * am:+.3f}%  win {(xa > 0).mean():.3f}  "
                      f"t {tstat(xa):+.2f}  hold {np.median(ha):.0f}b")
                for (tag, f), m in zip(PARTS, [f(kt) for _, f in PARTS]):
                    if not (MIN_SIDE <= m.sum() <= len(m) - MIN_SIDE):
                        continue
                    row = describe(tag, (xa[m], ha[m]), am)
                    t_ = sep_t(xa, m)
                    print(f"      {'^ separation vs the rest':<34} t {t_:+6.2f}")
                    if row:
                        row.update(src=src, order=order, theta=theta, entry=entry,
                                   exit=exit_, sep_t=t_)
                        allrows.append(row)
                        printed.append((f"{src} o{order} {entry}/{exit_} {tag}", t_))
                if run_control:
                    trend_control(g, tr, ka, xa)
                    if order in (5, 10):
                        tcells = tradeable(g, tr, ka, src, swap, order, allrows)
                        if tcells:
                            tobs, tp95, tpv = maxstat([(y, [m]) for y, m, _ in tcells])
                            print(f"      max-stat over the {len(tcells)} bracket cells: "
                                  f"best|t| {tobs:.2f}  null p95 {tp95:.2f}  p {tpv:.4f}  -> "
                                  f"{'PASSES' if tpv < 0.05 else 'FAILS'}")
                            allrows.append(dict(src=src, order=order, cell="TRADE_MAXSTAT",
                                                t=tobs, maxstat_p95=tp95, maxstat_p=tpv))

        if cells:
            obs, p95, pval = maxstat(cells)
            ncell = sum(len(m) for _, m in cells)
            print(f"\n  {'=' * 100}")
            print(f"  MAX-STAT NULL over the WHOLE search — {len(cells)} scale x mode "
                  f"combinations, {ncell} partitions tried")
            print(f"    best |t| observed {obs:5.2f}   null p95 {p95:5.2f}   "
                  f"p {pval:.4f}   ->  {'PASSES' if pval < 0.05 else 'FAILS'}")
            best_cell = max(printed, key=lambda z: abs(z[1])) if printed else None
            if best_cell:
                print(f"    the arg-max was: {best_cell[0]}  (t {best_cell[1]:+.2f})")
            allrows.append(dict(src=src, cell="MAXSTAT", n=ncell, t=obs,
                                maxstat_p95=p95, maxstat_p=pval))

    if allrows:
        pd.DataFrame(allrows).to_csv(OUT / "oracle_ceiling.csv", index=False)
        print(f"\nwrote {OUT / 'oracle_ceiling.csv'}")


if __name__ == "__main__":
    main()
