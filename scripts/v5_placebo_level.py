#!/usr/bin/env python3
"""OCTOBER D21 — the PLACEBO LEVEL control. Is the prior-day high special, or is any level?

Pre-registered in data/v5_runs/october/PREREGISTRATION.md (D21 addendum). Design in OCTOBER.MD §4b.

WHY THIS RUNS BEFORE EVERYTHING ELSE IN TIER D. §3ci split the long-only prior-high break's
+10.161bp into **+2.807bp free drift / +4.729bp FILL ADVANTAGE (t +7.45) / ~+2.6bp genuine excess**,
and the fill advantage is 47% of the headline. But it is measured **conditional on the level being
crossed**, and *any* level that gets crossed has price travelling past it by construction: the bar
that crosses a level closes somewhere beyond it, or it would not have crossed. §3ci controlled for
gold's UNCONDITIONAL drift over the same remaining bars. It did not control for
**conditional-on-crossing** drift, which is a different and much harder benchmark.

That omission is the exact shape of five kills already in this file -- window extremes inverting by
window return sign, extreme-timing patterns being drift, the skipped sample, the dip detector
selecting losers, volume zones' premise running backwards. Each one looked like a property of a
chosen location and turned out to be a property of the selection that found it.

THE CONTROL. For every day, build a placebo level at the same distance above the day's open **in
ATR units** -- the distance drawn from a DIFFERENT day by circular shift, so the distribution of
"how hard is this level to reach" is preserved while any link to yesterday's high is destroyed.
ATR units rather than percent because the prior-day high's distance is correlated with yesterday's
range and therefore with today's volatility; a raw-percent permutation would hand easy levels to
volatile days and silently change the fill rate. Both are reported.

THREE OUTCOMES, ALL INFORMATIVE, PRE-REGISTERED SO NEITHER IS ROUNDED UP:
  1. placebo fill advantage ~= real AND placebo excess-over-drift ~= real
     -> the prior-day high is DECORATION. What survives is generic resting-stop breakout geometry,
        which is a different and far more multiplicity-exposed question than the one §3ch filed.
  2. placebo fill advantage ~= real BUT placebo excess-over-drift < real
     -> the fill capture is a tautology of crossing, and the LEVEL still carries information. The
        strategy survives with a corrected story and a smaller headline.
  3. both materially lower -> the prior-day high is special on both counts. Strongest case.

CONTROL CELL WITH A KNOWN ANSWER. Arm (a) below reproduces §3ch's first-break framing and must
return n ~= 1,446 and a fill advantage of ~ +4.729bp. If it does not, nothing further is read.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "v5_runs" / "october"
RT_BP = 0.504 * 1.5           # 0.756bp, the live round trip
N_PERM = 200                  # permutation draws for the placebo null
MIN_SHIFT = 20                # circular shift >= 20 days, per the maxstat_null design
RNG = np.random.default_rng(20261030)


def load(tf: str) -> pd.DataFrame:
    """XAUUSD M15 and H1 are the CLEAN pair (§3bb agreement 1.00). H4 and M30 are spliced."""
    d = pd.read_csv(ROOT / f"data/XAUUSD_{tf}_long.csv", parse_dates=["time"], index_col="time")
    return d[~d.index.duplicated(keep="last")].sort_index()


class Days:
    """Day-sliced OHLC plus the unconditional drift curve indexed by BARS LEFT."""

    def __init__(self, d: pd.DataFrame):
        day = d.index.normalize()
        self.o = d["open"].to_numpy(float)
        self.h = d["high"].to_numpy(float)
        self.l = d["low"].to_numpy(float)
        self.c = d["close"].to_numpy(float)
        nd = day.to_numpy()
        self.slices = []
        start = 0
        for i in range(1, len(d) + 1):
            if i == len(d) or nd[i] != nd[start]:
                self.slices.append((start, i))
                start = i
        # prior-day high / low / true range, strictly from COMPLETED days
        hi = np.array([self.h[a:b].max() for a, b in self.slices])
        lo = np.array([self.l[a:b].min() for a, b in self.slices])
        cl = np.array([self.c[b - 1] for a, b in self.slices])
        self.pdh = np.r_[np.nan, hi[:-1]]
        self.pdl = np.r_[np.nan, lo[:-1]]
        pc = np.r_[np.nan, cl[:-1]]
        tr = np.maximum(hi - lo, np.maximum(np.abs(hi - pc), np.abs(lo - pc)))
        self.atr = pd.Series(np.r_[np.nan, tr[:-1]]).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
        self.open = np.array([self.o[a] for a, b in self.slices])

        # ---- UNCONDITIONAL drift by bars-left: mean of close_last/close_j - 1 over every day ----
        bl, rr = [], []
        for a, b in self.slices:
            n = b - a
            if n < 2:
                continue
            last = self.c[b - 1]
            for j in range(n - 1):
                bl.append(n - 1 - j)
                rr.append(last / self.c[a + j] - 1.0)
        s = pd.Series(rr).groupby(pd.Series(bl)).mean()
        self.drift = s.to_dict()           # bars_left -> mean return (fraction)

    def cross_up(self, di: int, level: float):
        """First bar of day `di` whose HIGH exceeds `level`. Returns (k, bars_left) or None."""
        a, b = self.slices[di]
        n = b - a
        if n < 2 or not np.isfinite(level):
            return None
        w = np.flatnonzero(self.h[a:b] > level)
        if not len(w):
            return None
        k = int(w[0])
        # A cross on the LAST bar is kept, with zero bars left: the trade is entered at the level
        # and exited at that same bar's close, which is a real -- if very short -- trade. Dropping
        # them moved §3ch's control cell by 59 trades and 0.45bp, so they are counted, not binned.
        return k, n - 1 - k

    def cross_dn(self, di: int, level: float):
        """First bar of day `di` whose LOW falls below `level`, or None."""
        a, b = self.slices[di]
        if (b - a) < 2 or not np.isfinite(level):
            return None
        w = np.flatnonzero(self.l[a:b] < level)
        return int(w[0]) if len(w) else None


def arm(D: Days, levels: np.ndarray, eligible: np.ndarray,
        cancel_below: np.ndarray | None = None) -> dict:
    """Score a resting buy-stop at `levels[di]` for every eligible day.

    `cancel_below` is §3ch's first-break condition made operational: the order rests at the level
    and is CANCELLED if the downside level is taken out first. That is implementable in real time
    -- it is not hindsight -- and it is worth ~2.3bp, so the placebo has to be given the same rule
    or the comparison is rigged.
    """
    lv, cl, dr, bars = [], [], [], []
    for di in np.flatnonzero(eligible):
        L = levels[di]
        hit = D.cross_up(di, L)
        if hit is None:
            continue
        k, left = hit
        if cancel_below is not None:
            kd = D.cross_dn(di, cancel_below[di])
            if kd is not None and kd < k:
                continue
        a, b = D.slices[di]
        last = D.c[b - 1]
        lv.append((last / L - 1.0) * 1e4 - RT_BP)                    # filled AT the level
        cl.append((last / D.c[a + k] - 1.0) * 1e4 - RT_BP)           # filled at the bar's CLOSE
        dr.append(D.drift.get(left, np.nan) * 1e4)                   # matched unconditional drift
        bars.append(k)
    lv, cl, dr = np.array(lv), np.array(cl), np.array(dr)
    n = len(lv)
    if n < 50:
        return dict(n=n)

    def t(x):
        x = x[np.isfinite(x)]
        return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 1 else np.nan

    fill_adv = lv - cl
    exc_lv, exc_cl = lv - dr, cl - dr
    return dict(n=n, fill_rate=n / max(int(eligible.sum()), 1), med_bar=float(np.median(bars)),
                level_bp=float(lv.mean()), level_t=t(lv),
                close_bp=float(cl.mean()), close_t=t(cl),
                drift_bp=float(np.nanmean(dr)),
                fill_adv=float(fill_adv.mean()), fill_adv_t=t(fill_adv),
                exc_level=float(np.nanmean(exc_lv)), exc_level_t=t(exc_lv),
                exc_close=float(np.nanmean(exc_cl)), exc_close_t=t(exc_cl))


def replicate_3ch(d: pd.DataFrame) -> None:
    """Reproduce §3ch's `prior_day_break` loop EXACTLY, then show what the headline is made of.

    This is the control cell with a known answer (2,768 trades, LONG n 1,446, +10.161bp, t +4.71).
    It also isolates two conventions that D21's implementable arms resolve differently, because a
    resting stop order cannot honour either of them -- and together they are 13% of the headline.
    """
    day = d.index.normalize()
    hi = d["high"].groupby(day).max().shift(1)
    lo = d["low"].groupby(day).min().shift(1)
    pdh = hi.reindex(day).to_numpy(float); pdl = lo.reindex(day).to_numpy(float)
    h, l, c = (d[k].to_numpy(float) for k in ("high", "low", "close"))
    nd = day.to_numpy()
    rows, start = [], 0
    for i in range(1, len(d) + 1):
        if i == len(d) or nd[i] != nd[start]:
            sl = slice(start, i)
            if np.isfinite(pdh[start]) and np.isfinite(pdl[start]):
                up = np.flatnonzero(h[sl] > pdh[start]); dn = np.flatnonzero(l[sl] < pdl[start])
                fu = up[0] if len(up) else 10**9
                fd = dn[0] if len(dn) else 10**9
                if min(fu, fd) < 10**9:
                    side = 1 if fu < fd else -1
                    entry = pdh[start] if side == 1 else pdl[start]
                    rows.append((side, side * (c[i - 1] / entry - 1.0) * 1e4 - RT_BP,
                                 i - start, int(fu), int(fd)))
            start = i
    df = pd.DataFrame(rows, columns=["side", "bp", "nbars", "fu", "fd"])
    L = df[df.side == 1]

    def mt(x):
        return (len(x), float(x.bp.mean()),
                float(x.bp.mean() / (x.bp.std(ddof=1) / np.sqrt(len(x)))))

    n, m, t = mt(L)
    print(f"\n  CONTROL CELL — §3ch replicated exactly: all {len(df):,} trades, "
          f"LONG n={n:,} mean {m:+.3f}bp t {t:+.2f}   (filed: 1,446 / +10.161 / +4.71)")
    one = L[L.nbars < 2]
    rest = L[L.nbars >= 2]
    if len(one):
        print(f"    - {len(one)} of them are SINGLE-BAR days (Sundays: one 23:00 bar) at "
              f"{one.bp.mean():+.3f}bp. 'Exit at the session close' is one hour later and the "
              f"fill is a weekend GAP, not a level. Drop them -> n={len(rest):,} "
              f"{rest.bp.mean():+.3f}bp (worth {m - rest.bp.mean():+.3f}bp of the headline)")
    ties = df[(df.fu == df.fd) & (df.fu < 10**9)]
    if len(ties):
        print(f"    - {len(ties)} days have the SAME bar take out both levels (outside days). "
              f"§3ch's `side = 1 if fu < fd else -1` sends every one of them SHORT; a resting "
              f"buy-stop would have FILLED. The order of events inside that bar is unresolvable "
              f"at this frame -- cf. `order-of-events-needs-intraday`.")
    lastbar = L[L.fu == L.nbars - 1]
    if len(lastbar):
        print(f"    - {len(lastbar)} cross on the LAST bar of the day at "
              f"{lastbar.bp.mean():+.3f}bp (vs {L[L.fu < L.nbars-1].bp.mean():+.3f} for the rest); "
              f"entry and exit are then the same bar. D21 keeps them.")


def show(tag: str, r: dict) -> None:
    if r.get("n", 0) < 50:
        print(f"  {tag:<40} n={r.get('n',0)} — too few"); return
    print(f"  {tag:<40} {r['n']:>6} {r['fill_rate']:>7.1%} {r['med_bar']:>6.0f} "
          f"{r['level_bp']:>+9.3f} {r['fill_adv']:>+9.3f} {r['fill_adv_t']:>+7.2f} "
          f"{r['exc_level']:>+9.3f} {r['exc_level_t']:>+7.2f}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tf = sys.argv[1] if len(sys.argv) > 1 else "H1"
    d = load(tf)
    D = Days(d)
    nd = len(D.slices)
    print("=" * 112)
    print(f"OCTOBER D21 — PLACEBO LEVEL CONTROL on XAUUSD {tf}")
    print("=" * 112)
    print(f"  {d.index[0].date()} -> {d.index[-1].date()}   {len(d):,} bars, {nd:,} days, "
          f"round trip {RT_BP:.3f}bp")

    op, pdh, pdl, atr = D.open, D.pdh, D.pdl, D.atr
    ok = np.isfinite(pdh) & np.isfinite(pdl) & np.isfinite(atr) & (atr > 0)
    # A resting buy-stop can only be PLACED if the level is above where price already is.
    above = ok & (pdh > op)
    print(f"  days with a usable prior day: {int(ok.sum()):,}   "
          f"of which the prior high is ABOVE today's open (a stop is placeable): "
          f"{int(above.sum()):,} ({above.sum()/max(ok.sum(),1):.1%})")
    print(f"  *** the other {int((ok & ~above).sum()):,} days OPENED above the level: a 'break' "
          f"there is a gap, filled at the open, NOT at the level. §3ch counted them.")

    replicate_3ch(d)

    # ---------------- arm (a): §3ch's first-break framing — the control cell with a known answer
    print(f"\n  {'arm':<40} {'n':>6} {'fill':>7} {'bar':>6} {'level bp':>9} "
          f"{'fillAdv':>9} {'t':>7} {'excess':>9} {'t':>7}")
    print("  " + "-" * 108)
    a_first = arm(D, pdh, ok, cancel_below=pdl)
    show("(a) REAL first-break-up  [§3ch control]", a_first)
    a_any = arm(D, pdh, ok)
    show("(b) REAL any up-break, no cancel rule", a_any)
    a_rest = arm(D, pdh, above, cancel_below=pdl)
    show("(c) REAL (a) restricted to placeable", a_rest)
    a_rest_nc = arm(D, pdh, above)
    show("(d) REAL (b) restricted to placeable", a_rest_nc)

    # ---------------- the placebo families, scored on the SAME eligible set as (c) --------------
    dist_atr = (pdh - op) / atr                       # how far above the open, in ATR
    print(f"\n  prior-high distance above the open: median {np.nanmedian(dist_atr[above]):.3f} ATR "
          f"(p25 {np.nanpercentile(dist_atr[above],25):.3f}, "
          f"p75 {np.nanpercentile(dist_atr[above],75):.3f})")

    print(f"\n  FIXED-DISTANCE placebo — a level k x ATR above the open, no reference to yesterday")
    print(f"  {'arm':<40} {'n':>6} {'fill':>7} {'bar':>6} {'level bp':>9} "
          f"{'fillAdv':>9} {'t':>7} {'excess':>9} {'t':>7}")
    print("  " + "-" * 108)
    fixed_rows = []
    for k in (0.25, 0.50, 0.75, 1.00, 1.50):
        r = arm(D, op + k * atr, above)
        show(f"    placebo fixed {k:.2f} ATR", r)
        fixed_rows.append(dict(family="fixed", k=k, **r))

    # ---------------- circular-shift permutation: same distance DISTRIBUTION, wrong day ---------
    print(f"\n  PERMUTED placebo — the SAME {int(above.sum()):,} days, each given another day's "
          f"distance by circular shift (>= {MIN_SHIFT} days), {N_PERM} draws")
    # The placebo gets BOTH levels permuted and the SAME cancel rule, so it is the identical
    # strategy with only the location of the two levels replaced.
    dist_dn_atr = (op - pdl) / atr
    perm_adv, perm_exc, perm_lvl, perm_n = [], [], [], []
    for _ in range(N_PERM):
        s = int(RNG.integers(MIN_SHIFT, nd - MIN_SHIFT))
        pl = op + np.roll(dist_atr, s) * atr
        pd_ = op - np.roll(dist_dn_atr, s) * atr
        r = arm(D, pl, above, cancel_below=pd_)
        if r.get("n", 0) >= 50:
            perm_adv.append(r["fill_adv"]); perm_exc.append(r["exc_level"])
            perm_lvl.append(r["level_bp"]); perm_n.append(r["n"])
    perm_adv, perm_exc = np.array(perm_adv), np.array(perm_exc)
    perm_lvl, perm_n = np.array(perm_lvl), np.array(perm_n)

    real = a_rest
    def band(name, obs, draws):
        p = float((draws >= obs).mean())
        return (f"  {name:<24} observed {obs:>+8.3f}   placebo p50 {np.median(draws):>+8.3f}  "
                f"p05 {np.percentile(draws,5):>+8.3f}  p95 {np.percentile(draws,95):>+8.3f}   "
                f"P(placebo >= observed) = {p:.4f}")
    print(f"\n  {'':<24} {'n':>8}         placebo n p50 {np.median(perm_n):.0f}")
    print(band("FILL ADVANTAGE bp", real["fill_adv"], perm_adv))
    print(band("EXCESS over drift bp", real["exc_level"], perm_exc))
    print(band("level-arm net bp", real["level_bp"], perm_lvl))

    pd.DataFrame(fixed_rows).to_csv(OUT / f"d21_placebo_fixed_{tf}.csv", index=False)
    pd.DataFrame(dict(fill_adv=perm_adv, exc_level=perm_exc, level_bp=perm_lvl,
                      n=perm_n)).to_csv(OUT / f"d21_placebo_perm_{tf}.csv", index=False)

    print(f"""
  HOW TO READ THIS. 'fillAdv' is the quantity §3ci attributes 47% of the edge to: the gap between
  the level asked for and where the crossing bar closed. If the placebo rows match it, the gap is a
  property of CROSSING A LEVEL, not of the prior-day high -- true, capturable with a resting order,
  but not evidence for this level. 'excess' is the level-arm return minus gold's unconditional drift
  over the same bars-left; that is the column that decides whether the prior-day high carries
  information. P(placebo >= observed) near or above 0.50 on BOTH is outcome 1 in the header.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
