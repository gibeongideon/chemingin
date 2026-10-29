#!/usr/bin/env python3
"""OCTOBER A4 — VOLUME-DERIVED supply and demand: volume profile, POC, value area, HVN/LVN.

Pre-registered in data/v5_runs/october/PREREGISTRATION.md. Design in OCTOBER.MD.

WHY THIS IS THE ONE GENUINELY UNKNOWN ROW IN THE DOCUMENT. A grep across all 6,232 lines of
V5_FINDINGS.md and every file in src/ returns **zero hits** for volume profile, market profile,
value area, POC, VAH, VAL, footprint or order-flow delta. Meanwhile the 12-concept ICT/SMC sweep
(3v/3w) tested order blocks, breakers, mitigation blocks, FVGs, IFVGs and liquidity pools -- and
**every zone it tested was SWING-DERIVED**, i.e. built from price pivots. A volume-derived zone is
a different construction with a different mechanism: it asks where business was actually done, not
where price turned. It is also the cheapest available approximation of the "real order flow /
tick-level microstructure" that 3r and BEST_FINDINGS both name as THE data type that could change
the answer here.

*** TWO HONEST LIMITATIONS, MEASURED BEFORE ANY TEST. ***
  1. `real_volume` is **0.0% populated** in this feed. Only `tick_volume` exists, which counts
     TICKS, not contracts. A tick-count profile is a proxy for a volume profile and must be called
     one. 3ag already measured volume/dollar/tick BARS at 0 of 17.
  2. `tick_volume`'s median grew from 3,749 (2015) to 15,325 (2026) -- roughly 4x. **Absolute
     volume is therefore non-stationary**, so everything below is normalised WITHIN its own rolling
     profile window and no level is ever compared across years.

THE CONTROL THAT KILLED EVERY ICT ZONE (3bc) IS THE ONE THAT MATTERS. There, FVG retests, order-block
taps and liquidity sweeps all looked excellent conditional on firing (+0.19% to +0.25%) and all lost
to **a flat 0.5-ATR limit, which beat the best order block on gain per fill AND filled 2.4x as
often**. So a volume zone is scored against that same flat-offset control, not against zero.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "v5_runs" / "october"
SPREAD_BP = 0.504
RT_BP = SPREAD_BP * 1.5
LOOKBACK = 120          # H1 bars = 5 trading days, the usual profile window
N_BINS = 50             # price bins per profile
VALUE_AREA = 0.70       # the conventional 70% value area
HORIZON = 24            # bars held after a zone interaction = 1 day


def wilder_atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def load_h1() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data/XAUUSD_H1_long.csv", parse_dates=["time"], index_col="time")
    d = d[~d.index.duplicated(keep="last")].sort_index()
    d["atr"] = wilder_atr(d["high"], d["low"], d["close"])
    return d.dropna(subset=["atr"])


def volume_profile(d: pd.DataFrame, lookback: int, n_bins: int) -> pd.DataFrame:
    """Strictly causal POC / VAH / VAL from the trailing `lookback` bars.

    Each bar's tick_volume is spread UNIFORMLY across its own high-low range, which is the standard
    approximation when only OHLCV is available (true profiles need intrabar prints). The window
    ends at t-1, so nothing at t is used.
    """
    h = d["high"].to_numpy(float); l = d["low"].to_numpy(float)
    v = pd.to_numeric(d["tick_volume"], errors="coerce").fillna(0.0).to_numpy(float)
    n = len(d)
    poc = np.full(n, np.nan); vah = np.full(n, np.nan); val = np.full(n, np.nan)
    hvn = np.full(n, np.nan)
    for i in range(lookback, n):
        s = slice(i - lookback, i)                       # ends at i-1: causal
        lo, hi = l[s].min(), h[s].max()
        if not np.isfinite(lo) or hi <= lo:
            continue
        edges = np.linspace(lo, hi, n_bins + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        hist = np.zeros(n_bins)
        for j in range(i - lookback, i):
            a, b = l[j], h[j]
            if b <= a:
                k = min(int((a - lo) / (hi - lo) * n_bins), n_bins - 1)
                hist[k] += v[j]
                continue
            k0 = max(int((a - lo) / (hi - lo) * n_bins), 0)
            k1 = min(int((b - lo) / (hi - lo) * n_bins), n_bins - 1)
            hist[k0:k1 + 1] += v[j] / (k1 - k0 + 1)      # uniform across the bar's range
        if hist.sum() <= 0:
            continue
        p = int(np.argmax(hist))
        poc[i] = centres[p]
        # value area: expand outward from the POC until 70% of volume is enclosed
        tot = hist.sum(); acc = hist[p]; a_i = b_i = p
        while acc < VALUE_AREA * tot and (a_i > 0 or b_i < n_bins - 1):
            take_lo = hist[a_i - 1] if a_i > 0 else -1.0
            take_hi = hist[b_i + 1] if b_i < n_bins - 1 else -1.0
            if take_hi >= take_lo:
                b_i += 1; acc += take_hi
            else:
                a_i -= 1; acc += take_lo
        val[i] = centres[a_i]; vah[i] = centres[b_i]
        # is the CURRENT price sitting in a high- or low-volume node?
        cur = min(max(int((d["close"].iloc[i - 1] - lo) / (hi - lo) * n_bins), 0), n_bins - 1)
        hvn[i] = hist[cur] / (hist.mean() if hist.mean() > 0 else np.nan)
    return pd.DataFrame(dict(poc=poc, vah=vah, val=val, hvn=hvn), index=d.index)


def score(name: str, mask: np.ndarray, fwd: np.ndarray, side: np.ndarray,
          base_ev: float, n_total: int) -> dict:
    n = int(mask.sum())
    if n < 200:
        return dict(rule=name, fires=n, note="too few")
    r = fwd[mask] * side[mask]
    ev = float(np.nanmean(r) * 1e4)
    return dict(rule=name, fires=n, fill=n / n_total, hit=float(np.nanmean(r > 0)),
                ev_bp=ev, net_bp=ev - RT_BP, vs_hold=ev - base_ev)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_h1()
    vp = volume_profile(d, LOOKBACK, N_BINS)
    c = d["close"].to_numpy(float)
    atr = d["atr"].to_numpy(float)
    fwd = np.full(len(d), np.nan)
    fwd[:-HORIZON] = c[HORIZON:] / c[:-HORIZON] - 1.0

    print("=" * 104)
    print("OCTOBER A4 — VOLUME-DERIVED SUPPLY/DEMAND (volume profile, POC, value area)")
    print("=" * 104)
    print(f"  XAUUSD H1  {d.index[0].date()} -> {d.index[-1].date()}  {len(d):,} bars")
    print(f"  profile: {LOOKBACK} bars (5 days), {N_BINS} price bins, {VALUE_AREA:.0%} value area")
    print(f"  *** real_volume is 0.0% populated -> this is a TICK-COUNT profile, not a volume one")
    print(f"  *** tick_volume median grew 3,749 (2015) -> 15,325 (2026), so all levels are")
    print(f"      normalised inside their own rolling window and never compared across years")
    ok0 = np.isfinite(vp["poc"].to_numpy()) & np.isfinite(fwd)
    print(f"  profiled bars {int(ok0.sum()):,}   base fwd {HORIZON}h "
          f"{float(np.nanmean(fwd[ok0]))*1e4:+.3f}bp   round trip {RT_BP:.2f}bp")

    poc, vah, val, hvn = (vp[k].to_numpy(float) for k in ("poc", "vah", "val", "hvn"))
    prev_c = np.r_[np.nan, c[:-1]]
    d_poc = (prev_c - poc) / atr                     # ATR units from the point of control
    base_ev = float(np.nanmean(fwd[ok0]) * 1e4)
    nt = int(ok0.sum())
    long_side = np.ones(len(d)); short_side = -np.ones(len(d))

    rows = []
    rows.append(dict(rule="HOLD (always long)", fires=nt, fill=1.0,
                     hit=float(np.nanmean(fwd[ok0] > 0)), ev_bp=base_ev,
                     net_bp=base_ev - RT_BP, vs_hold=0.0))
    # --- mean reversion TO the POC: price far below it -> long, far above -> short -----------
    for k in (1.0, 1.5, 2.0):
        rows.append(score(f"below POC by {k}xATR -> long", ok0 & (d_poc < -k), fwd, long_side,
                          base_ev, nt))
        rows.append(score(f"above POC by {k}xATR -> short", ok0 & (d_poc > k), fwd, short_side,
                          base_ev, nt))
    # --- value-area edges: fade the edge, and break beyond it --------------------------------
    rows.append(score("at/below VAL -> long (fade)", ok0 & (prev_c <= val), fwd, long_side,
                      base_ev, nt))
    rows.append(score("at/above VAH -> short (fade)", ok0 & (prev_c >= vah), fwd, short_side,
                      base_ev, nt))
    rows.append(score("broke above VAH -> long (cont)", ok0 & (prev_c > vah), fwd, long_side,
                      base_ev, nt))
    rows.append(score("broke below VAL -> short (cont)", ok0 & (prev_c < val), fwd, short_side,
                      base_ev, nt))
    # --- volume NODES: is price in thin (LVN) or thick (HVN) structure? ----------------------
    rows.append(score("in LVN (hvn<0.5) -> long", ok0 & (hvn < 0.5), fwd, long_side, base_ev, nt))
    rows.append(score("in HVN (hvn>1.5) -> long", ok0 & (hvn > 1.5), fwd, long_side, base_ev, nt))
    # --- THE CONTROL THAT KILLED EVERY ICT ZONE (3bc): a flat ATR offset, no zone at all ------
    pull = (prev_c - c) / atr
    rows.append(score("CONTROL flat 0.5xATR dip -> long", ok0 & (pull > 0.5), fwd, long_side,
                      base_ev, nt))
    rows.append(score("CONTROL flat 1.0xATR dip -> long", ok0 & (pull > 1.0), fwd, long_side,
                      base_ev, nt))

    print(f"\n  {'rule':<34} {'fires':>8} {'fill':>7} {'hit':>7} {'EV bp':>9} "
          f"{'net bp':>9} {'vs HOLD':>9}")
    print("  " + "-" * 92)
    for r in rows:
        if r.get("note"):
            print(f"  {r['rule']:<34} {r['fires']:>8}  {r['note']}"); continue
        print(f"  {r['rule']:<34} {r['fires']:>8} {r['fill']:>6.1%} {r['hit']:>7.4f} "
              f"{r['ev_bp']:>+9.3f} {r['net_bp']:>+9.3f} {r['vs_hold']:>+9.3f}")
    pd.DataFrame(rows).to_csv(OUT / "a4_volume_zones.csv", index=False)
    print(f"""
  THE TWO COLUMNS THAT DECIDE IT. 'vs HOLD' is EV against simply being long over the same bars
  (CONTROL #2) -- 3bh's identity means any rule that is not always-long starts from a deficit
  against gold's drift. And the two CONTROL rows are 3bc's killer: if a flat ATR dip with no zone
  at all matches or beats the volume zones on EV AND fires more often, the zone is decoration.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
