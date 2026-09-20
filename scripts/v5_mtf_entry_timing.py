"""Does an M15 confirmation improve the entry on an ALREADY-MEASURED H4 directional call?

THE QUESTION, AND WHY IT IS NOT THE ONE §3v/§3w CLOSED. That sweep asked whether ICT patterns
PREDICT DIRECTION on XAU H4/H1 and answered no, twelve concepts and four confluence combos deep
(Breaker Block -0.67, Mitigation Block -1.06 and backwards, Inversion FVG -0.60, EQH/EQL -0.44).
This asks something different: given a directional call that is already measured and validated
(`1.0/9/BASE`: UP 60.3%, DOWN 56.2%, both tails holding across halves), is a better PRICE
available by waiting for an M15 trigger? That is entry timing — execution economics — which is
where §3ay concluded the remaining lever is once the signal side is exhausted. The sweep never
reached M15 at all, because nothing survived H4/H1 to justify going lower.

MATCHED TRADE COUNT IS WHAT MAKES IT A TIMING TEST. If an unconfirmed signal were skipped, the
arms would differ in WHICH trades they take and the result would be a filtering test wearing a
timing test's clothes — and filtering is the thing already disproven. So when no trigger appears
inside the window, the trade is entered at the window's end anyway. Every arm takes the identical
823 trades, exits at the identical instant, and pays the identical costs; the ONLY difference is
the entry price. Costs therefore cancel in the paired difference, and the absolute net of each
arm is printed separately so it is still visible whether either is profitable at all.

    long:   r = (exit - entry) / entry     -> a lower entry is better
    short:  r = (entry - exit) / entry     -> a higher entry is better

THE ORACLE SETS THE SCALE (`v5_mtf_entry_oracle.py`, run first as MANDATORY CONTROL #5). A
perfect, cheating timer is worth **+0.2547% = $6.88 = 14.6 spreads** per trade at the 2h window,
against a naive baseline of +0.0910%. So the prize is real and roughly 2.8x the whole trade. Every
trigger below is reported as a CAPTURE RATE of that oracle, because "+0.03%" means nothing without
knowing whether the ceiling was 0.04% or 4%. §3ag spent a run finding detectors worth +0.090
against an oracle worth +1.252; the capture column exists so that cannot happen quietly.

THE ZONES MUST PRE-EXIST THE DECISION. A trigger is the RETEST of an imbalance, not its
formation: inside a 2h window there is rarely time for a zone to form and be revisited. So zones
are detected on M15 bars strictly BEFORE the decision instant, must still be unfilled, and must
sit on the correct side of price; the trigger is the first touch inside the window. This is both
the realistic use (the H4 call fires, you enter at the nearest standing imbalance) and causal by
construction.

WHAT A TRIGGER HAS TO BEAT. `momentum` is a deliberately plain control — no ICT vocabulary, just
"wait for the M15 close to come back through a short EMA in the bias direction". If the pattern
triggers cannot beat that, they are not adding pattern information, only delay.

    python scripts/v5_mtf_entry_timing.py
    python scripts/v5_mtf_entry_timing.py --windows 1,2,4 --draws 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_advisor_measure import load_frames  # noqa: E402
from scripts.v5_mtf_entry_oracle import DOWN_AT, SPREAD_USD, UP_AT, load_signal  # noqa: E402
from scripts.v5_range_multitf import atr_series  # noqa: E402
from src.features.smc_signals import fair_value_gaps, order_blocks  # noqa: E402
from src.v5.advisor_calibration import block_bootstrap_ci  # noqa: E402
from src.v5.mtf_frames import frames, verify_against_filed  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "mtf_entry"
TRIGGERS = ("fvg", "ob", "sweep", "momentum")
LOOKBACK = 48          # M15 bars (12h) in which a zone may have formed
EMA_N = 8


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    a = 2.0 / (n + 1.0)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def zone_levels(m15: pd.DataFrame, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar zone level for bearish (sell-side, above) and bullish (buy-side, below) setups.

    Returns the PRICE of the zone created at each bar, NaN where none formed. Both detectors
    are causal — `order_blocks` and `fair_value_gaps` read only bars up to i — and the level is
    taken from bars at or before formation, never after.
    """
    o = m15["open"].values
    h, l, c = m15["high"].values, m15["low"].values, m15["close"].values
    n = len(c)
    bear = np.full(n, np.nan)
    bull = np.full(n, np.nan)
    if kind == "ob":
        ob_bull, ob_bear = order_blocks(m15)
        ib, ir = ob_bull.values > 0, ob_bear.values > 0
        # the block is the last opposite-colour candle before the impulse: bar i-1
        bear[ir] = np.where(ir, np.roll(h, 1), np.nan)[ir]
        bull[ib] = np.where(ib, np.roll(l, 1), np.nan)[ib]
    elif kind == "fvg":
        fb, fr, _ = fair_value_gaps(m15)
        ib, ir = fb.values > 0, fr.values > 0
        # bullish FVG gap = [high[i-2], low[i]]  -> buy-side level is its lower edge
        # bearish FVG gap = [high[i], low[i-2]]  -> sell-side level is its upper edge
        h2, l2 = np.roll(h, 2), np.roll(l, 2)
        bull[ib] = h2[ib]
        bear[ir] = l2[ir]
    else:
        raise ValueError(kind)
    return bear, bull


def find_entry(trigger: str, side: int, s: int, e: int, arr: dict) -> int | None:
    """Index of the first M15 bar in (s, e) that triggers, or None.

    `s` is the decision bar (its close is the baseline entry), so scanning starts at s+1: the
    decision bar itself has already closed and cannot be re-entered.
    """
    h, l, c = arr["h"], arr["l"], arr["c"]
    if trigger == "momentum":
        ema = arr["ema"]
        for j in range(s + 1, e):
            # wait for the close to come back through the EMA in the bias direction
            if side < 0 and c[j] < ema[j] and c[j - 1] >= ema[j - 1]:
                return j
            if side > 0 and c[j] > ema[j] and c[j - 1] <= ema[j - 1]:
                return j
        return None
    if trigger == "sweep":
        for j in range(s + 1, e):
            w = slice(max(0, j - 12), j)
            if side < 0 and h[j] > h[w].max() and c[j] < h[w].max():
                return j          # swept a local high and closed back under it
            if side > 0 and l[j] < l[w].min() and c[j] > l[w].min():
                return j
        return None
    # fvg / ob: the first touch of a standing zone on the correct side of price
    bear, bull = arr[f"{trigger}_bear"], arr[f"{trigger}_bull"]
    px = c[s]
    lb = slice(max(0, s - LOOKBACK), s + 1)
    if side < 0:
        z = bear[lb]
        z = z[np.isfinite(z) & (z > px)]          # resistance ABOVE, still unfilled
        if not len(z):
            return None
        lvl = float(z.min())                      # the nearest one
        for j in range(s + 1, e):
            if h[j] >= lvl:
                return j
    else:
        z = bull[lb]
        z = z[np.isfinite(z) & (z < px)]
        if not len(z):
            return None
        lvl = float(z.max())
        for j in range(s + 1, e):
            if l[j] <= lvl:
                return j
    return None


def run(windows, hold_h: int) -> pd.DataFrame:
    h4_filed, m15 = load_frames()
    F = frames(m15)
    verify_against_filed(F["H4"], h4_filed)
    sig = load_signal()
    atr = atr_series(F["H4"], 20)
    M = F["M15"]
    arr = {"h": M["high"].values, "l": M["low"].values, "c": M["close"].values,
           "ema": _ema(M["close"].values, EMA_N)}
    for k in ("fvg", "ob"):
        arr[f"{k}_bear"], arr[f"{k}_bull"] = zone_levels(M, k)
    mi = M.index

    fires = sig[(sig["p"] >= DOWN_AT) | (sig["p"] <= UP_AT)]
    rows = []
    for t, r in fires.iterrows():
        side = -1 if r["p"] >= DOWN_AT else +1
        dec = t + pd.Timedelta(hours=4)              # the H4 bar's CLOSE, not its stamp
        a = atr.get(t, np.nan)
        if not np.isfinite(a) or a <= 0:
            continue
        s = int(np.searchsorted(mi.values, np.datetime64(dec), side="left"))
        if s >= len(mi) or mi[s] != dec:
            continue
        xi = int(np.searchsorted(mi.values,
                                 np.datetime64(dec + pd.Timedelta(hours=hold_h)), side="left"))
        if xi >= len(mi):
            continue
        base_px, exit_px = arr["c"][s], arr["c"][xi]
        row = dict(time=t, decision=dec, side=side, p=float(r["p"]), y=float(r["y"]),
                   atr=float(a), base_px=base_px, exit_px=exit_px,
                   base_ret=side * (exit_px - base_px) / base_px)
        for w in windows:
            e = min(int(np.searchsorted(mi.values,
                                        np.datetime64(dec + pd.Timedelta(hours=w)),
                                        side="right")), xi)
            if e <= s + 1:
                for tg in TRIGGERS:
                    row[f"{tg}_{w}"] = np.nan
                    row[f"{tg}_fired_{w}"] = np.nan
                continue
            # the oracle for this window, recomputed so the capture rate is self-contained
            seg_h, seg_l = arr["h"][s + 1:e], arr["l"][s + 1:e]
            best = seg_h.max() if side < 0 else seg_l.min()
            row[f"oracle_{w}"] = side * (exit_px - best) / best
            fallback = arr["c"][e - 1]               # matched count: enter at the window's end
            for tg in TRIGGERS:
                j = find_entry(tg, side, s, e, arr)
                px = arr["c"][j] if j is not None else fallback
                row[f"{tg}_{w}"] = side * (exit_px - px) / px
                row[f"{tg}_fired_{w}"] = 1.0 if j is not None else 0.0
                row[f"{tg}_bars_{w}"] = (j - s) if j is not None else (e - 1 - s)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="1,2,4")
    ap.add_argument("--hold", type=int, default=9)
    ap.add_argument("--draws", type=int, default=200)
    a = ap.parse_args()
    windows = [float(x) for x in a.windows.split(",")]

    df = run(windows, a.hold)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "timing.parquet")

    px = df["base_px"].mean()
    cost_pct = SPREAD_USD / px                      # one spread, paid on entry
    print("=" * 104)
    print(f"M15 ENTRY TIMING on the measured H4 call — {len(df):,} fires, matched trade count, "
          f"common exit at +{a.hold}h")
    print("=" * 104)
    print(f"  baseline (enter at the H4 close) : {df.base_ret.mean()*100:+.4f}% gross, "
          f"{(df.base_ret.mean()-2*cost_pct)*100:+.4f}% net of 2 spreads")
    print(f"  mean price ${px:.2f}   one spread ${SPREAD_USD:.2f} = {cost_pct*100:.4f}%")
    print("  costs CANCEL in the paired difference: every arm takes the same trades.\n")

    res = []
    for w in windows:
        orc = (df[f"oracle_{w}"] - df["base_ret"]).mean()
        print(f"-- window {w}h   oracle ceiling {orc*100:+.4f}% "
              f"(= {orc*px/SPREAD_USD:.1f} spreads) " + "-" * 40)
        print(f"   {'trigger':10s} {'fired':>6s} {'gain vs base':>13s} {'CI90':>20s} "
              f"{'capture':>8s} {'$/trade':>8s} {'yrs+':>6s} {'bars':>5s}")
        for tg in TRIGGERS:
            c = df[[f"{tg}_{w}", f"{tg}_fired_{w}", f"{tg}_bars_{w}", "base_ret",
                    "base_px", "time"]].dropna()
            if not len(c):
                continue
            g = (c[f"{tg}_{w}"] - c["base_ret"]).values
            lo, hi, se = block_bootstrap_ci(g, block=20, alpha=0.10)
            yr = pd.DatetimeIndex(c["time"]).year
            per = pd.Series(g, index=yr).groupby(level=0).mean()
            res.append(dict(window=w, trigger=tg, n=int(len(c)),
                            fired=float(c[f"{tg}_fired_{w}"].mean()), gain=float(g.mean()),
                            ci_lo=float(lo), ci_hi=float(hi), capture=float(g.mean() / orc),
                            usd=float((g * c["base_px"]).mean()),
                            years_pos=int((per > 0).sum()), years=int(len(per)),
                            bars=float(c[f"{tg}_bars_{w}"].mean())))
            r = res[-1]
            print(f"   {tg:10s} {r['fired']*100:5.1f}% {r['gain']*100:+12.4f}% "
                  f"[{lo*100:+7.4f},{hi*100:+7.4f}] {r['capture']*100:7.1f}% "
                  f"{r['usd']:+8.2f} {r['years_pos']:3d}/{r['years']:<2d} {r['bars']:5.1f}")
        print()

    out = pd.DataFrame(res)
    out.to_csv(OUT / "timing_summary.csv", index=False)
    best = out.loc[out["gain"].idxmax()]
    print("=" * 104)
    print(f"BEST CELL: {best.trigger} @ {best.window}h — gain {best.gain*100:+.4f}% "
          f"(${best.usd:+.2f} = {best.usd/SPREAD_USD:+.1f} spreads), "
          f"capture {best.capture*100:.1f}%, {best.years_pos}/{best.years} years positive")
    mom = out[out.trigger == "momentum"]["gain"].max()
    print(f"  the plain momentum control's best cell: {mom*100:+.4f}%  -> the ICT triggers "
          f"{'BEAT' if best.gain > mom and best.trigger != 'momentum' else 'DO NOT BEAT'} it")
    print(f"\nwrote {(OUT / 'timing_summary.csv').relative_to(ROOT)}")
    print("Gate 6 (max-statistic null over this 4x%d grid) is run by "
          "v5_mtf_entry_null.py, and only if a cell looks worth pricing." % len(windows))


if __name__ == "__main__":
    main()
