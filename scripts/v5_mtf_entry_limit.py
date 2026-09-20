"""The decisive control: is the ICT zone doing the work, or is it just "price pulled back"?

WHAT THE TIMING RUN FOUND, AND WHY IT NEEDS THIS TEST. As a matched-count TIMING overlay every
trigger loses (`v5_mtf_entry_timing.py`: all 12 cells negative, best -0.0004%). But CONDITIONAL
ON FIRING the ICT triggers capture 89-99% of the oracle — order-block tap at the 2h window gains
+0.2515% against an oracle ceiling of +0.2547%, on 96 fills. That is causal (the trigger reads
only bars up to the entry) and it is a paired comparison on the same trades, so it is not a
lookahead. It reframes the product: not "wait, then market-order", but **rest a LIMIT order at
the zone and trade only if it fills**.

THE CONTROL THAT DECIDES IT. A limit order resting below the market on a long will fill exactly
when price dips, and a dip is a better entry whatever drew the line. So the ICT zone has to beat
a line drawn with NO pattern at all: a flat offset of k*ATR against the signal, resting in the
same window. If plain-offset limits capture the same, then "order block" and "fair value gap" are
decoration on "buy the dip", and the honest description of the finding is a limit-order
mechanic rather than a Smart-Money one. §3v already closed these patterns as PREDICTORS; this
asks whether they are better than arithmetic as PLACEMENT.

The offset ladder is matched to the zones' own fill rates so the comparison is like-for-like: a
control that fills 60% of the time cannot be compared to a zone that fills 12%.

EVERY APPROACH IS SCORED PER SIGNAL, NOT PER FILL. A strategy that fills 12% of the time and
wins 0.25% earns 0.03% per signal; the baseline earns 0.091% by always trading. Quoting per-fill
returns beside a baseline quoted per-signal is the arithmetic that makes thin filters look
brilliant, so both columns are printed and the per-signal one is the verdict.

Also reported: what the skipped signals did. If the unfilled ones would have been winners, the
filter is discarding edge, and that cost belongs in the comparison rather than out of sight.

    python scripts/v5_mtf_entry_limit.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_advisor_measure import load_frames  # noqa: E402
from scripts.v5_mtf_entry_oracle import DOWN_AT, SPREAD_USD, UP_AT, load_signal  # noqa: E402
from scripts.v5_mtf_entry_timing import (  # noqa: E402
    EMA_N, LOOKBACK, _ema, find_entry, zone_levels,
)
from scripts.v5_range_multitf import atr_series  # noqa: E402
from src.v5.advisor_calibration import block_bootstrap_ci  # noqa: E402
from src.v5.mtf_frames import frames, verify_against_filed  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "mtf_entry"
OFFSETS = (0.10, 0.20, 0.30, 0.50)        # in ATR, against the signal direction
ZONES = ("fvg", "ob", "sweep")


def offset_fill(side: int, s: int, e: int, arr: dict, lvl: float) -> int | None:
    """First bar in (s, e) whose range reaches a resting limit at `lvl`."""
    h, l = arr["h"], arr["l"]
    for j in range(s + 1, e):
        if side < 0 and h[j] >= lvl:      # short: limit ABOVE the market
            return j
        if side > 0 and l[j] <= lvl:      # long: limit BELOW the market
            return j
    return None


def run(window: float, hold_h: int) -> pd.DataFrame:
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
        dec = t + pd.Timedelta(hours=4)
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
        e = min(int(np.searchsorted(mi.values,
                                    np.datetime64(dec + pd.Timedelta(hours=window)),
                                    side="right")), xi)
        if e <= s + 1:
            continue
        base_px, exit_px = arr["c"][s], arr["c"][xi]
        row = dict(time=t, side=side, p=float(r["p"]), atr=float(a), base_px=base_px,
                   base_ret=side * (exit_px - base_px) / base_px)
        seg_h, seg_l = arr["h"][s + 1:e], arr["l"][s + 1:e]
        best = seg_h.max() if side < 0 else seg_l.min()
        row["oracle"] = side * (exit_px - best) / best
        # --- the ICT zones, entered as LIMITS (fill only) ---
        for tg in ZONES:
            j = find_entry(tg, side, s, e, arr)
            row[f"{tg}_fill"] = 1.0 if j is not None else 0.0
            row[f"{tg}_ret"] = (side * (exit_px - arr["c"][j]) / arr["c"][j]
                               if j is not None else np.nan)
        # --- the pattern-free control: a flat k*ATR offset against the signal ---
        for k in OFFSETS:
            lvl = base_px - side * k * a          # short -> above, long -> below
            j = offset_fill(side, s, e, arr, lvl)
            row[f"off{k}_fill"] = 1.0 if j is not None else 0.0
            # a resting limit fills AT its level, not at the bar's close
            row[f"off{k}_ret"] = (side * (exit_px - lvl) / lvl) if j is not None else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, names: list, label: str) -> pd.DataFrame:
    n = len(df)
    orc = (df["oracle"] - df["base_ret"]).mean()
    out = []
    for nm in names:
        f = df[f"{nm}_fill"] > 0
        if f.sum() < 20:
            continue
        sub = df[f]
        g = (sub[f"{nm}_ret"] - sub["base_ret"]).values
        lo, hi, _ = block_bootstrap_ci(g, block=20, alpha=0.10)
        yr = pd.DatetimeIndex(sub["time"]).year
        per = pd.Series(g, index=yr).groupby(level=0).mean()
        # per SIGNAL: an unfilled signal earns nothing at all
        ev_signal = float(sub[f"{nm}_ret"].sum() / n)
        skipped = df[~f]["base_ret"].mean() if (~f).sum() else np.nan
        out.append(dict(name=nm, fill=f.mean(), n_fill=int(f.sum()),
                        gain_per_fill=float(g.mean()), ci_lo=float(lo), ci_hi=float(hi),
                        capture=float(g.mean() / orc),
                        ret_per_fill=float(sub[f"{nm}_ret"].mean()),
                        ev_per_signal=ev_signal, skipped_base=float(skipped),
                        years_pos=int((per > 0).sum()), years=int(len(per))))
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--hold", type=int, default=9)
    a = ap.parse_args()

    df = run(a.window, a.hold)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "limit.parquet")
    px = df["base_px"].mean()
    orc = (df["oracle"] - df["base_ret"]).mean()

    print("=" * 108)
    print(f"LIMIT-ORDER PLACEMENT at the {a.window}h window — {len(df):,} signals, "
          f"common exit at +{a.hold}h")
    print("=" * 108)
    print(f"  baseline: market order at the H4 close, ALWAYS trades — "
          f"{df.base_ret.mean()*100:+.4f}% per signal")
    print(f"  oracle ceiling (hindsight-best entry in the window): {orc*100:+.4f}%")
    print(f"  one spread ${SPREAD_USD:.2f} = {SPREAD_USD/px*100:.4f}%\n")

    names = list(ZONES) + [f"off{k}" for k in OFFSETS]
    s = summarise(df, names, "all")
    s.to_csv(OUT / "limit_summary.csv", index=False)
    print(f"  {'placement':10s} {'fill':>6s} {'n':>5s} {'gain/fill':>11s} {'CI90':>20s} "
          f"{'capture':>8s} {'EV/signal':>10s} {'skipped':>9s} {'yrs+':>6s}")
    for _, r in s.iterrows():
        tag = "ICT " if r["name"] in ZONES else "flat"
        print(f"  {tag}{r['name']:6s} {r['fill']*100:5.1f}% {r['n_fill']:5d} "
              f"{r['gain_per_fill']*100:+10.4f}% [{r['ci_lo']*100:+7.4f},{r['ci_hi']*100:+7.4f}] "
              f"{r['capture']*100:7.1f}% {r['ev_per_signal']*100:+9.4f}% "
              f"{r['skipped_base']*100:+8.4f}% {r['years_pos']:3d}/{r['years']:<2d}")

    print(f"\n  BASELINE EV/signal for comparison: {df.base_ret.mean()*100:+.4f}%")
    best_ict = s[s.name.isin(ZONES)].sort_values("ev_per_signal").iloc[-1]
    best_flat = s[~s.name.isin(ZONES)].sort_values("ev_per_signal").iloc[-1]
    print(f"\n  best ICT zone : {best_ict['name']:6s} fill {best_ict['fill']*100:4.1f}%  "
          f"gain/fill {best_ict['gain_per_fill']*100:+.4f}%  "
          f"EV/signal {best_ict['ev_per_signal']*100:+.4f}%")
    print(f"  best flat     : {best_flat['name']:6s} fill {best_flat['fill']*100:4.1f}%  "
          f"gain/fill {best_flat['gain_per_fill']*100:+.4f}%  "
          f"EV/signal {best_flat['ev_per_signal']*100:+.4f}%")

    # like-for-like: the flat offset whose fill rate is closest to the best zone's
    cand = s[~s.name.isin(ZONES)].copy()
    cand["dist"] = (cand["fill"] - best_ict["fill"]).abs()
    m = cand.sort_values("dist").iloc[0]
    print(f"\n  LIKE-FOR-LIKE (matched fill rate, which is what makes the comparison fair):")
    print(f"    {best_ict['name']} fills {best_ict['fill']*100:.1f}% and gains "
          f"{best_ict['gain_per_fill']*100:+.4f}%/fill")
    print(f"    {m['name']} fills {m['fill']*100:.1f}% and gains "
          f"{m['gain_per_fill']*100:+.4f}%/fill  (no pattern, just arithmetic)")
    d = best_ict["gain_per_fill"] - m["gain_per_fill"]
    print(f"    the ICT zone is worth {d*100:+.4f}% ({d*px:+.2f} = {d*px/SPREAD_USD:+.1f} "
          f"spreads) beyond the flat line")
    print(f"    -> the pattern {'ADDS' if d > 0 else 'ADDS NOTHING'} over a plain offset")
    print(f"\nwrote {(OUT / 'limit_summary.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
