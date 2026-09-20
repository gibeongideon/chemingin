"""MANDATORY CONTROL #5 for the M15 entry-timing idea: hindsight-trade the label FIRST.

THE QUESTION. The advisor gives a measured H4 directional call (`1.0/9/BASE`: UP 60.3%,
DOWN 56.2%, both tails validated across halves). The proposal is to delay entry until an M15
confirmation appears. Before building any detector, this asks what the BEST POSSIBLE entry in
the window would have been worth — because if a perfect, cheating entry-timer is worth little,
no honest trigger can be worth more. §3ap's lesson, applied before the work rather than after:
"hindsight-trade a label BEFORE building a detector". §3ag's top-detector run spent a week
finding detectors worth +0.090 against an oracle worth +1.252; the reverse mistake, spending it
on an oracle worth nothing, is what this file exists to prevent.

THE DESIGN ISOLATES ENTRY PRICE AND NOTHING ELSE. Both arms exit at the SAME wall-clock instant,
so the trade captures the identical market move and the only difference is the price paid to get
in. A fixed HOLD from entry would instead shift the exit window too, confounding timing with a
different horizon. With a common exit:

    long:   r = (exit - entry) / entry      -> a LOWER entry is better
    short:  r = (entry - exit) / entry      -> a HIGHER entry is better

so "best entry in the window" is the window's low for an UP call and its high for a DOWN call.

WHY THE ANTI-ORACLE IS REPORTED TOO. The oracle bounds what a perfect timer could gain; the
anti-oracle bounds what a bad one could lose. A wide band means timing matters and the detector
has to earn its place; a narrow band means the whole idea is inside the noise whatever detector
is used. Reporting only the upside would make any mediocre trigger look like progress.

EVERYTHING IS IN ATR AND IN DOLLARS, BESIDE THE SPREAD. §3ax: gold's daily range went from 1.34%
to 2.73% of price, so a percentage prize is a different thing in 2020 and 2026; and a prize
smaller than the $0.47 live spread cannot be collected however real it is.

    python scripts/v5_mtf_entry_oracle.py
    python scripts/v5_mtf_entry_oracle.py --windows 1,2,4 --hold 9
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
from scripts.v5_range_multitf import atr_series  # noqa: E402
from src.v5.advisor_calibration import block_bootstrap_ci  # noqa: E402
from src.v5.mtf_frames import frames, verify_against_filed  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "mtf_entry"
SPREAD_USD = 0.47          # measured live on bridge 18814, 2026-09-19
DOWN_AT, UP_AT = 0.60, 0.40


def load_signal() -> pd.DataFrame:
    """The advisor's own walk-forward OOS probabilities for the shipped cell."""
    f = ROOT / "data" / "v5_runs" / "xau_advisor" / "oos_adv_cleanfeed.parquet"
    if not f.exists():
        raise SystemExit(f"missing {f} — run scripts/v5_advisor_feedcheck.py first")
    d = pd.read_parquet(f)
    d = d[d["cell"] == "1.0_9_BASE"].sort_values("time")
    d["time"] = pd.DatetimeIndex(d["time"])
    return d.set_index("time")


def build(windows, hold_h: int) -> pd.DataFrame:
    h4_filed, m15 = load_frames()
    F = frames(m15)
    verify_against_filed(F["H4"], h4_filed)          # raises if the feed changed under us
    sig = load_signal()
    atr = atr_series(F["H4"], 20)

    m15i = F["M15"].index
    hi, lo, cl = (F["M15"]["high"].values, F["M15"]["low"].values, F["M15"]["close"].values)

    fires = sig[(sig["p"] >= DOWN_AT) | (sig["p"] <= UP_AT)]
    rows = []
    for t, r in fires.iterrows():
        side = -1 if r["p"] >= DOWN_AT else +1          # -1 short (DOWN), +1 long (UP)
        # The decision is the H4 bar's CLOSE, not its stamp: an H4 bar stamped t closes at
        # t+4h. Reading it at t is the §3az stamp-vs-close bug that produced AUC 0.901.
        dec = t + pd.Timedelta(hours=4)
        a = atr.get(t, np.nan)
        if not np.isfinite(a) or a <= 0:
            continue
        s = int(np.searchsorted(m15i.values, np.datetime64(dec), side="left"))
        if s >= len(m15i) or m15i[s] != dec:
            # the decision instant must itself be an M15 bar, or the baseline entry price is
            # being invented rather than observed
            continue
        base_px = cl[s]
        exit_i = int(np.searchsorted(m15i.values,
                                     np.datetime64(dec + pd.Timedelta(hours=hold_h)),
                                     side="left"))
        if exit_i >= len(m15i):
            continue
        exit_px = cl[exit_i]
        row = dict(time=t, decision=dec, side=side, p=float(r["p"]), y=float(r["y"]),
                   atr=float(a), base_px=base_px, exit_px=exit_px,
                   base_ret=side * (exit_px - base_px) / base_px)
        for w in windows:
            e = int(np.searchsorted(m15i.values,
                                    np.datetime64(dec + pd.Timedelta(hours=w)), side="right"))
            e = min(e, exit_i)
            if e <= s + 1:
                row[f"or_{w}"] = np.nan
                row[f"an_{w}"] = np.nan
                continue
            seg_hi, seg_lo = hi[s + 1:e], lo[s + 1:e]
            # best entry: the window's HIGH for a short, its LOW for a long
            best = seg_hi.max() if side < 0 else seg_lo.min()
            worst = seg_lo.min() if side < 0 else seg_hi.max()
            row[f"or_{w}"] = side * (exit_px - best) / best
            row[f"an_{w}"] = side * (exit_px - worst) / worst
            row[f"or_px_{w}"] = best
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="0.5,1,2,4",
                    help="hours to wait for a confirmation before entering")
    ap.add_argument("--hold", type=int, default=9, help="hours to the common exit")
    a = ap.parse_args()
    windows = [float(x) for x in a.windows.split(",")]

    df = build(windows, a.hold)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "oracle.parquet")

    print("=" * 94)
    print(f"ORACLE CEILING for M15 entry timing — {len(df):,} fires, common exit at "
          f"decision+{a.hold}h")
    print("=" * 94)
    print(f"  baseline (enter at the H4 close): mean net-of-nothing return "
          f"{df.base_ret.mean()*100:+.4f}%")
    print(f"  mean ATR ${df.atr.mean():.2f}   mean price ${df.base_px.mean():.2f}   "
          f"spread ${SPREAD_USD:.2f} = {SPREAD_USD/df.base_px.mean()*100:.4f}%")
    print()
    print(f"{'window':>8} {'n':>5} {'ORACLE gain':>13} {'in ATR':>8} {'in $':>9} "
          f"{'vs spread':>10} {'ANTI gain':>11} {'band':>9}")
    for w in windows:
        c = df[[f"or_{w}", f"an_{w}", "base_ret", "atr", "base_px"]].dropna()
        if not len(c):
            continue
        g = (c[f"or_{w}"] - c["base_ret"])
        b = (c[f"an_{w}"] - c["base_ret"])
        g_usd = (g * c["base_px"]).mean()
        print(f"{w:8.1f} {len(c):5d} {g.mean()*100:+12.4f}% "
              f"{(g * c['base_px'] / c['atr']).mean():8.3f} {g_usd:+9.2f} "
              f"{g_usd / SPREAD_USD:9.1f}x {b.mean()*100:+10.4f}% "
              f"{(g.mean()-b.mean())*100:8.4f}%")

    print("\nTHE GATE. A detector can only ever capture a fraction of the oracle, and it must")
    print("clear the spread to be collectable. Read the 'vs spread' column: it is how many")
    print("spreads a PERFECT, cheating entry-timer would have won per trade.")

    w0 = windows[min(2, len(windows) - 1)]
    c = df[[f"or_{w0}", "base_ret", "base_px", "atr", "side", "p"]].dropna()
    g = c[f"or_{w0}"] - c["base_ret"]
    lo, hi, se = block_bootstrap_ci(g.values, block=20, alpha=0.10)
    print(f"\n  at the {w0}h window: oracle gain {g.mean()*100:+.4f}% "
          f"CI90 [{lo*100:+.4f}, {hi*100:+.4f}]  (block bootstrap, block=20)")
    for side, nm in ((-1, "DOWN/short"), (1, "UP/long")):
        s = c[c.side == side]
        if len(s):
            gs = s[f"or_{w0}"] - s["base_ret"]
            print(f"    {nm:11s} n {len(s):4d}  baseline {s.base_ret.mean()*100:+.4f}%  "
                  f"oracle gain {gs.mean()*100:+.4f}%  "
                  f"(${(gs * s.base_px).mean():+.2f} = "
                  f"{(gs * s.base_px).mean()/SPREAD_USD:.1f} spreads)")
    print(f"\nwrote {(OUT / 'oracle.parquet').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
