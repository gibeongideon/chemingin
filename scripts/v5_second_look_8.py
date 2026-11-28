#!/usr/bin/env python3
"""Second look #8: §3ak's single best trigger, evaluated ONCE on data it never saw (pre-registered in
data/v5_runs/second_look/PREREG.md). §3ak (2016-26 broker H4) found `trend=1 & mom24=0 & vol=2` -> LONG
24 bars: discovery non-overlapping t 1.43 (n 19), holdout t 0.70 (n 24), same sign; it died for lack of
events. OANDA gold 2007-2014 has never been used by that search.
Everything (state_vars, terciles on a trailing-2y rank, forward 24-bar log return, excess over the
unconditional mean, block-bootstrap null with the firing mask fixed, non-overlapping t, 2.76bp cost) is
imported unchanged from scripts/v5_trigger_search.py.
Pass: non-overlapping t >= 2 with the recorded (positive) sign.

    python scripts/v5_second_look_8.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from scripts.v5_trigger_search import COST_RT_BP, fwd_returns, state_vars, trigger_null_p  # noqa: E402
from v5_second_look_1 import s2_h4  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "second_look" / "8"
H = 24


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = s2_h4()
    df["tick_volume"] = 1.0                       # not part of this trigger; state_vars needs the column
    df = df[df.index < "2015-01-01"]
    S = state_vars(df)
    y = fwd_returns(df["close"], H)
    ok = S[["trend", "mom24", "vol"]].notna().all(axis=1) & y.notna()
    fire = ((S.trend == 1) & (S.mom24 == 0) & (S.vol == 2) & ok).to_numpy()
    yv = y[ok].to_numpy()
    mask = fire[ok.to_numpy()]
    p, t_nl, n_nl = trigger_null_p(mask, yv, +1.0, H)
    exc = (yv[mask].mean() - yv.mean()) * 1e4
    net = exc - COST_RT_BP
    rng = (df.index[ok.to_numpy()][0], df.index[-1])
    lines = [f"OANDA H4 gold, evaluated window {rng[0]:%Y-%m-%d} -> {rng[1]:%Y-%m-%d} (never seen by §3ak)",
             f"trigger trend=1 & mom24=0 & vol=2 -> LONG {H} H4 bars",
             f"  fires {int(mask.sum())}  non-overlapping n {n_nl}",
             f"  excess {exc:+.1f}bp  net {net:+.1f}bp (cost {COST_RT_BP:.2f}bp)",
             f"  block-bootstrap p {p:.3f}   NON-OVERLAPPING t {t_nl:+.2f}",
             f"  §3ak reference: discovery t +1.43 (n 19), holdout t +0.70 (n 24), same sign",
             f"VERDICT: {'REVIVED' if (t_nl >= 2 and exc > 0) else 'not revived'}"]
    print("\n".join(lines))
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
