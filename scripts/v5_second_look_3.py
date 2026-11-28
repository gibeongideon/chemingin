#!/usr/bin/env python3
"""Second look #3: §3bz probability-HYSTERESIS (long-only, P>0.65) re-judged on the PROP yardstick
(pre-registered in data/v5_runs/second_look/PREREG.md, addendum 2026-11-28).

§3bz reported the hysteresis long-only cell matched the "champion" Sharpe (+0.932) with maxDD -19.6% vs
-33.1% and turnover 33 vs 114/yr. But that "champion" ran H4 speeds on H1 bars with raw (not vol-targeted)
positions, and the hysteresis arm held far less exposure, so lower drawdown may just be lower size.
Here both arms are scaled to the SAME 9% vol dial in the prop simulator and compared at matched vol:
  hysteresis : model/features/walk-forward imported unchanged from scripts/v5_prob_hysteresis.py
               (XAUUSD_H1_long.csv, single broker), P>0.65 long, P<0.35 flat, else hold; cost as §3bz
  champion   : the deployed vol-targeted H4 champion on the same broker's M15->H4 (v5_second_look_1.s1_h4)

    python scripts/v5_second_look_3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import v5_prob_hysteresis as ph  # noqa: E402
import v5_xau_turn_prob as tp  # noqa: E402
from v5_second_look_1 import matched_t, prop, s1_h4, sharpe  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "second_look" / "3"


def dd(r):
    e = (1 + r).cumprod()
    return float((e / e.cummax() - 1).min() * 100)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = ph.load_h1()
    c = d["close"].astype(float)
    ret = c.pct_change()
    X = ph.features(d)
    y = (c.shift(-ph.HORIZON) > c).astype(float).where(c.shift(-ph.HORIZON).notna()).to_numpy()
    p = ph.walk_forward_proba(X, y, purge=ph.HORIZON)
    pos = pd.Series(ph.hysteresis(p, 0.65, True), index=ret.index).shift(1).fillna(0.0)
    net = (pos * ret - pos.diff().abs().fillna(0.0) * (ph.RT / 1e4)).dropna()
    first = ret.index[np.argmax(np.isfinite(p))]
    hyst = (1 + net[net.index >= first]).resample("D").prod().sub(1)
    hyst = hyst[hyst.index.dayofweek < 5]

    h4 = s1_h4()
    _, champ = tp.engine(h4, champion_signal(h4["close"]), eval_start=str(first.date()))
    j = hyst.index.intersection(champ.index)
    hyst, champ = hyst.reindex(j).fillna(0.0), champ.reindex(j).fillna(0.0)
    mid = j[len(j) // 2]
    t, da = matched_t(hyst, champ)
    h1, h2 = matched_t(hyst[:mid], champ[:mid])[0], matched_t(hyst[mid:], champ[mid:])[0]
    # known-answer: §3bz headline (raw, unscaled) Sharpe ~ +0.93, maxDD ~ -19.6%
    print(f"common window {j[0].date()} -> {j[-1].date()} ({len(j)} days)")
    print(f"  KNOWN ANSWER  hysteresis raw: SR {sharpe(hyst):+.3f} maxDD {dd(hyst):+.1f}% exposure {pos.abs().mean():.2f}"
          f"   (§3bz: +0.932 / -19.6%)")
    hm = hyst * (champ.std() / hyst.std())
    print(f"  champion H4 vol-targeted: SR {sharpe(champ):+.3f} maxDD {dd(champ):+.1f}%")
    print(f"  hysteresis AT CHAMPION VOL: SR {sharpe(hm):+.3f} maxDD {dd(hm):+.1f}%")
    print(f"  matched-vol paired t {t:+.2f} ({100 * da:+.2f}%/yr)  halves {h1:+.2f}/{h2:+.2f}  "
          f"corr {np.corrcoef(hyst, champ)[0, 1]:+.2f}")
    pc, pr = prop(champ), prop(hyst)
    print(f"  prop GFT 2-step static @9% dial: champion pass {pc[0]:.1f}% daily-fail {pc[2]:.1f}% dd-fail {pc[3]:.1f}% | "
          f"hysteresis pass {pr[0]:.1f}% daily-fail {pr[2]:.1f}% dd-fail {pr[3]:.1f}%")
    ok = (pr[0] - pc[0] >= 5) and (pr[2] <= pc[2]) and (t > -1)
    print(f"\nVERDICT: {'REVIVED (prop arm)' if ok else 'not revived'}")
    pd.DataFrame([dict(sr_hyst=sharpe(hyst), dd_hyst=dd(hyst), sr_champ=sharpe(champ), dd_champ=dd(champ),
                       dd_hyst_matched=dd(hm), t_matched=t, h1=h1, h2=h2, prop_champ=pc[0], prop_hyst=pr[0],
                       dayfail_champ=pc[2], dayfail_hyst=pr[2], revived=ok)]).to_csv(OUT / "cells.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
