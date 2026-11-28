#!/usr/bin/env python3
"""Second look PHASE D: put the round's survivor into the DEPLOYED XAU+BTC book (pre-registered addendum in
data/v5_runs/second_look/PREREG.md). Survivor = #1 champion meta-trim at b=0.5 on the gold sleeve.
The book is rebuilt exactly as v5_account_chooser_2026_10.xau_btc_book, with the gold forecast optionally
multiplied by (1 - 0.5 * p_bad), p_bad from #1's S1 walk-forward. Prop arms use v5_gft_check.challenge.

    python scripts/v5_second_look_d.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import v5_xau_turn_prob as tp  # noqa: E402
from v5_gft_check import challenge  # noqa: E402
from v5_second_look_1 import K, Q, matched_t, s1_h4, sharpe  # noqa: E402
from v5_xau_champion_lifts import champion_recipe  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "second_look" / "D"


def book(mult: pd.Series | None) -> pd.Series:
    """xau_btc_book, verbatim, with an optional multiplier on the gold forecast."""
    h4 = s1_h4()
    legs = []
    for name, close, scale, sp, carry in (
            ("xau", h4["close"].astype(float), 1.0, 0.50, -5.76),
            ("btc", pd.read_csv(ROOT / "data/BTC_D1_long.csv", parse_dates=["time"], index_col="time")
             .sort_index()["close"].astype(float), 1 / 6, 4.46, -15.0)):
        close = close[~close.index.duplicated(keep="last")]
        bpy = len(close) / max((close.index[-1] - close.index[0]).days / 365.25, 1e-9)
        ret = close.pct_change()
        vol = ret.ewm(halflife=42, min_periods=20).std() * np.sqrt(bpy)
        fc = champion_recipe(close, scale, 1.5).clip(-2, 2)
        if name == "xau" and mult is not None:
            fc = fc * mult.reindex(fc.index).fillna(1.0)
        pos = (fc * (0.10 / vol)).clip(-8, 8).shift(1)
        r = (pos * ret - pos.diff().abs().fillna(0.0) * (sp / 1e4) * 1.5 - pos.abs() * (abs(carry) / 100.0) / bpy).dropna()
        legs.append((1 + r).resample("1D").prod().sub(1))
    bd = pd.bdate_range(max(l.index[0] for l in legs).normalize(), min(l.index[-1] for l in legs).normalize())
    return (0.5 * legs[0].reindex(bd).fillna(0.0) + 0.5 * legs[1].reindex(bd).fillna(0.0)).dropna()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    h4 = s1_h4()
    tp.OOS_START_YEAR = 2018
    fc = champion_signal(h4["close"])
    p, _ = tp.walk_forward_prob(h4, tp.build_features(h4), K, Q, tp.badness(h4, fc, "champmeta", K))
    mult = 1.0 - 0.5 * p.fillna(0.0)
    base, trim = book(None), book(mult)
    j = base.loc["2018-01-01":].index.intersection(trim.index)
    b, t = base.reindex(j), trim.reindex(j)
    tm, da = matched_t(t, b)
    mid = j[len(j) // 2]
    h1, h2 = matched_t(t[:mid], b[:mid])[0], matched_t(t[mid:], b[mid:])[0]
    e = lambda r: float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min() * 100)  # noqa: E731
    lines = [f"PHASE D: deployed XAU+BTC book vs the same book with the gold sleeve trimmed (b=0.5), {j[0].date()} -> {j[-1].date()}",
             f"  KNOWN ANSWER: base book SR {sharpe(b):+.3f} (account-chooser book on this window; full-history ~1.28)",
             f"  base  SR {sharpe(b):+.3f}  maxDD {e(b):+.1f}%",
             f"  trim  SR {sharpe(t):+.3f}  maxDD {e(t):+.1f}%",
             f"  matched-vol paired t {tm:+.2f} ({100 * da:+.2f}%/yr at equal vol)  halves {h1:+.2f}/{h2:+.2f}"]
    res = {}
    for lab, g, s, cs in (("raw left tail (no guard)", None, 0.0, False), ("deployed GFT setup (guard 2% + cushion)", 0.02, 0.002, True)):
        from v5_gft_check import challenge as ch  # noqa: PLC0415
        out = []
        for name, r in (("base", b), ("trim", t)):
            rv = r.to_numpy()
            out.append(ch(rv, rv.std() * np.sqrt(252), 0.09, g, s, False, cs))
        res[lab] = out
        lines.append(f"  prop GFT 2-step static @9%, {lab}: base pass {out[0][0]:.1f}% ({out[0][1]:.1f} mo, daily-fail "
                     f"{out[0][2]:.1f}%) | trim pass {out[1][0]:.1f}% ({out[1][1]:.1f} mo, daily-fail {out[1][2]:.1f}%)")
    dep = res["deployed GFT setup (guard 2% + cushion)"]
    ok_s = tm >= 2 and h1 > 0 and h2 > 0
    ok_p = (dep[1][0] - dep[0][0] >= 5) and (dep[1][2] <= dep[0][2]) and h1 > 0 and h2 > 0
    lines.append(f"VERDICT: Sharpe arm {'PASS' if ok_s else 'fail'} | prop arm {'PASS' if ok_p else 'fail'}")
    print("\n".join(lines))
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
