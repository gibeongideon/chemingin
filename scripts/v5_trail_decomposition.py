"""GATE B — decompose the champion's trailing stop into its three separate parameters.

WHY THIS ARM EXISTS. The live Maven config (`configs/v5_xau_maven.json`, magic 360571) ships
`sl_atr: 3.0, trail_atr: 3.0` with `trail_activation_atr` unset, so it silently defaults to 1.0
(`src/v5/xau_trend.py:70`). The 3.0 came from §3f (`V5_FINDINGS.md:394`), a sweep in which
`scripts/v5_multi_tf_trend.py:117-118` passes `sl_atr=trail_atr, trail_atr=trail_atr` -- the
protective stop and the trail distance MOVED TOGETHER. So the number in production was never
separated from the one it was confounded with, and §3f's own verdict line reads "sharp parameter
peak ... a robust edge has a flat surface."

Separately: §3aa's 64 exit cells contained NO RATCHET at all and resolved on closes rather than
the path, so "does ratcheting beat not-ratcheting on a signal that works" has never been asked.

THE GATE IS A SHAPE TEST, NOT AN ARGMAX. In this repo a sharp peak is a FAILURE signal -- §3f
killed H4/med precisely because it peaked (0.82/1.02/1.27/1.14/1.02) and §3ab found
parameter-cloud averaging HURTS the champion. A plateau qualifies; a peak disqualifies.

FIDELITY. `scripts/v5_maven_champ.py:99` runs the live bot by monkeypatching
`xau_trend.xau_signal` to `champion_signal` (LONG-ONLY, .clip(0,2)) and calling `run_trades`.
This script does exactly the same thing, so the incumbent cell here IS the live configuration.
S3 swaps in `ls_signal` to measure the long/short variant the user asked about.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import src.v5.xau_trend as xt                                    # noqa: E402
from src.v5.xau_dual_signals import SIGNALS                      # noqa: E402
from src.v5 import mtf_frames                                    # noqa: E402
from src.evaluation.dsr_pbo import pbo_cscv                      # noqa: E402
from v5_xau_champion_lifts import vol_match                      # noqa: E402

OUT = ROOT / "data/v5_runs/trail_decomp"
INCUMBENT = dict(sl_atr=3.0, trail_atr=3.0, trail_activation_atr=1.0)
EVAL_START = "2018-01-01"


def clean_h4() -> pd.DataFrame:
    """H4 rebuilt from M15. NEVER data/XAUUSD_H4_long.csv -- §3bb measured its agreement with
    M15 at 0.04 from 2023-06-28 (FTMO quotes spliced in by refresh_xau_h4.py)."""
    m15 = pd.read_csv(ROOT / "data/XAUUSD_M15_long.csv", parse_dates=["time"],
                      index_col="time").sort_index()
    m15 = m15[~m15.index.duplicated(keep="last")]
    h4 = mtf_frames.aggregate(m15, "H4")
    filed = pd.read_csv(ROOT / "data/XAUUSD_H4_long.csv", parse_dates=["time"],
                        index_col="time").sort_index()
    v = mtf_frames.verify_against_filed(h4, filed, clean_until=2022, tol=0.10)
    print(f"  H4 rebuilt from M15: {len(h4):,} bars, {h4.index[0].date()}..{h4.index[-1].date()}")
    print(f"  verify_against_filed (pre-2023): n={v.get('n')} median|d|={v.get('median_abs', 0):.4f}")
    return h4[["open", "high", "low", "close", "spread"]].dropna()


def daily_returns(res: dict) -> pd.Series:
    eq = res["equity"].ffill().dropna()
    return eq.resample("1D").last().dropna().pct_change().dropna()


def run(h4: pd.DataFrame, params: dict, signal: str = "champ",
        exit_mode: str = "trail") -> pd.Series:
    xt.xau_signal = lambda close: SIGNALS[signal](close)     # the live bot's own pattern
    res = xt.run_trades(h4, equity0=100_000.0, exit_mode=exit_mode,
                        flip_mode="confidence", params={**INCUMBENT, **params})
    return daily_returns(res)


def stats(r: pd.Series, base: pd.Series, bdays) -> dict:
    a = r.reindex(bdays, fill_value=0.0)
    b = base.reindex(bdays, fill_value=0.0)
    am = vol_match(a, b)                       # rule 1.3.1 -- a wider trail DE-RISKS
    d = am - b
    t = float(d.mean() / d.std() * np.sqrt(len(d))) if d.std() > 0 else np.nan
    def sh(x):
        return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else np.nan
    eq = (1 + a).cumprod()
    h1 = a[a.index.year <= 2021]; h2 = a[a.index.year >= 2022]
    hb1 = b[b.index.year <= 2021]; hb2 = b[b.index.year >= 2022]
    return dict(sharpe=round(sh(a), 3), cagr=round(100 * (eq.iloc[-1] ** (252 / len(a)) - 1), 2),
                maxdd=round(100 * float((eq / eq.cummax() - 1).min()), 1),
                t_matched=round(t, 2),
                half_18_21=round(sh(h1) - sh(hb1), 3), half_22_26=round(sh(h2) - sh(hb2), 3),
                both_halves=bool(sh(h1) > sh(hb1) and sh(h2) > sh(hb2)))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 112)
    print("GATE B — trailing-stop decomposition on the CHAMPION (the live Maven configuration)")
    print("=" * 112)
    h4 = clean_h4()
    h4 = h4[h4.index >= EVAL_START]

    base = run(h4, {})
    bdays = pd.bdate_range(base.index[0], base.index[-1])
    print(f"  incumbent sl 3.0 / trail 3.0 / activation 1.0 — "
          f"{len(base)} days, {base.index[0].date()}..{base.index[-1].date()}")

    rows, series = [], {}
    axes = {"sl_atr": (2.0, 3.0, 4.0),
            "trail_atr": (1.5, 2.0, 3.0, 4.0, 6.0),
            "trail_activation_atr": (0.0, 1.0, 2.0, 3.0)}
    for axis, cells in axes.items():
        for v in cells:
            key = f"{axis}={v}"
            r = base if v == INCUMBENT[axis] else run(h4, {axis: v})
            series[key] = r
            s = stats(r, base, bdays)
            s.update(axis=axis, cell=v, incumbent=(v == INCUMBENT[axis]), arm=key)
            rows.append(s)

    # secondary: does ratcheting beat not-ratcheting at all? (never asked -- §3aa had no ratchet)
    for em in ("flip", "sltp"):
        r = run(h4, {}, exit_mode=em)
        series[f"exit={em}"] = r
        s = stats(r, base, bdays); s.update(axis="exit_mode", cell=em, incumbent=False,
                                            arm=f"exit={em}")
        rows.append(s)

    # side arm: the short-leg question, measured not cited
    r = run(h4, {}, signal="ls")
    series["S3 long/short"] = r
    s = stats(r, base, bdays); s.update(axis="side", cell="long/short", incumbent=False,
                                        arm="S3 long/short")
    rows.append(s)

    tb = pd.DataFrame(rows)[["arm", "axis", "cell", "incumbent", "sharpe", "cagr", "maxdd",
                             "t_matched", "half_18_21", "half_22_26", "both_halves"]]
    for axis in list(axes) + ["exit_mode", "side"]:
        s = tb[tb.axis == axis]
        if len(s):
            print(f"\n--- {axis} (incumbent marked) ---")
            print(s.drop(columns=["axis"]).to_string(index=False))

    # PBO over the declared grid
    m = min(len(v.reindex(bdays, fill_value=0.0)) for v in series.values())
    M = np.column_stack([v.reindex(bdays, fill_value=0.0).values[:m] for v in series.values()])
    pbo = pbo_cscv(M, n_partitions=10)
    print(f"\n  PBO over the {M.shape[1]}-cell grid: {pbo.pbo:.3f}  (bar < 0.30)")

    tb.to_csv(OUT / "gate_b_profiles.csv", index=False)
    pd.DataFrame({k: v for k, v in series.items()}).to_csv(OUT / "gate_b_daily.csv")
    json.dump(dict(pbo=float(pbo.pbo), n_cells=int(M.shape[1])),
              open(OUT / "gate_b_pbo.json", "w"), indent=2)
    print(f"\n  wrote {OUT}/gate_b_profiles.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
