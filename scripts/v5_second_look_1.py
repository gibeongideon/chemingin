#!/usr/bin/env python3
"""Second look #1: the §3p champion meta-label TRIM, re-judged (pre-registered in
data/v5_runs/second_look/PREREG.md).

§3p (2026-07-29) found the champ-meta trim lifted Sharpe 1.041 -> 1.338 and cut maxDD 16.9% -> 9.4%,
beat a block-shuffled-p null and a matched-constant control, and was REJECTED on a RAW paired t of
-2.11 (it holds less gold, so its raw daily delta is negative in a bull market). The repo later
established that de-risking overlays must be gated at MATCHED VOLATILITY (§3ab, §3ap, xau-champion-
five-lifts-fail). §3p also read XAUUSD_H4_long.csv, which carries an FTMO splice from 2023-06-28.

This script changes ONLY the data and the yardstick. Model, label, features, walk-forward, overlay
and engine are imported unchanged from scripts/v5_xau_turn_prob.py.
  S1 = H4 rebuilt from the single-feed XAUUSD_M15_long.csv (OOS 2018+, as §3p)
  S2 = H4 resampled from OANDA H1 mid (OOS 2012+; 2012-2017 never seen by §3p)
Statistics: raw paired t (old), MATCHED-VOL paired t (new), halves, yrs+, 60-draw block-shuffled-p
null, 2x cost, and the prop arm: GFT 2-step static pass rate at a 9% dial (v5_gft_check.challenge).

    python scripts/v5_second_look_1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import v5_xau_turn_prob as tp  # noqa: E402
from src.v5.mtf_frames import aggregate  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "second_look" / "1"
COST = 0.448
K, Q = 24, 0.30
BS = (1.0, 0.5)                     # PRIMARY first


def s1_h4() -> pd.DataFrame:
    m15 = pd.read_csv(ROOT / "data/XAUUSD_M15_long.csv", parse_dates=["time"], index_col="time").sort_index()
    m15 = m15[~m15.index.duplicated(keep="last")]
    h4 = aggregate(m15, "H4").dropna(subset=["close"])
    exp_med = h4["spread"].expanding(min_periods=20).median().bfill()
    h4["spread_px"] = np.maximum(np.maximum(h4["spread"], exp_med) * 0.1, COST)
    return h4


def s2_h4() -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "data/oanda_hf/XAUUSD_H1.parquet")
    d.index = pd.to_datetime(d.time, unit="s")
    d = d[["open", "high", "low", "close"]].sort_index()
    d = d[d.index >= "2007-01-01"]
    g = d.groupby(d.index.floor("4h"))
    h4 = pd.DataFrame(dict(open=g.open.first(), high=g.high.max(), low=g.low.min(), close=g.close.last()))
    h4["spread_px"] = COST
    return h4.dropna()


def matched_t(daily: pd.Series, base: pd.Series) -> tuple[float, float]:
    j = daily.index.intersection(base.index)
    a, b = daily.reindex(j).fillna(0.0), base.reindex(j).fillna(0.0)
    a = a * (b.std() / a.std())
    d = a - b
    return float(d.mean() / d.std() * np.sqrt(len(d))), float(d.mean() * 252)


def sharpe(r):
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0


def prop(daily: pd.Series) -> tuple:
    from v5_gft_check import challenge  # noqa: PLC0415
    r = daily.to_numpy()
    sd = r.std() * np.sqrt(252)
    return challenge(r, sd, 0.09, None, 0.0, False, False)   # pass%, months, day%, dd%, time%


def run(name: str, df: pd.DataFrame, oos_year: int) -> list[dict]:
    tp.OOS_START_YEAR = oos_year
    ev = f"{oos_year}-01-01"
    fc = champion_signal(df["close"])
    F = tp.build_features(df)
    ch_m, ch_d = tp.engine(df, fc, eval_start=ev)
    print(f"\n=== {name}: {df.index[0]:%Y-%m} -> {df.index[-1]:%Y-%m}, {len(df):,} H4 bars, OOS {ev}+ ===")
    print(f"  CHAMPION  SR {ch_m['sharpe']:+.3f}  CAGR {ch_m['cagr']:+.1f}%  maxDD {ch_m['dd']:+.1f}%  "
          f"(known-answer: §3p 1.041 on the spliced file)")
    tgt = tp.badness(df, fc, "champmeta", K)
    p, folds = tp.walk_forward_prob(df, F, K, Q, tgt)
    FD = pd.DataFrame(folds)
    print(f"  per-fold AUC: " + " ".join(f"{r.year}:{r.auc}" for r in FD.itertuples())
          + f"   mean {FD.auc.mean():.3f}")
    mid = ch_d.index[len(ch_d) // 2]
    base_prop = prop(ch_d)
    print(f"  prop (GFT 2-step static, 9% dial, no guard) CHAMPION: pass {base_prop[0]:.1f}%  "
          f"{base_prop[1]:.1f} mo  daily-fail {base_prop[2]:.1f}%  dd-fail {base_prop[3]:.1f}%")
    rows = []
    pf = p.fillna(0.0)
    for b in BS:
        m, d = tp.engine(df, fc * (1.0 - b * pf), eval_start=ev)
        _, t_raw, _ = tp.paired(d, ch_d)
        t_mv, da = matched_t(d, ch_d)
        h1 = matched_t(d[d.index < mid], ch_d[ch_d.index < mid])[0]
        h2 = matched_t(d[d.index >= mid], ch_d[ch_d.index >= mid])[0]
        yp, yn = tp.per_year(d * (ch_d.std() / d.std()), ch_d)
        rng = np.random.default_rng(5)
        nulls = [sharpe(tp.engine(df, fc * (1.0 - b * tp.block_shuffle(p, 30, rng).fillna(0.0)), eval_start=ev)[1])
                 for _ in range(60)]
        p95 = float(np.percentile(nulls, 95))
        # 2x cost: double spread and slippage
        df2 = df.copy(); df2["spread_px"] = df["spread_px"] * 2
        slip0 = tp.SLIP_USD; tp.SLIP_USD = slip0 * 2
        m2c, d2c = tp.engine(df2, fc * (1.0 - b * pf), eval_start=ev)
        _, c2d = tp.engine(df2, fc, eval_start=ev)
        tp.SLIP_USD = slip0
        t2c = matched_t(d2c, c2d)[0]
        pr = prop(d)
        sharpe_pass = (t_mv >= 2.0) and h1 > 0 and h2 > 0 and t2c > 0 and sharpe(d) > p95
        prop_pass = (pr[0] - base_prop[0] >= 5.0) and (pr[2] <= base_prop[2])
        print(f"  TRIM b={b}: SR {m['sharpe']:+.3f} (dSR {m['sharpe'] - ch_m['sharpe']:+.3f})  maxDD {m['dd']:+.1f}%  "
              f"CAGR {m['cagr']:+.1f}%  avg pos {m['avg_pos']:.2f} vs {ch_m['avg_pos']:.2f}")
        print(f"      raw paired t {t_raw:+.2f} (old yardstick) | MATCHED-VOL t {t_mv:+.2f} "
              f"(+{100 * da:.2f}%/yr at equal vol)  halves {h1:+.2f}/{h2:+.2f}  yrs+ {yp}/{yn}")
        print(f"      shuffled-p null p95 {p95:+.3f} vs {sharpe(d):+.3f} | 2x cost matched t {t2c:+.2f}")
        print(f"      prop: pass {pr[0]:.1f}% ({pr[0] - base_prop[0]:+.1f}pp)  {pr[1]:.1f} mo  daily-fail "
              f"{pr[2]:.1f}% (base {base_prop[2]:.1f}%)  dd-fail {pr[3]:.1f}%")
        print(f"      SHARPE ARM {'PASS' if sharpe_pass else 'fail'} | PROP ARM {'PASS' if prop_pass else 'fail'}")
        rows.append(dict(source=name, b=b, champ_sr=ch_m["sharpe"], sr=m["sharpe"], dd=m["dd"], champ_dd=ch_m["dd"],
                         t_raw=round(t_raw, 2), t_matched=round(t_mv, 2), h1=round(h1, 2), h2=round(h2, 2),
                         yrs_pos=yp, yrs=yn, null_p95=round(p95, 3), t_2x=round(t2c, 2), mean_auc=round(FD.auc.mean(), 3),
                         prop_pass=pr[0], prop_base=base_prop[0], prop_dayfail=pr[2], prop_base_dayfail=base_prop[2],
                         prop_months=pr[1], sharpe_arm=sharpe_pass, prop_arm=prop_pass))
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = run("S1 M15->H4 (single feed)", s1_h4(), 2018)
    rows += run("S2 OANDA H1->H4", s2_h4(), 2012)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "cells.csv", index=False)
    prim = R[R.b == 1.0]
    s_ok = bool(prim.iloc[0].t_matched >= 2.0 and prim.iloc[1].t_matched > 0
                and prim.sharpe_arm.iloc[0])
    p_ok = bool(prim.prop_arm.all())
    print(f"\nPRIMARY (b=1.0) VERDICT: Sharpe arm {'REVIVED' if s_ok else 'not revived'} | "
          f"prop arm {'REVIVED' if p_ok else 'not revived'}")
    print(f"wrote {OUT / 'cells.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
