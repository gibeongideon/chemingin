"""THE PROPOSED 6-COMPONENT STACK, tested component-by-component then stacked.

An external proposal (2026-09-03) recommended layering onto the champion, in order:
  1 volatility-regime filter        2 trend-strength ratio EWMA(r)/EWMA(|r|)
  3 volatility-scaled sizing        4 higher-timeframe regime confirmation
  5 time-of-day filter              6 transaction-cost-aware minimum-edge threshold
...and stacking them: EWMA -> trend strength -> vol regime -> HTF -> hours -> vol sizing.

STATUS BEFORE TESTING (verified in code, not assumed):
  #3 ALREADY IMPLEMENTED. Continuous engine: `pos = fc * (TARGET_VOL / vol)`
     (v5_xau_turn_prob.py:94). Deployed discrete engine: `lots = risk*eq /
     (sl_atr*ATR)` (xau_trend.py:167,175). This is exactly target_vol/current_vol.
  #6 ALREADY IMPLEMENTED (partially). A causal no-trade band, `BUFFER = 0.1`, scaled
     by the vol target (v5_xau_turn_prob.py:52,96-97) — the position only moves when the
     move exceeds the band, which is the cost-aware rule. Its WIDTH is swept below.
  #1 and #4 are close relatives of two already-disproven results: §3u (ADX-percentile
     regime GATE on this exact champion: SR 1.084->1.007, DD WORSE, paired-t -2.88, FTMO
     pass 95%->80%) and §3ab idea 2 (agreement across the champion's own 6 horizons,
     t -0.63). They are still tested here in the PROPOSED form (vol LEVEL rather than
     trend strength; a genuinely higher timeframe rather than internal horizons).
  #5's prior verdict is "session/regime/carry/RL all failed" (memory
     sharpe-to-12-fast-basket) but was never published as a per-hour attribution, and the
     proposal explicitly says measure rather than assume — so it is measured here.

THE GATE (§3ab): every component below multiplies exposure by <=1, i.e. DE-RISKS, so a
raw paired-t marks risk-adjusted gains as losses. Each candidate is levered to the base's
realised vol first, then compared (`t@matched`). And per §3ab's hard lesson — a pooled
t of +2.25 with 7/9 years STILL turned out to be a 2022+ artifact — anything that passes
must also survive the SPLIT-SAMPLE, which is printed for every survivor.

    python scripts/v5_xau_stack_proposal.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_xau_turn_prob import load_h4, paired, per_year  # noqa: E402
from scripts.v5_xau_champion_lifts import (  # noqa: E402
    engine_v, vol_match, sharpe, dd_of, ANN_H4, VOL_HL, EVAL_START, COST_USD, XAU_REGIMES,
)
from src.v5.xau_dual_signals import champion_signal, ewmac_fc  # noqa: E402
from src.evaluation.walk_forward_grid import regime_split_sharpe  # noqa: E402

SPLITS = [("2018-01-01", "2021-12-31", "2018-21"), ("2022-01-01", "2026-12-31", "2022-26")]
PCT_WIN = 252 * 6 * 2       # trailing 2y for causal percentile ranks


def gate(d: pd.Series, base: pd.Series) -> tuple[float, str]:
    dm = vol_match(d, base)
    _, t, _ = paired(dm, base)
    yp, yn = per_year(dm, base)
    return t, f"{yp}/{yn}"


def split_line(d: pd.Series, base: pd.Series) -> str:
    out = []
    for s0, s1, lbl in SPLITS:
        ds, bs = d.loc[s0:s1], base.loc[s0:s1]
        t, yrs = gate(ds, bs)
        out.append(f"{lbl}: SR{sharpe(ds):+.2f} t{t:+.2f} {yrs}")
    return "   |   ".join(out)


def pct_rank(s: pd.Series, window: int = PCT_WIN) -> pd.Series:
    """Causal trailing percentile rank of the current value in its own history."""
    return s.rolling(window, min_periods=252).apply(lambda x: (x[-1] > x).mean(), raw=True)


def main() -> None:
    df = load_h4(COST_USD)
    close = df["close"]
    ret = close.pct_change()
    fc = champion_signal(close)
    base = engine_v(df, fc)
    base_sr = sharpe(base)
    print(f"XAUUSD H4, eval {EVAL_START}+, cost ${COST_USD}")
    print(f"BASE champion: SR={base_sr:+.3f}  DD={dd_of(base):+.1f}%   "
          f"turnover is ~24 position-changes/yr (a WEEKS-long holder)\n")

    rows = []

    def run(label: str, comp: str, fcx: pd.Series | None = None, **kw):
        d = engine_v(df, (fcx if fcx is not None else fc).clip(0, 2), **kw)
        t, yrs = gate(d, base)
        rows.append(dict(component=comp, label=label, sr=sharpe(d), dd=dd_of(d),
                         t=t, years=yrs, series=d))
        print(f"  {label:38} SR={sharpe(d):+.3f}  DD={dd_of(d):+.1f}%  "
              f"t@matched={t:+.2f}  years {yrs}")
        return d

    # ---------------------------------------------------------------- #1 vol regime
    print("=== #1 VOLATILITY-REGIME FILTER (the proposal's first choice) ===")
    # the proposal's own formula: sigma_t^2 = lam*sigma_{t-1}^2 + (1-lam)*r_t^2
    var_ewma = ret.pow(2).ewm(halflife=VOL_HL, min_periods=20).mean()
    vol_ewma = np.sqrt(var_ewma) * np.sqrt(ANN_H4)
    vr = pct_rank(vol_ewma)

    # REDUNDANCY DIAGNOSTIC — this is the crux for #1
    pos_base = (fc.clip(0, 2) * (0.10 / vol_ewma)).clip(0, 8)
    q_lo, q_hi = vr <= 0.2, vr >= 0.8
    print(f"  redundancy check: the engine ALREADY sizes 1/vol, so extreme vol is already")
    print(f"    de-weighted. mean position in the HIGHEST vol quintile "
          f"{pos_base[q_hi].mean():.2f} vs the LOWEST {pos_base[q_lo].mean():.2f} "
          f"= {pos_base[q_lo].mean()/max(pos_base[q_hi].mean(),1e-9):.1f}x smaller already.")
    print(f"    corr(vol percentile, existing position) = "
          f"{float(pd.concat([vr, pos_base], axis=1).dropna().corr().iloc[0,1]):+.2f}")

    schedules = {
        "as proposed (lo.5/norm1/hi.5/ext0)": {0: 0.5, 1: 1.0, 2: 1.0, 3: 0.5, 4: 0.0},
        "cut extreme only (q5=0)":            {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.0},
        "cut low only (q1=0)":                {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
        "cut both tails":                     {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.0},
        "monotone taper 1.0->0.2":            {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2},
    }
    q = np.clip((vr * 5).fillna(2).astype(int), 0, 4)
    for name, sched in schedules.items():
        m = q.map(sched).astype(float)
        run(name, "1_volregime", fc * m)

    # ---------------------------------------------------------------- #2 trend strength
    print("\n=== #2 TREND-STRENGTH RATIO  EWMA(r)/EWMA(|r|) ===")
    print("  collinearity check FIRST — if it duplicates the champion it cannot add:")
    ts_by_hl = {}
    for hl in (12, 24, 48, 120):
        ts = (ret.ewm(halflife=hl, min_periods=20).mean()
              / ret.abs().ewm(halflife=hl, min_periods=20).mean())
        ts_by_hl[hl] = ts
        j = pd.concat([ts.rename("ts"), fc.rename("fc")], axis=1).dropna()
        print(f"    hl={hl:>3}  corr(TrendStrength, champion forecast) = "
              f"{float(j.corr().iloc[0,1]):+.2f}   (champion's EWMAC is itself a "
              f"signal/noise ratio)")
    for hl in (24, 48, 120):
        for thr in (0.0, 0.1, 0.2):
            ts = ts_by_hl[hl]
            m = (ts > thr).astype(float)
            run(f"hl={hl} require TS>{thr}", "2_trendstrength", fc * m)

    # ---------------------------------------------------------------- #4 HTF regime
    print("\n=== #4 HIGHER-TIMEFRAME REGIME CONFIRMATION ===")
    d1_close = close.resample("D").last().dropna()
    for name, speeds in {"D1 ewmac(16,64)": ((16, 64),),
                         "D1 ewmac(32,128)": ((32, 128),),
                         "D1 ewmac(8,32)": ((8, 32),)}.items():
        htf = ewmac_fc(d1_close, speeds)
        # causal: only the PREVIOUS completed daily bar is knowable intraday
        htf_h4 = htf.shift(1).reindex(close.index, method="ffill")
        for red in (0.0, 0.5):
            m = np.where(htf_h4 > 0, 1.0, red)
            run(f"{name} agree else x{red}", "4_htf", fc * pd.Series(m, index=close.index))

    # ---------------------------------------------------------------- #5 session
    print("\n=== #5 TIME-OF-DAY — MEASURED, per the proposal's own instruction ===")
    pos = engine_v.__wrapped__ if False else None
    # attribute the base strategy's daily P&L to the H4 bar-of-day that produced it
    vol_b = ret.ewm(halflife=VOL_HL, min_periods=20).std() * np.sqrt(ANN_H4)
    p = (fc.clip(0, 2) * (0.10 / vol_b)).clip(0, 8).shift(1).fillna(0.0)
    bar_pnl = (p * ret).fillna(0.0)
    print(f"  {'hour(UTC)':>10} {'bars':>7} {'total P&L':>11} {'mean bp':>9} {'hit%':>7}")
    hrs = sorted(close.index.hour.unique())
    for h in hrs:
        m = close.index.hour == h
        seg = bar_pnl[m]
        print(f"  {h:>10} {int(m.sum()):>7} {seg.sum()*100:>+10.1f}% "
              f"{seg.mean()*1e4:>+9.2f} {float((seg>0).mean()*100):>6.1f}")
    print("  STRUCTURAL NOTE: the champion changes position ~24x/YEAR and holds for weeks.")
    print("  An hours filter cannot 'trade only London' without forcing intraday exit/re-entry,")
    print("  which multiplies turnover against a $0.448 spread. Tested anyway:")
    best_hours = sorted(bar_pnl.groupby(close.index.hour).sum().nlargest(3).index)
    m_h = pd.Series(np.where(np.isin(close.index.hour, best_hours), 1.0, 0.0),
                    index=close.index)
    run(f"hold only in best hours {best_hours}", "5_session", fc * m_h)

    # ---------------------------------------------------------------- #6 cost band
    print("\n=== #6 COST-AWARE MINIMUM EDGE — the band already exists; sweep its width ===")
    for b in (0.0, 0.05, 0.10, 0.20, 0.40):
        d = engine_v(df, fc, buffer_frac=b)
        t, yrs = gate(d, base)
        tag = " <- DEPLOYED" if abs(b - 0.10) < 1e-9 else ""
        rows.append(dict(component="6_costband", label=f"buffer={b}", sr=sharpe(d),
                         dd=dd_of(d), t=t, years=yrs, series=d))
        print(f"  buffer_frac={b:<5} SR={sharpe(d):+.3f}  DD={dd_of(d):+.1f}%  "
              f"t@matched={t:+.2f}  years {yrs}{tag}")
    for k in (0.3, 0.5, 0.8):
        run(f"min-edge: trade only fc>{k}", "6_costband", fc.where(fc > k, 0.0))

    # ---------------------------------------------------------------- the FULL STACK
    print("\n=== THE FULL STACK, exactly as proposed (all components multiplied) ===")
    R = pd.DataFrame([{k: v for k, v in r.items() if k != "series"} for r in rows])
    best = {}
    for comp, sub in R.groupby("component"):
        best[comp] = sub.loc[sub["t"].idxmax()]
    for comp, row in best.items():
        print(f"  best of {comp:16} -> {row['label']:38} t={row['t']:+.2f}")

    m_stack = pd.Series(1.0, index=close.index)
    # vol regime (best schedule), trend strength (best hl/thr), HTF (best), hours (best)
    vb = best["1_volregime"]["label"]
    m_stack *= q.map(schedules[vb]).astype(float)
    tb = best["2_trendstrength"]["label"]
    hl_b = int(tb.split("hl=")[1].split(" ")[0]); thr_b = float(tb.split("TS>")[1])
    m_stack *= (ts_by_hl[hl_b] > thr_b).astype(float)
    hb = best["4_htf"]["label"]
    sp = {"D1 ewmac(16,64)": ((16, 64),), "D1 ewmac(32,128)": ((32, 128),),
          "D1 ewmac(8,32)": ((8, 32),)}[hb.split(" agree")[0]]
    red_b = float(hb.split("else x")[1])
    htf_b = ewmac_fc(d1_close, sp).shift(1).reindex(close.index, method="ffill")
    m_stack *= pd.Series(np.where(htf_b > 0, 1.0, red_b), index=close.index)
    m_stack *= m_h if best["5_session"]["t"] > 0 else 1.0

    d_stack = engine_v(df, (fc * m_stack).clip(0, 2))
    t_s, yrs_s = gate(d_stack, base)
    _, t_raw, _ = paired(d_stack, base)
    print(f"\n  STACK: SR={sharpe(d_stack):+.3f}  DD={dd_of(d_stack):+.1f}%  "
          f"t@matched={t_s:+.2f} (raw {t_raw:+.2f})  years {yrs_s}")
    print(f"  BASE : SR={base_sr:+.3f}  DD={dd_of(base):+.1f}%")
    print(f"  mean exposure retained by the stack: {float(m_stack.mean()*100):.0f}% "
          f"(time fully flat: {float((m_stack == 0).mean()*100):.0f}% of bars)")

    # ---------------------------------------------------------------- verdict
    print(f"\n{'='*80}\nVERDICT — gate is t@matched > +1.64, then MUST survive the split\n{'='*80}")
    winners = R[R["t"] > 1.64].sort_values("t", ascending=False)
    print(f"components beating base at t@matched > +1.64: {len(winners)}/{len(R)}")
    if len(winners):
        by_series = {r["label"]: r["series"] for r in rows}
        for _, w in winners.iterrows():
            print(f"\n  {w['label']}  (SR {w['sr']:+.3f}, t {w['t']:+.2f})")
            print(f"    SPLIT: {split_line(by_series[w['label']], base)}")
    else:
        print("  none — so there is nothing to split-test.")
    print(f"\n  stack t@matched = {t_s:+.2f}")
    if t_s > 1.64:
        print(f"    SPLIT: {split_line(d_stack, base)}")

    print("\nregime split (stack vs base):")
    rs, rb = regime_split_sharpe(d_stack, XAU_REGIMES), regime_split_sharpe(base, XAU_REGIMES)
    for k in rs:
        print(f"  {k:24s} stack {rs[k]:+.2f}   base {rb.get(k, float('nan')):+.2f}")

    out = ROOT / "data" / "v5_runs" / "xau_stack_proposal.csv"
    R.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
