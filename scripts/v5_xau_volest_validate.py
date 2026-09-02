"""IDEA 1 DEEP VALIDATION — is the range-based vol denominator a REAL lift, and
does it reach the LIVE bot? XAUUSD H4 (+ EURUSD/GBPUSD/USDJPY/BTC/NDX/SPX externally).

`v5_xau_champion_lifts.py` screened five untried mechanisms against the champion. Four
failed the matched-vol gate. One passed, cleanly and as a FAMILY rather than a lucky cell:
replacing the engine's close-to-close EWMA vol denominator with a range-based estimator.

    estimator          SR      t@matched-vol   years better
    close2close(base)  +1.084   —              —
    parkinson          +1.117   +1.63          7/9
    garman_klass       +1.131   +1.75          7/9
    rogers_satchell    +1.149   +2.11          7/9
    yang_zhang         +1.145   +2.25          7/9

All four beat the base, 7/9 years each, ordered by their known statistical efficiency —
the signature of a real effect, not selection. Note the claim needs NO cell selection:
the estimator is chosen a priori on theory (Yang-Zhang: handles overnight gaps AND drift;
Rogers-Satchell: drift-robust), so this is not a "best of N" result.

THE CATCH, CHECKED HEAD-ON. The lift is measured in the CONTINUOUS research engine, which
divides by close-to-close EWMA vol. The DEPLOYED bot (`xau_trend.run_trades`, cent magic
360542 / Maven 360571) does NOT: it sizes off `wilder_atr`, i.e. TRUE range including
|H-prevC| / |L-prevC|, which is already range-based AND already gap-aware. So part of this
"lift" may be a fix to the research harness rather than money in the live book. Part B
tests the actionable version directly: in the DISCRETE engine, does an efficient vol
estimator beat Wilder ATR for the stop/sizing distance? Level-calibrated to ATR's mean so
the test isolates the SHAPE/TIMING of the estimate, not its magnitude (a magnitude change
is just risk_frac, already mapped in §3z).

    python scripts/v5_xau_volest_validate.py
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

import src.v5.xau_trend as xt  # noqa: E402
from scripts.v5_xau_turn_prob import load_h4, paired, per_year  # noqa: E402
import scripts.v5_xau_champion_lifts as lifts  # noqa: E402
from scripts.v5_xau_champion_lifts import (  # noqa: E402  (reuse, no duplication)
    engine_v, vol_estimators, vol_match, sharpe, dd_of, ANN_H4, VOL_HL, XAU_REGIMES,
    EVAL_START, COST_USD,
)
from src.v5.xau_dual_signals import champion_signal  # noqa: E402
from src.evaluation.dsr_pbo import deflated_sharpe_ratio, pbo_cscv  # noqa: E402
from src.evaluation.walk_forward_grid import regime_split_sharpe  # noqa: E402

A_PRIORI = "yang_zhang"          # fixed on theory BEFORE seeing part-B results
SPLITS = [("2018-01-01", "2021-12-31", "first half"),
          ("2022-01-01", "2026-12-31", "second half")]


def t_matched(d: pd.Series, base: pd.Series) -> tuple[float, str]:
    dm = vol_match(d, base)
    _, t, _ = paired(dm, base)
    yp, yn = per_year(dm, base)
    return t, f"{yp}/{yn}"


def main() -> None:
    df = load_h4(COST_USD)
    close = df["close"]
    fc = champion_signal(close)
    vols = vol_estimators(df)
    base = engine_v(df, fc)
    print(f"XAUUSD H4, eval {EVAL_START}+, cost ${COST_USD}. "
          f"BASE (close2close vol) SR={sharpe(base):+.3f} DD={dd_of(base):+.1f}%")
    print(f"a-priori estimator = {A_PRIORI} (chosen on theory, not on the numbers)\n")

    # ---------------- A1: does the WHOLE FAMILY hold in BOTH halves?
    print("=== A1: split-sample — the family must hold in BOTH halves, not just pooled ===")
    print(f"{'estimator':20} " + " ".join(f"{lbl:>26}" for _, _, lbl in SPLITS))
    for vname, v in vols.items():
        d = engine_v(df, fc, vol=v)
        cells = []
        for s0, s1, _ in SPLITS:
            ds, bs = d.loc[s0:s1], base.loc[s0:s1]
            t, yrs = t_matched(ds, bs)
            cells.append(f"SR{sharpe(ds):+.2f} t{t:+.2f} {yrs:>5}")
        print(f"{vname:20} " + " ".join(f"{c:>26}" for c in cells))

    # ---------------- A2: cost robustness
    print("\n=== A2: cost robustness (the a-priori estimator vs base at each spread) ===")
    print(f"{'spread':>10} {'base SR':>9} {A_PRIORI + ' SR':>16} {'t@matched':>10} {'years':>7}")
    for cost in (0.12, 0.34, 0.448, 0.60):
        d2 = load_h4(cost)
        v2 = vol_estimators(d2)
        b2 = engine_v(d2, champion_signal(d2["close"]))
        a2 = engine_v(d2, champion_signal(d2["close"]), vol=v2[A_PRIORI])
        t, yrs = t_matched(a2, b2)
        print(f"${cost:>9.3f} {sharpe(b2):>+9.3f} {sharpe(a2):>+16.3f} {t:>+10.2f} {yrs:>7}")

    # ---------------- A3: vol-halflife robustness (a fluke would need hl=42 specifically)
    print("\n=== A3: halflife robustness — must not depend on the incumbent hl=42 ===")
    print(f"{'halflife':>9} {'base SR':>9} {A_PRIORI + ' SR':>16} {'t@matched':>10} {'years':>7}")
    for hl in (21, 42, 84, 126):
        vb = close.pct_change().ewm(halflife=hl, min_periods=20).std() * np.sqrt(ANN_H4)
        va = vol_estimators(df, hl=hl)[A_PRIORI]
        b3 = engine_v(df, fc, vol=vb)
        a3 = engine_v(df, fc, vol=va)
        t, yrs = t_matched(a3, b3)
        print(f"{hl:>9} {sharpe(b3):>+9.3f} {sharpe(a3):>+16.3f} {t:>+10.2f} {yrs:>7}")

    # ---------------- A4: deflation over the 5-cell family only (honest trial count)
    print("\n=== A4: deflation over the 5-estimator family (n_trials=5, the real search) ===")
    fam = {k: engine_v(df, fc, vol=v) for k, v in vols.items()}
    trials = np.array([sharpe(v) / np.sqrt(252) for v in fam.values()])
    d_ap = fam[A_PRIORI]
    dsr = deflated_sharpe_ratio(vol_match(d_ap, base).values, trials)
    common = None
    for v in fam.values():
        common = v.index if common is None else common.intersection(v.index)
    pbo = pbo_cscv(np.column_stack([v.reindex(common).fillna(0.0).values
                                    for v in fam.values()]), n_partitions=10)
    print(f"  DSR={dsr['dsr']:.3f}  PBO={pbo.pbo:.3f}  "
          f"(PBO here is benign-by-construction: all 5 cells are the same signal with "
          f"different denominators, so any pick is fine — the family effect is the claim)")

    print("\n  regime split (a-priori estimator vs base):")
    ra, rb = regime_split_sharpe(d_ap, XAU_REGIMES), regime_split_sharpe(base, XAU_REGIMES)
    for k in ra:
        print(f"    {k:24s} {A_PRIORI} {ra[k]:+.2f}   base {rb.get(k, float('nan')):+.2f}")

    # ---------------- A5: does it generalise to OTHER instruments? (external validation)
    print("\n=== A5: external validation — same swap elsewhere, GROSS (cost-free) ===")
    print("  Gross (zero spread AND zero slippage), and FX is EXCLUDED. Both are units")
    print("  decisions, not conveniences: this harness's dollar constants are XAU-specific")
    print("  (spread*0.1; SLIP_USD=$0.10). On EURUSD at 1.1 a $0.10 slip is ~9% PER TRADE,")
    print("  which produced SR -7.7 on a first pass — an artefact, not a finding. BTC/NDX/SPX")
    print("  trade in the hundreds-to-thousands, so those constants are harmless there.")
    print(f"{'instrument':12} {'bars':>7} {'base SR':>9} {'range-vol SR':>13} "
          f"{'t@matched':>10} {'years':>7}")
    others = [("BTC", "data/BTC_D1_long.csv", 252),
              ("NDX", "data/NDX_D1_long.csv", 252),
              ("SPX", "data/SPX_D1_long.csv", 252)]
    for name, path, ann in others:
        fp = ROOT / path
        if not fp.exists():
            continue
        o = pd.read_csv(fp, parse_dates=["time"], index_col="time").sort_index()
        o = o[~o.index.duplicated(keep="last")]
        if "spread" not in o:
            continue
        o["spread_px"] = 0.0        # gross — see the note printed above
        lifts.SLIP_USD = 0.0        # ditto: XAU-tuned dollar constant, zeroed for A5
        of = champion_signal(o["close"])
        # reuse the same estimator maths, annualised for this instrument's bar size
        oc, oh, ol, oo = o["close"], o["high"], o["low"], o["open"]
        rs_var = (np.log(oh / oc) * np.log(oh / oo) + np.log(ol / oc) * np.log(ol / oo))
        on_var = (np.log(oo / oc.shift(1)) ** 2).ewm(halflife=VOL_HL, min_periods=20).mean()
        oc_var = (np.log(oc / oo) ** 2).ewm(halflife=VOL_HL, min_periods=20).mean()
        k = 0.34 / (1.34 + (VOL_HL + 1) / (VOL_HL - 1))
        yz = on_var + k * oc_var + (1 - k) * rs_var.clip(lower=0).ewm(
            halflife=VOL_HL, min_periods=20).mean()
        v_range = np.sqrt(yz.clip(lower=0)) * np.sqrt(ann)
        v_base = oc.pct_change().ewm(halflife=VOL_HL, min_periods=20).std() * np.sqrt(ann)
        bo = engine_v(o, of, vol=v_base)
        ao = engine_v(o, of, vol=v_range)
        if bo.empty or ao.empty:
            continue
        t, yrs = t_matched(ao, bo)
        print(f"{name:12} {len(o):>7} {sharpe(bo):>+9.3f} {sharpe(ao):>+13.3f} "
              f"{t:>+10.2f} {yrs:>7}")

    lifts.SLIP_USD = 0.10   # restore before PART B
    # ---------------- PART B: the actionable test — does it beat ATR in the LIVE engine?
    print(f"\n{'='*78}\nPART B: does an efficient estimator beat WILDER ATR in the DISCRETE\n"
          f"(deployed) engine? Level-calibrated, so only shape/timing is tested.\n{'='*78}")
    BASE_PARAMS = dict(sl_atr=3.0, trail_atr=3.0,
                       conf_risk_scale={"low": 0.5, "med": 1.0, "high": 1.5})
    xt.xau_signal = champion_signal
    atr_real = xt.wilder_atr(df, 14)

    def discrete_daily(vol_price: pd.Series | None) -> tuple[pd.Series, pd.DataFrame]:
        """Run the deployed engine, optionally replacing its ATR with a
        level-calibrated price-unit vol series."""
        if vol_price is None:
            xt.wilder_atr = _orig_atr
        else:
            cal = vol_price * (atr_real.mean() / vol_price.mean())   # same average stop
            xt.wilder_atr = lambda d, period, _c=cal: _c
        r = xt.run_trades(df, equity0=100_000.0, exit_mode="trail",
                          flip_mode="confidence", params=BASE_PARAMS)
        e = r["equity"].dropna().loc[EVAL_START:]
        dd = e.resample("D").last().dropna().pct_change().dropna()
        return dd, r["trades"]

    _orig_atr = xt.wilder_atr
    d_atr, tr_atr = discrete_daily(None)
    print(f"{'stop/size vol':22} {'SR':>8} {'DD':>8} {'trades':>7} {'t@matched':>10} {'years':>7}")
    print(f"{'wilder_atr (DEPLOYED)':22} {sharpe(d_atr):>+8.3f} {dd_of(d_atr):>+8.1f} "
          f"{len(tr_atr):>7} {'—':>10} {'—':>7}")
    for vname in ("parkinson", "garman_klass", "rogers_satchell", "yang_zhang"):
        # annualised fraction -> per-bar price units, then level-calibrated inside
        vp = vols[vname] / np.sqrt(ANN_H4) * close
        d_v, tr_v = discrete_daily(vp)
        t, yrs = t_matched(d_v, d_atr)
        print(f"{vname:22} {sharpe(d_v):>+8.3f} {dd_of(d_v):>+8.1f} {len(tr_v):>7} "
              f"{t:>+10.2f} {yrs:>7}")
    xt.wilder_atr = _orig_atr
    print("\n  (t@matched here is vs the DEPLOYED ATR version, not vs the continuous base)")


if __name__ == "__main__":
    main()
