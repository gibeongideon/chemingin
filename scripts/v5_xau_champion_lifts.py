"""FIVE UNTRIED WAYS TO LIFT THE LONG-ONLY CHAMPION'S SHARPE. XAUUSD H4.

User ask (2026-09-03): improve the long-only champion's Sharpe; list five ways NOT based
on anything already tried; research independently.

Checked against every closed line first (V5_FINDINGS §3a-§3aa + memory): none of the five
mechanisms below appears anywhere. Specifically these are NOT: meta-labeling/per-trade
probability sizing (§4, disproven 2x), an ADX/Hurst/Choppiness regime GATE (§3u,
paired-t -2.88), regime features fed to a direction classifier (§3r), turning-point
overlays (§3p), speed/timeframe sweeps (§3f), profit-take or dip re-entry (§3e), scale-in
(§3y), risk_frac (§3z), breadth/basket dilution (§3-breadth), SuperTrend (§3s), ICT/SMC
(§3v-w), or anything long/short (§3aa).

  IDEA 1  RISK-NORMALISATION QUALITY. The engine divides by close-to-close EWMA vol.
          Range-based estimators (Parkinson / Garman-Klass / Rogers-Satchell /
          Yang-Zhang) are 5-8x more statistically efficient for the SAME data because
          they use the whole bar, not just its endpoints. A less noisy denominator means
          exposure that tracks true risk more closely. Touches only the risk normaliser —
          the directional signal is untouched, so it cannot be a disguised new bet.
  IDEA 2  ENDOGENOUS CONVICTION. The champion is a blend of 6 constituent forecasts
          (3 EWMAC speed pairs + 3 breakout windows). Their AGREEMENT is free information
          the blend throws away: 6/6 pointing up is a different state from 3/6, at the
          same blended level. Scale exposure by agreement. Distinct from §3u's regime gate
          because the conditioning variable is the signal's own internal structure, not a
          separate lagging indicator, and nothing is fitted.
  IDEA 3  MOMENTUM LIFE-CYCLE. Trends age. Forward returns conditional on how long a
          trend has already run are documented to decay (late-stage momentum is weaker
          and crash-prone). The champion is age-blind. Measure the age curve, then taper.
  IDEA 4  PARAMETER-CLOUD AVERAGING. The champion is a POINT estimate chosen from a
          sweep. Averaging its forecast over a dense neighbourhood of nearby
          parameterisations (lookback scalings x concentration exponents) reduces
          parameter-estimation error without changing the family. Standard CTA robustness
          practice; never done here.
  IDEA 5  THIRD-MOMENT CONDITIONING. Every prior regime study used first/second moment
          (trend strength, vol, Hurst, choppiness). Trailing SKEW / downside-vs-upside
          semi-vol is a different statistic, and gold's crisis asymmetry gives it a real
          mechanism.

THE GATE. §3y's lesson applies with full force: for an OVERLAY on an already-good base,
DSR against the overlay grid is near-vacuous (all cells inherit the base's real edge).
The gate here is the PAIRED daily-return t-stat vs the champion, plus per-year win count,
plus walk-forward selection WITHIN each idea's own grid. DSR/PBO are reported across all
cells for completeness, not used as the pass criterion.

    python scripts/v5_xau_champion_lifts.py
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
from src.v5.xau_dual_signals import (  # noqa: E402
    champion_signal, ewmac_fc, breakout_fc, _conc, EWMAC_MID, BKO_FAST, D,
)
from src.evaluation.dsr_pbo import deflated_sharpe_ratio, pbo_cscv  # noqa: E402
from src.evaluation.walk_forward_grid import (  # noqa: E402
    walk_forward_select, regime_split_sharpe, print_selections,
)

EVAL_START = "2018-01-01"
YEARS = list(range(2018, 2027))
SELECT_WINDOW_BARS = 3 * 252         # trailing 3y of DAILY rows
ANN_H4 = 252 * 6
VOL_HL = 42
TARGET_VOL = 0.10
BUFFER = 0.1
MAX_LEV = 8.0
SLIP_USD = 0.10
COST_USD = 0.448                     # live FTMO/Maven-class floor (MANDATORY CONTROL #1)

XAU_REGIMES = [
    ("2015-01-01", "2018-12-31", "chop"),
    ("2019-01-01", "2020-12-31", "bull"),
    ("2021-01-01", "2022-12-31", "flat"),
    ("2023-01-01", "2026-12-31", "bull"),
]


# ============================================================ engine (vol-pluggable)
def engine_v(df: pd.DataFrame, fc: pd.Series, *, vol: pd.Series | None = None,
             buffer_frac: float = BUFFER, delay: int = 1) -> pd.Series:
    """`v5_xau_turn_prob.engine`, verbatim formula, with the vol denominator
    exposed so IDEA 1 can swap the estimator. Returns DAILY net returns."""
    close = df["close"]
    ret = close.pct_change()
    if vol is None:
        vol = ret.ewm(halflife=VOL_HL, min_periods=20).std() * np.sqrt(ANN_H4)
    pos = (fc.clip(-2.0, 2.0) * (TARGET_VOL / vol)).clip(-MAX_LEV, MAX_LEV)
    if buffer_frac > 0:
        band = (buffer_frac * (TARGET_VOL / vol).clip(0, MAX_LEV)).values
        p, out, held = pos.values.copy(), np.zeros(len(pos)), 0.0
        for i in range(len(p)):
            if np.isfinite(p[i]):
                b = band[i] if np.isfinite(band[i]) else 0.0
                if abs(p[i] - held) > b:
                    held = p[i] - np.sign(p[i] - held) * b
            out[i] = held
        pos = pd.Series(out, index=pos.index)
    pos = pos.shift(delay).fillna(0.0)
    cost_frac = (df["spread_px"] / 2.0 + SLIP_USD) / close
    net = (pos * ret - pos.diff().abs().fillna(0.0) * cost_frac).fillna(0.0)
    eq = (1.0 + net).cumprod().loc[EVAL_START:]
    if eq.empty:
        return pd.Series(dtype=float)
    eq = eq / eq.iloc[0]
    return eq.resample("D").last().pct_change(fill_method=None).dropna()


def sharpe(d: pd.Series) -> float:
    return float(d.mean() / d.std() * np.sqrt(252)) if len(d) > 20 and d.std() > 0 else np.nan


def dd_of(d: pd.Series) -> float:
    e = (1 + d).cumprod()
    return float((e / e.cummax() - 1).min() * 100)


# ============================================================ IDEA 1: vol estimators
def vol_estimators(df: pd.DataFrame, hl: int = VOL_HL) -> dict[str, pd.Series]:
    """All causal: each uses only the current bar's OHLC, then an EWMA over the
    past. Annualised to match the engine's convention."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    ann = np.sqrt(ANN_H4)
    out = {}
    out["close2close(base)"] = c.pct_change().ewm(halflife=hl, min_periods=20).std() * ann

    hl_log = np.log(h / l)
    park_var = (hl_log ** 2) / (4.0 * np.log(2.0))
    out["parkinson"] = np.sqrt(park_var.ewm(halflife=hl, min_periods=20).mean()) * ann

    gk_var = 0.5 * hl_log ** 2 - (2.0 * np.log(2.0) - 1.0) * (np.log(c / o) ** 2)
    out["garman_klass"] = np.sqrt(gk_var.clip(lower=0).ewm(halflife=hl, min_periods=20).mean()) * ann

    rs_var = (np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o))
    out["rogers_satchell"] = np.sqrt(rs_var.clip(lower=0).ewm(halflife=hl, min_periods=20).mean()) * ann

    # Yang-Zhang: overnight + open-close + Rogers-Satchell, k per the paper's rule
    on_var = (np.log(o / prev_c) ** 2).ewm(halflife=hl, min_periods=20).mean()
    oc_var = (np.log(c / o) ** 2).ewm(halflife=hl, min_periods=20).mean()
    k = 0.34 / (1.34 + (hl + 1) / (hl - 1))
    yz = on_var + k * oc_var + (1 - k) * rs_var.clip(lower=0).ewm(halflife=hl, min_periods=20).mean()
    out["yang_zhang"] = np.sqrt(yz.clip(lower=0)) * ann
    return out


# ============================================================ IDEA 2: ensemble agreement
def constituent_forecasts(close: pd.Series) -> pd.DataFrame:
    """The champion's 6 building blocks, each on its own."""
    cols = {}
    for i, pair in enumerate(EWMAC_MID):
        cols[f"ewmac{i}"] = ewmac_fc(close, (pair,))
    for i, w in enumerate(BKO_FAST):
        cols[f"bko{i}"] = breakout_fc(close, (w,))
    return pd.DataFrame(cols)


def agreement_multiplier(close: pd.Series, lo: float) -> pd.Series:
    """Fraction of the 6 constituents pointing UP (the champion is long-only, so
    'up' is the only direction that matters), mapped linearly onto [lo, 1.0].
    agreement 1.0 -> full size; agreement 0 -> `lo` x size."""
    C = constituent_forecasts(close)
    frac_up = (C > 0).sum(axis=1) / C.shape[1]
    return (lo + (1.0 - lo) * frac_up).clip(lo, 1.0)


# ============================================================ IDEA 3: trend age
def trend_age(fc: pd.Series, thresh: float = 0.5) -> pd.Series:
    """Bars since the forecast last crossed UP through `thresh` from below.
    NaN/0 while below the threshold. Causal by construction."""
    above = (fc >= thresh).values
    age = np.zeros(len(fc))
    run = 0
    for i in range(len(fc)):
        run = run + 1 if above[i] else 0
        age[i] = run
    return pd.Series(age, index=fc.index)


def age_multiplier(fc: pd.Series, taper_start: int, taper_floor: float,
                   taper_len: int = 60) -> pd.Series:
    """Full size until `taper_start` bars of trend age, then decay linearly to
    `taper_floor` over `taper_len` bars."""
    age = trend_age(fc).values
    m = np.ones(len(fc))
    over = np.clip((age - taper_start) / max(taper_len, 1), 0.0, 1.0)
    m = 1.0 - over * (1.0 - taper_floor)
    return pd.Series(m, index=fc.index)


# ============================================================ IDEA 4: parameter cloud
def champion_recipe(close: pd.Series, scale: float, conc_p: float) -> pd.Series:
    """The champion's exact recipe with all lookbacks scaled by `scale` and the
    concentration exponent set to `conc_p`. scale=1.0, conc_p=1.5 reproduces
    `champion_signal` (asserted in main())."""
    ewm = tuple((max(2, int(round(f * scale))), max(3, int(round(s * scale))))
                for f, s in EWMAC_MID)
    bko = tuple(max(3, int(round(w * scale))) for w in BKO_FAST)
    base = ewmac_fc(close, ewm)
    b = breakout_fc(close, bko)
    maxewbko = np.maximum(base.clip(lower=0.0), b.clip(lower=0.0))
    return (0.5 * (_conc(maxewbko, conc_p) * 0.8 + 0.15)
            + 0.5 * (_conc(b, conc_p) * 0.8 + 0.15)).clip(0, 2)


def parameter_cloud(close: pd.Series, scales: tuple, concs: tuple) -> pd.Series:
    fcs = [champion_recipe(close, s, p) for s in scales for p in concs]
    return pd.concat(fcs, axis=1).mean(axis=1).clip(0, 2)


# ============================================================ IDEA 5: skew / semi-vol
def skew_multiplier(close: pd.Series, window: int, lo: float, sign: int) -> pd.Series:
    """Trailing realised skew mapped to a size multiplier. `sign=+1` sizes UP on
    positive skew, -1 sizes up on negative skew. Both directions are tested
    because presuming the sign would be a free parameter — and both are counted
    in the trial set."""
    r = close.pct_change()
    sk = r.rolling(window, min_periods=window // 2).skew()
    z = (sk / sk.rolling(500, min_periods=100).std()).clip(-2, 2).fillna(0.0)
    raw = 0.5 * (1.0 + sign * z / 2.0)          # -> roughly [0,1]
    return (lo + (1.0 - lo) * raw.clip(0, 1)).clip(lo, 1.0)


def semivol_multiplier(close: pd.Series, window: int, lo: float) -> pd.Series:
    """Downside/upside semi-vol ratio: size DOWN when downside vol dominates."""
    r = close.pct_change()
    dn = r.clip(upper=0).rolling(window, min_periods=window // 2).std()
    up = r.clip(lower=0).rolling(window, min_periods=window // 2).std()
    ratio = (up / (dn + 1e-12)).clip(0, 3).fillna(1.0)
    raw = (ratio / 2.0).clip(0, 1)
    return (lo + (1.0 - lo) * raw).clip(lo, 1.0)


# ============================================================ harness
def vol_match(d: pd.Series, base: pd.Series) -> pd.Series:
    """Lever `d` to the base's realised vol. Sharpe is scale-invariant, so this
    changes nothing about the ranking — but it makes the RETURN comparison
    apples-to-apples. Every overlay here multiplies the forecast by <=1, i.e.
    de-risks; a raw paired-t on returns would therefore mark a genuine
    risk-adjusted improvement as a loss. Same correction as the leverage-matched
    control that settled the scale-in question (§3y)."""
    if d.std() == 0 or not np.isfinite(d.std()):
        return d
    return d * (base.std() / d.std())


def report(name: str, d: pd.Series, base: pd.Series) -> dict:
    dm = vol_match(d, base)
    _, t_raw, _ = paired(d, base)
    _, t_m, _ = paired(dm, base)          # THE GATE: more return at equal risk?
    yp, yn = per_year(dm, base)
    sr = sharpe(d)
    return dict(name=name, sharpe=sr, dd=dd_of(d), dd_matched=dd_of(dm),
                t_raw=t_raw, t_vs_base=t_m, years_better=f"{yp}/{yn}", yp=yp, yn=yn,
                calmar=(sr and dd_of(d) and abs(sr / (dd_of(d) / 100.0))) or np.nan)


def main() -> None:
    df = load_h4(COST_USD)
    close = df["close"]
    base_fc = champion_signal(close)
    base = engine_v(df, base_fc)
    print(f"XAUUSD H4 {df.index[0].date()}..{df.index[-1].date()}  eval {EVAL_START}+  "
          f"cost floored ${COST_USD}")
    print(f"BASE champion: SR={sharpe(base):+.3f}  DD={dd_of(base):+.1f}%\n")

    # sanity: the cloud's centre must reproduce the champion exactly
    centre = champion_recipe(close, 1.0, 1.5)
    assert float((centre - base_fc).abs().max()) < 1e-9, "champion_recipe != champion_signal"
    print("regression check: champion_recipe(scale=1, conc=1.5) == champion_signal  PASS\n")

    all_cells: dict[str, pd.Series] = {}
    idea_rows: list[dict] = []

    # ---------------- IDEA 1
    print("=== IDEA 1: range-based vol estimator for the risk denominator ===")
    vols = vol_estimators(df)
    for vname, v in vols.items():
        d = engine_v(df, base_fc, vol=v)
        r = report(f"vol:{vname}", d, base)
        idea_rows.append(dict(idea="1_vol", **r))
        all_cells[r["name"]] = d
        print(f"  {vname:20} SR={r['sharpe']:+.3f}  DD={r['dd']:+.1f}%  "
              f"DD@matched={r['dd_matched']:+.1f}%  t@matched={r['t_vs_base']:+.2f}  "
              f"years {r['years_better']}")

    # ---------------- IDEA 2
    print("\n=== IDEA 2: endogenous ensemble-agreement conviction ===")
    for lo in (0.3, 0.5, 0.7):
        m = agreement_multiplier(close, lo)
        d = engine_v(df, (base_fc * m).clip(0, 2))
        r = report(f"agree:lo={lo}", d, base)
        idea_rows.append(dict(idea="2_agreement", **r))
        all_cells[r["name"]] = d
        print(f"  lo={lo:<4} SR={r['sharpe']:+.3f}  DD={r['dd']:+.1f}%  "
              f"DD@matched={r['dd_matched']:+.1f}%  t@matched={r['t_vs_base']:+.2f}  "
              f"years {r['years_better']}")

    # ---------------- IDEA 3 (research first: the age curve, then the taper)
    print("\n=== IDEA 3: momentum life-cycle (trend age) ===")
    age = trend_age(base_fc)
    fwd = close.pct_change(6).shift(-6)          # forward ~1 day, for the CURVE ONLY
    buckets = [(1, 12), (13, 30), (31, 60), (61, 120), (121, 10_000)]
    print("  age curve (forward 6-bar return by trend age; descriptive, uses future "
          "data BY DESIGN — not tradeable, only to choose the taper):")
    for a0, a1 in buckets:
        m = (age >= a0) & (age <= a1)
        seg = fwd[m].dropna()
        if len(seg) > 50:
            print(f"    age {a0:>4}-{a1:<6} n={len(seg):>5}  mean fwd6 "
                  f"{seg.mean()*1e4:+7.1f}bp  hit {float((seg>0).mean()*100):.1f}%")
    # the curve says the WEAK phase is the FIRST ~12 bars (whipsaw right after the
    # cross), not only the late phase — so test an ignition delay too. NOTE: these
    # buckets were read off a FULL-SAMPLE forward-return curve, so the choice is
    # in-sample-informed; it only counts if it survives the walk-forward below.
    for ign in (6, 12, 24):
        for ign_lo in (0.3, 0.6):
            age_v = trend_age(base_fc).values
            m = np.where(age_v <= ign, ign_lo, 1.0)
            m = pd.Series(m, index=base_fc.index)
            d = engine_v(df, (base_fc * m).clip(0, 2))
            r = report(f"ignition:bars<={ign},lo={ign_lo}", d, base)
            idea_rows.append(dict(idea="3_age", **r))
            all_cells[r["name"]] = d
            print(f"  ignition bars<={ign:<3} lo={ign_lo:<4} SR={r['sharpe']:+.3f}  "
                  f"DD={r['dd']:+.1f}%  DD@m={r['dd_matched']:+.1f}%  "
                  f"t@matched={r['t_vs_base']:+.2f}  years {r['years_better']}")
    for start in (30, 60, 120):
        for floor in (0.4, 0.7):
            m = age_multiplier(base_fc, start, floor)
            d = engine_v(df, (base_fc * m).clip(0, 2))
            r = report(f"age:start={start},floor={floor}", d, base)
            idea_rows.append(dict(idea="3_age", **r))
            all_cells[r["name"]] = d
            print(f"  taper start={start:<4} floor={floor:<4} SR={r['sharpe']:+.3f}  "
                  f"DD={r['dd']:+.1f}%  DD@m={r['dd_matched']:+.1f}%  "
                  f"t@matched={r['t_vs_base']:+.2f}  years {r['years_better']}")

    # ---------------- IDEA 4
    print("\n=== IDEA 4: parameter-cloud forecast averaging ===")
    clouds = {
        "tight(0.85-1.15,conc1.25-1.75)": ((0.85, 1.0, 1.15), (1.25, 1.5, 1.75)),
        "wide(0.7-1.3,conc1.25-1.75)":    ((0.7, 0.85, 1.0, 1.15, 1.3), (1.25, 1.5, 1.75)),
        "speed-only(0.7-1.3,conc1.5)":    ((0.7, 0.85, 1.0, 1.15, 1.3), (1.5,)),
    }
    for cname, (scales, concs) in clouds.items():
        fc = parameter_cloud(close, scales, concs)
        d = engine_v(df, fc)
        r = report(f"cloud:{cname}", d, base)
        idea_rows.append(dict(idea="4_cloud", **r))
        all_cells[r["name"]] = d
        print(f"  {cname:32} SR={r['sharpe']:+.3f}  DD={r['dd']:+.1f}%  "
              f"DD@m={r['dd_matched']:+.1f}%  t@matched={r['t_vs_base']:+.2f}  "
              f"years {r['years_better']}")

    # ---------------- IDEA 5
    print("\n=== IDEA 5: third-moment (skew / semi-vol) conditioning ===")
    for w in (60, 120):
        for sgn in (+1, -1):
            m = skew_multiplier(close, w, 0.5, sgn)
            d = engine_v(df, (base_fc * m).clip(0, 2))
            r = report(f"skew:w={w},sign={sgn:+d}", d, base)
            idea_rows.append(dict(idea="5_skew", **r))
            all_cells[r["name"]] = d
            print(f"  skew w={w:<4} sign={sgn:+d}  SR={r['sharpe']:+.3f}  "
                  f"DD={r['dd']:+.1f}%  DD@m={r['dd_matched']:+.1f}%  "
                  f"t@matched={r['t_vs_base']:+.2f}  years {r['years_better']}")
    for w in (60, 120):
        m = semivol_multiplier(close, w, 0.5)
        d = engine_v(df, (base_fc * m).clip(0, 2))
        r = report(f"semivol:w={w}", d, base)
        idea_rows.append(dict(idea="5_skew", **r))
        all_cells[r["name"]] = d
        print(f"  semivol w={w:<4}      SR={r['sharpe']:+.3f}  DD={r['dd']:+.1f}%  "
              f"DD@m={r['dd_matched']:+.1f}%  t@matched={r['t_vs_base']:+.2f}  "
              f"years {r['years_better']}")

    R = pd.DataFrame(idea_rows)

    # ---------------- verdict per idea
    print(f"\n{'='*78}\nPER-IDEA VERDICT (gate = paired-t vs champion, NOT DSR — see §3y)\n{'='*78}")
    base_sr, base_dd = sharpe(base), dd_of(base)
    print(f"  (base: SR {base_sr:+.3f}, DD {base_dd:+.1f}%)   gate = t@matched-vol > +1.64")
    print(f"{'idea':14} {'cells':>5} {'best SR':>8} {'best DD@m':>10} "
          f"{'best t@m':>9} {'cells t@m>1.64':>15}")
    for idea, sub in R.groupby("idea"):
        print(f"{idea:14} {len(sub):>5} {sub.sharpe.max():>+8.3f} "
              f"{sub.dd_matched.max():>+10.1f} {sub.t_vs_base.max():>+9.2f} "
              f"{int((sub.t_vs_base > 1.64).sum()):>15}")

    winners = R[(R.t_vs_base > 1.64) & (R.sharpe > base_sr)]
    print(f"\ncells beating base with paired-t > +1.64: {len(winners)}/{len(R)}")
    if len(winners):
        for _, w in winners.sort_values("t_vs_base", ascending=False).iterrows():
            print(f"  {w['name']:34} SR={w.sharpe:+.3f} t={w.t_vs_base:+.2f} "
                  f"years {w.years_better}")

    # ---------------- COMBINATION of the best cell per idea (if any survived)
    print(f"\n{'='*78}\nCOMBINATION of each idea's best cell\n{'='*78}")
    best_per_idea = R.loc[R.groupby("idea")["t_vs_base"].idxmax()]
    print("  stacking:", ", ".join(best_per_idea["name"].tolist()))
    fc_comb = base_fc.copy()
    vol_comb = None
    for _, row in best_per_idea.iterrows():
        nm = row["name"]
        if nm.startswith("vol:"):
            vol_comb = vols[nm.split("vol:")[1]]
        elif nm.startswith("agree:"):
            fc_comb = fc_comb * agreement_multiplier(close, float(nm.split("lo=")[1]))
        elif nm.startswith("age:"):
            st = int(nm.split("start=")[1].split(",")[0]); fl = float(nm.split("floor=")[1])
            fc_comb = fc_comb * age_multiplier(base_fc, st, fl)
        elif nm.startswith("cloud:"):
            key = nm.split("cloud:")[1]
            fc_comb = fc_comb / base_fc.replace(0, np.nan) * parameter_cloud(close, *clouds[key])
            fc_comb = fc_comb.fillna(0.0)
        elif nm.startswith("skew:"):
            w = int(nm.split("w=")[1].split(",")[0]); sg = int(nm.split("sign=")[1])
            fc_comb = fc_comb * skew_multiplier(close, w, 0.5, sg)
        elif nm.startswith("semivol:"):
            fc_comb = fc_comb * semivol_multiplier(close, int(nm.split("w=")[1]), 0.5)
    d_comb = engine_v(df, fc_comb.clip(0, 2), vol=vol_comb)
    rc = report("COMBINED", d_comb, base)
    all_cells["COMBINED"] = d_comb
    print(f"  COMBINED SR={rc['sharpe']:+.3f}  DD={rc['dd']:+.1f}%  "
          f"DD@matched={rc['dd_matched']:+.1f}%  t@matched={rc['t_vs_base']:+.2f}  "
          f"t@raw={rc['t_raw']:+.2f}  years {rc['years_better']}")

    # ---------------- walk-forward over EVERYTHING + honest deflation
    print(f"\n{'='*78}\nWALK-FORWARD over all {len(all_cells)} cells (trailing 3y) + deflation\n{'='*78}")
    oos, sel = walk_forward_select(all_cells, YEARS, SELECT_WINDOW_BARS, min_train_bars=100)
    print_selections(sel)
    if not oos.empty and oos.std() > 0:
        oos_m = vol_match(oos, base)
        _, t_raw_wf, _ = paired(oos, base)
        _, t_wf, _ = paired(oos_m, base)
        yp, yn = per_year(oos_m, base)
        print(f"\nWALK-FORWARD selected: SR={sharpe(oos):+.3f}  DD={dd_of(oos):+.1f}%  "
              f"DD@matched={dd_of(oos_m):+.1f}%")
        print(f"BASE champion        : SR={base_sr:+.3f}  DD={base_dd:+.1f}%")
        print(f"paired t @matched vol = {t_wf:+.2f} (raw {t_raw_wf:+.2f})   "
              f"years better {yp}/{yn}")
        trials = np.array([s / np.sqrt(252) for s in R.sharpe if np.isfinite(s)])
        dsr = deflated_sharpe_ratio(oos.values, trials)
        common = None
        for v in all_cells.values():
            common = v.index if common is None else common.intersection(v.index)
        pbo = pbo_cscv(np.column_stack([v.reindex(common).fillna(0.0).values
                                        for v in all_cells.values()]), n_partitions=10)
        print(f"DSR={dsr['dsr']:.3f} (n_trials={dsr['n_trials']})  PBO={pbo.pbo:.3f}   "
              f"[DSR is near-vacuous for overlays — see §3y]")
        print("\nregime split (selected vs base):")
        rv, rb = regime_split_sharpe(oos, XAU_REGIMES), regime_split_sharpe(base, XAU_REGIMES)
        for k in rv:
            print(f"  {k:24s} lift {rv[k]:+.2f}   base {rb.get(k, float('nan')):+.2f}")

    out = ROOT / "data" / "v5_runs" / "xau_champion_lifts.csv"
    R.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
