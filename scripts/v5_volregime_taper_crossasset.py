"""PRE-REGISTERED TEST: does the high-vol taper replicate on BTC / NDX / BRENT + portfolio?

§3ac found the volatility-regime taper is the only one of six proposed components with a
real signal on XAU: SR +1.084 -> +1.287, DD -19.8% -> -12.7%, consistent in BOTH halves,
but t@matched only +1.23 (5/9 years) = REAL BUT UNPROVEN. The recorded next step was
explicit: do NOT add more XAU schedule variants (multiple-testing trap on one asset) —
test the SAME taper on the other book sleeves. A general trend-following property should
replicate; a gold-specific fluke will not. This is that test.

PRE-REGISTRATION (fixed before looking at any result):
  * PRIMARY schedule = the monotone taper 1.0/0.8/0.6/0.4/0.2 over trailing vol quintiles,
    the winner on XAU. The other two XAU schedules are reported for completeness but the
    verdict rests on the primary — no per-instrument re-optimisation.
  * PRIMARY claim = the PORTFOLIO (the book is what actually trades), gated at matched vol
    with a split-sample.
  * Replication counts if the primary schedule lifts the majority of sleeves AND the
    portfolio, in BOTH halves.

TWO UNITS TRAPS HANDLED (both bit earlier work — see §3ab):
  1. COST IN BASIS POINTS, not dollars. `SLIP_USD=$0.10` is meaningless across
     instruments ($0.10 on BRENT at ~$80 is 12bp; on BTC at ~$60k it is 0.002bp). Costs
     below are one-way bp built from LIVE-VERIFIED spreads in configs/v5_ftmo_challenge.json
     (BTC ~1.5bp, BRENT 7.58bp verified 2026-07-24, NDX ~0.5bp) plus slippage at 50% of
     the half-spread. Sanity anchor: this reproduces 0.75bp one-way for XAU, matching what
     the dollar-based engine charges at $0.448 + $0.10 on ~$4400.
  2. CHAMPION SPEEDS RESCALED FOR D1. `champion_signal` hardcodes D=6 (H4 bars/day), so
     applied to DAILY bars it silently becomes a 96-1536 DAY trend follower instead of
     16-256 days. `champion_recipe(close, scale=1/6)` restores the intended economic
     horizon. XAU stays on H4 as deployed; BTC/NDX/BRENT are D1 as deployed.

    python scripts/v5_volregime_taper_crossasset.py
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
from scripts.v5_xau_champion_lifts import champion_recipe, vol_match, sharpe, dd_of  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402

EVAL_START = "2018-01-01"
TARGET_VOL, MAX_LEV, BUFFER = 0.10, 8.0, 0.1
VOL_HL_DAYS = 7           # ~7 trading days, the SAME economic halflife as XAU's H4 hl=42
PCT_WIN_DAYS = 504        # trailing 2y, matching the XAU test's 2y percentile window
SPLITS = [("2018-01-01", "2021-12-31", "2018-21"), ("2022-01-01", "2026-12-31", "2022-26")]

# one-way cost in bp = half-spread + slippage(50% of half-spread). Spreads live-verified.
SLEEVES = {
    "XAU (H4, deployed)": dict(path="data/XAUUSD_H4_long.csv", ann=252 * 6,
                               cost_bp=0.75, scale=1.0, hl=42, pct_win=252 * 6 * 2),
    "BTC (D1)":           dict(path="data/BTC_D1_long.csv", ann=252,
                               cost_bp=1.13, scale=1 / 6, hl=VOL_HL_DAYS, pct_win=PCT_WIN_DAYS),
    "NDX (D1)":           dict(path="data/NDX_D1_long.csv", ann=252,
                               cost_bp=0.38, scale=1 / 6, hl=VOL_HL_DAYS, pct_win=PCT_WIN_DAYS),
    "BRENT (D1)":         dict(path="data/BRENT_D1_long.csv", ann=252,
                               cost_bp=5.69, scale=1 / 6, hl=VOL_HL_DAYS, pct_win=PCT_WIN_DAYS),
}

SCHEDULES = {
    "monotone taper (PRIMARY)":    {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2},
    "as proposed (lo.5/hi.5/ext0)": {0: 0.5, 1: 1.0, 2: 1.0, 3: 0.5, 4: 0.0},
    "cut extreme only":            {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.0},
}
PRIMARY = "monotone taper (PRIMARY)"


def engine_bp(df: pd.DataFrame, fc: pd.Series, cost_bp: float, ann: int,
              hl: int, vol_override: pd.Series | None = None) -> pd.Series:
    """Same vol-target engine as everywhere else, but cost is a one-way FRACTION
    OF PRICE (bp) so it is dimensionally valid on any instrument."""
    close = df["close"]
    ret = close.pct_change()
    vol = (vol_override if vol_override is not None
           else ret.ewm(halflife=hl, min_periods=20).std() * np.sqrt(ann))
    pos = (fc.clip(-2.0, 2.0) * (TARGET_VOL / vol)).clip(-MAX_LEV, MAX_LEV)
    band = (BUFFER * (TARGET_VOL / vol).clip(0, MAX_LEV)).values
    p, out, held = pos.values.copy(), np.zeros(len(pos)), 0.0
    for i in range(len(p)):
        if np.isfinite(p[i]):
            b = band[i] if np.isfinite(band[i]) else 0.0
            if abs(p[i] - held) > b:
                held = p[i] - np.sign(p[i] - held) * b
        out[i] = held
    pos = pd.Series(out, index=pos.index).shift(1).fillna(0.0)
    net = (pos * ret - pos.diff().abs().fillna(0.0) * (cost_bp * 1e-4)).fillna(0.0)
    eq = (1.0 + net).cumprod().loc[EVAL_START:]
    if eq.empty:
        return pd.Series(dtype=float)
    eq = eq / eq.iloc[0]
    return eq.resample("D").last().pct_change(fill_method=None).dropna()


def vol_quintile(df: pd.DataFrame, ann: int, hl: int, pct_win: int) -> pd.Series:
    """The proposal's own EWMA-vol formula, ranked causally against its trailing
    2y history, bucketed into quintiles."""
    ret = df["close"].pct_change()
    vol = np.sqrt(ret.pow(2).ewm(halflife=hl, min_periods=20).mean()) * np.sqrt(ann)
    vr = vol.rolling(pct_win, min_periods=252).apply(lambda x: (x[-1] > x).mean(), raw=True)
    return np.clip((vr * 5).fillna(2).astype(int), 0, 4)


def gate(d: pd.Series, base: pd.Series) -> tuple[float, str]:
    dm = vol_match(d, base)
    _, t, _ = paired(dm, base)
    yp, yn = per_year(dm, base)
    return t, f"{yp}/{yn}"


def load_sleeve(cfg: dict) -> pd.DataFrame | None:
    fp = ROOT / cfg["path"]
    if not fp.exists():
        return None
    d = pd.read_csv(fp, parse_dates=["time"], index_col="time").sort_index()
    return d[~d.index.duplicated(keep="last")]


def main() -> None:
    print("PRE-REGISTERED: same high-vol taper, other sleeves + portfolio. "
          f"eval {EVAL_START}+\n")
    print(f"{'sleeve':22} {'one-way cost':>13} {'champion speeds':>18}")
    for nm, c in SLEEVES.items():
        sp = "16-256d (H4 native)" if c["scale"] == 1.0 else "16-256d (rescaled)"
        print(f"{nm:22} {c['cost_bp']:>11.2f}bp {sp:>18}")

    base_by, taper_by, split_rows = {}, {}, []
    print(f"\n{'='*94}\nPER-SLEEVE\n{'='*94}")
    for nm, cfg in SLEEVES.items():
        d = load_sleeve(cfg)
        if d is None:
            print(f"{nm}: data missing, skipped")
            continue
        fc = (champion_signal(d["close"]) if cfg["scale"] == 1.0
              else champion_recipe(d["close"], cfg["scale"], 1.5))
        base = engine_bp(d, fc, cfg["cost_bp"], cfg["ann"], cfg["hl"])
        if base.empty or base.std() == 0:
            print(f"{nm}: no usable series, skipped")
            continue
        base_by[nm] = base
        print(f"\n{nm}  ({len(d):,} bars {d.index[0].date()}..{d.index[-1].date()})")
        print(f"  {'variant':30} {'SR':>8} {'DD':>8} {'t@matched':>10} {'years':>7}")
        print(f"  {'base champion':30} {sharpe(base):>+8.3f} {dd_of(base):>+8.1f} "
              f"{'—':>10} {'—':>7}")
        q = vol_quintile(d, cfg["ann"], cfg["hl"], cfg["pct_win"])
        for sname, sched in SCHEDULES.items():
            m = q.map(sched).astype(float)
            dt = engine_bp(d, (fc * m).clip(-2, 2), cfg["cost_bp"], cfg["ann"], cfg["hl"])
            t, yrs = gate(dt, base)
            star = "  <- PRIMARY" if sname == PRIMARY else ""
            print(f"  {sname:30} {sharpe(dt):>+8.3f} {dd_of(dt):>+8.1f} {t:>+10.2f} "
                  f"{yrs:>7}{star}")
            if sname == PRIMARY:
                taper_by[nm] = dt
                for s0, s1, lbl in SPLITS:
                    ds, bs = dt.loc[s0:s1], base.loc[s0:s1]
                    if len(ds) > 30 and ds.std() > 0:
                        ts, ys = gate(ds, bs)
                        split_rows.append(dict(sleeve=nm, half=lbl, sr_taper=sharpe(ds),
                                               sr_base=sharpe(bs), t=ts, years=ys))

    # ------------------------------------------------ split-sample, primary only
    print(f"\n{'='*94}\nSPLIT-SAMPLE, PRIMARY schedule only (the control that caught §3ab's artifact)\n{'='*94}")
    print(f"{'sleeve':22} {'half':>9} {'taper SR':>9} {'base SR':>9} {'t@matched':>10} {'years':>7}")
    for r in split_rows:
        print(f"{r['sleeve']:22} {r['half']:>9} {r['sr_taper']:>+9.3f} {r['sr_base']:>+9.3f} "
              f"{r['t']:>+10.2f} {r['years']:>7}")

    # ------------------------------------------------ PORTFOLIO (the primary claim)
    print(f"\n{'='*94}\nPORTFOLIO — equal-weight sleeves, taper applied to EVERY sleeve\n{'='*94}")
    common = None
    for s in base_by.values():
        common = s.index if common is None else common.intersection(s.index)
    if common is None or len(common) < 200:
        print("insufficient overlap for a portfolio test")
        return
    pb = pd.concat([s.reindex(common).fillna(0.0) for s in base_by.values()],
                   axis=1).mean(axis=1)
    pt = pd.concat([taper_by[k].reindex(common).fillna(0.0) for k in base_by],
                   axis=1).mean(axis=1)
    t_p, yrs_p = gate(pt, pb)
    _, t_raw, _ = paired(pt, pb)
    print(f"  overlap {len(common)} days ({common[0].date()}..{common[-1].date()}), "
          f"{len(base_by)} sleeves equal-weight")
    print(f"  {'portfolio':30} {'SR':>8} {'DD':>8} {'t@matched':>10} {'years':>7}")
    print(f"  {'base (no taper)':30} {sharpe(pb):>+8.3f} {dd_of(pb):>+8.1f} {'—':>10} {'—':>7}")
    print(f"  {'tapered (PRIMARY)':30} {sharpe(pt):>+8.3f} {dd_of(pt):>+8.1f} "
          f"{t_p:>+10.2f} {yrs_p:>7}   (raw t {t_raw:+.2f})")
    print("  portfolio split-sample:")
    for s0, s1, lbl in SPLITS:
        a, b = pt.loc[s0:s1], pb.loc[s0:s1]
        if len(a) > 30 and a.std() > 0:
            ts, ys = gate(a, b)
            print(f"    {lbl}: taper SR {sharpe(a):+.3f}  base SR {sharpe(b):+.3f}  "
                  f"t{ts:+.2f}  years {ys}")

    # ------------------------------------------------ verdict
    print(f"\n{'='*94}\nVERDICT vs the pre-registered criterion\n{'='*94}")
    n_sleeve_up = sum(1 for k in taper_by if sharpe(taper_by[k]) > sharpe(base_by[k]))
    n_sig = 0
    for k in taper_by:
        t, _ = gate(taper_by[k], base_by[k])
        n_sig += int(t > 1.64)
    print(f"  sleeves where PRIMARY taper lifts Sharpe : {n_sleeve_up}/{len(taper_by)}")
    print(f"  sleeves where it is significant (t>1.64) : {n_sig}/{len(taper_by)}")
    print(f"  portfolio t@matched                      : {t_p:+.2f}  (years {yrs_p})")
    both_halves = {}
    for r in split_rows:
        both_halves.setdefault(r["sleeve"], []).append(r["sr_taper"] > r["sr_base"])
    n_both = sum(1 for v in both_halves.values() if all(v))
    print(f"  sleeves improving in BOTH halves         : {n_both}/{len(both_halves)}")

    out = ROOT / "data" / "v5_runs" / "volregime_taper_crossasset.csv"
    pd.DataFrame(split_rows).to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
