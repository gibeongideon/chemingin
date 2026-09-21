"""Opening-range trigger on XAUUSD sessions: a PRE-DECLARED 54-cell grid with a max-stat null.

THE ASK. One signal with a real edge, firing roughly twice a week, from session-open behaviour.
Not trading the whole period — acting only when the trigger comes.

THE GRID IS DECLARED HERE, BEFORE IT RUNS, AND IT IS SMALL ON PURPOSE. §3ak searched 21,300
conditional triggers on XAU H4 and its best one was indistinguishable from what a no-edge dataset
hands you (Reality-Check p 0.110; the null's MEDIAN max |z| was 9.34). The lesson recorded there
is that searching harder raises the bar faster than the statistic, so this run declares:

    session   ASIA / LONDON / NY                                    3
    open range  first 15 / 30 / 60 minutes                          3
    rule      BREAKOUT (follow the break) / FADE (take the other side)   2
    width     ALL / NARROW (bottom 30%) / WIDE (top 30%) of the trailing
              60-session opening-range width                        3
                                                                  = 54 cells

and nothing else. One exit convention (session close), because §3ab already exhausted a 64-cell
exit sweep on XAU and found 0/64 clearing +0.50. One entry convention. No parameter tuning.
The max-statistic null is then run over exactly these 54, which is the number that was declared.

CAUSALITY. The opening range is the first k minutes; it cannot be acted on until it has closed.
Entry is at the CLOSE of the M15 bar that breaks the range — deliberately worse than a resting
stop order filled at the level, because a stop's fill is an assumption and a close is observed.
The trailing width percentile uses the previous 60 sessions only, never the current one.

COSTS ARE NOT OPTIONAL HERE. §3ay's breakeven table: at M15 the spread demands a 0.587 hit rate
against 0.510 at D1. The live Maven XAUUSD spread is $0.51 on $4,364 = 1.17bp one way, so a round
trip is 2.34bp, and `ftmo-xau-diversifier-search` warns that CSV spreads understate the real
thing 20-50x, so the LIVE figure is used and floored rather than anything from a file. Gross and
net are both printed; only net decides.

THE ORACLE, for scale (mean |session return|): ASIA 30.8bp, LONDON 28.4bp, NY 41.8bp. That is
what a perfect session-direction call is worth, so a trigger returning 3bp has captured under a
tenth of the available move and should be read that way.

    python scripts/v5_session_orb.py --draws 400
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "v5_runs" / "session_study"

from scripts.v5_session_profile import SESSIONS, session_open_utc  # noqa: E402

OR_MINUTES = (15, 30, 60)
RULES = ("BREAKOUT", "FADE")
WIDTH = ("ALL", "NARROW", "WIDE")
SPREAD_BP_ROUND_TRIP = 2.34          # live Maven XAUUSD $0.51 on $4,364, both ways
TRAIL = 60                            # sessions for the width percentile


def trades(m15: pd.DataFrame, session: str, or_min: int) -> pd.DataFrame:
    """One row per session-day: the opening range, the first break, and the outcome."""
    tz, hh, mm, hours = SESSIONS[session]
    days = pd.DatetimeIndex(sorted(set(m15.index.normalize())))
    days = days[days.dayofweek < 5]
    opens = session_open_utc(days, tz, hh, mm)
    mi = m15.index
    hi, lo, cl, op = (m15["high"].values, m15["low"].values,
                      m15["close"].values, m15["open"].values)
    nb = or_min // 15
    rows = []
    for d, t0 in zip(days, opens):
        s = int(np.searchsorted(mi.values, np.datetime64(t0), side="left"))
        if s >= len(mi) or mi[s] != t0:
            continue
        e = int(np.searchsorted(mi.values,
                                np.datetime64(t0 + pd.Timedelta(hours=hours)), side="left"))
        if e >= len(mi) or e - s < nb + 4:
            continue
        or_hi, or_lo = hi[s:s + nb].max(), lo[s:s + nb].min()
        or_open = op[s]
        width_bp = (or_hi - or_lo) / or_open * 1e4
        # first break AFTER the opening range has closed
        j, side, entry = None, 0, np.nan
        for k in range(s + nb, e):
            up, dn = hi[k] > or_hi, lo[k] < or_lo
            if up and dn:                 # both in one bar: genuinely ambiguous, skip the day
                j, side = None, 0
                break
            if up:
                j, side, entry = k, +1, cl[k]
                break
            if dn:
                j, side, entry = k, -1, cl[k]
                break
        if j is None:
            continue
        exit_px = cl[e]
        rows.append(dict(day=d, t0=t0, width_bp=width_bp, side=side, entry=entry,
                         exit=exit_px, bars_to_break=j - (s + nb),
                         fwd_bp=side * (exit_px - entry) / entry * 1e4))
    df = pd.DataFrame(rows)
    if len(df):
        # trailing width percentile: PREVIOUS 60 sessions only
        df["w_pct"] = (df["width_bp"].rolling(TRAIL, closed="left")
                       .apply(lambda x: (x < x.iloc[-1]).mean() if len(x) else np.nan))
        df["w_pct"] = df["width_bp"].shift(0)  # placeholder replaced below
        w = df["width_bp"]
        ranks = []
        for i in range(len(w)):
            if i < TRAIL:
                ranks.append(np.nan)
            else:
                prev = w.iloc[i - TRAIL:i]
                ranks.append(float((prev < w.iloc[i]).mean()))
        df["w_pct"] = ranks
    return df


def cells(m15: pd.DataFrame) -> pd.DataFrame:
    res, store = [], {}
    for ses in SESSIONS:
        for om in OR_MINUTES:
            t = trades(m15, ses, om)
            if not len(t):
                continue
            for rule in RULES:
                sgn = 1.0 if rule == "BREAKOUT" else -1.0
                for w in WIDTH:
                    if w == "ALL":
                        m = t["w_pct"].notna()
                    elif w == "NARROW":
                        m = t["w_pct"] <= 0.30
                    else:
                        m = t["w_pct"] >= 0.70
                    sub = t[m]
                    if len(sub) < 100:
                        continue
                    gross = sgn * sub["fwd_bp"]
                    net = gross - SPREAD_BP_ROUND_TRIP
                    yrs = pd.DatetimeIndex(sub["day"]).year
                    per = net.groupby(yrs).mean()
                    span_wk = (sub["day"].max() - sub["day"].min()).days / 7
                    key = f"{ses}_{om}_{rule}_{w}"
                    store[key] = pd.Series(net.values, index=pd.DatetimeIndex(sub["day"]))
                    res.append(dict(cell=key, session=ses, or_min=om, rule=rule, width=w,
                                    n=len(sub), per_week=len(sub) / span_wk,
                                    gross_bp=float(gross.mean()), net_bp=float(net.mean()),
                                    sd=float(net.std()),
                                    t=float(net.mean() / (net.std() / np.sqrt(len(net)))),
                                    hit=float((gross > SPREAD_BP_ROUND_TRIP).mean()),
                                    years_pos=int((per > 0).sum()), years=int(len(per))))
    return pd.DataFrame(res), store


def maxstat_null(store: dict, draws: int, block: int = 20, seed: int = 11) -> dict:
    """Null of the MAXIMUM |t| across the declared 54 cells, stationary block bootstrap.

    Cells share the same underlying days, so their statistics are correlated and the maximum is
    what must be priced — not each cell on its own. Blocks preserve the serial dependence that
    an analytic t assumes away.
    """
    rng = np.random.default_rng(seed)
    keys = list(store)
    arrs = {k: store[k].values for k in keys}
    obs = {k: abs(a.mean() / (a.std() / np.sqrt(len(a)))) for k, a in arrs.items()}
    best_cell = max(obs, key=obs.get)
    mx = np.empty(draws)
    for i in range(draws):
        m = 0.0
        for k, a in arrs.items():
            n = len(a)
            nb = int(np.ceil(n / block))
            st = rng.integers(0, max(n - block, 1), nb)
            b = np.concatenate([a[x:x + block] for x in st])[:n]
            b = b - a.mean()                     # impose the null of zero mean
            m = max(m, abs(b.mean() / (b.std() / np.sqrt(n))))
        mx[i] = m
    q = lambda v: float(np.percentile(mx, v))
    return dict(cells=len(keys), draws=draws, observed_best_cell=best_cell,
                observed_max_t=float(obs[best_cell]), null_p50=q(50), null_p95=q(95),
                null_p99=q(99), null_max=float(mx.max()),
                p_value=float(np.mean(mx >= obs[best_cell])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=400)
    a = ap.parse_args()
    from scripts.v5_advisor_measure import load_frames
    _, m15 = load_frames()
    df, store = cells(m15)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "orb_cells.csv", index=False)

    print("=" * 108)
    print(f"OPENING-RANGE TRIGGER — {len(df)} of 54 declared cells admissible, "
          f"net of {SPREAD_BP_ROUND_TRIP}bp round trip")
    print("=" * 108)
    print(f"  {'cell':26s} {'n':>5s} {'/wk':>5s} {'gross':>7s} {'net bp':>8s} {'t':>6s} "
          f"{'hit':>6s} {'yrs+':>7s}")
    for _, r in df.sort_values("t", ascending=False).iterrows():
        print(f"  {r['cell']:26s} {r['n']:5d} {r['per_week']:5.2f} {r['gross_bp']:+7.2f} "
              f"{r['net_bp']:+8.2f} {r['t']:+6.2f} {r['hit']*100:5.1f}% "
              f"{r['years_pos']:3d}/{r['years']:<3d}")

    print(f"\nrunning the max-statistic null over all {len(store)} cells, {a.draws} draws ...")
    null = maxstat_null(store, a.draws)
    (OUT / "orb_null.json").write_text(json.dumps(null, indent=2))
    print("=" * 108)
    print(f"  best cell            {null['observed_best_cell']}")
    print(f"  observed max |t|     {null['observed_max_t']:.3f}")
    print(f"  null of the maximum  p50 {null['null_p50']:.3f}   p95 {null['null_p95']:.3f}   "
          f"p99 {null['null_p99']:.3f}   max {null['null_max']:.3f}")
    print(f"  Reality-Check p      {null['p_value']:.4f}  -> "
          f"{'SURVIVES' if null['p_value'] < 0.05 else 'DOES NOT SURVIVE'}")
    print(f"\nwrote {(OUT / 'orb_cells.csv').relative_to(ROOT)}, orb_null.json")


if __name__ == "__main__":
    main()
