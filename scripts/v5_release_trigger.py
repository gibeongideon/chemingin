"""Scheduled-release trigger on XAUUSD: a PRE-DECLARED 72-cell grid with a max-stat null.

THE ASK. §3be closed opening-range breakouts (real direction, 1-3bp gross against a 2.34bp round
trip, t ~1 against a null median of 2.58). The one untested branch left is the SCHEDULED RELEASE:
the 13:30 UTC slot runs 13.04bp mean |return| against a ~6bp all-day baseline, more than double,
and unlike a session open it has a known calendar.

WHAT THE CLOCK CALIBRATION ALREADY REVEALED. §3be measured server = UTC with a one-hour seasonal
shift, which means the 13:30 UTC peak is TWO DIFFERENT EVENTS by season:

    Apr-Oct (EDT, UTC-4):  12:30 UTC = 08:30 ET release,  13:30 UTC = 09:30 ET equity open
    Nov-Mar (EST, UTC-5):  13:30 UTC = 08:30 ET release,  14:30 UTC = 09:30 ET equity open

Pooled in UTC those overlap, which is exactly why 13:30 looked like the single biggest slot.
Anchoring in America/New_York separates them, and it is the reason this test is not just §3be
again with different hours. 09:30 ET is deliberately NOT an anchor here — that is the NY equity
open and §3be already tested it as a session.

NO ECONOMIC CALENDAR IS AVAILABLE, and that bounds the claim honestly. This cannot say "CPI days
behave differently from PPI days". What it CAN do is anchor on the RELEASE CLOCK — the times the
US publishes at — and condition on the size of the initial reaction, which is observable. So the
question is not "does CPI move gold" (obviously) but **"does the first 15-30 minutes after a
release-time window predict the next few hours?"** That is a tradeable question and it needs no
calendar.

THE GRID, DECLARED HERE BEFORE IT RUNS:

    anchor    08:30 ET (BLS/BEA window) / 10:00 ET (ISM, confidence) / 14:00 ET (FOMC)   3
    initial   first 15 / 30 minutes after the anchor                                     2
    rule      CONTINUE (follow the initial move) / FADE (the first move is the fake one)  2
    hold      1h / 3h after the initial window closes                                    2
    size      ALL / BIG (top 30% initial move) / SMALL (bottom 30%), by trailing 60-day   3
                                                                                      = 72

One entry convention (the close of the initial window — it is the first knowable price), costs
charged at the live 2.34bp round trip, and the max-statistic null run over exactly these 72.
Nothing is tuned afterwards. §3ak's null had a MEDIAN max |z| of 9.34 over 21,300 tests; the
whole point of declaring 72 is that its null median is a number a real effect could beat.

FADE IS IN THE GRID ON PURPOSE. "The first move after the news is the fake move" is one of the
most widely repeated claims in retail trading, and it is the exact mirror of CONTINUE, so the two
together cannot both be noise-positive: a consistent sign across cells with the mirror negative
is evidence, and that is how §3be established the breakout direction was real even while being
too small to trade.

    python scripts/v5_release_trigger.py --draws 400
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "v5_runs" / "session_study"

from scripts.v5_session_profile import session_open_utc  # noqa: E402

ANCHORS = {"0830ET": (8, 30), "1000ET": (10, 0), "1400ET": (14, 0)}
INIT_MIN = (15, 30)
RULES = ("CONTINUE", "FADE")
HOLD_H = (1.0, 3.0)
SIZE = ("ALL", "BIG", "SMALL")
COST_BP = 2.34
TRAIL = 60


def events(m15: pd.DataFrame, anchor: str, init_min: int, hold_h: float) -> pd.DataFrame:
    hh, mm = ANCHORS[anchor]
    days = pd.DatetimeIndex(sorted(set(m15.index.normalize())))
    days = days[days.dayofweek < 5]
    t0s = session_open_utc(days, "America/New_York", hh, mm)
    mi = m15.index
    cl = m15["close"].values
    nb = init_min // 15
    rows = []
    for d, t0 in zip(days, t0s):
        s = int(np.searchsorted(mi.values, np.datetime64(t0), side="left"))
        if s >= len(mi) or mi[s] != t0:
            continue                       # the anchor must land on a real M15 bar
        j = s + nb - 1                     # last bar OF the initial window
        e = int(np.searchsorted(mi.values,
                               np.datetime64(t0 + pd.Timedelta(minutes=init_min)
                                             + pd.Timedelta(hours=hold_h)), side="left"))
        if j >= len(mi) or e >= len(mi) or e <= j:
            continue
        p0, p1, p2 = cl[s - 1] if s > 0 else cl[s], cl[j], cl[e]
        init_bp = (p1 - p0) / p0 * 1e4
        if init_bp == 0:
            continue
        side = 1 if init_bp > 0 else -1
        rows.append(dict(day=d, init_bp=init_bp, abs_init=abs(init_bp), side=side,
                         fwd_bp=side * (p2 - p1) / p1 * 1e4,
                         dow=d.dayofweek,
                         first_friday=(d.dayofweek == 4 and d.day <= 7)))
    df = pd.DataFrame(rows)
    if len(df):
        a = df["abs_init"]
        df["sz_pct"] = [np.nan if i < TRAIL
                        else float((a.iloc[i - TRAIL:i] < a.iloc[i]).mean())
                        for i in range(len(a))]
    return df


def cells(m15: pd.DataFrame):
    res, store = [], {}
    for anchor in ANCHORS:
        for im in INIT_MIN:
            for hh in HOLD_H:
                ev = events(m15, anchor, im, hh)
                if not len(ev):
                    continue
                for rule in RULES:
                    sgn = 1.0 if rule == "CONTINUE" else -1.0
                    for sz in SIZE:
                        if sz == "ALL":
                            m = ev["sz_pct"].notna()
                        elif sz == "BIG":
                            m = ev["sz_pct"] >= 0.70
                        else:
                            m = ev["sz_pct"] <= 0.30
                        sub = ev[m]
                        if len(sub) < 100:
                            continue
                        gross = sgn * sub["fwd_bp"]
                        net = gross - COST_BP
                        yrs = pd.DatetimeIndex(sub["day"]).year
                        per = net.groupby(yrs).mean()
                        span_wk = (sub["day"].max() - sub["day"].min()).days / 7
                        key = f"{anchor}_{im}_{rule}_{int(hh)}h_{sz}"
                        store[key] = net.values
                        res.append(dict(
                            cell=key, anchor=anchor, init=im, rule=rule, hold=hh, size=sz,
                            n=len(sub), per_week=len(sub) / span_wk,
                            init_bp=float(sub["abs_init"].mean()),
                            gross_bp=float(gross.mean()), net_bp=float(net.mean()),
                            t=float(net.mean() / (net.std() / np.sqrt(len(net)))),
                            hit=float((gross > COST_BP).mean()),
                            years_pos=int((per > 0).sum()), years=int(len(per))))
    return pd.DataFrame(res), store


def maxstat(store: dict, draws: int, block: int = 10, seed: int = 11) -> dict:
    """Null of the maximum |t| across the declared 72. Block bootstrap under a zero-mean null."""
    rng = np.random.default_rng(seed)
    obs = {k: abs(a.mean() / (a.std() / np.sqrt(len(a)))) for k, a in store.items()}
    best = max(obs, key=obs.get)
    mx = np.empty(draws)
    for i in range(draws):
        m = 0.0
        for a in store.values():
            n = len(a)
            nb = int(np.ceil(n / block))
            st = rng.integers(0, max(n - block, 1), nb)
            b = np.concatenate([a[x:x + block] for x in st])[:n] - a.mean()
            m = max(m, abs(b.mean() / (b.std() / np.sqrt(n))))
        mx[i] = m
    q = lambda v: float(np.percentile(mx, v))
    return dict(cells=len(store), draws=draws, best_cell=best,
                observed_max_t=float(obs[best]), p50=q(50), p95=q(95), p99=q(99),
                null_max=float(mx.max()), p_value=float(np.mean(mx >= obs[best])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=400)
    a = ap.parse_args()
    from scripts.v5_advisor_measure import load_frames
    _, m15 = load_frames()
    df, store = cells(m15)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "release_cells.csv", index=False)

    print("=" * 106)
    print(f"RELEASE-TIME TRIGGER — {len(df)} of 72 declared cells admissible, net of "
          f"{COST_BP}bp round trip")
    print("=" * 106)
    print("  how big is the initial reaction at each anchor? (mean |first-window move|)")
    for anchor in ANCHORS:
        for im in INIT_MIN:
            sub = df[(df.anchor == anchor) & (df.init == im)]
            if len(sub):
                print(f"    {anchor} first {im:2d}min: {sub.init_bp.iloc[0]:6.2f}bp")
    print()
    print(f"  {'cell':34s} {'n':>5s} {'/wk':>5s} {'gross':>7s} {'net bp':>8s} {'t':>6s} "
          f"{'hit':>6s} {'yrs+':>7s}")
    for _, r in df.sort_values("net_bp", ascending=False).head(14).iterrows():
        print(f"  {r['cell']:34s} {r['n']:5d} {r['per_week']:5.2f} {r['gross_bp']:+7.2f} "
              f"{r['net_bp']:+8.2f} {r['t']:+6.2f} {r['hit']*100:5.1f}% "
              f"{r['years_pos']:3d}/{r['years']:<3d}")
    print(f"  ... {len(df)-14} more, worst net {df.net_bp.min():+.2f}bp")

    print("\n  MIRROR CHECK (CONTINUE vs its exact opposite FADE):")
    for rule in RULES:
        sub = df[df.rule == rule]
        print(f"    {rule:9s} mean gross {sub.gross_bp.mean():+6.3f}bp   "
              f"positive in {(sub.gross_bp > 0).sum()}/{len(sub)} cells")
    print(f"    net-positive cells overall: {(df.net_bp > 0).sum()} of {len(df)}")

    print(f"\nrunning the max-stat null over all {len(store)} cells, {a.draws} draws ...")
    n = maxstat(store, a.draws)
    (OUT / "release_null.json").write_text(json.dumps(n, indent=2))
    print("=" * 106)
    print(f"  best cell by |t|     {n['best_cell']}   observed |t| {n['observed_max_t']:.3f}")
    print(f"  null of the maximum  p50 {n['p50']:.3f}  p95 {n['p95']:.3f}  p99 {n['p99']:.3f}  "
          f"max {n['null_max']:.3f}")
    print(f"  Reality-Check p      {n['p_value']:.4f}  -> "
          f"{'SURVIVES' if n['p_value'] < 0.05 else 'DOES NOT SURVIVE'}")
    bp = df[df.net_bp > 0].sort_values("t", ascending=False)
    if len(bp):
        r = bp.iloc[0]
        print(f"\n  best NET-POSITIVE cell: {r['cell']}  net {r['net_bp']:+.2f}bp  "
              f"t {r['t']:+.2f}  {r['per_week']:.2f}/wk  {r['years_pos']}/{r['years']} yrs")
        print(f"    its t vs the null median {n['p50']:.3f}: "
              f"{'clears' if r['t'] > n['p95'] else 'NOWHERE NEAR'}")
    else:
        print("\n  no net-positive cell at all.")
    print(f"\nwrote {(OUT / 'release_cells.csv').relative_to(ROOT)}, release_null.json")


if __name__ == "__main__":
    main()
