#!/usr/bin/env python3
"""Does BTC's WEEKEND move predict gold's Monday gap? (independent exploration, 2026-10-21)

THE MECHANISM, stated before the test so it can be falsified rather than fitted. Gold shuts Friday
~21:00 UTC and reopens Sunday ~22:00 -- a ~49 hour blackout. BTC never shuts. Macro information
arriving over a weekend (geopolitics, policy, risk events) therefore has only one liquid
debasement-asset venue to price into, and gold must catch up at its re-open. This is NOT a
correlation study: correlation between two assets that trade simultaneously is a different claim.
It is an INFORMATION-RELEASE ORDERING, and it is only testable across a closure.

WHY THE WEEKEND RATHER THAN THE NIGHTLY BREAK. The nightly break is ~1h and would need intraday
BTC, of which the broker serves only 7 months (~150 events, SE 0.041 -- underpowered). The weekend
is a 49h closure, and BTC D1 covers every calendar day back to 2014 while gold D1 covers business
days, so the pair is directly computable over ~600 weekends from data this repo already trusts.

CONTROLS CARRIED:
  * The benchmark is gold's OWN Friday return (autocorrelation), not zero. If BTC adds nothing
    beyond gold's own momentum, the mechanism is not what is being measured.
  * Base rate printed beside every hit rate -- gold gaps up more often than down, and a threshold
    at 0.5 would be wrong (xau-drift-baseline-inverts-3r).
  * EV per fire in bp against the live 0.76bp round trip, and against HOLDING gold (control #2).
  * The SKIPPED sample is measured: what happens on the weekends the filter declines.
  * Split-half, because a 2020-2021 crypto-macro co-move could carry the whole result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v5_runs" / "intraday_explore"
RT_BP = 0.504 * 2 * 0.75      # live Switch round trip = 0.76bp


def load(alias: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / f"{alias}_D1_long.csv", parse_dates=["time"])
    d = d.set_index("time").sort_index()
    return d[~d.index.duplicated(keep="last")]


def build() -> pd.DataFrame:
    gold, btc = load("GOLD"), load("BTC")
    g = gold.copy()
    g["dow"] = g.index.dayofweek
    rows = []
    idx = list(g.index)
    for i in range(1, len(idx)):
        prev, cur = idx[i - 1], idx[i]
        # a weekend = a gap of 2+ calendar days between consecutive GOLD sessions
        if (cur - prev).days < 2:
            continue
        gap = g.at[cur, "open"] / g.at[prev, "close"] - 1.0          # gold's catch-up
        mon = g.at[cur, "close"] / g.at[cur, "open"] - 1.0           # Monday's own session
        fri = g.at[prev, "close"] / g.at[prev, "open"] - 1.0         # gold's Friday session
        # BTC across exactly the same wall-clock window
        bw = btc.loc[(btc.index > prev) & (btc.index <= cur), "close"]
        bprev = btc.loc[btc.index <= prev, "close"]
        if len(bw) == 0 or len(bprev) == 0:
            continue
        btc_wknd = bw.iloc[-1] / bprev.iloc[-1] - 1.0
        rows.append(dict(date=cur, gap=gap, monday=mon, fri=fri, btc_wknd=btc_wknd,
                         days=(cur - prev).days))
    return pd.DataFrame(rows).set_index("date")


def report(d: pd.DataFrame, name: str, sig: pd.Series, target: str) -> dict:
    """Score a signal: hit rate vs base, EV vs cost, and the SKIPPED sample."""
    y = d[target]
    fired = sig.fillna(False)
    n = int(fired.sum())
    if n < 20:
        return {}
    side = np.sign(d.loc[fired, "btc_wknd"]).replace(0, 1)
    ret = (y[fired] * side)
    base_up = float((y > 0).mean())
    hit = float((ret > 0).mean())
    ev = float(ret.mean() * 1e4) - RT_BP
    skipped = float(y[~fired].mean() * 1e4)
    return dict(name=name, n=n, hit=hit, base=base_up, ev_bp=ev,
                hold_bp=float(y.mean() * 1e4), skipped_bp=skipped,
                t=float(ret.mean() / (ret.std() / np.sqrt(n))) if ret.std() > 0 else np.nan)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = build()
    d.to_csv(OUT / "weekend_btc_gold.csv")
    print("=" * 100)
    print("C.  DOES BTC'S WEEKEND MOVE PREDICT GOLD'S MONDAY GAP?")
    print("=" * 100)
    print(f"  {len(d):,} weekends, {d.index.min().date()} -> {d.index.max().date()}  "
          f"(median closure {d['days'].median():.0f} days)")
    print(f"  gold weekend GAP: mean {d.gap.mean()*1e4:+.2f}bp  sd {d.gap.std()*1e4:.1f}bp  "
          f"up-rate {(d.gap>0).mean():.3f}")
    print(f"  BTC weekend move: mean {d.btc_wknd.mean()*1e4:+.2f}bp  sd {d.btc_wknd.std()*1e4:.1f}bp")

    print(f"\n  --- the raw relationship ---")
    for lbl, a, b in (("BTC weekend -> gold GAP", "btc_wknd", "gap"),
                      ("BTC weekend -> gold MONDAY session", "btc_wknd", "monday"),
                      ("gold FRIDAY -> gold GAP  [autocorr control]", "fri", "gap"),
                      ("gold FRIDAY -> gold MONDAY [control]", "fri", "monday")):
        c = d[a].corr(d[b])
        t = c * np.sqrt((len(d) - 2) / max(1e-12, 1 - c ** 2))
        print(f"    {lbl:<44} corr {c:+.4f}   t {t:+.2f}")

    print(f"\n  --- conditional: gold's gap given BTC's weekend direction ---")
    print(f"    {'BTC weekend':<22} {'n':>5} {'gold gap bp':>12} {'up-rate':>9} {'t':>7}")
    base = d.gap.mean() * 1e4
    for lbl, m in (("UP  (any)", d.btc_wknd > 0), ("DOWN (any)", d.btc_wknd < 0),
                   ("UP  > +2%", d.btc_wknd > 0.02), ("DOWN < -2%", d.btc_wknd < -0.02),
                   ("UP  > +5%", d.btc_wknd > 0.05), ("DOWN < -5%", d.btc_wknd < -0.05)):
        s = d.gap[m]
        if len(s) < 10:
            continue
        t = s.mean() / (s.std() / np.sqrt(len(s))) if s.std() > 0 else np.nan
        print(f"    {lbl:<22} {len(s):>5} {s.mean()*1e4:>+12.2f} {(s>0).mean():>9.3f} {t:>+7.2f}")
    print(f"    {'ALL (base)':<22} {len(d):>5} {base:>+12.2f} {(d.gap>0).mean():>9.3f}")

    print(f"\n  --- tradeable? directional bet on the gap, scored net of the live spread ---")
    print(f"    {'rule':<26} {'n':>5} {'hit':>7} {'base':>7} {'EV bp':>8} {'hold':>7} "
          f"{'SKIPPED':>9} {'t':>7}")
    res = []
    for thr in (0.0, 0.02, 0.05):
        r = report(d, f"|BTC wknd| > {thr:.0%}", d.btc_wknd.abs() > thr, "gap")
        if r:
            res.append(r)
            print(f"    {r['name']:<26} {r['n']:>5} {r['hit']:>7.3f} {r['base']:>7.3f} "
                  f"{r['ev_bp']:>+8.2f} {r['hold_bp']:>+7.2f} {r['skipped_bp']:>+9.2f} "
                  f"{r['t']:>+7.2f}")

    print(f"\n  --- split-half (is it a 2020-21 crypto-macro co-move?) ---")
    mid = d.index[len(d) // 2]
    for lbl, part in (("H1 " + str(d.index.min().date()) + "..", d[d.index < mid]),
                      ("H2 " + str(mid.date()) + "..", d[d.index >= mid])):
        c = part.btc_wknd.corr(part.gap)
        up = part.gap[part.btc_wknd > 0]
        dn = part.gap[part.btc_wknd < 0]
        print(f"    {lbl:<22} n {len(part):>4}  corr {c:+.4f}   "
              f"gap|BTC up {up.mean()*1e4:+7.2f}bp   gap|BTC dn {dn.mean()*1e4:+7.2f}bp   "
              f"spread {(up.mean()-dn.mean())*1e4:+7.2f}bp")

    print(f"\n  --- per-year spread (gap when BTC up minus gap when BTC down) ---")
    yrs = []
    for y, g in d.groupby(d.index.year):
        u, v = g.gap[g.btc_wknd > 0], g.gap[g.btc_wknd < 0]
        if len(u) < 3 or len(v) < 3:
            continue
        sp = (u.mean() - v.mean()) * 1e4
        yrs.append((y, len(g), sp))
        print(f"    {y}  n {len(g):>3}  spread {sp:>+8.2f}bp")
    pos = sum(1 for _, _, s in yrs if s > 0)
    print(f"\n    years with POSITIVE spread: {pos} of {len(yrs)}")


if __name__ == "__main__":
    main()
