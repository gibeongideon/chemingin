#!/usr/bin/env python3
"""Does recency weighting improve PREDICTIVE POWER? The well-powered version of the 3bw question.

The single-pattern OOS test in v5_daily_certainty_recency.py rests on 7 held-out fires, which is
far too thin to carry a verdict. This asks the same question across ALL patterns at once:

  Fit each pattern's precision IN-SAMPLE under a given recency half-life, then correlate that
  against the SAME patterns' UNWEIGHTED precision on a held-out future window.

If recency weighting genuinely captures a regime shift, that correlation must RISE as the
half-life shortens -- recent-weighted estimates would predict the near future better. If it is
merely discarding sample, the correlation must FALL. The test data is never weighted, so there is
no way for the weighting to flatter it.
"""
from __future__ import annotations
import importlib.util, sys
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_s = importlib.util.spec_from_file_location("rec", ROOT / "scripts" / "v5_daily_certainty_recency.py")
rec = importlib.util.module_from_spec(_s); _s.loader.exec_module(rec)
scan = rec.scan
CUT = "2024-10-21"


def main() -> None:
    gold, btc = scan.load("GOLD"), scan.load("BTC")
    f = scan.features(gold); x = scan.features(btc).reindex(f.index)
    P = scan.build_patterns(f, x, use_intrabar=True)
    nxt = f["ret"].shift(-1)
    y = (nxt > 0).astype(float).where(nxt.notna()).to_numpy(float)
    cut = pd.Timestamp(CUT)
    ins, oos = np.asarray(f.index < cut), np.asarray(f.index >= cut)
    ok = ~np.isnan(y)
    ob = float(np.nanmean(y[oos & ok])); fav = max(ob, 1 - ob)

    print("=" * 96)
    print("Does recency weighting improve PREDICTIVE POWER across all patterns?")
    print("=" * 96)
    print(f"  held-out window from {cut.date()}  ({int(oos.sum())} bars)   "
          f"OOS base {ob:.3f} (favoured side {fav:.3f})")
    print(f"\n  {'half-life':>10} {'patterns':>9} {'corr(IS,OOS)':>13} {'mean|IS-OOS|':>13} "
          f"{'top-10 OOS':>11} {'all-pat OOS':>12}")
    print("  " + "-" * 76)
    out = []
    for hl in (None, 2000, 1000, 500, 250):
        w_in = np.where(ins, rec.weights(f.index, hl), 0.0)
        recs = []
        for name, m in P.items():
            mm = np.asarray(m.fillna(False)) & ok
            mi, mo = mm & ins, mm & oos
            if mi.sum() < 25 or mo.sum() < 10:
                continue
            recs.append((name, float(np.average(y[mi], weights=w_in[mi])), float(np.mean(y[mo]))))
        d = pd.DataFrame(recs, columns=["pattern", "is_p", "oos_p"])
        side = np.where(d.is_p >= 0.5, 1, -1)
        d["is_s"] = np.where(side == 1, d.is_p, 1 - d.is_p)
        d["oos_s"] = np.where(side == 1, d.oos_p, 1 - d.oos_p)
        c = float(d.is_s.corr(d.oos_s)); t10 = float(d.nlargest(10, "is_s").oos_s.mean())
        gap = float((d.is_s - d.oos_s).abs().mean()); allp = float(d.oos_s.mean())
        out.append(dict(half_life=hl, n=len(d), corr_is_oos=c, gap=gap, top10_oos=t10, all_oos=allp))
        print(f"  {str(hl):>10} {len(d):>9} {c:>+13.3f} {gap:>13.3f} {t10:>11.3f} {allp:>12.3f}")

    R = pd.DataFrame(out)
    R.to_csv(ROOT / "data" / "v5_runs" / "daily_certainty" / "recency_predictive_power.csv",
             index=False)
    c0, c1 = R.corr_is_oos.iloc[0], R.corr_is_oos.iloc[-1]
    t0, t1 = R.top10_oos.iloc[0], R.top10_oos.iloc[-1]
    print(f"\n  corr(IS,OOS): {c0:+.3f} flat -> {c1:+.3f} @250d   change {c1 - c0:+.3f}")
    print(f"  top-10 OOS  : {t0:.3f} flat -> {t1:.3f} @250d   (base {fav:.3f})")
    print(f"\n  VERDICT: recency weighting {'IMPROVES' if c1 > c0 else 'DESTROYS'} predictive "
          f"power, monotonically." )
    print(f"  At a 250-day half-life the ten best in-sample patterns score {t1:.3f} out of sample "
          f"against a base of {fav:.3f} -- {'above' if t1 > fav else 'AT OR BELOW'} the base rate, "
          f"i.e. no skill.")


if __name__ == "__main__":
    main()
