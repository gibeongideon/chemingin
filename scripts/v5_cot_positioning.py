"""CFTC Managed-Money positioning as a XAUUSD signal. Pre-registered in COT_PREREGISTRATION.md.

WHY THIS IS DIFFERENT FROM EVERYTHING ELSE IN V5_FINDINGS. Every prior study here is price on
price. §3ao, §3be and §3bf each ended with the same sentence in different words — the limit is the
DATA TYPE, not the model. Positioning is a genuinely different observable: who holds the contracts,
published by the regulator, unavailable from any price series.

IT ALSO ESCAPES THE COST WALL FOR THE FIRST TIME. §3ay's breakeven table and §3be/§3bf's verdicts
all turned on the same arithmetic: 0.3-3bp of directional content against a 2.34bp round trip. At
a weekly horizon gold moves ~150bp, so the same round trip is **1.5% of the signal instead of
~100% of it**. Whatever kills this, it will not be the spread.

*** THE PUBLICATION LAG IS THE WHOLE GAME. *** COT is surveyed TUESDAY at close and published
FRIDAY 15:30 ET. A study that uses Tuesday's positioning to trade Tuesday's close is reading a
number that will not exist for three more days, and in published critiques that lookahead is worth
about the whole reported effect. So:

    declared rule: a report dated Tuesday T is actionable only from Friday T+3, 20:30 UTC.
    every signal is aligned to the FOLLOWING MONDAY's open, strictly later than the release.

An UNLAGGED arm is also run, deliberately, and reported as the SIZE OF THE LOOKAHEAD rather than
as a result. If the lagged arm is flat and the unlagged arm looks good, that difference IS the
finding, and it is the one thing a reader of a COT study most needs to know.

NORMALISATION, because a raw contract count is not comparable across a decade of changing open
interest (gold OI roughly doubled over this sample). All three use PRIOR weeks only:
    net_oi   net managed-money position / open interest
    z156     z-score of net_oi over the trailing 156 weeks
    cotidx   percentile rank of net_oi in its trailing 156 weeks (the retail-standard "COT index")

    python scripts/v5_cot_positioning.py --draws 400
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "v5_runs" / "session_study"
COT_DIR = Path("/tmp/claude-1000/-home-rock-Desktop-2026-Projects-Trader36-MT5/"
               "9fbdfea4-245d-40ee-814f-563e7efd71db/scratchpad/cot")

MARKET = "GOLD - COMMODITY EXCHANGE INC."
MEASURES = ("net_oi", "z156", "cotidx")
RULES = ("CONTRARIAN", "MOMENTUM")
EXTREME = (0.20, 0.30)
HORIZON_W = (1, 2, 4)
TRAIL = 156
COST_BP = 2.34


def load_cot() -> pd.DataFrame:
    files = sorted(glob.glob(str(COT_DIR / "t*" / "*.txt")))
    if not files:
        raise SystemExit(f"no COT text files under {COT_DIR}")
    keep = ["Market_and_Exchange_Names", "Report_Date_as_YYYY-MM-DD", "Open_Interest_All",
            "M_Money_Positions_Long_All", "M_Money_Positions_Short_All"]
    parts = []
    for f in files:
        d = pd.read_csv(f, low_memory=False)
        d.columns = [c.strip().strip('"') for c in d.columns]
        if not set(keep).issubset(d.columns):
            continue
        d = d[d["Market_and_Exchange_Names"].astype(str).str.strip().str.strip('"') == MARKET]
        parts.append(d[keep])
    if not parts:
        raise SystemExit("gold rows not found in any COT file")
    d = pd.concat(parts, ignore_index=True)
    d["date"] = pd.to_datetime(d["Report_Date_as_YYYY-MM-DD"])
    for c in ("Open_Interest_All", "M_Money_Positions_Long_All", "M_Money_Positions_Short_All"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Open_Interest_All"]).sort_values("date")
    d = d[~d["date"].duplicated(keep="last")].set_index("date")
    d["net"] = d["M_Money_Positions_Long_All"] - d["M_Money_Positions_Short_All"]
    d["net_oi"] = d["net"] / d["Open_Interest_All"]
    # causal trailing normalisations: prior weeks only, never the current one
    v = d["net_oi"]
    z, ci = [], []
    for i in range(len(v)):
        if i < TRAIL:
            z.append(np.nan); ci.append(np.nan)
        else:
            w = v.iloc[i - TRAIL:i]
            sd = w.std()
            z.append((v.iloc[i] - w.mean()) / sd if sd > 0 else np.nan)
            ci.append(float((w < v.iloc[i]).mean()))
    d["z156"], d["cotidx"] = z, ci
    return d


def price_at(mi: np.ndarray, px: np.ndarray, t: pd.Timestamp):
    """First observed price at or after `t`. Returns None if beyond the series."""
    i = int(np.searchsorted(mi, np.datetime64(t), side="left"))
    return (px[i], i) if i < len(px) else (None, None)


def build(cot: pd.DataFrame, m15: pd.DataFrame, lagged: bool) -> pd.DataFrame:
    mi, px = m15.index.values, m15["close"].values
    rows = []
    for dt, r in cot.iterrows():
        if not np.isfinite(r.get("z156", np.nan)):
            continue
        # LAGGED: Tuesday report -> published Friday 15:30 ET -> act at the NEXT Monday open.
        # UNLAGGED: act at the Tuesday close itself, which is the lookahead being measured.
        t_entry = (dt + pd.Timedelta(days=6)) if lagged else dt
        p0, i0 = price_at(mi, px, t_entry)
        if p0 is None:
            continue
        row = dict(report=dt, entry_t=mi[i0], net_oi=r["net_oi"], z156=r["z156"],
                   cotidx=r["cotidx"])
        for h in HORIZON_W:
            p1, _ = price_at(mi, px, pd.Timestamp(mi[i0]) + pd.Timedelta(weeks=h))
            row[f"fwd{h}"] = np.nan if p1 is None else (p1 - p0) / p0 * 1e4
        rows.append(row)
    return pd.DataFrame(rows)


def cells(ev: pd.DataFrame):
    res, store = [], {}
    for meas in MEASURES:
        v = ev[meas]
        # the extreme is taken on the measure's own trailing distribution, causally
        pct = [np.nan if i < TRAIL else float((v.iloc[i - TRAIL:i] < v.iloc[i]).mean())
               for i in range(len(v))]
        ev[f"{meas}_pct"] = pct
        for ex in EXTREME:
            hi_m, lo_m = ev[f"{meas}_pct"] >= 1 - ex, ev[f"{meas}_pct"] <= ex
            for rule in RULES:
                # CONTRARIAN: crowded LONG -> short gold.  MOMENTUM: crowded long -> long gold.
                sgn = -1.0 if rule == "CONTRARIAN" else +1.0
                side = pd.Series(0.0, index=ev.index)
                side[hi_m] = sgn
                side[lo_m] = -sgn
                for h in HORIZON_W:
                    m = (side != 0) & ev[f"fwd{h}"].notna()
                    if m.sum() < 40:
                        continue
                    sub = ev[m]
                    gross = (side[m] * sub[f"fwd{h}"]).values
                    net = gross - COST_BP
                    yrs = pd.DatetimeIndex(sub["report"]).year
                    per = pd.Series(net, index=yrs).groupby(level=0).mean()
                    span_wk = (sub["report"].max() - sub["report"].min()).days / 7
                    key = f"{meas}_{rule}_{int(ex*100)}pct_{h}w"
                    store[key] = gross
                    # the skipped sample: what a passive long earned on the weeks it did NOT fire
                    skipped = ev.loc[~m & ev[f"fwd{h}"].notna(), f"fwd{h}"]
                    res.append(dict(cell=key, measure=meas, rule=rule, extreme=ex, horizon=h,
                                    n=int(m.sum()), per_week=m.sum() / span_wk,
                                    gross_bp=float(gross.mean()), net_bp=float(net.mean()),
                                    t=float(gross.mean() / (gross.std() / np.sqrt(len(gross)))),
                                    hit=float((gross > COST_BP).mean()),
                                    skipped_long_bp=float(skipped.mean()) if len(skipped) else np.nan,
                                    years_pos=int((per > 0).sum()), years=int(len(per))))
    return pd.DataFrame(res), store


def maxstat_signed(store: dict, draws: int, block: int = 4, seed: int = 11) -> dict:
    """Null of the max SIGNED t on GROSS returns (§3bf: never bootstrap a net-of-cost series)."""
    rng = np.random.default_rng(seed)
    obs = {k: a.mean() / (a.std() / np.sqrt(len(a))) for k, a in store.items()}
    best = max(obs, key=lambda k: obs[k])
    mx = np.empty(draws)
    for i in range(draws):
        m = -9e9
        for a in store.values():
            n = len(a)
            nb = int(np.ceil(n / block))
            st = rng.integers(0, max(n - block, 1), nb)
            b = np.concatenate([a[x:x + block] for x in st])[:n] - a.mean()
            m = max(m, b.mean() / (b.std() / np.sqrt(n)))
        mx[i] = m
    q = lambda v: float(np.percentile(mx, v))
    return dict(cells=len(store), draws=draws, best_cell=best, observed_t=float(obs[best]),
                p50=q(50), p95=q(95), p99=q(99), p_value=float(np.mean(mx >= obs[best])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=400)
    a = ap.parse_args()
    from scripts.v5_advisor_measure import load_frames
    from src.v5.mtf_frames import frames
    _, m15 = load_frames()
    px = frames(m15)["M15"]
    cot = load_cot()
    print(f"COT gold: {len(cot):,} weekly reports  {cot.index.min().date()} -> "
          f"{cot.index.max().date()}")
    print(f"  managed-money net / OI: mean {cot.net_oi.mean():+.3f}  "
          f"range [{cot.net_oi.min():+.3f}, {cot.net_oi.max():+.3f}]")
    print(f"  open interest: {cot.Open_Interest_All.iloc[0]:,.0f} -> "
          f"{cot.Open_Interest_All.iloc[-1]:,.0f}  (why raw contract counts cannot be compared)")

    OUT.mkdir(parents=True, exist_ok=True)
    for lagged in (True, False):
        tag = "LAGGED (actionable: Monday after Friday release)" if lagged else \
              "UNLAGGED (trades Tuesday's close = THE LOOKAHEAD)"
        ev = build(cot, px, lagged)
        df, store = cells(ev)
        if not len(df):
            print(f"\n{tag}: no admissible cells"); continue
        df.to_csv(OUT / f"cot_cells_{'lag' if lagged else 'peek'}.csv", index=False)
        print("\n" + "=" * 104)
        print(f"{tag} — {len(df)} cells, {len(ev)} reports")
        print("=" * 104)
        print(f"  {'cell':34s} {'n':>4s} {'/wk':>5s} {'gross':>8s} {'net bp':>8s} {'t':>6s} "
              f"{'hit':>6s} {'skip long':>10s} {'yrs+':>7s}")
        for _, r in df.sort_values("t", ascending=False).head(8).iterrows():
            print(f"  {r['cell']:34s} {r['n']:4d} {r['per_week']:5.2f} {r['gross_bp']:+8.1f} "
                  f"{r['net_bp']:+8.1f} {r['t']:+6.2f} {r['hit']*100:5.1f}% "
                  f"{r['skipped_long_bp']:+10.1f} {r['years_pos']:3d}/{r['years']:<3d}")
        for rule in RULES:
            s = df[df.rule == rule]
            print(f"    {rule:11s} mean gross {s.gross_bp.mean():+8.2f}bp   "
                  f"positive {(s.gross_bp > 0).sum()}/{len(s)}")
        n = maxstat_signed(store, a.draws)
        (OUT / f"cot_null_{'lag' if lagged else 'peek'}.json").write_text(json.dumps(n, indent=2))
        print(f"  best signed t {n['observed_t']:+.3f} ({n['best_cell']})   "
              f"null p50 {n['p50']:+.3f} p95 {n['p95']:+.3f}   p {n['p_value']:.4f}  -> "
              f"{'SURVIVES' if n['p_value'] < 0.05 else 'does not survive'}")


if __name__ == "__main__":
    main()
