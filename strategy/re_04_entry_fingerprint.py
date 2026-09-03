"""STAGE 4 — fingerprint the ENTRY rule against matched control moments.

Stage 3 settled the decomposition: the entries carry a weak, tail-driven, short-horizon
edge (+1.66 mean move at 15m, t=+2.33, gone by 60m; hit rate only 48.5%), while the
headline performance comes from a sub-second exit that banks 27% of the favourable
excursion and absorbs 2% of the adverse one. The exit is not reproducible from bars. The
ENTRY is the part that can be reverse-engineered, so this stage does that properly.

METHOD. For every real entry, compute features using ONLY bars strictly before it, plus
where inside its own bar the fill sat. Compare against CONTROL moments drawn from the same
hours-of-day and same days (so time-of-day, which is clearly non-uniform in the log, cannot
masquerade as signal). Anything that separates entries from controls is part of the rule;
anything that does not, is not — regardless of how plausible it sounds.

Direction is fingerprinted separately from timing: WHEN it trades and WHICH WAY it trades
are different questions and may have different answers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OFFSET_MIN = -120


def feats(px: pd.DataFrame) -> pd.DataFrame:
    """Causal features on M15 bars: value at bar i uses bars <= i."""
    c, h, l, o = px.close, px.high, px.low, px.open
    f = pd.DataFrame(index=px.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    f["atr"] = atr
    for n in (4, 8, 16, 48):
        f[f"mom{n}"] = (c - c.shift(n)) / atr
    for n in (20, 96):
        m, s = c.rolling(n).mean(), c.rolling(n).std()
        f[f"z{n}"] = (c - m) / s
        hi, lo = h.rolling(n).max(), l.rolling(n).min()
        f[f"rng_pos{n}"] = (c - lo) / (hi - lo).replace(0, np.nan)
    dl = c.diff()
    up = dl.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-dl.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    f["bar_range_atr"] = (h - l) / atr
    f["body_frac"] = (c - o).abs() / (h - l).replace(0, np.nan)
    f["atr_pctl"] = atr.rolling(2000, min_periods=200).apply(lambda x: (x[-1] > x).mean(), raw=True)
    f["ret1_atr"] = (c - c.shift(1)) / atr
    f["gap_atr"] = (o - c.shift(1)) / atr
    return f


def main() -> None:
    d = pd.read_csv(HERE / "re_trades_aligned.csv", parse_dates=["open_date", "close_date", "t"])
    px = pd.read_csv(ROOT / "data" / "XAUUSD_M15_long.csv",
                     parse_dates=["time"], index_col="time").sort_index()
    px = px[~px.index.duplicated(keep="last")]
    F = feats(px)
    idx = px.index

    # entry rows: features from the bar BEFORE the entry bar (strictly pre-entry info)
    pos = idx.searchsorted(d.t.values, side="right") - 1
    d["prev"] = pos - 1
    keep = (d.prev > 250) & (d.prev < len(idx) - 10)
    d = d[keep].reset_index(drop=True)
    pos, prev = pos[keep.values], d.prev.values

    # intra-bar fill position (where in its own bar did the fill land?)
    lo_b, hi_b = px.low.values[pos], px.high.values[pos]
    d["fill_in_bar"] = (d.open_price.values - lo_b) / np.where(hi_b > lo_b, hi_b - lo_b, np.nan)

    E = F.iloc[prev].reset_index(drop=True)

    # ---------------- CONTROLS matched on hour-of-day AND date
    rng = np.random.default_rng(7)
    hours = pd.Series(idx.hour, index=range(len(idx)))
    dates = pd.Series(idx.normalize(), index=range(len(idx)))
    # dtype matters here: entry_days must be datetime64 to match dates.values,
    # otherwise np.isin silently matches nothing and every control pool comes back empty
    entry_days = pd.to_datetime(d.t).dt.normalize().unique().astype("datetime64[ns]")
    ctrl_pool_by_hour: dict[int, np.ndarray] = {}
    for hr in sorted(set(pd.to_datetime(d.t).dt.hour)):
        mask = (hours.values == hr) & np.isin(dates.values.astype("datetime64[ns]"), entry_days)
        p = np.where(mask)[0]
        ctrl_pool_by_hour[hr] = p[(p > 250) & (p < len(idx) - 10)]
    ctrl_idx = []
    for hr in pd.to_datetime(d.t).dt.hour:
        pool = ctrl_pool_by_hour.get(hr, np.array([]))
        if len(pool):
            ctrl_idx.append(rng.choice(pool))
    ctrl_idx = np.array(ctrl_idx)
    C = F.iloc[ctrl_idx - 1].reset_index(drop=True)
    print(f"entries {len(E)}   hour+date-matched controls {len(C)}\n")

    # ---------------- WHEN does it trade? (entry vs control)
    print("=== WHEN IT TRADES: entry bars vs matched control bars ===")
    print(f"{'feature':16} {'entry mean':>11} {'ctrl mean':>10} {'t-stat':>8} {'verdict':>14}")
    cols = [c for c in F.columns if c != "atr"]
    sep = []
    for col in cols:
        a, b = E[col].dropna(), C[col].dropna()
        if len(a) < 50 or len(b) < 50:
            continue
        t = (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))
        sep.append((col, a.mean(), b.mean(), t))
    for col, am, bm, t in sorted(sep, key=lambda x: -abs(x[3])):
        v = "SEPARATES" if abs(t) > 3 else ("weak" if abs(t) > 2 else "no")
        print(f"{col:16} {am:>11.3f} {bm:>10.3f} {t:>+8.2f} {v:>14}")

    # ---------------- WHICH WAY? (buy vs sell)
    print("\n=== WHICH DIRECTION IT PICKS: buy entries vs sell entries ===")
    isbuy = d.sgn.values > 0
    print(f"{'feature':16} {'buy mean':>10} {'sell mean':>10} {'t-stat':>8} {'verdict':>14}")
    dsep = []
    for col in cols:
        a, b = E[col][isbuy].dropna(), E[col][~isbuy].dropna()
        if len(a) < 50 or len(b) < 50:
            continue
        t = (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))
        dsep.append((col, a.mean(), b.mean(), t))
    for col, am, bm, t in sorted(dsep, key=lambda x: -abs(x[3])):
        v = "SEPARATES" if abs(t) > 3 else ("weak" if abs(t) > 2 else "no")
        print(f"{col:16} {am:>10.3f} {bm:>10.3f} {t:>+8.2f} {v:>14}")

    print(f"\nintra-bar fill position (0=bar low, 1=bar high): "
          f"buys {np.nanmedian(d.fill_in_bar[isbuy]):.2f}   "
          f"sells {np.nanmedian(d.fill_in_bar[~isbuy]):.2f}")
    print("  (buys near the LOW / sells near the HIGH = fading the intrabar extreme;")
    print("   the reverse = breakout entry)")

    # ---------------- does the entry predict direction at all, per side?
    cc = px.close.values
    fwd15 = (cc[np.clip(pos + 1, 0, len(cc) - 1)] - d.open_price.values) * d.sgn.values
    print(f"\nforward 15m move in trade direction: buys {fwd15[isbuy].mean():+.3f} "
          f"(hit {float((fwd15[isbuy]>0).mean()*100):.1f}%)   "
          f"sells {fwd15[~isbuy].mean():+.3f} (hit {float((fwd15[~isbuy]>0).mean()*100):.1f}%)")

    out = pd.concat([d[["t", "sgn", "lots", "profit", "move", "fill_in_bar"]], E], axis=1)
    out.to_csv(HERE / "re_entry_features.csv", index=False)
    print(f"\n-> {HERE/'re_entry_features.csv'}")


if __name__ == "__main__":
    main()
