#!/usr/bin/env python3
"""TRIAL LAB — extensive trial-and-error, run so that whatever survives means something.

User, 2026-11-04: "do an extensive trial and error experiments until you find something that
makes sense". The repo's own law (trigger-search-max-stat-null) is that searching harder raises
the bar faster than it finds signal, so the search is wrapped in three guards:

  1. LOCKED HOLDOUT. Everything from HOLDOUT_START on is never read by the discovery pass.
     `--holdout` scores ONLY the finalists written by the discovery pass, once.
  2. EVERY TRIAL IS LOGGED (data/v5_runs/trial_lab/trials.csv), so the trial count K is known and
     the bar is set from it: discovery |t| must clear the max-|t| of K noise trials.
  3. SCORED AS ALPHA, NOT RETURN. Daily strategy P&L is regressed on the asset's own daily
     return; the intercept (Newey-West, 5 lags) is the edge. Being long a rising asset scores 0.

ENGINE. A position decided at the close of H1 bar t earns close(t)->close(t+1). Cost = max(quoted,
floor) half-spread per unit of position change; carry accrues per hour held at the broker's
annual long/short financing (Switch, approximations from memory: XAU L -6% S +2.5%, BTC L -15%
S -10%, USTEC L -6% S 0%). Quoted spread is ZERO in the feed before 2025, hence the floor.

    python scripts/v5_trial_lab.py              # discovery on data < HOLDOUT_START
    python scripts/v5_trial_lab.py --holdout    # finalists only, on data >= HOLDOUT_START
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sweep_switch"
OUT = ROOT / "data" / "v5_runs" / "trial_lab"
HOLDOUT_START = pd.Timestamp("2024-07-01")
ASSETS = ["XAUUSDs", "BTCUSDs", "USTECs"]
FLOOR_BP = {"XAUUSDs": 0.55, "BTCUSDs": 4.5, "USTECs": 0.7}
CARRY = {"XAUUSDs": (-0.06, 0.025), "BTCUSDs": (-0.15, -0.10), "USTECs": (-0.06, 0.0)}
SHORT = {"XAU": "XAUUSDs", "BTC": "BTCUSDs", "NDX": "USTECs"}


# --------------------------------------------------------------------------- data ------
def load_all(holdout: bool) -> dict[str, pd.DataFrame]:
    meta = json.loads((DATA / "meta.json").read_text())
    out = {}
    for s in ASSETS:
        d = pd.read_csv(DATA / f"{s}_H1.csv", parse_dates=["time"]).set_index("time")
        d = d[~d.index.duplicated()].sort_index()
        d = d[d.index >= HOLDOUT_START] if holdout else d[d.index < HOLDOUT_START]
        q = d["spread"] * meta[s]["point"] / d["close"]
        d["cost"] = np.maximum(q, FLOOR_BP[s] / 1e4)            # full spread, as a fraction
        pc = d["close"].shift(1)
        tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(1)
        d["atr"] = tr.rolling(20).mean()
        d["r1"] = d["close"].pct_change().shift(-1)            # return earned by pos[t]
        d["dt_h"] = (d.index.to_series().shift(-1) - d.index.to_series()).dt.total_seconds() / 3600
        out[s] = d
    return out


# ------------------------------------------------------------------------- engine ------
def hold_latest(ev: pd.Series, H: int) -> pd.Series:
    """Events (+1/-1 at decision bars, NaN elsewhere) -> position held H bars from each event."""
    return (ev.ffill(limit=H - 1) if H > 1 else ev).fillna(0.0)


def nw_t(y: np.ndarray, X: np.ndarray, lags: int = 5) -> tuple[np.ndarray, np.ndarray]:
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return b, b / np.sqrt(np.diag(V))


def score(d: pd.DataFrame, sym: str, pos: pd.Series) -> dict:
    pos = pos.reindex(d.index).fillna(0.0).clip(-1, 1)
    cl, cs = CARRY[sym]
    carry = np.where(pos > 0, pos * cl, -pos * cs) * d["dt_h"].fillna(1) / 8760
    turn = pos.diff().abs().fillna(pos.abs())
    net = pos * d["r1"] - turn * d["cost"] / 2 + carry
    net = net.fillna(0.0)
    day = d.index.normalize()
    nd = net.groupby(day).sum()
    hold = d["r1"].fillna(0) + cl * d["dt_h"].fillna(1) / 8760     # what HOLDING earns, net
    ad = hold.groupby(day).sum()
    act = nd.index[(pos.groupby(day).apply(lambda x: x.abs().sum()) > 0)]
    if len(nd) < 60 or pos.abs().sum() == 0:
        return dict(n_days=len(nd), alpha_ann=np.nan, t=np.nan, sharpe=np.nan)
    X = np.column_stack([np.ones(len(nd)), ad.to_numpy()])
    b, t = nw_t(nd.to_numpy(), X)
    yrs = (nd.index[-1] - nd.index[0]).days / 365.25
    tdy = len(nd) / max(yrs, 0.1)                                  # trading days per year
    half = len(nd) // 2
    a1 = nd.iloc[:half].mean() - b[1] * ad.iloc[:half].mean()
    a2 = nd.iloc[half:].mean() - b[1] * ad.iloc[half:].mean()
    by = (nd - b[1] * ad).groupby(nd.index.year).sum()
    return dict(alpha_ann=round(b[0] * tdy, 4), t=round(t[0], 2), beta=round(b[1], 3),
                sharpe=round(nd.mean() / nd.std() * np.sqrt(tdy), 2) if nd.std() > 0 else np.nan,
                exposure=round(pos.abs().mean(), 3),
                turns_yr=round(turn.sum() / max(yrs, 0.1), 1),
                h1=round(a1 * tdy, 4), h2=round(a2 * tdy, 4),
                yrs_pos=f"{int((by > 0).sum())}/{len(by)}", n_days=len(nd),
                active_days=len(act))


# ----------------------------------------------------------------------- families ------
def fam_ghost(d):
    """F1 ghost-limit refusal at an intraday scale (fill rates near 50%), and its momentum twin."""
    for k, N, H in itertools.product((0.25, 0.5, 1.0), (2, 4, 8), (4, 12, 24)):
        ghost = (d.close - k * d.atr).shift(N)
        refused = d.low.rolling(N).min() > ghost
        fill = 1 - refused.mean()
        yield (f"ghost k{k} N{N} H{H}", hold_latest(refused.astype(float).where(refused), H),
               dict(fill=round(fill, 3)))
        # TWIN: same window, momentum only, matched frequency (top quantile of window return)
        wret = d.close / d.close.shift(N) - 1
        thr = wret.quantile(fill)
        mom = wret > thr
        yield (f"ghostTWIN k{k} N{N} H{H}", hold_latest(mom.astype(float).where(mom), H), {})


def fam_compress(d):
    """F4 volatility compression, then follow the box breakout (long-only or both)."""
    for L, q, H, mode in itertools.product((6, 12, 24), (0.1, 0.2), (8, 24), ("LO", "LS")):
        hi, lo = d.high.rolling(L).max().shift(1), d.low.rolling(L).min().shift(1)
        width = (hi - lo) / d.atr
        rank = width.rolling(500, min_periods=200).rank(pct=True)
        comp = rank.shift(1) <= q
        up = comp & (d.close > hi)
        dn = comp & (d.close < lo)
        ev = pd.Series(np.nan, index=d.index)
        ev[up] = 1.0
        if mode == "LS":
            ev[dn] = -1.0
        yield f"compress L{L} q{q} H{H} {mode}", hold_latest(ev, H), {}


def fam_lag24(d):
    """F7 same-hour seasonality: yesterday's (or last D days') return at this hour."""
    for D, mode in itertools.product((1, 5, 20), ("LO", "LS")):
        same = sum(d.r1.shift(24 * i) for i in range(1, D + 1)) / D   # r(t+1-24i)
        sig = np.sign(same)
        if mode == "LO":
            sig = sig.clip(lower=0)
        yield f"lag24 D{D} {mode}", sig.fillna(0.0), {}


def fam_spread(d):
    """F8 liquidity provision: after a quoted-spread spike (2025+ only has quotes)."""
    sp = d["spread"].replace(0, np.nan)
    rel = sp / sp.rolling(500, min_periods=100).median()
    for m, H, side in itertools.product((2.0, 3.0, 5.0), (4, 24), (1, -1)):
        ev = pd.Series(np.nan, index=d.index)
        ev[rel >= m] = float(side)
        yield f"spreadspike m{m} H{H} side{side:+d}", hold_latest(ev, H), {}


def fam_gap(d):
    """F5 the first bar after a >=2h gap (session break / weekend): fade or follow."""
    gap_h = d.index.to_series().diff().dt.total_seconds() / 3600
    first = gap_h >= 2
    g = (d.open - d.close.shift(1)) / d.atr
    for thr, H, mode in itertools.product((0.25, 0.5, 1.0), (1, 4, 8), ("fade", "follow")):
        ev = pd.Series(np.nan, index=d.index)
        s = np.sign(g) * (1 if mode == "follow" else -1)
        ev[first & (g.abs() >= thr)] = s[first & (g.abs() >= thr)]
        yield f"gap thr{thr} H{H} {mode}", hold_latest(ev, H), {}


def fam_tom(d):
    """F13 turn of month: long the last K1 and first K2 trading days."""
    day = d.index.normalize()
    days = pd.Series(sorted(set(day)))
    mon = days.dt.to_period("M")
    rank_f = days.groupby(mon).cumcount()
    rank_b = days.groupby(mon).cumcount(ascending=False)
    for K1, K2 in ((1, 3), (2, 3), (1, 1), (0, 4)):
        on = set(days[(rank_b < K1) | (rank_f < K2)])
        yield f"tom last{K1} first{K2}", pd.Series(day.isin(on).astype(float), index=d.index), {}


def fam_dip(d):
    """F14 long-only only while below the N-day high by > x ATR (D1-equivalent via 24 bars)."""
    for N, x, mode in itertools.product((5, 20, 60), (1.0, 2.0, 4.0), ("below", "above")):
        hh = d.high.rolling(24 * N).max()
        dd = (hh - d.close) / d.atr
        on = dd > x if mode == "below" else dd <= x
        yield f"dip N{N}d x{x} {mode}", on.astype(float), {}


def fam_hour(d):
    """F15 hour-of-day exposure learned on a TRAILING window: long the hours whose trailing
    mean return is in the top third, flat otherwise (walk-forward, no full-sample fit)."""
    hr = d.index.hour
    for W, mode in itertools.product((60, 250), ("LO", "LS")):
        m = pd.Series(d.r1.to_numpy(), index=d.index).groupby(hr).transform(
            lambda s: s.shift(1).rolling(W, min_periods=W // 2).mean())
        rk = m.groupby(d.index.normalize()).rank(pct=True)
        sig = (rk >= 2 / 3).astype(float)
        if mode == "LS":
            sig = sig - (rk <= 1 / 3).astype(float)
        yield f"hourWF W{W}d {mode}", sig, {}


def fam_cross(D):
    """F2 lead-lag across assets: sign of the leader's last L hours sets the target for H hours.
    Leader bars are asof-joined (a leader bar stamped t is complete only at t+1h -> shift 1)."""
    for (ln, tn), L, H, mode in itertools.product(
            itertools.permutations(SHORT, 2), (1, 4, 24), (1, 4, 24), ("LO", "LS")):
        lead, tgt = D[SHORT[ln]], D[SHORT[tn]]
        lr = (lead.close / lead.close.shift(L) - 1).shift(1)           # completed bars only
        lr = lr.reindex(tgt.index, method="ffill")
        sig = np.sign(lr)
        if mode == "LO":
            sig = sig.clip(lower=0)
        ev = sig.where(np.arange(len(sig)) % H == 0)
        yield SHORT[tn], f"lead {ln}->{tn} L{L} H{H} {mode}", hold_latest(ev, H), {}


def fam_weekend(D):
    """F2w BTC trades the weekend; gold and NDX do not. At each target's first bar of the week,
    trade the sign of BTC's move since the target's last pre-weekend close."""
    btc = D["BTCUSDs"]
    for tn, H, mode in itertools.product(("XAU", "NDX"), (4, 24, 96), ("LO", "LS")):
        tgt = D[SHORT[tn]]
        gap_h = tgt.index.to_series().diff().dt.total_seconds() / 3600
        first = gap_h >= 40
        ev = pd.Series(np.nan, index=tgt.index)
        for t in tgt.index[first]:
            prev = tgt.index[tgt.index.get_loc(t) - 1]
            b0 = btc.close.asof(prev)
            b1 = btc.close.asof(t - pd.Timedelta(hours=1))
            if np.isfinite(b0) and np.isfinite(b1):
                s = np.sign(b1 / b0 - 1)
                ev[t] = max(s, 0) if mode == "LO" else s
        yield SHORT[tn], f"weekendBTC->{tn} H{H} {mode}", hold_latest(ev, H), {}


# ------------------------------------------------------------ round 2: DAILY scale -----
def _daily(d):
    """Server-day bars + the H1 index of each day's LAST bar (the decision point)."""
    day = d.index.normalize()
    g = d.groupby(day)
    D1 = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(),
                       "close": g.close.last()})
    pc = D1.close.shift(1)
    tr = pd.concat([D1.high - D1.low, (D1.high - pc).abs(), (D1.low - pc).abs()], axis=1).max(1)
    D1["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    last = pd.Series(d.index, index=day).groupby(level=0).max()
    D1["last"] = last.reindex(D1.index).to_numpy()
    bpd = int(g.size().median())
    return D1.iloc[20:], bpd


def _ev_daily(d, D1, cond_long, cond_short=None):
    ev = pd.Series(np.nan, index=d.index)
    ev.loc[D1["last"][cond_long.fillna(False)].to_numpy()] = 1.0
    if cond_short is not None:
        ev.loc[D1["last"][cond_short.fillna(False)].to_numpy()] = -1.0
    return ev


def fam_daily(d):
    D1, bpd = _daily(d)
    r = (D1.close - D1.close.shift(1)) / D1.atr.shift(1)
    rng_ = (D1.high - D1.low).replace(0, np.nan)
    clv = (D1.close - D1.low) / rng_
    for x, H, mode in itertools.product((0.5, 1.0, 2.0), (1, 3, 5), ("LO", "LS")):
        ev = _ev_daily(d, D1, r >= x, (r <= -x) if mode == "LS" else None)
        yield f"dmove x{x} H{H}d {mode}", hold_latest(ev, H * bpd), {}
    for q, H, mode in itertools.product((0.8, 0.9), (1, 3, 5), ("LO", "LS")):
        ev = _ev_daily(d, D1, clv >= q, (clv <= 1 - q) if mode == "LS" else None)
        yield f"clv q{q} H{H}d {mode}", hold_latest(ev, H * bpd), {}
    for k, H in itertools.product((2, 3, 4), (1, 3, 5)):
        hl = (D1.low > D1.low.shift(1)).astype(int)
        run_ = hl.groupby((hl == 0).cumsum()).cumsum()
        yield f"higherlows k{k} H{H}d", hold_latest(_ev_daily(d, D1, run_ >= k), H * bpd), {}
        dn = (D1.close < D1.close.shift(1)).astype(int)
        rd = dn.groupby((dn == 0).cumsum()).cumsum()
        yield f"downrun k{k} H{H}d", hold_latest(_ev_daily(d, D1, rd >= k), H * bpd), {}
    nr7 = (D1.high - D1.low) <= (D1.high - D1.low).rolling(7).min()
    inside = (D1.high <= D1.high.shift(1)) & (D1.low >= D1.low.shift(1))
    for nm, cond in (("nr7", nr7), ("inside", inside)):
        # next day: follow the FIRST H1 close beyond that day's high/low
        for H, mode in itertools.product((1, 3), ("LO", "LS")):
            ev = pd.Series(np.nan, index=d.index)
            for t, row in D1[cond.fillna(False)].iterrows():
                nxt = d.index[(d.index > row["last"]) & (d.index <= row["last"] + pd.Timedelta(hours=30))]
                c = d.close.reindex(nxt)
                up, dn_ = c[c > row.high], c[c < row.low]
                fu = up.index[0] if len(up) else None
                fd = dn_.index[0] if len(dn_) else None
                if fu is not None and (fd is None or fu < fd):
                    ev[fu] = 1.0
                elif fd is not None and mode == "LS":
                    ev[fd] = -1.0
            yield f"{nm}break H{H}d {mode}", hold_latest(ev, H * bpd), {}


SINGLE = [fam_ghost, fam_compress, fam_lag24, fam_spread, fam_gap, fam_tom, fam_dip, fam_hour,
          fam_daily]


def run(D: dict, only: set | None = None) -> pd.DataFrame:
    rows = []

    def add(sym, name, pos, extra):
        if only is not None and (sym, name) not in only:
            return
        rows.append(dict(asset=sym, trial=name, family=name.split()[0], **extra,
                         **score(D[sym], sym, pos)))

    for sym in ASSETS:
        d = D[sym]
        for f in SINGLE:
            for name, pos, extra in f(d):
                add(sym, name, pos, extra)
        add(sym, "buyhold", pd.Series(1.0, index=d.index), {})
        print(f"  {sym}: {len(rows)} trials so far", flush=True)
    for f in (fam_cross, fam_weekend):
        for sym, name, pos, extra in f(D):
            add(sym, name, pos, extra)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 230, "display.max_columns", 30, "display.max_rows", 400)
    if not a.holdout:
        D = load_all(holdout=False)
        res = run(D)
        res.to_csv(OUT / "trials.csv", index=False)
        K = int(res.t.notna().sum())
        # max |t| of K independent N(0,1) draws, p95 — the bar a lone winner must clear
        rng = np.random.default_rng(0)
        bar = float(np.percentile(np.abs(rng.standard_normal((2000, K))).max(1), 95))
        print(f"\nK = {K} trials. Noise bar (p95 of max|t| over K): {bar:.2f}")
        print(res.sort_values("t", ascending=False).head(30).to_string(index=False))
        print("\nBOTTOM 10 (an edge that is reliably NEGATIVE is also information):")
        print(res.sort_values("t").head(10).to_string(index=False))
        # FINALISTS: clear the bar, OR the same trial positive at t>=2 on >=2 assets
        fin = res[res.t >= bar]
        multi = res[res.t >= 2.0].groupby("trial").filter(lambda g: g.asset.nunique() >= 2)
        fin = pd.concat([fin, multi]).drop_duplicates(["asset", "trial"])
        fin[["asset", "trial"]].to_csv(OUT / "finalists.csv", index=False)
        fam = res.groupby(["family", "asset"]).t.agg(["count", "mean", "max"]).round(2)
        print("\nFAMILY x ASSET mean/max t:\n" + fam.to_string())
        print(f"\n{len(fin)} finalist(s) written to finalists.csv (holdout NOT read)")
        print(fin.to_string(index=False) if len(fin) else "  none")
    else:
        fin = pd.read_csv(OUT / "finalists.csv")
        only = set(map(tuple, fin[["asset", "trial"]].to_numpy()))
        D = load_all(holdout=True)
        res = run(D, only=only)
        res.to_csv(OUT / "holdout.csv", index=False)
        print(res.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
