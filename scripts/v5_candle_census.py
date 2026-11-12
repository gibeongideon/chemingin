#!/usr/bin/env python3
"""High-frequency CANDLE-PATTERN CENSUS on H4 (user 2026-11-12: "search for same ideas pattern that
repeat at high frequency ... we focus on repeated candle patterns then profitability next").
PRE-REGISTERED.

ALPHABET. Every H4 candle is coded against the previous one, 32 codes:
    HH   high > previous high                     (2)
    LL   low  < previous low                      (2)
    ZONE close below prev low | lower half of prev range | upper half | above prev high   (4)
    BODY close > open                             (2)
Patterns = sequences of 1, 2, 3 codes ending at bar t (32 + 1,024 + 32,768 candidates). The
friend's sweep-and-reject is one of them (HH, LL, ZONE 0).

PANEL: every CME future in data/databento/ohlcv_1h (single provenance, back-adjusted, bars that
span a roll dropped). Switch XAUUSDs / BTCUSDs / USTECs are scored separately as the
execution check (never pooled with the CME copy of the same asset).

TARGET: return from OPEN[t+1] to CLOSE[t+h], h = 1, 3, 6 H4 bars, in units of ATR20[t], minus
that asset's mean over all bars of the same period (EXCESS over drift). Hit rate is printed
beside the asset's base rate, never beside 0.5.

FREQUENCY FILTER (pattern codes only, no returns): >= 20 fires per asset-year in discovery.

PERIODS: DISCOVERY 2010-06..2019-12 (the search), VALIDATION 2020-01..2023-12, HOLDOUT
2024-01.. LOCKED (printed only for survivors, once).

REPEATED = all of
  1 |z| in discovery above the 95th percentile of the MAX |z| over every candidate in a null
    where each asset's excess-return series is circularly shifted against its codes (200 draws)
  2 the same sign in validation, with validation z beyond +/-2
  3 the same sign in >= 70% of assets (discovery + validation)
z = sum(excess) / sqrt(sum(excess^2)) (self-normalised; the null carries the overlap/dependence).

    python scripts/v5_candle_census.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_sweep_reject_h4 import load_switch, to_h4  # noqa: E402

CME = ROOT / "data" / "databento" / "ohlcv_1h"
OUT = ROOT / "data" / "v5_runs" / "candle_census"
HS = (1, 3, 6)
DISC_END, VAL_END = pd.Timestamp("2020-01-01"), pd.Timestamp("2024-01-01")
MIN_PER_YR = 20
N_NULL = 200
RNG = np.random.default_rng(2026111202)
ZONES = ("<pL", "loH", "upH", ">pH")


def load_cme(root: str) -> pd.DataFrame:
    d = pd.read_csv(CME / f"{root}.csv.gz", parse_dates=["time"]).set_index("time")
    d = d[(d.close > 0) & (d.close_adj > 0)]
    k = d.close_adj / d.close
    h1 = pd.DataFrame({"open": d.open * k, "high": d.high * k, "low": d.low * k, "close": d.close_adj, "spr": 0.0})
    roll = d.instrument_id.ne(d.instrument_id.shift()).astype(float)
    roll.iloc[0] = 0.0
    h4 = to_h4(h1)
    has = roll.resample("4h", label="left", closed="left").max().reindex(h4.index).fillna(0) > 0
    return h4[~has]


def codes(d: pd.DataFrame) -> np.ndarray:
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    ph, pl = np.r_[np.nan, h[:-1]], np.r_[np.nan, l[:-1]]
    mid = (ph + pl) / 2
    zone = np.where(c < pl, 0, np.where(c < mid, 1, np.where(c <= ph, 2, 3)))
    cd = (h > ph) * 16 + (l < pl) * 8 + zone * 2 + (c > o)
    cd = cd.astype(np.int64)
    cd[0] = -1
    return cd


def seq(cd: np.ndarray, L: int) -> np.ndarray:
    out = cd.copy()
    for j in range(1, L):
        prev = np.r_[np.full(j, -1), cd[:-j]]
        out = np.where((out < 0) | (prev < 0), -1, prev * 32 ** j + out if j == 1 else prev * 32 ** j + out)
    return out


def decode(p: int, L: int) -> str:
    parts = []
    for j in range(L):
        c = (p // 32 ** (L - 1 - j)) % 32
        parts.append(f"{'HH' if c & 16 else '--'}{'LL' if c & 8 else '--'}{ZONES[(c >> 1) & 3]}{'+' if c & 1 else '-'}")
    return " | ".join(parts)


def fwd_excess(d: pd.DataFrame, h: int, periods: np.ndarray) -> np.ndarray:
    o, c = d.open.to_numpy(), d.close.to_numpy()
    pc = np.r_[np.nan, c[:-1]]
    tr = np.maximum.reduce([d.high.to_numpy() - d.low.to_numpy(), np.abs(d.high.to_numpy() - pc),
                            np.abs(d.low.to_numpy() - pc)])
    atr = pd.Series(tr).rolling(20).mean().to_numpy()
    n = len(d)
    f = np.full(n, np.nan)
    f[: n - h] = (c[h:] - o[1: n - h + 1]) / atr[: n - h]
    ex = f.copy()
    for p in np.unique(periods):
        m = (periods == p) & np.isfinite(f)
        ex[m] = f[m] - f[m].mean()
    return ex


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    roots = sorted(p.name.split(".")[0] for p in CME.glob("*.csv.gz"))
    panel = {r: load_cme(r) for r in roots}
    panel = {k: v for k, v in panel.items() if len(v) > 3000}
    exe = {"XAU-switch": load_switch("XAUUSDs", 0.30, None), "BTC-switch": load_switch("BTCUSDs", None, 4.5),
           "USTEC-switch": load_switch("USTECs", None, 1.0)}
    lines = [f"PANEL {len(panel)} CME roots: {' '.join(panel)}"]
    for k, d in exe.items():
        bpd = d.groupby(d.index.normalize()).size().median()
        assert 5 <= bpd <= 7, (k, bpd)
    lines.append("bars/day: " + " ".join(f"{k}:{d.groupby(d.index.normalize()).size().median():.0f}"
                                         for k, d in panel.items()))
    # per asset: period labels, codes per L, excess per h
    A = {}
    for k, d in {**panel, **exe}.items():
        per = np.where(d.index < DISC_END, 0, np.where(d.index < VAL_END, 1, 2))
        cd = codes(d)
        span = [d.index[per == p] for p in range(3)]
        A[k] = dict(per=per, yrs=np.array([(x[-1] - x[0]).days / 365.25 if len(x) > 1 else 0.0 for x in span]),
                    seq={L: seq(cd, L) for L in (1, 2, 3)},
                    ex={h: fwd_excess(d, h, per) for h in HS},
                    hit_base={h: None for h in HS})
    rows = []
    for L in (1, 2, 3):
        K = 32 ** L
        # frequency filter on discovery codes only
        cnt = np.zeros(K)
        ay = 0.0
        for k in panel:
            s, per = A[k]["seq"][L], A[k]["per"]
            m = (per == 0) & (s >= 0)
            cnt += np.bincount(s[m], minlength=K)
            ay += A[k]["yrs"][0]
        keep = np.flatnonzero(cnt / max(ay, 1e-9) >= MIN_PER_YR)
        if not len(keep):
            continue
        for h in HS:
            def zstat(per_sel: int, shift: dict | None = None, assets=panel):
                S = np.zeros(K)
                Q = np.zeros(K)
                N = np.zeros(K)
                for k in assets:
                    s, per, ex = A[k]["seq"][L], A[k]["per"], A[k]["ex"][h]
                    m = per == per_sel
                    s_, e_ = s[m], ex[m]
                    if shift is not None:
                        e_ = np.roll(e_, shift[k])
                    ok = (s_ >= 0) & np.isfinite(e_)
                    S += np.bincount(s_[ok], weights=e_[ok], minlength=K)
                    Q += np.bincount(s_[ok], weights=e_[ok] ** 2, minlength=K)
                    N += np.bincount(s_[ok], minlength=K)
                with np.errstate(invalid="ignore", divide="ignore"):
                    return S / np.sqrt(Q), S / N, N
            z0, m0, n0 = zstat(0)
            z1, m1, n1 = zstat(1)
            mx = []
            for _ in range(N_NULL):
                sh = {k: int(RNG.integers(100, max(101, (A[k]["per"] == 0).sum() - 100))) for k in panel}
                zn, _, _ = zstat(0, sh)
                mx.append(np.nanmax(np.abs(zn[keep])))
            thr = float(np.quantile(mx, 0.95))
            # per-asset sign agreement (disc + val)
            agree = np.zeros(K)
            nas = np.zeros(K)
            for k in panel:
                s, per, ex = A[k]["seq"][L], A[k]["per"], A[k]["ex"][h]
                m = (per <= 1) & (s >= 0) & np.isfinite(ex)
                sm = np.bincount(s[m], weights=ex[m], minlength=K)
                nn = np.bincount(s[m], minlength=K)
                has = nn >= 30
                agree += has * (np.sign(sm) == np.sign(z0))
                nas += has
            for p in keep:
                rows.append(dict(L=L, h=h, pattern=decode(p, L), code=int(p),
                                 per_yr=round(cnt[p] / ay, 1), n_disc=int(n0[p]),
                                 ex_disc_atr=round(m0[p], 4), z_disc=round(z0[p], 2),
                                 ex_val_atr=round(m1[p], 4), z_val=round(z1[p], 2),
                                 asset_agree=round(agree[p] / max(nas[p], 1), 2), null95=round(thr, 2)))
            print(f"L{L} h{h}: {len(keep)} patterns kept, null max|z| 95% = {thr:.2f}", flush=True)
    R = pd.DataFrame(rows)
    R["repeated"] = ((R.z_disc.abs() > R.null95) & (np.sign(R.z_val) == np.sign(R.z_disc))
                     & (R.z_val.abs() > 2) & (R.asset_agree >= 0.70))
    S = R[R.repeated].sort_values("z_disc", key=np.abs, ascending=False)
    # execution check on Switch (no pooling) + LOCKED holdout, survivors only
    ex_rows = []
    for r in S.itertuples():
        for k in exe:
            s, ex, per = A[k]["seq"][r.L], A[k]["ex"][r.h], A[k]["per"]
            m = (s == r.code) & np.isfinite(ex)
            yrs = A[k]["yrs"].sum()
            ex_rows.append(dict(pattern=r.pattern, h=r.h, mkt=k, per_yr=round(m.sum() / yrs, 1),
                                ex_atr=round(ex[m].mean(), 4) if m.any() else np.nan))
        s, per = None, None
    hold = []
    for r in S.itertuples():
        e = np.concatenate([A[k]["ex"][r.h][(A[k]["seq"][r.L] == r.code) & (A[k]["per"] == 2)] for k in panel])
        e = e[np.isfinite(e)]
        hold.append(dict(pattern=r.pattern, h=r.h, n=len(e), ex_atr=round(e.mean(), 4) if len(e) else np.nan,
                         z=round(e.sum() / np.sqrt((e ** 2).sum()), 2) if len(e) else np.nan,
                         same_sign=bool(np.sign(e.mean()) == np.sign(r.z_disc)) if len(e) else None))
    cost_atr = {}
    for k, d in exe.items():
        a = (d.high - d.low).rolling(20).mean()
        cost_atr[k] = float((d.spr / a)[d.index >= "2025-01-01"].median())
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 80)
    tot = R.groupby(["L", "h"]).size()
    lines += ["", "candidates kept by frequency filter (per L, h):", tot.to_string(),
              "", f"REPEATED: {len(S)} of {len(R)} (pattern, horizon) cells",
              S.drop(columns=["code"]).head(60).to_string(index=False) if len(S) else "(none)",
              "", "Top 25 by |z_disc| regardless of status:",
              R.reindex(R.z_disc.abs().sort_values(ascending=False).index).drop(columns=["code"]).head(25).to_string(index=False),
              "", "Execution check (Switch, not pooled), survivors only:",
              pd.DataFrame(ex_rows).to_string(index=False) if ex_rows else "(none)",
              "", "LOCKED HOLDOUT 2024-01.. (opened once), survivors only:",
              pd.DataFrame(hold).to_string(index=False) if hold else "(none)",
              "", "Cost yardstick (round trip in H4-ATR, median 2025+): " + " ".join(
                  f"{k} {cost_atr[k]:.3f}" for k in exe)]
    txt = "\n".join(lines)
    print(txt)
    (OUT / "report.txt").write_text(txt + "\n")
    R.to_csv(OUT / "cells.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())
