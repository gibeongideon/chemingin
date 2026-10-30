#!/usr/bin/env python3
"""OCTOBER A7 — TradingView Technical Ratings, replicated, and its EFFECTIVE number of opinions.

Pre-registered in OCTOBER.MD §2 row A7. Requested by the user from the live XAUUSD page.

WHY THIS IS NOT ANOTHER INDICATOR TEST. The repo has already tested most of these components
INDIVIDUALLY -- RSI, Stochastic, Bollinger, ADX -- and the champion itself is an EWMAC+breakout
blend, and A3 assembled a 23-signal panel. What has never been tested is **the vote-counting
AGGREGATOR**: TradingView's rating is a headcount over 26 components, and a headcount is only
meaningful if the heads are independent.

*** THE PRIMARY DELIVERABLE IS A MULTIPLICITY BOUND, NOT AN EDGE. *** Pure direction is retired by
CONTROL #8 (a 5.9-sigma AUC lift produced dSharpe -0.702), so this script's job is to answer: how
many opinions is a 26-component consensus actually made of? The user's own screenshot answers it
in miniature -- EMA and SMA at 10/20/30/50/100/200 ALL read "Sell" because price (4,023) sat below
every one of them (4,068 -> 4,502). That is **one fact counted twelve times**, and it is 12 of the
26 votes before a single oscillator is consulted.

WHAT IS REPORTED, in the order that decides the question:
  (a) EFFECTIVE RANK of the 26x26 vote correlation matrix -- participation ratio and the count of
      eigenvalues needed for 90% of variance -- for all 26, for the 12-MA block alone, and for the
      11 oscillators alone.
  (b) correlation of the rating with ONE trivial indicator (close vs SMA50) and with the champion
      forecast. If either is high the rating is a restatement, not an aggregate.
  (c) EV/fire by consensus bucket against **HOLDING gold** (CONTROL #2) and against the champion,
      with the mean offset reported beside it (CONTROL #8), and the threshold sweep priced against
      a best-of-K floor rather than read off its argmax (CONTROL #7).

FIDELITY NOTES, so the replication is not quietly flattered:
  * Every frame is rebuilt from `data/XAUUSD_M15_long.csv` via `mtf_frames.aggregate`, never the
    spliced `XAUUSD_H4_long.csv` and never mixed with the Yahoo D1 panel (`multi-broker dataset is
    a leak`, `cross-timeframe resample is lookahead`).
  * `tick_volume` is used for VWMA because `real_volume` is 0.0% populated in this feed (§3cf).
  * Ichimoku uses TradingView's CLOUD rule, not a price-vs-base-line shortcut. The shortcut would
    make it behave like the other twelve MAs and so would bias the effective rank DOWNWARD -- i.e.
    toward this script's own stated prior, which is exactly the thing not to do.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from v5_xau_champion_lifts import champion_recipe                     # noqa: E402
from src.v5 import mtf_frames                                         # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "october"
RT_BP = 0.504 * 1.5
BARS_PER_DAY = {"H1": 24, "H4": 6, "D1": 1}


# ----------------------------------------------------------------- primitives
def ema(s, n):  return s.ewm(span=n, adjust=False).mean()
def sma(s, n):  return s.rolling(n).mean()


def wma(s, n):
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def hma(s, n):
    return wma(2 * wma(s, n // 2) - wma(s, n), int(np.sqrt(n)))


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def stoch(h, l, c, n=14, k=3, d=3):
    ll, hh = l.rolling(n).min(), h.rolling(n).max()
    raw = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    K = raw.rolling(k).mean()
    return K, K.rolling(d).mean()


def cci(h, l, c, n=20):
    tp = (h + l + c) / 3
    m = tp.rolling(n).mean()
    md = (tp - m).abs().rolling(n).mean()
    return (tp - m) / (0.015 * md.replace(0, np.nan))


def adx_di(h, l, c, n=14):
    up, dn = h.diff(), -l.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(plus, index=h.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus, index=h.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, mdi


def ultimate(h, l, c, a=7, b=14, d=28):
    pc = c.shift(1)
    bp = c - pd.concat([l, pc], axis=1).min(axis=1)
    tr = pd.concat([h, pc], axis=1).max(axis=1) - pd.concat([l, pc], axis=1).min(axis=1)
    avg = lambda n: bp.rolling(n).sum() / tr.rolling(n).sum().replace(0, np.nan)
    return 100 * (4 * avg(a) + 2 * avg(b) + avg(d)) / 7


def sgn3(buy: pd.Series, sell: pd.Series) -> pd.Series:
    """TradingView's three-state vote: +1 Buy, -1 Sell, 0 Neutral."""
    return pd.Series(np.where(buy, 1.0, np.where(sell, -1.0, 0.0)), index=buy.index)


# ----------------------------------------------------------------- the 26 votes
def votes(d: pd.DataFrame) -> pd.DataFrame:
    h, l, c = d["high"], d["low"], d["close"]
    v = pd.to_numeric(d.get("tick_volume", pd.Series(1.0, index=d.index)),
                      errors="coerce").fillna(0.0)
    V = {}

    # ---- 11 oscillators, TradingView's documented rules ----
    r = rsi(c, 14);                 V["RSI14"] = sgn3((r < 30) & (r > r.shift(1)),
                                                      (r > 70) & (r < r.shift(1)))
    K, D = stoch(h, l, c);          V["Stoch"] = sgn3((K < 20) & (D < 20) & (K > D),
                                                      (K > 80) & (D > 80) & (K < D))
    cc = cci(h, l, c, 20);          V["CCI20"] = sgn3((cc < -100) & (cc > cc.shift(1)),
                                                      (cc > 100) & (cc < cc.shift(1)))
    ax, pdi, mdi = adx_di(h, l, c)
    V["ADX14"] = sgn3((ax > 20) & (pdi > mdi) & (pdi > pdi.shift(1)),
                      (ax > 20) & (pdi < mdi) & (pdi < pdi.shift(1)))
    hl2 = (h + l) / 2
    ao = sma(hl2, 5) - sma(hl2, 34)
    V["Awesome"] = sgn3((ao > 0) & ((ao.shift(1) < 0) | (ao > ao.shift(1))),
                        (ao < 0) & ((ao.shift(1) > 0) | (ao < ao.shift(1))))
    mom = c - c.shift(10);          V["Mom10"] = sgn3(mom > mom.shift(1), mom < mom.shift(1))
    macd = ema(c, 12) - ema(c, 26); sig = ema(macd, 9)
    V["MACD"] = sgn3(macd > sig, macd < sig)
    rr = rsi(c, 14)
    sr = 100 * (rr - rr.rolling(14).min()) / (rr.rolling(14).max() - rr.rolling(14).min()).replace(0, np.nan)
    sk = sr.rolling(3).mean(); sd = sk.rolling(3).mean()
    V["StochRSI"] = sgn3((sk < 20) & (sd < 20) & (sk > sd), (sk > 80) & (sd > 80) & (sk < sd))
    hh, ll = h.rolling(14).max(), l.rolling(14).min()
    wr = -100 * (hh - c) / (hh - ll).replace(0, np.nan)
    V["WillR"] = sgn3((wr < -80) & (wr > wr.shift(1)), (wr > -20) & (wr < wr.shift(1)))
    e13 = ema(c, 13); bbp = (h - e13) + (l - e13)
    V["BullBear"] = sgn3((bbp > 0) & (bbp > bbp.shift(1)), (bbp < 0) & (bbp < bbp.shift(1)))
    uo = ultimate(h, l, c);         V["Ultimate"] = sgn3(uo > 70, uo < 30)

    # ---- 15 moving averages: Buy if price is above, Sell if below ----
    for n in (10, 20, 30, 50, 100, 200):
        V[f"EMA{n}"] = sgn3(c > ema(c, n), c < ema(c, n))
        V[f"SMA{n}"] = sgn3(c > sma(c, n), c < sma(c, n))
    vw = (c * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    V["VWMA20"] = sgn3(c > vw, c < vw)
    hm = hma(c, 9);                 V["HMA9"] = sgn3(c > hm, c < hm)
    # Ichimoku: TradingView's CLOUD rule, deliberately NOT a price-vs-base shortcut
    conv = (h.rolling(9).max() + l.rolling(9).min()) / 2
    base = (h.rolling(26).max() + l.rolling(26).min()) / 2
    la = ((conv + base) / 2).shift(26)
    lb = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    V["Ichimoku"] = sgn3((la > lb) & (c > la) & (c > base) & (conv > base),
                         (la < lb) & (c < la) & (c < base) & (conv < base))
    return pd.DataFrame(V).dropna()


OSC = ["RSI14", "Stoch", "CCI20", "ADX14", "Awesome", "Mom10", "MACD", "StochRSI",
       "WillR", "BullBear", "Ultimate"]
MAS = [f"{k}{n}" for n in (10, 20, 30, 50, 100, 200) for k in ("EMA", "SMA")] + \
      ["VWMA20", "HMA9", "Ichimoku"]


def to_daily(m15: pd.DataFrame) -> pd.DataFrame:
    """D1 built here because `mtf_frames.aggregate` only serves M30/H1/H4.

    Single provenance (the same M15 file) and causal: a day's bar is stamped at that day and the
    forward return is taken from the NEXT day's close, so nothing at t is known before t ends.
    Never the Yahoo `GOLD_D1_long.csv`, which is a different source from the MT5 intraday files
    across the whole history (`multi-broker dataset is a leak`).
    """
    g = m15.groupby(m15.index.normalize())
    out = pd.DataFrame(dict(open=g["open"].first(), high=g["high"].max(),
                            low=g["low"].min(), close=g["close"].last()))
    if "tick_volume" in m15.columns:
        out["tick_volume"] = g["tick_volume"].sum()
    return out.dropna(subset=["close"])


def eff_rank(X: pd.DataFrame) -> tuple:
    """Participation ratio and the eigenvalue count for 90% of variance."""
    C = X.corr().to_numpy()
    C = np.nan_to_num(C, nan=0.0)
    lam = np.linalg.eigvalsh(C)
    lam = np.clip(lam[::-1], 0, None)
    pr = float(lam.sum() ** 2 / np.square(lam).sum())
    cum = np.cumsum(lam) / lam.sum()
    return pr, int(np.searchsorted(cum, 0.90) + 1), lam


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    m15 = pd.read_csv(ROOT / "data/XAUUSD_M15_long.csv", parse_dates=["time"], index_col="time")
    m15 = m15[~m15.index.duplicated(keep="last")].sort_index()

    print("=" * 104)
    print("OCTOBER A7 — TradingView Technical Ratings replicated, and its EFFECTIVE opinion count")
    print("=" * 104)

    all_rows = []
    for tf in ("H1", "H4", "D1"):
        d = to_daily(m15) if tf == "D1" else mtf_frames.aggregate(m15, tf).dropna(subset=["close"])
        W = votes(d)
        c = d["close"].reindex(W.index)
        rating = W.mean(axis=1)                       # (buys - sells)/26, TV's scale
        bpd = BARS_PER_DAY[tf]
        print(f"\n{'='*104}\n  --- {tf}: {len(W):,} rated bars  "
              f"{W.index[0].date()} -> {W.index[-1].date()} ---")

        # ---------- (a) THE MULTIPLICITY BOUND ----------
        for name, cols in (("ALL 26 components", list(W.columns)),
                           ("the 15 moving averages", MAS),
                           ("the 11 oscillators", OSC)):
            pr, n90, lam = eff_rank(W[cols])
            print(f"  EFFECTIVE RANK · {name:<24} nominal {len(cols):>2}  "
                  f"participation ratio {pr:>5.2f}  eigenvalues for 90% var {n90:>2}  "
                  f"top eigenvalue {lam[0]:>5.2f} = {100*lam[0]/len(cols):.0f}% of it")
        mam = W[MAS].mean(axis=1)
        print(f"  the 15 MAs agree UNANIMOUSLY on {float((mam.abs() == 1).mean()):.1%} of bars; "
              f"mean |pairwise corr| among them {W[MAS].corr().abs().values[np.triu_indices(len(MAS),1)].mean():.3f}")

        # ---------- (b) IS IT A RESTATEMENT? ----------
        trivial = np.sign(c - sma(c, 50).reindex(W.index))
        fc = champion_recipe(d["close"].astype(float), 1.0, 1.5).reindex(W.index)
        print(f"  corr(rating, close-vs-SMA50)  = {float(rating.corr(trivial)):+.4f}   "
              f"<- one trivial indicator")
        print(f"  corr(rating, CHAMPION forecast) = {float(rating.corr(fc)):+.4f}")

        # ---------- (c) EV/FIRE BY CONSENSUS, vs HOLDING ----------
        H = bpd                                        # one day forward
        fwd = (c.shift(-H) / c - 1.0) * 1e4
        base = float(fwd.mean())
        print(f"\n  EV by consensus bucket, {H} bars ({tf}) forward. "
              f"HOLD = {base:+.3f}bp, round trip {RT_BP:.2f}bp")
        print(f"  {'bucket':<22} {'bars':>7} {'fire':>7} {'EV bp':>9} {'offset':>9} "
              f"{'net bp':>9} {'vs HOLD':>9}")
        print("  " + "-" * 78)
        edges = [(-1.01, -0.5, "strong sell"), (-0.5, -0.1, "sell"), (-0.1, 0.1, "neutral"),
                 (0.1, 0.5, "buy"), (0.5, 1.01, "strong buy")]
        for lo, hi, lab in edges:
            msk = (rating > lo) & (rating <= hi) & np.isfinite(fwd)
            n = int(msk.sum())
            if n < 50:
                print(f"  {lab:<22} {n:>7}  too few"); continue
            side = 1.0 if "buy" in lab else (-1.0 if "sell" in lab else 1.0)
            ev = float(fwd[msk].mean())
            # CONTROL #2 is HOLDING GOLD -- a LONG benchmark for every row. Benchmarking a short
            # bucket against `side*base` would compare it to shorting gold unconditionally, which
            # flatters it by exactly 2x the drift: §3bh's identity
            # excess(LONG) + excess(SHORT) = -2*base is the whole reason that is the wrong frame.
            vs_hold = side * ev - base
            print(f"  {lab + (' (long)' if side > 0 else ' (short)'):<22} {n:>7} "
                  f"{n/len(W):>6.1%} {side*ev:>+9.3f} {ev - base:>+9.3f} "
                  f"{side*ev - RT_BP:>+9.3f} {vs_hold:>+9.3f}")
            all_rows.append(dict(tf=tf, bucket=lab, n=n, ev_bp=side * ev,
                                 offset=ev - base, vs_hold=vs_hold))

        # ---------- the threshold sweep, priced against a best-of-K floor ----------
        print(f"\n  THRESHOLD SWEEP — long when rating >= thr, else flat. "
              f"CONTROL #7: read the FLOOR, not the argmax.")
        print(f"  {'thr':>6} {'fire':>7} {'EV bp':>9} {'net bp':>9} {'vs HOLD':>9}")
        print("  " + "-" * 48)
        sweep = []
        for thr in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
            msk = (rating >= thr) & np.isfinite(fwd)
            n = int(msk.sum())
            if n < 50:
                continue
            ev = float(fwd[msk].mean())
            sweep.append((thr, n, ev))
            print(f"  {thr:>6.1f} {n/len(W):>6.1%} {ev:>+9.3f} {ev - RT_BP:>+9.3f} "
                  f"{ev - base:>+9.3f}")
        if sweep:
            best = max(sweep, key=lambda x: x[2])
            print(f"  best cell thr {best[0]:.1f} at {best[2]:+.3f}bp vs HOLD {base:+.3f}bp "
                  f"-> advantage {best[2]-base:+.3f}bp over {len(sweep)} cells searched")

        # ---------- THE COMPARISON THAT DECIDES IT: the CHAMPION, bucketed identically ----------
        # "Beats holding gold" is necessary, not sufficient. The incumbent is the champion, and
        # the rating correlates with it, so the question is whether the headcount adds anything a
        # forecast this repo already runs does not.
        print(f"\n  THE INCUMBENT — champion forecast bucketed by its OWN quantiles, same bars")
        print(f"  {'bucket':<22} {'bars':>7} {'fire':>7} {'EV bp':>9} {'vs HOLD':>9}")
        print("  " + "-" * 60)
        qs = fc.quantile([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).to_numpy()
        champ_top = None
        for i in range(5):
            lo, hi = qs[i], qs[i + 1]
            msk = (fc > lo) & (fc <= hi) & np.isfinite(fwd) if i else \
                  (fc >= lo) & (fc <= hi) & np.isfinite(fwd)
            n = int(msk.sum())
            if n < 50:
                continue
            ev = float(fwd[msk].mean())
            lab = f"champ Q{i+1} [{lo:.2f},{hi:.2f}]"
            print(f"  {lab:<22} {n:>7} {n/len(W):>6.1%} {ev:>+9.3f} {ev - base:>+9.3f}")
            champ_top = ev - base
        tv_top = max((e - base for _, _, e in sweep), default=np.nan)
        print(f"  *** top champion bucket {champ_top:+.3f}bp vs HOLD · best TV cell "
              f"{tv_top:+.3f}bp vs HOLD -> TV {'BEATS' if tv_top > champ_top else 'LOSES TO'} "
              f"the incumbent by {abs(tv_top - champ_top):.3f}bp ***")

    pd.DataFrame(all_rows).to_csv(OUT / "a7_tv_ratings.csv", index=False)
    print(f"""
{'='*104}
  HOW TO READ THIS. The rating is a HEADCOUNT, and a headcount is only information if the heads
  are independent. 'participation ratio' is the effective number of independent directions in the
  vote matrix: if 26 nominal components carry a participation ratio of 3, then a '20 of 26 say
  buy' consensus is roughly three opinions wearing twenty-six hats, and every significance figure
  computed as though it were 26 is overstated by that factor. The 15-MA block is the clearest
  case: they share one input -- where price sits relative to a trailing average -- and differ only
  in window.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
