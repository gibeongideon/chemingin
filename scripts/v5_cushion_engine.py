"""CUSHION ENGINE — maximise return subject to a HARD 5% drawdown floor.

A DIFFERENT OBJECTIVE, not another signal. Everything in this repo so far
maximises Sharpe and then discovers what drawdown falls out (typically -8% to
-13%). That is the wrong optimisation for the stated goal. Here the 5% floor is a
CONSTRAINT and return is the objective, which changes the architecture entirely.

Why the usual book cannot get there: at Sharpe ~1 a 5% max-DD cap forces roughly
2% vol and therefore ~2% annual return. High return under a tight floor is only
reachable three ways, so this engine uses all three:

  1. BREADTH. IR ~ IC x sqrt(N). One-to-three sleeves cannot clear the bar; ~28
     weakly-correlated markets can. Breadth is the only free lunch available.
  2. POSITIVE SKEW. The floor is a constraint on the LEFT TAIL, so the payoff
     shape matters more than the mean. Discrete breakout bets with a hard ATR
     stop produce many tiny losses and few large wins; that thins the DD tail in
     a way a continuously-exposed vol-targeted book never does.
  3. PATH-DEPENDENT LEVERAGE (CPPI). Risk is budgeted from the CUSHION
     (equity - floor), not from equity. Deep in profit the cushion is large and
     the engine presses; near the floor the cushion collapses and size goes to
     zero MECHANICALLY, before any discretionary rule fires.

WEEKEND-FLAT IS A FEATURE HERE, NOT A TAX. CPPI's classic failure is gapping
straight through the floor. Being flat every Friday close removes weekend gap
risk entirely — the same constraint that COST the trend follower ~45% of its edge
(v5_xau_weekly_cycle.py) is what makes a hard floor enforceable. Mon->Fri is
therefore load-bearing, not an imposition.

Costs floored at live FTMO quotes (see v5_ftmo_xau_diversifiers.py — the CSV
spreads understate by 20-50x on illiquid names and manufacture fake edges).
"""
from __future__ import annotations
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = "/home/rock/Desktop/2026_Projects/Trader36/MT5"

EQUITY0 = 100_000.0
SLIP_BP = 0.5           # extra execution slip on top of half-spread

# asset -> live FTMO spread bp (verified on account 1514025597, 2026-07-24).
# FX excluded (dead post-2016). Ags/energy kept but honestly priced.
UNIVERSE = {
    "GOLD": 1.11, "SILVER": 9.12, "PLAT": 46.56, "PALL": 56.36, "COPPER": 8.89,
    "WTI": 9.67, "BRENT": 7.58, "NATGAS": 199.37, "HEATOIL": 80.13,
    "BTC": 0.15, "ETH": 3.18, "LTC": 32.18, "SOL": 3.96, "AVA": 32.05,
    "SPX": 0.81, "DJI": 0.40, "NDX": 0.69, "DAX": 0.49, "FTSE": 0.70,
    "STOXX": 1.70, "NIKKEI": 1.55, "ASX": 1.14,
    "CORN": 22.70, "WHEAT": 28.55, "SOY": 17.78, "SUGAR": 83.68,
    "COFFEE": 10.89, "COTTON": 48.78,
}


def load_panel(start="2012-01-01"):
    """Aligned OHLC + ATR + per-bar one-way cost fraction for every asset."""
    out = {}
    for a, bp in UNIVERSE.items():
        try:
            df = pd.read_csv(f"{ROOT}/data/{a}_D1_long.csv",
                             parse_dates=["time"], index_col="time").sort_index()
        except Exception:                                   # noqa: BLE001
            continue
        df = df[~df.index.duplicated(keep="last")].loc[start:]
        if len(df) < 500:
            continue
        df = df[df.index.dayofweek < 5]
        pc = df["close"].shift(1)
        tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                        (df["low"] - pc).abs()], axis=1).max(axis=1)
        df["atr"] = tr.rolling(20, min_periods=10).mean()
        # one-way cost as a FRACTION of price: half the spread plus slip
        df["cost"] = (bp / 2.0 + SLIP_BP) / 1e4
        out[a] = df.dropna(subset=["atr"])
    return out


def signals(df, lookback):
    """Donchian breakout state: +1 new N-day high, -1 new N-day low, else 0.

    Deliberately DISCRETE. A continuous forecast keeps you exposed all the time,
    which is what makes drawdowns fat; a breakout only puts capital at risk when
    the market has already moved in your favour, which is where the positive skew
    comes from.
    """
    hi = df["high"].rolling(lookback).max().shift(1)
    lo = df["low"].rolling(lookback).min().shift(1)
    up = (df["close"] > hi).astype(int)
    dn = (df["close"] < lo).astype(int)
    return up - dn


def run(panel, lookback=20, stop_atr=2.0, risk_frac=0.02, max_dd=0.05,
        long_only=False, max_pos=12, weekend_flat=True, trail=True,
        static_floor=False, start=None, end=None):
    """Event-driven daily simulation. Returns (equity_series, trades_df)."""
    days = sorted(set().union(*[set(d.index) for d in panel.values()]))
    days = [d for d in days if (start is None or d >= pd.Timestamp(start))
            and (end is None or d <= pd.Timestamp(end))]
    sig = {a: signals(d, lookback) for a, d in panel.items()}

    equity, hwm = EQUITY0, EQUITY0
    floor = EQUITY0 * (1 - max_dd)
    book: dict[str, dict] = {}
    eq_curve, trades = [], []

    for i, t in enumerate(days):
        is_fri = (t.dayofweek == 4) or (i + 1 == len(days)) or \
                 (days[i + 1].dayofweek < t.dayofweek if i + 1 < len(days) else False)

        # ---- 1. mark open positions, honour stops intraday, then weekend flat
        for a in list(book.keys()):
            d = panel[a]
            if t not in d.index:
                continue
            p, row = book[a], d.loc[t]
            px_prev = p["px"]
            hit = (row["low"] <= p["stop"]) if p["dir"] > 0 else (row["high"] >= p["stop"])
            if hit:
                px = p["stop"]                       # stop fills at the level
                equity += p["units"] * p["dir"] * (px - px_prev)
                equity -= abs(p["units"]) * px * float(row["cost"])
                trades.append(dict(asset=a, exit=t, why="stop",
                                   pnl=p["units"] * p["dir"] * (px - p["entry_px"])))
                del book[a]
                continue
            px = float(row["close"])
            equity += p["units"] * p["dir"] * (px - px_prev)
            p["px"] = px
            if trail:                                # ratchet the stop, never loosen
                s = px - p["dir"] * stop_atr * float(row["atr"])
                p["stop"] = max(p["stop"], s) if p["dir"] > 0 else min(p["stop"], s)
            if weekend_flat and is_fri:
                equity -= abs(p["units"]) * px * float(row["cost"])
                trades.append(dict(asset=a, exit=t, why="friday",
                                   pnl=p["units"] * p["dir"] * (px - p["entry_px"])))
                del book[a]

        # ---- 2. CPPI bookkeeping: ratchet the floor with the high-water mark
        hwm = max(hwm, equity)
        if not static_floor:
            floor = max(floor, hwm * (1 - max_dd))
        cushion = max(0.0, equity - floor)

        # ---- 3. new entries, risk budgeted from the CUSHION not from equity
        if cushion > 0 and not (weekend_flat and is_fri):
            budget = risk_frac * cushion
            for a, d in panel.items():
                if a in book or len(book) >= max_pos or t not in d.index:
                    continue
                s = sig[a]
                if t not in s.index or s.loc[t] == 0:
                    continue
                direction = int(s.loc[t])
                if long_only and direction < 0:
                    continue
                row = d.loc[t]
                px, atr = float(row["close"]), float(row["atr"])
                dist = stop_atr * atr
                if dist <= 0 or px <= 0:
                    continue
                units = budget / dist                # lose exactly `budget` if stopped
                notional = units * px
                if notional > equity * 3.0:          # per-name leverage sanity cap
                    units = equity * 3.0 / px
                equity -= abs(units) * px * float(row["cost"])
                book[a] = dict(dir=direction, units=units, px=px, entry_px=px,
                               stop=px - direction * dist, entry=t)
        eq_curve.append((t, equity))

    eq = pd.Series(dict(eq_curve)).sort_index()
    return eq, pd.DataFrame(trades)


def report(eq, tr, label):
    if len(eq) < 100:
        return None
    yrs = len(eq) / 252
    r = eq.pct_change().dropna()
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1)
    mdd = float(dd.min())
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    sr = float(r.mean() / r.std() * np.sqrt(252)) if r.std() else np.nan
    wr = float((tr["pnl"] > 0).mean() * 100) if len(tr) else np.nan
    return dict(label=label, cagr=cagr * 100, mdd=mdd * 100, calmar=calmar,
                sr=sr, n=len(tr), wr=wr, skew=float(r.skew()),
                worst=float(r.min() * 100), final=float(eq.iloc[-1]))


HDR = (f"{'variant':38s} {'CAGR%':>7} {'maxDD%':>7} {'Calmar':>7} {'Sharpe':>7} "
       f"{'trades':>7} {'win%':>6} {'skew':>6} {'worstday%':>9}")


def show(r):
    if r is None:
        return
    print(f"{r['label']:38s} {r['cagr']:+7.1f} {r['mdd']:7.2f} {r['calmar']:7.2f} "
          f"{r['sr']:+7.2f} {r['n']:7d} {r['wr']:6.1f} {r['skew']:+6.2f} "
          f"{r['worst']:9.2f}")




# ---------------------------------------------------------------------------
# CONTINUOUS ENGINE. The discrete-breakout version above loses money once the
# Friday exit is enforced: a breakout needs weeks to pay and a 2xATR trail plus a
# weekly flat cuts every winner short (measured: -0.3% CAGR, 50.2% win rate, and
# -11% CAGR when sized off equity). That is a signal problem, not a risk-engine
# problem — so keep the cushion architecture and put PROVEN alpha underneath it:
# the champion recipe run weekend-flat across the broad universe.
# ---------------------------------------------------------------------------
sys.path.insert(0, ROOT + "/scripts")
import v5_basket_challenge as vbc  # noqa: E402

# Names whose live spread cannot support frequent trading. Kept out of the book
# rather than silently subsidised (NATGAS 199bp, HEATOIL 80bp, SUGAR 84bp,
# PALL 56bp, COTTON 49bp, PLAT 47bp round-trip against ~1% daily moves).
COSTLY = {"NATGAS", "HEATOIL", "SUGAR", "PALL", "COTTON", "PLAT"}


def asset_stream(df, bp, long_only=True, weekend_flat=True, target_vol=0.10):
    """Weekend-flat daily net returns for one asset at `target_vol`."""
    close, open_ = df["close"], df["open"]
    spread_px = close * bp / 1e4
    cost_px = spread_px / 2 + close * SLIP_BP / 1e4
    ret = close.pct_change()
    vol = ret.ewm(halflife=42, min_periods=20).std() * np.sqrt(252)
    fc = vbc.champ_recipe_lo(close) if long_only else \
        vbc.ewmac_fc(close, ((16, 64), (32, 128), (64, 256)))
    raw = (fc * (target_vol / vol)).clip(-8, 8)
    if long_only:
        raw = raw.clip(lower=0.0)
    pos = raw.shift(1).fillna(0.0)

    idx = df.index
    wk = pd.Series(idx.isocalendar().year * 100 + idx.isocalendar().week, index=idx)
    first, last = ~wk.duplicated(keep="first"), ~wk.duplicated(keep="last")
    out = pd.Series(0.0, index=idx)
    pv, cv, ov, kv = pos.values, close.values, open_.values, cost_px.values
    fv, lv = first.values, last.values
    prev_end = 0.0
    for i in range(1, len(idx)):
        p = pv[i]
        if weekend_flat:
            gross = p * (cv[i] / ov[i] - 1.0) if fv[i] else p * (cv[i] / cv[i - 1] - 1.0)
            turn = abs(p - prev_end) + (abs(p) if lv[i] else 0.0)
            prev_end = 0.0 if lv[i] else p
        else:
            gross = p * (cv[i] / cv[i - 1] - 1.0)
            turn = abs(p - pv[i - 1])
        out.iloc[i] = gross - turn * (kv[i] / cv[i])
    return out


def build_book(panel, long_only=True, weekend_flat=True, drop_costly=True,
               book_vol=0.10):
    """Equal-risk combine every asset, then scale the book to `book_vol`."""
    streams = {}
    for a, df in panel.items():
        if drop_costly and a in COSTLY:
            continue
        streams[a] = asset_stream(df, UNIVERSE[a], long_only, weekend_flat)
    M = pd.DataFrame(streams).fillna(0.0)
    M = M.loc[(M != 0).any(axis=1)]
    inv = 1.0 / M.std().replace(0, np.nan)
    W = (inv / inv.sum()).fillna(0.0)
    book = (M * W).sum(axis=1)
    g = book_vol / (book.std() * np.sqrt(252))
    return book * g, M.shape[1]


def cppi(book, max_dd=0.05, mult=4.0, cap=6.0, static_floor=False):
    """Grow equity at exposure = mult x cushion/equity, capped.

    The floor ratchets with the high-water mark, so the constraint is on
    PEAK-TO-TROUGH drawdown. `cap` stops the multiplier exploding when the
    cushion is large — without it CPPI takes unbounded leverage in calm markets.
    """
    eq, hwm = EQUITY0, EQUITY0
    floor = EQUITY0 * (1 - max_dd)
    curve, expo = [], []
    for t, r in book.items():
        cush = max(0.0, eq - floor)
        e = min(cap, mult * cush / eq) if eq > 0 else 0.0
        eq *= (1.0 + e * r)
        hwm = max(hwm, eq)
        if not static_floor:
            floor = max(floor, hwm * (1 - max_dd))
        curve.append((t, eq))
        expo.append(e)
    return pd.Series(dict(curve)), float(np.mean(expo))


def rep2(eq, label, expo=np.nan):
    yrs = len(eq) / 252
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dd = float((eq / eq.cummax() - 1).min())
    r = eq.pct_change().dropna()
    return dict(label=label, cagr=cagr * 100, mdd=dd * 100,
                calmar=cagr / abs(dd) if dd < 0 else np.nan,
                sr=float(r.mean() / r.std() * np.sqrt(252)) if r.std() else np.nan,
                expo=expo, final=float(eq.iloc[-1]))


H2 = (f"{'variant':40s} {'CAGR%':>7} {'maxDD%':>7} {'Calmar':>7} {'Sharpe':>7} "
      f"{'avgLev':>7} {'final$':>12}")


def show2(r):
    print(f"{r['label']:40s} {r['cagr']:+7.2f} {r['mdd']:7.2f} {r['calmar']:7.2f} "
          f"{r['sr']:+7.2f} {r['expo']:7.2f} {r['final']:12,.0f}")



# Equal-weighting all 22 names gives a book Sharpe of only +0.37: the ags and
# weak indices have ~zero standalone edge, and the Fundamental Law only pays for
# breadth when each bet has POSITIVE IC. Adding zero-IC bets adds noise, not
# breadth. So the universe must be selected — walk-forward, never in-sample.
QUALITY = ["GOLD", "BTC", "ETH", "SOL", "NDX", "SPX", "DJI", "DAX", "NIKKEI",
           "BRENT", "WTI", "SILVER", "COPPER", "STOXX", "FTSE", "ASX", "LTC"]


def walk_forward_book(panel, long_only=True, weekend_flat=True, book_vol=0.10,
                      lookback_y=3, top_k=8):
    """Re-select the traded universe each year from TRAILING Sharpe only.

    Picking the winners the FTMO study already found would be selection bias on
    the same sample. Here the universe is rebuilt every January from the prior
    `lookback_y` years, so every trade is out-of-sample with respect to the
    choice of what to trade.
    """
    streams = {a: asset_stream(panel[a], UNIVERSE[a], long_only, weekend_flat)
               for a in panel}
    M = pd.DataFrame(streams).fillna(0.0)
    M = M.loc[(M != 0).any(axis=1)]
    parts, chosen = [], {}
    for yr in sorted({d.year for d in M.index}):
        tr = M.loc[f"{yr - lookback_y}-01-01":f"{yr - 1}-12-31"]
        fwd = M.loc[f"{yr}-01-01":f"{yr}-12-31"]
        if len(tr) < 250 or len(fwd) == 0:
            continue
        sr = (tr.mean() / tr.std().replace(0, np.nan) * np.sqrt(252)).dropna()
        keep = list(sr[sr > 0].sort_values(ascending=False).head(top_k).index)
        if not keep:
            continue
        chosen[yr] = keep
        sub = fwd[keep]
        inv = 1.0 / sub.std().replace(0, np.nan)
        W = (inv / inv.sum()).fillna(0.0)
        parts.append((sub * W).sum(axis=1))
    book = pd.concat(parts).sort_index()
    g = book_vol / (book.std() * np.sqrt(252))
    return book * g, chosen


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2012-01-01")
    a = ap.parse_args()
    panel = load_panel(a.start)

    def sr_of(b):
        return float(b.mean() / b.std() * np.sqrt(252))

    all_b, n_all = build_book(panel)
    qpanel = {k: v for k, v in panel.items() if k in QUALITY}
    q_b, n_q = build_book(qpanel, drop_costly=False)
    wf_b, chosen = walk_forward_book(panel)
    q_alwaysin, _ = build_book(qpanel, drop_costly=False, weekend_flat=False)

    print(f"window {a.start} -> {all_b.index[-1].date()}  ({len(all_b) / 252:.1f}y)")
    print(f"  all-{n_all} equal-risk  weekend-flat   Sharpe {sr_of(all_b):+.2f}")
    print(f"  quality-{n_q} (in-sample pick)         Sharpe {sr_of(q_b):+.2f}")
    print(f"  WALK-FORWARD top-8 by trailing 3y      Sharpe {sr_of(wf_b):+.2f}")
    print(f"  quality-{n_q} ALWAYS-IN (weekend risk) Sharpe {sr_of(q_alwaysin):+.2f}")
    print(f"  walk-forward universe 2019: {chosen.get(2019)}")
    print(f"  walk-forward universe 2025: {chosen.get(2025)}\n")

    print("=" * 96)
    print("STEP 3 — best honest book, CPPI vs fixed vol-target, hard 5% floor")
    print("=" * 96)
    print(H2)
    for m in (3, 4, 6, 8):
        eq, e = cppi(wf_b, max_dd=0.05, mult=m)
        show2(rep2(eq, f"WF book | CPPI mult {m}", e))
    print("-" * 96)
    for v in (0.02, 0.025, 0.03, 0.04):
        b2, _ = walk_forward_book(panel, book_vol=v)
        eq = EQUITY0 * (1 + b2).cumprod()
        show2(rep2(eq, f"WF book | fixed vol {v:.1%}", v / 0.10))

    print("\n" + "=" * 96)
    print("STEP 4 — THE FRONTIER: what return is actually buyable at each DD cap")
    print("=" * 96)
    print(f"{'DD cap':>8} {'best CAGR%':>11} {'via':>22} {'realised maxDD%':>17} "
          f"{'Calmar':>7}")
    for cap in (0.03, 0.05, 0.075, 0.10, 0.15):
        best = None
        for m in (2, 3, 4, 6, 8, 10):
            eq, e = cppi(wf_b, max_dd=cap, mult=m)
            r = rep2(eq, f"CPPI m={m}", e)
            if r["mdd"] >= -cap * 100 and (best is None or r["cagr"] > best["cagr"]):
                best = r
        for v in (0.01, 0.02, 0.03, 0.05, 0.07, 0.10):
            b2, _ = walk_forward_book(panel, book_vol=v)
            eq = EQUITY0 * (1 + b2).cumprod()
            r = rep2(eq, f"vol-target {v:.0%}", v)
            if r["mdd"] >= -cap * 100 and (best is None or r["cagr"] > best["cagr"]):
                best = r
        if best:
            print(f"{cap * 100:7.1f}% {best['cagr']:11.2f} {best['label']:>22} "
                  f"{best['mdd']:17.2f} {best['calmar']:7.2f}")
