"""XAUUSD WEEKLY-CYCLE trend follower: enter Monday, flat by Friday (<=5 days).

Motivation is a hard broker rule, not a hunch: FundingPips MASTER/FUNDED accounts
have not allowed weekend holding since 29-Jan-2026 (positions are auto-closed
Friday). The deployed champion is an always-in H4 trend follower that carries
through weekends, so it cannot run as-is on a funded account. This asks whether
the gold trend edge survives being chopped into Mon->Fri slices.

MECHANICS (strictly causal):
  * forecast is computed on D1 closes and read from the LAST close STRICTLY
    BEFORE the entry day — i.e. the prior Friday. No same-bar lookahead.
  * enter at MONDAY'S OPEN (the first trading day of the ISO week), which is what
    a Monday-morning bot pass can actually get.
  * hold at most `max_days` trading days, and never past the week's last day, so
    the book is flat over every weekend.
  * exit at that day's CLOSE.

COSTS are floored at FTMO's live XAUUSD quote (1.11 bp = ~$0.448/oz, measured
2026-07-24), never the optimistic CSV spread, and slippage is charged both ways.
This matters far more here than for the always-in book: forcing a round trip
every single week is ~52 round trips a year against the champion's handful, so
the strategy pays turnover the always-in version never does. Getting this wrong
is how the HEATOIL "diversifier" and the XAU fade signal both looked profitable.
"""
from __future__ import annotations
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = "/home/rock/Desktop/2026_Projects/Trader36/MT5"
sys.path.insert(0, ROOT + "/scripts")
import v5_basket_challenge as vbc  # noqa

LIVE_BP = 1.11          # FTMO XAUUSD live spread, 2026-07-24
SLIP_USD = 0.10         # project convention, 1 pip
TARGET_VOL = 0.10       # annualised, for clean Sharpe comparison
START = "2008-01-01"    # overridden by --start


def load_gold():
    df = pd.read_csv(f"{ROOT}/data/GOLD_D1_long.csv",
                     parse_dates=["time"], index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    csv_spr = df["spread"].clip(lower=df["spread"].median())
    df["spread_px"] = np.maximum(csv_spr, df["close"] * LIVE_BP / 1e4)
    df["cost_px"] = df["spread_px"] / 2 + SLIP_USD          # one way
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=5).mean()
    return df


def forecasts(close):
    """Champion recipe (long-only) plus the raw components, on D1."""
    ew = vbc.ewmac_fc(close, ((16, 64), (32, 128), (64, 256)))
    bk = vbc.breakout_fc(close, (10, 20, 40))
    champ_lo = vbc.champ_recipe_lo(close)
    # faster variants: a 5-day horizon may want quicker signals than the
    # always-in book, so give the sweep something short to pick.
    ew_f = vbc.ewmac_fc(close, ((4, 16), (8, 32), (16, 64)))
    bk_f = vbc.breakout_fc(close, (5, 10, 20))
    return {
        "champ_lo": champ_lo,
        "ewmac": ew,
        "breakout": bk,
        "ewmac_fast": ew_f,
        "breakout_fast": bk_f,
    }


def run_cycle(df, fc, max_days=5, long_only=True, early_exit=False,
              tp_pct=None, tp_atr=None):
    """Weekly Mon->Fri cycle. Returns (daily_net_series, trade_list).

    tp_pct / tp_atr add a PROFIT TARGET: a resting limit order that takes the
    trade off any day the price trades through it, leaving the book flat until
    the next Monday. Fill realism: the target is only counted as hit when the
    day's HIGH (long) or LOW (short) actually reaches it, and the fill is booked
    at the target price — correct for a resting limit, which provides liquidity
    rather than crossing the spread. The entry day's range is fair game because
    entry is at the open. The exit still pays one-way cost, so a target that
    fires often is not free.
    """
    close, open_, cost_px = df["close"], df["open"], df["cost_px"]
    high, low, atr = df["high"], df["low"], df["atr"]
    ret = close.pct_change()
    vol = ret.ewm(halflife=42, min_periods=20).std() * np.sqrt(252)

    idx = df.index
    weeks = pd.Series(idx.isocalendar().year * 100 + idx.isocalendar().week,
                      index=idx)
    daily = pd.Series(0.0, index=idx)
    trades = []

    for _, days in df.groupby(weeks).groups.items():
        days = pd.DatetimeIndex(sorted(days))
        if len(days) == 0:
            continue
        entry = days[0]
        prior = fc.index[fc.index < entry]
        if len(prior) == 0:
            continue
        sig = float(fc.loc[prior[-1]])
        v = float(vol.loc[prior[-1]]) if prior[-1] in vol.index else np.nan
        if not np.isfinite(sig) or not np.isfinite(v) or v <= 0:
            continue
        if long_only:
            sig = max(0.0, sig)
        size = float(np.clip(sig * (TARGET_VOL / v), -8, 8))
        if abs(size) < 1e-9:
            continue

        hold = days[:max_days]
        # optional early exit: leave the moment the signal stops agreeing
        if early_exit:
            keep = []
            for d in hold:
                keep.append(d)
                p = fc.index[fc.index < d]
                if len(p) and d != hold[0]:
                    s2 = float(fc.loc[p[-1]])
                    if long_only and s2 <= 0:
                        break
                    if not long_only and np.sign(s2) != np.sign(size):
                        break
            hold = pd.DatetimeIndex(keep)
        e_px = float(open_.loc[entry])
        direction = 1.0 if size > 0 else -1.0

        # profit target: first day whose range trades through it ends the trade
        target, hit = None, False
        if tp_pct is not None:
            target = e_px * (1.0 + direction * tp_pct)
        elif tp_atr is not None:
            a = float(atr.loc[prior[-1]]) if prior[-1] in atr.index else np.nan
            if np.isfinite(a) and a > 0:
                target = e_px + direction * tp_atr * a
        exit_day, x_px = hold[-1], float(close.loc[hold[-1]])
        if target is not None:
            for d in hold:
                if (direction > 0 and float(high.loc[d]) >= target) or \
                   (direction < 0 and float(low.loc[d]) <= target):
                    exit_day, x_px, hit = d, target, True
                    break
        hold = hold[:list(hold).index(exit_day) + 1]

        c_in = float(cost_px.loc[entry]) / e_px
        c_out = float(cost_px.loc[exit_day]) / x_px

        # day 1 is open->close; subsequent days are close->close; the exit day
        # marks to the fill price rather than the close
        prev = e_px
        for i, d in enumerate(hold):
            c = x_px if d == exit_day else float(close.loc[d])
            r = size * (c / prev - 1.0)
            if i == 0:
                r -= abs(size) * c_in
            if d == exit_day:
                r -= abs(size) * c_out
            daily.loc[d] = daily.loc[d] + r
            prev = c
        trades.append(dict(entry=entry, exit=exit_day, days=len(hold), size=size,
                           hit=hit,
                           gross=size * (x_px / e_px - 1.0),
                           net=size * (x_px / e_px - 1.0) - abs(size) * (c_in + c_out)))
    tdf = pd.DataFrame(trades)
    # slice trades to the SAME window as the daily series, or win%/hit%/n would
    # be reported over all history while the Sharpe came from the window only
    if len(tdf):
        tdf = tdf[tdf["entry"] >= pd.Timestamp(START)].reset_index(drop=True)
    return daily.loc[START:], tdf


def run_weekend_flat(df, fc, long_only=True):
    """Keep the DAILY signal (resize every day) but hold nothing over a weekend.

    This separates the two things a Monday-only cycle conflates: (a) losing the
    weekend gap and paying a forced round trip, versus (b) freezing one size on
    Friday's read and refusing to react until the next Monday. It is also the
    variant that maps cleanly onto the funded-account rule — the bot can trade
    normally Mon-Fri, it just may not carry risk through Friday's close.
    """
    close, open_, cost_px = df["close"], df["open"], df["cost_px"]
    ret = close.pct_change()
    vol = ret.ewm(halflife=42, min_periods=20).std() * np.sqrt(252)
    raw = (fc * (TARGET_VOL / vol)).clip(-8, 8)
    if long_only:
        raw = raw.clip(lower=0.0)
    pos = raw.shift(1).fillna(0.0)              # size carried into day t

    idx = df.index
    wk = idx.isocalendar().year * 100 + idx.isocalendar().week
    wk = pd.Series(wk, index=idx)
    first = ~wk.duplicated(keep="first")
    last = ~wk.duplicated(keep="last")

    daily = pd.Series(0.0, index=idx)
    pv, cv, ov = pos.values, close.values, open_.values
    fv, lv, kv = first.values, last.values, cost_px.values
    prev_end = 0.0
    for i in range(1, len(idx)):
        p = pv[i]
        # entry day of the week prices off the OPEN: the weekend gap is not ours
        gross = p * (cv[i] / ov[i] - 1.0) if fv[i] else p * (cv[i] / cv[i - 1] - 1.0)
        turn = abs(p - prev_end)
        if lv[i]:
            turn += abs(p)                      # forced flat into the weekend
        daily.iloc[i] = gross - turn * (kv[i] / cv[i])
        prev_end = 0.0 if lv[i] else p
    return daily.loc[START:]


def sharpe_se(sr, n_days):
    """Lo (2002) standard error of an annualised Sharpe.

    Reported because short samples invite over-reading: at 4.5 years the SE of a
    Sharpe is ~0.47, so a +0.5 point estimate is not distinguishable from zero.
    """
    yrs = n_days / 252.0
    return float(np.sqrt((1.0 + 0.5 * sr ** 2) / yrs)) if yrs > 0 else np.nan


def stats(daily, trades, label):
    d = daily.dropna()
    if d.std() == 0 or len(d) < 100:
        return None
    sr = float(d.mean() / d.std() * np.sqrt(252))
    eq = (1 + d).cumprod()
    dd = float((eq / eq.cummax() - 1).min() * 100)
    ann = float(d.mean() * 252 * 100)
    # split the sample in half by DATE so the halves adapt to --start
    mid = d.index[len(d) // 2]
    h1, h2 = d.loc[:mid], d.loc[mid:]
    sr1 = float(h1.mean() / h1.std() * np.sqrt(252)) if h1.std() else np.nan
    sr2 = float(h2.mean() / h2.std() * np.sqrt(252)) if h2.std() else np.nan
    wr = float((trades["net"] > 0).mean() * 100) if len(trades) else np.nan
    gross_sr = np.nan
    if len(trades):
        g = trades["gross"]
        gross_sr = float(g.mean() / g.std() * np.sqrt(52)) if g.std() else np.nan
    return dict(label=label, sr=sr, gross_sr=gross_sr, ann=ann, dd=dd,
                n=len(trades), wr=wr, avgd=float(trades["days"].mean()) if len(trades) else np.nan,
                sr1=sr1, sr2=sr2, se=sharpe_se(sr, len(d)))


HDR = (f"{'variant':34s} {'netSR':>6} {'+-SE':>5} {'grossSR':>8} {'ann%':>6} "
       f"{'maxDD':>7} {'trades':>7} {'win%':>6} {'avgd':>5} {'1stH':>6} {'2ndH':>6}")


def show(r):
    if r is None:
        return
    print(f"{r['label']:34s} {r['sr']:+6.2f} {r['se']:5.2f} {r['gross_sr']:+8.2f} "
          f"{r['ann']:6.1f} {r['dd']:6.1f}% {r['n']:7d} {r['wr']:6.1f} "
          f"{r['avgd']:5.1f} {r['sr1']:+6.2f} {r['sr2']:+6.2f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2008-01-01",
                    help="restrict the evaluation window, e.g. 2022-01-01")
    _a = ap.parse_args()
    START = _a.start
    df = load_gold()
    FC = forecasts(df["close"])
    print(f"gold D1 {df.index.min().date()} -> {df.index.max().date()}  "
          f"{len(df)} bars | spread floored at {LIVE_BP}bp "
          f"(~${df['close'].iloc[-1] * LIVE_BP / 1e4:.3f}/oz) + ${SLIP_USD} slip")
    print(f"EVALUATION WINDOW: {START} -> {df.index.max().date()}\n")

    # STEP 0 — the control that decides whether any of the rest means anything.
    # A long-only gold trend follower in a gold bull market will post a fine
    # Sharpe without any timing skill at all. If the strategy does not beat
    # simply owning the metal over the same window, what it is selling is
    # drawdown control, not return generation. Never read the tables below
    # without this line. (2026-07-24)
    _b = df.loc[START:]["close"].pct_change().dropna()
    _sr = float(_b.mean() / _b.std() * np.sqrt(252))
    _eq = (1 + _b).cumprod()
    print("=" * 108)
    print(f"STEP 0 — CONTROL: buy & hold gold over the same window   "
          f"SR {_sr:+.2f} +-{sharpe_se(_sr, len(_b)):.2f}   "
          f"vol {_b.std() * np.sqrt(252) * 100:.1f}%   "
          f"maxDD {(_eq / _eq.cummax() - 1).min() * 100:.1f}%")
    print("=" * 108)
    print("Any variant below that does not clear this is capturing gold beta, "
          "not timing skill.\n")

    print("=" * 108)
    print("STEP 1 — max-hold sweep, champion recipe long-only, enter Mon open")
    print("=" * 108)
    print(HDR)
    for md in (1, 2, 3, 4, 5):
        daily, tr = run_cycle(df, FC["champ_lo"], max_days=md)
        show(stats(daily, tr, f"champ_lo  hold<={md}d"))

    print("\n" + "=" * 108)
    print("STEP 2 — signal sweep at hold<=5d (long-only vs long-short)")
    print("=" * 108)
    print(HDR)
    for name, f in FC.items():
        daily, tr = run_cycle(df, f, max_days=5, long_only=True)
        show(stats(daily, tr, f"{name}  LO"))
    print("-" * 108)
    for name, f in FC.items():
        if name == "champ_lo":
            continue                       # already long-only by construction
        daily, tr = run_cycle(df, f, max_days=5, long_only=False)
        show(stats(daily, tr, f"{name}  LS"))

    print("\n" + "=" * 108)
    print("STEP 3 — early exit on signal flip vs always hold to Friday")
    print("=" * 108)
    print(HDR)
    for name in ("champ_lo", "ewmac", "breakout_fast"):
        daily, tr = run_cycle(df, FC[name], max_days=5, early_exit=False)
        show(stats(daily, tr, f"{name}  hold-to-Fri"))
        daily, tr = run_cycle(df, FC[name], max_days=5, early_exit=True)
        show(stats(daily, tr, f"{name}  early-exit"))

    print("\n" + "=" * 108)
    print("STEP 4 — what the weekend flat costs: always-in D1 benchmark")
    print("=" * 108)
    close = df["close"]
    ret = close.pct_change()
    vol = ret.ewm(halflife=42, min_periods=20).std() * np.sqrt(252)
    print(HDR)
    for name in ("champ_lo", "ewmac"):
        f = FC[name]
        pos = (f * (TARGET_VOL / vol)).clip(-8, 8).shift(1).fillna(0.0)
        cost = pos.diff().abs().fillna(0.0) * (df["cost_px"] / close)
        net = (pos * ret - cost).fillna(0.0).loc[START:]
        fake = pd.DataFrame(dict(net=[0.0], days=[0.0], gross=[0.0]))
        r = stats(net, fake, f"{name}  ALWAYS-IN (no weekly cut)")
        if r:
            r["n"], r["wr"], r["avgd"], r["gross_sr"] = 0, np.nan, np.nan, np.nan
            show(r)

    print("\n" + "=" * 108)
    print("STEP 5 — WEEKEND-FLAT: keep the daily signal, only refuse weekend risk")
    print("=" * 108)
    print(HDR)
    blank = pd.DataFrame(dict(net=[0.0], days=[0.0], gross=[0.0]))
    for name in ("champ_lo", "ewmac", "breakout"):
        d = run_weekend_flat(df, FC[name], long_only=True)
        r = stats(d, blank, f"{name}  weekend-flat LO")
        if r:
            r["n"], r["wr"], r["avgd"], r["gross_sr"] = 0, np.nan, np.nan, np.nan
            show(r)

    print("\n" + "=" * 108)
    print("STEP 6 — challenge viability, WITH the book's vol-target + DD scaler")
    print("=" * 108)
    print("(same treatment the basket engine applies, so these pass% are directly")
    print(" comparable to the XAU+NDX+BRENT / XAU+BTC+NDX tables)\n")
    print(f"{'variant':34s} {'Sharpe':>7} {'maxDD':>7} {'FTMOpass%':>10} "
          f"{'FLEXpass%':>10} {'median':>8}")

    def challenged(daily, label):
        d = daily.dropna()
        rv = d.ewm(halflife=vbc.VT_HALFLIFE, min_periods=20).std() * np.sqrt(252)
        for model in ("ftmo", "flex"):
            pass
        out = {}
        for model in ("ftmo", "flex"):
            m = vbc.MODELS[model]
            vs = (m["vol"] / rv).clip(0.0, vbc.VT_MAXSCALE)
            eqb = (1 + d).cumprod()
            ds = (1 + (eqb / eqb.cummax() - 1) * 3.0).clip(lower=vbc.DD_FLOOR)
            vt = (d * (vs * ds).shift(1)).dropna()
            if vt.std() == 0:
                return
            r10 = (vt * (0.10 / (vt.std() * np.sqrt(252)))).values
            fp = vbc.fp_sim(r10, m["vol"] / vbc.TARGET_VOL, day_safety=1.5,
                            p1=m["p1"], p2=m["p2"], dayloss=m["daily"],
                            maxloss=m["maxloss"])
            eq = (1 + vt).cumprod()
            out[model] = (float(vt.mean() / vt.std() * np.sqrt(252)),
                          float((eq / eq.cummax() - 1).min() * 100),
                          fp["passpct"], fp["med_mo"])
        f = out["ftmo"]
        print(f"{label:34s} {f[0]:+7.2f} {f[1]:6.1f}% {f[2]:10.1f} "
              f"{out['flex'][2]:10.1f} {f[3]:7.1f}mo")

    for md, lbl in ((4, "champ_lo Mon->Thu (hold<=4d)"), (5, "champ_lo Mon->Fri (hold<=5d)")):
        daily, _ = run_cycle(df, FC["champ_lo"], max_days=md)
        challenged(daily, lbl)
    challenged(run_weekend_flat(df, FC["champ_lo"]), "champ_lo weekend-flat")
    close = df["close"]
    ret = close.pct_change()
    vol = ret.ewm(halflife=42, min_periods=20).std() * np.sqrt(252)
    pos = (FC["champ_lo"] * (TARGET_VOL / vol)).clip(-8, 8).shift(1).fillna(0.0)
    cost = pos.diff().abs().fillna(0.0) * (df["cost_px"] / close)
    challenged((pos * ret - cost).fillna(0.0).loc[START:],
               "champ_lo ALWAYS-IN (benchmark)")

    print("\n" + "=" * 108)
    print("STEP 7 — PROFIT TARGET sweep: take profit any day the level trades")
    print("=" * 108)
    THDR = (f"{'variant':34s} {'netSR':>6} {'+-SE':>5} {'ann%':>6} {'maxDD':>7} "
            f"{'trades':>7} {'win%':>6} {'hit%':>6} {'avgd':>5} {'1stH':>6} {'2ndH':>6}")

    def tshow(daily, tr, label):
        r = stats(daily, tr, label)
        if r is None:
            return None
        hit = float(tr["hit"].mean() * 100) if "hit" in tr and len(tr) else np.nan
        print(f"{r['label']:34s} {r['sr']:+6.2f} {r['se']:5.2f} {r['ann']:6.1f} "
              f"{r['dd']:6.1f}% {r['n']:7d} {r['wr']:6.1f} {hit:6.1f} "
              f"{r['avgd']:5.1f} {r['sr1']:+6.2f} {r['sr2']:+6.2f}")
        return r

    for md in (4, 5):
        print(f"\n-- champion recipe long-only, hold<={md}d --")
        print(THDR)
        d0, t0 = run_cycle(df, FC["champ_lo"], max_days=md)
        t0["hit"] = False
        tshow(d0, t0, f"no target (baseline, {md}d)")
        for pct in (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03):
            d, t = run_cycle(df, FC["champ_lo"], max_days=md, tp_pct=pct)
            tshow(d, t, f"TP {pct * 100:.2f}% move")
        for k in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            d, t = run_cycle(df, FC["champ_lo"], max_days=md, tp_atr=k)
            tshow(d, t, f"TP {k:.2f} x ATR14")

    print("\n" + "=" * 108)
    print("STEP 8 — best profit targets through the challenge sim")
    print("=" * 108)
    print(f"{'variant':34s} {'Sharpe':>7} {'maxDD':>7} {'FTMOpass%':>10} "
          f"{'FLEXpass%':>10} {'median':>8}")
    for md in (4, 5):
        for k in (1.0, 1.5, 2.0):
            d, _ = run_cycle(df, FC["champ_lo"], max_days=md, tp_atr=k)
            challenged(d, f"hold<={md}d  TP {k:.1f}xATR")
        for pct in (0.01, 0.015):
            d, _ = run_cycle(df, FC["champ_lo"], max_days=md, tp_pct=pct)
            challenged(d, f"hold<={md}d  TP {pct * 100:.1f}%")
    d0, _ = run_cycle(df, FC["champ_lo"], max_days=5)
    challenged(d0, "hold<=5d  no target (baseline)")

    print("\nnetSR is after real spread+slippage; grossSR is per-trade before cost.")
    print("A weekly cycle pays ~52 round trips/yr — the gap between the two is the toll.")
