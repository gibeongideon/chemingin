#!/usr/bin/env python3
"""What actually happens at the hand-over between trading sessions?

THE ASK. "Study what usually happens transitioning from one session to another — Asian, London,
etc. If we can build a strategy around the transition, better." Descriptive first, strategy only
if the description supports one.

*** SESSIONS ARE DEFINED IN LOCAL MARKET TIME, DST-AWARE, NOT IN FIXED UTC HOURS. ***
London and New York shift with their own daylight rules, and those rules do NOT move together —
the US changes two to three weeks before Europe each spring and a week after each autumn. A
fixed-UTC session boundary is therefore WRONG for several weeks a year, always at the hand-over,
which is exactly the minutes this study is about. This repo has already paid for that mistake
once (§3be's session-DST correction of 1h).

BLOCKS (non-overlapping, so a bar belongs to exactly one):
    ASIA     London local < 08:00                      the quiet book
    LONDON   London >= 08:00, before the NY open       Europe alone
    OVERLAP  NY local 09:30 .. London local 16:30      both books open — the deepest hours
    NY       London closed, NY local < 17:00           America alone
    LATE     everything after                          the thin tail

WHAT IS ALREADY CLOSED, so that this does not re-run it:
  * SESSION OPENING-RANGE BREAKOUT — §3be, and the correction in §3bx-D. The "real but sub-cost"
    verdict was a COST BUG; the honest p is 0.8425. Do not re-test an ORB.
  * THE 08:30 ET RELEASE — §3bg. That bar is 2.3x a normal one and 0 of 72 cells clear cost.
  * Any M15 entry — §3cp: every M15 bucket loses net, and the hit rate points the opposite way
    to the EV.

WHAT IS GENUINELY OPEN, and is what this measures: the RELATION BETWEEN CONSECUTIVE SESSIONS.
Does London continue Asia or fade it? Does the overlap continue London? That is a cross-session
momentum/reversal question, not a breakout, and it has not been asked here.

CONTROLS. #2 every conditional number is printed beside the unconditional one for the same
block — a session that rises 55% of the time is worthless if it rises 55% of the time anyway.
#1 cost at the live round trip. #4 the unit is the DAY, not the bar. Both halves everywhere.

    python scripts/v5_session_transitions.py --port 18815 --symbol XAUUSDs
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "v5_runs"
LON, NYC = ZoneInfo("Europe/London"), ZoneInfo("America/New_York")
SERVER_UTC_OFFSET = 3                      # MT5 server clock runs UTC+3
ORDER = ["ASIA", "LONDON", "OVERLAP", "NY", "LATE"]


def fetch(m, sym: str, n: int) -> pd.DataFrame:
    r = m.copy_rates_from_pos(sym, 15, 0, n)          # M15
    if r is None:
        return pd.DataFrame()
    d = pd.DataFrame([(x[0], x[1], x[2], x[3], x[4], x[5]) for x in r],
                     columns=["ts", "open", "high", "low", "close", "tick_volume"])
    d["time"] = pd.to_datetime(d.ts, unit="s")
    return d.set_index("time").drop(columns=["ts"]).iloc[:-1]


def label(idx: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """(session block, trading day) per bar.

    *** WHY THIS IS DONE IN UTC WITH DST-DERIVED BOUNDARIES, NOT IN LOCAL HOURS. ***
    The first version compared each bar's LOCAL London and New York hour directly, which looks
    natural and is wrong, because the local hour WRAPS THROUGH MIDNIGHT. A bar at 23:00 UTC in
    British Summer Time is 00:00 in London, so `london_hour < 16.5` was TRUE and
    `new_york_hour >= 9.5` was also TRUE, and the bar was labelled OVERLAP. The overlap block
    then ran 13:30 to 23:45 instead of 13:30 to 15:30 — swallowing the whole New York session.
    The tell was in the output and is why this was caught: OVERLAP had 983 days where every
    other block had 873, and LONDON->OVERLAP printed t = 14.54, which is not a number an
    honest session study produces.

    Converting the market opens INTO UTC instead keeps every comparison on one monotonic clock,
    where no wrap is possible. DST is still respected — the boundaries move with each market's
    own offset, and those offsets do not move together (the US switches two to three weeks
    before Europe in spring).

    The trading DAY is shifted +2h so it starts at 22:00 UTC, matching the server day
    (01:00-23:59 at UTC+3) and gold's 21:00-22:00 UTC break. Grouping on the plain UTC date
    would cut the Sydney open away from the Asian session that continues it.
    """
    utc = pd.DatetimeIndex(idx - pd.Timedelta(hours=SERVER_UTC_OFFSET))
    u = utc.tz_localize("UTC")
    lon_off = np.array([x.utcoffset().total_seconds() / 3600.0 for x in u.tz_convert(LON)])
    ny_off = np.array([x.utcoffset().total_seconds() / 3600.0 for x in u.tz_convert(NYC)])

    lon_open = 8.0 - lon_off           # London 08:00 local, expressed in UTC
    lon_close = 16.5 - lon_off         # London 16:30 local
    ny_open = 9.5 - ny_off             # New York 09:30 local
    ny_close = 17.0 - ny_off           # New York 17:00 local
    uh = utc.hour + utc.minute / 60.0

    out = np.full(len(idx), "LATE", dtype=object)
    out[(uh >= 22.0) | (uh < lon_open)] = "ASIA"          # Sydney/Tokyo through the London open
    out[(uh >= lon_open) & (uh < ny_open)] = "LONDON"
    out[(uh >= ny_open) & (uh < lon_close)] = "OVERLAP"
    out[(uh >= lon_close) & (uh < ny_close)] = "NY"
    day = pd.Series((utc + pd.Timedelta(hours=2)).date, index=idx)
    return pd.Series(out, index=idx, name="block"), day


def blocks(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (trading day, session block)."""
    b, day = label(df.index)
    g = df.groupby([day, b])
    out = pd.DataFrame(dict(open=g["open"].first(), high=g["high"].max(), low=g["low"].min(),
                            close=g["close"].last(), vol=g["tick_volume"].sum(),
                            bars=g["close"].size()))
    out.index.names = ["day", "block"]
    out = out.reset_index()
    out["ret"] = (out["close"] / out["open"] - 1.0) * 1e4           # bp, open-to-close
    out["rng"] = (out["high"] - out["low"]) / out["open"] * 1e4     # bp, high-low
    return out[out["bars"] >= 4]                                     # at least an hour of tape


def halves(x: pd.Series) -> tuple[float, float]:
    h = len(x) // 2
    return (float(x.iloc[:h].mean()), float(x.iloc[h:].mean())) if h > 2 else (np.nan, np.nan)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18815)
    ap.add_argument("--symbol", default="XAUUSDs")
    ap.add_argument("--bars", type=int, default=200_000)
    ap.add_argument("--rt-bp", type=float, default=1.08, help="live round trip")
    a = ap.parse_args()

    from mt5linux import MetaTrader5
    m, last = None, None
    for i in range(3):
        if i:
            time.sleep(3 * i)
        m = MetaTrader5(host="localhost", port=a.port)
        if m.initialize():
            break
        last = m.last_error()
    else:
        print("bridge init failed:", last); return 2
    m.symbol_select(a.symbol, True)
    df = fetch(m, a.symbol, a.bars)
    m.shutdown()
    if df.empty:
        print("no bars"); return 1
    print(f"{len(df):,} M15 bars  {df.index[0]}  ..  {df.index[-1]}  (server, UTC+3)\n")

    # ---- STRUCTURAL CHECK, before any statistic is computed ---------------------------------
    # The wrap bug made OVERLAP span ten hours instead of two and it was only visible as a
    # suspiciously good t-stat. Assert the shape of the blocks instead of trusting it: each one
    # must occupy roughly its intended width, and the blocks must appear in clock order within
    # a day. A study whose buckets are wrong cannot be rescued by careful statistics downstream.
    b_chk, day_chk = label(df.index)
    utc_chk = pd.DatetimeIndex(df.index - pd.Timedelta(hours=SERVER_UTC_OFFSET))
    span = pd.DataFrame({"b": b_chk.values, "h": utc_chk.hour + utc_chk.minute / 60.0})
    print("   block spans (UTC hours, median first/last bar and bar count per day):")
    bad = []
    for k in ORDER:
        s = span[span.b == k]["h"]
        if s.empty:
            continue
        per_day = span[span.b == k].groupby(day_chk[b_chk == k].values).size()
        width = float(per_day.median()) * 0.25                  # M15 bars -> hours
        print(f"     {k:<9} {s.min():5.2f} .. {s.max():5.2f}   {width:4.1f}h/day")
        if k == "OVERLAP" and width > 4.0:
            bad.append(f"{k} spans {width:.1f}h — it should be ~2h; a wrap bug is back")
        if k in ("LONDON", "NY") and width > 8.0:
            bad.append(f"{k} spans {width:.1f}h — too wide")
    if bad:
        print("\n*** REFUSING TO REPORT: " + "; ".join(bad))
        return 1
    print()

    B = blocks(df)
    piv = B.pivot_table(index="day", columns="block", values="ret")
    rng = B.pivot_table(index="day", columns="block", values="rng")
    vol = B.pivot_table(index="day", columns="block", values="vol")
    piv = piv.sort_index()

    # ---------- 1. WHAT EACH SESSION IS, UNCONDITIONALLY --------------------------------------
    # *** NO VOLUME COLUMN. *** This feed reports tick_volume == 0 on 60.2% of M15 bars,
    # uniformly across every UTC hour, while 48,128 of those 48,133 bars have a non-zero
    # high-low range — price moved and the volume was simply not recorded. A "share of daily
    # volume" computed on that column is a share of an arbitrary 40% sample, so it is not
    # printed. RANGE is the honest activity measure here and is shown instead.
    zero_share = float((df["tick_volume"] == 0).mean())
    print("1. THE SESSIONS THEMSELVES  (bp, open-to-close; range is high-low)")
    print(f"   [tick_volume is 0 on {zero_share:.1%} of bars on this feed — volume columns "
          f"are omitted as unusable]")
    print(f"   {'block':<9}{'days':>7}{'mean ret':>10}{'median':>9}{'P(up)':>8}"
          f"{'mean rng':>10}{'rng share':>11}")
    tot_rng = float(np.nansum(rng.to_numpy()))
    for k in ORDER:
        if k not in piv:
            continue
        r = piv[k].dropna()
        share = float(np.nansum(rng[k].to_numpy())) / tot_rng if k in rng else np.nan
        print(f"   {k:<9}{len(r):>7}{r.mean():>10.2f}{r.median():>9.2f}"
              f"{(r > 0).mean():>8.1%}{rng[k].mean():>10.1f}{share:>10.1%}")

    # ---------- 2. THE TRANSITION: does the next block CONTINUE or FADE the last? -------------
    print(f"\n2. THE HAND-OVER — next block's return, split by the sign of the one before")
    print("   (CONTROL #2: 'all' is the unconditional mean for that block. A conditional number")
    print("    only means something as a DIFFERENCE from it.)")
    print(f"   {'transition':<20}{'n':>6}{'after UP':>10}{'after DN':>10}{'all':>9}"
          f"{'spread':>9}{'t':>7}{'1st/2nd half':>18}")
    pairs = [("ASIA", "LONDON"), ("LONDON", "OVERLAP"), ("OVERLAP", "NY"), ("ASIA", "OVERLAP"),
             ("LONDON", "NY")]
    rows = []
    for a_, b_ in pairs:
        if a_ not in piv or b_ not in piv:
            continue
        d = piv[[a_, b_]].dropna()
        if len(d) < 200:
            continue
        up, dn = d[d[a_] > 0][b_], d[d[a_] <= 0][b_]
        allm = float(d[b_].mean())
        spread = float(up.mean() - dn.mean())
        # two-sample t on the DIFFERENCE; the unit is the day and days do not overlap
        se = np.sqrt(up.var(ddof=1) / len(up) + dn.var(ddof=1) / len(dn))
        t = spread / se if se > 0 else np.nan
        d2 = d.copy()
        d2["sig"] = np.where(d2[a_] > 0, d2[b_], -d2[b_])     # continuation P&L, sign-flipped
        h1, h2 = halves(d2["sig"])
        rows.append((f"{a_}->{b_}", len(d), float(up.mean()), float(dn.mean()), allm, spread, t,
                     h1, h2))
        print(f"   {a_+'->'+b_:<20}{len(d):>6}{up.mean():>10.2f}{dn.mean():>10.2f}{allm:>9.2f}"
              f"{spread:>9.2f}{t:>7.2f}   {h1:>7.2f} /{h2:>7.2f}")

    print("\n   'spread' is what a CONTINUATION trade would capture GROSS: go with the prior")
    print(f"   session's sign and you earn (after UP - after DN)/2 per trade, before the "
          f"{a.rt_bp:.2f}bp round trip.")
    for name, n, u_, dn_, allm, sp, t, h1, h2 in rows:
        gross = sp / 2.0
        print(f"   {name:<20} gross {gross:+7.2f}bp   net {gross - a.rt_bp:+7.2f}bp   "
              f"halves {h1/2 if h1 == h1 else float('nan'):+7.2f} /"
              f"{h2/2 if h2 == h2 else float('nan'):+7.2f}")

    # ---------- 3. THE TRANSITION HOUR ITSELF -------------------------------------------------
    print(f"\n3. THE FIRST HOUR OF EACH BLOCK vs THE REST OF IT  (mean |ret| per M15 bar, bp)")
    b, day = label(df.index)
    ret15 = (df["close"] / df["open"] - 1.0) * 1e4
    first = df.groupby([day, b]).cumcount() < 4
    print(f"   {'block':<9}{'first hour':>12}{'rest':>9}{'ratio':>8}{'first mean ret':>16}")
    for k in ORDER:
        sel = (b == k)
        if sel.sum() < 500:
            continue
        f_, r_ = ret15[sel & first].abs(), ret15[sel & ~first].abs()
        print(f"   {k:<9}{f_.mean():>12.2f}{r_.mean():>9.2f}{f_.mean()/r_.mean():>8.2f}"
              f"{ret15[sel & first].mean():>16.3f}")
    print("   a high ratio means the OPEN is violent, which is a VOLATILITY fact and says")
    print("   nothing about direction — 'first mean ret' is the directional part (§3bg: the")
    print("   08:30 ET bar is 2.3x a normal one and 0 of 72 cells clear cost).")

    # ---------- 4. THE ONLY COMPARISON THAT DECIDES ANYTHING ----------------------------------
    # Holding gold all day earns the SUM of the blocks and pays the round trip once, ever. A
    # session strategy earns one block and pays the round trip EVERY DAY. Any session that does
    # not clear the full-day drift by more than a daily round trip is a worse way to own gold.
    day_sum = piv[[k for k in ORDER if k in piv]].sum(axis=1, min_count=1).dropna()
    hold = float(day_sum.mean())
    h1, h2 = halves(day_sum)
    print(f"\n4. EACH SESSION AS A STRATEGY, AGAINST SIMPLY HOLDING  (CONTROL #2)")
    print(f"   holding all day: {hold:+.2f}bp/day  (halves {h1:+.2f} / {h2:+.2f}), "
          f"and it pays the spread ONCE, not daily")
    print(f"   {'long only in':<14}{'gross':>9}{'net of 1 RT':>13}{'vs HOLD':>10}"
          f"{'t vs hold':>11}{'halves':>18}")
    for k in ORDER:
        if k not in piv:
            continue
        r = piv[k].dropna()
        d = (piv[k] - day_sum).dropna()                   # paired: same days, both legs
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        net = float(r.mean()) - a.rt_bp
        k1, k2 = halves(r)
        print(f"   {k:<14}{r.mean():>9.2f}{net:>13.2f}{net - hold:>10.2f}"
              f"{(net - hold) / se if se else float('nan'):>11.2f}"
              f"   {k1 - a.rt_bp:>7.2f} /{k2 - a.rt_bp:>7.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    piv.to_csv(OUT / "session_blocks.csv")
    print(f"\nwrote {OUT / 'session_blocks.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
