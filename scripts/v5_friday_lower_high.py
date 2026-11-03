"""§3cx  FRIDAY LOWER-HIGH -> MONDAY VISITS FRIDAY'S LOW. User hypothesis, 2026-11-03:

    "If Friday's high is not as high as Thursday's high, the odds are overwhelmingly high that
     whatever low you made on Friday will be visited on Monday during the regular session."

DATA. Switch's OWN H1 feed for both symbols (data/{XAUUSDs,BTCUSDs}_H1_switch.csv, pulled
2026-11-03, one broker, the 60,000-bar terminal cap: gold 2016-09+, BTC 2019-03+). Server clock
is New York + 7h all year (verified: gold's daily break sits at server hour 0 in January AND
July), so a server DATE is the NY 17:00->17:00 trading day and the NY regular session
09:00-16:00 is server hours 16..22.

WHAT "OVERWHELMINGLY HIGH" HAS TO BEAT. A hit rate on its own is not evidence; the controls:
  1. the COMPLEMENT: the same outcome on weeks where Friday did NOT make a lower high.
  2. DISTANCE. A lower-high Friday is usually a down Friday that closes near its low, and a
     low that is close is touched more often for that reason alone. Two answers:
       a. stratify by (Fri close - Fri low)/ATR and compare inside each bin, plus a logit
          Y ~ C + distance + Monday gap;
       b. a MATCHED-DISTANCE PLACEBO: for each week, the probability that ANY level that far
          below Monday's open gets touched, from the empirical Monday excursion distribution.
          If the Friday low is special it must beat a placebo level at the same distance.
  3. WEEKDAY PLACEBO: the identical rule on Mon/Tue->Wed, Tue/Wed->Thu, Wed/Thu->Fri. If those
     match, the weekend is not part of the story.
  4. MONEY: short at the window open, take profit AT Friday's low, otherwise exit at the window
     close, net of the live Switch spread ($0.25 gold, $32.5 BTC, measured 2026-11-03). Compared
     against the same trade on complement weeks, because gold drifts UP and a short has to beat
     that (memory: drift-baseline-inverts-3r).

    python scripts/v5_friday_lower_high.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v5_runs" / "friday_lower_high"
NY_HOURS = range(16, 23)                     # server 16:00..22:59 = NY 09:00..15:59
ASSETS = {"XAUUSD": dict(file="data/XAUUSDs_H1_switch.csv", spread=0.25, min_bars=18),
          "BTCUSD": dict(file="data/BTCUSDs_H1_switch.csv", spread=32.5, min_bars=22)}


def load(path: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / path, parse_dates=["time"], index_col="time")
    return d[~d.index.duplicated(keep="last")].sort_index()


def daily(h1: pd.DataFrame, min_bars: int) -> pd.DataFrame:
    g = h1.groupby(h1.index.normalize())
    d = pd.DataFrame(dict(open=g["open"].first(), high=g["high"].max(), low=g["low"].min(),
                          close=g["close"].last(), n=g["close"].size()))
    d = d[d.n >= min_bars]
    d = d.iloc[:-1]                                       # today's bar is still forming
    pc = d["close"].shift(1)
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()  # known at that day's close
    return d


def window(h1: pd.DataFrame, day: pd.Timestamp, hours=None) -> pd.DataFrame:
    x = h1[(h1.index >= day) & (h1.index < day + pd.Timedelta(days=1))]
    return x[x.index.hour.isin(hours)] if hours is not None else x


def events(h1: pd.DataFrame, d: pd.DataFrame, a_off: int, b_off: int, t_off: int,
           weekday_a: int, is_btc: bool) -> pd.DataFrame:
    """Day A (e.g. Thu), day B = A + b_off (Fri), target day T = A + t_off (Mon). Calendar
    offsets, so a holiday on any of the three drops the week rather than shifting it."""
    rows = []
    for a in d.index[d.index.dayofweek == weekday_a]:
        b, t = a + pd.Timedelta(days=b_off), a + pd.Timedelta(days=t_off)
        if b not in d.index or t not in d.index:
            continue
        A, B = d.loc[a], d.loc[b]
        full, ny = window(h1, t), window(h1, t, NY_HOURS)
        if len(full) < 10 or len(ny) < 5:
            continue
        between = h1[(h1.index >= b + pd.Timedelta(days=1)) & (h1.index < t)]
        rows.append(dict(
            week=t, C=bool(B.high < A.high), atr=B.atr, b_low=B.low, b_close=B.close,
            b_down=bool(B.close < B.open),
            dist=(B.close - B.low) / B.atr,
            gap=(full.open.iloc[0] - B.close) / B.atr,
            pre_hit=bool(len(between) and between.low.min() <= B.low),  # BTC weekend
            t_open=full.open.iloc[0], t_close=full.close.iloc[-1], t_low=full.low.min(),
            ny_open=ny.open.iloc[0], ny_close=ny.close.iloc[-1], ny_low=ny.low.min(),
            pre_ny_low=full[full.index.hour < 16].low.min() if (full.index.hour < 16).any() else np.nan))
    e = pd.DataFrame(rows).set_index("week")
    e["Y_day"] = e.t_low <= e.b_low
    e["Y_ny"] = e.ny_low <= e.b_low
    return e


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def rate_row(e, col):
    out = {}
    for lab, s in (("C", e[e.C]), ("notC", e[~e.C]), ("all", e)):
        k, n = int(s[col].sum()), len(s)
        lo, hi = wilson(k, n)
        out[lab] = (k, n, k / n if n else np.nan, lo, hi)
    tab = [[out["C"][0], out["C"][1] - out["C"][0]], [out["notC"][0], out["notC"][1] - out["notC"][0]]]
    out["fisher_p"] = stats.fisher_exact(tab)[1]
    return out


def fmt(o):
    k, n, p, lo, hi = o
    return f"{100*p:5.1f}% [{100*lo:4.1f},{100*hi:4.1f}] n={n}"


def placebo_expected(e: pd.DataFrame, open_col: str, low_col: str) -> pd.Series:
    """Matched-distance placebo. Excursion x = (open - window low)/ATR over ALL weeks. For a
    week whose Friday low sits d = (open - Fri low)/ATR below the open, the placebo probability
    is P(x >= d) — how often a level that far down gets touched regardless of what it is.
    Leave-one-out so a week never scores itself."""
    x = ((e[open_col] - e[low_col]) / e.atr).values
    d = ((e[open_col] - e.b_low) / e.atr).values
    n = len(x)
    srt = np.sort(x)
    ge = n - np.searchsorted(srt, d, side="left")           # count x >= d
    own = (x >= d).astype(int)
    return pd.Series(np.where(d <= 0, 1.0, (ge - own) / (n - 1)), index=e.index)


def trade(e: pd.DataFrame, open_col: str, close_col: str, low_col: str, spread: float) -> pd.Series:
    """Short at the window open, TP at Fri low, else out at the window close. bp of price, net.
    Weeks already through the low at the open are excluded: there is nothing to sell into."""
    ok = e[open_col] > e.b_low
    s = e[ok]
    exit_px = np.where(s[low_col] <= s.b_low, s.b_low, s[close_col])
    return pd.Series((s[open_col] - exit_px - spread) / s[open_col] * 1e4, index=s.index)


def tstat(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std() * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []

    def p(s=""):
        print(s); lines.append(s)

    for name, cfg in ASSETS.items():
        h1 = load(cfg["file"]); d = daily(h1, cfg["min_bars"]); btc = name == "BTCUSD"
        e = events(h1, d, 0, 1, 4, 3, btc)                    # Thu, Fri, Mon
        e.to_csv(OUT / f"{name}_events.csv")
        p("=" * 100)
        p(f"{name}   {e.index[0].date()} .. {e.index[-1].date()}   {len(e)} Thu/Fri/Mon weeks   "
          f"C = Fri high < Thu high on {e.C.mean():.1%}")
        p("=" * 100)

        p("\n1. HIT RATE — Monday trades at or below Friday's low")
        for col, lab in (("Y_day", "any time Monday"), ("Y_ny", "Monday NY session 09-16")):
            r = rate_row(e, col)
            p(f"   {lab:26s} C {fmt(r['C'])}   notC {fmt(r['notC'])}   all {100*r['all'][2]:.1f}%"
              f"   diff {100*(r['C'][2]-r['notC'][2]):+5.1f}pp  Fisher p {r['fisher_p']:.3g}")
        if btc:
            r = rate_row(e, "pre_hit")
            p(f"   {'already hit Sat/Sun':26s} C {fmt(r['C'])}   notC {fmt(r['notC'])}")
            ee = e[~e.pre_hit]
            r = rate_row(ee, "Y_day")
            p(f"   {'Monday, weekend missed':26s} C {fmt(r['C'])}   notC {fmt(r['notC'])}"
              f"   Fisher p {r['fisher_p']:.3g}")
        already = (e.t_open <= e.b_low)
        p(f"   Monday OPENS at/below Fri low: C {already[e.C].mean():.1%}  notC {already[~e.C].mean():.1%}"
          f"  (those count as 'visited' with no move at all)")
        nyfresh = e[(e.pre_ny_low > e.b_low)]
        r = rate_row(nyfresh, "Y_ny")
        p(f"   NY session, NOT already hit before 09:00 NY:  C {fmt(r['C'])}   notC {fmt(r['notC'])}"
          f"   Fisher p {r['fisher_p']:.3g}")

        p("\n2. DISTANCE CONTROL — is it the lower high, or just that the low is close?")
        p(f"   median (Fri close - Fri low)/ATR:  C {e[e.C].dist.median():.2f}   notC {e[~e.C].dist.median():.2f}"
          f"   Fri DOWN-close: C {e[e.C].b_down.mean():.1%}  notC {e[~e.C].b_down.mean():.1%}")
        e["dq"] = pd.qcut(e.dist, 4, labels=["Q1 near", "Q2", "Q3", "Q4 far"])
        for q, s in e.groupby("dq", observed=True):
            c, nc = s[s.C], s[~s.C]
            p(f"   {q:8s} dist {s.dist.min():.2f}-{s.dist.max():.2f}   C {c.Y_day.mean():5.1%} (n={len(c):3d})"
              f"   notC {nc.Y_day.mean():5.1%} (n={len(nc):3d})   diff {100*(c.Y_day.mean()-nc.Y_day.mean()):+5.1f}pp")
        X = sm.add_constant(pd.DataFrame(dict(C=e.C.astype(float), dist=e.dist, gap=e.gap)))
        try:
            lg = sm.Logit(e.Y_day.astype(float), X).fit(disp=0)
            p(f"   logit Y_day ~ C + dist + gap:  C coef {lg.params['C']:+.3f} (z {lg.tvalues['C']:+.2f}, "
              f"p {lg.pvalues['C']:.3g})   dist z {lg.tvalues['dist']:+.2f}   gap z {lg.tvalues['gap']:+.2f}")
        except Exception as exc:                                # noqa: BLE001
            p(f"   logit failed: {exc}")
        for col, oc, lc, lab in (("Y_day", "t_open", "t_low", "any time Monday"),
                                 ("Y_ny", "ny_open", "ny_low", "NY session")):
            ex = placebo_expected(e, oc, lc)
            for grp, m in (("C", e.C), ("notC", ~e.C)):
                act, exp = e.loc[m, col].mean(), ex[m].mean()
                z = (e.loc[m, col].astype(float) - ex[m]).sum() / np.sqrt((ex[m] * (1 - ex[m])).sum())
                p(f"   placebo {lab:16s} {grp:5s} actual {act:5.1%}  vs a level the SAME distance away "
                  f"{exp:5.1%}   excess {100*(act-exp):+5.1f}pp  z {z:+.2f}")

        p("\n3. WEEKDAY PLACEBO — same rule, no weekend in between (day A high > day B high -> day C)")
        for wa, lab in ((0, "Mon/Tue -> Wed"), (1, "Tue/Wed -> Thu"), (2, "Wed/Thu -> Fri")):
            w = events(h1, d, 0, 1, 2, wa, btc)
            r = rate_row(w, "Y_day")
            p(f"   {lab:15s} C {fmt(r['C'])}   notC {fmt(r['notC'])}   diff "
              f"{100*(r['C'][2]-r['notC'][2]):+5.1f}pp")

        p("\n4. MONEY — short at the open, TP at Fri low, else exit at the close (net of "
          f"${cfg['spread']} spread)")
        for oc, cc, lc, lab in (("t_open", "t_close", "t_low", "Monday day"),
                                ("ny_open", "ny_close", "ny_low", "Monday NY session")):
            tc, tn = trade(e[e.C], oc, cc, lc, cfg["spread"]), trade(e[~e.C], oc, cc, lc, cfg["spread"])
            hold = (e[cc] - e[oc]) / e[oc] * 1e4
            p(f"   {lab:18s} C: {tc.mean():+6.1f}bp/trade (t {tstat(tc):+.2f}, n={len(tc)}, win {(tc>0).mean():.0%}, "
              f"total {tc.sum():+.0f}bp)   notC: {tn.mean():+6.1f}bp (t {tstat(tn):+.2f})   "
              f"LONG same window all weeks {hold.mean():+5.1f}bp")
            tt = stats.ttest_ind(tc, tn, equal_var=False)
            p(f"   {'':18s} C minus notC: {tc.mean()-tn.mean():+.1f}bp  Welch t {tt.statistic:+.2f}")
            if lab == "Monday day":
                yr = tc.groupby(tc.index.year).mean()
                p("   per-year C bp: " + "  ".join(f"{y}:{v:+.0f}" for y, v in yr.items()))
                h = len(tc) // 2
                p(f"   halves: first {tc.iloc[:h].mean():+.1f}bp  second {tc.iloc[h:].mean():+.1f}bp")
        p("")
    (OUT / "report.txt").write_text("\n".join(lines))
    print(f"wrote {OUT.relative_to(ROOT)}/report.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
