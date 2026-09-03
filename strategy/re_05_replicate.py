"""STAGE 5 — build the replica from the fingerprint and backtest it on OLDER data.

WHAT THE FORENSICS ESTABLISHED (stages 1-4, all measured, none assumed):
  * log is genuine: balance reconciles to the cent ($18,251.80); vendor log runs +2h vs
    our feed (at -120min, 85.3% of fills land inside their own M15 bar, median miss $0.00).
  * fixed 0.10 lots, one position at a time, no grid/martingale.
  * ENTRY TIMING carries NO bar-detectable information: against hour-and-date-matched
    control bars, every feature separates at |t| <= 2.6, and the two that come closest
    (atr_pctl, bar_range_atr) point to slightly CALMER conditions, not a trigger.
  * ENTRY DIRECTION is momentum, overwhelmingly: buys at 20-bar range-position 0.76 and
    z20 +0.95, sells at 0.35 and -0.83 (t ~ +30 on both); intrabar fills at 0.74 of the
    bar for buys and 0.31 for sells. It buys strength and sells weakness.
  * EXIT is sub-second and is where the performance lives: it banks 27% of the favourable
    excursion while absorbing 2% of the adverse one, cutting losers 17.4x earlier than the
    bar range implies. The DECLARED bracket (SL 2.4 / TP 10) is not the real exit — applied
    as stated to the vendor's own entries it returns -$14,702 at a 7.9% win rate.
  * on the vendor's own entries with neutral exits: +15m +$9,438, +30m +$11,590,
    +120m -$2,229. Entry edge at 15m is real but weak and tail-driven (t=+2.33, hit 48.5%).

THE REPLICA therefore reproduces what is reproducible — the momentum DIRECTION rule at a
comparable trade rate inside the log's active hours — and tests it against a range of
bar-implementable exits. It cannot reproduce a sub-second exit, and the point of this
stage is to quantify that gap honestly rather than curve-fit an equity curve to match.

Backtested on the vendor window AND on 2015-2025 (true out-of-sample, ~10 years).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

LOTS = 0.10
CONTRACT = 100.0
SPREAD = 0.44          # measured live XAUUSD spread, charged once per round trip
ACTIVE_HOURS = range(1, 17)   # log hours 3..18 in vendor time = 1..16 in feed time (-2h)
VENDOR0, VENDOR1 = "2025-10-15", "2026-04-14"


def load_m15() -> pd.DataFrame:
    px = pd.read_csv(ROOT / "data" / "XAUUSD_M15_long.csv",
                     parse_dates=["time"], index_col="time").sort_index()
    px = px[~px.index.duplicated(keep="last")]
    c, h, l = px.close, px.high, px.low
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    px["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    m, s = c.rolling(20).mean(), c.rolling(20).std()
    px["z20"] = (c - m) / s
    hi, lo = h.rolling(20).max(), l.rolling(20).min()
    px["rng_pos20"] = (c - lo) / (hi - lo).replace(0, np.nan)
    px["mom8"] = (c - c.shift(8)) / px["atr"]
    return px.dropna(subset=["z20", "atr", "rng_pos20", "mom8"])


def signals(px: pd.DataFrame, z_thr: float, rng_thr: float) -> pd.Series:
    """The fingerprinted direction rule: long on strength, short on weakness.
    Decided on a COMPLETED bar, acted on at the NEXT bar's open."""
    long = (px.z20 >= z_thr) & (px.rng_pos20 >= rng_thr) & (px.mom8 > 0)
    short = (px.z20 <= -z_thr) & (px.rng_pos20 <= 1 - rng_thr) & (px.mom8 < 0)
    s = pd.Series(0.0, index=px.index)
    s[long] = 1.0
    s[short] = -1.0
    s[~px.index.hour.isin(list(ACTIVE_HOURS))] = 0.0
    s[px.index.dayofweek >= 5] = 0.0
    return s


def backtest(px: pd.DataFrame, sig: pd.Series, exit_mode: str,
             hold_bars: int = 2, sl: float = 2.4, tp: float = 10.0,
             max_per_day: int = 5) -> pd.DataFrame:
    """One position at a time, entry at next bar open, exits are all
    bar-implementable. `first_profit` is the closest honest approximation of the
    vendor's 'bank any favourable move' behaviour that M15 data can express."""
    o, h, l, c = (px[k].values for k in ("open", "high", "low", "close"))
    idx, sg = px.index, sig.values
    n = len(px)
    trades = []
    i, day, cnt = 0, None, 0
    while i < n - 1:
        if sg[i] == 0:
            i += 1
            continue
        d0 = idx[i].date()
        if d0 != day:
            day, cnt = d0, 0
        if cnt >= max_per_day:
            i += 1
            continue
        s = sg[i]
        ent = o[i + 1] + s * SPREAD / 2.0
        sl_px, tp_px = ent - s * sl, ent + s * tp
        ex_px, ex_i, reason = None, None, ""
        for k in range(i + 1, min(i + 1 + hold_bars, n)):
            if exit_mode in ("bracket", "atr_bracket"):
                hit_sl = (l[k] <= sl_px) if s > 0 else (h[k] >= sl_px)
                hit_tp = (h[k] >= tp_px) if s > 0 else (l[k] <= tp_px)
                if hit_sl:
                    ex_px, ex_i, reason = sl_px, k, "sl"; break
                if hit_tp:
                    ex_px, ex_i, reason = tp_px, k, "tp"; break
            elif exit_mode == "first_profit":
                if (c[k] - ent) * s > 0:
                    ex_px, ex_i, reason = c[k], k, "profit"; break
        if ex_px is None:
            ex_i = min(i + hold_bars, n - 1)
            ex_px, reason = c[ex_i], "timeout"
        gross = (ex_px - ent) * s - SPREAD / 2.0
        trades.append(dict(t=idx[i + 1], dir=s, entry=ent, exit=ex_px, reason=reason,
                           move=gross, profit=gross * LOTS * CONTRACT))
        cnt += 1
        i = ex_i + 1
    return pd.DataFrame(trades)


def summarise(tr: pd.DataFrame, label: str, days: float) -> dict:
    if tr.empty:
        return dict(label=label, n=0)
    p = tr.profit
    eq = 3000 + p.cumsum()
    dd = float((eq / eq.cummax() - 1).min() * 100)
    return dict(label=label, n=len(tr), per_day=len(tr) / days,
                win=float((p > 0).mean() * 100), mean=p.mean(), total=p.sum(),
                payoff=abs(p[p > 0].mean() / p[p < 0].mean()) if (p < 0).any() else np.nan,
                dd=dd, gain_pct=float(p.sum() / 3000 * 100))


def main() -> None:
    px = load_m15()
    print("VENDOR BENCHMARK (what we are trying to replicate)")
    print("  773 trades / 181d = 4.27 per day | win 83.2% | payoff 10.6 | "
          "mean $19.73 | total $15,252 | +508% | balance-DD -0.18%\n")

    grids = [(0.8, 0.60), (1.2, 0.70), (1.6, 0.80), (2.0, 0.85)]
    exits = [("first_profit", dict(exit_mode="first_profit", hold_bars=4)),
             ("hold 30m", dict(exit_mode="timeout", hold_bars=2)),
             ("hold 60m", dict(exit_mode="timeout", hold_bars=4)),
             ("declared SL2.4/TP10", dict(exit_mode="bracket", hold_bars=8, sl=2.4, tp=10.0)),
             ("tight SL1/TP2", dict(exit_mode="bracket", hold_bars=8, sl=1.0, tp=2.0))]

    for wname, (w0, w1) in (("VENDOR WINDOW 2025-10-15..2026-04-14", (VENDOR0, VENDOR1)),
                            ("OUT-OF-SAMPLE 2015..2025-10-14", ("2015-01-01", "2025-10-14"))):
        sub = px.loc[w0:w1]
        days = max((sub.index[-1] - sub.index[0]).days, 1)
        print(f"{'='*100}\n{wname}   ({len(sub):,} M15 bars, {days} days)\n{'='*100}")
        print(f"{'z_thr':>6} {'rngp':>5} {'exit':22} {'n':>5} {'t/day':>6} {'win%':>6} "
              f"{'payoff':>7} {'mean$':>8} {'total$':>10} {'gain%':>8} {'DD%':>7}")
        for z_thr, rng_thr in grids:
            sig = signals(sub, z_thr, rng_thr)
            for ename, kw in exits:
                tr = backtest(sub, sig, **kw)
                s = summarise(tr, ename, days)
                if s["n"] == 0:
                    print(f"{z_thr:>6.1f} {rng_thr:>5.2f} {ename:22} {'0':>5}")
                    continue
                print(f"{z_thr:>6.1f} {rng_thr:>5.2f} {ename:22} {s['n']:>5} "
                      f"{s['per_day']:>6.2f} {s['win']:>6.1f} "
                      f"{(s['payoff'] if np.isfinite(s['payoff']) else 0):>7.2f} "
                      f"{s['mean']:>8.2f} {s['total']:>10.0f} {s['gain_pct']:>8.1f} "
                      f"{s['dd']:>7.1f}")
        print()


if __name__ == "__main__":
    main()
