#!/usr/bin/env python3
"""OCTOBER E27 — the move-to-breakeven stop, isolated. Requested 2026-10-30.

The user's spec was: catch an intraday trend, tight initial SL, **move the SL to breakeven once
the trade reaches breakeven**, then trail slowly, no take-profit, manage in real time. Five of
those six elements are already measured here (OCTOBER.MD §4c). **The move-to-breakeven is the one
thing this repo has never tested**, so it is the only thing this script varies.

EVERYTHING ELSE IS PINNED AT ITS VALIDATED SETTING, so the comparison is clean:
  * long-only (§3bn: long-only 1.091 vs long/short 0.954, BOTH halves negative for shorts -- the
    sixth refutation of the short leg)
  * no take-profit (§3bl: a fixed bracket costs even a PERFECT-FORESIGHT oracle -0.94 to -1.43bp
    against simply holding, because barriers truncate the moves that pay for the losers)
  * a RATCHETING 3.0-ATR trail (§3bn: peaks at 3.0 with smooth decay both sides, worth +0.077
    Sharpe over a fixed SL/TP and +0.263 over signal-flip)
  * brackets in ATR, never percent (`gold-range-exploded-fixed-brackets-dead`: 2026 daily range is
    ~$125, so a 0.50% stop is 0.18 of ONE day)

*** THE CONTROL THAT DECIDES IT IS NOT "vs DOING NOTHING". *** A breakeven stop REDUCES risk, so
it will always flatter raw drawdown. `xau-champion-five-lifts-fail` made matched-vol gating a
method rule for exactly this, and `overnight-ban-is-turnover-not-gaps` showed the shape: a +$200
take-profit cost 0.26 Sharpe and **0.89x sizing beat it at matched drawdown**. So the control here
is the plain trail SIZED DOWN to each variant's own realised volatility. If sizing down wins, the
breakeven move is dead by the same argument that killed the take-profit.

*** WHY M15 IS ENOUGH HERE, WHICH IS NOT OBVIOUS. *** `order-of-events-needs-intraday` showed an
unresolvable intrabar convention inflating a result 19x (NR7 first-break: +8.23 on D1 vs +0.43 on
M15), so a stop study on bar data needs a reason, not an assumption. The reason is the ATR the
stop is measured in: **this uses the DAILY ATR (~$78), not the bar ATR (~$7.6 at M15)**, so a
1.5-ATR stop sits about **15 M15 bars of range** away from entry. A convention only bites when the
bracket is comparable to one bar's range; at 15x it is nearly irrelevant. PART 2 verifies that
empirically by running the identical rule on **H1** -- if M15 and H1 agree, bar resolution is not
driving the answer. (An M1 pull was attempted and the broker caps at 100,000 bars = 73 dates
starting 2026-07-22, while the M15 history ends 2026-06-15, so the two do not even overlap. M1 is
not needed and is not used.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "v5_runs" / "october"
RT_BP = 0.504 * 1.5
INIT_STOP_ATR = 1.5
TRAIL_ATR = 3.0
MAX_HOLD_DAYS = 5


def wilder_atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, parse_dates=["time"], index_col="time")
    return d[~d.index.duplicated(keep="last")].sort_index()


def simulate(d: pd.DataFrame, be_trigger_atr: float | None, be_offset_atr: float = 0.0,
             bars_per_day: int = 96, same_events: bool = True) -> pd.DataFrame:
    """Long-only break of the prior-day high; ratcheting 3.0-ATR trail; optional breakeven move.

    `be_trigger_atr`  None = never move to breakeven (the §3bn control).
                      x    = once unrealised profit reaches x*ATR, lift the stop.
    `be_offset_atr`   0.0  = lift it to exactly the entry ("free trade").
                      k    = lift it to entry - k*ATR, which truncates less.

    *** THE INTRABAR CONVENTION IS ADVERSE-FIRST. *** Within one bar we cannot know whether the
    high or the low came first, so the stop is tested against the LOW before the high is used to
    ratchet. That is the conservative choice, and PART 2 measures how wrong it is.
    """
    day = d.index.normalize()
    g = d.groupby(day)
    pdh = g["high"].max().shift(1).reindex(day).to_numpy(float)
    atr_d = wilder_atr(g["high"].max(), g["low"].min(), g["close"].last()).shift(1)
    atr = atr_d.reindex(day).to_numpy(float)
    h, l, c = (d[k].to_numpy(float) for k in ("high", "low", "close"))
    nd = day.to_numpy()

    trades, i, n = [], 1, len(d)
    while i < n:
        if not (np.isfinite(pdh[i]) and np.isfinite(atr[i]) and atr[i] > 0):
            i += 1; continue
        if h[i] <= pdh[i]:
            i += 1; continue
        entry = pdh[i]                                   # resting buy-stop at the level
        a0 = atr[i]
        stop = entry - INIT_STOP_ATR * a0
        hwm = entry
        moved = False
        limit = min(n - 1, i + MAX_HOLD_DAYS * bars_per_day)
        j, exit_px, why = i, np.nan, "open"
        while j <= limit:
            if l[j] <= stop:                             # adverse-first convention
                exit_px, why = stop, ("stop_be" if moved else "stop_init")
                break
            if h[j] > hwm:
                hwm = h[j]
            if be_trigger_atr is not None and not moved and (hwm - entry) >= be_trigger_atr * a0:
                cand = entry - be_offset_atr * a0
                if cand > stop:
                    stop, moved = cand, True
            trail = hwm - TRAIL_ATR * a0                 # ratchet: never loosens
            if trail > stop:
                stop = trail
            j += 1
        if not np.isfinite(exit_px):
            exit_px, why = c[min(j, n - 1)], "max_hold"
        trades.append(dict(entry_time=d.index[i], entry=entry, exit=exit_px, why=why,
                           bars=j - i, moved_to_be=moved,
                           ret_bp=(exit_px / entry - 1.0) * 1e4 - RT_BP))
        # *** EVENT SET MUST BE IDENTICAL ACROSS VARIANTS, OR THE PAIRED t IS MEANINGLESS. ***
        # Advancing to the day after the EXIT lets a variant that exits earlier take MORE trades
        # (492 vs 522 on the first run), so the variants were being compared on different event
        # sets and the "improvement" partly measured extra exposure. With same_events the next
        # entry is the day after the ENTRY, so every variant trades exactly the same days and the
        # difference is the exit rule alone. Overlapping holds are accepted for that.
        i = (int(np.searchsorted(nd, nd[i], side="right")) if same_events
             else max(j + 1, int(np.searchsorted(nd, nd[min(j, n - 1)], side="right"))))
    return pd.DataFrame(trades)


def daily(tr: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.Series:
    if tr.empty:
        return pd.Series(0.0, index=idx)
    s = (tr.set_index(tr["entry_time"].dt.normalize())["ret_bp"] / 1e4).groupby(level=0).sum()
    return s.reindex(idx).fillna(0.0)


def stats(r: pd.Series, ppy: float) -> dict:
    if r.std() == 0:
        return dict(sharpe=np.nan, cagr=np.nan, maxdd=np.nan)
    eq = (1 + r).cumprod()
    return dict(sharpe=float(r.mean() / r.std() * np.sqrt(ppy)),
                cagr=float(eq.iloc[-1] ** (ppy / len(r)) - 1),
                maxdd=float((eq / eq.cummax() - 1).min()))


VARIANTS = [
    ("A  no breakeven move  [§3bn control]", None, 0.0),
    ("B  -> breakeven at +1.0 ATR", 1.0, 0.0),
    ("C  -> breakeven at +2.0 ATR", 2.0, 0.0),
    ("D  -> entry-0.5ATR at +1.0 ATR", 1.0, 0.5),
    ("E  -> entry-1.0ATR at +1.5 ATR", 1.5, 1.0),
]


def run_block(d: pd.DataFrame, bpd: int, title: str) -> pd.DataFrame:
    idx = pd.DatetimeIndex(sorted(set(d.index.normalize())))
    yrs = max((idx[-1] - idx[0]).days / 365.25, 1e-9)
    ppy = len(idx) / yrs
    print(f"\n  {title}")
    print(f"  {'variant':<38} {'trades':>7} {'BE hit':>7} {'mean bp':>9} {'hit':>7} "
          f"{'Sharpe':>8} {'maxDD':>8} {'t vs A':>8}")
    print("  " + "-" * 100)
    base_r, rows = None, []
    for lab, trig, off in VARIANTS:
        tr = simulate(d, trig, off, bpd, same_events=True)
        if tr.empty:
            print(f"  {lab:<38} no trades"); continue
        r = daily(tr, idx)
        st = stats(r, ppy)
        be = float(tr["moved_to_be"].mean())
        if base_r is None:
            base_r, t = r, np.nan
        else:
            dd = (r - base_r).dropna()
            t = float(dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))) if dd.std(ddof=1) > 0 else np.nan
        print(f"  {lab:<38} {len(tr):>7} {be:>6.1%} {tr['ret_bp'].mean():>+9.3f} "
              f"{float((tr.ret_bp > 0).mean()):>7.4f} {st['sharpe']:>+8.3f} "
              f"{100*st['maxdd']:>7.1f}% {t:>+8.2f}")
        rows.append(dict(block=title, variant=lab, n=len(tr), be_rate=be,
                         mean_bp=float(tr["ret_bp"].mean()), **st, t_vs_A=t))
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 112)
    print("OCTOBER E27 — the move-to-breakeven stop, isolated (everything else pinned validated)")
    print("=" * 112)
    print(f"  entry: long-only break of the prior-day high · initial stop {INIT_STOP_ATR} ATR · "
          f"ratcheting {TRAIL_ATR} ATR trail · NO take-profit · max hold {MAX_HOLD_DAYS} days")
    print(f"  intrabar convention: ADVERSE-FIRST (the low is tested before the high ratchets)")

    m15 = load(ROOT / "data/XAUUSD_M15_long.csv")

    # Why the bar frame does not decide this: the stop is in DAILY ATR, so it sits many bar-ranges
    # from entry. Printed rather than asserted, because the whole PART 2 design rests on it.
    day = m15.index.normalize()
    g = m15.groupby(day)
    datr = float(wilder_atr(g["high"].max(), g["low"].min(), g["close"].last()).iloc[-1])
    batr = float(wilder_atr(m15["high"], m15["low"], m15["close"]).iloc[-1])
    print(f"  daily ATR ${datr:,.2f} vs M15 bar ATR ${batr:,.2f} -> a {INIT_STOP_ATR} ATR stop is "
          f"{INIT_STOP_ATR*datr/batr:.1f} M15 bar-ranges from entry,")
    print(f"  which is why the intrabar convention is not the binding issue here. PART 2 checks "
          f"it on H1 rather than assuming.")

    all_rows = [run_block(m15, 96, f"PART 1 — M15, FULL HISTORY "
                                   f"{m15.index[0].date()} -> {m15.index[-1].date()}")]

    # ---------------- PART 2 — does the bar frame change the answer? -------------------------
    from src.v5 import mtf_frames
    h1 = mtf_frames.aggregate(m15, "H1").dropna(subset=["close"])
    print(f"\n{'='*112}")
    print(f"  PART 2 — SAME RULE ON H1. If M15 and H1 agree, bar resolution is not driving it.")
    all_rows.append(run_block(h1, 24, f"PART 2 — H1, FULL HISTORY "
                                      f"{h1.index[0].date()} -> {h1.index[-1].date()}"))

    # ---------------- PART 3 — WHICH TIMEFRAME SHOULD THIS RUN ON? ---------------------------
    # The entry (prior-day high) and the stop (daily ATR) are both frame-INDEPENDENT by
    # construction, so the frame changes exactly one thing: how finely the stop resolves. A finer
    # frame sees the adverse excursion sooner and stops out earlier. Any difference here is
    # therefore execution granularity, not signal -- which is the honest way to read it.
    print(f"\n{'='*112}")
    print("  PART 3 — TIMEFRAME SWEEP. Entry and stop are frame-independent; the frame only")
    print("  decides how finely the stop resolves, so this measures granularity, not signal.")
    print(f"  {'frame':>6} {'bars':>9} {'trades':>7} {'mean bp':>9} {'hit':>7} {'Sharpe':>8} "
          f"{'maxDD':>8}   best breakeven variant")
    print("  " + "-" * 100)
    tf_rows = []
    for tf, bpd in (("M15", 96), ("M30", 48), ("H1", 24), ("H4", 6)):
        dd = m15 if tf == "M15" else mtf_frames.aggregate(m15, tf).dropna(subset=["close"])
        idx = pd.DatetimeIndex(sorted(set(dd.index.normalize())))
        ppy = len(idx) / max((idx[-1] - idx[0]).days / 365.25, 1e-9)
        cells = {}
        for lab, trig, off in VARIANTS:
            tr = simulate(dd, trig, off, bpd, same_events=True)
            if tr.empty:
                continue
            st = stats(daily(tr, idx), ppy)
            cells[lab] = (st, tr)
        if "A  no breakeven move  [§3bn control]" not in cells:
            continue
        sa, ta = cells["A  no breakeven move  [§3bn control]"]
        bestlab = max(cells, key=lambda k: cells[k][0]["sharpe"])
        print(f"  {tf:>6} {len(dd):>9,} {len(ta):>7} {ta['ret_bp'].mean():>+9.3f} "
              f"{float((ta.ret_bp > 0).mean()):>7.4f} {sa['sharpe']:>+8.3f} "
              f"{100*sa['maxdd']:>7.1f}%   {bestlab.split()[0]} ({cells[bestlab][0]['sharpe']:+.3f})")
        for lab, (st, tr) in cells.items():
            tf_rows.append(dict(block="PART3 timeframe", frame=tf, variant=lab,
                                n=len(tr), mean_bp=float(tr["ret_bp"].mean()), **st))
    all_rows.append(pd.DataFrame(tf_rows))
    sw = pd.DataFrame(tf_rows)
    if not sw.empty:
        k = len(sw)
        print(f"\n  *** CONTROL #7 — {k} cells were examined here ({sw.frame.nunique()} frames x "
              f"{sw.variant.nunique()} variants). ***")
        print(f"  Spread of the control variant across frames: "
              f"{sw[sw.variant.str.startswith('A')].sharpe.min():+.3f} to "
              f"{sw[sw.variant.str.startswith('A')].sharpe.max():+.3f} — read the SPREAD, not the")
        print(f"  argmax. A frame choice that only wins by less than that spread is a coin flip.")

    df = pd.concat([x for x in all_rows if not x.empty], ignore_index=True)
    df.to_csv(OUT / "e27_breakeven.csv", index=False)
    print(f"""
  HOW TO READ 't vs A'. Variant A is §3bn's validated trail with NO breakeven move. A positive t
  means the breakeven move ADDS; negative means it costs. Note this is the RAW paired t -- the
  matched-risk comparison is the one that decides, and it is run next only if any variant is
  positive here, because a de-risking overlay that already loses on raw terms cannot win on
  matched terms.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
