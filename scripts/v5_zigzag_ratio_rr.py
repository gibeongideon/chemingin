#!/usr/bin/env python3
"""ZigZag on the BTC/XAU ratio -- the ACCURACY-vs-RR question, answered by arithmetic first.

The request was to re-explore the ZigZag on "BTCXAU" while "using the correct accuracy and RR".
That phrasing points at the right method, so this script inverts the usual order: instead of
building a detector and hoping the economics work, it computes WHAT PRECISION EACH RISK-REWARD
REQUIRES and compares that against two hard references:

  (a) the ORACLE bracket precision -- what a PERFECT pivot detector achieves at that RR. This is
      the ceiling, and it is NOT 1.0: knowing a bar is a true ZigZag pivot does not mean a
      TP/SL bracket placed there resolves favourably. CONTROL #5 (oracle ceiling before detector).

  (b) 0.645 -- the best precision any detector in this repo has actually reached (the gold H1
      ZigZag at threshold 0.60: AUC 0.7589, 3,726 fires / 698 non-overlapping, within_tol 0.6455,
      mean_offset -1.065 so it is already anticipatory). Source: data/v5_runs/zigzag_atr/gates.json.

If required precision exceeds the oracle ceiling, the cell is dead for ANY detector and no amount
of model work reopens it. If it sits between 0.645 and the ceiling, it is dead for every detector
we have actually built. Only cells below 0.645 are worth a model.

*** BTCXAU IS NOT A TRADEABLE SYMBOL. *** All 2,067 Switch Markets symbols were checked (3bu):
only XAUUSDs, XAUEURs, BTCUSDs, BTCEURs. The ratio is SYNTHESISED here and would need TWO LEGS,
costing 7.44bp round trip against gold's 0.75bp -- about 10x. That cost is the binding constraint
and it is why this is an arithmetic question before it is a modelling one.

*** D1 INTRABAR AMBIGUITY IS REPORTED, NOT HIDDEN. *** With only daily OHLC we cannot know whether
TP or SL was touched first inside a bar. 'Order-of-events needs intraday' is a filed lesson here:
the NR7 first-break study scored Sharpe +8.23 on D1 and +0.43 on M15, a 19x illusion built partly
on exactly this ambiguity. So every both-touched bar is resolved AS A LOSS (pessimistic) and the
ambiguous fraction is printed beside every cell. A result that depends on the optimistic
convention is not a result.

GOLD is run as a CONTROL CELL with a known answer: 3bl already measured that exit changes on this
family recover only 4.3% of a 33.3bp oracle gap, so gold must come back with thin-to-negative
headroom. If it does not, the harness is wrong, not the finding.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "v5_runs" / "zigzag_ratio"

# Round trip in bp, measured live at Switch Markets 2026-10-19 (3bs). one-way = 0.75 x spread.
COST_BP = {"GOLD": 0.50 * 1.5, "BTC": 4.46 * 1.5, "RATIO": (0.50 + 4.46) * 1.5}
# The best precision any detector in this repo has reached (gold H1 ZigZag @ 0.60).
DETECTOR_BEST = 0.645
RR_GRID = (0.5, 1.0, 1.5, 2.0, 3.0)
SL_ATR_GRID = (0.5, 1.0, 2.0, 3.0)
ZZ_K = (1.0, 2.0, 3.0)


def load(alias: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / f"{alias}_D1_long.csv", parse_dates=["time"])
    d = d.set_index("time").sort_index()
    return d[~d.index.duplicated(keep="last")]


def wilder_atr(h, l, c, n: int = 14) -> pd.Series:
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def zigzag_pivots(h: pd.Series, l: pd.Series, c: pd.Series, atr: pd.Series,
                  k: float) -> pd.DataFrame:
    """Confirmed ZigZag pivots as (pivot_i, confirm_i, side): side +1 trough, -1 peak.

    A pivot is confirmed only once price has retraced k x ATR from the extreme, which is why the
    label is knowable ONLY IN HINDSIGHT -- that is the whole reason an oracle ceiling is needed
    before a detector is built. The confirmation lag is not a flaw in the label, it IS the label.

    *** BUG FIXED 2026-10-21. *** The first version kept `direction` at 0 until a pivot fired and
    allowed BOTH update branches while direction == 0, so the running extreme tracked price every
    single bar, the retracement `move` was therefore always exactly 0, and no pivot could ever
    confirm -- a deadlock that returned 0 pivots on every asset and every k. It was caught
    immediately because GOLD is run as a control cell with a known answer (3bl measured this family
    on gold), and a control returning nothing is unmistakable. The fix: the extreme may only track
    in the CURRENT direction, and an undecided series is seeded as provisionally rising.

    *** AND THE CIRCULARITY THE FIRST RUN WALKED INTO -- the reason this returns CONFIRM indices.
    *** A trough is confirmed only once price has risen k x ATR above it, so BY CONSTRUCTION price
    rises at least k x ATR after a true trough. Entering at the PIVOT bar's close therefore makes a
    TP of less than k x ATR a mathematical certainty, not a prediction: the first run returned
    oracle precision of 0.97-1.00 and declared almost every cell "worth a model". That is the same
    tautology class already killed four times here (window extremes are set by the window's return
    sign). A detector cannot fire at the pivot -- the pivot is only knowable once the confirming
    move has ALREADY happened. So this returns (pivot_i, confirm_i, side) and the oracle must enter
    at CONFIRM_I, which is the earliest bar the information exists on.
    """
    cv = c.to_numpy(float)
    av = atr.to_numpy(float)
    n = len(cv)
    start = int(np.argmax(np.isfinite(av) & (av > 0)))
    last_ext_i, last_ext_v, direction = start, cv[start], +1
    rows = []
    for i in range(start + 1, n):
        a = av[i]
        if not (np.isfinite(a) and a > 0):
            continue
        if direction > 0:
            if cv[i] > last_ext_v:                     # extend the rally
                last_ext_i, last_ext_v = i, cv[i]
            elif cv[i] < last_ext_v - k * a:           # retraced enough: that extreme was a PEAK
                rows.append((last_ext_i, i, -1))
                direction, last_ext_i, last_ext_v = -1, i, cv[i]
        else:
            if cv[i] < last_ext_v:                     # extend the decline
                last_ext_i, last_ext_v = i, cv[i]
            elif cv[i] > last_ext_v + k * a:           # that extreme was a TROUGH
                rows.append((last_ext_i, i, +1))
                direction, last_ext_i, last_ext_v = +1, i, cv[i]
    return pd.DataFrame(rows, columns=["pivot_i", "confirm_i", "side"])


def bracket_outcome(entry: float, side: int, sl_px: float, tp_px: float,
                    hi: np.ndarray, lo: np.ndarray, horizon: int) -> tuple[int, bool]:
    """Walk forward bar by bar. Returns (outcome, ambiguous).

    outcome: +1 TP first, -1 SL first, 0 neither inside the horizon.
    ambiguous: both levels touched in the SAME bar -- unresolvable on daily OHLC, so it is
    resolved as -1 (a loss) and counted here so the reader can see how much of the result rests
    on a convention we cannot verify without intraday data.
    """
    for j in range(len(hi)):
        if j >= horizon:
            break
        if side > 0:
            hit_tp, hit_sl = hi[j] >= tp_px, lo[j] <= sl_px
        else:
            hit_tp, hit_sl = lo[j] <= tp_px, hi[j] >= sl_px
        if hit_tp and hit_sl:
            return -1, True
        if hit_tp:
            return +1, False
        if hit_sl:
            return -1, False
    return 0, False


def oracle_precision(d: pd.DataFrame, atr: pd.Series, piv: pd.DataFrame,
                     sl_atr: float, rr: float, horizon: int = 30) -> dict:
    """Precision a PERFECT detector achieves -- entering at CONFIRMATION, the earliest honest bar.

    This is the ceiling for any detector, because a detector that fires before confirmation is
    guessing, and one that fires after is late. Entry is the close of `confirm_i`, by which point
    price has already travelled k x ATR away from the pivot: the oracle knows WHAT just happened,
    not what happens next, which is precisely the information a real model could hold.
    """
    hi, lo, cl = d["high"].to_numpy(float), d["low"].to_numpy(float), d["close"].to_numpy(float)
    av = atr.to_numpy(float)
    wins = losses = neither = amb = 0
    lags, spent = [], []
    for pi, ci, side in piv.itertuples(index=False):
        if ci + 1 >= len(cl) or not np.isfinite(av[ci]) or av[ci] <= 0:
            continue
        entry = cl[ci]                      # <-- CONFIRMATION bar, never the pivot bar
        lags.append(ci - pi)
        spent.append(abs(entry - cl[pi]) / av[ci])
        sl_d, tp_d = sl_atr * av[ci], sl_atr * rr * av[ci]
        # After a confirmed PEAK (side -1) the new direction is DOWN, so the trade is short.
        trade = int(side)
        sl_px = entry - trade * sl_d
        tp_px = entry + trade * tp_d
        o, ambiguous = bracket_outcome(entry, trade, sl_px, tp_px,
                                       hi[ci + 1:], lo[ci + 1:], horizon)
        amb += ambiguous
        if o > 0:
            wins += 1
        elif o < 0:
            losses += 1
        else:
            neither += 1
    resolved = wins + losses
    return dict(fires=len(piv), resolved=resolved, wins=wins, losses=losses,
                unresolved=neither, ambiguous=amb,
                precision=(wins / resolved) if resolved else np.nan,
                amb_frac=(amb / resolved) if resolved else np.nan,
                lag=float(np.median(lags)) if lags else np.nan,
                spent_atr=float(np.median(spent)) if spent else np.nan)


def required_precision(sl_bp: float, rr: float, cost_bp: float) -> float:
    """Breakeven precision for a TP = rr x SL bracket, net of cost.

        EV = p*(rr*SL) - (1-p)*SL - cost = 0   ->   p = (SL + cost) / (SL*(1+rr))

    Note cost enters as a FRACTION OF SL. A wide stop dilutes the spread; a tight stop is
    dominated by it. This is the exact arithmetic that makes the ratio's 7.44bp a real
    constraint while gold's 0.75bp is nearly free.
    """
    return (sl_bp + cost_bp) / (sl_bp * (1.0 + rr))


def run(name: str, d: pd.DataFrame, cost_bp: float) -> pd.DataFrame:
    atr = wilder_atr(d["high"], d["low"], d["close"])
    atr_bp = float((atr / d["close"]).median() * 1e4)
    sigma_bp = float(d["close"].pct_change().std() * 1e4)

    print(f"\n{'=' * 100}")
    print(f"{name}   bars {len(d):,}  {d.index.min().date()} -> {d.index.max().date()}   "
          f"median ATR {atr_bp:,.0f}bp   daily sigma {sigma_bp:,.0f}bp   cost {cost_bp:.2f}bp")
    print(f"{'=' * 100}")

    rows = []
    for k in ZZ_K:
        piv = zigzag_pivots(d["high"], d["low"], d["close"], atr, k)
        npiv = len(piv)
        if npiv < 20:
            print(f"  k={k}: only {npiv} pivots, skipped")
            continue
        _probe = oracle_precision(d, atr, piv, 1.0, 1.0)
        print(f"\n  --- ZigZag k={k} ATR : {npiv} confirmed pivots "
              f"({npiv / (len(d) / 252):.1f}/yr), median confirmation lag "
              f"{_probe['lag']:.0f} bars, {_probe['spent_atr']:.2f} ATR already spent "
              f"by entry ---")
        print(f"    {'SL(ATR)':>8} {'RR':>5} {'SL bp':>7} {'need p':>8} {'oracle p':>9} "
              f"{'headroom':>9} {'amb%':>6} {'vs 0.645':>9}  verdict")
        for sl_atr in SL_ATR_GRID:
            sl_bp = sl_atr * atr_bp
            for rr in RR_GRID:
                need = required_precision(sl_bp, rr, cost_bp)
                orc = oracle_precision(d, atr, piv, sl_atr, rr)
                head = orc["precision"] - need
                if head <= 0:
                    verdict = "DEAD (oracle below breakeven)"
                elif need > DETECTOR_BEST:
                    verdict = "dead for any detector we have"
                else:
                    verdict = "** worth a model **"
                print(f"    {sl_atr:>8.1f} {rr:>5.1f} {sl_bp:>7.0f} {need:>8.3f} "
                      f"{orc['precision']:>9.3f} {head:>+9.3f} {orc['amb_frac']:>5.1%} "
                      f"{need - DETECTOR_BEST:>+9.3f}  {verdict}")
                rows.append(dict(asset=name, k=k, sl_atr=sl_atr, rr=rr, sl_bp=sl_bp,
                                 need_p=need, oracle_p=orc["precision"], headroom=head,
                                 amb_frac=orc["amb_frac"], fires=orc["fires"],
                                 resolved=orc["resolved"], unresolved=orc["unresolved"],
                                 detector_gap=need - DETECTOR_BEST, verdict=verdict))
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gold, btc = load("GOLD"), load("BTC")
    common = gold.index.intersection(btc.index)
    # The synthetic cross. high/low of a ratio are NOT the ratio of highs/lows -- the extremes of
    # B/G come from B-high paired with G-LOW and vice versa. Getting this backwards would understate
    # the range and flatter every bracket.
    ratio = pd.DataFrame({
        "open": btc["open"].reindex(common) / gold["open"].reindex(common),
        "high": btc["high"].reindex(common) / gold["low"].reindex(common),
        "low": btc["low"].reindex(common) / gold["high"].reindex(common),
        "close": btc["close"].reindex(common) / gold["close"].reindex(common),
    }).dropna()

    print("=" * 100)
    print("ZigZag x BTC/XAU -- required precision vs RR, against the oracle ceiling and 0.645")
    print("=" * 100)
    print(f"  reference: best detector precision ever reached in this repo = {DETECTOR_BEST:.3f}")
    print("  (gold H1 ZigZag @ threshold 0.60, AUC 0.7589, 698 non-overlapping fires)")

    res = pd.concat([
        run("RATIO (BTC/XAU, 2 legs)", ratio, COST_BP["RATIO"]),
        run("GOLD [CONTROL, known answer]", gold.loc[common], COST_BP["GOLD"]),
    ], ignore_index=True)
    res.to_csv(OUT / "rr_accuracy_grid.csv", index=False)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for asset, g in res.groupby("asset", sort=False):
        alive = g[g["headroom"] > 0]
        modelable = g[(g["headroom"] > 0) & (g["need_p"] <= DETECTOR_BEST)]
        print(f"\n  {asset}")
        print(f"    cells: {len(g)}   oracle clears breakeven: {len(alive)}   "
              f"AND within reach of a real detector: {len(modelable)}")
        if len(modelable):
            print(modelable[["k", "sl_atr", "rr", "need_p", "oracle_p", "headroom",
                             "amb_frac"]].to_string(index=False,
                                                    float_format=lambda v: f"{v:,.3f}"))
        else:
            best = g.loc[g["headroom"].idxmax()]
            print(f"    best cell k={best['k']} SL={best['sl_atr']}ATR RR={best['rr']}: "
                  f"need {best['need_p']:.3f} vs oracle {best['oracle_p']:.3f} "
                  f"(headroom {best['headroom']:+.3f}), and a real detector tops out at "
                  f"{DETECTOR_BEST:.3f}")
    print(f"\n  grid -> {OUT / 'rr_accuracy_grid.csv'}")


if __name__ == "__main__":
    main()
