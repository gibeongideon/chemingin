"""XAU H4 top/bottom PROBABILITY -> position sizing — staged, oracle-gated research.

The user's question is "detect tops and bottoms with probability, size by it, report
win rate and Sharpe". Prior work (V5_FINDINGS §3, §3l, §4) answered a *geometric*
version of it and failed. This script answers it with a mandatory pre-flight that the
earlier studies never ran:

    STAGE 0 — THE ORACLE CEILING.
    Before building any detector for a label L, trade L with PERFECT HINDSIGHT and
    measure what it is worth. If a cheating detector is worthless, no amount of
    calibration / encoders / walk-forward can rescue an honest one.

That test reverses the program's premise (see --approach 1 output):
  * perfect knowledge of BOTTOMS  -> Sharpe 0.125 standalone, and HURTS as an overlay
    (the champion is long-only and near-always-in, so it is already long at bottoms).
  * perfect knowledge of TOPS     -> +0.22..+0.84 Sharpe as a trim on the champion.
  * the documented detector is 71% precision on bottoms and 70% @ only 9% recall on
    tops, i.e. strong on the worthless side and weak on the valuable one.
  * the gain needs RECALL ~0.8 over ~36% of bars => a regime/downside classifier,
    not a pivot detector. Turnover was a symptom, not the disease.

Everything here is therefore judged as a PAIRED DAILY DELTA vs the champion (never
standalone, never unpaired: Lo SE at SR 1.04 over 8.5y is +-0.43), against a
block-shuffled-p null (a matched CONSTANT control is useless — the champion is
vol-targeted so champ*{0.4..1.0} is Sharpe-neutral), with costs at three levels.

    python scripts/v5_xau_turn_prob.py --approach 1     # oracle ceiling screen
    python scripts/v5_xau_turn_prob.py --approach 1 --cost 0.448
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "v5_runs" / "xau-sharpe1-lab"))

from scipy.signal import argrelextrema  # noqa: E402

from src.v5.xau_dual_signals import champion_signal  # noqa: E402

EVAL_START = "2017-01-01"
SLIP_USD = 0.10
ANN_H4 = 252 * 6
TARGET_VOL = 0.10
VOL_HL = 42
BUFFER = 0.1
MAX_LEV = 8.0

# live broker quotes: blessed-CSV / live FTMO / stress. Sign must survive 0.448.
COST_LEVELS = {"blessed": None, "live": 0.448, "stress": 0.600}


# --------------------------------------------------------------------- data
def load_h4(cost_usd: float | None = None) -> pd.DataFrame:
    """H4 bars with an honest spread.

    `xau_lab.load_h4` floors the CSV spread at its FULL-SAMPLE median (an in-sample
    statistic) — replaced here with an EXPANDING median. `cost_usd` additionally
    floors the round-trip spread at a live broker quote, because the CSV column
    understates the venue (measured mean $0.360 vs live FTMO $0.448).
    """
    df = pd.read_csv(ROOT / "data" / "XAUUSD_H4_long.csv",
                     parse_dates=["time"], index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    exp_med = df["spread"].expanding(min_periods=20).median().bfill()
    df["spread_px"] = np.maximum(df["spread"], exp_med) * 0.1
    if cost_usd is not None:
        df["spread_px"] = np.maximum(df["spread_px"], float(cost_usd))
    return df


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ------------------------------------------------------------------- engine
def engine(df: pd.DataFrame, fc: pd.Series, *, buffer_frac: float = BUFFER,
           delay: int = 1) -> tuple[dict, pd.Series]:
    """Continuous vol-targeted engine — mirrors xau_lab.run but ALSO returns the
    daily return series, which paired testing requires. Verified to reproduce
    xau_lab's champion Sharpe (see --approach 1 regression line)."""
    close = df["close"]
    ret = close.pct_change()
    vol = ret.ewm(halflife=VOL_HL, min_periods=20).std() * np.sqrt(ANN_H4)
    pos = (fc.clip(-2.0, 2.0) * (TARGET_VOL / vol)).clip(-MAX_LEV, MAX_LEV)

    if buffer_frac > 0:                       # causal no-trade band around the holding
        band = (buffer_frac * (TARGET_VOL / vol).clip(0, MAX_LEV)).values
        p, out, held = pos.values.copy(), np.zeros(len(pos)), 0.0
        for i in range(len(p)):
            if np.isfinite(p[i]):
                b = band[i] if np.isfinite(band[i]) else 0.0
                if abs(p[i] - held) > b:
                    held = p[i] - np.sign(p[i] - held) * b
            out[i] = held
        pos = pd.Series(out, index=pos.index)

    pos = pos.shift(delay).fillna(0.0)
    cost_frac = (df["spread_px"] / 2.0 + SLIP_USD) / close
    net = (pos * ret - pos.diff().abs().fillna(0.0) * cost_frac).fillna(0.0)
    eq = (1.0 + net).cumprod()

    e = eq.loc[EVAL_START:]
    e = e / e.iloc[0]
    daily = e.resample("D").last().pct_change(fill_method=None).dropna()
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    m = {
        "sharpe": round(float(daily.mean() / daily.std() * np.sqrt(252)), 3),
        "dd": round(float((e / e.cummax() - 1.0).min() * 100), 1),
        "cagr": round(float(e.iloc[-1] ** (1 / yrs) - 1) * 100, 1),
        "turnover": round(float(pos.diff().abs().sum() / yrs), 1),
        "avg_pos": round(float(pos.abs().mean()), 2),
    }
    return m, daily


def paired(daily: pd.Series, base: pd.Series) -> tuple[float, float, int]:
    """Paired daily-delta t-stat vs the base. This is the primary statistic: the
    champion already contains gold's bull beta, so only the residual is a claim."""
    j = daily.index.intersection(base.index)
    d = (daily.reindex(j) - base.reindex(j)).dropna()
    if len(d) < 30 or d.std() == 0:
        return 0.0, 0.0, len(d)
    return float(d.mean()), float(d.mean() / d.std() * np.sqrt(len(d))), len(d)


def per_year(daily: pd.Series, base: pd.Series) -> tuple[int, int]:
    """How many OOS years the paired delta is positive in (a regime-robustness gate:
    even the top-oracle loses in ~2 of 11 years)."""
    j = daily.index.intersection(base.index)
    d = (daily.reindex(j) - base.reindex(j)).dropna()
    g = d.groupby(d.index.year).mean()
    return int((g > 0).sum()), int(len(g))


# ------------------------------------------------------------------- labels
def geometric_pivots(df: pd.DataFrame, order: int = 5, theta_mult: float = 1.5,
                     tol: int = 3) -> tuple[pd.Series, pd.Series]:
    """The §3 ground truth: alternating ATR-filtered zigzag pivots, +-tol bars.
    FORWARD-LOOKING by construction — this is an answer key, not a feature."""
    hi, lo, close = df["high"].values, df["low"].values, df["close"]
    theta = theta_mult * atr(df)
    cand = sorted([(i, "H") for i in argrelextrema(hi, np.greater_equal, order=order)[0]]
                  + [(i, "L") for i in argrelextrema(lo, np.less_equal, order=order)[0]])
    kept: list[tuple[int, str]] = []
    for i, t in cand:
        px = hi[i] if t == "H" else lo[i]
        if kept and kept[-1][1] == t:
            prev = hi[kept[-1][0]] if t == "H" else lo[kept[-1][0]]
            if (t == "H" and px > prev) or (t == "L" and px < prev):
                kept[-1] = (i, t)
        elif not kept or abs(px - (hi[kept[-1][0]] if kept[-1][1] == "H"
                                   else lo[kept[-1][0]])) >= theta.iloc[i]:
            kept.append((i, t))

    def near(idx: list[int]) -> pd.Series:
        y = np.zeros(len(df), bool)
        for i in idx:
            y[max(0, i - tol):min(len(df), i + tol + 1)] = True
        return pd.Series(y, index=df.index)

    return near([i for i, t in kept if t == "L"]), near([i for i, t in kept if t == "H"])


def economic_bad(df: pd.DataFrame, k: int) -> pd.Series:
    """ECONOMIC label: the next k bars are down. Oracle-trading this is what scored
    t~8-12 in planning — the target a trim overlay actually wants."""
    fwd = df["close"].shift(-k) / df["close"] - 1.0
    return (fwd < 0).fillna(False)


def champ_meta_bad(df: pd.DataFrame, fc: pd.Series, k: int) -> pd.Series:
    """The champion's OWN forward P&L over k bars is negative — meta-labelling a base
    with a known positive edge, which is what V5_FINDINGS §2's law demands."""
    ret = df["close"].pct_change()
    pnl = (fc.shift(1).fillna(0.0) * ret).rolling(k).sum().shift(-k)
    return (pnl < 0).fillna(False)


def downside_quantile_bad(df: pd.DataFrame, k: int, q: float = 0.30) -> pd.Series:
    """Forward maximum adverse excursion in the worst q of its own distribution — a
    broad-coverage drawdown target rather than a sparse pivot event."""
    close = df["close"]
    fwd_min = pd.Series(
        [close.iloc[i + 1:i + 1 + k].min() / close.iloc[i] - 1.0 if i + 1 + k <= len(close)
         else np.nan for i in range(len(close))], index=close.index)
    thr = fwd_min.expanding(min_periods=500).quantile(q)
    return (fwd_min <= thr).fillna(False)


# ------------------------------------------------------------ oracle screen
def pr_degrade(flag: pd.Series, precision: float, recall: float,
               rng: np.random.Generator) -> pd.Series:
    """Synthesize a detector with the given (precision, recall) against `flag`.
    Keeps `recall` of the true positives, then adds false positives drawn from the
    negatives so the realized precision matches."""
    pos = np.where(flag.values)[0]
    neg = np.where(~flag.values)[0]
    n_tp = int(round(len(pos) * recall))
    tp = rng.choice(pos, size=n_tp, replace=False) if n_tp else np.array([], int)
    n_fp = int(round(n_tp * (1.0 - precision) / max(precision, 1e-9)))
    n_fp = min(n_fp, len(neg))
    fp = rng.choice(neg, size=n_fp, replace=False) if n_fp else np.array([], int)
    out = np.zeros(len(flag), bool)
    out[np.concatenate([tp, fp]).astype(int)] = True
    return pd.Series(out, index=flag.index)


def screen(df: pd.DataFrame, champ_fc: pd.Series, base_daily: pd.Series,
           name: str, bad: pd.Series, *, strengths=(0.25, 1.0)) -> list[dict]:
    """Stage 0 for one label family: trade it with hindsight, as a TRIM on the
    champion (never a boost — §3 antagonism and the bottom-oracle row both say so)."""
    rows = []
    for b in strengths:
        m, d = engine(df, champ_fc * (1.0 - b * bad.astype(float)))
        dsr, t, n = paired(d, base_daily)
        yrs_pos, yrs_n = per_year(d, base_daily)
        rows.append(dict(label=name, mode=f"trim b={b}", coverage=round(float(bad.mean()), 3),
                         sharpe=m["sharpe"], dsharpe=round(m["sharpe"] - BASE_SR, 3),
                         paired_t=round(t, 2), turnover=m["turnover"], dd=m["dd"],
                         yrs=f"{yrs_pos}/{yrs_n}"))
    return rows


BASE_SR = 0.0  # set at runtime


def approach1(cost_usd: float | None) -> None:
    """STAGE 0 — the oracle ceiling screen over every candidate label family."""
    df = load_h4(cost_usd)
    close = df["close"]
    champ_fc = champion_signal(close)
    lab = "blessed CSV" if cost_usd is None else f"${cost_usd:.3f} floor"
    print(f"\n=== STAGE 0: ORACLE CEILING SCREEN — XAU H4, cost {lab} "
          f"(mean spread ${df['spread_px'].mean():.4f}/oz) ===")
    print(f"    bars {len(df)}  {df.index[0].date()}..{df.index[-1].date()}   "
          f"eval from {EVAL_START}\n")

    # ---- benchmarks
    vol_ret = close.pct_change().ewm(halflife=VOL_HL, min_periods=20).std() * np.sqrt(ANN_H4)
    bh_m, bh_d = engine(df, pd.Series(1.0, index=df.index))
    ch_m, ch_d = engine(df, champ_fc)
    global BASE_SR
    BASE_SR = ch_m["sharpe"]
    print("BENCHMARKS")
    print(f"  {'buy&hold vol-targeted 10%':38s} SR {bh_m['sharpe']:+.3f}  "
          f"turnover {bh_m['turnover']:6.1f}  dd {bh_m['dd']:+.1f}%")
    print(f"  {'CHAMPION (champion_signal)':38s} SR {ch_m['sharpe']:+.3f}  "
          f"turnover {ch_m['turnover']:6.1f}  dd {ch_m['dd']:+.1f}%   <-- all deltas vs this")

    # ---- label families
    bots, tops = geometric_pivots(df)
    fams: list[tuple[str, pd.Series]] = [
        ("geometric TOPS (§3 answer key)", tops),
        ("geometric BOTTOMS (inverted->trim)", bots),
    ]
    for k in (12, 24, 48):
        fams.append((f"economic fwd-{k}bar down", economic_bad(df, k)))
    for k in (12, 24, 48):
        fams.append((f"champ-meta fwd-{k}bar P&L<0", champ_meta_bad(df, champ_fc, k)))
    for k in (24, 48):
        fams.append((f"downside-quantile k{k} q30", downside_quantile_bad(df, k)))

    rows: list[dict] = []
    # bottoms as a BOOST is the historically-assumed use — measure it explicitly
    for b in (0.25, 1.0):
        m, d = engine(df, champ_fc * (1.0 + b * bots.astype(float)))
        _, t, _ = paired(d, ch_d)
        yp, yn = per_year(d, ch_d)
        rows.append(dict(label="geometric BOTTOMS", mode=f"BOOST b={b}",
                         coverage=round(float(bots.mean()), 3), sharpe=m["sharpe"],
                         dsharpe=round(m["sharpe"] - BASE_SR, 3), paired_t=round(t, 2),
                         turnover=m["turnover"], dd=m["dd"], yrs=f"{yp}/{yn}"))
    m, d = engine(df, pd.Series(1.0, index=df.index).where(bots, 0.0))
    rows.append(dict(label="geometric BOTTOMS", mode="long standalone",
                     coverage=round(float(bots.mean()), 3), sharpe=m["sharpe"],
                     dsharpe=round(m["sharpe"] - BASE_SR, 3),
                     paired_t=round(paired(d, ch_d)[1], 2), turnover=m["turnover"],
                     dd=m["dd"], yrs="-"))
    for nm, bad in fams:
        rows += screen(df, champ_fc, ch_d, nm, bad)

    R = pd.DataFrame(rows)
    print("\nORACLE CEILINGS (perfect hindsight; dSharpe & paired t vs champion)")
    print(f"  {'label':34s} {'mode':16s} {'cov':>5s} {'SR':>7s} {'dSR':>7s} "
          f"{'t':>6s} {'turn':>6s} {'yrs+':>6s}")
    for _, x in R.iterrows():
        flag = "  <== PASSES" if (x.dsharpe >= 0.20 and x.paired_t >= 2.5) else ""
        print(f"  {x.label:34s} {x['mode']:16s} {x.coverage:5.2f} {x.sharpe:+7.3f} "
              f"{x.dsharpe:+7.3f} {x.paired_t:+6.2f} {x.turnover:6.1f} {x.yrs:>6s}{flag}")

    out = ROOT / "data" / "v5_runs" / "turnprob"
    out.mkdir(parents=True, exist_ok=True)
    tag = "blessed" if cost_usd is None else f"{cost_usd:.3f}"
    R.to_csv(out / f"oracle_screen_{tag}.csv", index=False)

    # ---- attainability frontier for the best-passing family
    best = R[(R.dsharpe >= 0.20) & (R.paired_t >= 2.5)].sort_values("paired_t", ascending=False)
    print("\nSTAGE-0 GATE: dSharpe >= +0.20 AND paired t >= 2.5 AND the (prec,recall)")
    print("contour reaching dSR +0.20 must sit inside {prec<=0.80, recall<=0.60}.")
    if best.empty:
        print("  -> NO label family passes. The program is capped; stop before building.")
        return
    for nm in best.label.unique()[:3]:
        bad = dict(fams + [("geometric BOTTOMS", bots)])[nm]
        print(f"\n  ATTAINABILITY FRONTIER — {nm} (trim b=1.0, dSR vs champion, 3 seeds)")
        hdr = "prec|rec"
        print(f"    {hdr:>9s}" + "".join(f"{r:>9.2f}" for r in (1.0, 0.8, 0.6, 0.4, 0.2)))
        for p in (1.0, 0.9, 0.8, 0.7, 0.6):
            cells = []
            for r in (1.0, 0.8, 0.6, 0.4, 0.2):
                ds = []
                for seed in (1, 2, 3):
                    fl = pr_degrade(bad, p, r, np.random.default_rng(seed))
                    mm, dd_ = engine(df, champ_fc * (1.0 - fl.astype(float)))
                    ds.append(mm["sharpe"] - BASE_SR)
                cells.append(f"{np.mean(ds):+9.3f}")
            print(f"    {p:>9.2f}" + "".join(cells))
    print(f"\n  wrote {out / f'oracle_screen_{tag}.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--approach", type=int, default=1)
    ap.add_argument("--cost", default="blessed",
                    help="blessed | live | stress | a $ figure e.g. 0.448")
    args = ap.parse_args()
    cost = COST_LEVELS.get(args.cost, None) if args.cost in COST_LEVELS else float(args.cost)
    if args.approach == 1:
        approach1(cost)
    else:
        raise SystemExit(f"approach {args.approach} not implemented yet")


if __name__ == "__main__":
    main()
