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
           delay: int = 1, eval_start: str = EVAL_START) -> tuple[dict, pd.Series]:
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

    e = eq.loc[eval_start:]
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


# ============================ APPROACH 4 ==================================
# Forward-DOWNSIDE probability. Stage 0 says this is the best-conditioned target:
# oracle dSR +0.70 at t=10.8, 10/10 years, and its attainability frontier clears
# +0.20 even at precision 0.60 / recall 0.20 (unlike the economic label, which goes
# NEGATIVE at 0.60/0.60). It is a DRAWDOWN-AVOIDANCE filter, not a top detector.

OOS_START_YEAR = 2018
PURGE = 8                      # tol+order: a pivot at i is unconfirmable until i+5


def fwd_min_atr(df: pd.DataFrame, k: int) -> pd.Series:
    """Forward maximum adverse excursion over the next k bars, in ATR units.
    Negative = drawdown ahead. Label only — deliberately forward-looking."""
    close = df["close"].values
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(n - k):
        out[i] = close[i + 1:i + 1 + k].min() / close[i] - 1.0
    a = (atr(df) / df["close"]).values          # ATR as a fraction of price
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.Series(out / np.where(a > 0, a, np.nan), index=df.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """The §3 causal feature set MINUS `hour` (on H4 gold that is a within-day
    position index and finds a session artifact off ragged Sunday bars), plus
    vol/trend context that a downside model needs."""
    from scripts.v5_xau_turning_ml import features as f23
    F = f23(df).drop(columns=["hour"], errors="ignore")
    close = df["close"]
    a = atr(df)
    F["atr_frac"] = a / close
    F["ema200_dist"] = (close - close.ewm(span=200, min_periods=100).mean()) / a
    F["dd_from_hi"] = close / close.rolling(120, min_periods=60).max() - 1.0
    F["vol_ratio"] = (close.pct_change().rolling(12).std()
                      / close.pct_change().rolling(72).std())
    return F


def _calibrated_fit_predict(Xtr, ytr, Xte, seed=7):
    """Fit on the head of train, fit an isotonic calibrator on a PURGED tail slice.
    (src/v5/xau_meta_oos.py uses unpurged StratifiedKFold(cv=3) for this, which on
    overlapping labels yields over-confident probabilities.)"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    cut = int(len(Xtr) * 0.80)
    Xf, yf = Xtr[:cut], ytr[:cut]
    Xc, yc = Xtr[cut + PURGE:], ytr[cut + PURGE:]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                         l2_regularization=1.0, random_state=seed)
    clf.fit(Xf, yf)
    p_te = clf.predict_proba(Xte)[:, 1]
    if len(Xc) > 50 and 0 < yc.mean() < 1:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(clf.predict_proba(Xc)[:, 1], yc)
        p_te = iso.predict(p_te)
    return p_te


def badness(df: pd.DataFrame, champ_fc: pd.Series, kind: str, k: int) -> pd.Series:
    """Continuous forward 'badness' target — LOW values = bad window ahead. Label
    only (forward-looking by construction). The three families Stage 0 endorsed."""
    if kind == "dsq":                 # forward max adverse excursion, ATR units
        return fwd_min_atr(df, k)
    if kind == "econ":                # plain forward k-bar return
        return df["close"].shift(-k) / df["close"] - 1.0
    if kind == "champmeta":           # the CHAMPION's own forward k-bar P&L
        ret = df["close"].pct_change()
        return (champ_fc.shift(1).fillna(0.0) * ret).rolling(k).sum().shift(-k)
    raise ValueError(kind)


def walk_forward_prob(df: pd.DataFrame, F: pd.DataFrame, k: int, q: float,
                      tgt: pd.Series, seed: int = 7) -> tuple[pd.Series, list[dict]]:
    """Expanding yearly refits. Purge: a train row's label window (t..t+k) must end
    at least PURGE bars before test_start, else the label peeks into the test set.
    The 'bad' threshold is taken from the TRAIN slice only (no expanding-quantile
    peek). Returns calibrated p_bad on OOS bars + per-fold diagnostics."""
    X = F.values
    ok = np.isfinite(X).all(axis=1) & np.isfinite(tgt.values)
    years = sorted({t.year for t in df.index if t.year >= OOS_START_YEAR})
    p_out = pd.Series(np.nan, index=df.index)
    folds = []
    for yr in years:
        ts, te = pd.Timestamp(f"{yr}-01-01"), pd.Timestamp(f"{yr+1}-01-01")
        pos = np.arange(len(df))
        # purge: label window (t..t+k) must close PURGE bars before test start
        tr_m = ok & (df.index < ts) & (pos < np.searchsorted(df.index, ts) - k - PURGE)
        te_m = ok & (df.index >= ts) & (df.index < te)
        if tr_m.sum() < 800 or te_m.sum() < 30:
            continue
        thr = np.quantile(tgt.values[tr_m], q)          # train-only threshold
        y = (tgt.values <= thr).astype(int)
        p = _calibrated_fit_predict(X[tr_m], y[tr_m], X[te_m], seed)
        p_out.iloc[np.where(te_m)[0]] = p
        yt = y[te_m]
        folds.append(dict(year=yr, n_tr=int(tr_m.sum()), n_te=int(te_m.sum()),
                          base=round(float(yt.mean()), 3), thr=round(float(thr), 3),
                          auc=_auc(yt, p), ic=_ic(p, tgt.values[te_m])))
    return p_out, folds


def _auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    return round(float(roc_auc_score(y, p)), 3) if 0 < np.mean(y) < 1 else float("nan")


def _ic(p, t) -> float:
    from scipy.stats import spearmanr
    m = np.isfinite(p) & np.isfinite(t)
    if m.sum() < 30:
        return float("nan")
    return round(float(spearmanr(p[m], t[m]).correlation), 3)


def block_shuffle(s: pd.Series, block: int = 30, rng=None) -> pd.Series:
    """Resample in blocks: preserves the marginal AND the autocorrelation (hence the
    exposure profile and turnover) while destroying TIMING. This is the control that
    a matched CONSTANT multiplier cannot provide — the champion is vol-targeted, so
    champ*{0.4..1.0} is Sharpe-neutral and free to beat."""
    rng = rng or np.random.default_rng(0)
    v = s.values.copy()
    fin = np.where(np.isfinite(v))[0]
    if len(fin) < block * 3:
        return s
    seg = v[fin]
    nb = int(np.ceil(len(seg) / block))
    starts = rng.integers(0, max(1, len(seg) - block), size=nb)
    out = np.concatenate([seg[st:st + block] for st in starts])[:len(seg)]
    new = v.copy()
    new[fin] = out
    return pd.Series(new, index=s.index)


def approach4(cost_usd: float | None, ks=(24, 48), qs=(0.30,), kind: str = "dsq") -> None:
    df = load_h4(cost_usd)
    close = df["close"]
    champ_fc = champion_signal(close)
    F = build_features(df)
    ev = f"{OOS_START_YEAR}-01-01"
    lab = "blessed CSV" if cost_usd is None else f"${cost_usd:.3f} floor"

    ch_m, ch_d = engine(df, champ_fc, eval_start=ev)
    bh_m, bh_d = engine(df, pd.Series(1.0, index=df.index), eval_start=ev)
    NAMES = {"dsq": "FORWARD-DOWNSIDE (max adverse excursion)",
             "econ": "ECONOMIC (forward k-bar return)",
             "champmeta": "CHAMPION-TRADE META (champion's own fwd P&L)"}
    print(f"\n=== {NAMES[kind]} PROBABILITY -> TRIM  (cost {lab}) ===")
    print(f"    walk-forward: expanding yearly refits, first OOS {OOS_START_YEAR}, "
          f"purge {PURGE} bars + label window k")
    print(f"    eval {ev}+   CHAMPION SR {ch_m['sharpe']:+.3f}   "
          f"buy&hold(vt) SR {bh_m['sharpe']:+.3f}\n")

    rows = []
    for k in ks:
        for q in qs:
            tgt = badness(df, champ_fc, kind, k)
            p, folds = walk_forward_prob(df, F, k, q, tgt)
            FD = pd.DataFrame(folds)
            cov = float(p.notna().mean())
            print(f"--- k={k} q={q:.2f} --- OOS bars {int(p.notna().sum())} ({cov*100:.0f}%)")
            print(f"    per-fold AUC: " + " ".join(f"{r.year}:{r.auc}" for r in FD.itertuples()))
            print(f"    per-fold  IC: " + " ".join(f"{r.year}:{r.ic}" for r in FD.itertuples()))
            mean_auc = float(FD.auc.mean())
            ic_all = _ic(p.values, tgt.values)
            sign_ok = int((FD.ic < 0).sum())   # p_bad HIGH should mean fwd_min LOW
            # ---- STAGE 1: information gates
            msk = p.notna() & tgt.notna()
            top = p[msk] >= p[msk].quantile(0.90)
            dec_spread = float(tgt[msk][top].mean() - tgt[msk].mean())
            # block-permutation null for IC
            rng = np.random.default_rng(11)
            null = [_ic(block_shuffle(p, 30, rng).values, tgt.values) for _ in range(60)]
            null_sd = float(np.nanstd(null))
            print(f"    mean AUC {mean_auc:.3f} | full IC {ic_all:+.4f} "
                  f"(null sd {null_sd:.4f}, gate |IC|>={2*null_sd:.4f}) | "
                  f"folds with correct sign {sign_ok}/{len(FD)}")
            sd_t = float(tgt[msk].std())
            spread_gate = -0.15 if kind == 'dsq' else -0.15 * sd_t
            print(f"    top-decile fwd-badness spread {dec_spread:+.4f} "
                  f"(gate <= {spread_gate:+.4f})  coverage {float(top.mean()):.2f}")
            s1 = (abs(ic_all) >= 2*null_sd) and (sign_ok >= 7) and (dec_spread <= spread_gate)
            print(f"    STAGE 1 {'PASS' if s1 else 'FAIL'}")

            # ---- STAGE 2: paired overlay vs champion + shuffled-p null
            pf = p.fillna(0.0)
            for b in (0.5, 1.0):
                fc = champ_fc * (1.0 - b * pf)
                m, d = engine(df, fc, eval_start=ev)
                dm, t, n = paired(d, ch_d)
                yp, yn = per_year(d, ch_d)
                # shuffled-p null (timing destroyed, exposure preserved)
                rng2 = np.random.default_rng(5)
                nulls = []
                for _ in range(60):
                    ps = block_shuffle(p, 30, rng2).fillna(0.0)
                    mm, _ = engine(df, champ_fc * (1.0 - b * ps), eval_start=ev)
                    nulls.append(mm["sharpe"])
                p95 = float(np.percentile(nulls, 95))
                dsr = m["sharpe"] - ch_m["sharpe"]
                ok2 = (dsr >= 0.20) and (t >= 2.0) and (m["sharpe"] > p95) and (yp >= 7)
                print(f"    trim b={b}: SR {m['sharpe']:+.3f} (dSR {dsr:+.3f}) t {t:+.2f} "
                      f"turn {m['turnover']:.1f} dd {m['dd']:+.1f}% yrs+ {yp}/{yn} | "
                      f"shuffled-p null p95 {p95:+.3f} | STAGE 2 "
                      f"{'PASS' if ok2 else 'FAIL'}")
                rows.append(dict(k=k, q=q, b=b, sharpe=m["sharpe"], dsr=round(dsr, 3),
                                 t=round(t, 2), turnover=m["turnover"], dd=m["dd"],
                                 yrs_pos=yp, yrs=yn, null_p95=round(p95, 3),
                                 mean_auc=round(mean_auc, 3), ic=ic_all,
                                 null_sd=round(null_sd, 4), stage1=s1, stage2=ok2,
                                 kind=kind))
    R = pd.DataFrame(rows)
    out = ROOT / "data" / "v5_runs" / "turnprob"
    out.mkdir(parents=True, exist_ok=True)
    tag = ("blessed" if cost_usd is None else f"{cost_usd:.3f}") + f"_{kind}"
    R.to_csv(out / f"approach4_{tag}.csv", index=False)
    print(f"\nwrote {out / f'approach4_{tag}.csv'}")
    if not R.empty and R.stage2.any():
        print("VERDICT: a cell PASSED Stage 2 -> proceed to Stage 3 sizing + certification")
    else:
        print("VERDICT: no cell passed Stage 2 -> DISPROVEN at this stage; do not deploy")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--approach", type=int, default=1)
    ap.add_argument("--label", default="dsq", choices=["dsq","econ","champmeta"])
    ap.add_argument("--cost", default="blessed",
                    help="blessed | live | stress | a $ figure e.g. 0.448")
    args = ap.parse_args()
    cost = COST_LEVELS.get(args.cost, None) if args.cost in COST_LEVELS else float(args.cost)
    if args.approach == 1:
        approach1(cost)
    elif args.approach == 4:
        approach4(cost, kind=args.label)
    else:
        raise SystemExit(f"approach {args.approach} not implemented yet")


if __name__ == "__main__":
    main()
