"""XAUUSD H4 trend/turning-point PREDICTION ACCURACY — new feature families.

Pure classification accuracy, NOT P&L (per user request 2026-08-10: "i dont care
the profitability for now, we need to increase accuracy"). This extends the
established turning-point benchmark (`scripts/v5_xau_turning_ml.py`, price-only
features: OOS BUY 71% precision @ 22% recall tol=3 [lift 2.1x]; SELL stuck at
~9% recall @ 70% precision — see memory `xau-turning-point-detectability`) with
FOUR feature families that prior XAU studies in this repo never fed a classifier:

  ZZSTATE   causal zigzag STATE (not the label): direction of the last CONFIRMED
            leg, bars since that pivot, running extension in ATR units, amplitude
            of the last leg, streak of same-direction pivots. Zero lookahead —
            distinct from src/features/trend_labels.zigzag_labels which is a
            TARGET generator (uses the future tail on purpose).
  REGIME    Hurst exponent (rolling generalized-variance estimator), ADX(14),
            Choppiness Index(14), short/long ATR ratio, D1 EMA-slope + z-score
            (a higher-timeframe filter), all lagged so nothing peeks past the
            bar being predicted.
  INTER     intermarket panel, lagged 1 full calendar day so a still-open daily
            bar can never leak in: DXY proxy (weighted FX basket), UST10Y/UST2Y
            (bond price, i.e. INVERSE real-yield proxy) + curve slope, SPX/NDX/DJI,
            BTC, SILVER/PLAT/PALL (+ gold/silver ratio), COPPER, WTI. This is the
            genuinely new angle: prior turning-point work here used price-derived
            features only, despite the repo already holding this data.
  (PRICE is the reproduced baseline: multi-window z-score/percentile-rank, RSI,
   ATR-normalised momentum/accel, wick rejection, RSI divergence, streaks, hour.)

REGIME (esp. its default Hurst window=100) turned out to be the one family that
actually moves accuracy (see memory `xau-regime-features-fwd-accuracy`: +2.12pp
over the persistence baseline on fwd6, block-bootstrap confirmed, 8/9 years).
2026-08-10 round 2 pushes on THAT thread with three more causal families, all
layered ON TOP of price+regime rather than replacing it:
  HURSTMULTI  Hurst exponent at 6 windows (50..300) in parallel, instead of
              picking one — lets the model use whichever scale carries signal.
  HMM         a hand-rolled 3-state Gaussian HMM (diagonal cov, scaled
              Baum-Welch) on (return, vol-ratio), refit every walk-forward
              year on strictly-past data, decoded with a forward-ONLY filter
              (not hmmlearn's smoothing decode, which would leak future bars
              into an earlier state estimate) — state probabilities + regime
              persistence (bars in the current state).
  WAVELET     causal rolling Haar multiresolution decomposition of log-returns
              (trailing 64-bar window, 4 dyadic scales): each scale's share of
              total energy, i.e. is recent action choppy noise (short-scale
              energy) or a smooth extended run (long-scale/residual-trend
              energy) — a spectral profile, not a single summary number like
              Hurst/Choppiness.
`--hurst-sweep` runs a cheap fwd6/HistGBoost-only comparison across Hurst
windows on their own, if you just want that answer without the full grid.

Two target families, both forward-looking BY DESIGN (targets only — every
feature above is strictly causal):
  pivot   near-swing labels from `v5_xau_turning_ml.zigzag_swings` (order=5,
          theta=1.5*ATR, tol=3) — SAME ground truth as the benchmark, so numbers
          are directly comparable.
  fwd6    sign(close[t+6] - close[t]) — "will price be higher in ~1 trading day"
          (H4 x 6 bars), the literal "predict the short trend" framing. Baseline
          is NOT 50/50 — it's the PERSISTENCE rule (repeat the sign of the last
          6-bar return), the honest random-walk-aware benchmark.

Walk-forward, expanding, refit each calendar year on strictly-past data, purge
gap = tol (pivot) or H (fwd6) bars at the train/test boundary, eval pooled
2018+ (matches V5_FINDINGS MANDATORY CONTROLS: walk-forward per year, not one
split). Ablation = which feature blocks are fed to which model, so we can see
WHICH family moves accuracy, not just a final number.

    python scripts/v5_xau_intermarket_accuracy.py
    python scripts/v5_xau_intermarket_accuracy.py --target fwd6 --models histgb,xgb
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
warnings.filterwarnings("ignore")

EVAL_START_YEAR = 2018   # first walk-forward test year (matches repo convention)
TOL = 3                   # pivot tolerance, bars — same as v5_xau_turning_ml
ORDER = 5
THETA_MULT = 1.5
FWD_H = 6                 # ~1 trading day on H4

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except ImportError:
    HAVE_LGBM = False


# ============================================================== data loading
def load_xau_h4() -> pd.DataFrame:
    d = pd.read_csv(DATA / "XAUUSD_H4_long.csv", parse_dates=["time"])
    return d.set_index("time").sort_index()


def load_d1(sym: str) -> pd.DataFrame | None:
    fp = DATA / f"{sym}_D1_long.csv"
    if not fp.exists():
        return None
    d = pd.read_csv(fp, parse_dates=["time"])
    return d.set_index("time").sort_index()[["open", "high", "low", "close"]]


def atr(d: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([(d.high - d.low), (d.high - d.close.shift()).abs(),
                    (d.low - d.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


# ======================================================== ground-truth pivots
def zigzag_swings(d: pd.DataFrame, order: int, theta: pd.Series):
    """Verbatim logic from v5_xau_turning_ml.zigzag_swings (kept local so this
    script has no import-order coupling to another research script)."""
    hi, lo = d.high.values, d.low.values
    imax = set(argrelextrema(hi, np.greater_equal, order=order)[0])
    imin = set(argrelextrema(lo, np.less_equal, order=order)[0])
    cand = sorted([(i, "H") for i in imax] + [(i, "L") for i in imin])
    piv = []
    for i, t in cand:
        if piv and piv[-1][1] == t:
            j, _ = piv[-1]
            if (hi[i] > hi[j]) if t == "H" else (lo[i] < lo[j]):
                piv[-1] = (i, t)
        else:
            piv.append((i, t))
    kept = []
    for i, t in piv:
        p = hi[i] if t == "H" else lo[i]
        if not kept:
            kept.append((i, t, p)); continue
        j, tj, pj = kept[-1]
        if abs(p - pj) >= theta.iloc[i]:
            kept.append((i, t, p))
        elif t == tj and ((t == "H" and p > pj) or (t == "L" and p < pj)):
            kept[-1] = (i, t, p)
    sells = np.array([i for i, t, _ in kept if t == "H"], int)
    buys = np.array([i for i, t, _ in kept if t == "L"], int)
    return sells, buys


def label_near(truth: np.ndarray, n: int, tol: int) -> np.ndarray:
    y = np.zeros(n, dtype=int)
    for i in truth:
        y[max(0, i - tol):min(n, i + tol + 1)] = 1
    return y


# ================================================================ PRICE block
def price_features(d: pd.DataFrame) -> pd.DataFrame:
    """Reproduces v5_xau_turning_ml.features() — the established baseline."""
    c, h, l = d.close, d.high, d.low
    a = atr(d)
    rng = (h - l).replace(0, np.nan)
    f = pd.DataFrame(index=d.index)
    for w in (10, 20, 50):
        m = c.rolling(w).mean(); sd = c.rolling(w).std()
        f[f"z{w}"] = (c - m) / sd
        f[f"pctl{w}"] = c.rolling(w).apply(lambda x: (x[-1] > x).mean(), raw=True)
    for n in (7, 14):
        f[f"rsi{n}"] = rsi(c, n)
    f["ema_dist"] = (c - c.ewm(span=50).mean()) / a
    f["mom"] = (c - c.shift(5)) / a
    f["accel"] = f["mom"] - (c.shift(5) - c.shift(10)) / a
    f["upwick"] = (h - np.maximum(c, d.open)) / rng
    f["lowick"] = (np.minimum(c, d.open) - l) / rng
    f["clpos"] = (c - l) / rng
    f["atr_z"] = (a - a.rolling(100).mean()) / a.rolling(100).std()
    up = (c.diff() > 0).astype(int); dn = (c.diff() < 0).astype(int)
    f["run_up"] = up * (up.groupby((up != up.shift()).cumsum()).cumcount() + 1)
    f["run_dn"] = dn * (dn.groupby((dn != dn.shift()).cumsum()).cumcount() + 1)
    r14 = f["rsi14"]
    f["bull_div"] = ((l <= l.rolling(20).min()) & (r14 > r14.rolling(20).min() + 3)).astype(int)
    f["bear_div"] = ((h >= h.rolling(20).max()) & (r14 < r14.rolling(20).max() - 3)).astype(int)
    f["hour"] = d.index.hour
    f["dow"] = d.index.dayofweek
    return f


# =============================================================== ZZSTATE block
def zzstate_features(d: pd.DataFrame, a: pd.Series, k: float = THETA_MULT) -> pd.DataFrame:
    """Causal zigzag STATE (not the retrospective label). At every bar i, only
    bars <= i are used: direction/age/extension of the swing-in-progress and
    stats of the last CONFIRMED leg. Same ATR-pivot mechanics as
    src/features/trend_labels.zigzag_labels, but exposing state instead of
    assigning the (forward-looking) segment label."""
    h = d.high.values; lo = d.low.values; c = d.close.values; av = a.values
    n = len(d)
    a_mean = float(np.nanmean(av)) if np.isfinite(np.nanmean(av)) else 0.0
    dirn = np.zeros(n); age = np.zeros(n); ext = np.zeros(n)
    last_amp = np.zeros(n); streak = np.zeros(n)
    last_pivot, hi_i, lo_i, direction = 0, 0, 0, 0
    confirmed_dir, confirmed_streak, last_leg_amp = 0, 0, 0.0
    for i in range(1, n):
        if h[i] >= h[hi_i]: hi_i = i
        if lo[i] <= lo[lo_i]: lo_i = i
        thr = k * (av[i] if np.isfinite(av[i]) and av[i] > 0 else a_mean)
        if thr > 0:
            if direction >= 0 and h[hi_i] - lo[i] >= thr and hi_i > last_pivot:
                amp = c[hi_i] - c[last_pivot]
                new_dir = int(np.sign(amp)) or 1
                confirmed_streak = confirmed_streak + 1 if new_dir == confirmed_dir else 1
                confirmed_dir, last_leg_amp = new_dir, abs(amp)
                last_pivot, direction, hi_i, lo_i = hi_i, -1, i, i
            elif direction <= 0 and h[i] - lo[lo_i] >= thr and lo_i > last_pivot:
                amp = c[lo_i] - c[last_pivot]
                new_dir = int(np.sign(amp)) or -1
                confirmed_streak = confirmed_streak + 1 if new_dir == confirmed_dir else 1
                confirmed_dir, last_leg_amp = new_dir, abs(amp)
                last_pivot, direction, hi_i, lo_i = lo_i, 1, i, i
        # state as known causally AT bar i (post any confirmation above)
        dirn[i] = confirmed_dir
        age[i] = i - last_pivot
        run_extreme = h[hi_i] if direction >= 0 else lo[lo_i]
        ext[i] = (run_extreme - c[last_pivot]) / (av[i] if av[i] > 0 else a_mean)
        last_amp[i] = last_leg_amp / (av[i] if av[i] > 0 else a_mean)
        streak[i] = confirmed_streak * confirmed_dir
    f = pd.DataFrame(index=d.index)
    f["zz_dir"] = dirn
    f["zz_age"] = age
    f["zz_ext_atr"] = ext
    f["zz_last_amp_atr"] = last_amp
    f["zz_streak"] = streak
    return f


# ================================================================ REGIME block
def hurst_rolling(close: pd.Series, window: int = 100) -> pd.Series:
    """Generalized Hurst exponent via variance scaling at lags {1,2,4,8,16},
    rolling & causal. H~0.5 random walk, >0.5 trending, <0.5 mean-reverting."""
    lags = np.array([1, 2, 4, 8, 16])
    logc = np.log(close.values)

    def _h(x):
        if len(x) < window:
            return np.nan
        vs = []
        for lag in lags:
            diffs = x[lag:] - x[:-lag]
            v = np.var(diffs)
            vs.append(v if v > 0 else np.nan)
        vs = np.array(vs)
        ok = np.isfinite(vs) & (vs > 0)
        if ok.sum() < 3:
            return np.nan
        slope = np.polyfit(np.log(lags[ok]), np.log(vs[ok]), 1)[0]
        return slope / 2.0

    out = pd.Series(logc, index=close.index).rolling(window).apply(_h, raw=True)
    return out.rename("hurst")


def hurst_multi_features(close: pd.Series, windows=(50, 75, 100, 150, 200, 300)) -> pd.DataFrame:
    """Multiple Hurst windows as parallel features, so a scale that the single
    default window=100 misses (e.g. faster regime flips at 50, slower
    persistent trends at 300) is still available to the model."""
    f = pd.DataFrame(index=close.index)
    for w in windows:
        f[f"hurst_w{w}"] = hurst_rolling(close, w)
    return f


def choppiness(d: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([(d.high - d.low), (d.high - d.close.shift()).abs(),
                    (d.low - d.close.shift()).abs()], axis=1).max(axis=1)
    atr_sum = tr.rolling(n).sum()
    rng = d.high.rolling(n).max() - d.low.rolling(n).min()
    return (100 * np.log10(atr_sum / rng.replace(0, np.nan)) / np.log10(n)).rename("chop")


def regime_features(d: pd.DataFrame, a: pd.Series, hurst_window: int = 100) -> pd.DataFrame:
    from src.features.indicators import adx as adx_fn
    f = pd.DataFrame(index=d.index)
    f["hurst100"] = hurst_rolling(d.close, hurst_window)
    f["adx14"] = adx_fn(d, 14)
    f["chop14"] = choppiness(d, 14)
    f["vol_ratio"] = a / d.high.sub(d.low).abs().ewm(alpha=1 / 50, adjust=False).mean().add(
        (d.high - d.close.shift()).abs().ewm(alpha=1 / 50, adjust=False).mean()).replace(0, np.nan)
    f["vol_ratio"] = atr(d, 14) / atr(d, 50)   # simpler & robust: short/long ATR ratio
    # D1 higher-timeframe filter, built from XAU's own H4->D1 resample (no lookahead:
    # each D1 bar is only "known" once that calendar day is over -> lag 1 day, then
    # asof-align back onto H4, same mechanism as the intermarket panel below).
    d1 = d.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    d1_ema = d1.close.ewm(span=20).mean()
    d1_slope = np.sign(d1_ema.diff())
    d1_z = (d1.close - d1.close.rolling(20).mean()) / d1.close.rolling(20).std()
    avail = d1.index + pd.Timedelta(days=1)
    d1f = pd.DataFrame({"d1_slope": d1_slope.values, "d1_z": d1_z.values}, index=avail).sort_index()
    d1f.index.name = "avail_time"
    merged = pd.merge_asof(pd.DataFrame(index=d.index).reset_index(), d1f.reset_index(),
                            left_on="time", right_on="avail_time", direction="backward").set_index("time")
    f["d1_slope"] = merged["d1_slope"].values
    f["d1_z"] = merged["d1_z"].values
    return f


# ================================================================== HMM block
class DiagGaussianHMM:
    """Minimal K-state Gaussian HMM, diagonal covariance, fit via scaled
    Baum-Welch EM. Hand-rolled (no hmmlearn — this session's network took
    25+ min and never finished installing xgboost/lightgbm, too risky to
    depend on another slow install for a small K=3, D=2 model).

    `filter_causal` is the point of writing this by hand: hmmlearn's public
    decode/predict_proba run smoothing (forward-BACKWARD) over the whole
    sequence passed in, which uses FUTURE observations to inform an earlier
    bar's state probability — leakage if used as a live feature. The
    forward-only alpha pass here uses strictly X[0..t] for row t, so once the
    model is fit on past-only data it is safe to filter forward through a
    test period."""

    def __init__(self, n_states: int = 3, n_iter: int = 25, seed: int = 7):
        self.k = n_states
        self.n_iter = n_iter
        self.rng = np.random.default_rng(seed)

    def _emission_prob(self, X: np.ndarray, means: np.ndarray, varr: np.ndarray) -> np.ndarray:
        var = np.maximum(varr, 1e-8)
        diff2 = (X[:, None, :] - means[None, :, :]) ** 2
        logB = -0.5 * (np.log(2 * np.pi * var).sum(axis=1)[None, :] +
                       (diff2 / var[None, :, :]).sum(axis=2))
        return np.exp(logB - logB.max(axis=1, keepdims=True))   # rescaled, not normalised — fine, c[t] normalises

    def fit(self, X: np.ndarray) -> "DiagGaussianHMM":
        T, D = X.shape
        k = self.k
        order = np.argsort(X[:, 0])
        means = np.array([X[order[int(i * T / k):int((i + 1) * T / k)]].mean(axis=0) for i in range(k)])
        varr = np.tile(X.var(axis=0) + 1e-6, (k, 1))
        A = np.full((k, k), 0.1 / max(k - 1, 1)); np.fill_diagonal(A, 0.9)
        pi = np.full(k, 1.0 / k)

        for _ in range(self.n_iter):
            B = self._emission_prob(X, means, varr)
            alpha = np.zeros((T, k)); c = np.zeros(T)
            ah = pi * B[0]; c[0] = ah.sum() + 1e-300; alpha[0] = ah / c[0]
            for t in range(1, T):
                ah = (alpha[t - 1] @ A) * B[t]
                c[t] = ah.sum() + 1e-300
                alpha[t] = ah / c[t]
            beta = np.zeros((T, k)); beta[T - 1] = 1.0
            for t in range(T - 2, -1, -1):
                beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
            xi_sum = np.zeros((k, k))
            for t in range(T - 1):
                xi_sum += (alpha[t][:, None] * A * (B[t + 1] * beta[t + 1])[None, :]) / c[t + 1]
            pi = gamma[0]
            A = xi_sum / (gamma[:-1].sum(axis=0)[:, None] + 1e-300)
            for j in range(k):
                w = gamma[:, j]; wsum = w.sum() + 1e-300
                means[j] = (w[:, None] * X).sum(axis=0) / wsum
                varr[j] = (w[:, None] * (X - means[j]) ** 2).sum(axis=0) / wsum + 1e-6
        self.means_, self.varr_, self.A_, self.pi_ = means, varr, A, pi
        return self

    def filter_causal(self, X: np.ndarray) -> np.ndarray:
        """Forward-only filtered P(state_t | obs_0..t), shape (T, K)."""
        T, k = len(X), self.k
        B = self._emission_prob(X, self.means_, self.varr_)
        alpha = np.zeros((T, k))
        ah = self.pi_ * B[0]; alpha[0] = ah / (ah.sum() + 1e-300)
        for t in range(1, T):
            ah = (alpha[t - 1] @ self.A_) * B[t]
            alpha[t] = ah / (ah.sum() + 1e-300)
        return alpha


def hmm_obs(d: pd.DataFrame, a: pd.Series) -> pd.DataFrame:
    """2D (return, vol-ratio) observation pair — the classic Hamilton-style
    regime-switching input. Distinct from REGIME's summary stats: the HMM's
    signal is the TRANSITION dynamics (how sticky/persistent each regime is),
    not an instantaneous reading."""
    ret = (np.log(d.close).diff() / a.replace(0, np.nan)).rename("r")
    chg = d.close.pct_change()
    vol = (chg.rolling(10).std() / chg.rolling(50).std()).rename("v")
    return pd.concat([ret, vol], axis=1)


def hmm_regime_walkforward(d: pd.DataFrame, a: pd.Series, years: list[int],
                            n_states: int = 3, warmup_years: int = 2) -> pd.DataFrame:
    """Per-year refit, SAME expanding-window discipline as the classifiers:
    fit on data strictly before the test year, then causally filter from
    series start through that year's end with THAT year's own parameters.
    No target-label purge needed — the HMM never sees a label, only past
    OHLC, so there is nothing to leak beyond 'not using future price bars'.

    Fills starting `warmup_years` BEFORE the first requested eval year, not
    just the eval years themselves — a classifier's earliest eval fold still
    needs several years of PRE-eval training data, and if this only filled
    the eval years, every pre-eval training row would be NaN in the HMM
    columns, silently zeroing out that fold's training set (caught via a
    smoke test: the 2024-only run below showed train_mask.sum()==0 for the
    first eval year until this was fixed)."""
    obs = hmm_obs(d, a)
    idx = d.index
    cols = [f"hmm_p{k}" for k in range(n_states)] + ["hmm_persist"]
    out = pd.DataFrame(np.nan, index=idx, columns=cols)
    ok = obs.notna().all(axis=1).values
    fit_years = range(min(years) - warmup_years, max(years) + 1) if years else []
    for yr in fit_years:
        train_end = np.searchsorted(idx.values, np.datetime64(pd.Timestamp(f"{yr}-01-01")))
        year_end = np.searchsorted(idx.values, np.datetime64(pd.Timestamp(f"{yr + 1}-01-01")))
        train_mask = ok.copy(); train_mask[train_end:] = False
        if train_mask.sum() < 500:
            continue
        hmm = DiagGaussianHMM(n_states=n_states, n_iter=25, seed=7).fit(obs.values[train_mask])
        seg_mask = ok.copy(); seg_mask[year_end:] = False
        seg_idx = np.where(seg_mask)[0]
        probs = hmm.filter_causal(obs.values[seg_mask])
        test_idx = seg_idx[(seg_idx >= train_end) & (seg_idx < year_end)]
        pos_in_seg = np.searchsorted(seg_idx, test_idx)
        for j in range(n_states):
            out.iloc[test_idx, j] = probs[pos_in_seg, j]
        state = probs.argmax(axis=1)
        persist = np.zeros(len(state))
        run = 0
        for i in range(len(state)):
            run = run + 1 if (i == 0 or state[i] == state[i - 1]) else 1
            persist[i] = run
        out.iloc[test_idx, n_states] = persist[pos_in_seg]
    return out


# =============================================================== WAVELET block
def wavelet_features(close: pd.Series, window: int = 64, levels: int = 4) -> pd.DataFrame:
    """Causal rolling Haar multiresolution decomposition of log-returns: for
    every bar, decompose the TRAILING `window` returns into `levels` dyadic
    scales (2,4,8,16-bar cycles at window=64,levels=4) and report each
    scale's share of total energy — a spectral fingerprint of whether recent
    price action is choppy noise (energy at short scales) or a smooth
    extended run (energy at long scales / residual trend). Distinct from
    Hurst/Choppiness (single summary numbers): this is the full profile
    across scales. Strictly causal — each row's window ends AT that bar, no
    future returns enter it."""
    logret = np.log(close).diff().fillna(0.0).values
    n = len(logret)
    out_cols = [f"wv_e{i + 1}_frac" for i in range(levels)] + ["wv_trend_frac"]
    out = pd.DataFrame(np.nan, index=close.index, columns=out_cols)
    if n < window:
        return out
    W = np.lib.stride_tricks.sliding_window_view(logret, window).copy()  # (n-window+1, window)
    cur = W
    energies = []
    for _ in range(levels):
        L = cur.shape[1] - (cur.shape[1] % 2)
        cur = cur[:, :L]
        pairs = cur.reshape(cur.shape[0], L // 2, 2)
        avg = pairs.mean(axis=2)
        diff = (pairs[:, :, 0] - pairs[:, :, 1]) / 2.0
        energies.append(np.mean(diff ** 2, axis=1))
        cur = avg
    total = np.sum(energies, axis=0) + np.mean(cur ** 2, axis=1) + 1e-12
    for i, e in enumerate(energies):
        out.iloc[window - 1:, i] = e / total
    out.iloc[window - 1:, levels] = np.mean(cur ** 2, axis=1) / total
    return out


# =============================================================== INTER block
DXY_WEIGHTS = {  # renormalised ICE-DXY weights over the pairs we have (missing SEK ~4.2%)
    "EURUSD": -0.601, "USDJPY": 0.142, "GBPUSD": -0.124,
    "USDCAD": 0.095, "USDCHF": 0.038,
}
COMPLEX = ["SILVER", "PLAT", "PALL", "COPPER", "WTI", "BRENT", "SPX", "NDX", "DJI", "BTC",
           "UST10Y", "UST2Y"]


def intermarket_panel(h4_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily intermarket features, lagged 1 day and asof-aligned onto the H4
    index so nothing from a still-open daily bar can leak into an H4 feature.

    Each instrument has its OWN holiday/weekend calendar (equities closed
    weekends, bonds closed some days equities trade, crypto trades every day,
    ...). Naively pd.DataFrame(dict-of-Series)-ing them unions the indices, so
    a day only one asset is missing produces a row with a real NaN for that
    column even though its last value is only a day or two stale — and
    merge_asof then locks the WHOLE row (all columns) to that nearest-in-time
    but partially-NaN row instead of each column's own last known value. Fix:
    forward-fill every close series independently across a shared daily
    calendar FIRST, so no column's carried-forward value is lost to another
    asset's gap."""
    daily = {}
    for sym in list(DXY_WEIGHTS) + COMPLEX:
        dd = load_d1(sym)
        if dd is None:
            print(f"  ! intermarket series missing, skipped: {sym}")
            continue
        daily[sym] = dd.close

    cal = pd.date_range(min(s.index.min() for s in daily.values()),
                         max(s.index.max() for s in daily.values()), freq="D")
    daily = {sym: s.reindex(cal).ffill() for sym, s in daily.items()}

    dxy_ret = None
    for sym, w in DXY_WEIGHTS.items():
        if sym not in daily:
            continue
        r = np.log(daily[sym]).diff() * w
        dxy_ret = r if dxy_ret is None else dxy_ret.add(r, fill_value=0)
    feats = {}
    if dxy_ret is not None:
        feats["dxy_ret"] = dxy_ret
        feats["dxy_z"] = (dxy_ret.rolling(20).mean()) / dxy_ret.rolling(20).std()

    for sym in COMPLEX:
        if sym not in daily:
            continue
        c = daily[sym]
        r = np.log(c).diff()
        feats[f"{sym.lower()}_ret"] = r
        feats[f"{sym.lower()}_z"] = (c - c.rolling(20).mean()) / c.rolling(20).std()

    if "SILVER" in daily:
        xau_d1 = load_d1("GOLD")
        if xau_d1 is not None:
            gold_c = xau_d1.close.reindex(cal).ffill()
            ratio = gold_c / daily["SILVER"]
            feats["gs_ratio_z"] = (ratio - ratio.rolling(60).mean()) / ratio.rolling(60).std()
    if "UST10Y" in daily and "UST2Y" in daily:
        # bond PRICE curve (inverse of the usual yield curve, fine for ML either way)
        feats["curve_slope"] = daily["UST10Y"] - daily["UST2Y"]

    panel = pd.DataFrame(feats).sort_index().dropna(how="all")
    avail = panel.index + pd.Timedelta(days=1)
    panel = panel.set_index(avail)
    panel.index.name = "avail_time"
    merged = pd.merge_asof(pd.DataFrame(index=h4_index).reset_index(), panel.reset_index(),
                            left_on="time", right_on="avail_time", direction="backward").set_index("time")
    merged = merged.drop(columns=["avail_time"], errors="ignore")
    return merged


# =================================================================== targets
def make_targets(d: pd.DataFrame, a: pd.Series):
    n = len(d)
    theta = THETA_MULT * a
    sells, buys = zigzag_swings(d, ORDER, theta)
    y_buy = label_near(buys, n, TOL)
    y_sell = label_near(sells, n, TOL)
    fwd_ret = d.close.shift(-FWD_H) - d.close
    y_fwd = (fwd_ret > 0).astype(int)
    y_fwd[fwd_ret.isna()] = -1          # sentinel: no forward data (tail), excluded later
    return dict(buy=y_buy, sell=y_sell, fwd6=y_fwd.values), dict(buy=buys, sell=sells)


# =============================================================== model zoo
def build_models(names: list[str]) -> dict:
    zoo = {}
    if "logreg" in names:
        zoo["logreg"] = ("scale", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"))
    if "rf" in names:
        zoo["rf"] = ("raw", RandomForestClassifier(n_estimators=300, max_depth=6,
                                                    min_samples_leaf=50, n_jobs=-1,
                                                    class_weight="balanced_subsample", random_state=7))
    if "histgb" in names:
        zoo["histgb"] = ("raw", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                                max_depth=4, l2_regularization=1.0,
                                                                validation_fraction=0.15, random_state=7))
    if "xgb" in names and HAVE_XGB:
        zoo["xgb"] = ("raw", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                           subsample=0.8, colsample_bytree=0.8,
                                           reg_lambda=1.0, eval_metric="logloss",
                                           n_jobs=-1, random_state=7))
    if "lgbm" in names and HAVE_LGBM:
        zoo["lgbm"] = ("raw", LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                             subsample=0.8, colsample_bytree=0.8,
                                             reg_lambda=1.0, verbosity=-1, random_state=7))
    return zoo


# ============================================================ walk-forward
def walk_forward(X: pd.DataFrame, y: np.ndarray, model_kind, model, purge: int,
                  years: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Expanding-window, refit each calendar year on strictly-past data with a
    `purge`-bar gap before the test year, pooled OOS predictions returned."""
    idx = X.index
    valid = np.isfinite(X.values).all(axis=1) & (y >= 0)
    oos_p, oos_y = [], []
    for yr in years:
        test_mask = (idx.year == yr) & valid
        if test_mask.sum() == 0:
            continue
        cutoff = pd.Timestamp(f"{yr}-01-01")
        train_end_pos = np.searchsorted(idx.values, np.datetime64(cutoff)) - purge
        if train_end_pos < 200:
            continue
        train_mask = np.zeros(len(idx), dtype=bool)
        train_mask[:train_end_pos] = True
        train_mask &= valid
        if train_mask.sum() < 200:
            continue
        Xtr, ytr = X.values[train_mask], y[train_mask]
        Xte, yte = X.values[test_mask], y[test_mask]
        if len(np.unique(ytr)) < 2:
            continue
        if model_kind == "scale":
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        m = model.__class__(**model.get_params())
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        oos_p.append(p); oos_y.append(yte)
    if not oos_p:
        return np.array([]), np.array([])
    return np.concatenate(oos_p), np.concatenate(oos_y)


def score_pivot(p: np.ndarray, y: np.ndarray) -> dict:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return dict(n=len(y), base=np.nan, ap=np.nan, p_at_r20=np.nan, p_at_r10=np.nan,
                    rec_at_p70=np.nan)
    base = y.mean()
    ap = average_precision_score(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    def p_at_r(target):
        mask = rec[:-1] >= target
        return float(prec[:-1][mask].max()) if mask.any() else np.nan
    def r_at_p(target):
        mask = prec[:-1] >= target
        return float(rec[:-1][mask].max()) if mask.any() else 0.0
    return dict(n=len(y), base=base, ap=ap, p_at_r20=p_at_r(0.20), p_at_r10=p_at_r(0.10),
                rec_at_p70=r_at_p(0.70))


def score_fwd(p: np.ndarray, y: np.ndarray, persist: np.ndarray) -> dict:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return dict(n=len(y), acc=np.nan, auc=np.nan, persist_acc=np.nan)
    pred = (p >= 0.5).astype(int)
    acc = (pred == y).mean()
    auc = roc_auc_score(y, p)
    persist_acc = (persist == y).mean() if len(persist) == len(y) else np.nan
    return dict(n=len(y), acc=acc, auc=auc, persist_acc=persist_acc)


# ==================================================================== main
def hurst_sweep(d, a, years, windows):
    """Cheap, targeted answer to 'which Hurst window is best': swap ONLY the
    Hurst window inside the already-proven price+regime champion, evaluate
    fwd6/HistGBoost only (the config with the block-bootstrap-confirmed
    edge), skip the full grid."""
    price = price_features(d)
    targets, _ = make_targets(d, a)
    y = targets["fwd6"]
    persist_fwd = np.sign(d.close - d.close.shift(FWD_H)).fillna(0).clip(lower=0).astype(int).values
    zoo = build_models(["histgb"])
    kind, model = zoo["histgb"]
    print(f"{'hurst_window':12} {'n':>6} {'acc':>7} {'auc':>7} {'delta_vs_persist':>16}")
    for w in windows:
        reg = regime_features(d, a, hurst_window=w)
        X = pd.concat([price, reg], axis=1)
        p, yte = walk_forward(X, y, kind, model, FWD_H, years)
        valid = np.isfinite(X.values).all(axis=1) & (y >= 0)
        idx = X.index
        persist_aligned = np.concatenate([persist_fwd[(idx.year == yr) & valid] for yr in years
                                          if ((idx.year == yr) & valid).sum() > 0])
        s = score_fwd(p, yte, persist_aligned)
        print(f"{w:12d} {s['n']:6.0f} {s['acc']*100:6.1f}% {s['auc']:7.3f} "
              f"{(s['acc']-s['persist_acc'])*100:+15.2f}pp")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="all", choices=["buy", "sell", "fwd6", "all"])
    ap.add_argument("--models", default="logreg,rf,histgb,xgb,lgbm")
    ap.add_argument("--ablations", default="price,zzstate,regime,inter,full")
    ap.add_argument("--eval-start-year", type=int, default=EVAL_START_YEAR)
    ap.add_argument("--hurst-sweep", action="store_true",
                    help="quick fwd6/histgb-only sweep of Hurst windows, then exit")
    ap.add_argument("--hurst-windows", default="50,75,100,150,200,300")
    args = ap.parse_args()

    print("Loading XAUUSD H4...")
    d = load_xau_h4()
    a = atr(d)
    years = list(range(args.eval_start_year, d.index.year.max() + 1))

    if args.hurst_sweep:
        hurst_sweep(d, a, years, [int(w) for w in args.hurst_windows.split(",")])
        return

    print("Building feature blocks (HMM/wavelet added on top of the price+regime champion)...")
    price = price_features(d)
    zz = zzstate_features(d, a)
    reg = regime_features(d, a)
    inter = intermarket_panel(d.index)
    hurstmulti = hurst_multi_features(d.close)
    hmm = hmm_regime_walkforward(d, a, years)
    wavelet = wavelet_features(d.close)
    print(f"  bars={len(d)}  price={price.shape[1]}f  zzstate={zz.shape[1]}f  "
          f"regime={reg.shape[1]}f  inter={inter.shape[1]}f  hurstmulti={hurstmulti.shape[1]}f  "
          f"hmm={hmm.shape[1]}f  wavelet={wavelet.shape[1]}f")

    blocks = {"price": price, "zzstate": zz, "regime": reg, "inter": inter,
             "hurstmulti": hurstmulti, "hmm": hmm, "wavelet": wavelet}
    abl_names = args.ablations.split(",")
    ablations = {
        "price": ["price"],
        "zzstate": ["price", "zzstate"],
        "regime": ["price", "regime"],
        "inter": ["price", "inter"],
        "full": ["price", "zzstate", "regime", "inter"],
        "hurstmulti": ["price", "regime", "hurstmulti"],
        "hmm": ["price", "regime", "hmm"],
        "wavelet": ["price", "regime", "wavelet"],
        "regime2": ["price", "regime", "hurstmulti", "hmm", "wavelet"],
    }
    ablations = {k: v for k, v in ablations.items() if k in abl_names}

    targets, truth = make_targets(d, a)
    tnames = list(targets) if args.target == "all" else [args.target]
    persist_fwd = np.sign(d.close - d.close.shift(FWD_H)).fillna(0).clip(lower=0).astype(int).values

    zoo = build_models(args.models.split(","))
    if not HAVE_XGB and "xgb" in args.models.split(","):
        print("  ! xgboost not installed — skipped (pip install xgboost)")
    if not HAVE_LGBM and "lgbm" in args.models.split(","):
        print("  ! lightgbm not installed — skipped (pip install lightgbm)")

    rows = []
    for tname in tnames:
        y = targets[tname]
        purge = FWD_H if tname == "fwd6" else TOL
        base_rate = (y[y >= 0].mean()) if tname != "fwd6" else np.nan
        n_truth = len(truth.get(tname, [])) if tname != "fwd6" else np.nan
        print(f"\n=== target={tname} " + (f"(n_swings={n_truth}) " if tname != "fwd6" else "") + "===")
        for abl_name, abl_blocks in ablations.items():
            X = pd.concat([blocks[b] for b in abl_blocks], axis=1)
            for mname, (kind, model) in zoo.items():
                p, yte = walk_forward(X, y, kind, model, purge, years)
                if tname == "fwd6":
                    pm = persist_fwd[np.isin(np.arange(len(d)), np.where(np.isin(d.index.year, years))[0])]
                    # recompute persist aligned to the same valid/test rows walk_forward used
                    valid = np.isfinite(X.values).all(axis=1) & (y >= 0)
                    persist_aligned = []
                    idx = X.index
                    for yr in years:
                        test_mask = (idx.year == yr) & valid
                        if test_mask.sum() == 0:
                            continue
                        persist_aligned.append(persist_fwd[test_mask])
                    persist_aligned = np.concatenate(persist_aligned) if persist_aligned else np.array([])
                    s = score_fwd(p, yte, persist_aligned)
                    rows.append(dict(target=tname, ablation=abl_name, model=mname, **s))
                else:
                    s = score_pivot(p, yte)
                    rows.append(dict(target=tname, ablation=abl_name, model=mname, **s))

        sub = pd.DataFrame([r for r in rows if r["target"] == tname])
        if tname == "fwd6":
            print(f"{'ablation':10} {'model':8} {'n':>6} {'acc':>7} {'auc':>7} {'persist_acc':>12}")
            for _, r in sub.iterrows():
                print(f"{r['ablation']:10} {r['model']:8} {r['n']:6.0f} {r['acc']*100:6.1f}% "
                      f"{r['auc']:7.3f} {r['persist_acc']*100:11.1f}%")
        else:
            print(f"{'ablation':10} {'model':8} {'n':>6} {'base':>6} {'PR-AUC':>7} "
                  f"{'P@R20':>7} {'P@R10':>7} {'R@P70':>7}")
            for _, r in sub.iterrows():
                print(f"{r['ablation']:10} {r['model']:8} {r['n']:6.0f} {r['base']*100:5.1f}% "
                      f"{r['ap']:7.3f} {r['p_at_r20']*100:6.1f}% {r['p_at_r10']*100:6.1f}% "
                      f"{r['rec_at_p70']*100:6.1f}%")

    out = pd.DataFrame(rows)
    out_path = DATA / "v5_runs" / "turning_predict" / "intermarket_accuracy.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
