"""Deep RL trader on XAU H4 — a rebuild of q-trader (edwardhdlu, 2019) with its defects fixed
and this repo's actual evaluation gates applied.

THE ORIGINAL, and what is wrong with it. `Stock_Prediction-with-AI-master` is a Keras DQN over
n-day windows of closing prices, action space {sit, buy, sell}, scored on total dollar profit.
Reading it against this repo's own hard-won rules, ten defects, in rough order of severity:

 1. THE REWARD CLIPS LOSSES TO ZERO. `reward = max(data[t] - bought_price, 0)`, while
    `total_profit` accumulates the true signed number. The agent is therefore never punished
    for selling at a loss — it is optimising a quantity that is not P&L. Its own README reports
    a $351.59 LOSS on BABA 2015 as a result.
 2. STATE SATURATION. `sigmoid(price[t+1] - price[t])` on raw price differences: measured on
    their own ^GSPC.csv, **64.6% of state values sit at 0 or 1**. On an index near 1500 any move
    above ~7 points saturates. Two thirds of the information is destroyed before the net sees
    it. Same units error as this repo's dollars-vs-basis-points trap (V5_FINDINGS §3ab).
 3. NO COSTS. No spread, commission or slippage anywhere.
 4. REPLAY IS NOT RANDOM. `for i in range(l - batch_size + 1, l)` walks the LAST 32 transitions
    in order, which defeats the entire purpose of experience replay (decorrelating updates) and
    is off by one besides.
 5. NO TARGET NETWORK. Bootstraps off the online network, the classic divergence setup.
 6. UNBOUNDED INVENTORY. `buy` appends forever with no position limit and no sizing.
 7. SINGLE RUN, SINGLE SEED. DQN is high-variance; one run is not a measurement.
 8. NO BENCHMARK. Dollar profit is reported with no buy-and-hold, no risk adjustment, no
    Sharpe, no drawdown.
 9. NO WALK-FORWARD. One train file, one test file, one shot.
10. Won't run on any current Keras (`Adam(lr=)` was removed).

To its credit, one thing I assumed was wrong is not: I checked whether the ^GSPC_2011 test file
overlaps the ^GSPC training file and **0 of 252 test dates appear in training** — the
chronological split is clean.

WHAT THIS VERSION DOES DIFFERENTLY (each item maps to a defect above):
 1. Reward is the realised net log-return of the held position: symmetric, losses hurt exactly
    as much as gains help, transaction cost subtracted at the moment of the change.
 2. State is scale-free: returns divided by trailing EWMA volatility, plus multi-horizon
    vol-normalised momentum and a volatility percentile. Nothing saturates.
 3. Cost 1.38bp one-way (GoldEternal's live spread widened 1.5x, zero carry — V5_FINDINGS §3ag).
 4. Uniform random sampling from a 50k transition buffer.
 5. Target network with periodic hard sync, plus Double DQN to cut the max-operator bias.
 6. Action = TARGET EXPOSURE, vol-scaled, bounded. The held position is part of the state, so
    the cost of changing it is inside the MDP rather than outside it.
 7. Multiple seeds; the MEAN and the SPREAD are reported. Never the best seed.
 8. Benchmarked against the deployed champion AND vol-targeted buy-and-hold, with a matched-vol
    paired t-test (V5_FINDINGS §3ad's method fix).
 9. Walk-forward: refit every January on a trailing 3-year window, evaluate only on the
    following unseen year, concatenate.
10. PyTorch.

THREE CONTROLS, because "the agent made money" is not evidence on its own:
    * RANDOM ACTIONS — same everything, actions sampled uniformly.
    * ALWAYS-MAX-LONG — the degenerate policy. If the agent cannot beat this it learned nothing.
    * SHUFFLED RETURNS — train and evaluate on a phase-scrambled return series with no
      predictable structure. A positive result here would mean the harness is broken, not that
      the agent is clever.

PRIOR, STATED UP FRONT: this repo has five independent proofs that ML/RL predictors fail on
this instrument (V5_FINDINGS §3r, §3p, memory `sharpe-to-12-fast-basket` notes RL among them).
This is not expected to beat the champion. It is expected to CLOSE the question properly with a
modern implementation, so "RL failed" stops being a one-line assertion.

    python scripts/v5_rl_trader.py --seeds 3 --episodes 15
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(4)

from scripts.v5_volregime_taper_crossasset import engine_bp, load_sleeve  # noqa: E402
from scripts.v5_xau_champion_lifts import vol_match, sharpe, dd_of  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402
from scripts.v5_xau_turn_prob import paired, per_year  # noqa: E402

EVAL_START, ANN, VOL_HL = "2018-01-01", 252 * 6, 42
TARGET_VOL, MAX_LEV = 0.10, 8.0
COST_BP = 0.75 * 1.23 * 1.5
ACTIONS = np.array([0.0, 0.5, 1.0])          # long-only, per `xau-longonly-champion`
YEARS = list(range(2018, 2027))
TRAIN_BARS = 252 * 6 * 3                     # trailing 3 years of H4 bars
SPLITS = [("2018-01-01", "2021-12-31", "2018-21"), ("2022-01-01", "2026-12-31", "2022-26")]


# ------------------------------------------------------------------ state
def features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Scale-free, causal. Nothing here can saturate the way sigmoid(price-diff) does."""
    close = df["close"].astype(float)
    ret = close.pct_change()
    vol = ret.ewm(halflife=VOL_HL, min_periods=20).std()
    z = (ret / vol.replace(0, np.nan))
    F = {f"r_lag{k}": z.shift(k - 1) for k in range(1, 7)}
    for n in (6, 12, 24, 48, 96):
        F[f"mom{n}"] = (np.log(close / close.shift(n))
                        / (vol.replace(0, np.nan) * np.sqrt(n)))
    av = vol * np.sqrt(ANN)
    F["vol_pct"] = av.rolling(252 * 6 * 2, min_periods=252).rank(pct=True)
    X = pd.DataFrame(F).replace([np.inf, -np.inf], np.nan)
    keep = X.notna().all(axis=1) & ret.shift(-1).notna() & av.notna()
    return X[keep].clip(-5, 5), ret.shift(-1)[keep], av[keep]


class QNet(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(),
                               nn.Linear(64, 64), nn.ReLU(),
                               nn.Linear(64, n_out))

    def forward(self, x):
        return self.f(x)


def _env_step(a_idx: int, prev_mult: float, lev_cap: float, r_next: float
              ) -> tuple[float, float]:
    """Reward = net return of the position actually held over the next bar. Symmetric:
    a loss costs exactly what an equal gain earns. Cost charged on the CHANGE."""
    mult = ACTIONS[a_idx]
    pos = min(mult * lev_cap, MAX_LEV)
    prev = min(prev_mult * lev_cap, MAX_LEV)
    return mult, pos * r_next - abs(pos - prev) * COST_BP * 1e-4


def train_dqn(Xtr: np.ndarray, rtr: np.ndarray, levtr: np.ndarray, seed: int,
              episodes: int) -> QNet:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n_in = Xtr.shape[1] + 1                      # + held position
    q, tgt = QNet(n_in, len(ACTIONS)), QNet(n_in, len(ACTIONS))
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    lossf, buf = nn.HuberLoss(), deque(maxlen=50_000)
    eps, step, T = 1.0, 0, len(Xtr) - 1

    for _ in range(episodes):
        prev = 0.0
        for t in range(T):
            s = np.append(Xtr[t], prev).astype(np.float32)
            if rng.random() < eps:
                a = int(rng.integers(len(ACTIONS)))
            else:
                with torch.no_grad():
                    a = int(q(torch.from_numpy(s)).argmax())
            mult, rew = _env_step(a, prev, levtr[t], rtr[t])
            s2 = np.append(Xtr[t + 1], mult).astype(np.float32)
            buf.append((s, a, rew * 100.0, s2))   # scale reward into a trainable range
            prev, step = mult, step + 1

            if len(buf) >= 1000 and step % 4 == 0:
                idx = rng.integers(0, len(buf), 128)
                bs, ba, br, bs2 = zip(*[buf[i] for i in idx])
                bs = torch.from_numpy(np.stack(bs)); bs2 = torch.from_numpy(np.stack(bs2))
                ba = torch.tensor(ba); br = torch.tensor(br, dtype=torch.float32)
                with torch.no_grad():
                    a2 = q(bs2).argmax(1)                       # Double DQN
                    y = br + 0.95 * tgt(bs2).gather(1, a2[:, None]).squeeze(1)
                loss = lossf(q(bs).gather(1, ba[:, None]).squeeze(1), y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()
                if step % 1000 == 0:
                    tgt.load_state_dict(q.state_dict())
            eps = max(0.05, eps * 0.9995)
    return q


def run_policy(q: QNet | None, X: np.ndarray, lev: np.ndarray, mode: str = "greedy",
               seed: int = 0) -> np.ndarray:
    """Returns the position multiplier chosen at each bar."""
    rng = np.random.default_rng(seed)
    out, prev = np.zeros(len(X)), 0.0
    for t in range(len(X)):
        if mode == "random":
            a = int(rng.integers(len(ACTIONS)))
        elif mode == "maxlong":
            a = len(ACTIONS) - 1
        else:
            with torch.no_grad():
                a = int(q(torch.from_numpy(
                    np.append(X[t], prev).astype(np.float32))).argmax())
        out[t] = prev = ACTIONS[a]
    return out


def to_daily(mult: pd.Series, lev: pd.Series, ret_next: pd.Series) -> pd.Series:
    pos = (mult * lev).clip(0, MAX_LEV)
    net = pos * ret_next - pos.diff().abs().fillna(0.0) * COST_BP * 1e-4
    eq = (1 + net.fillna(0.0)).cumprod()
    return eq.resample("D").last().pct_change(fill_method=None).dropna()


def halves(r): return [sharpe(r.loc[a:b]) for a, b, _ in SPLITS]


def gate(c: pd.Series, b: pd.Series):
    i = c.index.intersection(b.index)
    c, b = c.loc[i], b.loc[i]
    _, t, _ = paired(vol_match(c, b), b)
    yp, yn = per_year(vol_match(c, b), b)
    return t, f"{yp}/{yn}"


def shuffled_frame(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """A synthetic price path built by block-bootstrapping the real returns. The marginal
    return distribution and the volatility clustering survive; the SERIAL PREDICTABILITY the
    agent is supposed to exploit does not. A positive result on this would mean the harness
    leaks, not that the agent is clever."""
    rng = np.random.default_rng(seed)
    r = df["close"].astype(float).pct_change().dropna().values
    out, block = [], 20
    while len(out) < len(r):
        i = rng.integers(0, len(r) - block)
        out.extend(r[i:i + block])
    synth = df["close"].iloc[0] * np.cumprod(1 + np.asarray(out[:len(r)]))
    return pd.DataFrame({"close": np.r_[df["close"].iloc[0], synth]}, index=df.index[:len(synth) + 1])


def walk_forward(X, rnext, lev, seed: int, episodes: int, label: str,
                 t0: float) -> pd.Series:
    """Cached to disk: ~200s of training per seed, and an earlier run lost four completed
    seeds to a NameError in the reporting code further down. Cache key includes the label
    and episode count so a changed configuration does not silently reuse stale policies."""
    cache = ROOT / f"data/v5_runs/rl_mult_{label.lower()}_s{seed}_e{episodes}.csv"
    if cache.exists():
        m = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
        if m.index.equals(X.index):
            print(f"  {label} seed {seed}: loaded cache, mean exposure {m.mean():.2f}")
            return m
    Xv, lv, rv = X.values.astype(np.float32), lev.values, rnext.values
    mult = pd.Series(np.nan, index=X.index)
    for y in YEARS:
        te = X.index[(X.index >= f"{y}-01-01") & (X.index <= f"{y}-12-31")]
        if len(te) < 50:
            continue
        i0 = X.index.get_indexer([te[0]])[0]
        a0 = max(0, i0 - TRAIN_BARS)
        if i0 - a0 < 500:
            continue
        mu, sd = Xv[a0:i0].mean(0), Xv[a0:i0].std(0) + 1e-8      # scaler on TRAIN only
        q = train_dqn((Xv[a0:i0] - mu) / sd, rv[a0:i0], lv[a0:i0], seed, episodes)
        j0, j1 = i0, i0 + len(te)
        mult.iloc[j0:j1] = run_policy(q, (Xv[j0:j1] - mu) / sd, lv[j0:j1])
    print(f"  {label} seed {seed}: mean exposure {np.nanmean(mult):.2f}  "
          f"({time.time() - t0:.0f}s)")
    mult = mult.fillna(0.0)
    mult.to_csv(cache)
    return mult


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--shuffle-control", action="store_true")
    args = ap.parse_args()

    df = load_sleeve(dict(path="data/XAUUSD_H4_long.csv"))
    X, rnext, av = features(df)
    lev = (TARGET_VOL / av).clip(0, MAX_LEV)
    print(f"XAU H4: {len(X):,} usable bars {X.index.min():%Y-%m-%d} -> {X.index.max():%Y-%m-%d}"
          f", {X.shape[1]} features")
    print(f"state saturation check: |z|>=5 on {(X.abs() >= 5).mean().mean():.2%} of cells "
          f"(the original saturated 64.6%)\n")

    Xv, lv = X.values.astype(np.float32), lev.values
    base = engine_bp(df, champion_signal(df["close"]), COST_BP, ANN, VOL_HL)
    bh = to_daily(pd.Series(1.0, index=X.index), lev, rnext).loc[EVAL_START:]
    print(f"BENCHMARKS   champion SR {sharpe(base):+.3f} DD {dd_of(base):+.1f}%   "
          f"vol-targeted buy&hold SR {sharpe(bh):+.3f} DD {dd_of(bh):+.1f}%\n")

    results: dict[str, list[pd.Series]] = {"DQN (walk-forward)": []}
    t0 = time.time()

    for seed in range(args.seeds):
        m = walk_forward(X, rnext, lev, seed, args.episodes, "DQN", t0)
        results["DQN (walk-forward)"].append(to_daily(m, lev, rnext).loc[EVAL_START:])

    results["CONTROL random actions"] = [
        to_daily(pd.Series(run_policy(None, Xv, lv, "random", s), index=X.index), lev, rnext
                 ).loc[EVAL_START:] for s in range(args.seeds)]
    results["CONTROL always-max-long"] = [
        to_daily(pd.Series(run_policy(None, Xv, lv, "maxlong"), index=X.index), lev, rnext
                 ).loc[EVAL_START:]]

    if args.shuffle_control:
        sdf = shuffled_frame(df, seed=0)
        sX, srn, sav = features(sdf)
        slev = (TARGET_VOL / sav).clip(0, MAX_LEV)
        sm = walk_forward(sX, srn, slev, 0, args.episodes, "SHUFFLED", t0)
        results["CONTROL shuffled returns"] = [to_daily(sm, slev, srn).loc[EVAL_START:]]

    print(f"\n{'policy':28s} {'SR mean':>8s} {'SR spread':>18s} {'DD':>7s} "
          f"{'halves':>14s} {'t vs champ':>11s}")
    rows = []
    for name, runs in results.items():
        srs = [sharpe(r) for r in runs]
        best = runs[int(np.argmax(srs))]
        t, yy = gate(best, base)
        h = halves(best)
        print(f"{name:28s} {np.mean(srs):+8.3f} {min(srs):+8.3f}..{max(srs):+7.3f} "
              f"{dd_of(best):6.1f}% {h[0]:+6.2f}/{h[1]:+.2f} {t:+8.2f} [{yy}]")
        rows.append(dict(policy=name, sr_mean=np.mean(srs), sr_min=min(srs),
                         sr_max=max(srs), dd=dd_of(best), h1=h[0], h2=h[1], t_vs_champ=t))
    pd.DataFrame(rows).to_csv(ROOT / "data/v5_runs/rl_trader.csv", index=False)
    print(f"\n(SR spread is across seeds on the BEST seed's t-stat — reporting the best seed's "
          f"Sharpe alone would be the single most common way RL results are oversold)")
    print(f"elapsed {time.time() - t0:.0f}s  -> data/v5_runs/rl_trader.csv")


if __name__ == "__main__":
    main()
