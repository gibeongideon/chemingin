#!/usr/bin/env python3
"""OCTOBER A1 — candle structure as a LANGUAGE. Markov / n-gram over the 45-token bar alphabet.

Pre-registered in data/v5_runs/october/PREREGISTRATION.md (committed before the first fit).

WHY THIS IS NOT CLOSED BY THE REPRESENTATION CEILING (3ao), in three parts, all from the repo's
own text:
  1. 3ao's window W is `[z-scored log return, normalised range] x 64` -- it contains NO open, NO
     body/range ratio and NO wick asymmetry. `src/features/bar_tokenizer.py`'s alphabet is
     DIR_SIZE_WICK, so it is NOT a measurable function of W and `I(Y;R(W)) <= I(Y;W)` does not
     bind it.
  2. That argument fixes the label Y. This predicts the NEXT SYMBOL, and CONTROL #8's own scope
     note exempts "programmes that change the TARGET".
  3. 3ao was a FEATURE test: `f_ceiling` emits 128 independent lagged columns into i.i.d.-row
     classifiers, and nothing consumes order. A corpus grep finds n-gram 0 hits, markov 0,
     next-token 0, LSTM 0, transformer 0.

WHY IT ATTACKS THE BINDING CONSTRAINT. CONTROL #7: XAU H4 yields ~247 decided events, SE(AUC)
0.0367, and a best-of-8 noise floor of +0.053 -- "any approach that only nudges AUC is
unfalsifiable here". This run has **270,458 M15 bars**, which is the largest power gain available
without new data. It is the one corner where a sequence objective is measurable.

MARKOV AND N-GRAMS FIRST, TRANSFORMER ONLY IF THESE SHOW SIGNAL. `src/models/bar_lm_model.py` is a
~200k-parameter Transformer over this exact vocabulary, built and registered and never trained. If
an order-3 n-gram cannot beat an order-1 Markov chain, a transformer over the same alphabet is
fitting noise with more parameters, and the cheap test says so for a thousandth of the compute.

THREE CONTROLS THAT DECIDE IT, pre-declared:
  * order-0 (token frequency) and order-1 (Markov) baselines -- a higher order must EARN its
    context, and on 47 symbols an order-3 table has 47^3 = 103,823 cells against 270k bars, so
    sparsity alone can beat it.
  * a SHUFFLED-TOKEN control: the identical pipeline on a permuted token stream. Sequence
    structure must beat its own absence.
  * EV per fire net of the live 0.504bp spread (CONTROL #8). A perplexity gain is an information
    measurement, not money -- 3as logged a 5.9-sigma AUC gain that produced dSharpe -0.702.

PROVENANCE. `data/XAUUSD_M15_long.csv` is the ONE file 3ba establishes was never refreshed, so it
is single-feed throughout (the splice lives in XAUUSD_H4_long.csv, which this never touches).
Measured intrabar_leak M15 = 0.0166 against a 0.15 bar.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features.bar_tokenizer import BarTokenizer, VOCAB_SIZE, UNK_TOKEN   # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "october"
SPREAD_BP = 0.504                      # live Switch XAUUSDs
RT_BP = SPREAD_BP * 1.5                # round trip, repo convention
N_FOLDS = 6
RNG = np.random.default_rng(20261029)


def wilder_atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def load_m15() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data/XAUUSD_M15_long.csv", parse_dates=["time"], index_col="time")
    d = d[~d.index.duplicated(keep="last")].sort_index()
    d["atr"] = wilder_atr(d["high"], d["low"], d["close"])
    return d.dropna(subset=["atr"])


def intrabar_leak(d: pd.DataFrame) -> float:
    """|corr(clpos[t], ret[t+1])| -- the forward-stamping harm itself. Bar 0.15."""
    c = d["close"].pct_change()
    x = (d["close"] - d["low"]) / (d["high"] - d["low"]).replace(0, np.nan)
    nx = c.shift(-1)
    ok = x.notna() & nx.notna()
    return float(abs(x[ok].corr(nx[ok])))


def tokenize(d: pd.DataFrame, tok: BarTokenizer) -> np.ndarray:
    """One token id per bar. The tokenizer's SIZE bins must already be fit on TRAIN ONLY."""
    o, h, l, c, a = (d[k].to_numpy(float) for k in ("open", "high", "low", "close", "atr"))
    ids = np.empty(len(d), dtype=np.int16)
    vocab = tok.vocab
    unk = vocab[UNK_TOKEN]
    for i in range(len(d)):
        ids[i] = vocab.get(tok.encode_bar(o[i], h[i], l[i], c[i], a[i]), unk)
    return ids


def ngram_logloss(ids: np.ndarray, tr: slice, te: slice, order: int,
                  alpha: float = 0.5) -> float:
    """Mean negative log-likelihood per held-out token, counts from TRAIN ONLY.

    Add-alpha (Lidstone) smoothing. A context unseen in training falls back to the order-0
    train distribution rather than to a uniform, which is the harder and fairer baseline.
    """
    V = VOCAB_SIZE
    uni = np.bincount(ids[tr], minlength=V).astype(float) + alpha
    uni /= uni.sum()
    if order == 0:
        return float(-np.mean(np.log(uni[ids[te]])))

    ctx = defaultdict(lambda: np.zeros(V))
    a = ids[tr]
    for i in range(order, len(a)):
        ctx[a[i - order:i].tobytes()][a[i]] += 1.0

    b = ids
    lo, hi = te.start, te.stop
    tot, n = 0.0, 0
    for i in range(max(lo, order), hi):
        row = ctx.get(b[i - order:i].tobytes())
        if row is None:
            p = uni[b[i]]
        else:
            s = row.sum()
            p = (row[b[i]] + alpha) / (s + alpha * V)
        tot -= np.log(max(p, 1e-12))
        n += 1
    return tot / max(n, 1)


def folds(n: int) -> list[tuple[slice, slice]]:
    edges = np.linspace(int(n * 0.35), n, N_FOLDS + 1).astype(int)
    return [(slice(0, edges[i]), slice(edges[i], edges[i + 1])) for i in range(N_FOLDS)]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_m15()
    leak = intrabar_leak(d)
    print("=" * 104)
    print("OCTOBER A1 — CANDLE STRUCTURE AS A LANGUAGE (Markov / n-gram, 45-token alphabet)")
    print("=" * 104)
    print(f"  XAUUSD M15  {d.index[0].date()} -> {d.index[-1].date()}   {len(d):,} bars")
    print(f"  intrabar_leak = {leak:.4f}  (bar 0.15) -> "
          f"{'CLEAN' if leak <= 0.15 else '*** LEAKING, STOP ***'}")
    if leak > 0.15:
        return 1

    # --- tokenizer fit on the FIRST fold's training span only, then frozen ------------------
    fl = folds(len(d))
    tok = BarTokenizer()
    tok.fit(d.iloc[fl[0][0]])
    ids = tokenize(d, tok)

    # --- PRE-REGISTERED SANITY: the vocabulary must actually populate ----------------------
    used, counts = np.unique(ids, return_counts=True)
    share = counts / counts.sum()
    print(f"\n  VOCAB SANITY: {len(used)} of {VOCAB_SIZE} symbols used; "
          f"top symbol holds {share.max():.1%}; top-3 hold {np.sort(share)[-3:].sum():.1%}")
    inv = {v: k for k, v in tok.vocab.items()}
    top = sorted(zip(counts, used), reverse=True)[:8]
    print("    most frequent: " + ", ".join(f"{inv[i]} {c/counts.sum():.1%}" for c, i in top))
    if len(used) < 20 or share.max() > 0.5:
        print("    *** ALPHABET COLLAPSED — a bad ATR scale would invalidate everything below ***")
        return 1

    # --- held-out perplexity by order, plus the shuffled-token control ---------------------
    print(f"\n  --- HELD-OUT NEGATIVE LOG-LIKELIHOOD per token ({N_FOLDS} expanding folds) ---")
    print(f"  {'order':>6} {'real':>10} {'shuffled':>10} {'gain vs o0':>11} "
          f"{'gain vs o1':>11} {'vs shuffled':>12}")
    print("  " + "-" * 66)
    sh = ids.copy()
    RNG.shuffle(sh)                      # destroys sequence, preserves the token distribution
    res = {}
    for order in (0, 1, 2, 3):
        r = float(np.mean([ngram_logloss(ids, tr, te, order) for tr, te in fl]))
        s = float(np.mean([ngram_logloss(sh, tr, te, order) for tr, te in fl]))
        res[order] = (r, s)
        g0 = res[0][0] - r if 0 in res else 0.0
        g1 = (res[1][0] - r) if 1 in res else float("nan")
        print(f"  {order:>6} {r:>10.5f} {s:>10.5f} {g0:>+11.5f} "
              f"{g1 if order >= 1 else float('nan'):>+11.5f} {s - r:>+12.5f}")

    print("""
  READING. 'gain vs o0' is what CONTEXT buys over knowing only the token frequencies. 'vs
  shuffled' is the same model on a permuted stream -- if a positive gain survives shuffling, it is
  the smoothing and the vocabulary, not the sequence. On 47 symbols an order-3 table has 103,823
  cells against 270k bars, so a FALLING gain from order 2 to 3 is sparsity, not absence of
  structure.""")
    pd.DataFrame([dict(order=k, nll=v[0], nll_shuffled=v[1]) for k, v in res.items()]).to_csv(
        OUT / "a1_perplexity.csv", index=False)
    economics(d, ids, tok)
    return 0




# ─────────────────────────────────────────────────────────────────────────────────────────────
# STAGE 2 — THE ECONOMIC TEST (CONTROL #8). Run only because stage 1 found real order-1 structure.
#
# Stage 1 result: order-1 NLL 3.12590 vs order-0 3.13545 = +0.00954 nats/token of genuine context,
# and orders 2-3 collapse on sparsity (47^3 = 103,823 cells against 270k bars). The order-0
# real-vs-shuffled gap of +0.00552 is the NOISE FLOOR of that comparison -- order-0 holds no
# sequence information by construction, so whatever separates it from its own shuffle is
# non-stationarity, not structure. Order-1's +0.02013 against shuffled is therefore worth about
# +0.0146 net of that floor.
#
# 0.0095 nats on a base of 3.135 is a 0.3% perplexity improvement. 3as logged a 5.9-SIGMA AUC gain
# that produced dSharpe -0.702, so the only question that matters now is EV per fire.
# ─────────────────────────────────────────────────────────────────────────────────────────────

def up_mask(tok: BarTokenizer) -> np.ndarray:
    """Boolean over token ids: is this an UP-bodied bar?"""
    m = np.zeros(VOCAB_SIZE, dtype=bool)
    for name, i in tok.vocab.items():
        m[i] = name.startswith("UP_")
    return m


def economics(d: pd.DataFrame, ids: np.ndarray, tok: BarTokenizer) -> None:
    """Walk-forward order-1 P(next bar UP) -> position -> EV per fire, net of the live spread."""
    V = VOCAB_SIZE
    up = up_mask(tok)
    ret = d["close"].pct_change().shift(-1).to_numpy()      # the bar we would hold
    fl = folds(len(d))
    p_hat = np.full(len(d), np.nan)
    for tr, te in fl:
        cnt = np.zeros((V, V))
        a = ids[tr]
        np.add.at(cnt, (a[:-1], a[1:]), 1.0)
        # *** SMOOTHING MUST BE CONSISTENT ACROSS NUMERATOR AND DENOMINATOR. ***
        # The first version added 0.5 to the numerator and 0.5*V = 23.5 to the denominator, which
        # biases every probability DOWN -- with 1000 observations and 500 UP it returns
        # 500.5/1023.5 = 0.489, so nothing ever fires above 0.50 and the whole grid came back
        # empty while the "short" arm fired on 142,814 of 175,797 bars. The prior mass on the UP
        # event is alpha x (number of UP tokens), not alpha.
        n_up = int(up.sum())                                   # 15 = 5 sizes x 3 wicks
        pu = (cnt[:, up].sum(axis=1) + 0.5 * n_up) / (cnt.sum(axis=1) + 0.5 * V)
        lo, hi = te.start, te.stop
        p_hat[lo:hi] = pu[ids[lo - 1:hi - 1]] if lo > 0 else np.nan

    ok = np.isfinite(p_hat) & np.isfinite(ret)
    # *** THRESHOLDS MUST SIT AT THE BASE RATE OF THE PREDICTED EVENT, NEVER AT 0.5. ***
    # p_hat is P(next TOKEN is UP_*), and roughly a fifth of bars are DOJI, so that probability
    # centres near 0.40 -- not 0.50. The first grid asked for p_hat > 0.50/0.52/0.55 and fired on
    # ZERO of 175,797 bars while the "short" arm fired on 142,203, which is the signature of a
    # threshold set against the wrong base. This is the repo's own drift-baseline rule
    # (xau-drift-baseline-inverts-3r) and I broke it while citing it.
    p_base = float(np.nanmean(p_hat[ok]))
    base_up = float((ret[ok] > 0).mean())
    base_ev = float(ret[ok].mean() * 1e4)
    print("\n" + "=" * 104)
    print("STAGE 2 — ECONOMICS (CONTROL #8: EV per fire, not perplexity)")
    print("=" * 104)
    print(f"  scored bars {ok.sum():,}   base P(next bar up) {base_up:.4f}   "
          f"base mean fwd {base_ev:+.3f}bp   round trip {RT_BP:.2f}bp")
    print(f"  P(next token UP) centres at {p_base:.4f}  <- the base the thresholds must straddle")
    print(f"  M15 breakeven hit rate at this cost (3bx) = 0.5343\n")
    print(f"  {'rule':<26} {'fires':>8} {'hit':>8} {'base':>8} {'EV bp':>9} "
          f"{'net EV':>9} {'vs always-long':>15}")
    print("  " + "-" * 88)
    arms = [("ALL bars (always long)", ok)]
    for off in (0.01, 0.02, 0.04):
        arms.append((f"P(up) > base+{off:.2f} ({p_base + off:.3f})", ok & (p_hat > p_base + off)))
    for off in (0.02, 0.04):
        arms.append((f"P(up) < base-{off:.2f}  [short]", ok & (p_hat < p_base - off)))
    for lbl, m in arms:
        n = int(m.sum())
        if n < 100:
            print(f"  {lbl:<26} {n:>8}  (too few fires)"); continue
        side = -1.0 if "short" in lbl else 1.0
        r = ret[m] * side
        hit = float((r > 0).mean())
        ev = float(r.mean() * 1e4)
        print(f"  {lbl:<26} {n:>8} {hit:>8.4f} {base_up:>8.4f} {ev:>+9.3f} "
              f"{ev - RT_BP:>+9.3f} {ev - base_ev * side:>+15.3f}")
    print(f"""
  THE DECIDING COMPARISON is the last column: EV against SIMPLY BEING LONG, not against zero.
  3bh's identity -- excess(LONG) + excess(SHORT) = -2*base -- means any rule that is not always-long
  starts from a deficit against gold's drift, and CONTROL #2 requires the buy-and-hold benchmark.""")


if __name__ == "__main__":
    raise SystemExit(main())
