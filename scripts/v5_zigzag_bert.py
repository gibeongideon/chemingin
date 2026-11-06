#!/usr/bin/env python3
"""ZigZag-BERT: a masked-candle-model (BERT-style) encoder pretrained on candle TOKENS, fed into the
live ZigZag harness — and combined with the §3dk payoff label (the user's "combinations").

WHY THIS IS NOT §3cc AGAIN: §3cc measured n-gram context on the 45-token candle alphabet and found
+0.0095 nats/token at order 1, with orders 2-3 collapsing on SPARSITY (47^3 cells). It deliberately
did not run the Transformer. A transformer shares statistics across contexts through embeddings, so
sparsity does not bind it the same way — that is the open question here.

STAGE 1 — PRETRAIN (frozen, leak-free): `src/features/bar_tokenizer.BarTokenizer` (45 bar tokens,
  size bins fitted on gold BEFORE 2024-01) over Switch H1 gold + EURUSD/GBPUSD/USDJPY. 4-layer
  encoder, d=64, context 64 bars, 15% masking (80/10/10). Trained ONLY on windows ending before
  2024-01-01, then frozen. Diagnostic on 2024+ gold: masked-token NLL vs unigram and vs the
  order-1 (previous-token) model. If BERT does not beat order-1, it learned nothing new.
STAGE 2 — FEED ZIGZAG: embedding of the last 64 closed bars (mean of final hidden states, 64 dims)
  at every bar. Monthly walk-forward 2024-01.. exactly as `v5_zigzag_madl.py`:
    BASE           live features, pivot label, P >= 0.60
    BERT           embedding only, pivot label, P >= 0.60
    ZZ+BERT        live features + embedding, pivot label, P >= 0.60
    ZZ+BERT_MADL   live features + embedding, payoff label sign(R) weighted |R|, P >= 0.50
  CONTROL: 200 circular-shift timing replays (same fires, same trade rules), lock on and off.

    python scripts/v5_zigzag_bert.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_zigzag_live_wf import SPREAD, START, SYMS, load, sharpe, simulate  # noqa: E402
import scripts.v5_zigzag_live_wf as lw  # noqa: E402
from scripts.v5_zigzag_madl import PURGE_R, series_r, shifted  # noqa: E402
from scripts.v5_zigzag_signal import LEAK_MAX, MIN_TRAIN, TOL, TRAIN_BARS, _frame, atr, intrabar_leak  # noqa: E402
from src.features.bar_tokenizer import UNK_ID, BarTokenizer  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "zigzag_bert"
CTX, D, LAYERS, HEADS = 64, 64, 4, 4
VOCAB, MASK_ID = 47, 47                     # 47 tokenizer ids + [MASK]
EPOCHS, BATCH, LR = 6, 256, 3e-4
N_NULL, SEED = 200, 7
torch.manual_seed(0)
torch.set_num_threads(8)


class Bert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tok = nn.Embedding(VOCAB + 1, D)
        self.pos = nn.Embedding(CTX, D)
        layer = nn.TransformerEncoderLayer(D, HEADS, 2 * D, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, LAYERS)
        self.head = nn.Linear(D, VOCAB)

    def hidden(self, x):
        return self.enc(self.tok(x) + self.pos(torch.arange(x.shape[1])))

    def forward(self, x):
        return self.head(self.hidden(x))


def tokens(tok: BarTokenizer, d: pd.DataFrame) -> np.ndarray:
    a = atr(d).to_numpy(float)
    o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    return np.array([tok.vocab.get(tok.encode_bar(o[i], h[i], l[i], c[i], a[i]), UNK_ID)
                     if np.isfinite(a[i]) else UNK_ID for i in range(len(d))], np.int64)


def windows(ids: np.ndarray, ends: np.ndarray) -> np.ndarray:
    return np.stack([ids[e - CTX + 1:e + 1] for e in ends])


def mask(x: torch.Tensor, g: torch.Generator):
    m = torch.rand(x.shape, generator=g) < 0.15
    y = torch.where(m, x, torch.full_like(x, -100))
    r = torch.rand(x.shape, generator=g)
    x = x.clone()
    x[m & (r < 0.8)] = MASK_ID
    rnd = m & (r >= 0.8) & (r < 0.9)
    x[rnd] = torch.randint(2, VOCAB, (int(rnd.sum()),), generator=g)
    return x, y


def pretrain(train_win: np.ndarray, lines: list) -> Bert:
    net = Bert()
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    g = torch.Generator().manual_seed(1)
    X = torch.tensor(train_win)
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(X), generator=g)
        tot = n = 0
        for b in range(0, len(X), BATCH):
            xb, yb = mask(X[perm[b:b + BATCH]], g)
            loss = ce(net(xb).reshape(-1, VOCAB), yb.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        print(f"pretrain epoch {ep + 1}: masked NLL {tot / n:.4f}", flush=True)
    lines.append(f"pretrain: {len(X):,} windows (all < {START:%Y-%m-%d}), final train masked NLL {tot / n:.4f}")
    return net.eval()


def diagnostic(net: Bert, ids: np.ndarray, ends: np.ndarray, train_ids: np.ndarray, lines: list) -> None:
    """Masked-token NLL on OOS gold vs unigram and order-1 (prev-token) baselines, same positions."""
    uni = np.bincount(train_ids, minlength=VOCAB) + 1.0
    uni /= uni.sum()
    bi = np.ones((VOCAB, VOCAB))
    np.add.at(bi, (train_ids[:-1], train_ids[1:]), 1)
    bi /= bi.sum(1, keepdims=True)
    g = torch.Generator().manual_seed(2)
    W = torch.tensor(windows(ids, ends[::4]))
    nb = nu = n1 = cnt = 0.0
    with torch.no_grad():
        for b in range(0, len(W), 1024):
            xb = W[b:b + 1024]
            m = torch.zeros(xb.shape, dtype=torch.bool)
            pos = torch.randint(1, CTX, (len(xb),), generator=g)          # one masked slot per window
            m[torch.arange(len(xb)), pos] = True
            xin = xb.clone()
            xin[m] = MASK_ID
            lp = torch.log_softmax(net(xin), -1)
            true = xb[m].numpy()
            prev = xb[torch.arange(len(xb)), pos - 1].numpy()
            nb += -lp[m].numpy()[np.arange(len(true)), true].sum()
            nu += -np.log(uni[true]).sum()
            n1 += -np.log(bi[prev, true]).sum()
            cnt += len(true)
    lines.append(f"OOS gold masked-token NLL (nats): BERT {nb / cnt:.4f} | order-1 {n1 / cnt:.4f} | "
                 f"unigram {nu / cnt:.4f}  -> BERT gain over order-1 {n1 / cnt - nb / cnt:+.4f}, "
                 f"order-1 over unigram {nu / cnt - n1 / cnt:+.4f}  ({int(cnt):,} tokens)")
    print(lines[-1], flush=True)


def embed(net: Bert, ids: np.ndarray) -> np.ndarray:
    E = np.full((len(ids), D), np.nan, np.float32)
    ends = np.arange(CTX + 20, len(ids))
    with torch.no_grad():
        for b in range(0, len(ends), 2048):
            e = ends[b:b + 2048]
            E[e] = net.hidden(torch.tensor(windows(ids, e))).mean(1).numpy()
    return E


def probs(series: dict, emb: dict) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415
    gold = series["XAUUSDs"]
    Fg, _, okg = _frame(gold)
    Eg = emb["XAUUSDs"]
    full = {"BASE": Fg.values, "BERT": Eg, "ZZ+BERT": np.hstack([Fg.values, Eg]),
            "ZZ+BERT_MADL": np.hstack([Fg.values, Eg])}
    okall = okg & np.isfinite(Eg).all(1)
    out = {k: pd.Series(np.nan, index=gold.index) for k in full}
    months = pd.date_range(START, gold.index[-1] + pd.offsets.MonthBegin(1), freq="MS")
    for m0, m1 in zip(months[:-1], months[1:]):
        test = (gold.index >= m0) & (gold.index < m1)
        if not test.any():
            continue
        Fs, Es, yp, yr = [], [], [], []
        for s, d0 in series.items():
            n = int((d0.index < m0).sum())
            d = d0.iloc[:n]
            if n < MIN_TRAIN + PURGE_R:
                continue
            F, y, ok = _frame(d)
            r = series_r(d, SPREAD if s == "XAUUSDs" else 0.0)
            E = emb[s][:n]
            hi = n - max(TOL + 1, PURGE_R)
            lo = max(0, hi - TRAIN_BARS)
            msk = np.zeros(n, bool)
            msk[lo:hi] = True
            msk &= ok & np.isfinite(r) & np.isfinite(E).all(1)
            Fs.append(F.values[msk])
            Es.append(E[msk])
            yp.append(y[msk])
            yr.append(r[msk])
        F, E, yp, yr = np.vstack(Fs), np.vstack(Es), np.concatenate(yp), np.concatenate(yr)
        w = np.abs(yr)
        rows = test & okall
        kw = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0)
        spec = {"BASE": (F, yp, None), "BERT": (E, yp, None), "ZZ+BERT": (np.hstack([F, E]), yp, None),
                "ZZ+BERT_MADL": (np.hstack([F, E]), (yr > 0).astype(int), w)}
        for name, (X, lab, wt) in spec.items():
            clf = HistGradientBoostingClassifier(**kw).fit(X, lab, sample_weight=wt)
            out[name][rows] = clf.predict_proba(full[name][rows])[:, 1]
        print(f"{m0:%Y-%m} rows {len(yp):,} fires " + " ".join(
            f"{k} {int((out[k][test] >= (0.5 if 'MADL' in k else 0.6)).sum())}" for k in out), flush=True)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []
    series = {"XAUUSDs": load("XAUUSDs")}
    for s in SYMS[1:]:
        d = load(s)
        if intrabar_leak(d) <= LEAK_MAX:
            series[s] = d
    gold = series["XAUUSDs"]
    tok = BarTokenizer()
    g0 = gold[gold.index < START].copy()
    g0["atr_14"] = atr(g0)
    tok.fit(g0)
    ids = {s: tokens(tok, d) for s, d in series.items()}
    tw = [windows(ids[s], np.arange(CTX + 20, int((d.index < START).sum())))
          for s, d in series.items()]
    train_win = np.vstack(tw)
    net = pretrain(train_win, lines)
    torch.save(net.state_dict(), OUT / "bert.pt")
    pre_ids = np.concatenate([ids[s][:int((d.index < START).sum())] for s, d in series.items()])
    oos_ends = np.flatnonzero(gold.index >= START)
    diagnostic(net, ids["XAUUSDs"], oos_ends[oos_ends >= CTX], pre_ids, lines)
    emb = {s: embed(net, ids[s]) for s in series}
    P = probs(series, emb)
    pd.DataFrame(P).to_csv(OUT / "probs.csv")
    rng = np.random.default_rng(SEED)
    n_oos = int((gold.index >= START).sum())
    shifts = rng.integers(24 * 20, n_oos - 24 * 20, N_NULL)
    lw.THR = 0.5
    rows = []
    for name, thr in (("BASE", .6), ("BERT", .6), ("ZZ+BERT", .6), ("ZZ+BERT_MADL", .5)):
        p = (P[name] >= thr).astype(float)
        for lock in (True, False):
            tr, ret = simulate(gold, p, lock)
            null = np.array([sharpe(simulate(gold, shifted(p, int(k)), lock)[1]) for k in shifts])
            sh = sharpe(ret)
            rows.append(dict(variant=name, lock=lock, trades=len(tr), avgR=round(tr.R.mean(), 3),
                             totR=round(tr.R.sum(), 1), sharpe=round(sh, 2),
                             null_p50=round(float(np.median(null)), 2),
                             null_p95=round(float(np.percentile(null, 95)), 2),
                             p=round(float((null >= sh).mean()), 3),
                             by_year={y: round(v, 1) for y, v in tr.groupby(tr.t_out.dt.year).R.sum().items()}))
            print(rows[-1], flush=True)
    gd = gold[gold.index >= START].close.resample("D").last().dropna()
    lines.append(f"hold gold same days (with swap): SHARPE {sharpe(gd.pct_change().dropna() - 0.06 / 252):+.2f}")
    txt = "\n".join(lines) + "\n" + pd.DataFrame(rows).to_string(index=False) + (
        f"\n\np = share of {N_NULL} circular-shift timing replays with Sharpe >= the variant's.\n")
    print(txt)
    (OUT / "report.txt").write_text(txt)


if __name__ == "__main__":
    sys.exit(main())
