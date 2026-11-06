#!/usr/bin/env python3
"""MADL loss (Michańków, Sakowski & Ślepaczuk 2024, J. Comp. Sci. 81:102375) — replication + controls.

THE PAPER: LSTM on daily simple returns (single feature, sequence 20 BTC / 10 oil), walk-forward
4y train (33% validation) -> 1y test, retrain yearly; long/short on the forecast's sign. Loss
    MADL = mean( -sign(R * R_hat) * |R| )
i.e. minus the return of trading the forecast's sign. Claimed: MADL beats MAE on BTC and USO.

WHAT WE ADD
  * MADL_EXACT has zero gradient almost everywhere (d sign/dx = 0), so backprop cannot train it:
    the network stays at its random init (weight decay only shrinks it). The UNTRAINED control
    (same seeds, zero epochs) tests whether MADL_EXACT is anything more than that.
  * MADL_SOFT: -mean(tanh(k * R_hat / s) * R), a differentiable version (the paper's own
    "future research" suggestion) — directly the P&L of a tanh-sized position.
  * MAE and MSE as the classical losses. 5 seeds per (asset, loss, year); report the per-seed
    Sharpe spread and the seed-ensemble (mean forecast).
  * Benchmarks: buy-and-hold, and the drift identity: on a drift asset any long/short rule must
    beat "always long" to show skill. LONG-ONLY variant (flat when the forecast is < 0).
  * Cost: per unit of position change, BTC 0.05%, WTI 0.03%, gold 0.01% (Switch-class quotes).
INTEGRATION: written to data/v5_runs/madl/forecasts.csv for `--overlay`, which gates the
  champion sleeves (XAU, BTC) by the sign forecast and judges at MATCHED VOL.

    python scripts/v5_madl_lstm.py            # ~minutes on CPU
    python scripts/v5_madl_lstm.py --overlay
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "v5_runs" / "madl"
ASSETS = {"BTC": ("BTC_D1_long.csv", 20, 0.0005),
          "WTI": ("WTI_D1_long.csv", 10, 0.0003),
          "GOLD": ("GOLD_D1_long.csv", 10, 0.0001)}
LOSSES = ("MADL_EXACT", "MADL_SOFT", "MAE", "MSE", "UNTRAINED")
SEEDS = (0, 1, 2, 3, 4)
TRAIN_Y, MIN_TRAIN_Y, VAL_FRAC = 4, 1, 0.33
EPOCHS, PATIENCE, LR, WD = 200, 30, 1e-3, 1e-4
torch.set_num_threads(8)


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, 64, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        return self.head(self.lstm(x)[0][:, -1]).squeeze(-1)


def loss_fn(name: str, pred, y):
    if name == "MADL_EXACT":
        return (-torch.sign(y * pred) * y.abs()).mean()
    if name == "MADL_SOFT":
        return -(torch.tanh(4.0 * pred) * y).mean()
    if name == "MAE":
        return (pred - y).abs().mean()
    return ((pred - y) ** 2).mean()


def windows(r: np.ndarray, seq: int):
    X = np.lib.stride_tricks.sliding_window_view(r, seq)[:-1]        # X[i] = r[i : i+seq]
    y = r[seq:]                                                       # next return
    return X, y


def fit_predict(name, seed, Xtr, ytr, Xte):
    torch.manual_seed(seed)
    net = Net()
    if name == "UNTRAINED":
        with torch.no_grad():
            return net(torch.tensor(Xte[..., None], dtype=torch.float32)).numpy()
    nv = int(len(Xtr) * VAL_FRAC)
    xt = torch.tensor(Xtr[:-nv, :, None], dtype=torch.float32)
    yt = torch.tensor(ytr[:-nv], dtype=torch.float32)
    xv = torch.tensor(Xtr[-nv:, :, None], dtype=torch.float32)
    yv = torch.tensor(ytr[-nv:], dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    best, best_state, bad = np.inf, None, 0
    for _ in range(EPOCHS):
        net.train()
        opt.zero_grad()
        loss_fn(name, net(xt), yt).backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_fn(name, net(xv), yv).item()
        if v < best - 1e-9:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(Xte[..., None], dtype=torch.float32)).numpy()


def load(fname: str) -> pd.Series:
    d = pd.read_csv(ROOT / "data" / fname, parse_dates=["time"], index_col="time")
    c = d.close[~d.index.duplicated()].sort_index()
    c = c[c.diff() != 0] if fname.startswith("GOLD") else c          # drop stale flat prints
    return c.pct_change().dropna()


def strat(pos: np.ndarray, r: np.ndarray, cost: float) -> np.ndarray:
    return pos * r - np.abs(np.diff(np.r_[0.0, pos])) * cost


def sharpe(x) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0.0


def main_study() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines, fc_all = [], {}
    for asset, (fname, seq, cost) in ASSETS.items():
        r = load(fname)
        X, y = windows(r.to_numpy(), seq)
        t = r.index[seq:]                                              # timestamp of y
        years = sorted(set(t.year))
        first = years[0] + MIN_TRAIN_Y + 1
        preds = {(L, s): pd.Series(np.nan, index=t) for L in LOSSES for s in SEEDS}
        for yr in range(first, years[-1] + 1):
            te = t.year == yr
            tr = (t.year < yr) & (t.year >= yr - TRAIN_Y)
            if te.sum() < 50 or tr.sum() < 200:
                continue
            sc = y[tr].std()
            Xtr, ytr, Xte = X[tr] / sc, y[tr] / sc, X[te] / sc
            for L in LOSSES:
                for s in SEEDS:
                    preds[(L, s)][te] = fit_predict(L, s, Xtr, ytr, Xte)
            print(f"{asset} {yr} done (train {tr.sum()}, test {te.sum()})", flush=True)
        oos = preds[(LOSSES[0], 0)].notna().to_numpy()
        ro = y[oos]
        lines.append(f"\n{asset}: OOS {t[oos][0]:%Y-%m-%d}..{t[oos][-1]:%Y-%m-%d} ({oos.sum()} days), "
                     f"buy&hold SR {sharpe(ro):+.2f}, cost {cost*100:.2f}%/change")
        for L in LOSSES:
            ls = [sharpe(strat(np.sign(preds[(L, s)].to_numpy()[oos]), ro, cost)) for s in SEEDS]
            lo = [sharpe(strat((preds[(L, s)].to_numpy()[oos] > 0).astype(float), ro, cost)) for s in SEEDS]
            ens = np.mean([preds[(L, s)].to_numpy()[oos] for s in SEEDS], axis=0)
            longfrac = np.mean([(preds[(L, s)].to_numpy()[oos] > 0).mean() for s in SEEDS])
            hit = np.mean([(np.sign(preds[(L, s)].to_numpy()[oos]) == np.sign(ro)).mean() for s in SEEDS])
            lines.append(f"  {L:10} L/S SR seeds [{', '.join(f'{v:+.2f}' for v in ls)}] median "
                         f"{np.median(ls):+.2f} ens {sharpe(strat(np.sign(ens), ro, cost)):+.2f} | "
                         f"LONG-ONLY median {np.median(lo):+.2f} | long {100*longfrac:.0f}% hit {100*hit:.1f}%")
            fc_all[f"{asset}_{L}"] = pd.Series(ens, index=t[oos])
        print("\n".join(lines[-len(LOSSES) - 1:]), flush=True)
    pd.DataFrame(fc_all).to_csv(OUT / "forecasts.csv")
    txt = "\n".join(lines) + ("\n\nSR = net daily Sharpe; seeds = 5 independent inits; ens = sign of the "
                              "5-seed mean forecast; 'long' = share of days the forecast is > 0.\n")
    (OUT / "report.txt").write_text(txt)
    print(txt)


def overlay() -> None:
    from scripts.v5_book_overlays import sleeve_returns  # noqa: PLC0415
    from scripts.v5_xau_champion_lifts import dd_of, vol_match  # noqa: PLC0415
    from scripts.v5_xau_champion_lifts import sharpe as shp  # noqa: PLC0415
    from scripts.v5_xau_turn_prob import paired, per_year  # noqa: PLC0415
    fc = pd.read_csv(OUT / "forecasts.csv", index_col=0, parse_dates=True)
    R = sleeve_returns()[["XAU", "BTC"]]
    book = R.mean(axis=1)
    lines = [f"BOOK SR {shp(book):+.3f} DD {dd_of(book):+.1f}%"]
    for L in LOSSES:
        for floor in (0.0, 0.5):
            Rg = R.copy()
            for sleeve, asset in (("XAU", "GOLD"), ("BTC", "BTC")):
                f = fc[f"{asset}_{L}"].dropna()
                f.index = f.index.normalize()
                g = (f > 0).astype(float).replace(0.0, floor)              # forecast FOR day t
                Rg[sleeve] = R[sleeve] * g.reindex(R.index).fillna(1.0)
            b2 = Rg.mean(axis=1)
            mm = vol_match(b2, book)
            _, tm, _ = paired(mm, book)
            yp, yn = per_year(mm, book)
            lines.append(f"  gate {L:10} floor {floor}: SR {shp(b2):+.3f} DD {dd_of(b2):+.1f}% "
                         f"t@matched {tm:+.2f} years {yp}/{yn}")
    txt = "\n".join(lines) + "\n"
    print(txt)
    (OUT / "overlay.txt").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", action="store_true")
    a = ap.parse_args()
    sys.exit(overlay() if a.overlay else main_study())
