#!/usr/bin/env python3
"""ZigZag + FOOTPRINT (§3do part B): does aggressor flow improve the live ZigZag (360601)?

Same harness as `v5_zigzag_live_wf.py` / §3dk: pooled HistGB (gold + EUR/GBP/JPY H1, Switch),
monthly walk-forward 2024-01.., fire P >= 0.60, next open + $0.25, 3 ATR trail, max hold 120,
+$100 -> +$50 lock, swap -6%/yr. Footprint from PAXGUSDT minutes (§3do part A; clock checked).
VARIANTS (pre-registered):
  BASE      the live model                                        (must reproduce §3dg)
  FP_FEAT   live features + the 8 footprint features (gold rows only; FX rows NaN — HistGB
            routes missing values natively, so the pooled FX data still trains the price part)
  FP_GATE   BASE fires, kept only when the textbook footprint bottom holds on the signal bar:
            buy_lo > 0.5 (buyers dominated the bar's bottom 25% = absorption at the low).
            The SKIPPED fires are traded too and reported (control: measure the skipped sample).
CONTROLS: 200 circular-shift timing replays per variant; hold gold same days.

    python scripts/v5_footprint_zigzag.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.v5_zigzag_live_wf as lw  # noqa: E402
from scripts.v5_footprint_screen import FEATS, features, load_fp, utc_to_server  # noqa: E402
from scripts.v5_zigzag_live_wf import START, SYMS, load, sharpe, simulate  # noqa: E402
from scripts.v5_zigzag_madl import shifted  # noqa: E402
from scripts.v5_zigzag_signal import LEAK_MAX, MIN_TRAIN, TOL, TRAIN_BARS, _frame, intrabar_leak  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "footprint"
N_NULL, SEED = 200, 7


def fp_frame(gold: pd.DataFrame) -> pd.DataFrame:
    p = load_fp("PAXGUSDT")
    p.index = utc_to_server(p.index)
    p = p[~p.index.duplicated()].reindex(gold.index)
    return features(p, gold.close).replace([np.inf, -np.inf], np.nan)[FEATS].add_prefix("fp_")


def probs(gold, aux, FP):
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415
    Fg, _, okg = _frame(gold)
    out = {k: pd.Series(np.nan, index=gold.index) for k in ("BASE", "FP_FEAT")}
    months = pd.date_range(START, gold.index[-1] + pd.offsets.MonthBegin(1), freq="MS")
    kw = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0)
    for m0, m1 in zip(months[:-1], months[1:]):
        test = (gold.index >= m0) & (gold.index < m1)
        if not test.any():
            continue
        X, XF, Y = [], [], []
        for s, d in [("XAUUSDs", gold)] + sorted(aux.items()):
            d = d[d.index < m0]
            if len(d) < MIN_TRAIN:
                continue
            F, y, ok = _frame(d)
            hi = len(d) - (TOL + 1)
            lo = max(0, hi - TRAIN_BARS)
            msk = np.zeros(len(d), bool)
            msk[lo:hi] = True
            msk &= ok
            f = FP.reindex(d.index).values if s == "XAUUSDs" else np.full((len(d), FP.shape[1]), np.nan)
            X.append(F.values[msk])
            XF.append(np.hstack([F.values[msk], f[msk]]))
            Y.append(y[msk])
        X, XF, Y = np.vstack(X), np.vstack(XF), np.concatenate(Y)
        rows = test & okg
        out["BASE"][rows] = HistGradientBoostingClassifier(**kw).fit(X, Y).predict_proba(Fg.values[rows])[:, 1]
        xt = np.hstack([Fg.values[rows], FP.values[rows]])
        out["FP_FEAT"][rows] = HistGradientBoostingClassifier(**kw).fit(XF, Y).predict_proba(xt)[:, 1]
        print(f"{m0:%Y-%m} fires BASE {int((out['BASE'][test] >= .6).sum())} "
              f"FP_FEAT {int((out['FP_FEAT'][test] >= .6).sum())}", flush=True)
    return out


def row(gold, name, fire, rng_shifts, lock=True):
    tr, ret = simulate(gold, fire, lock)
    null = np.array([sharpe(simulate(gold, shifted(fire, int(k)), lock)[1]) for k in rng_shifts])
    sh = sharpe(ret)
    eq = (1 + ret).cumprod()
    return dict(variant=name, fires=int((fire > 0).sum()), trades=len(tr),
                win=round(100 * (tr.R > 0).mean(), 1) if len(tr) else np.nan,
                avgR=round(tr.R.mean(), 3) if len(tr) else np.nan, totR=round(tr.R.sum(), 1),
                ann=round(100 * (eq.iloc[-1] ** (252 / len(ret)) - 1), 2),
                mdd=round(100 * (1 - eq / eq.cummax()).max(), 2), sharpe=round(sh, 2),
                null_p50=round(float(np.median(null)), 2), null_p95=round(float(np.percentile(null, 95)), 2),
                p=round(float((null >= sh).mean()), 3))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gold = load("XAUUSDs")
    aux = {s: d for s in SYMS[1:] if intrabar_leak(d := load(s)) <= LEAK_MAX}
    FP = fp_frame(gold)
    print(f"footprint coverage on gold OOS rows: {100 * FP.loc[START:].notna().all(axis=1).mean():.1f}%")
    P = probs(gold, aux, FP)
    pd.DataFrame(P).to_csv(OUT / "zigzag_probs.csv")
    live = ROOT / "data/v5_runs/zigzag_live_wf/probs.csv"
    lines = [f"Switch XAUUSDs H1 walk-forward {START:%Y-%m-%d}..{gold.index[-1]}, pooled {sorted(aux)}"]
    if live.exists():
        lp = pd.read_csv(live, index_col=0, parse_dates=True).iloc[:, 0]
        j = lp.dropna().index.intersection(P["BASE"].dropna().index)
        lines.append(f"BASE vs §3dg live probs: corr {np.corrcoef(lp[j], P['BASE'][j])[0, 1]:.4f}")
    lw.THR = 0.5                                          # simulate() fires on mask > 0.5
    base = (P["BASE"] >= .6).astype(float)
    gate = (FP["fp_buy_lo"].reindex(gold.index) > 0.5).astype(float)
    shifts = np.random.default_rng(SEED).integers(24 * 20, int((gold.index >= START).sum()) - 24 * 20, N_NULL)
    rows = [row(gold, "BASE", base, shifts),
            row(gold, "FP_FEAT", (P["FP_FEAT"] >= .6).astype(float), shifts),
            row(gold, "FP_GATE kept", base * gate, shifts),
            row(gold, "FP_GATE skipped", base * (1 - gate), shifts)]
    gd = gold[gold.index >= START].close.resample("D").last().dropna()
    lines.append(f"hold gold same days (with swap): SHARPE {sharpe(gd.pct_change().dropna() - 0.06 / 252):+.2f}")
    txt = "\n".join(lines) + "\n" + pd.DataFrame(rows).to_string(index=False) + (
        f"\n\np = share of {N_NULL} circular-shift timing replays (same fire count, same trade rules) with\n"
        "Sharpe >= the variant's. 'skipped' = the BASE fires the footprint gate would have refused, traded.\n")
    print(txt)
    (OUT / "zigzag.txt").write_text(txt)


if __name__ == "__main__":
    sys.exit(main())
