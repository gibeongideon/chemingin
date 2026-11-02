#!/usr/bin/env python3
"""Does the ZigZag's TRAINING WINDOW LENGTH matter? Walk-forward, sliding vs expanding.

THE QUESTION. `TRAIN_BARS = 20_000` (~3.2y of H1) has never been compared against anything. It
is the last untested knob on a detector whose CLASSIFICATION numbers are the best in this repo
(71% precision at 22% recall) and whose traded form is disproven (§3x walk-forward SR -0.76).

THE SCAR THIS IS DESIGNED AROUND. "Use fresher data" is the single most seductive wrong answer
here. §3bw measured recency weighting cutting corr(IS,OOS) from +0.364 to +0.140 and
MANUFACTURING a 0.974 pattern by shrinking the effective sample to 33. A short window is the
same operation by another name, so a short window winning on a point estimate is exactly what
that failure looks like — which is why both halves and an episode SE are reported, and why the
expanding window is in the list as the opposite extreme.

DESIGN.
  * Walk-forward with PERIODIC REFIT: fit at t, score the next `--block` bars, step. That is how
    the live system behaves in aggregate (it refits hourly, and a model fitted at t is nearly
    identical to one fitted at t+1), and it makes a 20,000-bar scoring run affordable.
  * PURGE of TOL+1 bars between the end of training and the first scored bar, same as the live
    path: the label at bar i depends on pivots to i+TOL.
  * GOLD-ONLY for the sweep. The window question is per-series and pooling multiplies cost 4x;
    the top two windows are then re-run POOLED to confirm the ranking survives the deployed arm.
  * CONTROL #8: AUC is reported but is not the verdict. Precision at the live 0.60 threshold is
    what the strategy actually consumes, and the forward return of fired bars against the base
    rate is what it is worth.

    python scripts/v5_zigzag_window.py --port 18815 --symbol XAUUSDs
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_zigzag_signal import TOL, _frame  # noqa: E402

OUT = ROOT / "data" / "v5_runs"
WINDOWS = [2_500, 5_000, 10_000, 20_000, 30_000, 0]      # 0 = EXPANDING
THRESHOLD = 0.60                                          # the live operating point


def fetch(m, sym: str, n: int) -> pd.DataFrame:
    r = m.copy_rates_from_pos(sym, 16385, 0, n)
    if r is None:
        return pd.DataFrame()
    d = pd.DataFrame([(x[0], x[1], x[2], x[3], x[4], x[5]) for x in r],
                     columns=["ts", "open", "high", "low", "close", "tick_volume"])
    d["time"] = pd.to_datetime(d.ts, unit="s")
    return d.set_index("time").drop(columns=["ts"]).iloc[:-1]      # drop the forming bar


def walk(F: np.ndarray, y: np.ndarray, ok: np.ndarray, start: int, block: int,
         window: int) -> np.ndarray:
    """Out-of-sample probability for every bar from `start` on. NaN where not scored."""
    out = np.full(len(y), np.nan)
    purge = TOL + 1
    for s in range(start, len(y), block):
        hi = s - purge                                   # training may not touch the purge gap
        lo = 0 if window == 0 else max(0, hi - window)   # 0 = expanding
        tr = np.zeros(len(y), bool)
        tr[lo:hi] = True
        tr &= ok
        if tr.sum() < 2_000 or len(np.unique(y[tr])) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                             max_leaf_nodes=31, random_state=0)
        clf.fit(F[tr], y[tr])
        e = min(s + block, len(y))
        sl = np.arange(s, e)
        sl = sl[ok[sl]]
        if len(sl):
            out[sl] = clf.predict_proba(F[sl])[:, 1]
    return out


def episode_se(mask: np.ndarray, vals: np.ndarray) -> tuple[float, int]:
    """SE over contiguous FIRING runs. A detector sits in a state for many bars and those bars
    are one observation, not many — bar-count SE overstates significance by the run length."""
    runs, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            v = vals[i:j]
            v = v[np.isfinite(v)]
            if len(v):
                runs.append(float(v.mean()))
            i = j
        else:
            i += 1
    if len(runs) < 3:
        return float("nan"), len(runs)
    return float(np.std(runs, ddof=1) / np.sqrt(len(runs))), len(runs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18815)
    ap.add_argument("--symbol", default="XAUUSDs")
    ap.add_argument("--bars", type=int, default=60_000)
    ap.add_argument("--block", type=int, default=250, help="bars scored per refit")
    ap.add_argument("--hold", type=int, default=48, help="H1 bars held, for the economics")
    ap.add_argument("--pooled", nargs="*", default=None,
                    help="aux symbols; omit for the gold-only sweep")
    a = ap.parse_args()

    from mt5linux import MetaTrader5
    m, last = None, None
    for i in range(3):
        if i:
            time.sleep(3 * i)
        m = MetaTrader5(host="localhost", port=a.port)
        if m.initialize():
            break
        last = m.last_error()
    else:
        print("bridge init failed:", last); return 2
    m.symbol_select(a.symbol, True)
    df = fetch(m, a.symbol, a.bars)
    m.shutdown()
    if df.empty:
        print("no bars"); return 1
    print(f"{len(df):,} closed H1 bars  {df.index[0]}  ..  {df.index[-1]}")

    F, y, ok = _frame(df)
    Fv = F.values
    c = df["close"].to_numpy(float)
    fwd = np.full(len(c), np.nan)
    fwd[:-a.hold] = (c[a.hold:] / c[:-a.hold] - 1.0) * 1e4        # bp over the hold

    # Scoring starts once the LARGEST finite window could have been filled, so every config is
    # judged on the SAME bars. Giving a short window extra early bars would be a free head start.
    start = max(w for w in WINDOWS if w) + TOL + 1
    if start >= len(y) - 1_000:
        print("not enough history for the largest window"); return 1
    scored = len(y) - start
    print(f"scoring {scored:,} bars from {df.index[start]}  (block {a.block}, "
          f"~{scored//a.block} refits per config)\n")

    base = float(np.nanmean(fwd[start:]))
    base_rate = float(np.nanmean(y[start:][ok[start:]]))
    print(f"base: P(bottom) {base_rate:.4f}   forward {a.hold}h return {base:+.2f}bp "
          f"— both are what every row must beat (CONTROL #2)\n")

    rows = []
    for w in WINDOWS:
        t0 = time.time()
        p = walk(Fv, y, ok, start, a.block, w)
        sel = np.isfinite(p) & ok & np.isfinite(fwd)
        if sel.sum() < 1_000:
            continue
        auc = roc_auc_score(y[sel], p[sel])
        fire = sel & (p >= THRESHOLD)
        n_f = int(fire.sum())
        prec = float(y[fire].mean()) if n_f else float("nan")
        ev = float(np.nanmean(fwd[fire])) if n_f else float("nan")
        se, n_ep = episode_se(fire, fwd)
        half = start + scored // 2
        a1 = roc_auc_score(y[sel & (np.arange(len(y)) < half)],
                           p[sel & (np.arange(len(y)) < half)])
        a2 = roc_auc_score(y[sel & (np.arange(len(y)) >= half)],
                           p[sel & (np.arange(len(y)) >= half)])
        rows.append(dict(window=("EXPAND" if w == 0 else f"{w:,}"), auc=auc, auc1=a1, auc2=a2,
                         fires=n_f, prec=prec, ev=ev, vs_base=ev - base, se=se, ep=n_ep,
                         secs=time.time() - t0))
        r = rows[-1]
        print(f"{r['window']:>8}  AUC {auc:.4f} ({a1:.4f}/{a2:.4f})  fires {n_f:>6}  "
              f"prec {prec:.4f}  EV {ev:+7.2f}bp  vsBase {ev-base:+7.2f}  SE {se:>6.2f}  "
              f"ep {n_ep:>4}  [{r['secs']:.0f}s]")

    if rows:
        k = len(rows)
        b = max(rows, key=lambda r: r["auc"])
        print(f"\nbest AUC: {b['window']} at {b['auc']:.4f}  "
              f"(halves {b['auc1']:.4f}/{b['auc2']:.4f})")
        bp = max((r for r in rows if r["prec"] == r["prec"]), key=lambda r: r["prec"])
        print(f"best precision at the live {THRESHOLD:.2f} threshold: {bp['window']} "
              f"at {bp['prec']:.4f} against a {base_rate:.4f} base")
        be = max((r for r in rows if r["vs_base"] == r["vs_base"]), key=lambda r: r["vs_base"])
        floor = be["se"] * np.sqrt(2 * np.log(max(k, 2))) if be["se"] == be["se"] else np.nan
        print(f"best economics: {be['window']} at {be['vs_base']:+.2f}bp vs base; "
              f"best-of-K floor (K={k}) ~{floor:+.2f}bp (CONTROL #7)")
        print("\n*** AUC IS NOT THE VERDICT (CONTROL #8). *** A 5.9-sigma AUC lift in this repo "
              "once produced\n    dSharpe -0.702. Precision at the live threshold and vs-base "
              "are what the strategy consumes.")
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "zigzag_window.json").write_text(json.dumps(
            dict(symbol=a.symbol, bars=len(df), scored=scored, block=a.block, hold=a.hold,
                 base=base, base_rate=base_rate, pooled=bool(a.pooled), rows=rows), indent=2))
        print(f"\nwrote {OUT / 'zigzag_window.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
