"""QUEUE #14/#16 (rebuilt from a mechanism, not a hunch) — an OFFSET-AWARE bottom label.

WHY THIS EXISTS. Tonight's pooled detector produced the largest and most robust AUC lift in
this repo's history: gold OOS AUC 0.6876 -> 0.7452 (+0.0576, SE 0.0092), **15 of 15** test
years, sign-test p 0.00003, shuffled-label control clean at 0.50, and a TRANSFER arm trained
without gold at all scoring +0.0570. It then lost money in every structure tested (panel
dSharpe **-0.702**, cross-sectional gross |0.31| at 1.2 SE, gold overlay +0.066 at t 1.81).

The mechanism was measured rather than guessed. `label_near` tags TOL=3 bars either side of a
pivot, so one label covers 7 bars whose economics differ by a factor of three:

    offset      -3     -2     -1      0     +1     +2     +3   non-pivot
    5d return -0.6%  +0.4%  +1.8%  +2.6%  +1.8%  +1.3%  +0.9%     -0.3%
    fire rate  6.4%   6.1%   5.1%  10.2%  28.0%  27.5%  16.7%      3.8%

**The detector fires 28% of the time one bar AFTER the low and 5% one bar before.** Mean
true-positive offset +0.967. It is a CONFIRMATION detector: it waits until the rally is visible
in the features, because that is what makes the pattern easy. Confirmation costs 52% of the
available move (captured +1.24% of +2.60%), and 4,484 false-positive fires at -2.01% each —
34% of all fires — erase what is left. Net ~+0.12% per 5-day fire, indistinguishable from
holding.

**The label, not the features, is the defect.** It pays the same reward for anticipating a low
and for noticing it two days late. So this script varies the TARGET, which §3ao's Tier-4 listed
and no study here has ever done, and gates on the ECONOMIC axis rather than AUC — the lesson
from the Phase 0b gate that had an unresolvable band.

FOUR ARMS, one question: can a label be written that makes the model fire EARLY?

  INCUMBENT   label_near +/-3, the 7-bar tag every prior study used.
  ANTICIPATE  offsets -3..0 only. Firing after the low is a NEGATIVE, not a positive.
  PIVOT-ONLY  offset 0 only. Maximally strict, ~1/7 the positives.
  RET-WEIGHT  +/-3 tag, but each positive is sample-weighted by the 5-day return actually
              available at that offset, so offset +3 is worth a third of offset 0.

PRE-REGISTERED PRIMARY METRIC — captured return per fire, NOT AUC. Tonight proved AUC and
money are decoupled here, so AUC is reported for continuity and decides nothing. An arm wins
only if mean true-positive offset falls AND captured/available rises AND the panel economics
beat buy-and-hold on a paired t.

    python scripts/v5_anticipation_label.py
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from scripts.v5_pooled_bottom_detector import GOLD, YEARS, auc, hm_se, load_d1  # noqa: E402
from scripts.v5_xau_turning_ml import atr, features, zigzag_swings  # noqa: E402

TOL, ZZ_ORDER, ZZ_THETA = 3, 5, 1.5
MIN_BARS, H, THR = 1200, 5, 0.60
PURGE_DAYS = TOL + 1
ARMS = ("INCUMBENT", "ANTICIPATE", "PIVOT-ONLY", "RET-WEIGHT")


def build_panel() -> dict:
    """Features plus FOUR labels and the offset/return bookkeeping every arm is scored on.

    `zigzag_swings` returns `buys` as an INDEX ARRAY. Treating it as a boolean mask silently
    measures the first ~200 bars of each series instead of the pivots — a bug that produced
    two wrong diagnostics tonight before the event path exposed it. Indexing is positional
    throughout and asserted below.
    """
    syms = sorted(p.name.replace("_D1_long.csv", "")
                  for p in (ROOT / "data").glob("*_D1_long.csv"))
    X, S, T, OFF, FWD = [], [], [], [], []
    lab = {a: [] for a in ARMS}
    kept = 0
    for s in syms:
        d = load_d1(s)
        if d is None:
            continue
        try:
            F = features(d)
            if "hour" in F.columns:
                F = F.drop(columns=["hour"])
            _, buys = zigzag_swings(d, ZZ_ORDER, ZZ_THETA * atr(d))
        except Exception:
            continue
        piv = np.asarray(buys, int)
        if len(piv) < 10:
            continue
        n = len(d)
        assert piv.max() < n, "buys must be indices into the series"
        c = d["close"].values
        # signed offset to the NEAREST pivot; 99 = not within TOL of any
        off = np.full(n, 99, int)
        for i in piv:
            for k in range(-TOL, TOL + 1):
                j = i + k
                if 0 <= j < n and abs(k) < abs(off[j]):
                    off[j] = k
        fwd = np.full(n, np.nan)
        fwd[: n - H] = c[H:] / c[: n - H] - 1

        y_inc = (np.abs(off) <= TOL) & (off != 99)
        y_ant = np.isin(off, [-3, -2, -1, 0])
        y_piv = off == 0
        ok = F.notna().all(axis=1).values & ~np.isnan(fwd)
        if ok.sum() < MIN_BARS:
            continue
        kept += 1
        X.append(F[ok]); S.append(np.full(ok.sum(), s)); T.append(d.index[ok])
        OFF.append(off[ok]); FWD.append(fwd[ok])
        lab["INCUMBENT"].append(y_inc[ok].astype(int))
        lab["ANTICIPATE"].append(y_ant[ok].astype(int))
        lab["PIVOT-ONLY"].append(y_piv[ok].astype(int))
        lab["RET-WEIGHT"].append(y_inc[ok].astype(int))
    print(f"panel: {kept} instruments, {sum(len(x) for x in X):,} usable bars")
    return dict(X=pd.concat(X), S=np.concatenate(S),
                T=pd.DatetimeIndex(np.concatenate([t.values for t in T])),
                off=np.concatenate(OFF), fwd=np.concatenate(FWD),
                y={a: np.concatenate(v) for a, v in lab.items()})


def sample_weight(arm: str, off: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """RET-WEIGHT's whole idea: a positive at offset +3 must not be worth as much as one at
    the low. Weights are the AVERAGE available return by offset measured on the panel, not
    the bar's own realised return — using the realised return would leak the outcome."""
    if arm != "RET-WEIGHT":
        return None
    tbl = {-3: 0.0, -2: 0.15, -1: 0.69, 0: 1.00, 1: 0.70, 2: 0.49, 3: 0.35}
    w = np.ones(len(y))
    for k, v in tbl.items():
        w[(off == k) & (y == 1)] = max(v, 0.02)
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-bp", type=float, default=3.0)
    a = ap.parse_args()
    t0 = time.time()
    P = build_panel()
    X, S, T, off, fwd = P["X"], P["S"], P["T"], P["off"], P["fwd"]
    print(f"base rates: " + "  ".join(f"{k} {v.mean():.3f}" for k, v in P["y"].items()))

    oos = {a_: dict(p=[], idx=[]) for a_ in ARMS}
    for yr in YEARS:
        te = np.flatnonzero(T.year == yr)
        cut = pd.Timestamp(f"{yr}-01-01") - pd.Timedelta(days=PURGE_DAYS)
        tr = np.flatnonzero((T.year < yr) & (T <= cut))
        if len(tr) < 2000 or len(te) < 50:
            continue
        for arm in ARMS:
            y = P["y"][arm]
            if len(np.unique(y[tr])) < 2:
                continue
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                               max_leaf_nodes=31, min_samples_leaf=40,
                                               l2_regularization=1.0, random_state=0)
            sw = sample_weight(arm, off[tr], y[tr])
            m.fit(X.values[tr], y[tr], sample_weight=sw)
            oos[arm]["p"].append(m.predict_proba(X.values[te])[:, 1])
            oos[arm]["idx"].append(te)
        print(f"  {yr}: train {len(tr):7d}  test {len(te):6d}  ({time.time()-t0:.0f}s)")

    print("\n" + "=" * 96)
    print("PRIMARY METRIC — WHEN does each arm fire, and how much of the move does it get?")
    print("=" * 96)
    print(f"{'arm':12s} {'AUC(own)':>9s} {'AUC(inc)':>9s} {'mean_off':>9s} {'fire%':>7s} "
          f"{'capt5d':>8s} {'avail5d':>8s} {'ratio':>7s} {'FP%':>6s} {'FP_ret':>8s} "
          f"{'EV/fire':>9s}")
    summary = {}
    for arm in ARMS:
        if not oos[arm]["p"]:
            continue
        p = np.concatenate(oos[arm]["p"]); ix = np.concatenate(oos[arm]["idx"])
        o, f = off[ix], fwd[ix]
        y_own = P["y"][arm][ix]; y_inc = P["y"]["INCUMBENT"][ix]
        fire = p >= THR
        tp = fire & (o != 99)
        fp = fire & (o == 99)
        mean_off = float(np.average(o[tp])) if tp.sum() else np.nan
        capt = float(np.nanmean(f[tp])) if tp.sum() else np.nan
        avail = float(np.nanmean(f[o == 0]))
        fpret = float(np.nanmean(f[fp])) if fp.sum() else np.nan
        ev = float(np.nanmean(f[fire])) if fire.sum() else np.nan
        summary[arm] = dict(mean_off=mean_off, capt=capt, ratio=capt / avail,
                            ev=ev, fire=fire, p=p, ix=ix, ev_n=int(fire.sum()))
        print(f"{arm:12s} {auc(p, y_own):9.4f} {auc(p, y_inc):9.4f} {mean_off:+9.3f} "
              f"{fire.mean()*100:7.2f} {capt:+8.4f} {avail:+8.4f} {capt/avail:7.3f} "
              f"{fp.mean()*100:6.2f} {fpret:+8.4f} {ev:+9.4f}")
    print("\nmean_off  = average offset of a true-positive fire. NEGATIVE = anticipates the low.")
    print("ratio     = captured / available-at-the-pivot. The incumbent forfeits 52%.")
    print("EV/fire   = mean 5-day return over ALL fires, false positives included. This is the")
    print("            only number that has ever predicted P&L in this repo.")

    # ---------------- economics: is EV/fire big enough to beat holding? ----------------
    print("\n" + "=" * 96)
    print("ECONOMICS — EV per fire against the cost of the trade and the drift it displaces")
    print("=" * 96)
    hold5 = float(np.nanmean(fwd))
    rt = 2 * a.cost_bp / 1e4
    print(f"panel mean 5-day return of simply being long (the benchmark): {hold5:+.4f}")
    print(f"round-trip cost at {a.cost_bp:.0f}bp one-way:                          {rt:+.4f}")
    print(f"\n{'arm':12s} {'EV/fire':>9s} {'net of cost':>12s} {'vs holding':>11s} "
          f"{'fires':>7s} {'t':>7s}")
    for arm, v in summary.items():
        n = v["ev_n"]
        f_all = fwd[v["ix"]][v["fire"]]
        sd = np.nanstd(f_all)
        t = (np.nanmean(f_all) - rt - hold5) / (sd / np.sqrt(max(n, 1)))
        print(f"{arm:12s} {v['ev']:+9.4f} {v['ev']-rt:+12.4f} "
              f"{v['ev']-rt-hold5:+11.4f} {n:7d} {t:+7.2f}")
    print("\nDECISION: an arm is worth building only if EV/fire net of cost EXCEEDS the drift")
    print("it displaces, with t >= 2. Otherwise the detector is a worse way to be long.")

    # ---------------- gold-specific, for continuity with 3x ----------------
    print("\n--- GOLD only (continuity with 3x / the live harness) ---")
    print(f"{'arm':12s} {'AUC':>8s} {'SE':>7s} {'mean_off':>9s} {'EV/fire':>9s} {'fires':>7s}")
    for arm, v in summary.items():
        gi = S[v["ix"]] == GOLD
        if gi.sum() < 200:
            continue
        p, o, f = v["p"][gi], off[v["ix"]][gi], fwd[v["ix"]][gi]
        y = P["y"][arm][v["ix"]][gi]
        fire = p >= THR
        aa = auc(p, y)
        print(f"{arm:12s} {aa:8.4f} {hm_se(aa, y.sum(), len(y)-y.sum()):7.4f} "
              f"{np.average(o[fire & (o != 99)]) if (fire & (o != 99)).sum() else np.nan:+9.3f} "
              f"{np.nanmean(f[fire]) if fire.sum() else np.nan:+9.4f} {int(fire.sum()):7d}")
    print(f"\nelapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
