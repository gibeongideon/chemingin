#!/usr/bin/env python3
"""ADVISOR -> ZIGZAG: does the advisor's reading improve the best ZigZag? (user 2026-11-12: "can
this advisor give ZigZag some information to improve it, try that again"). PRE-REGISTERED.

Both advisors are rebuilt OUT-OF-SAMPLE on the SAME gold bars the ZigZag backtest trades (no
spliced H4 file, §3ba), with the advisor pipeline (`f_base` + `walk_forward`, yearly refits):
  SHORT    H1 model, P(+1.0 ATR before -1.0 ATR within 6h)           (the new entry advisor)
  GENERAL  H4 model, P(-1.0 H4-ATR before +1.0 within 9h)            (the shipped cell 1.0/9/BASE)
           read from the H4 bar's CLOSE (stamp + 4h), never its stamp (§3az)
GATES on ZigZag variant B entries (fires at P(bottom) >= 0.60), declared before running:
  G_SHORT    skip the buy when SHORT says SELL          (p_up <= 0.45)
  G_GENERAL  skip the buy when GENERAL is DOWN          (p_down >= 0.60, the shipped DOWN edge)
  G_EITHER   skip when either does
  G_AGREE    take the buy ONLY when SHORT says BUY      (p_up >= 0.55)
Window 2018-01..2026 for both sources (the advisor walk-forward's OOS span), V0 on the same window.
Null per gate: the SAME block mask shifted to random dates inside the window (200 draws); family
max-stat over the 4 gates. Also printed: what the BLOCKED trades earned on their own.

    python scripts/v5_zigzag_advisor_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_advisor_measure import walk_forward  # noqa: E402
from scripts.v5_repr_ceiling import PURGE_EXTRA, f_base  # noqa: E402
from scripts.v5_h1_features import f_base_h1  # noqa: E402
from scripts.v5_zigzag_long_history import OUT as ZZ, SRC, THR, load, sharpe  # noqa: E402
from scripts.v5_zigzag_regime import bt  # noqa: E402
from src.v5.advisor_labels import first_touch_atr  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "zigzag_advisor_gate"
START = pd.Timestamp("2018-01-01")
N_NULL = 200
RNG = np.random.default_rng(20261113)


def atr14(d: pd.DataFrame) -> pd.Series:
    pc = d.close.shift()
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def short_p_up(g: pd.DataFrame) -> np.ndarray:
    ft = first_touch_atr(g.index, g.close, atr14(g), g.high, g.low, g.index, k_atr=1.0, horizon_bars=6)
    y = 1.0 - ft["y"].to_numpy(dtype=float)
    return walk_forward(f_base_h1(g), y, purge=6 + PURGE_EXTRA, index=g.index)["p_cal"]


def general_p_down(g: pd.DataFrame) -> np.ndarray:
    r = g.resample("4h", label="left", closed="left")
    h4 = pd.DataFrame({"open": r.open.first(), "high": r.high.max(), "low": r.low.min(),
                       "close": r.close.last()}).dropna()
    ft = first_touch_atr(h4.index, h4.close, atr14(h4), g.high, g.low, g.index, k_atr=1.0,
                         horizon_bars=9, dec_bar_minutes=240)
    y = ft["y"].to_numpy(dtype=float)                      # 1 = DOWN barrier first
    p = walk_forward(f_base(h4), y, purge=3 + PURGE_EXTRA, index=h4.index)["p_cal"]
    s = pd.Series(p, index=h4.index + pd.Timedelta(hours=4))    # usable from the bar's CLOSE
    return s.reindex(g.index, method="ffill").to_numpy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []
    for src, cfg in SRC.items():
        g = load(src, cfg["gold"])
        p = pd.read_csv(ZZ / src / "probs.csv", index_col=0, parse_dates=True).iloc[:, 0].reindex(g.index).to_numpy()
        fires = np.isfinite(p) & (p >= THR)
        pu, pdn = short_p_up(g), general_p_down(g)
        cov = np.isfinite(pu) & np.isfinite(pdn)
        st = max(START, pd.Timestamp(cfg["start"]))
        s0 = int((g.index < st).sum())
        win = (g.index >= st)
        fires = fires & cov                                   # same event set for every arm
        z = np.zeros(len(g), bool)
        tr0, r0 = bt(g, fires, z, st, cfg["swap"], 0)
        base = sharpe(r0)
        with np.errstate(invalid="ignore"):
            blocks = {"G_SHORT": pu <= 0.45, "G_GENERAL": pdn >= 0.60,
                      "G_EITHER": (pu <= 0.45) | (pdn >= 0.60), "G_AGREE": ~(pu >= 0.55)}
        shifts = RNG.integers(24 * 30, int(win.sum()) - 24 * 30, N_NULL)
        rows, nulls = [], []
        for k, blk in blocks.items():
            blk = blk & cov
            tr, r = bt(g, fires & ~blk, z, st, cfg["swap"], 0)
            s = sharpe(r)
            trb, rb = bt(g, fires & blk, z, st, cfg["swap"], 0)
            nl = []
            for sh in shifts:
                b2 = blk.copy()
                b2[s0:] = np.roll(blk[s0:], int(sh))
                nl.append(sharpe(bt(g, fires & ~b2, z, st, cfg["swap"], 0)[1]))
            nulls.append(nl)
            rows.append(dict(gate=k, fires_blocked_pct=round(100 * (fires & blk)[win].sum() / max(fires[win].sum(), 1), 0),
                             trades=len(tr), SR=round(s, 2), dSR=round(s - base, 2),
                             blocked_trades=len(trb), SR_of_blocked=round(sharpe(rb), 2),
                             null_med=round(float(np.median(nl)), 2), p=round(float((np.array(nl) >= s).mean()), 3)))
        A = pd.DataFrame(rows)
        mx = np.array(nulls).max(axis=0)
        # does the advisor rank ZigZag's OWN trades? mean advisor reading on winners vs losers
        lines += [f"\n{src.upper()} {st:%Y-%m}..: ZigZag V0 Sharpe {base:+.2f}, {len(tr0)} trades "
                  f"(fires with advisor coverage {int(fires[win].sum())})",
                  A.to_string(index=False),
                  f"family: best {A.gate[A.SR.idxmax()]} {A.SR.max():+.2f} vs shifted-max median "
                  f"{np.median(mx):+.2f}, p {(mx >= A.SR.max()).mean():.3f}"]
        print("done", src, flush=True)
    txt = "ADVISOR -> ZIGZAG gates (see docstring)\n" + "\n".join(lines) + "\n"
    print(txt)
    (OUT / "report.txt").write_text(txt)


if __name__ == "__main__":
    sys.exit(main())
