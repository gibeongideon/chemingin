"""PHASE 2 — can a SIGNAL-BASED take-profit rescue the 4-15h long/short book?

Phase 1 (`v5_xau_fast_ls_screen.py`) established two things:
  1. COST WALL: gross Sharpe is only +0.6..+0.8, and at Maven's measured $0.44 spread
     net Sharpe collapses. Turnover is the discriminator — 297 turns/yr survived
     (+0.34), 1389 turns/yr did not (-1.22).
  2. The SHORT side lost on 10 of 10 signals; long-only roughly doubled every
     long/short Sharpe.

Turnover being the binding constraint is exactly why the user's "take profit using
discovered signal" idea deserves a real test rather than a shrug: an exit rule that
holds a position through noise instead of flip-flopping on every forecast wiggle
LOWERS turnover, and lower turnover is the only lever that beats a fixed spread.

So this stages a trade-state machine over the forecast (enter on conviction, hold at
fixed size, exit on a rule) and measures turnover AND net Sharpe for each exit rule:

  fc_follow   continuous forecast, no discrete exit (the phase-1 baseline)
  sig_decay   THE "discovered signal" TP — exit once |forecast| falls to `decay` x
              its value at entry, i.e. the signal itself says the trend is spent
  sig_flip    exit when the forecast crosses zero (trend regime change)
  time_cap    exit after N hours — the literal reading of "follow 4 to 15 hour trends"
  r_target    exit at a fixed R multiple (ATR-denominated), the classic fixed TP
  decay_or_cap  sig_decay with a hard time cap as a backstop

Each is run LONG/SHORT (as specified) and LONG-ONLY (the control that phase 1 says
should win), at all three cost levels, with walk-forward selection over the whole grid.

    python scripts/v5_xau_fast_ls_exits.py
"""
from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.dsr_pbo import deflated_sharpe_ratio, pbo_cscv  # noqa: E402
from src.evaluation.walk_forward_grid import (  # noqa: E402
    walk_forward_select, regime_split_sharpe, print_selections,
)
from scripts.v5_xau_fast_ls_screen import (  # noqa: E402  (reuse verbatim, no duplication)
    load_h1, ewmac_fast, breakout_fast, zigzag_structure, engine_h1, sharpe,
    ANN_H1, VOL_HL_H1, TARGET_VOL, MAX_LEV, SLIP_USD, EVAL_START, YEARS,
    SELECT_WINDOW_BARS, COST_LEVELS, XAU_REGIMES,
)

# phase-1 survivors + the zigzag the user explicitly asked about
BASE_SIGNALS = {
    "ewmac(12-60h)": lambda d: ewmac_fast(d, ((12, 48), (15, 60))),
    "ewmac(6-48h)":  lambda d: ewmac_fast(d, ((6, 24), (9, 36), (12, 48))),
    "breakout(12-48h)": lambda d: breakout_fast(d, (12, 24, 48)),
    "zigzag(order5)": lambda d: zigzag_structure(d, 5),
}

ENTER_THRESH = 0.5      # same conviction gate the champion's discrete engine uses
EXIT_RULES = ("fc_follow", "sig_decay", "sig_flip", "time_cap", "r_target", "decay_or_cap")
DECAYS = (0.3, 0.5, 0.7)
CAPS_H = (6, 12, 15, 24)      # hours; H1 bars
R_TARGETS = (1.0, 2.0, 3.0)


def staged_position(df: pd.DataFrame, fc: pd.Series, rule: str, *, side: str = "both",
                    decay: float = 0.5, cap_h: int = 12, r_target: float = 2.0,
                    sl_atr: float = 3.0) -> pd.Series:
    """Trade-state machine over the forecast, returning a per-bar target position
    in vol-target units. Enter when |fc| >= ENTER_THRESH (size = vol-target x fc,
    FIXED at entry so the position does not churn on forecast wiggles), hold until
    the exit rule fires, then flat until the next entry. Strictly causal: every
    decision at bar i uses fc/price data at or before i, and the caller shifts by
    one bar before any P&L is taken."""
    close = df["close"]
    ret = close.pct_change()
    vol = ret.ewm(halflife=VOL_HL_H1, min_periods=20).std() * np.sqrt(ANN_H1)
    scale = (TARGET_VOL / vol).clip(0, MAX_LEV)
    atr = (df["high"] - df["low"]).ewm(alpha=1 / 14, adjust=False).mean()

    f = fc.values
    sc = scale.values
    c = close.values
    a = atr.values
    n = len(f)
    out = np.zeros(n)

    in_pos = False
    pdir = 0.0
    psize = 0.0
    entry_fc = 0.0
    entry_px = 0.0
    bars_held = 0

    for i in range(n):
        if not np.isfinite(f[i]) or not np.isfinite(sc[i]):
            out[i] = psize if in_pos else 0.0
            continue

        if in_pos:
            bars_held += 1
            exit_now = False
            if rule in ("sig_decay", "decay_or_cap"):
                if abs(f[i]) <= decay * abs(entry_fc):
                    exit_now = True
            if rule in ("sig_flip",):
                if np.sign(f[i]) != pdir and abs(f[i]) >= ENTER_THRESH:
                    exit_now = True
            if rule in ("time_cap",) or (rule == "decay_or_cap" and not exit_now):
                if bars_held >= cap_h:
                    exit_now = True
            if rule in ("r_target",):
                move = (c[i] - entry_px) * pdir
                if np.isfinite(a[i]) and move >= r_target * sl_atr * a[i]:
                    exit_now = True
            # universal protective stop so no rule can hold an unbounded loser
            move = (c[i] - entry_px) * pdir
            if np.isfinite(a[i]) and move <= -sl_atr * a[i]:
                exit_now = True
            # a hard flip in the forecast always closes (regime change)
            if np.sign(f[i]) != pdir and abs(f[i]) >= ENTER_THRESH:
                exit_now = True
            if exit_now:
                in_pos, psize, pdir = False, 0.0, 0.0

        if not in_pos and abs(f[i]) >= ENTER_THRESH:
            d = 1.0 if f[i] > 0 else -1.0
            if (side == "long" and d < 0) or (side == "short" and d > 0):
                out[i] = 0.0
                continue
            in_pos, pdir = True, d
            psize = np.clip(f[i], -2.0, 2.0) * sc[i]
            entry_fc, entry_px, bars_held = f[i], c[i], 0

        out[i] = psize if in_pos else 0.0

    return pd.Series(out, index=df.index)


def pos_to_daily(df: pd.DataFrame, pos: pd.Series, spread_usd: float,
                 gross: bool = False) -> tuple[pd.Series, float]:
    """Net daily returns + turnover/yr for an explicit position path."""
    close = df["close"]
    ret = close.pct_change()
    p = pos.shift(1).fillna(0.0)
    cost_frac = 0.0 if gross else (spread_usd / 2.0 + SLIP_USD) / close
    net = (p * ret - p.diff().abs().fillna(0.0) * cost_frac).fillna(0.0)
    yrs = (df.index[-1] - df.index[0]).days / 365.25
    turn = float(p.diff().abs().sum() / yrs)
    eq = (1.0 + net).cumprod().loc[EVAL_START:]
    if eq.empty:
        return pd.Series(dtype=float), turn
    eq = eq / eq.iloc[0]
    return eq.resample("D").last().pct_change(fill_method=None).dropna(), turn


def main() -> None:
    df = load_h1()
    print(f"XAUUSD H1 {df.index[0].date()}..{df.index[-1].date()}  eval {EVAL_START}+")
    print("Maven measured spread $0.44 | phase-1: turnover is the binding constraint\n")

    fcs = {k: fn(df) for k, fn in BASE_SIGNALS.items()}

    # ---- build the exit-rule grid
    combos = []
    for sig in BASE_SIGNALS:
        for rule in EXIT_RULES:
            if rule in ("sig_decay",):
                combos += [(sig, rule, dict(decay=d)) for d in DECAYS]
            elif rule in ("time_cap",):
                combos += [(sig, rule, dict(cap_h=ch)) for ch in CAPS_H]
            elif rule == "r_target":
                combos += [(sig, rule, dict(r_target=r)) for r in R_TARGETS]
            elif rule == "decay_or_cap":
                combos += [(sig, rule, dict(decay=d, cap_h=ch))
                           for d in (0.5,) for ch in CAPS_H]
            else:
                combos += [(sig, rule, {})]

    print(f"=== STAGE 1: exit rules x signals — turnover and NET Sharpe ({len(combos)} cells) ===")
    print(f"{'signal':16} {'exit rule':13} {'params':16} {'turn/yr':>8} {'GROSS':>7} "
          f"{'$0.12':>7} {'$0.34':>7} {'$0.44':>7} {'LONG-ONLY $0.44':>16}")
    rows = []
    grid_ls: dict = {}
    for sig, rule, kw in combos:
        pos = staged_position(df, fcs[sig], rule, side="both", **kw)
        g, _ = pos_to_daily(df, pos, 0.0, gross=True)
        r12, turn = pos_to_daily(df, pos, 0.12)
        r34, _ = pos_to_daily(df, pos, 0.34)
        r44, _ = pos_to_daily(df, pos, 0.44)
        pos_l = staged_position(df, fcs[sig], rule, side="long", **kw)
        rl, _ = pos_to_daily(df, pos_l, 0.44)
        key = (sig, rule, tuple(sorted(kw.items())))
        grid_ls[key] = r44
        pstr = ",".join(f"{k}={v}" for k, v in kw.items()) or "-"
        rows.append(dict(signal=sig, rule=rule, params=pstr, turnover=turn,
                         gross=sharpe(g), net12=sharpe(r12), net34=sharpe(r34),
                         net44=sharpe(r44), long_only44=sharpe(rl)))
        print(f"{sig:16} {rule:13} {pstr:16} {turn:>8.0f} {sharpe(g):>+7.2f} "
              f"{sharpe(r12):>+7.2f} {sharpe(r34):>+7.2f} {sharpe(r44):>+7.2f} "
              f"{sharpe(rl):>+16.2f}")
    R = pd.DataFrame(rows)

    base = R[R.rule == "fc_follow"]
    print(f"\n  baseline (fc_follow, no discrete exit): median turn/yr {base.turnover.median():.0f}, "
          f"median net@$0.44 {base.net44.median():+.2f}")
    for rule in EXIT_RULES:
        sub = R[R.rule == rule]
        print(f"  {rule:13}: median turn/yr {sub.turnover.median():>6.0f}  "
              f"median net@$0.44 {sub.net44.median():>+6.2f}  "
              f"best {sub.net44.max():>+6.2f}   (long-only best {sub.long_only44.max():+.2f})")

    print(f"\n  best NET @$0.44 long/short: {R.net44.max():+.2f}  "
          f"({R.loc[R.net44.idxmax(), 'signal']} / {R.loc[R.net44.idxmax(), 'rule']} "
          f"{R.loc[R.net44.idxmax(), 'params']})")
    print(f"  best NET @$0.44 LONG-ONLY : {R.long_only44.max():+.2f}  "
          f"({R.loc[R.long_only44.idxmax(), 'signal']} / "
          f"{R.loc[R.long_only44.idxmax(), 'rule']} "
          f"{R.loc[R.long_only44.idxmax(), 'params']})")
    n_beat = int((R.net44 > 0.5).sum())
    print(f"  long/short cells with net@$0.44 > +0.50: {n_beat}/{len(R)}   "
          f"(champion, for reference, is +1.02 on H4)")

    # ---- did lower turnover actually buy net Sharpe? (the mechanism check)
    ok = R[np.isfinite(R.turnover) & np.isfinite(R.net44)]
    if len(ok) > 5:
        corr = float(np.corrcoef(ok.turnover, ok.net44)[0, 1])
        print(f"\n  corr(turnover, net Sharpe @$0.44) = {corr:+.2f}  "
              f"-> {'lower turnover DOES buy net Sharpe' if corr < -0.3 else 'turnover is not the whole story'}")

    print("\n=== STAGE 2: WALK-FORWARD over the whole exit-rule grid, $0.44, LONG/SHORT ===")
    oos, sel = walk_forward_select(grid_ls, YEARS, SELECT_WINDOW_BARS, min_train_bars=100)
    print_selections(sel)
    if not oos.empty and oos.std() > 0:
        e = (1 + oos).cumprod()
        print(f"\nWALK-FORWARD net SR={sharpe(oos):+.3f}  "
              f"DD={float((e/e.cummax()-1).min()*100):+.1f}%")
        trials = np.array([s / np.sqrt(252) for s in R.net44 if np.isfinite(s)])
        dsr = deflated_sharpe_ratio(oos.values, trials)
        common = None
        for v in grid_ls.values():
            common = v.index if common is None else common.intersection(v.index)
        pbo = pbo_cscv(np.column_stack([v.reindex(common).fillna(0.0).values
                                        for v in grid_ls.values()]), n_partitions=10)
        print(f"DSR={dsr['dsr']:.3f} (n_trials={dsr['n_trials']})   PBO={pbo.pbo:.3f}")
        print("regime split:")
        for k, v in regime_split_sharpe(oos, XAU_REGIMES).items():
            print(f"  {k:24s} {v:+.2f}")

    out = ROOT / "data" / "v5_runs" / "xau_fast_ls_exits.csv"
    R.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
