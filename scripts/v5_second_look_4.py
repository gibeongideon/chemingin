#!/usr/bin/env python3
"""Second look #4: six ZigZag VETOES re-scored as SIZE DIALS (pre-registered in
data/v5_runs/second_look/PREREG.md).

§3eu showed that on the one-position-at-a-time ZigZag harness a veto reshuffles the event set (skipping a
fire frees the slot for a later one) and destroys the information, while a SIZE DIAL keeps the same trades
and passes 16/16. Six filters were only ever scored as vetoes and each failed in one source:
  G_SHORT (§3em)     bad = H1 SHORT advisor P(up) <= 0.45            (v5_zigzag_advisor_gate.short_p_up)
  SMA50 (§3er)       bad = close <= H1 SMA50                         (v5_zigzag_elliott_veto)
  PRIOR_LOW (§3er)   bad = trend up AND low broke the last confirmed swing low (§3eq cell)
  LEMA (§3dy)        bad = 50-day EMA not above its value 10 days ago (v5_zigzag_regime.daily_regimes)
  CHAMP (§3dt)       bad = champion H4 forecast < 0.5                 (v5_zigzag_downtrend_guards.champion_ok)
  Y2UP (§3ea)        bad = 2-year yield above its level 63 business days ago (v5_free_data_gates)
Dial: size 0.5 on trades whose ENTRY bar is 'bad', 1.0 otherwise; the trades are identical (asserted).
Control per gate: a random dial halving the SAME NUMBER of trades, 200 draws -> percentile.
Family: max-stat over the 6 gates (z vs each gate's random-dial null), per source.
Known-answer control: §3eu downside-vol taper (CME +0.109 / Switch +0.215 at 0.5, >= 98th pctile).
Base = live variant B + circuit breaker k=1.0, CME GC 2012+ and Switch 2018+ cached walk-forward probs.

    python scripts/v5_second_look_4.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from scripts.v5_free_data_gates import day_to_bars, macro_flags  # noqa: E402
from scripts.v5_zigzag_advisor_gate import short_p_up  # noqa: E402
from scripts.v5_zigzag_downtrend_guards import champion_ok, circuit_ok, guards  # noqa: E402
from scripts.v5_zigzag_elliott_veto import prior_confirmed_low  # noqa: E402
from scripts.v5_zigzag_long_history import OUT as ZZ, SRC, THR, backtest, load, sharpe  # noqa: E402
from scripts.v5_zigzag_regime import daily_regimes  # noqa: E402
from scripts.v5_zigzag_vol_taper import daily  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "second_look" / "4"
N_RANDOM, SIZE = 200, 0.5
RNG = np.random.default_rng(20261128)


def bad_masks(g: pd.DataFrame) -> dict[str, np.ndarray]:
    close, low = g.close.to_numpy(), g.low.to_numpy()
    sma = pd.Series(close).rolling(50).mean().to_numpy()
    trend_up = close > sma
    pl = prior_confirmed_low(g)
    broke = np.isfinite(pl) & (low < pl)
    with np.errstate(invalid="ignore"):
        pu = short_p_up(g)
    macro = macro_flags()
    G = guards(g)
    return {
        "KNOWN: downside-vol taper": ~G["downside vol below median"],
        "G_SHORT": np.nan_to_num(pu, nan=1.0) <= 0.45,
        "SMA50": ~trend_up,
        "PRIOR_LOW": trend_up & broke,
        "LEMA": ~daily_regimes(g)["EMA"],
        "CHAMP": ~champion_ok(g, 0.5),
        "Y2UP": day_to_bars(macro["Y2UP"], g.index, 0),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, lines = [], []

    def say(s=""):
        print(s, flush=True); lines.append(s)

    for src, cfg in SRC.items():
        g = load(src, cfg["gold"])
        p = pd.read_csv(ZZ / src / "probs.csv", index_col=0, parse_dates=True).iloc[:, 0].reindex(g.index).to_numpy()
        start, swap = cfg["start"], cfg["swap"]
        yrs = (g.index[-1] - pd.Timestamp(start)).days / 365.25
        fire = np.isfinite(p) & (p >= THR) & circuit_ok(g, 1.0)
        tr, _ = backtest(g, fire, start, swap)
        ent = pd.Series(np.arange(len(g)), index=g.index).reindex(tr.t_in).to_numpy() - 1   # signal bar
        base = daily(tr, np.ones(len(tr)), start, g.index[-1])
        b_sr = sharpe(base)
        mid = base.index[len(base) // 2]
        b_h = (sharpe(base[:mid]), sharpe(base[mid:]))
        masks = bad_masks(g)
        say("=" * 112)
        say(f"{src.upper()} {start} -> {g.index[-1]:%Y-%m-%d} ({yrs:.1f}y): {len(tr)} trades, base SR {b_sr:+.3f} "
            f"(halves {b_h[0]:+.2f}/{b_h[1]:+.2f})   dial: size {SIZE} when the gate says bad")
        say("=" * 112)
        say(f"  {'gate':<28}{'bad%':>6}{'SR':>8}{'dSR':>8}{'pctile':>8}{'dSR h1':>8}{'dSR h2':>8}{'z':>7}")
        zs, nulls_z = {}, {}
        for name, bad in masks.items():
            risky = bad[ent]
            n = int(risky.sum())
            r = daily(tr, np.where(risky, SIZE, 1.0), start, g.index[-1])
            sr = sharpe(r)
            h1, h2 = sharpe(r[:mid]) - b_h[0], sharpe(r[mid:]) - b_h[1]
            nl = np.empty(N_RANDOM)
            for i in range(N_RANDOM):
                pick = np.zeros(len(tr), bool)
                pick[RNG.choice(len(tr), size=n, replace=False)] = True
                nl[i] = sharpe(daily(tr, np.where(pick, SIZE, 1.0), start, g.index[-1]))
            pct = float((nl < sr).mean())
            z = (sr - nl.mean()) / nl.std() if nl.std() > 0 else 0.0
            if not name.startswith("KNOWN"):
                zs[name] = z
                nulls_z[name] = (nl - nl.mean()) / nl.std()
            say(f"  {name:<28}{100 * n / len(tr):>5.0f}%{sr:>+8.3f}{sr - b_sr:>+8.3f}{100 * pct:>7.1f}%"
                f"{h1:>+8.3f}{h2:>+8.3f}{z:>+7.2f}")
            rows.append(dict(src=src, gate=name, n_bad=n, n=len(tr), sr=sr, base_sr=b_sr, dsr=sr - b_sr,
                             pctile=pct, dsr_h1=h1, dsr_h2=h2, z=z))
        mx = max(zs.values())
        null_max = np.max(np.vstack(list(nulls_z.values())), axis=0)   # draws are independent per gate:
        fam_p = float((null_max >= mx).mean())                          # max over 6 independent nulls
        say(f"  family max-stat over the 6 gates: best z {mx:+.2f} ({max(zs, key=zs.get)}), family p {fam_p:.3f}")
        for r_ in rows:
            if r_["src"] == src:
                r_["family_p"] = fam_p
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "cells.csv", index=False)
    say("\nVERDICT per gate (pass = >=95th pctile AND both halves positive in BOTH sources, family p < 0.05 in both):")
    for gname in [k for k in R.gate.unique() if not k.startswith("KNOWN")]:
        s = R[R.gate == gname]
        ok = bool(((s.pctile >= 0.95) & (s.dsr_h1 > 0) & (s.dsr_h2 > 0) & (s.family_p < 0.05)).all() and len(s) == 2)
        say(f"  {gname:<10} {'REVIVED' if ok else 'not revived'}   " + "  ".join(
            f"{r.src}: dSR {r.dsr:+.3f} pct {100 * r.pctile:.0f}% halves {r.dsr_h1:+.2f}/{r.dsr_h2:+.2f}"
            for r in s.itertuples()))
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
