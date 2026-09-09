"""M15 intrabar-path trim at BOOK level — the one combination §3ap left untested.

§3ap disproved the M15 down-signal as a trim on the XAU SLEEVE (0/16: 12 of 16 cells cleared
t@matched >= +1.50 but every cell was negative in 2018-21 and positive in 2022-26, PBO 0.687).
§3ac's recorded lesson was "test overlays at BOOK level, diversification already does the
smoothing a per-sleeve overlay attempts", so the book-level version is a distinct test even
though §3ag went 0/11 on book-level RISK overlays. This one is DIRECTIONAL, not a risk overlay.

Two readings of "book level", both run:
  A. WHOLE-BOOK SCALAR — the XAU-derived down-signal scales total book exposure.
  B. SLEEVE-INSIDE-BOOK — trim only the XAU forecast, then rebuild the book, and measure the
     effect on the BOOK rather than on the sleeve. §3ap measured the sleeve.

PRE-REGISTERED, and the prior is bad: §3ac predicts dilution, §3ag went 0/11 on book overlays,
and §3ap's failure was in the HALF-SPLIT rather than in precision — so more or better-placed
signal is not obviously the fix. Run because it is cheap and because "argued from precedent" is
weaker than "measured".

Disclosed grid: 2 horizons x 4 thresholds x 2 strengths x 2 variants = 32 cells, all printed,
all counted in DSR/PBO. Gate: t@matched >= +1.50 AND both half-deltas positive.

    python scripts/v5_m15_trim_book.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_volregime_taper_crossasset import load_sleeve  # noqa: E402
from scripts.v5_financing_aware_book import engine_fin  # noqa: E402
from scripts.v5_xau_champion_lifts import champion_recipe, vol_match, sharpe, dd_of  # noqa: E402
from scripts.v5_xau_turn_prob import paired, per_year  # noqa: E402
from scripts.v5_repr_ceiling import first_touch, f_source  # noqa: E402
from scripts.v5_m15_trim_overlay import oos_prob_all_bars  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402
from src.evaluation.dsr_pbo import deflated_sharpe_ratio, pbo_cscv  # noqa: E402

SPLITS = [("2018-01-01", "2021-12-31", "2018-21"), ("2022-01-01", "2026-12-31", "2022-26")]
TILT = 0.50           # gold weight, per configs/v5_gold_max_sharpe.json
# name -> (path, ann, one-way bp, champion speed scale, vol halflife, LONG carry %/yr)
SL = {"XAU":   ("data/XAUUSD_H4_long.csv", 252 * 6, 0.75 * 1.23 * 1.5, 1.0, 42, 0.0),
      "BTC":   ("data/BTC_D1_long.csv", 252, 0.75 * 9.04 * 1.5, 1 / 6, 7, -0.30),
      "NDX":   ("data/NDX_D1_long.csv", 252, 0.75 * 0.31 * 1.5, 1 / 6, 7, -0.0394),
      "BRENT": ("data/BRENT_D1_long.csv", 252, 0.75 * 7.46 * 1.5, 1 / 6, 7, -0.0481)}


def bday(r: pd.Series) -> pd.Series:
    """Compound onto a common business-day index — BTC quotes weekends and mixing day counts
    inverted a book ranking once already (§3ah)."""
    e = (1 + r).cumprod()
    return e.resample("B").last().ffill().pct_change().dropna()


def sleeves(xau_fc: pd.Series | None = None) -> dict[str, pd.Series]:
    out = {}
    for n, (path, ann, bp, sc, hl, cy) in SL.items():
        d = load_sleeve(dict(path=path))
        if n == "XAU":
            fc = xau_fc if xau_fc is not None else champion_signal(d["close"])
        else:
            fc = champion_recipe(d["close"], sc, 1.5)
        out[n] = bday(engine_fin(d, fc, bp, ann, hl, cy, True))
    return out


def book_of(sl: dict[str, pd.Series]) -> pd.Series:
    R = pd.DataFrame(sl).dropna(how="all").fillna(0.0)
    return TILT * R["XAU"] + (1 - TILT) * R[["BTC", "NDX", "BRENT"]].mean(axis=1)


def halves(r): return [sharpe(r.loc[a:b]) for a, b, _ in SPLITS]


def gate(c: pd.Series, b: pd.Series):
    i = c.index.intersection(b.index)
    c, b = c.loc[i], b.loc[i]
    _, t, _ = paired(vol_match(c, b), b)
    yp, yn = per_year(vol_match(c, b), b)
    dh = [sharpe(vol_match(c.loc[a:x], b.loc[a:x])) - sharpe(b.loc[a:x]) for a, x, _ in SPLITS]
    return t, f"{yp}/{yn}", dh


def main() -> None:
    t0 = time.time()
    dx = load_sleeve(dict(path="data/XAUUSD_H4_long.csv"))
    m15 = load_sleeve(dict(path="data/XAUUSD_M15_long.csv"))
    base_book = book_of(sleeves())
    hb = halves(base_book)
    print(f"BOOK baseline (gold-tilted core-4, Maven carry, GoldEternal gold): "
          f"SR {sharpe(base_book):+.3f}  DD {dd_of(base_book):+.2f}%  "
          f"halves {hb[0]:+.2f}/{hb[1]:+.2f}")
    print(f"GATE: t@matched >= +1.50 AND both half-deltas positive. 32 cells disclosed.\n")

    Xs = f_source(dx, m15)
    rows, streams = [], {}
    for k in (12, 30):
        y = first_touch(dx, k)
        p = oos_prob_all_bars(Xs, y, k).ffill()
        # daily trim, using the LAST H4 bar of the prior day -> no lookahead
        for tag, trim_h4 in (
                ("q90", (p >= p.expanding(min_periods=250).quantile(0.90)).astype(float)),
                ("q80", (p >= p.expanding(min_periods=250).quantile(0.80)).astype(float)),
                ("q70", (p >= p.expanding(min_periods=250).quantile(0.70)).astype(float)),
                ("cont", ((p - 0.5) * 2).clip(0, 1))):
            trim_d = trim_h4.resample("B").last().shift(1)
            for b in (0.5, 1.0):
                # --- A: whole-book exposure scalar ---
                sc_a = (1.0 - b * trim_d).clip(0, 1).reindex(base_book.index).fillna(1.0)
                ra = (base_book * sc_a).dropna()
                # --- B: trim the XAU forecast, rebuild the book ---
                fcb = (champion_signal(dx["close"]) * (1.0 - b * trim_h4.fillna(0.0))).clip(0, 2)
                rb = book_of(sleeves(fcb))
                for vtag, r in (("A book-scalar", ra), ("B sleeve-in-book", rb)):
                    t, yy, dh = gate(r, base_book)
                    name = f"{vtag[0]} K{k} {tag} b={b}"
                    keep = (t >= 1.50 and min(dh) > 0)
                    rows.append(dict(cell=name, variant=vtag, sr=sharpe(r), dd=dd_of(r),
                                     t=t, yy=yy, d1=dh[0], d2=dh[1], keep=keep))
                    streams[name] = r
        print(f"  K={k} done ({time.time()-t0:.0f}s)")

    R = pd.DataFrame(rows).sort_values("t", ascending=False)
    print(f"\n{'cell':22s} {'SR':>7s} {'DD':>8s} {'t@matched':>10s} {'yrs':>6s} "
          f"{'dSR 18-21':>10s} {'dSR 22-26':>10s}")
    for _, r in R.iterrows():
        print(f"{r.cell:22s} {r.sr:+7.3f} {r.dd:7.2f}% {r.t:+10.2f} {r.yy:>6s} "
              f"{r.d1:+10.3f} {r.d2:+10.3f}" + ("   ** KEEPER **" if r.keep else ""))
    per = pd.DataFrame(streams).dropna()
    if per.shape[1] >= 2:
        trials = [sharpe(per[c]) / np.sqrt(252) for c in per.columns]
        best = per[max(per.columns, key=lambda c: sharpe(per[c]))]
        print(f"\nDSR {deflated_sharpe_ratio(best.values, trials)['dsr']:.4f}   "
              f"PBO {pbo_cscv(per.values).pbo:.4f}   over {len(trials)} disclosed cells")
    print(f"\nby variant, best t: " + "  ".join(
        f"{v}: {R[R.variant==v].t.max():+.2f}" for v in R.variant.unique()))
    print(f"cells with both halves positive: {int((R.d1>0)&(R.d2>0)).sum() if False else int(((R.d1>0)&(R.d2>0)).sum())}/{len(R)}")
    R.to_csv(ROOT / "data/v5_runs/m15_trim_book.csv", index=False)
    print(f"keepers {int(R.keep.sum())}/{len(R)}   elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
