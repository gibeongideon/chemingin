#!/usr/bin/env python3
"""Champion book + FOOTPRINT overlay (§3do part C), judged at MATCHED VOL.

Sleeves: XAU = champion_signal on Switch XAUUSDs H4 (H1 resampled on the server clock) with
PAXGUSDT flow summed into the same H4 bars; BTC = champion_recipe(close, 1/6, 1.5) on Binance
BTCUSDT D1 with BTC's own flow. Engine T.engine_bp (vol target, buffer), cost 1bp XAU / 5bp BTC.
OVERLAYS (pre-registered, the forecast is multiplied by m):
  TILT6     m = 1 + 0.5 tanh(z(cvd6)),  z vs trailing 120 bars    (buyers in control -> more)
  TILT30    same on cvd30
  DIVCUT    m = 0.5 when close is at its 30-bar high but cvd30 < 0 (rally without buyers), else 1
GATE per cell: matched-vol paired t >= 2 AND more years better than worse (report() from
v5_xau_champion_lifts). FAMILY NULL: 200 circular shifts of the multiplier (same values, random
timing); p = share whose MAX t over the 6 cells beats the observed max.

    python scripts/v5_footprint_book.py
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
import scripts.v5_volregime_taper_crossasset as T  # noqa: E402
from scripts.v5_footprint_screen import load_fp, utc_to_server  # noqa: E402
from scripts.v5_xau_champion_lifts import champion_recipe, report  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "footprint"
N_NULL, SEED = 200, 11


def sleeves() -> dict:
    g = pd.read_csv(ROOT / "data/v5_runs/zigzag_live_wf/XAUUSDs_H1.csv", parse_dates=["time"], index_col="time")
    g = g[~g.index.duplicated()].sort_index()
    gx = g.resample("4h").agg(dict(open="first", high="max", low="min", close="last")).dropna()
    p = load_fp("PAXGUSDT")
    p.index = utc_to_server(p.index)
    pf = p[~p.index.duplicated()][["v", "tb"]].resample("4h").sum(min_count=1).reindex(gx.index)
    b = load_fp("BTCUSDT")
    bx = b.resample("D").agg(dict(o="first", h="max", l="min", c="last", v="sum", tb="sum")).dropna()
    bpx = bx.rename(columns=dict(o="open", h="high", l="low", c="close"))
    return {"XAU": (gx, pf, champion_signal(gx.close), 1.0, 1512, 42, "2021-07-01"),
            "BTC": (bpx, bx[["v", "tb"]], champion_recipe(bpx.close, 1 / 6, 1.5), 5.0, 365, 7, "2019-01-01")}


def overlays(close: pd.Series, flow: pd.DataFrame) -> dict[str, pd.Series]:
    d = (2 * flow.tb - flow.v)
    out = {}
    for n in (6, 30):
        cvd = d.rolling(n).sum() / flow.v.rolling(n).sum()
        z = (cvd - cvd.rolling(120).mean()) / cvd.rolling(120).std()
        out[f"TILT{n}"] = (1 + 0.5 * np.tanh(z)).fillna(1.0)
    cvd30 = d.rolling(30).sum() / flow.v.rolling(30).sum()
    out["DIVCUT"] = pd.Series(np.where((close >= close.rolling(30).max()) & (cvd30 < 0), 0.5, 1.0), index=close.index)
    return out


def roll(m: pd.Series, k: int, start: str) -> pd.Series:
    q = m.copy()
    seg = q.index >= start
    q[seg] = np.roll(q[seg].to_numpy(), k)
    return q


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows, cells = [], []
    for name, (px, flow, fc, cost, ann, hl, start) in sleeves().items():
        T.EVAL_START = start
        base = T.engine_bp(px, fc, cost, ann, hl)
        print(f"{name}: base Sharpe {base.mean() / base.std() * np.sqrt(252):+.3f} ({start}..)", flush=True)
        for oname, m in overlays(px.close, flow).items():
            r = report(f"{name} {oname}", T.engine_bp(px, fc * m, cost, ann, hl), base)
            r["share_cut"] = round(float((m.loc[start:] < 1).mean()), 3)
            rows.append(r)
            cells.append((px, fc, cost, ann, hl, start, base, m))
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    obs = max(r["t_vs_base"] for r in rows)
    null = []
    for _ in range(N_NULL):
        best = -np.inf
        for px, fc, cost, ann, hl, start, base, m in cells:
            T.EVAL_START = start
            k = int(rng.integers(30, int((m.index >= start).sum()) - 30))
            best = max(best, report("n", T.engine_bp(px, fc * roll(m, k, start), cost, ann, hl), base)["t_vs_base"])
        null.append(best)
    null = np.array(null)
    R = pd.DataFrame(rows)[["name", "sharpe", "dd", "dd_matched", "t_raw", "t_vs_base", "years_better", "share_cut"]]
    R["PASS"] = (R.t_vs_base >= 2) & (pd.DataFrame(rows).yp > pd.DataFrame(rows).yn / 2)   # (better, total)
    txt = (R.round(3).to_string(index=False) + f"\n\nfamily max-stat: observed max t {obs:+.2f}; null median "
           f"{np.median(null):+.2f}, 95th {np.percentile(null, 95):+.2f}; p = {(null >= obs).mean():.3f} "
           f"({N_NULL} circular shifts of the multiplier)\n")
    print(txt)
    (OUT / "book.txt").write_text(txt)


if __name__ == "__main__":
    sys.exit(main())
