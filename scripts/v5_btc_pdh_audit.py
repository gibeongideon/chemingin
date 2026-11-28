#!/usr/bin/env python3
"""BTC-PDH Stage 1 (pre-registered in data/v5_runs/btc_pdh/PREREG.md): replicate the frozen PDH rule on a
SECOND source (Binance spot BTCUSDT 1-minute, 2017-08..2026-10) and measure how much survives a late entry.
Rule and statistics are imported unchanged from scripts/v5_pdh_m1_stress.py / v5_intraday_m1.py.
Days are built from all 7 days (the live executor reads the broker's prior D1 bar, so Monday's level is
Sunday's high); the weekday cell simply blocks Saturday/Sunday entries.

    python scripts/v5_btc_pdh_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v5_intraday_m1 import sharpe, vol_weight  # noqa: E402
from scripts.v5_pdh_m1_stress import run, to_days  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "btc_pdh"
EXIT_AT, STOP_K = 16 * 60 + 40, 1.0


def load() -> pd.DataFrame:
    z = np.load(ROOT / "data/binance_1m/BTCUSDT.npz")
    t = pd.to_datetime(z["t"], unit="ms" if z["t"].max() > 1e12 else "s")
    d = pd.DataFrame(dict(open=z["o"], high=z["h"], low=z["l"], close=z["c"]), index=t).astype(float)
    d.index = d.index.tz_localize("UTC").tz_convert("America/New_York").tz_localize(None)
    return d[~d.index.duplicated(keep="last")].sort_index()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    A, mod, start, end, bar = to_days(d, 720)          # >= half of a 1440-minute day
    n = len(bar)
    lv = bar.high.shift(1).to_numpy()
    O = A["open"]
    gap = np.array([np.isfinite(lv[i]) and O[start[i]] >= lv[i] for i in range(n)])
    weekend = np.asarray(bar.index.dayofweek.isin([5, 6]))
    hold = (bar.close / bar.open - 1).to_numpy()
    w = vol_weight(hold)
    half = n // 2
    lines = [f"Binance BTCUSDT 1m -> {n} trading days {bar.index[0].date()} -> {bar.index[-1].date()} (17:00 NY days)"]
    rows = []
    for days_lab, block in (("weekdays (PRIMARY)", gap | weekend), ("all 7 days", gap)):
        for entry in ("stop", "mclose", "delay5"):
            r, on, frac, e, tin = run(A, mod, start, end, bar, stop_k=STOP_K, exit_at=EXIT_AT, block=block, entry=entry)
            gross = w * r
            expo = w * frac
            a = gross - expo.mean() * hold
            t_a = float(a.mean() / a.std() * np.sqrt(len(a)))
            for cbp in (4.5, 1.0):
                c1 = w * on * cbp * 1e-4
                net, net2 = gross - c1, gross - 2 * c1
                row = dict(days=days_lab, entry=entry, cost_bp=cbp, trades=int(on.sum()),
                           gross_bp=round(1e4 * r[on.astype(bool)].mean(), 2), hit=round(100 * (r[on.astype(bool)] > 0).mean(), 1),
                           sr_gross=round(sharpe(gross), 2), sr_net=round(sharpe(net), 2), sr_2x=round(sharpe(net2), 2),
                           h1=round(sharpe(net[:half]), 2), h2=round(sharpe(net[half:]), 2), t_a=round(t_a, 2))
                rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "stage1_cells.csv", index=False)
    lines.append(R.to_string(index=False))
    p = R[(R.days == "weekdays (PRIMARY)") & (R.entry == "stop") & (R.cost_bp == 4.5)].iloc[0]
    late = R[(R.days == "weekdays (PRIMARY)") & (R.entry == "mclose") & (R.cost_bp == 4.5)].iloc[0]
    ok = p.sr_net >= 0.5 and p.h1 > 0 and p.h2 > 0 and p.t_a >= 2 and late.sr_net > 0
    kill = p.sr_net <= 0 or p.h1 < 0 or p.h2 < 0
    lines.append(f"\nOANDA M5 reference (§3fa #6): net +1.46, t_a 4.72, halves 2.30/0.48")
    lines.append(f"STAGE 1: {'PASS -> Stage 2 live fill probe' if ok else ('KILL' if kill else 'FAIL (no kill) -> stop')}"
                 f"  (primary net {p.sr_net:+.2f}, halves {p.h1:+.2f}/{p.h2:+.2f}, t_a {p.t_a:+.2f}; 1-min-late net {late.sr_net:+.2f})")
    print("\n".join(lines))
    (OUT / "stage1_report.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
