"""Audit every price file in `data/` for the provenance defect that cost §3ao and §3az.

WHAT IT LOOKS FOR. Two files of the same instrument at different timeframes must agree at the
instants they share: the coarse bar's close equals the close of the fine bar that ends on it. If
they diverge, the files came from different brokers (or different eras of the same broker), and
any study that resolves a label on one while building features from the other is comparing arms
that had different information — a leak in the data's PROVENANCE, which no timestamp control in
this repo can see. §3ba is the worked example: agreement fell from 100% before 2023 to 0.5% in
2026 and fabricated an entire "M15 intrabar edge".

WHY PER YEAR AND NOT POOLED. The defect is introduced by a REFRESH, so it is confined to
whatever window the refresher reaches back to — 2.5 years for `--bars 5000` at H4. Pooled over
eleven years that is a minority of rows and the median difference stays at 0.0000, which is
exactly why it went unnoticed. The per-year table makes the boundary obvious.

WHY MEDIAN AND A FRACTION, NOT A MEAN. Two feeds of the same asset differ by a spread and by
timing jitter, so a handful of large differences is normal. What is not normal is the CENTRE
moving. `frac_within_tol` is the honest headline: on one series it is ~1.00, and it collapses
toward 0 the moment the files stop being the same series.

THE TOLERANCE IS RELATIVE, AND THAT IS NOT A DETAIL. The first version of this audit used a flat
$0.10, which is about right for gold at $4,378 and absurd everywhere else: on EURUSD at 1.08 it
is a thousand pips, so every FX pair "passed" without the test ever having been applied. A
relative 0.2 basis points is ~$0.09 on gold, ~0.3 pip on USDJPY and ~0.2 pip on EURUSD — the
same question asked of each instrument. Any fixed-unit threshold in a multi-asset check is a
silent pass for whichever assets happen to be cheap.

D1 NEEDS ITS SESSION BOUNDARY FOUND, NOT ASSUMED. A daily bar's close depends on the broker's
rollover hour, so the audit searches the offset that best aligns D1 with its intraday file and
reports the winner. A convention mismatch lands near 1.00 once the right offset is used; a feed
splice does not improve at any offset, which is how the two are told apart.

    python scripts/v5_feed_audit.py                 # audit everything it can pair
    python scripts/v5_feed_audit.py --symbol XAUUSD
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
REL_TOL = 2e-5                 # 0.2 bp: a spread-sized tolerance at any price level
PASS_FRAC = 0.95               # below this in any year, the pair is not one series


def read(p: Path) -> pd.DataFrame | None:
    try:
        d = pd.read_csv(p, parse_dates=["time"], index_col="time").sort_index()
    except Exception:
        return None
    d = d[~d.index.duplicated(keep="last")]
    d.columns = [c.lower() for c in d.columns]
    return d if "close" in d.columns and len(d) > 500 else None


def discover() -> dict:
    """Group price files by symbol -> {timeframe: path}, from the repo's naming convention."""
    out: dict = {}
    for p in sorted(DATA.glob("*.csv")):
        m = re.match(r"^([A-Za-z0-9._]+?)_(M1|M5|M15|M30|H1|H4|D1)(_long)?\.csv$", p.name)
        if m:
            out.setdefault(m.group(1), {})[m.group(2)] = p
    return {k: v for k, v in out.items() if len(v) >= 2}


def _agree_at(coarse, fine, off: pd.Timedelta):
    t = coarse.index + off
    common = fine.index.intersection(t)
    if len(common) < 200:
        return None, None
    fc = fine.loc[common, "close"].values
    cc = coarse.loc[common - off, "close"].values
    rel = np.abs(fc - cc) / np.maximum(np.abs(cc), 1e-12)
    return pd.Series(rel, index=common), pd.Series(np.abs(fc - cc), index=common)


def agreement(coarse: pd.DataFrame, fine: pd.DataFrame, coarse_min: int,
              fine_min: int) -> tuple:
    """Per-year agreement between a coarse close and the fine bar that ends on it.

    Returns (table, offset_minutes). For D1 the session boundary is SEARCHED rather than
    assumed, because a rollover-hour mismatch and a feed splice look identical until you try
    the other offsets.
    """
    base = coarse_min - fine_min
    offsets = [base]
    if coarse_min >= 1440:
        offsets = sorted({base + h * 60 for h in range(-24, 25)})
    best = (None, None, -1.0)
    for om in offsets:
        rel, _ = _agree_at(coarse, fine, pd.Timedelta(minutes=om))
        if rel is None:
            continue
        frac = float((rel <= REL_TOL).mean())
        if frac > best[2]:
            best = (om, rel, frac)
    om, rel, _ = best
    if rel is None:
        return None, None
    _, absd = _agree_at(coarse, fine, pd.Timedelta(minutes=om))
    g, ga = rel.groupby(rel.index.year), absd.groupby(absd.index.year)
    return pd.DataFrame({"n": g.size(), "median_abs": ga.median(),
                         "frac_within_tol": g.apply(lambda x: float((x <= REL_TOL).mean()))}), om


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--quiet", action="store_true", help="only print pairs that FAIL")
    a = ap.parse_args()

    groups = discover()
    if a.symbol:
        groups = {k: v for k, v in groups.items() if k == a.symbol}
    if not groups:
        raise SystemExit("no multi-timeframe symbol groups found under data/")

    verdicts = []
    for sym, tfs in groups.items():
        frames = {tf: read(p) for tf, p in tfs.items()}
        frames = {k: v for k, v in frames.items() if v is not None}
        order = sorted(frames, key=lambda t: MINUTES[t])
        for i, fine in enumerate(order):
            for coarse in order[i + 1:]:
                tab, om = agreement(frames[coarse], frames[fine], MINUTES[coarse],
                                    MINUTES[fine])
                if tab is None:
                    continue
                bad = tab[tab.frac_within_tol < PASS_FRAC]
                ok = bad.empty
                verdicts.append(dict(symbol=sym, pair=f"{coarse} vs {fine}", ok=ok,
                                     bad_years=list(bad.index), offset_min=om,
                                     worst=float(tab.frac_within_tol.min())))
                if ok and a.quiet:
                    continue
                print("=" * 76)
                print(f"{sym}   {coarse} vs {fine}   (offset {om:+d} min)   "
                      f"{'OK — one series' if ok else '*** SPLICED — NOT ONE SERIES ***'}")
                print("=" * 76)
                print(tab.to_string(float_format=lambda x: f"{x:.4f}"))
                if not ok:
                    print(f"  diverges from {bad.index.min()} onward "
                          f"(worst year {tab.frac_within_tol.idxmin()}: "
                          f"{tab.frac_within_tol.min()*100:.1f}% within {REL_TOL*1e4:.1f}bp)")
                    print("  Any study pairing these two frames compares arms that had "
                          "different information. See V5_FINDINGS §3ba.")
                print()

    print("=" * 76)
    print("SUMMARY")
    print("=" * 76)
    for v in sorted(verdicts, key=lambda r: r["worst"]):
        flag = "OK    " if v["ok"] else "SPLICE"
        yrs = "" if v["ok"] else f"  from {min(v['bad_years'])}"
        print(f"  {flag}  {v['symbol']:<12} {v['pair']:<14} "
              f"worst year {v['worst']*100:5.1f}% within {REL_TOL*1e4:.1f}bp{yrs}")
    n_bad = sum(1 for v in verdicts if not v["ok"])
    print(f"\n{len(verdicts)} pairs checked, {n_bad} spliced.")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
