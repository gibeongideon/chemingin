#!/usr/bin/env python3
"""Fetch the 2-year Treasury yield (FRED DGS2) for ZigZag arm D's Y2UP size dial
(data/v5_runs/zigzag_dial_ab/PREREG.md). FRED is intermittently unreachable from the VPS, so it falls back to the
US Treasury's daily par-yield-curve CSV ('2 Yr' column), which is the source FRED's DGS2 republishes.
Writes data/free_data/fred_DGS2.csv (date,value) atomically; keeps the old file on any failure.

    python scripts/fetch_dgs2.py
"""
from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "free_data" / "fred_DGS2.csv"
UA = {"User-Agent": "Mozilla/5.0 (research-script/1.0)"}


def get(url: str) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45).read()


def from_fred() -> pd.DataFrame:
    d = pd.read_csv(io.BytesIO(get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2")))
    d.columns = ["date", "DGS2"]
    return d


def from_treasury() -> pd.DataFrame:
    year = pd.Timestamp.utcnow().year
    parts = []
    for y in (year - 1, year):
        url = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
               f"{y}/all?type=daily_treasury_yield_curve&field_tdr_date_value={y}&page&_format=csv")
        t = pd.read_csv(io.BytesIO(get(url)))
        parts.append(pd.DataFrame({"date": pd.to_datetime(t["Date"]).dt.strftime("%Y-%m-%d"), "DGS2": t["2 Yr"]}))
    d = pd.concat(parts).drop_duplicates("date").sort_values("date")
    if OUT.exists():                       # keep the long FRED history and append the Treasury rows
        old = pd.read_csv(OUT)
        old.columns = ["date", "DGS2"]
        d = pd.concat([old[old["date"] < d["date"].iloc[0]], d])
    return d


def ok(d: pd.DataFrame) -> bool:
    v = pd.to_numeric(d["DGS2"], errors="coerce").dropna()
    age = (pd.Timestamp.utcnow().tz_localize(None) - pd.to_datetime(d["date"].iloc[-1])).days
    return len(v) > 150 and age < 15


def main() -> int:
    for name, fn in (("FRED", from_fred), ("Treasury", from_treasury)):
        try:
            d = fn()
            if ok(d):
                OUT.parent.mkdir(parents=True, exist_ok=True)
                tmp = OUT.with_suffix(".tmp")
                d.to_csv(tmp, index=False)
                tmp.replace(OUT)
                print(f"DGS2 ok from {name}: {len(d)} rows, last {d['date'].iloc[-1]} = {d['DGS2'].iloc[-1]}")
                return 0
            print(f"{name}: data failed the sanity check")
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: {type(exc).__name__}: {exc}")
    print("DGS2 fetch FAILED from both sources; kept the old file")
    return 1


if __name__ == "__main__":
    sys.exit(main())
