#!/usr/bin/env python3
"""Fetch OANDA intraday bars from the public Hugging Face library `bunqi/OANDA_*` (one provenance,
mid prices, UTC bar-open stamps, 2002-2006 -> present; no licence stated: private research only).

    python scripts/fetch_oanda_hf.py 5 XAUUSD EURUSD ...     # timeframe folder: 5S | 1 | 5 | 60 | D
    python scripts/fetch_oanda_hf.py 5 --panel               # the 24-instrument §3ez panel + XAUEUR

Writes data/oanda_hf/<tf>/<SYMBOL>.parquet (time, open, high, low, close, volume), git-ignored.
Raw monthly files are kept under data/oanda_hf/raw/<tf>/<SYMBOL>/ so a re-run only fetches what is
missing. Downloaded files are DATA: they are only ever opened with pyarrow.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "oanda_hf"
DATASETS = {
    "NAS100USD": "OANDA_NAS100USD_US-Nas-100", "US30USD": "OANDA_US30USD_US-Wall-St-30",
    "SPX500USD": "OANDA_SPX500USD_US-SPX-500", "DE30EUR": "OANDA_DE30EUR_Germany-30",
    "JP225USD": "OANDA_JP225USD_Japan-225", "UK100GBP": "OANDA_UK100GBP_UK-100",
    "AU200AUD": "OANDA_AU200AUD_Australia-200", "US2000USD": "OANDA_US2000USD_US-Russ-2000",
    "EURUSD": "OANDA_EURUSD_EURUSD", "GBPUSD": "OANDA_GBPUSD_GBPUSD", "USDJPY": "OANDA_USDJPY_USDJPY",
    "USDCAD": "OANDA_USDCAD_USDCAD", "USDCHF": "OANDA_USDCHF_USDCHF", "AUDUSD": "OANDA_AUDUSD_AUDUSD",
    "NZDUSD": "OANDA_NZDUSD_NZDUSD", "EURJPY": "OANDA_EURJPY_EURJPY", "GBPJPY": "OANDA_GBPJPY_GBPJPY",
    "EURGBP": "OANDA_EURGBP_EURGBP", "AUDJPY": "OANDA_AUDJPY_AUDJPY",
    "XAUUSD": "OANDA_XAUUSD_Gold", "XAGUSD": "OANDA_XAGUSD_Silver",
    "WTICOUSD": "OANDA_WTICOUSD_West-Texas-Oil", "BCOUSD": "OANDA_BCOUSD_Brent-Crude-Oil",
    "BTCUSD": "OANDA_BTCUSD_Bitcoin", "XAUEUR": "OANDA_XAUEUR_GoldEUR",
}
PANEL = [k for k in DATASETS if k != "XAUEUR"]
UA = {"User-Agent": "Mozilla/5.0 (research fetch)"}


def get(url: str, tries: int = 5) -> bytes:
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 + 4 * i)
    raise RuntimeError(f"{url}: {last}")


def fetch(sym: str, tf: str) -> None:
    ds = DATASETS[sym]
    meta = json.loads(get(f"https://huggingface.co/api/datasets/bunqi/{ds}"))
    files = sorted(s["rfilename"] for s in meta["siblings"]
                   if s["rfilename"].startswith(f"{tf}/") and s["rfilename"].endswith(".parquet"))
    raw = BASE / "raw" / tf / sym
    raw.mkdir(parents=True, exist_ok=True)

    def one(f: str) -> None:
        dst = raw / Path(f).name
        if dst.exists() and dst.stat().st_size > 0:
            return
        tmp = dst.with_suffix(".part")
        tmp.write_bytes(get(f"https://huggingface.co/datasets/bunqi/{ds}/resolve/main/{f}"))
        tmp.replace(dst)

    with ThreadPoolExecutor(12) as ex:
        list(ex.map(one, files))
    parts = [pq.read_table(raw / Path(f).name).to_pandas() for f in files]
    d = pd.concat(parts, ignore_index=True).drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True)
    for c in ("open", "high", "low", "close"):
        d[c] = d[c].astype("float64")
    out = BASE / tf / f"{sym}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out, index=False)
    t = pd.to_datetime(d.time, unit="s")
    print(f"{sym:10s} {tf}: {len(files)} files, {len(d):,} bars, {t.iloc[0]} -> {t.iloc[-1]}  sha {meta.get('sha', '')[:8]}", flush=True)


def main() -> None:
    tf = sys.argv[1]
    syms = PANEL + ["XAUEUR"] if "--panel" in sys.argv else sys.argv[2:]
    for s in syms:
        try:
            fetch(s, tf)
        except Exception as e:  # noqa: BLE001
            print(f"{s}: FAILED {e}", flush=True)
    print("FETCH DONE", flush=True)


if __name__ == "__main__":
    main()
