#!/bin/bash
# Daily FRED DGS2 (2-year Treasury yield) for ZigZag arm D's Y2UP size dial (data/v5_runs/zigzag_dial_ab/PREREG.md).
# Atomic: the new file replaces the old one only if it parses and has recent rows; otherwise the old file stays
# (and the dial fails open by itself once it is more than 10 days stale).
set -u
cd /home/trader/MT5 || exit 1
mkdir -p data/free_data
TMP=$(mktemp)
if curl -sf --http1.1 --max-time 60 -A "research-script/1.0" "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2" -o "$TMP" \
   && /home/trader/miniconda3/envs/envmt5/bin/python -c "
import pandas as pd, sys
d = pd.read_csv('$TMP'); d.columns = ['date', 'v']
d['v'] = pd.to_numeric(d['v'], errors='coerce'); d = d.dropna()
age = (pd.Timestamp.utcnow().tz_localize(None) - pd.to_datetime(d['date'].iloc[-1])).days
sys.exit(0 if len(d) > 1000 and age < 15 else 1)"; then
  mv "$TMP" data/free_data/fred_DGS2.csv && echo "$(date -u +%FT%TZ) DGS2 ok $(tail -1 data/free_data/fred_DGS2.csv)" >> data/v5_runs/fetch-fred.log
else
  rm -f "$TMP"; echo "$(date -u +%FT%TZ) DGS2 fetch FAILED, kept the old file" >> data/v5_runs/fetch-fred.log
fi
