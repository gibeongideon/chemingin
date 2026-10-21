#!/usr/bin/env python3
"""Forex Factory economic calendar -> CSV, plus a growing historical archive.

    python3 scripts/ff_calendar.py --csv data/ff_this_week.csv
    python3 scripts/ff_calendar.py --impact High,Medium --currencies USD,EUR
    python3 scripts/ff_calendar.py --archive            # merge into the long-run archive
    python3 scripts/ff_calendar.py --from-archive --impact High --csv data/high_impact.csv

WHICH ENDPOINT. This uses `nfs.faireconomy.media/ff_calendar_thisweek.json`, the same feed
`src/v5/news_filter.py` already reads for the live news filter. The widely-circulated
`forexfactory.com/ffcal_week_this.json` is DEAD -- it returns HTTP 403 (verified 2026-10-21).

*** TWO LIMITATIONS THAT DECIDE WHAT THIS CAN BE USED FOR. Measured, not assumed. ***

1. ONLY THE CURRENT WEEK EXISTS. `ff_calendar_lastweek`, `_nextweek` and `_thismonth` all return
   HTTP 404. There is NO backfill, so a historical calendar can only be ACCUMULATED going forward
   -- which is what `--archive` is for. Run it weekly and the history builds from the day you
   start, not before.

2. THERE IS NO `actual` FIELD. The feed carries title / country / date / impact / forecast /
   previous and nothing else, and it does NOT add the realised value after release -- verified
   against 40 already-released events in the current week. So this feed gives event TIMING and
   EXPECTED impact, never the SURPRISE (actual - forecast), which is the part that moves price.

   The consequence for research is specific and worth stating plainly: event timing alone is
   exactly the information V5_FINDINGS 3bd already tested and closed (the 08:30 ET reaction is
   2.3x a normal bar, yet 0/72 cells cleared the spread, signed t +1.39 against a null median of
   +2.32). That study's own conclusion was that it "needs a real calendar to reopen" -- meaning
   one with ACTUALS. This feed is not that calendar and cannot reopen it. For surprise-based
   research use a vintage-aware source instead (FRED/ALFRED gives as-of-date macro releases free).

   Where this feed IS genuinely useful is the live book's news filter, which only ever needed to
   know WHEN a high-impact release lands so it can stand aside. That is already wired up.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
ARCHIVE = ROOT / "data" / "ff_calendar_archive.csv"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# Identity of an event. Deliberately excludes forecast/previous because BOTH get revised between
# the first publication and the release; keying on them would create duplicate rows for one event.
KEY = ["title", "country", "date_utc"]


def fetch(url: str = FEED_URL, timeout: int = 20, tries: int = 4) -> list[dict]:
    """Fetch the feed, backing off on 429 and falling back to the live filter's cache.

    *** THIS ENDPOINT RATE-LIMITS, and it is strict. *** Four probe requests in a couple of minutes
    was enough to earn HTTP 429 (measured 2026-10-21), which is also why `src/v5/news_filter.py`
    refreshes only every 60 minutes rather than every pass. Treating a 429 as a hard failure would
    make this script flaky by design, and -- worse -- an unhandled 429 inside a trading wrapper
    would abort a pass over a calendar refresh, so the fallback matters operationally.
    """
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
            if r.status_code == 429:
                wait = 15 * (2 ** i)
                print(f"  feed rate-limited (429), retrying in {wait}s "
                      f"[{i + 1}/{tries}]", file=sys.stderr)
                time.sleep(wait)
                last = "HTTP 429 (rate limited)"
                continue
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                raise SystemExit(f"unexpected payload type {type(data).__name__} from {url}")
            return data
        except requests.RequestException as e:
            last = f"{type(e).__name__}: {str(e)[:80]}"
            if i < tries - 1:
                time.sleep(10 * (2 ** i))

    # Fall back to whatever the live news filter last cached, clearly labelled as stale.
    cache = ROOT / "data" / "news_cache.json"
    if cache.exists():
        blob = json.loads(cache.read_text())
        evs = blob.get("events", [])
        age_h = (datetime.now(timezone.utc).timestamp() - float(blob.get("fetched_at", 0))) / 3600
        print(f"  feed unreachable ({last}); FALLING BACK to data/news_cache.json "
              f"({len(evs)} events, {age_h:.0f}h old)", file=sys.stderr)
        return evs
    raise SystemExit(f"feed unreachable ({last}) and no cache at {cache}")


def to_frame(events: list[dict]) -> pd.DataFrame:
    d = pd.DataFrame(events)
    if d.empty:
        return d
    # The feed stamps an explicit UTC offset (e.g. -04:00). Keep BOTH: the original local stamp a
    # trader reads, and a normalised UTC column, because every join in this repo is in UTC and a
    # mixed-offset string column silently sorts wrong across a DST boundary.
    ts = pd.to_datetime(d["date"], utc=True, errors="coerce")
    d["date_utc"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    d["date_local"] = d["date"]
    d["day_utc"] = ts.dt.strftime("%Y-%m-%d")
    d["time_utc"] = ts.dt.strftime("%H:%M")
    d["weekday"] = ts.dt.day_name()
    d = d.drop(columns=["date"])
    cols = ["day_utc", "time_utc", "weekday", "title", "country", "impact",
            "forecast", "previous", "date_utc", "date_local"]
    return d[[c for c in cols if c in d.columns]].sort_values("date_utc").reset_index(drop=True)


def apply_filters(d: pd.DataFrame, impact: str | None, currencies: str | None) -> pd.DataFrame:
    if d.empty:
        return d
    if impact:
        want = {s.strip().lower() for s in impact.split(",") if s.strip()}
        d = d[d["impact"].str.lower().isin(want)]
    if currencies:
        want = {s.strip().upper() for s in currencies.split(",") if s.strip()}
        d = d[d["country"].str.upper().isin(want)]
    return d.reset_index(drop=True)


def merge_archive(new: pd.DataFrame, path: Path = ARCHIVE) -> tuple[int, int, int]:
    """Append to the long-run archive without ever losing a row. Returns (added, revised, total).

    Forecast and previous are REVISED as a release approaches, so a plain drop_duplicates on the
    whole row would accumulate near-duplicates of the same event. Rows are keyed on
    (title, country, date_utc) and the newest values win, while `first_seen` preserves when the
    event was first published -- which is the only publication-lag information this feed carries.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new = new.copy()
    new["first_seen"] = now
    new["last_seen"] = now

    if path.exists():
        old = pd.read_csv(path, dtype=str).fillna("")
        for c in new.columns:
            if c not in old.columns:
                old[c] = ""
        combined = pd.concat([old, new.astype(str)], ignore_index=True)
        # keep the EARLIEST first_seen per event, the LATEST everything else
        first = combined.groupby(KEY, as_index=False)["first_seen"].min()
        combined = combined.drop_duplicates(subset=KEY, keep="last").drop(columns=["first_seen"])
        combined = combined.merge(first, on=KEY, how="left")
        added = len(combined) - len(old)
        revised = len(new) - added
    else:
        combined = new.astype(str)
        added, revised = len(new), 0

    combined = combined.sort_values("date_utc").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return added, revised, len(combined)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="write the (filtered) calendar here")
    ap.add_argument("--impact", help="comma list: High,Medium,Low,Holiday")
    ap.add_argument("--currencies", help="comma list: USD,EUR,GBP,...")
    ap.add_argument("--archive", action="store_true",
                    help=f"merge this week into {ARCHIVE.relative_to(ROOT)}")
    ap.add_argument("--from-archive", action="store_true",
                    help="read the archive instead of the network (for filtering/export)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    if a.from_archive:
        if not ARCHIVE.exists():
            raise SystemExit(f"no archive at {ARCHIVE} -- run with --archive first")
        d = pd.read_csv(ARCHIVE, dtype=str).fillna("")
        src = f"archive {ARCHIVE.relative_to(ROOT)}"
    else:
        d = to_frame(fetch())
        src = FEED_URL

    total_before = len(d)
    if a.archive and not a.from_archive:
        added, revised, total = merge_archive(d)
        if not a.quiet:
            print(f"archive {ARCHIVE.relative_to(ROOT)}: +{added} new, {revised} revised, "
                  f"{total} rows total")

    d = apply_filters(d, a.impact, a.currencies)

    if a.csv:
        out = Path(a.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(out, index=False)
        if not a.quiet:
            print(f"wrote {out} -- {len(d)} of {total_before} events")

    if not a.quiet:
        print(f"\nsource: {src}")
        if d.empty:
            print("no events match the filter")
        else:
            show = d[["day_utc", "time_utc", "country", "impact", "title",
                      "forecast", "previous"]].head(40)
            print(show.to_string(index=False))
            if len(d) > 40:
                print(f"... {len(d) - 40} more")
        if not a.from_archive:
            print("\nNOTE: this feed has NO `actual` column and covers ONLY the current week "
                  "(lastweek/nextweek return 404).\n"
                  "      Useful for standing aside around releases; NOT sufficient to train a "
                  "surprise signal -- see this file's docstring.")


if __name__ == "__main__":
    sys.exit(main())
