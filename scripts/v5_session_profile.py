"""DESCRIPTIVE session-open behaviour on XAUUSD M15. No trading rule yet — facts first.

WHY DESCRIPTIVE FIRST. §3ak already ran 21,300 conditional triggers on XAU H4 with a
maximum-statistic null and found the best one indistinguishable from a no-edge dataset (p 0.110,
null median |z| 9.34). Its lesson is that searching harder provably raises the bar faster than
the statistic, so the only defensible way into this space is a SMALL, pre-declared grid chosen
from measured behaviour — not another sweep. This file measures the behaviour; the grid comes
after, and is declared before it is run.

WHAT §3ak COULD NOT SEE. It coded `session` as a tercile on H4 bars. An H4 bar cannot represent
"the first fifteen minutes of the London open", which is exactly what was asked for. M15 has
270,458 clean bars back to 2015 and the question is genuinely untested here.

CALIBRATION, MEASURED NOT ASSUMED (2026-09-21, clean M15):
  * server clock = UTC. The single most volatile 15-min slot in the whole sample is 13:30
    (13.04bp mean |return|), which is the US 08:30 ET data release.
  * the seasonal shift is ONE hour, not two: peak hour 13h in Apr-Oct, 14h in Nov-Mar. This
    CORRECTS `ict_primitives.session_windows`, whose docstring asserts a measured 2-hour shift
    and hard-codes a 2h offset. That was calibrated on 2026-08-19, before §3bb found
    XAUUSD_H4_long.csv splices two brokers, and it means every ICT concept built on that
    primitive was session-aligned up to an hour wrong. Those concepts all failed anyway, so no
    conclusion changes, but the primitive is wrong and is now known to be.

Session anchors therefore use real exchange local time via zoneinfo, not a hard-coded offset:
Tokyo 09:00 JST (no DST), London 08:00 Europe/London, New York 09:30 America/New_York.

    python scripts/v5_session_profile.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "v5_runs" / "session_study"

SESSIONS = {
    "ASIA":   ("Asia/Tokyo",        9, 0, 7.0),    # 09:00 JST, 7h window
    "LONDON": ("Europe/London",     8, 0, 5.0),    # 08:00 local, 5h to the NY overlap
    "NY":     ("America/New_York",  9, 30, 6.5),   # 09:30 ET, to the 16:00 equity close
}


def session_open_utc(days: pd.DatetimeIndex, tz: str, hh: int, mm: int) -> pd.DatetimeIndex:
    """The UTC instant of a session's local open on each calendar day, DST-correct.

    Built from the exchange's own local clock rather than a fixed UTC offset, because the whole
    point of a session study is that the anchor is right: London's open is 08:00 UTC in winter
    and 07:00 in summer, and getting that wrong smears every measurement by an hour.
    """
    z = ZoneInfo(tz)
    loc = pd.DatetimeIndex([pd.Timestamp(d.date(), tz=z) + pd.Timedelta(hours=hh, minutes=mm)
                            for d in days])
    return loc.tz_convert("UTC").tz_localize(None)


def build(m15: pd.DataFrame) -> dict:
    out = {}
    days = pd.DatetimeIndex(sorted(set(m15.index.normalize())))
    days = days[days.dayofweek < 5]
    mi = m15.index
    hi, lo, cl, op = (m15["high"].values, m15["low"].values,
                      m15["close"].values, m15["open"].values)
    for name, (tz, hh, mm, hours) in SESSIONS.items():
        opens = session_open_utc(days, tz, hh, mm)
        rows = []
        for d, t0 in zip(days, opens):
            s = int(np.searchsorted(mi.values, np.datetime64(t0), side="left"))
            if s >= len(mi) or mi[s] != t0:
                continue                      # the open must land on a real M15 bar
            e = int(np.searchsorted(mi.values,
                                    np.datetime64(t0 + pd.Timedelta(hours=hours)),
                                    side="left"))
            if e >= len(mi) or e - s < 8:
                continue
            rows.append(dict(day=d, t0=t0, s=s, e=e,
                             o=op[s], c=cl[e], h=hi[s:e].max(), l=lo[s:e].min()))
        df = pd.DataFrame(rows)
        if not len(df):
            continue
        df["range_bp"] = (df.h - df.l) / df.o * 1e4
        df["ret_bp"] = (df.c - df.o) / df.o * 1e4
        out[name] = df
    return out


def main() -> None:
    from scripts.v5_advisor_measure import load_frames
    _, m15 = load_frames()
    S = build(m15)
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("SESSION BEHAVIOUR, XAUUSD, clean M15 2015-2026 (DST-correct local anchors)")
    print("=" * 96)
    print(f"  {'session':8s} {'n days':>7s} {'window':>7s} {'range bp':>10s} {'|ret| bp':>9s} "
          f"{'mean ret':>9s} {'t':>6s} {'up%':>6s}")
    for name, df in S.items():
        hours = SESSIONS[name][3]
        t = df.ret_bp.mean() / (df.ret_bp.std() / np.sqrt(len(df)))
        print(f"  {name:8s} {len(df):7,} {hours:6.1f}h {df.range_bp.mean():10.1f} "
              f"{df.ret_bp.abs().mean():9.1f} {df.ret_bp.mean():+9.2f} {t:+6.2f} "
              f"{(df.ret_bp > 0).mean()*100:5.1f}%")
    print("\n  'mean ret' is the session's own drift. A t above ~2 would be a tradeable")
    print("  time-of-day drift on its own; anything less is the backdrop, not a signal.")

    for name, df in S.items():
        df.to_parquet(OUT / f"session_{name}.parquet")
    print(f"\nwrote {len(S)} session frames to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
