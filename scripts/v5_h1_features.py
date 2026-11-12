"""`f_base` made causal for H1 bars.

`scripts/v5_repr_ceiling.f_base` builds `d1_slope` / `d1_z` from the DAILY close, stamped at the
day's 00:00, forward-filled, then `.shift(6)`. Six bars is 24h on H4 — correct there — but only
6h on H1, so from 06:00 an H1 row could see THAT day's close up to 18h early (found 2026-11-12:
it produced AUC 0.59-0.62 and 66-70% hit rates on every one of 24 H1 cells, §3em). Here the daily
value is stamped at the END of its day (00:00 + 1 day) before the forward-fill, so a bar only
ever sees days that have closed. Everything else in f_base is a trailing window and unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.v5_repr_ceiling import f_base


def f_base_h1(df: pd.DataFrame) -> pd.DataFrame:
    F = f_base(df)
    c = df["close"].astype(float)
    d1 = c.resample("D").last().dropna()
    known = d1.index + pd.Timedelta(days=1)                 # a day's close is known at its END
    slope = pd.Series(np.sign(d1.ewm(span=20, min_periods=10).mean().diff()).to_numpy(), index=known)
    z = pd.Series(((d1 - d1.rolling(20).mean()) / d1.rolling(20).std().replace(0, np.nan)).to_numpy(),
                  index=known)
    at = c.index + pd.Timedelta(hours=1)                    # an H1 bar stamped t is decided at t+1h
    F["d1_slope"] = slope.reindex(at, method="ffill").to_numpy()
    F["d1_z"] = z.reindex(at, method="ffill").to_numpy()
    return F
