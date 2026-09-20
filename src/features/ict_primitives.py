"""ICT / SMC primitive building blocks — shared by the concept-sweep scripts
(`scripts/v5_ict_*.py`, see the plan `~/.claude/plans/i-wanttt-you-to-playful-widget.md`).

Ten primitives cover the eleven concepts on the user's list plus the additional
ones (Equal Highs/Lows, Killzones, Turtle Soup, Unicorn) rather than one
implementation per concept — most named ICT setups are recombinations of "where
is structure" (P1-P3), "where is a gap/imbalance and has it since been
violated" (P4-P6), "what time/session is it" (P7), "where in the range is
price" (P8-P9), and "do two correlated assets agree" (P10).

Every function here is CAUSAL BY CONSTRUCTION — at bar i, only bars <= i are
ever read. `swing_levels()` in particular only reveals a fractal swing at
bar j+order (the earliest bar at which it is actually confirmable), never at
bar j itself. Every function in this module must pass
`src.evaluation.lookahead_probe.probe_lookahead` before any script trusts a
backtest built on it — P6 (`zone_lifecycle`) and P7 (`session_windows`) are the
two most-depended-on and get probed first and independently, per the plan.

P4 (order blocks) and P5 (fair value gaps) are reused directly from
`src.features.smc_signals` rather than reimplemented (P5's known pip-size bug
was fixed there directly, 2026-08-19, not duplicated/patched here).

Session-window calibration (P7) is EMPIRICAL, not an exact NY-timezone
conversion — see `session_windows()`'s docstring for the measured evidence and
its limits. Every concept built on P7 (PO3/AMD, Judas Swing, Silver Bullet,
Killzones) inherits that approximation and must say so when reporting results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.smc_signals import _atr14, _ohlcv, fair_value_gaps, order_blocks  # noqa: F401
from src.evaluation.lookahead_probe import synthetic_ohlc

__all__ = [
    "swing_levels", "break_of_structure", "liquidity_sweep",
    "zone_lifecycle", "session_windows", "daily_open_anchor",
    "premium_discount", "ote_zone", "displacement", "smt_divergence",
    "equal_levels", "build_smt_probe_frame",
]


# ================================================== regime-split reference boundaries
# MANDATORY per-instrument regime split (this session's rigor pipeline requires it for
# every trend/structure concept — it's the exact check that caught BOS's bull-beta
# problem). XAU's boundaries (2015-18 chop, 2019-20 bull, 2021-22 flat, 2023-26 bull)
# do NOT transfer to other instruments' own regime character — each derived here from
# that instrument's own annual returns + realized vol (2026-08-19), not assumed.
REGIME_BOUNDARIES: dict[str, list[tuple[str, str, str]]] = {
    "XAUUSD": [
        ("2015-01-01", "2018-12-31", "chop"),
        ("2019-01-01", "2020-12-31", "bull"),
        ("2021-01-01", "2022-12-31", "flat"),
        ("2023-01-01", "2026-12-31", "bull"),
    ],
    # EURUSD has no persistent multi-year drift (unlike gold) -- mostly chop, with two
    # identifiable directional stretches (2020 EUR-bull on COVID stimulus/USD weakness,
    # 2021-22 USD-bull/EUR-bear on the Fed hiking cycle).
    "EURUSD": [
        ("2015-01-01", "2019-12-31", "chop"),
        ("2020-01-01", "2020-12-31", "eur_bull"),
        ("2021-01-01", "2022-12-31", "usd_bull"),
        ("2023-01-01", "2026-12-31", "chop"),
    ],
    # SPX/NDX: predominantly bull with two clear bear/correction years and one
    # high-vol crash-then-recover year that should not be lumped into a smooth bull.
    "SPX": [
        ("2015-01-01", "2017-12-31", "bull"),
        ("2018-01-01", "2018-12-31", "bear"),
        ("2019-01-01", "2019-12-31", "bull"),
        ("2020-01-01", "2020-12-31", "crash_recover"),
        ("2021-01-01", "2021-12-31", "bull"),
        ("2022-01-01", "2022-12-31", "bear"),
        ("2023-01-01", "2026-12-31", "bull"),
    ],
    "NDX": [  # same boundaries as SPX (same underlying macro regime), own returns
        ("2015-01-01", "2017-12-31", "bull"),
        ("2018-01-01", "2018-12-31", "bear"),
        ("2019-01-01", "2019-12-31", "bull"),
        ("2020-01-01", "2020-12-31", "crash_recover"),
        ("2021-01-01", "2021-12-31", "bull"),
        ("2022-01-01", "2022-12-31", "bear"),
        ("2023-01-01", "2026-12-31", "bull"),
    ],
    # BRENT: event-driven, no persistent multi-year trend either direction (annual
    # returns alternate +50%/-19%/+51%/-18% with no clean multi-year run) -- treated as
    # one long "chop/event-driven" regime with flagged crisis years, not segmented into
    # bull/bear stretches the way an index or gold can be.
    "BRENT": [
        ("2015-01-01", "2026-12-31", "event_driven"),
    ],
}


def build_smt_probe_frame(n: int = 900, seed: int = 7) -> pd.DataFrame:
    """Two independent full-OHLC synthetic legs (`{open,high,low,close}_a/_b`),
    for lookahead-probing `smt_divergence` (which needs high/low per asset for
    `swing_levels`, unlike `v5_cross_asset_divergence.py`'s close-only
    `build_divergence_probe_frame`). Correlation between the legs is irrelevant
    for a truncation-invariance check — only causality of the computation is
    being tested."""
    a = synthetic_ohlc(n=n, seed=seed)
    b = synthetic_ohlc(n=n, seed=seed + 1000)
    out = pd.DataFrame(index=a.index)
    for col in ("open", "high", "low", "close"):
        out[f"{col}_a"] = a[col].values
        out[f"{col}_b"] = b[col].values
    return out


# ============================================================ P1 — swing structure
def swing_levels(df: pd.DataFrame, order: int = 5) -> tuple[pd.Series, pd.Series]:
    """Causal fractal swing high/low levels (mirrors `v5_smc_xau.py::swing_levels`,
    reimplemented fresh here for a clean module dependency).

    A bar j is a swing high if it is the max of [j-order, j+order]; that fact is
    only CONFIRMABLE at bar j+order (needs the `order` bars after j to know no
    later bar in the window beats it) — so it is only ever revealed there, never
    earlier. Returns the last-confirmed swing high/low, forward-filled, as of
    each bar.
    """
    o, h, l, c, idx = _ohlcv(df)
    n = len(c)
    last_sh, last_sl = np.nan, np.nan
    out_sh = np.full(n, np.nan)
    out_sl = np.full(n, np.nan)
    for i in range(n):
        j = i - order
        if j >= order:
            lo, hi = j - order, i  # hi == j + order
            if h[j] >= h[lo:hi + 1].max():
                last_sh = h[j]
            if l[j] <= l[lo:hi + 1].min():
                last_sl = l[j]
        out_sh[i] = last_sh
        out_sl[i] = last_sl
    return (pd.Series(out_sh, index=idx, name="swing_high"),
            pd.Series(out_sl, index=idx, name="swing_low"))


def equal_levels(swing_series: pd.Series, tol_atr: float, atr: pd.Series) -> pd.Series:
    """Equal Highs/Lows (liquidity pool) flag: True where the CURRENT confirmed
    swing level sits within `tol_atr` x ATR of the PREVIOUS distinct confirmed
    swing level of the same series — i.e. price has printed two nearly-equal
    swing points, the classic ICT 'liquidity resting at equal highs/lows' read.
    Causal: only compares the current confirmed value to the prior one."""
    prev_distinct = swing_series.where(swing_series != swing_series.shift(1)).ffill().shift(1)
    dist = (swing_series - prev_distinct).abs()
    return (dist <= tol_atr * atr).fillna(False)


# ==================================================================== P2 — BOS
def break_of_structure(df: pd.DataFrame, swing_high: pd.Series, swing_low: pd.Series) -> pd.Series:
    """+1 when close breaks above the last CONFIRMED swing high, -1 when close
    breaks below the last confirmed swing low, 0 otherwise. Causal given causal
    swing_high/swing_low (e.g. from `swing_levels`)."""
    close = df["close"]
    bos = pd.Series(0, index=df.index, dtype=int)
    bos[close > swing_high] = 1
    bos[close < swing_low] = -1
    return bos


# ============================================================= P3 — liquidity sweep
def liquidity_sweep(df: pd.DataFrame, swing_high: pd.Series, swing_low: pd.Series) -> pd.Series:
    """+1 = bullish sweep (this bar's low pierces below the confirmed swing low
    but the close reclaims back above it — a stop-hunt-and-reclaim); -1 = mirror
    bearish sweep. 0 otherwise."""
    low, high, close = df["low"], df["high"], df["close"]
    sweep = pd.Series(0, index=df.index, dtype=int)
    sweep[(low < swing_low) & (close > swing_low)] = 1
    sweep[(high > swing_high) & (close < swing_high)] = -1
    return sweep


# ===================================================== P6 — zone lifecycle state machine
def zone_lifecycle(
    df: pd.DataFrame,
    formed: pd.Series,
    top: pd.Series,
    bottom: pd.Series,
    kind: pd.Series,
    *,
    max_active: int = 20,
    expiry_bars: int = 2000,
) -> dict[str, pd.Series]:
    """Track a population of price zones (order blocks, FVGs, ...) through
    formed -> tested -> broken(role-flipped). Feeds Breaker Block, Mitigation
    Block, and Inversion FVG — all three are "a zone from P4/P5 that got broken
    and now acts with the opposite role," differing only in the entry rule a
    concept-script layers on top (this primitive stops at the flip event; retest
    -entry logic is concept-specific, not baked in here, to keep this reusable).

    Parameters
    ----------
    formed : bool Series, True at the bar a new zone is confirmed (e.g. from
        `order_blocks()` or `fair_value_gaps()` — must itself be causal).
    top, bottom : the zone's [bottom, top] bounds AT the formation bar.
    kind : +1 (bullish/support zone) or -1 (bearish/resistance zone) AT
        formation.
    max_active : cap on simultaneously-tracked zones (oldest dropped first) —
        keeps this O(n), not unbounded.
    expiry_bars : an untested/unbroken zone older than this is dropped.

    Returns a dict of per-bar Series: `nearest_zone_dir` (+1/-1/0, the closest
    active zone's CURRENT kind — reflects any flip), `nearest_zone_dist_atr`,
    `in_active_zone`, `broke_bull_zone`/`broke_bear_zone` (1 at the bar a zone's
    role just flipped), `broken_zone_top`/`broken_zone_bottom` (bounds of the
    zone that just flipped, nan otherwise).
    """
    o, h, l, c, idx = _ohlcv(df)
    n = len(c)
    formed_a = np.asarray(formed, dtype=bool)
    top_a = np.asarray(top, dtype=np.float64)
    bottom_a = np.asarray(bottom, dtype=np.float64)
    kind_a = np.asarray(kind, dtype=np.float64)
    a = _atr14(h, l, c)

    active: list[dict] = []
    nearest_dir = np.zeros(n)
    nearest_dist = np.full(n, np.nan)
    in_zone = np.zeros(n)
    broke_bull = np.zeros(n)
    broke_bear = np.zeros(n)
    broken_top = np.full(n, np.nan)
    broken_bot = np.full(n, np.nan)

    for i in range(n):
        if formed_a[i] and np.isfinite(top_a[i]) and np.isfinite(bottom_a[i]) and kind_a[i] != 0:
            active.append(dict(top=top_a[i], bottom=bottom_a[i], kind=float(kind_a[i]), formed_at=i))
            if len(active) > max_active:
                active.pop(0)

        still_active = []
        for z in active:
            if i - z["formed_at"] > expiry_bars:
                continue
            if z["kind"] > 0 and c[i] < z["bottom"]:
                broke_bear[i] = 1.0
                broken_top[i], broken_bot[i] = z["top"], z["bottom"]
                z["kind"] = -1.0
            elif z["kind"] < 0 and c[i] > z["top"]:
                broke_bull[i] = 1.0
                broken_top[i], broken_bot[i] = z["top"], z["bottom"]
                z["kind"] = 1.0
            still_active.append(z)
        active = still_active

        if active:
            best = None
            for z in active:
                mid = (z["top"] + z["bottom"]) / 2.0
                d = abs(c[i] - mid)
                if best is None or d < best[0]:
                    best = (d, z["kind"], z["bottom"] <= c[i] <= z["top"])
            nearest_dir[i] = best[1]
            nearest_dist[i] = best[0] / a[i]
            in_zone[i] = 1.0 if best[2] else 0.0

    return dict(
        nearest_zone_dir=pd.Series(nearest_dir, index=idx, name="nearest_zone_dir"),
        nearest_zone_dist_atr=pd.Series(nearest_dist, index=idx, name="nearest_zone_dist_atr"),
        in_active_zone=pd.Series(in_zone, index=idx, name="in_active_zone"),
        broke_bull_zone=pd.Series(broke_bull, index=idx, name="broke_bull_zone"),
        broke_bear_zone=pd.Series(broke_bear, index=idx, name="broke_bear_zone"),
        broken_zone_top=pd.Series(broken_top, index=idx, name="broken_zone_top"),
        broken_zone_bottom=pd.Series(broken_bot, index=idx, name="broken_zone_bottom"),
    )


# ======================================================== P7 — session / killzones
def session_windows(df: pd.DataFrame, *, dst_aware: bool = True) -> pd.DataFrame:
    """Approximate ICT killzone windows in UTC.

    CALIBRATION, NOT AN EXACT TIMEZONE CONVERSION: measured directly off this
    repo's own XAUUSD M15 tick-volume profile (2026-08-19) — the London/NY
    overlap volume peak sits at 13:00 UTC in Jun-Aug and 15:00 UTC in Dec-Feb, a
    2-HOUR seasonal shift. Standard NY-DST arithmetic alone only predicts a
    1-hour shift (EDT UTC-4 vs EST UTC-5), so the data's own broker/server
    timestamp most likely carries an additional DST convention layered on top of
    NY's. Rather than assert false precision, the windows below are fit directly
    to that measured 2-hour shift. Treat every concept built on this primitive
    (PO3/AMD, Judas Swing, Silver Bullet, Killzones) as APPROXIMATELY, not
    exactly, session-aligned — this is reported explicitly wherever those
    concepts' results are written up, not silently assumed away.

    `dst_aware=True` (default) shifts the NY windows by the measured 2h between
    Mar-Nov and Nov-Mar; `dst_aware=False` uses the Mar-Nov (summer) windows
    year-round, a simpler and slightly less precise baseline.

    Returns bool columns: asian, london, ny_am (Silver Bullet window proxy),
    ny_pm (Silver Bullet PM window proxy), ny_open (broader NY killzone).
    """
    idx = df.index
    hour = idx.hour.values
    if dst_aware:
        month = idx.month.values
        is_dst = (month >= 3) & (month <= 11)
        ny_am_start = np.where(is_dst, 13, 15)
        ny_am_end = np.where(is_dst, 15, 17)
        ny_pm_start = np.where(is_dst, 17, 19)
        ny_pm_end = np.where(is_dst, 18, 20)
        ny_open_start = np.where(is_dst, 12, 14)
        ny_open_end = np.where(is_dst, 16, 18)
    else:
        ny_am_start, ny_am_end = 13, 15
        ny_pm_start, ny_pm_end = 17, 18
        ny_open_start, ny_open_end = 12, 16

    out = pd.DataFrame(index=idx)
    out["asian"] = (hour >= 0) & (hour < 7)
    out["london"] = (hour >= 7) & (hour < 12)
    out["ny_am"] = (hour >= ny_am_start) & (hour < ny_am_end)
    out["ny_pm"] = (hour >= ny_pm_start) & (hour < ny_pm_end)
    out["ny_open"] = (hour >= ny_open_start) & (hour < ny_open_end)
    return out


def daily_open_anchor(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Causal running day-open / running day-high / running day-low, resetting
    at each UTC calendar-day boundary. `day_high_so_far`/`day_low_so_far`
    include the CURRENT bar (what a live system would know as of now) — feeds
    PO3/AMD (accumulation/manipulation/distribution needs 'today's range so
    far') and Judas Swing (needs the day's own open as the manipulation
    reference)."""
    idx = df.index
    day = idx.normalize()
    day_open = df["open"].groupby(day).transform("first")
    day_high = df["high"].groupby(day).cummax()
    day_low = df["low"].groupby(day).cummin()
    return day_open.rename("day_open"), day_high.rename("day_high_sofar"), day_low.rename("day_low_sofar")


# ============================================== P8 — premium/discount + OTE
def premium_discount(df: pd.DataFrame, swing_high: pd.Series, swing_low: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Position of close within [swing_low, swing_high] on a 0-1 scale (0 = at
    the low / deep discount, 1 = at the high / deep premium) plus a discrete
    zone label (-1 discount/favor-longs, 0 equilibrium, +1 premium/favor-shorts).
    Causal given causal swing_high/swing_low."""
    close = df["close"]
    rng = (swing_high - swing_low).replace(0, np.nan)
    pos = ((close - swing_low) / rng).clip(0, 1)
    zone = pd.Series(0, index=df.index, dtype=int)
    zone[pos < 0.5] = -1
    zone[pos > 0.5] = 1
    return pos.rename("pd_position"), zone.rename("pd_zone")


def ote_zone(df: pd.DataFrame, swing_high: pd.Series, swing_low: pd.Series, direction: pd.Series | int) -> pd.Series:
    """Optimal Trade Entry: True while price sits in the classic 62%-79%
    Fibonacci retracement zone of the [swing_low, swing_high] leg.

    `direction`: +1 = bullish leg (swing_low->swing_high; a LONG-continuation
    entry wants price to retrace DOWN to 62-79% retraced from the high, i.e.
    `pd_position` in [0.21, 0.38]); -1 = bearish leg (mirror: retrace UP,
    `pd_position` in [0.62, 0.79]). Scalar or per-bar Series."""
    pos, _ = premium_discount(df, swing_high, swing_low)
    dir_arr = np.full(len(df), direction, dtype=float) if np.isscalar(direction) else np.asarray(direction, dtype=float)
    p = pos.values
    in_ote = np.zeros(len(df), dtype=bool)
    bull = dir_arr > 0
    bear = dir_arr < 0
    in_ote[bull] = (p[bull] >= 0.21) & (p[bull] <= 0.38)
    in_ote[bear] = (p[bear] >= 0.62) & (p[bear] <= 0.79)
    return pd.Series(in_ote, index=df.index, name="in_ote")


# ========================================================== P9 — displacement
def displacement(df: pd.DataFrame, atr_mult: float = 2.0) -> pd.Series:
    """+1/-1/0: a bar whose body |close-open| exceeds `atr_mult` x ATR(14) and
    closes in the direction of its body — the ICT 'displacement' candle,
    associated with aggressive institutional participation and often the origin
    of a Fair Value Gap."""
    o, h, l, c, idx = _ohlcv(df)
    a = _atr14(h, l, c)
    body = c - o
    strong = np.abs(body) >= atr_mult * a
    disp = np.zeros(len(c))
    disp[strong & (body > 0)] = 1.0
    disp[strong & (body < 0)] = -1.0
    return pd.Series(disp, index=idx, name="displacement")


# =================================================== P10 — cross-asset SMT divergence
def smt_divergence(
    swing_high_a: pd.Series, swing_low_a: pd.Series, close_a: pd.Series,
    swing_high_b: pd.Series, swing_low_b: pd.Series, close_b: pd.Series,
) -> pd.Series:
    """Cross-asset structural (Smart Money Tool) divergence: asset A confirms a
    new swing extreme that asset B does NOT. +1 = bullish SMT (A swept to a new
    low, B did not confirm — fade A's low, expect reversal up); -1 = bearish SMT
    (mirror, A swept to a new high B did not confirm). All six inputs must share
    the same aligned index and each be causal (e.g. from `swing_levels` on each
    asset)."""
    new_low_a = close_a <= swing_low_a
    new_high_a = close_a >= swing_high_a
    new_low_b = close_b <= swing_low_b
    new_high_b = close_b >= swing_high_b
    smt = pd.Series(0, index=close_a.index, dtype=int)
    smt[new_low_a & ~new_low_b.reindex(close_a.index).fillna(False)] = 1
    smt[new_high_a & ~new_high_b.reindex(close_a.index).fillna(False)] = -1
    return smt.rename("smt_divergence")
