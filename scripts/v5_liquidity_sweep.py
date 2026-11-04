#!/usr/bin/env python3
"""Liquidity-sweep reversal, built as the user specified it (2026-11-04), tested cross-asset.

PRE-REGISTERED BEFORE THE FIRST RUN. Every rule below is fixed; nothing is tuned on the result.

WHAT IS NEW VERSUS WHAT IS ALREADY CLOSED ON XAU
  closed: H4/H1 EQH/EQL pool fade (§3v, SR -0.44); Asian-range sweep (§3bi, -2.95bp, 0/12 yrs);
          M15 sweep+reclaim as an ENTRY TIMER on the advisor call (§3bc, adverse selection t -5.96)
  new:    BTC / USTEC / FX at all; and the FULL recipe -- sweep candle closes back inside AND a
          lower-timeframe displacement + market-structure-shift (MSS) confirms -- as a standalone
          signal rather than a timer.

LEVELS (causal: known before the sweep bar opens), on H1, server clock
  PD  prior server-day high / low, still unswept today
  SW  most recent confirmed 3-bar fractal swing high / low within 120 bars, still unswept
      (a fractal at j is only known at j+3)

SWEEP on H1 bar t (bearish shown; bullish mirrors)
  high[t] > L  and  close[t] < L  and  upper wick >= 50% of the bar's range
  -> SELL (buy-side liquidity taken, rejected). Bullish sweep of a low -> BUY.

ARMS
  A  enter at the next H1 open.                                     (full H1 history)
  B  the full recipe: on M15, within 16 bars (4h) after the sweep bar closes, the first M15 close
     beyond the 2h M15 swing that preceded the extreme (MSS), whose body >= 0.8 x M15 ATR20
     (displacement), before price re-takes the extreme. Enter at the next M15 open. (M15 window)

EXIT (identical for every arm and every control)
  stop = sweep extreme +/- 0.1 x H1 ATR20; target = 2R; time stop 24h.
  Path on the arm's own bars; a bar touching BOTH stop and target counts as the STOP.
  Cost = the bar's quoted spread (Switch, bid bars), one per round trip. Swap ignored (<=24h).

CONTROLS (each one has killed an idea in this repo before)
  1  RANDOM-TIME BASE: same direction, same stop distance in ATR, same exit, entered at a random
     other bar. Excess = signal - base. This is the drift control (gold rises; BTC rises).
  2  NO-LEVEL WICK PLACEBO: identical candle shape (wick >= 50%, range >= that cell's median sweep
     range) that pierced NO active level. If this earns as much, the LEVEL is decoration.
  3  BREAKOUT (acceptance) on the same levels: close beyond L -> trade WITH it. The concept says
     sweep reverses and acceptance continues; both are measured.
  4  SKIPPED SAMPLE (arm B): what arm A earned on the sweeps that B's confirmation declined.
  5  MAX-STAT NULL over all A/B cells: signal times circularly shifted, 400 draws, GROSS R.
  6  per-year sign and halves.

    python scripts/v5_liquidity_sweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sweep_switch"
OUT = ROOT / "data" / "v5_runs" / "liquidity_sweep"

WICK_FRAC, SW_K, SW_LOOKBACK = 0.5, 3, 120
STOP_BUF_ATR, RR, MAX_HOLD_H = 0.1, 2.0, 24
MSS_WINDOW, MSS_REF_BARS, DISP_BODY_ATR = 16, 8, 0.8
N_NULL, SEED = 400, 7
SYMS = ["XAUUSDs", "BTCUSDs", "USTECs", "EURUSDs", "GBPUSDs"]


# ---------------------------------------------------------------------------- data -----
def load(sym: str, tf: str) -> pd.DataFrame | None:
    f = DATA / f"{sym}_{tf}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, parse_dates=["time"]).set_index("time").sort_index()
    d = d[~d.index.duplicated()]
    point = json.loads((DATA / "meta.json").read_text())[sym]["point"]
    d["spr"] = d["spread"].astype(float) * float(point)
    return d


def h1_from_m15(m15: pd.DataFrame) -> pd.DataFrame:
    """Only for FX, whose H1 fetch failed. Label = bar OPEN, so a bar stamped t is used only
    from t+1h onward exactly as a native H1 bar would be (no cross-timeframe lookahead)."""
    g = m15.resample("1h", label="left", closed="left")
    d = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                      "low": g["low"].min(), "close": g["close"].last(),
                      "spr": g["spr"].median()}).dropna()
    return d[g["open"].count().reindex(d.index) >= 3]


def atr(d: pd.DataFrame, n: int = 20) -> np.ndarray:
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()],
                   axis=1).max(axis=1)
    return tr.rolling(n).mean().to_numpy()


# -------------------------------------------------------------------------- levels -----
def pd_levels(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Prior server-day high/low, NaN once already swept earlier today (before bar t)."""
    day = d.index.normalize()
    dh = d.groupby(day)["high"].max()
    dl = d.groupby(day)["low"].min()
    days = dh.index
    prev_h = pd.Series(dh.shift(1).to_numpy(), index=days).reindex(day).to_numpy()
    prev_l = pd.Series(dl.shift(1).to_numpy(), index=days).reindex(day).to_numpy()
    # today's running extreme BEFORE bar t
    hi_before = d.groupby(day)["high"].transform(lambda s: s.cummax().shift(1)).to_numpy()
    lo_before = d.groupby(day)["low"].transform(lambda s: s.cummin().shift(1)).to_numpy()
    up = np.where(np.isnan(hi_before) | (hi_before <= prev_h), prev_h, np.nan)
    dn = np.where(np.isnan(lo_before) | (lo_before >= prev_l), prev_l, np.nan)
    return up, dn


def swing_levels(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Most recent confirmed, still-unswept fractal swing high/low, as known at bar t's open."""
    h, l = d["high"].to_numpy(), d["low"].to_numpy()
    n, k = len(d), SW_K
    up, dn = np.full(n, np.nan), np.full(n, np.nan)
    act_h, act_l = [], []                      # (index, level)
    for t in range(n):
        j = t - 1 - k                          # fractal at j is confirmed by bar t-1
        if j >= k:
            if h[j] == h[j - k:j + k + 1].max():
                act_h.append((j, h[j]))
            if l[j] == l[j - k:j + k + 1].min():
                act_l.append((j, l[j]))
        if t:                                  # bar t-1 sweeps (removes) any level it crossed
            act_h = [(i, v) for i, v in act_h if h[t - 1] <= v and t - i <= SW_LOOKBACK]
            act_l = [(i, v) for i, v in act_l if l[t - 1] >= v and t - i <= SW_LOOKBACK]
        if act_h:
            up[t] = act_h[-1][1]
        if act_l:
            dn[t] = act_l[-1][1]
    return up, dn


# -------------------------------------------------------------------------- events -----
def events(d: pd.DataFrame, up: np.ndarray, dn: np.ndarray, a: np.ndarray) -> pd.DataFrame:
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    rng = h - l
    uw = h - np.maximum(o, c)
    lw = np.minimum(o, c) - l
    with np.errstate(invalid="ignore", divide="ignore"):
        bear = (h > up) & (c < up) & (uw >= WICK_FRAC * rng) & (rng > 0)
        bull = (l < dn) & (c > dn) & (lw >= WICK_FRAC * rng) & (rng > 0)
        brk_up = (h > up) & (c > up)
        brk_dn = (l < dn) & (c < dn)
    rows = []
    for i in np.flatnonzero(bear):
        rows.append((i, -1, h[i], "sweep"))
    for i in np.flatnonzero(bull):
        rows.append((i, +1, l[i], "sweep"))
    for i in np.flatnonzero(brk_up):           # breakout WITH it; stop beyond the bar's low
        rows.append((i, +1, l[i], "breakout"))
    for i in np.flatnonzero(brk_dn):
        rows.append((i, -1, h[i], "breakout"))
    e = pd.DataFrame(rows, columns=["i", "side", "extreme", "kind"])
    e = e[np.isfinite(a[e["i"].to_numpy()])] if len(e) else e
    return e.sort_values("i").reset_index(drop=True)


def placebo_events(d: pd.DataFrame, up: np.ndarray, dn: np.ndarray, a: np.ndarray,
                   min_range: float) -> pd.DataFrame:
    """Same rejection candle, no level pierced."""
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    rng = h - l
    uw, lw = h - np.maximum(o, c), np.minimum(o, c) - l
    with np.errstate(invalid="ignore"):
        no_up = ~(h > up)                      # NaN level -> comparison False -> "no pierce"
        no_dn = ~(l < dn)
        bear = (uw >= WICK_FRAC * rng) & (rng >= min_range) & no_up & no_dn
        bull = (lw >= WICK_FRAC * rng) & (rng >= min_range) & no_up & no_dn
    rows = [(i, -1, h[i], "placebo") for i in np.flatnonzero(bear)]
    rows += [(i, +1, l[i], "placebo") for i in np.flatnonzero(bull)]
    e = pd.DataFrame(rows, columns=["i", "side", "extreme", "kind"])
    e = e[np.isfinite(a[e["i"].to_numpy()])] if len(e) else e
    return e.sort_values("i").reset_index(drop=True)


# --------------------------------------------------------------------------- trade -----
_ARR: dict = {}                                # frames live for the whole run, so id() is safe


def simulate(px: pd.DataFrame, entry_i: int, side: int, stop: float,
             bars_max: int) -> tuple[float, float, float] | None:
    """(gross R, net R, gross bp). Entry at px.open[entry_i]. Stop-first on a two-touch bar."""
    key = id(px)
    if key not in _ARR:
        _ARR[key] = tuple(px[x].to_numpy() for x in ("open", "high", "low", "close", "spr"))
    o, h, l, c, s = _ARR[key]
    if entry_i >= len(px):
        return None
    e = o[entry_i]
    risk = (e - stop) * side
    if not np.isfinite(risk) or risk <= 0:
        return None
    tp = e + side * RR * risk
    end = min(entry_i + bars_max, len(px)) - 1
    exit_px = c[end]
    for j in range(entry_i, end + 1):
        hit_stop = (l[j] <= stop) if side > 0 else (h[j] >= stop)
        hit_tp = (h[j] >= tp) if side > 0 else (l[j] <= tp)
        if hit_stop:
            exit_px = stop
            break
        if hit_tp:
            exit_px = tp
            break
    g = (exit_px - e) * side
    cost = s[entry_i] if np.isfinite(s[entry_i]) else 0.0
    return g / risk, (g - cost) / risk, 1e4 * g / e


def arm_a(d: pd.DataFrame, a: np.ndarray, ev: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in ev.itertuples():
        stop = r.extreme + (STOP_BUF_ATR * a[r.i] if r.side < 0 else -STOP_BUF_ATR * a[r.i])
        res = simulate(d, r.i + 1, r.side, stop, MAX_HOLD_H)
        if res:
            out.append((d.index[r.i], r.i, r.side, (stop - d["open"].iat[r.i + 1]) * -r.side
                        / a[r.i], *res))
    return pd.DataFrame(out, columns=["t", "i", "side", "stop_atr", "gR", "nR", "gbp"])


def arm_b(h1: pd.DataFrame, a: np.ndarray, ev: pd.DataFrame, m15: pd.DataFrame
          ) -> tuple[pd.DataFrame, set]:
    """Full recipe on M15. Returns trades and the set of H1 indices that confirmed."""
    ma = atr(m15)
    idx = m15.index
    mo, mh, ml, mc = (m15[x].to_numpy() for x in ("open", "high", "low", "close"))
    out, confirmed = [], set()
    for r in ev.itertuples():
        t0 = h1.index[r.i]
        if t0 < idx[0] or t0 + pd.Timedelta(hours=1) > idx[-1]:
            continue
        inbar = np.flatnonzero((idx >= t0) & (idx < t0 + pd.Timedelta(hours=1)))
        if not len(inbar):
            continue
        k_ext = inbar[np.argmax(mh[inbar])] if r.side < 0 else inbar[np.argmin(ml[inbar])]
        lo = max(0, k_ext - MSS_REF_BARS + 1)
        ref = ml[lo:k_ext + 1].min() if r.side < 0 else mh[lo:k_ext + 1].max()
        start = inbar[-1] + 1
        for j in range(start, min(start + MSS_WINDOW, len(m15) - 1)):
            if (r.side < 0 and mh[j] > r.extreme) or (r.side > 0 and ml[j] < r.extreme):
                break                                          # extreme re-taken: invalid
            body = abs(mc[j] - mo[j])
            mss = (mc[j] < ref) if r.side < 0 else (mc[j] > ref)
            if mss and np.isfinite(ma[j]) and body >= DISP_BODY_ATR * ma[j]:
                stop = r.extreme + (STOP_BUF_ATR * a[r.i] if r.side < 0
                                    else -STOP_BUF_ATR * a[r.i])
                res = simulate(m15, j + 1, r.side, stop, MAX_HOLD_H * 4)
                if res:
                    # stop distance kept in BOTH units: H1 ATR (comparable to arm A) and M15
                    # ATR at the signal bar, which is what the M15 null/base must re-use.
                    dist = (stop - mo[j + 1]) * -r.side
                    out.append((idx[j], r.i, j, r.side, dist / a[r.i], dist / ma[j], *res))
                    confirmed.add(r.i)
                break
    return (pd.DataFrame(out, columns=["t", "i", "j", "side", "stop_atr", "stop_m15",
                                       "gR", "nR", "gbp"]), confirmed)


def random_base(d: pd.DataFrame, a: np.ndarray, tr: pd.DataFrame, rng: np.random.Generator,
                draws: int = 10, hold: int = MAX_HOLD_H) -> float:
    """Same side, same stop distance in ATR, random other entry bars. Mean NET R."""
    if tr.empty:
        return float("nan")
    n, vals = len(d), []
    ok = np.flatnonzero(np.isfinite(a))
    for r in tr.itertuples():
        for k in rng.choice(ok[ok < n - 2], draws):
            stop = d["open"].iat[k + 1] - r.side * r.stop_atr * a[k]
            res = simulate(d, k + 1, r.side, stop, hold)
            if res:
                vals.append(res[1])
    return float(np.mean(vals)) if vals else float("nan")


def shifted_gross(d: pd.DataFrame, a: np.ndarray, tr: pd.DataFrame, shift: int,
                  hold: int = MAX_HOLD_H) -> float:
    """Null draw: the whole signal set circularly shifted (keeps count, clustering, sides)."""
    n, g = len(d), []
    for r in tr.itertuples():
        k = (r.i + shift) % (n - 2)
        if not np.isfinite(a[k]):
            continue
        stop = d["open"].iat[k + 1] - r.side * r.stop_atr * a[k]
        res = simulate(d, k + 1, r.side, stop, hold)
        if res:
            g.append(res[0])
    return float(np.mean(g)) if g else float("nan")


def summ(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return dict(n=0)
    yrs = pd.to_datetime(tr["t"]).dt.year
    by = tr.groupby(yrs)["nR"].mean()
    half = len(tr) // 2
    span = (pd.to_datetime(tr["t"]).max() - pd.to_datetime(tr["t"]).min()).days / 365.25
    return dict(n=len(tr), per_yr=round(len(tr) / max(span, 0.1), 1),
                gR=round(tr.gR.mean(), 4), nR=round(tr.nR.mean(), 4),
                t=round(tr.nR.mean() / (tr.nR.std(ddof=1) / np.sqrt(len(tr))), 2)
                if len(tr) > 2 else float("nan"),
                hit=round((tr.nR > 0).mean(), 3), gbp=round(tr.gbp.mean(), 2),
                h1=round(tr.nR.iloc[:half].mean(), 4), h2=round(tr.nR.iloc[half:].mean(), 4),
                yrs_pos=f"{int((by > 0).sum())}/{len(by)}")


# ---------------------------------------------------------------------------- main -----
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows, cells = [], []                       # cells feed the max-stat null
    for sym in SYMS:
        m15 = load(sym, "M15")
        h1 = load(sym, "H1")
        if h1 is None and m15 is not None:
            h1 = h1_from_m15(m15)
        if h1 is None:
            continue
        a = atr(h1)
        for lv_name, (up, dn) in (("PD", pd_levels(h1)), ("SW", swing_levels(h1))):
            ev = events(h1, up, dn, a)
            sw = ev[ev.kind == "sweep"]
            A = arm_a(h1, a, sw)
            BRK = arm_a(h1, a, ev[ev.kind == "breakout"])
            min_rng = float(np.median((h1["high"] - h1["low"]).to_numpy()[sw.i])) if len(sw) \
                else np.inf
            PL = arm_a(h1, a, placebo_events(h1, up, dn, a, min_rng))
            base = random_base(h1, a, A, rng)
            key = f"{sym} {lv_name}"
            rows.append(dict(cell=key, arm="A sweep (H1)", **summ(A), base_nR=round(base, 4),
                             excess=round(A.nR.mean() - base, 4) if len(A) else None))
            rows.append(dict(cell=key, arm="  placebo: wick, no level", **summ(PL)))
            rows.append(dict(cell=key, arm="  breakout (acceptance)", **summ(BRK)))
            cells.append((h1, a, A, f"{key} A", MAX_HOLD_H))
            if m15 is not None:
                B, conf = arm_b(h1, a, sw, m15)
                win = sw[h1.index[sw.i] >= m15.index[0]]
                A_win = arm_a(h1, a, win)
                skipped = A_win[~A_win.i.isin(conf)]
                kept = A_win[A_win.i.isin(conf)]
                ma = atr(m15)
                Bn = (B.assign(i=B.j, stop_atr=B.stop_m15) if len(B) else B)  # M15 units
                bb = random_base(m15, ma, Bn, rng, hold=MAX_HOLD_H * 4)
                rows.append(dict(cell=key, arm="B sweep+MSS (M15)", **summ(B),
                                 base_nR=round(bb, 4),
                                 excess=round(B.nR.mean() - bb, 4) if len(B) else None))
                rows.append(dict(cell=key, arm="  A on same window", **summ(A_win)))
                rows.append(dict(cell=key, arm="  A|B conf (LOOKAHEAD)", **summ(kept)))
                rows.append(dict(cell=key, arm="  A|B skipped", **summ(skipped)))
                if len(B):
                    cells.append((m15, ma, Bn, f"{key} B", MAX_HOLD_H * 4))
            print(f"done {key}", flush=True)

    res = pd.DataFrame(rows)
    # ---- max-stat null over the A/B cells (GROSS R, signed) ---------------------------
    obs = {name: tr.gR.mean() for _, _, tr, name, _ in cells if len(tr)}
    best_name = max(obs, key=obs.get) if obs else None
    null_max = []
    for _ in range(N_NULL):
        vals = []
        for d, a, tr, name, hold in cells:
            if len(tr) < 5:
                continue
            sh = int(rng.integers(200, len(d) - 200))
            vals.append(shifted_gross(d, a, tr, sh, hold))
        null_max.append(np.nanmax(vals))
    null_max = np.array(null_max)
    p = float((null_max >= obs[best_name]).mean()) if best_name else float("nan")

    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200)
    txt = res.to_string(index=False)
    tail = (f"\nMAX-STAT NULL over {len(cells)} A/B cells, {N_NULL} circular shifts, GROSS R:\n"
            f"  best cell {best_name}  gR {obs.get(best_name, float('nan')):+.4f}\n"
            f"  null max p50 {np.nanpercentile(null_max, 50):+.4f}  "
            f"p95 {np.nanpercentile(null_max, 95):+.4f}  ->  p = {p:.3f}\n")
    print(txt + tail)
    (OUT / "report.txt").write_text(txt + tail)
    res.to_csv(OUT / "summary.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())
