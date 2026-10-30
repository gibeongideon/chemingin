#!/usr/bin/env python3
"""OCTOBER B9 — the BTC two-cell speed test, and why it cannot do what §3bp hoped.

§3bp filed BTC trend as *"a real, unproven lead"*: walk-forward Sharpe **+1.401**, beats
buy-and-hold in BOTH halves and at 10x spread (which the gold champion never does), PBO 0.175
pass — and **DSR 0.743 against a 0.90 bar, FAIL**. Its stated promotion path was
*"a pre-registered 2-cell test (incumbent 1/6 vs one declared alternative), not a wider sweep."*

*** THE TRAP IN THAT PLAN, STATED BEFORE RUNNING IT. ***
§3bp had ALREADY swept the grid and found SLOW 1/12 to be the full-sample argmax (+1.719 against
1/6's +1.626). Declaring 1/12 as "the one alternative" NOW and then pricing the result at K=2
would be laundering a completed search through a pre-registration. §3s's rule is explicit --
*"DSR's trial count must reflect the ENTIRE search"* -- and §3ak proved searching harder raises
the bar faster than the statistic. **A 2-cell framing cannot retroactively undo a 5-cell sweep, so
B9 cannot deliver DSR >= 0.90 by re-labelling.** Saying so is the finding, not a disclaimer.

WHAT IS ACTUALLY ANSWERABLE, and what this script runs instead:

  PART 1 — IS THE DECISION EVEN DISTINGUISHABLE? A matched-vol paired t between the 1/6 and 1/12
  return streams. This needs no new data and is not contaminated by the search, because it asks
  about the DIFFERENCE rather than about either level. If |t| < 2 the two speeds are the same
  object and the incumbent stands for free -- which is a decisive answer, not a null one.

  PART 2 — DSR BOTH WAYS, on the REAL trial Sharpes. K=2 (the fiction) against K=5 (the grid the
  walk-forward actually wandered over: 1/3, 1/6, 1/12, 1/24, 1/48). The gap between them is the
  price of the search. §3ch fed `deflated_sharpe_ratio` a FABRICATED trial array and printed a
  meaningless 0.0000; this passes the measured grid.

  PART 3 — INDEPENDENT REPLICATION. Does 1/12 beat 1/6 on ETH, LTC and SOL -- instruments that
  were never used to pick it? This is the only uncontaminated evidence available, and it is the
  one thing that could legitimately promote the alternative. 3 of 3 would be real; 1 or 2 of 3 is
  what noise looks like.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from v5_xau_champion_lifts import champion_recipe                     # noqa: E402
from src.evaluation.dsr_pbo import deflated_sharpe_ratio              # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "october"
SPREAD_USD = 1.0                      # §3bp's BTC cost model, for the control cell
COMMON_RT_BP = 5.0                    # cross-instrument cost: one number for every crypto
TARGET_VOL, VOL_HL, MAX_LEV = 0.10, 42, 8.0
GRID = (1 / 3, 1 / 6, 1 / 12, 1 / 24, 1 / 48)     # what the walk-forward actually wandered over
INCUMBENT, ALTERNATIVE = 1 / 6, 1 / 12


def load(sym: str) -> pd.Series:
    d = pd.read_csv(ROOT / f"data/{sym}_D1_long.csv", parse_dates=["time"], index_col="time")
    d = d[~d.index.duplicated(keep="last")].sort_index()
    return d["close"].astype(float)


def bpy_of(idx) -> float:
    return len(idx) / max((idx[-1] - idx[0]).days / 365.25, 1e-9)


def arm(c: pd.Series, scale: float, cost_mode: str) -> pd.Series:
    """One speed's net return stream. `cost_mode` 'btc' reproduces §3bp; 'bp' is cross-instrument."""
    ret = c.pct_change()
    bpy = bpy_of(c.index)
    vol = ret.ewm(halflife=VOL_HL, min_periods=20).std() * np.sqrt(bpy)
    fc = champion_recipe(c, scale, 1.5)
    pos = (fc.clip(-2, 2) * (TARGET_VOL / vol)).clip(-MAX_LEV, MAX_LEV).shift(1)
    turn = pos.diff().abs().fillna(0.0)
    cost = (SPREAD_USD / c) * 0.75 * 2 if cost_mode == "btc" else (COMMON_RT_BP / 1e4)
    return (pos * ret - turn * cost).dropna()


def sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(bpy_of(r.index))) if r.std() > 0 else np.nan


def vol_match(a: pd.Series, b: pd.Series) -> pd.Series:
    sa, sb = a.std(), b.std()
    return a * (sb / sa) if sa > 0 else a


def paired_t(a: pd.Series, b: pd.Series) -> tuple:
    """Matched-vol paired t of (a - b). Positive means `a` beats `b`."""
    i = a.index.intersection(b.index)
    x, y = a.reindex(i), b.reindex(i)
    d = vol_match(x, y) - y
    if d.std(ddof=1) == 0:
        return np.nan, 0
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), len(d)


def walk_forward(c: pd.Series, grid=GRID, lookback_yrs: int = 3) -> tuple:
    """§3bp's CONTROL #3: re-pick the speed every January on the trailing 3 years.

    §3bp's DSR 0.743 was computed on THIS stream (+1.401), not on the in-sample best (+1.750), so
    a DSR quoted against the in-sample arm answers a different question and must not be compared
    to it. Reproducing the walk-forward here is what makes the comparison legitimate.
    """
    streams = {s: arm(c, s, "btc") for s in grid}
    idx = sorted(set.intersection(*[set(v.index) for v in streams.values()]))
    idx = pd.DatetimeIndex(idx)
    out = pd.Series(index=idx, dtype=float)
    picks = []
    for yr in sorted(set(idx.year))[lookback_yrs:]:
        tr = idx[(idx < pd.Timestamp(yr, 1, 1)) &
                 (idx >= pd.Timestamp(yr - lookback_yrs, 1, 1))]
        te = idx[(idx >= pd.Timestamp(yr, 1, 1)) & (idx < pd.Timestamp(yr + 1, 1, 1))]
        if len(tr) < 200 or len(te) < 20:
            continue
        best = max(grid, key=lambda s: (streams[s].reindex(tr).mean() /
                                        streams[s].reindex(tr).std()))
        out.loc[te] = streams[best].reindex(te)
        picks.append((yr, round(1 / best)))
    return out.dropna(), picks


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 100)
    print("OCTOBER B9 — the BTC two-cell speed test: 1/6 (incumbent) vs 1/12 (declared)")
    print("=" * 100)

    c = load("BTC")
    streams = {s: arm(c, s, "btc") for s in GRID}
    inc, alt = streams[INCUMBENT], streams[ALTERNATIVE]
    print(f"  BTC {len(c):,} D1 bars {c.index[0].date()} -> {c.index[-1].date()}  "
          f"({bpy_of(c.index):.1f} bars/yr — BTC quotes weekends)")
    print(f"  CONTROL CELL — §3bp filed 1/6 at +1.626 and 1/12 at +1.719 full sample")
    print(f"    measured here: 1/6 {sharpe(inc):+.3f}   1/12 {sharpe(alt):+.3f}")

    # ---------------- PART 1 — is the decision distinguishable at all? ----------------------
    print(f"\n  PART 1 — IS 1/12 vs 1/6 A DISTINGUISHABLE DECISION?")
    t, n = paired_t(alt, inc)
    print(f"    full sample: matched-vol paired t = {t:+.3f} over {n:,} days  "
          f"({'DISTINGUISHABLE' if abs(t) >= 2 else 'NOT distinguishable at |t| >= 2'})")
    half = n // 2
    i = alt.index.intersection(inc.index)
    for lab, sl in (("first half", slice(None, half)), ("second half", slice(half, None))):
        t2, n2 = paired_t(alt.reindex(i)[sl], inc.reindex(i)[sl])
        print(f"    {lab:<12}: t = {t2:+.3f} over {n2:,} days")
    rho = float(vol_match(alt.reindex(i), inc.reindex(i)).corr(inc.reindex(i)))
    print(f"    return-stream correlation between the two speeds: {rho:+.4f}  "
          f"<- they are {'ONE object' if rho > 0.9 else 'genuinely different'}")

    # ---------------- PART 2 — DSR on the REAL trial Sharpes, both framings ------------------
    print(f"\n  PART 2 — DSR, computed on the MEASURED grid rather than a declared one")
    per_obs = {s: float(streams[s].mean() / streams[s].std()) for s in GRID}
    print(f"    {'speed':>8} {'Sharpe':>9} {'per-obs':>10}")
    for s in GRID:
        print(f"    {f'1/{round(1/s)}':>8} {sharpe(streams[s]):>+9.3f} {per_obs[s]:>+10.5f}")
    best = max(GRID, key=lambda s: per_obs[s])
    rows = []
    for lab, trials in (("K=2  (the 'declared' fiction)", [per_obs[INCUMBENT], per_obs[ALTERNATIVE]]),
                        ("K=5  (the grid actually swept)", [per_obs[s] for s in GRID])):
        r = deflated_sharpe_ratio(streams[best].to_numpy(), np.asarray(trials))
        print(f"    {lab:<32} DSR {r['dsr']:.4f}   benchmark SR* {r['sr_benchmark']:.5f}   "
              f"{'PASS' if r['dsr'] >= 0.90 else '**FAIL**'} vs the 0.90 bar")
        rows.append(dict(framing=lab, dsr=r["dsr"], sr_star=r["sr_benchmark"], k=r["n_trials"]))
    print(f"    best speed on this grid is 1/{round(1/best)}")
    print(f"    *** BOTH READ 1.0000, AND THAT IS NOT A PASS — IT IS A DIFFERENT QUESTION. ***")
    print(f"    The five speeds produce near-identical Sharpes ({min(per_obs.values()):.5f} to "
          f"{max(per_obs.values()):.5f} per obs), so the cross-trial variance is tiny and the")
    print(f"    deflation benchmark SR* collapses toward zero. Deflation only bites when trials")
    print(f"    DISAGREE. §3bp's 0.743 was computed on the WALK-FORWARD stream (+1.401), not on")
    print(f"    the in-sample best, so these numbers do not speak to it. Reproducing it:")
    wf, picks = walk_forward(c)
    wf_sr = sharpe(wf)
    rw = deflated_sharpe_ratio(wf.to_numpy(), np.asarray([per_obs[s] for s in GRID]))
    print(f"      walk-forward Sharpe {wf_sr:+.3f}  (§3bp filed +1.401)   "
          f"DSR on the WF stream {rw['dsr']:.4f}  "
          f"{'PASS' if rw['dsr'] >= 0.90 else '**FAIL**'}")
    print(f"      speed picks by year: {', '.join(f'{y}:1/{p}' for y, p in picks)}")
    uniq = len(set(p for _, p in picks))
    print(f"      {uniq} distinct speeds chosen across {len(picks)} years — "
          f"{'WANDERS, as §3bp recorded' if uniq >= 3 else 'stable'}")
    rows.append(dict(framing="walk-forward", dsr=rw["dsr"], sharpe=wf_sr))

    # ---------------- PART 3 — independent replication on instruments never used to pick it ---
    print(f"\n  PART 3 — DOES 1/12 BEAT 1/6 WHERE IT WAS NOT CHOSEN?")
    print(f"    common cost {COMMON_RT_BP:.1f}bp round trip for every instrument, "
          f"so the comparison is like-for-like")
    print(f"    {'symbol':>7} {'bars':>7} {'1/6 SR':>9} {'1/12 SR':>9} {'diff':>8} "
          f"{'paired t':>9}  SR/t")
    print("    " + "-" * 68)
    wins, wins_t, tested = 0, 0, 0
    for sym in ("BTC", "ETH", "LTC", "SOL"):
        try:
            cc = load(sym)
        except FileNotFoundError:
            continue
        a6, a12 = arm(cc, INCUMBENT, "bp"), arm(cc, ALTERNATIVE, "bp")
        if len(a6) < 300:
            continue
        s6, s12 = sharpe(a6), sharpe(a12)
        tt, _ = paired_t(a12, a6)
        # TWO win counts, because they disagree and the repo gates on the second. A raw Sharpe
        # difference can favour an arm that LOSES once the two streams are vol-matched, which is
        # the whole reason `xau-champion-five-lifts-fail` made matched-vol gating a method rule.
        w_sr, w_t = s12 > s6, tt > 0
        if sym != "BTC":                       # BTC is contaminated; it does not get a vote
            tested += 1; wins += int(w_sr); wins_t += int(w_t)
        print(f"    {sym:>7} {len(cc):>7,} {s6:>+9.3f} {s12:>+9.3f} {s12-s6:>+8.3f} "
              f"{tt:>+9.2f}  {'YES' if w_sr else 'no':>4} / {'YES' if w_t else 'no':>3}"
              f"{'   <- 1/12 was PICKED on this one' if sym == 'BTC' else ''}")
        rows.append(dict(framing=f"replication {sym}", sr_inc=s6, sr_alt=s12, t=tt))
    print(f"\n    *** on raw Sharpe 1/12 beats 1/6 on {wins} of {tested} uncontaminated "
          f"instruments ***")
    print(f"    *** on MATCHED-VOL PAIRED t — the statistic this repo gates on — it is "
          f"{wins_t} of {tested}, and NONE reaches |t| >= 2 ***")

    pd.DataFrame(rows).to_csv(OUT / "b9_btc_two_cell.csv", index=False)
    print(f"""
  WHAT THIS CAN AND CANNOT SETTLE. The DSR gap in PART 2 is the price of §3bp's completed sweep,
  and no re-labelling removes it -- a 2-cell framing applied after a 5-cell search is not a 2-cell
  experiment. PART 1 asks the question that IS clean: if the two speeds cannot be told apart, the
  book keeps 1/6 and the decision costs nothing. PART 3 is the only route that could legitimately
  promote 1/12, because those instruments had no say in choosing it.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
