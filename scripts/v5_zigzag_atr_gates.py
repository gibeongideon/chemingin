"""ARM 1 stage 2 — Z0/Z1/Z3/Z4/Z5/Z6/Z7 for the ZigZag ATR bracket (queue #29).

Consumes the cached OOS probabilities and bracket outcomes written by v5_zigzag_atr_bracket.py.
Gates are evaluated in the pre-registered order: CONTROL #7 (answerability) before CONTROL #5
(oracle), because #7 is the more restrictive and kills cheaper.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from v5_xau_turning_ml import zigzag_swings, atr                      # noqa: E402
from v5_zigzag_atr_bracket import load, COST_USD, MAX_HOLD_H1, ZZ_ORDER, ZZ_THETA  # noqa: E402

OUT = ROOT / "data/v5_runs/zigzag_atr"
GAP = MAX_HOLD_H1
rng = np.random.default_rng(11)


def net(entry, px):
    return (px - entry - COST_USD) / entry


def nonoverlap(idx, gap=GAP):
    last, keep = -10**9, []
    for i in idx:
        if i - last >= gap:
            keep.append(i); last = i
    return np.array(keep, int)


def trade_sim(sel, r, close, index):
    """No pyramiding — one position at a time, exactly as v5_zigzag_ftmo.py:192 enforces."""
    daily, last_exit = {}, -1
    taken = []
    for i in sel:
        if i <= last_exit or not np.isfinite(r["exit_px"][i]):
            continue
        ret = net(close[i], r["exit_px"][i])
        bars = max(int(r["bars"][i]), 1)
        last_exit = i + int(np.ceil(bars / 4))
        taken.append(i)
        d = index[min(last_exit, len(index) - 1)].normalize()
        daily[d] = daily.get(d, 0.0) + ret
    s = pd.Series(daily).sort_index()
    return s, np.array(taken, int)


def perf(s, bdays):
    if len(s) < 10:
        return dict(sharpe=np.nan, cagr=np.nan, dd=np.nan, n=len(s))
    full = s.reindex(bdays, fill_value=0.0)
    mu, sd = full.mean(), full.std()
    eq = (1 + full).cumprod()
    yrs = len(full) / 252
    return dict(sharpe=float(mu / sd * np.sqrt(252)) if sd > 0 else np.nan,
                cagr=float(eq.iloc[-1] ** (1 / yrs) - 1),
                dd=float((eq / eq.cummax() - 1).min()), n=int(len(s)))


def main() -> int:
    st = np.load(OUT / "_outcomes.npy", allow_pickle=True).item()
    h1, _ = load("H1"), None
    close = h1["close"].values
    A = atr(h1)
    _, buys = zigzag_swings(h1, ZZ_ORDER, ZZ_THETA * A)
    p = pd.read_csv(OUT / "oos_prob_h1.csv", parse_dates=["time"], index_col="time")["p"]
    p = p.reindex(h1.index).values
    scored = np.isfinite(p)
    bdays = pd.bdate_range(h1.index[np.argmax(scored)].normalize(), h1.index[-1].normalize())
    piv = np.zeros(len(h1), bool); piv[buys] = True

    print("=" * 94)
    print("ARM 1 stage 2 -- gates Z0, Z2, Z1, Z3, Z5(specificity), Z4/Z6/Z7")
    print("=" * 94)

    # ---------------------------------------------------------------- Z0 answerability
    thr = 0.60
    fires = np.flatnonzero(scored & (p >= thr))
    no = nonoverlap(fires)
    print(f"\n[Z0] ANSWERABILITY (CONTROL #7) at threshold {thr}")
    print(f"     fires {len(fires):,}   non-overlapping (>={GAP}h apart) {len(no):,}   "
          f"bar is 400   ->  {'PASS' if len(no) >= 400 else 'FAIL'}")
    print(f"     fire rate {100*len(fires)/scored.sum():.2f}% of scored bars   "
          f"~{len(fires)/(scored.sum()/24/30.4):.1f} fires/month")

    # mean offset -- CONTROL #8's confirmation-detector test, at H1
    nxt = np.searchsorted(buys, fires)
    offs = []
    for f, k in zip(fires, nxt):
        cand = [buys[j] for j in (k - 1, k) if 0 <= j < len(buys)]
        if cand:
            offs.append(f - min(cand, key=lambda b: abs(f - b)))
    mean_off = float(np.mean(offs))
    near = float(np.mean(np.abs(offs) <= 3))
    print(f"     fire-weighted mean offset {mean_off:+.3f} bars  "
          f"({100*near:.1f}% within +/-3 of a pivot)  "
          f"-> {'CONFIRMATION detector' if mean_off > 0.3 else 'anticipatory'}")

    # ---------------------------------------------------------------- Z3 + Z1 per cell
    rows = []
    for name, r in st.items():
        ok = scored & np.isfinite(r["outcome"]) & np.isfinite(r["exit_px"])
        base_ev = float(np.mean(net(close[ok], r["exit_px"][ok])))
        base_hit = float(np.mean(r["outcome"][ok & (r["outcome"] != 0)] > 0))

        for arm, sel in (("DETECTOR", fires), ("ORACLE", buys)):
            s = sel[ok[sel]]
            if len(s) < 50:
                continue
            ev = float(np.mean(net(close[s], r["exit_px"][s])))
            hold = float(np.mean(net(close[s], r["hold_px"][s])))
            dec = s[r["outcome"][s] != 0]
            hit = float(np.mean(r["outcome"][dec] > 0)) if len(dec) else np.nan
            daily, taken = trade_sim(s, r, close, h1.index)
            m = perf(daily, bdays)
            rows.append(dict(cell=name, arm=arm, n=len(s), trades=len(taken),
                             hit=round(hit, 4), base_hit=round(base_hit, 4),
                             skill=round(hit - base_hit, 4),
                             ev_fire=round(1e4 * ev, 2), ev_anybar=round(1e4 * base_ev, 2),
                             ev_minus_anybar=round(1e4 * (ev - base_ev), 2),
                             ev_minus_hold=round(1e4 * (ev - hold), 2),
                             sharpe=round(m["sharpe"], 3), cagr=round(100 * m["cagr"], 2),
                             dd=round(100 * m["dd"], 1)))
    tb = pd.DataFrame(rows)
    tb.to_csv(OUT / "z1_z3_cells.csv", index=False)

    det = tb[tb.arm == "DETECTOR"].sort_values("ev_minus_anybar", ascending=False)
    orc = tb[tb.arm == "ORACLE"].sort_values("ev_minus_anybar", ascending=False)
    print("\n[Z1] ORACLE CEILING (CONTROL #5) -- fire at the TRUE pivot, perfect hindsight")
    print("     ev_* are in BASIS POINTS per fire, net of the $0.448 round trip")
    print(orc.drop(columns=["arm"]).to_string(index=False))
    print("\n[Z3] THE DETECTOR -- EV/fire vs what ANY bar earns with the same bracket")
    print(det.drop(columns=["arm"]).to_string(index=False))

    # ---------------------------------------------------------------- Z5 specificity sweep
    print("\n[Z5] SPECIFICITY SWEEP (CONTROL #7: gate on FP rate, never on recall)")
    best = det.iloc[0]["cell"]
    r = st[best]
    ok = scored & np.isfinite(r["exit_px"])
    base_ev = float(np.mean(net(close[ok], r["exit_px"][ok])))
    sw = []
    for t in (0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        s = np.flatnonzero(ok & (p >= t))
        if len(s) < 50:
            continue
        tp_fl = piv[np.clip(s[:, None] + np.arange(-3, 4), 0, len(piv) - 1)].any(1)
        ev = float(np.mean(net(close[s], r["exit_px"][s])))
        sw.append(dict(thr=t, fires=len(s), nonoverlap=len(nonoverlap(s)),
                       fp_rate=round(1 - float(tp_fl.mean()), 3),
                       ev_fire=round(1e4 * ev, 2),
                       ev_minus_anybar=round(1e4 * (ev - base_ev), 2)))
    sws = pd.DataFrame(sw)
    sws.to_csv(OUT / "z5_specificity.csv", index=False)
    print(f"     cell = {best}")
    print(sws.to_string(index=False))

    # ---------------------------------------------------------------- Z4 / Z6 walk-forward + halves
    print("\n[Z4/Z6] TRADE SIMULATION vs BUY-AND-HOLD (CONTROL #2 -- never vs zero)")
    bh = h1["close"].resample("1D").last().dropna().pct_change().dropna()
    bh = bh[bh.index >= h1.index[np.argmax(scored)]]
    bh_m = perf(bh, bdays)
    print(f"     buy-and-hold gold over the same window: Sharpe {bh_m['sharpe']:+.3f}  "
          f"CAGR {100*bh_m['cagr']:+.2f}%  maxDD {100*bh_m['dd']:.1f}%")
    hl = []
    for name, r in st.items():
        s = fires[np.isfinite(r["exit_px"][fires])]
        daily, _ = trade_sim(s, r, close, h1.index)
        if len(daily) < 30:
            continue
        full = daily.reindex(bdays, fill_value=0.0)
        pair = full - bh.reindex(bdays, fill_value=0.0)
        t_stat = float(pair.mean() / pair.std() * np.sqrt(len(pair))) if pair.std() > 0 else np.nan
        yrs = full.groupby(full.index.year).sum()
        byrs = bh.reindex(bdays, fill_value=0.0).groupby(bdays.year).sum()
        e1 = full[full.index.year <= 2021]; e2 = full[full.index.year >= 2022]
        hl.append(dict(cell=name, sharpe=round(perf(daily, bdays)["sharpe"], 3),
                       vs_bh_t=round(t_stat, 2),
                       yrs_beat=f"{int((yrs > byrs).sum())}/{len(yrs)}",
                       half_2018_21=round(1e4 * e1.mean(), 2),
                       half_2022_26=round(1e4 * e2.mean(), 2)))
    hs = pd.DataFrame(hl).sort_values("sharpe", ascending=False)
    hs.to_csv(OUT / "z4_z6_walkforward.csv", index=False)
    print(hs.to_string(index=False))

    json.dump(dict(threshold=thr, fires=int(len(fires)), nonoverlap=int(len(no)),
                   mean_offset=mean_off, within_tol=near,
                   best_cell=best, bh=bh_m), open(OUT / "gates.json", "w"), indent=2, default=str)
    print(f"\n  wrote {OUT}/z1_z3_cells.csv, z5_specificity.csv, z4_z6_walkforward.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
