"""LOT-INCREASE SCHEMES ON THE LIVE BOOK — and the probability of ruin.

Asked for: add lot-increase ("arbitrage") sizing and measure the chance of ruin.

WHAT IS NEW HERE. Lot progressions have been tested three times in this repo, but always on a
DETECTOR's bet stream (v5_martingale_predictor.py, v5_xau_fade_martingale.py,
v5_xau_detector_antimartingale.py). None was run on the CHAMPION -- the thing actually deployed
on Maven magic 360571 -- and none reported a ruin surface against the real prop-firm walls. That
is what this does.

THE GOVERNING LAW, PROVEN THREE TIMES AND NOT RE-LITIGATED HERE:
    "a martingale is an edge LEVER, not an edge CREATOR. On a positive-edge base it multiplied
     returns ~25x with bounded DD and 0% bust; on a negative-edge base it busted 100% of paths."
So the question is never "does escalation make money" -- it is "what does escalation cost in
RUIN at the return it buys". Every comparison below is therefore made at MATCHED MEDIAN RETURN:
the flat stake is scaled until it earns what the progression earns, and only then are the ruin
probabilities compared. Without that, a progression simply looks better because it is bigger.

STAGE 1 IS THE PRECONDITION, AND IT DECIDES EVERYTHING ELSE. A progression can only beat flat
sizing if outcomes CLUSTER -- martingale needs losses to cluster, pyramid needs wins to cluster.
If the sign of consecutive trade returns is serially independent, every progression is pure
leverage and the ruin comparison is the whole story. §3 measured win/loss lag-1 autocorrelation
on the detector at +0.05 to -0.03; it has never been measured on the champion.

Walls: FTMO's 8% halt ($92,000 on $100,000) and a 10% wall ($90,000), both STATIC from the
initial balance -- the prop-firm convention. A trailing/high-water wall is a different and much
harsher object (memory: static-vs-trailing is a real trap, 1.9% -> 45.9% breach at risk_frac 2%).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import src.v5.xau_trend as xt                                      # noqa: E402
from src.v5.xau_dual_signals import SIGNALS                        # noqa: E402
from src.v5 import mtf_frames                                      # noqa: E402
from v5_xau_fade_martingale import simulate                        # noqa: E402

OUT = ROOT / "data/v5_runs/lot_progression"
EQUITY0 = 100_000.0
WALLS = {"FTMO 8%": 92_000.0, "10%": 90_000.0}
LIVE = dict(sl_atr=3.0, trail_atr=3.0, trail_activation_atr=1.0)


def champion_trades(feed: str = "original") -> np.ndarray:
    """R-multiples of the LIVE champion. pnl = stake x r_multiple, so R maps exactly onto
    simulate()'s `pnl = stake * ri` with w = 1.0 (a full stop loses one unit of stake)."""
    f = "XAUUSD_M15_ftmo.csv" if feed == "ftmo" else "XAUUSD_M15_long.csv"
    m15 = pd.read_csv(ROOT / f"data/{f}", parse_dates=["time"], index_col="time").sort_index()
    m15 = m15[~m15.index.duplicated(keep="last")]
    h4 = mtf_frames.aggregate(m15, "H4")[["open", "high", "low", "close", "spread"]].dropna()
    if feed == "original":
        h4 = h4[h4.index >= "2018-01-01"]
    xt.xau_signal = lambda c: SIGNALS["champ"](c)      # exactly as v5_maven_champ.py:99 does
    res = xt.run_trades(h4, equity0=EQUITY0, exit_mode="trail",
                        flip_mode="confidence", params=LIVE)
    tr = res["trades"]
    r = tr["r_multiple"].astype(float).dropna().values
    print(f"  [{feed}] {len(r)} champion trades, {h4.index[0].date()}..{h4.index[-1].date()}")
    return r


def precondition(r: np.ndarray, seed=7) -> dict:
    """Do outcomes CLUSTER? If not, no progression can beat flat except by being bigger."""
    s = (r > 0).astype(int)
    ac1 = float(pd.Series(s).autocorr(1))
    rng = np.random.default_rng(seed)
    null = np.array([pd.Series(rng.permutation(s)).autocorr(1) for _ in range(2000)])
    z = (ac1 - null.mean()) / null.std()

    def streaks(x, want_loss=True):
        runs, c = [], 0
        for v in x:
            hit = (v == 0) if want_loss else (v == 1)
            if hit:
                c += 1
            elif c:
                runs.append(c); c = 0
        if c:
            runs.append(c)
        return np.array(runs) if runs else np.array([0])

    lo, wi = streaks(s, True), streaks(s, False)
    p = float(s.mean())
    print("\n" + "=" * 92)
    print("STAGE 1 — THE PRECONDITION: do wins or losses CLUSTER?")
    print("=" * 92)
    print(f"  trades {len(r)}   win rate {100*p:.1f}%   mean R {r.mean():+.3f}   "
          f"sum R {r.sum():+.1f}")
    print(f"  lag-1 autocorrelation of the WIN/LOSS SIGN: {ac1:+.4f}")
    print(f"    shuffled null: mean {null.mean():+.4f}  sd {null.std():.4f}  -> z = {z:+.2f}")
    print(f"  longest loss streak {lo.max()}  (iid expectation "
          f"{np.log(len(r))/np.log(1/max(1-p,1e-9)):.1f})"
          f"   longest win streak {wi.max()}")
    verdict = abs(z) < 2.0
    print(f"  -> {'NO CLUSTERING (|z| < 2). Any lot progression is PURE LEVERAGE.' if verdict else 'CLUSTERING DETECTED — a progression may have a basis.'}")
    return dict(p=p, ac1=ac1, z=float(z), max_loss_streak=int(lo.max()),
                max_win_streak=int(wi.max()), no_clustering=bool(verdict))


def ruin_frontier(r, engines, Ks, bases, floor, n_mc=2500, block=10, seed=7):
    """Sweep the base stake and report the (return, ruin) FRONTIER for each engine.

    NOT a bisection to a matched return. Median return is NOT monotone in stake: it rises,
    peaks, then COLLAPSES once ruin starts truncating paths at the wall. A bisection assuming
    monotonicity walks straight to its upper bound -- the first version of this script did
    exactly that, pinning every cell at a 20% stake with an identical -8.08% return, and the
    tell was that `flat` (whose answer was already known) came back wrong too.

    Reading the frontier at equal return is the honest comparison, and it is robust to the hump.
    """
    rng = np.random.default_rng(seed)
    N = len(r)
    seqs = []
    for _ in range(n_mc):
        idx = []
        while len(idx) < N:
            s0 = rng.integers(0, N)
            idx.extend(range(s0, s0 + block))
        seqs.append(r[np.array(idx[:N]) % N])

    rows = []
    for eng in engines:
        for K in (Ks if eng != "flat" else [0]):
            for bpct in bases:
                base = EQUITY0 * bpct / 100.0
                ends, dds, busts = [], [], 0
                for seq in seqs:
                    c, _s, bust = simulate(seq, eng, base, EQUITY0, floor, K, 50.0, 1.0)
                    ends.append(c[-1] / EQUITY0 - 1.0)
                    e = pd.Series(c); dds.append(float((e / e.cummax() - 1).min() * 100))
                    busts += bust >= 0
                rows.append(dict(engine=eng, K=K, base_pct=bpct,
                                 p_ruin=round(busts / n_mc, 4),
                                 med_ret_pct=round(float(np.median(ends) * 100), 2),
                                 med_maxdd_pct=round(float(np.median(dds)), 2)))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", choices=("original", "ftmo"), default="original")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 92)
    print("LOT-INCREASE SCHEMES ON THE LIVE CHAMPION — ruin measured, not asserted")
    print("=" * 92)
    r = champion_trades(a.feed)
    pre = precondition(r)

    BASES = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
    for wall_name, floor in WALLS.items():
        print("\n" + "=" * 100)
        print(f"STAGE 2 — RUIN FRONTIER at the {wall_name} wall "
              f"(${floor:,.0f}, STATIC from $100,000)")
        print("=" * 100)
        tb = ruin_frontier(r, ["flat", "double4", "pyramid", "recover4"], [3, 4, 6],
                           BASES, floor)
        tb["wall"] = wall_name
        piv_r = tb.pivot_table(index=["engine", "K"], columns="base_pct", values="med_ret_pct")
        piv_p = tb.pivot_table(index=["engine", "K"], columns="base_pct", values="p_ruin") * 100
        print("\n  MEDIAN RETURN % by base stake (% of equity per trade):")
        print(piv_r.to_string())
        print("\n  P(RUIN) % by base stake:")
        print(piv_p.round(2).to_string())

        # read the frontier at equal return: for each engine, interpolate the stake that
        # reaches the flat engine's best-performing return, then compare ruin there
        flat = tb[tb.engine == "flat"].sort_values("base_pct")
        tgt = float(flat.med_ret_pct.max())
        print(f"\n  COMPARISON AT MATCHED RETURN (target = flat's best median, {tgt:+.1f}%):")
        print(f"    {'engine':10s} {'K':>2s} {'stake%':>7s} {'ruin%':>7s} {'medDD%':>8s}")
        for (eng, K), g in tb.groupby(["engine", "K"]):
            g = g.sort_values("base_pct")
            if g.med_ret_pct.max() < tgt:
                print(f"    {eng:10s} {K:2d} {'—':>7s} {'—':>7s} {'—':>8s}   "
                      f"cannot reach {tgt:+.1f}% at any stake tested")
                continue
            row = g[g.med_ret_pct >= tgt].iloc[0]
            print(f"    {eng:10s} {K:2d} {row.base_pct:7.2f} {100*row.p_ruin:7.2f} "
                  f"{row.med_maxdd_pct:8.2f}")
        tb.to_csv(OUT / f"ruin_{a.feed}_{wall_name.split()[0].lower()}.csv", index=False)

    print(f"\n  wrote {OUT}/ruin_{a.feed}_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
