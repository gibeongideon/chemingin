"""FUNDINGPIPS 10K — find a replacement for the (now close-only) XAU sleeve.

FundingPips flipped XAUUSDmicro to trade_mode=CLOSEONLY between 2026-07-20 and
07-23, so the gold sleeve of the Flex 10K book cannot reopen. The obvious swap to
plain XAUUSD does NOT work: contract 100oz -> min-lot $4,048 against a ~$380
target, ~10.6x oversize on a $10,000 account.

So this asks a different question from v5_instrument_search.py (which hunts a 4th
sleeve for the 100K FTMO book): what can REPLACE the gold factor in the 10K book?

At $10,000 the binding constraint is not Sharpe, it is MIN-LOT GRANULARITY — most
of the FundingPips universe rounds to zero or to a wild oversize. So sizing is a
first-class gate here, checked BEFORE any performance claim:

  0. min-lot notional must fit the sleeve's own vol-targeted target notional
  1. positive standalone champion-recipe Sharpe (carry its own weight)
  2. low correlation to the surviving ETH + DJI sleeves (the whole point)
  3. the rebuilt book must beat the 2-sleeve ETH+DJI reality on pass% AND median,
     and hold up in BOTH half-samples (the check that killed H4/med)
"""
from __future__ import annotations
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = "/home/rock/Desktop/2026_Projects/Trader36/MT5"
sys.path.insert(0, ROOT + "/scripts")
import v5_basket_challenge as vbc  # noqa

M = vbc.MODELS["flex"]
DIAL = M["vol"] / vbc.TARGET_VOL
EQUITY = 10_000.0

# What the book is TODAY (gold sleeve dead) and what it was MEANT to be.
LIVE_2 = {"crypto": ["ETH"], "eq_us": ["DJI"]}
ORIG_3 = {"xau": ["XAUCHAMP"], "crypto": ["ETH"], "eq_us": ["DJI"]}

# FundingPips tradeable universe -> local D1 file, with min-lot notional VERIFIED
# LIVE on account 20414926 @ FundingPips-SIM1 (2026-07-24, 43 symbols total).
# Excluded: JPY crosses (min-lot $95k-218k, absurd at 10K), INRUSD (DISABLED),
# XAUUSDmicro (CLOSEONLY - the thing we are replacing).
FP_UNIVERSE = {
    # engine sym : (FundingPips symbol, min-lot notional $, spread bp)
    "ETH":    ("ETHUSD",   18.95,  2.69),
    "NZDUSD": ("NZDUSD",  578.78,  0.17),
    "STOXX":  ("STX50",   624.15,  2.40),
    "BTC":    ("BTCUSD",  654.65,  3.06),
    "AUDUSD": ("AUDUSD",  698.44,  0.14),
    "USDCHF": ("USDCHF",  816.20,  0.12),
    "EURGBP": ("EURGBP",  854.20,  0.59),
    "WTI":    ("USOIL",   911.83,  5.48),
    "EURCHF": ("EURCHF",  929.75,  0.54),
    "BRENT":  ("UKOIL",   986.83,  5.07),
    "FTSE":   ("FTSE100", 1066.70, 0.66),
    "GBPCHF": ("GBPCHF",  1088.41, 0.64),
    "EURUSD": ("EURUSD",  1139.14, 0.09),
    "AUDNZD": ("AUDNZD",  1206.69, 0.75),
    "GBPUSD": ("GBPUSD",  1333.54, 0.07),
    "USDCAD": ("USDCAD",  1407.47, 0.28),
    "EURAUD": ("EURAUD",  1630.96, 0.49),
    "GBPAUD": ("GBPAUD",  1909.26, 0.47),
    "DJI":    ("DJI30",   2594.55, 0.19),
    "SILVER": ("XAGUSD",  2907.50, 17.20),
    "SPX":    ("SPX500",  3710.51, 1.08),
    "GOLD":   ("XAUUSD",  4048.26, 0.44),
    "NDX":    ("NDX100",  5692.91, 0.56),
    "DAX":    ("GER40",   6225.28, 0.80),
    "NIKKEI": ("JP225",   6488.45, 1.54),
}

# The live book runs 0.89-1.18x of target notional (config _sizing_note), so a
# candidate whose MIN lot already exceeds ~1.5x its target is structurally
# oversized: it cannot be sized down, only switched off.
FIT_MAX = 1.5


def book_stats(classes):
    vbc.CLASSES = classes
    W, book, live = vbc.build(dial=DIAL)
    rv = book.ewm(halflife=vbc.VT_HALFLIFE, min_periods=20).std() * np.sqrt(252)
    vs = (M["vol"] / rv).clip(0.0, vbc.VT_MAXSCALE)
    eqb = (1 + book).cumprod()
    ds = (1 + (eqb / eqb.cummax() - 1) * 3.0).clip(lower=vbc.DD_FLOOR)
    vt = (book * (vs * ds).shift(1)).dropna().loc["2017-01-01":]
    sr = float(vt.mean() / vt.std() * np.sqrt(252))
    eq = (1 + vt).cumprod()
    dd = float((eq / eq.cummax() - 1).min() * 100)
    r10 = (vt * (0.10 / (vt.std() * np.sqrt(252)))).values
    fp = vbc.fp_sim(r10, DIAL, day_safety=1.5, p1=M["p1"], p2=M["p2"],
                    dayloss=M["daily"], maxloss=M["maxloss"])
    h1, h2 = vt.loc[:"2020-12-31"], vt.loc["2021-01-01":]
    return dict(sr=sr, dd=dd, passpct=fp["passpct"], med=fp["med_mo"],
                sr1=float(h1.mean() / h1.std() * np.sqrt(252)),
                sr2=float(h2.mean() / h2.std() * np.sqrt(252)))


def sleeve_fit(classes, eng, minlot):
    """min-lot / target-notional for one sleeve inside a given book."""
    lev = vbc.target_leverage("flex", classes)
    tgt = lev.get(eng, 0.0) * EQUITY
    return tgt, (minlot / tgt if tgt > 0 else float("inf"))


if __name__ == "__main__":
    print("=" * 78)
    print("STEP 0 — MIN-LOT FIT at $10,000 (the gate that kills most candidates)")
    print("=" * 78)
    print(f"{'engine':8s} {'FP symbol':10s} {'min-lot $':>10s} {'target $':>10s} "
          f"{'fit':>6s} {'spr bp':>7s} {'needs acct':>11s}  verdict")
    sizeable, oversized = [], []
    for eng, (fp_sym, minlot, spr) in sorted(FP_UNIVERSE.items(), key=lambda kv: kv[1][1]):
        if eng in ("ETH", "DJI"):
            continue                                    # already in the book
        cls = {k: list(v) for k, v in LIVE_2.items()}
        cls["x_cand"] = [eng]
        try:
            tgt, fit = sleeve_fit(cls, eng, minlot)
        except Exception as e:                          # noqa: BLE001
            print(f"{eng:8s} {fp_sym:10s} {minlot:10.2f}   (no data: {e})")
            continue
        ok = fit <= FIT_MAX
        # Target notional scales linearly with equity while target LEVERAGE is
        # scale-invariant, so the account that would make this sleeve tradeable
        # at FIT_MAX is just minlot / (lev * FIT_MAX).
        need = minlot / (tgt / EQUITY) / FIT_MAX if tgt > 0 else float("inf")
        if ok:
            sizeable.append((eng, fp_sym, minlot, tgt, fit, spr))
        else:
            oversized.append((eng, fp_sym, fit, need))
        print(f"{eng:8s} {fp_sym:10s} {minlot:10.2f} {tgt:10.2f} {fit:6.2f}x "
              f"{spr:7.2f} {'—' if ok else f'${need/1000:,.0f}k':>11s}  "
              f"{'SIZEABLE' if ok else 'oversize'}")

    if not sizeable:
        print("\nNo candidate fits at $10,000 — the book is size-constrained, "
              "not idea-constrained.")
        sys.exit(0)

    print("\n" + "=" * 78)
    print("STEP 1 — sleeve screen (standalone Sharpe + corr to surviving ETH/DJI)")
    print("=" * 78)
    base_nets = {}
    for s in ("ETH", "DJI"):
        _, nt = vbc._load_asset(s)
        base_nets[s] = nt.loc["2017-01-01":]
    base_df = pd.DataFrame(base_nets)

    print(f"{'engine':8s} {'FP symbol':10s} {'SR':>6} {'rETH':>6} {'rDJI':>6} "
          f"{'maxcorr':>8}  verdict")
    cands = []
    for eng, fp_sym, minlot, tgt, fit, spr in sizeable:
        _, nt = vbc._load_asset(eng)
        nt = nt.loc["2017-01-01":]
        j = pd.concat([base_df, nt.rename("cand")], axis=1).dropna()
        if len(j) < 500:
            print(f"{eng:8s} {fp_sym:10s}   (history too short: {len(j)}d)")
            continue
        sr = float(j["cand"].mean() / j["cand"].std() * np.sqrt(252))
        cor = {k: float(j["cand"].corr(j[k])) for k in ("ETH", "DJI")}
        mx = max(abs(v) for v in cor.values())
        ok = sr > 0.30 and mx < 0.45
        if ok:
            cands.append((eng, fp_sym, fit))
        print(f"{eng:8s} {fp_sym:10s} {sr:+6.2f} {cor['ETH']:+6.2f} {cor['DJI']:+6.2f} "
              f"{mx:8.2f}  {'CANDIDATE' if ok else ''}")

    print("\n" + "=" * 78)
    print("STEP 2 — rebuild the book with each survivor (FundingPips FLEX, 7% vol)")
    print("=" * 78)
    orig = book_stats({k: list(v) for k, v in ORIG_3.items()})
    live2 = book_stats({k: list(v) for k, v in LIVE_2.items()})
    print(f"{'book':30s} {'Sharpe':>7} {'maxDD':>7} {'pass%':>7} {'median':>8} "
          f"{'17-20':>7} {'21-26':>7}")
    print(f"{'WAS  XAU+ETH+DJI (dead)':30s} {orig['sr']:+7.2f} {orig['dd']:6.1f}% "
          f"{orig['passpct']:7.1f} {orig['med']:7.1f}mo {orig['sr1']:+7.2f} {orig['sr2']:+7.2f}")
    print(f"{'NOW  ETH+DJI (2 sleeves)':30s} {live2['sr']:+7.2f} {live2['dd']:6.1f}% "
          f"{live2['passpct']:7.1f} {live2['med']:7.1f}mo {live2['sr1']:+7.2f} {live2['sr2']:+7.2f}")
    print("-" * 78)

    for eng, fp_sym, fit in cands:
        cls = {k: list(v) for k, v in LIVE_2.items()}
        cls[f"x_{eng.lower()}"] = [eng]
        try:
            r = book_stats(cls)
        except Exception as e:                          # noqa: BLE001
            print(f"  +{eng:28s} ERROR {e}")
            continue
        better = (r["passpct"] >= live2["passpct"] and r["med"] <= live2["med"]
                  and min(r["sr1"], r["sr2"]) > 0.8)
        print(f"{'+ ' + eng + ' (' + fp_sym + ')':30s} {r['sr']:+7.2f} {r['dd']:6.1f}% "
              f"{r['passpct']:7.1f} {r['med']:7.1f}mo {r['sr1']:+7.2f} {r['sr2']:+7.2f}"
              f"{'   <-- RESTORES' if better else ''}")

    print("\nA replacement must beat the CURRENT 2-sleeve reality on pass% AND")
    print("median AND stay healthy in both half-samples — and fit min-lot at 10K.")
    print("Beating the dead 3-sleeve book is a bonus, not the bar.")

    if oversized:
        print("\n" + "=" * 78)
        print("BLOCKED BY SIZE ONLY — these have the edge but not the granularity")
        print("=" * 78)
        for eng, fp_sym, fit, need in sorted(oversized, key=lambda r: r[3]):
            print(f"  {eng:8s} {fp_sym:10s} {fit:6.2f}x oversize at $10k  "
                  f"-> tradeable from ~${need/1000:,.0f}k")
