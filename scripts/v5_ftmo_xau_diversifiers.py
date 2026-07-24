"""FTMO 100K — XAU trend follower ALONE, then build outward with diversifiers.

Framing differs from v5_instrument_search.py (which asks "does a 4th sleeve help
the finished XAU+BTC+NDX book?"). Here we start from the XAU trend follower on
its own and grow the book one uncorrelated sleeve at a time, so the marginal
value of EACH addition is visible rather than assumed.

Why this is worth re-asking on FTMO when it was hopeless on FundingPips 10K:
the 10K book is SIZE-constrained (see v5_fp_xau_replacement.py — every real
diversifier was 2.5x-21.8x oversize at min-lot). FTMO at $100k is not: verified
live on account 1514025597, 167 symbols, and the index CFDs have contract size
1.0 so min-lots are $62-$519, i.e. ~0.1-0.5% of equity. Granularity is a
non-issue here, so the search is free to be about EDGE and CORRELATION.

Bars a candidate must clear:
  1. min-lot fits its own vol-targeted target notional at $100k (cheap, checked)
  2. positive standalone champion-recipe Sharpe
  3. |corr| to XAU below the diversification bar
  4. the PAIR (XAU + candidate) must beat XAU alone on pass% AND median
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

M = vbc.MODELS["ftmo"]
DIAL = M["vol"] / vbc.TARGET_VOL
EQUITY = 100_000.0
FIT_MAX = 1.5
CORR_MAX = 0.45
SR_MIN = 0.30

# engine sym -> (FTMO symbol, min-lot notional $, spread bp), VERIFIED LIVE on
# account 1514025597 @ FTMO-Demo 2026-07-24 (167 symbols).
# Excluded on purpose: FX/exotics (comprehensively dead post-2016, V5_FINDINGS),
# UST2Y/5Y/10Y/30Y and GASOIL (FTMO does not offer them at all).
FTMO_UNIVERSE = {
    # metals
    "GOLD":   ("XAUUSD",      4047.55,  1.11),
    "SILVER": ("XAGUSD",      2906.20,  9.12),
    "PLAT":   ("XPTUSD",      1591.56, 46.56),
    "PALL":   ("XPDUSD",      1241.97, 56.36),
    "COPPER": ("XCUUSD",       629.97,  8.89),
    # energy
    "WTI":    ("USOIL.cash",    91.01,  9.67),
    "BRENT":  ("UKOIL.cash",    94.98,  7.58),
    "NATGAS": ("NATGAS.cash",   28.59, 199.37),
    "HEATOIL": ("HEATOIL.c",     4.24, 80.13),
    # crypto
    "BTC":    ("BTCUSD",       653.38,  0.15),
    "ETH":    ("ETHUSD",       188.80,  3.18),
    "LTC":    ("LTCUSD",        46.62, 32.18),
    "SOL":    ("SOLUSD",        75.74,  3.96),
    "AVA":    ("AVAUSD",        62.40, 32.05),
    # equity indices
    "SPX":    ("US500.cash",    74.17,  0.81),
    "DJI":    ("US30.cash",    518.66,  0.40),
    "NDX":    ("US100.cash",   284.35,  0.69),
    "DAX":    ("GER40.cash",   249.05,  0.49),
    "FTSE":   ("UK100.cash",   106.69,  0.70),
    "STOXX":  ("EU50.cash",     62.42,  1.70),
    "NIKKEI": ("JP225.cash",  6465.15,  1.55),
    "ASX":    ("AUS200.cash",   87.92,  1.14),
    # agriculture
    "CORN":   ("CORN.c",       462.61, 22.70),
    "WHEAT":  ("WHEAT.c",      700.53, 28.55),
    "SOY":    ("SOYBEAN.c",   1237.23, 17.78),
    "SUGAR":  ("SUGAR.c",       14.34, 83.68),
    "COFFEE": ("COFFEE.c",     312.11, 10.89),
    "COTTON": ("COTTON.c",      77.90, 48.78),
}

XAU_ALONE = {"xau": ["XAUCHAMP"]}
CHAMPION = {"xau": ["XAUCHAMP"], "crypto": ["BTC"], "eq_us": ["NDX"]}

# ---------------------------------------------------------------------------
# LIVE-COST FLOOR — the whole point of this block.
#
# The bundled D1 CSVs carry a nominal `spread` column that is WILDLY optimistic
# for the illiquid corners of the universe. Measured against FTMO's live quotes
# (2026-07-24): HEATOIL 4.0bp modelled vs 80.1bp real (20x), NATGAS 4.0 vs 199.4
# (50x), PALL 5.0 vs 56.4 (11x), BRENT/WTI/SILVER ~3x, and gold's H4 stream
# models $0.30/oz against a live $0.448/oz. Liquid instruments (NDX/SPX/DJI/BTC/
# SOL/NIKKEI/DAX) are already conservative and unaffected.
#
# Left uncorrected this manufactures fake diversifiers: a first pass ranked
# XAU+NDX+HEATOIL as the best book in the study purely on a 20x cost subsidy.
# This project has been bitten by exactly this before (the XAU fade signal
# looked like Sharpe 1.9 until true spread took it to -7.5), so every asset's
# cost is floored at what FTMO actually quotes. Never cheaper than reality.
# ---------------------------------------------------------------------------
LIVE_SPREAD_BP = {eng: spr for eng, (_s, _m, spr) in FTMO_UNIVERSE.items()}
XAU_LIVE_BP = 1.11                                  # FTMO XAUUSD, live

_orig_load_asset = vbc._load_asset
_orig_load_h4 = vbc.load_h4


def _load_h4_live():
    df = _orig_load_h4()
    df["spread_px"] = np.maximum(df["spread_px"], df["close"] * XAU_LIVE_BP / 1e4)
    return df


def _load_asset_live(sym):
    """vbc._load_asset with the per-asset spread floored at FTMO's live quote."""
    if sym == "XAUCHAMP":
        return _orig_load_asset(sym)                # uses the patched load_h4
    bp = LIVE_SPREAD_BP.get(sym)
    if bp is None:
        return _orig_load_asset(sym)
    df = pd.read_csv(f"{ROOT}/data/{sym}_D1_long.csv",
                     parse_dates=["time"], index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    spr = df["spread"].clip(lower=df["spread"].median())
    df["spread_px"] = np.maximum(spr, df["close"] * bp / 1e4)
    close = df["close"]
    fc = vbc.champ_recipe_lo(close)
    ret = close.pct_change()
    vol = ret.ewm(halflife=42, min_periods=20).std() * np.sqrt(252)
    pos = vbc._buffered_pos(fc, vol, df["spread_px"], close, 252).shift(1).fillna(0.0)
    cost = pos.diff().abs().fillna(0.0) * (df["spread_px"] / close)
    net = (pos * ret - cost).fillna(0.0).resample("D").sum()
    net = net[net.index.dayofweek < 5]
    return vbc._buffered_pos(fc, vol, df["spread_px"], close, 252), net


vbc.load_h4 = _load_h4_live
vbc._load_asset = _load_asset_live


def book_stats(classes):
    vbc.CLASSES = {k: list(v) for k, v in classes.items()}
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


def show(label, r, mark=""):
    print(f"{label:34s} {r['sr']:+7.2f} {r['dd']:6.1f}% {r['passpct']:7.1f} "
          f"{r['med']:7.1f}mo {r['sr1']:+7.2f} {r['sr2']:+7.2f}{mark}")


HDR = (f"{'book':34s} {'Sharpe':>7} {'maxDD':>7} {'pass%':>7} {'median':>8} "
       f"{'17-20':>7} {'21-26':>7}")

if __name__ == "__main__":
    print("=" * 92)
    print("STEP 1 — the XAU trend follower ALONE on FTMO 100K (2-Step rules, 7% vol)")
    print("=" * 92)
    print(HDR)
    alone = book_stats(XAU_ALONE)
    show("XAU alone (XAUCHAMP H4)", alone)
    champ = book_stats(CHAMPION)
    show("XAU+BTC+NDX (current champion)", champ)
    lev = vbc.target_leverage("ftmo", XAU_ALONE).get("XAUCHAMP", 0.0)
    tgt = lev * EQUITY
    print(f"\n  sizing: XAU-alone target notional ${tgt:,.0f} vs XAUUSD min-lot "
          f"$4,048 -> {4047.55 / tgt if tgt else 0:.2f}x  "
          f"({'fits' if tgt and 4047.55 / tgt <= FIT_MAX else 'OVERSIZE'})")

    print("\n" + "=" * 92)
    print("STEP 2 — screen every FTMO-tradeable instrument vs the XAU sleeve")
    print("=" * 92)
    _, xau_net = vbc._load_asset("XAUCHAMP")
    xau_net = xau_net.loc["2017-01-01":]

    print(f"{'engine':8s} {'FTMO symbol':13s} {'SR':>6} {'rXAU':>6} {'minlot$':>9} "
          f"{'fit':>6} {'spr bp':>7}  verdict")
    cands = []
    for eng, (sym, minlot, spr) in sorted(FTMO_UNIVERSE.items()):
        if eng == "GOLD":
            continue                                   # same factor as XAUCHAMP
        try:
            _, nt = vbc._load_asset(eng)
        except Exception:                              # noqa: BLE001
            print(f"{eng:8s} {sym:13s}   (no local D1 data)")
            continue
        nt = nt.loc["2017-01-01":]
        j = pd.concat([xau_net.rename("xau"), nt.rename("cand")], axis=1).dropna()
        if len(j) < 500:
            print(f"{eng:8s} {sym:13s}   (history too short: {len(j)}d)")
            continue
        sr = float(j["cand"].mean() / j["cand"].std() * np.sqrt(252))
        cor = float(j["cand"].corr(j["xau"]))
        cls = {"xau": ["XAUCHAMP"], "x_cand": [eng]}
        try:
            clev = vbc.target_leverage("ftmo", cls).get(eng, 0.0)
        except Exception:                              # noqa: BLE001
            clev = 0.0
        ctgt = clev * EQUITY
        fit = (minlot / ctgt) if ctgt > 0 else float("inf")
        ok = sr > SR_MIN and abs(cor) < CORR_MAX and fit <= FIT_MAX
        if ok:
            cands.append((eng, sym, sr, cor, fit))
        why = "CANDIDATE" if ok else (
            "weak SR" if sr <= SR_MIN else
            "correlated" if abs(cor) >= CORR_MAX else "oversize")
        print(f"{eng:8s} {sym:13s} {sr:+6.2f} {cor:+6.2f} {minlot:9.0f} "
              f"{fit:6.2f}x {spr:7.2f}  {why}")

    if not cands:
        print("\nNo candidate clears the bars.")
        sys.exit(0)

    print("\n" + "=" * 92)
    print("STEP 3 — PAIRS: XAU + each survivor (does the addition earn its place?)")
    print("=" * 92)
    print(HDR)
    show("XAU alone (baseline)", alone)
    print("-" * 92)
    pairs = []
    for eng, sym, sr, cor, fit in sorted(cands, key=lambda c: -c[2]):
        cls = {"xau": ["XAUCHAMP"], f"x_{eng.lower()}": [eng]}
        try:
            r = book_stats(cls)
        except Exception as e:                         # noqa: BLE001
            print(f"  + {eng:30s} ERROR {e}")
            continue
        better = (r["passpct"] >= alone["passpct"] and r["med"] <= alone["med"]
                  and min(r["sr1"], r["sr2"]) > 0.8)
        pairs.append((eng, sym, r, better))
        show(f"XAU + {eng} ({sym})", r, "   <-- IMPROVES" if better else "")

    print("\n" + "=" * 92)
    print("STEP 4 — best pair grown to a 3rd sleeve, vs the current champion")
    print("=" * 92)
    winners = [p for p in pairs if p[3]]
    if not winners:
        print("No pair improves on XAU alone — stop here.")
        sys.exit(0)
    winners.sort(key=lambda p: (-p[2]["passpct"], p[2]["med"]))
    best_eng, best_sym, best_r, _ = winners[0]
    print(f"best pair: XAU + {best_eng} ({best_sym})\n")
    print(HDR)
    show("XAU alone", alone)
    show(f"XAU + {best_eng}", best_r)
    print("-" * 92)
    trios = []
    for eng, sym, sr, cor, fit in cands:
        if eng == best_eng:
            continue
        cls = {"xau": ["XAUCHAMP"], f"x_{best_eng.lower()}": [best_eng],
               f"x_{eng.lower()}": [eng]}
        try:
            r = book_stats(cls)
        except Exception:                              # noqa: BLE001
            continue
        trios.append((eng, sym, r))
    trios.sort(key=lambda t: (-t[2]["passpct"], t[2]["med"]))
    for eng, sym, r in trios[:10]:
        better = (r["passpct"] >= best_r["passpct"] and r["med"] <= best_r["med"]
                  and min(r["sr1"], r["sr2"]) > 0.8)
        show(f"XAU + {best_eng} + {eng}", r, "   <-- IMPROVES" if better else "")
    print("-" * 92)
    show("XAU+BTC+NDX (current champion)", champ)

    print("\nA sleeve earns its place only if it raises pass% AND cuts median AND")
    print("both half-samples stay healthy. Anything else is regime luck.")
