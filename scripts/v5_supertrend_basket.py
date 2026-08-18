"""SuperTrend as a DIVERSIFIER, not an XAU replacement (2026-08-19 follow-up to §3s).

`scripts/v5_xau_supertrend.py` found SuperTrend genuinely real on XAU H4 (DSR 0.9948)
but too correlated to the champion's own XAU return stream (0.77) to add value there.
This tests the natural next step flagged in that finding: apply SuperTrend to the
SAME four instruments as the second independent basket
([[new-independent-basket-nikkei-coffee-eth-dax]]: NIKKEI/COFFEE/ETH/DAX) — an asset
set the champion doesn't already own — and see whether a DIFFERENT signal MECHANISM
on those same assets adds anything the champ-recipe version didn't.

Reuses `v5_basket_challenge.py`'s cost/vol-target/buffer machinery directly
(`_buffered_pos`, `TARGET_VOL`) so results are on the exact same footing as every
other basket number in this repo — only the forecast generator changes.

    python scripts/v5_supertrend_basket.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.v5_basket_challenge as vbc  # noqa: E402
from scripts.v5_xau_supertrend import supertrend_direction  # noqa: E402
from src.evaluation.dsr_pbo import deflated_sharpe_ratio, pbo_cscv  # noqa: E402

SYMBOLS = ("NIKKEI", "COFFEE", "ETH", "DAX")
ATR_PERIODS = (10, 14, 20)
MULTIPLIERS = (2.5, 3.0, 3.5)
EVAL_START = "2018-01-01"


def load_d1(sym: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / f"{sym}_D1_long.csv", parse_dates=["time"], index_col="time").sort_index()
    return df[~df.index.duplicated(keep="last")]


def supertrend_net_returns(df: pd.DataFrame, atr_period: int, multiplier: float, long_only: bool) -> pd.Series:
    """Same cost/vol-target/buffer mechanics as vbc._load_asset, forecast swapped
    for SuperTrend instead of the champ EWMAC+breakout recipe."""
    df = df.copy()
    df["spread_px"] = df["spread"].clip(lower=df["spread"].median())
    direction = supertrend_direction(df, atr_period, multiplier)
    fc = direction.clip(lower=0.0) if long_only else direction
    close = df["close"]
    ret = close.pct_change()
    vol = ret.ewm(halflife=42, min_periods=20).std() * np.sqrt(252)
    pos = vbc._buffered_pos(fc, vol, df["spread_px"], close, 252).shift(1).fillna(0.0)
    cost = pos.diff().abs().fillna(0.0) * (df["spread_px"] / close)
    net = (pos * ret - cost).fillna(0.0).resample("D").sum()
    return net[net.index.dayofweek < 5]


def sharpe(d: pd.Series, start: str = EVAL_START) -> float:
    d = d.loc[start:].dropna()
    return float(d.mean() / d.std() * np.sqrt(252)) if d.std() > 0 else float("nan")


def dd(d: pd.Series, start: str = EVAL_START) -> float:
    e = (1 + d.loc[start:]).cumprod()
    return float((e / e.cummax() - 1).min() * 100)


def cagr(d: pd.Series, start: str = EVAL_START) -> float:
    e = (1 + d.loc[start:]).cumprod()
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    return float(e.iloc[-1] ** (1 / yrs) - 1) * 100


def main():
    print("=== STAGE 1: per-instrument SuperTrend grid, long-only, standalone ===\n")
    dfs = {sym: load_d1(sym) for sym in SYMBOLS}
    best_per_symbol = {}

    for sym in SYMBOLS:
        grid_rets = {}
        for p in ATR_PERIODS:
            for m in MULTIPLIERS:
                grid_rets[(p, m)] = supertrend_net_returns(dfs[sym], p, m, long_only=True)
        M = pd.DataFrame(grid_rets).dropna(how="all").fillna(0.0)
        M = M.loc[EVAL_START:]
        sh = {k: sharpe(M[k], start=M.index[0].strftime("%Y-%m-%d")) for k in M.columns}
        best_key = max(sh, key=lambda k: sh[k] if np.isfinite(sh[k]) else -1e9)
        trial_sh_daily = np.array([sh[k] / np.sqrt(252) for k in sh])
        dsr = deflated_sharpe_ratio(M[best_key].values, trial_sh_daily)
        pbo = pbo_cscv(M.values, n_partitions=10)
        best_per_symbol[sym] = (best_key, M[best_key])
        print(f"{sym}: best (period={best_key[0]}, mult={best_key[1]})  "
              f"SR={sh[best_key]:+.3f}  DD={dd(M[best_key]):+.1f}%  "
              f"DSR={dsr['dsr']:.3f}  PBO={pbo.pbo:.3f} (n={pbo.n_splits})")
        for k, v in sorted(sh.items(), key=lambda kv: -kv[1] if np.isfinite(kv[1]) else 1e9):
            print(f"    period={k[0]:2d} mult={k[1]:.1f}  SR={v:+.3f}")
        print()

    print("=== STAGE 2: compare to the champ-recipe version of the SAME 4 assets ===\n")
    champ_rets = {}
    for sym in SYMBOLS:
        _, nt = vbc._load_asset(sym)
        champ_rets[sym] = nt.loc[EVAL_START:]
    for sym in SYMBOLS:
        st_ret = best_per_symbol[sym][1]
        j = pd.concat([st_ret.rename("st"), champ_rets[sym].rename("champ")], axis=1).dropna()
        corr = j.st.corr(j.champ)
        print(f"{sym}: SuperTrend SR {sharpe(st_ret):+.3f} vs champ-recipe SR {sharpe(champ_rets[sym]):+.3f}  "
              f"corr(ST, champ-recipe on same asset) = {corr:+.3f}")

    print("\n=== STAGE 3: build the SuperTrend BASKET (equal-class-risk, same 4 assets) ===\n")
    st_book_raw = sum(
        best_per_symbol[sym][1] * (0.10 / (best_per_symbol[sym][1].std() * np.sqrt(252)))
        for sym in SYMBOLS
    ) / len(SYMBOLS)
    rv = st_book_raw.ewm(halflife=vbc.VT_HALFLIFE, min_periods=20).std() * np.sqrt(252)
    vol_s = (vbc.TARGET_VOL / rv).clip(0, vbc.VT_MAXSCALE)
    eqb = (1 + st_book_raw).cumprod()
    dd_s = (1 + (eqb / eqb.cummax() - 1) * 3.0).clip(lower=vbc.DD_FLOOR)
    st_book = (st_book_raw * (vol_s * dd_s).shift(1)).dropna()
    print(f"SuperTrend basket (NIKKEI+COFFEE+ETH+DAX):  SR {sharpe(st_book):+.3f}  "
          f"CAGR {cagr(st_book):+.1f}%  DD {dd(st_book):+.1f}%")

    vbc.CLASSES = {"eq_ap": ["NIKKEI"], "softs": ["COFFEE"], "crypto_alt": ["ETH"], "eq_eu": ["DAX"]}
    _, champ_book_raw, _ = vbc.build(dial=1.0, start=EVAL_START)
    rv2 = champ_book_raw.ewm(halflife=vbc.VT_HALFLIFE, min_periods=20).std() * np.sqrt(252)
    vol_s2 = (vbc.TARGET_VOL / rv2).clip(0, vbc.VT_MAXSCALE)
    eqb2 = (1 + champ_book_raw).cumprod()
    dd_s2 = (1 + (eqb2 / eqb2.cummax() - 1) * 3.0).clip(lower=vbc.DD_FLOOR)
    champ_book = (champ_book_raw * (vol_s2 * dd_s2).shift(1)).dropna()
    print(f"champ-recipe basket (same 4 assets):        SR {sharpe(champ_book):+.3f}  "
          f"CAGR {cagr(champ_book):+.1f}%  DD {dd(champ_book):+.1f}%")

    j = pd.concat([st_book.rename("st"), champ_book.rename("champ")], axis=1).dropna()
    print(f"\ncorrelation(SuperTrend basket, champ-recipe basket, SAME 4 assets) = {j.st.corr(j.champ):+.3f}")

    print("\n=== STAGE 4: blend weight sweep — ST basket + champ-recipe basket (same assets) ===")
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        blend = (1 - w) * j.champ + w * j.st
        print(f"  w(ST)={w:.2f}  SR {sharpe(blend):+.3f}  DD {dd(blend):+.1f}%")

    print("\n=== STAGE 5: does the ST basket diversify the FLAGSHIP (XAU+BTC+NDX)? ===\n")
    vbc.CLASSES = {"xau": ["XAUCHAMP"], "crypto": ["BTC"], "eq_us": ["NDX"]}
    _, flagship_raw, _ = vbc.build(dial=1.0, start=EVAL_START)
    rv3 = flagship_raw.ewm(halflife=vbc.VT_HALFLIFE, min_periods=20).std() * np.sqrt(252)
    vol_s3 = (vbc.TARGET_VOL / rv3).clip(0, vbc.VT_MAXSCALE)
    eqb3 = (1 + flagship_raw).cumprod()
    dd_s3 = (1 + (eqb3 / eqb3.cummax() - 1) * 3.0).clip(lower=vbc.DD_FLOOR)
    flagship = (flagship_raw * (vol_s3 * dd_s3).shift(1)).dropna()
    jf = pd.concat([flagship.rename("f"), st_book.rename("st")], axis=1).dropna()
    print(f"flagship alone:                       SR {sharpe(jf.f):+.3f}  DD {dd(jf.f):+.1f}%")
    print(f"correlation(flagship, ST basket) = {jf.f.corr(jf.st):+.3f}\n")
    for w in (0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 1.0):
        blend = (1 - w) * jf.f + w * jf.st
        print(f"  w(ST-basket)={w:.2f}  SR {sharpe(blend):+.3f}  DD {dd(blend):+.1f}%  CAGR {cagr(blend):+.1f}%")


if __name__ == "__main__":
    main()
