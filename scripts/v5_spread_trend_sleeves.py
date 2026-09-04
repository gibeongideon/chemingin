"""PRE-REGISTERED: RELATIVE-VALUE TREND — trend-follow SPREADS, not outright prices.

The one structural gap left after §3ae/§3af: every sleeve this book has ever held is an
OUTRIGHT directional position, and adding more outright trend assets is closed (breadth
dilutes 1.67 -> 1.04; no weighting scheme beats equal weight). A spread trend sleeve is a
different object - the common market factor cancels between the two legs, so its correlation
to an outright trend book is near zero BY CONSTRUCTION rather than by luck. Classic CTA
relative-value positions (curve steepeners, crack/quality spreads, tech-vs-broad) live here.
grep confirms nothing in this repo trend-follows a ratio: the only spread work is
`gold-silver-spread-disproven`, which tested MEAN REVERSION on a rolling z-spread (dead OOS
2017+). Trend-following the same ratio is the opposite hypothesis, not a repeat.

PRE-REGISTRATION (fixed before any result):
  * SAME-SESSION PAIRS ONLY, decided a priori. §3b showed a cross-sectional signal built from
    non-synchronous closes can fake a Sharpe 3.5. Pairing e.g. SPX with NIKKEI would mix
    trading windows, so every pair below trades in one region/session or is 24h.
  * SYMMETRIC long/short forecast: 0.5*EWMAC + 0.5*breakout, no long-only concentration. The
    long-only concentration is a GOLD-SPECIFIC result (`xau-longonly-champion`) and a ratio
    has no reason to drift up, so imposing it would be a borrowed prior.
  * Speeds are the champion's own, rescaled to D1 via scale=1/6. No search over speeds.
  * COSTS ARE DOUBLED - both legs pay, and each leg's bp cost is the same conservative
    per-class figure used in §3af.
  * 15 pairs, each reported individually; the PRIMARY claim is the equal-risk BASKET of pairs
    as a single new sleeve added to the deployed core-4 book.
  * KEEPER BAR: basket standalone Sharpe >= 0.35 with both halves > 0, |corr| to the core book
    < 0.30, and book-level t@matched >= +1.50 with both half-deltas positive.

    python scripts/v5_spread_trend_sleeves.py
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

from scripts.v5_volregime_taper_crossasset import engine_bp, load_sleeve  # noqa: E402
from scripts.v5_xau_champion_lifts import champion_recipe, vol_match, sharpe, dd_of  # noqa: E402
from src.v5.xau_dual_signals import champion_signal, ewmac_fc, breakout_fc, EWMAC_MID, BKO_FAST  # noqa: E402
from scripts.v5_xau_turn_prob import paired, per_year  # noqa: E402

EVAL_START = "2018-01-01"
SPLITS = [("2018-01-01", "2021-12-31", "2018-21"), ("2022-01-01", "2026-12-31", "2022-26")]
TARGET_VOL, MAX_LEV, BUFFER, VOL_HL = 0.10, 6.0, 0.10, 20
SCALE = 1 / 6                     # champion speeds, rescaled from H4 to D1

# (leg A, leg B, session, bp cost A, bp cost B)
PAIRS = [
    ("GOLD", "SILVER", "metals LDN/NY", 0.75, 3.0),
    ("GOLD", "PLAT", "metals LDN/NY", 0.75, 3.0),
    ("PLAT", "PALL", "metals LDN/NY", 3.0, 3.0),
    ("COPPER", "GOLD", "metals LDN/NY", 3.0, 0.75),
    ("WTI", "BRENT", "energy ICE/NYM", 5.69, 5.69),
    ("HEATOIL", "GASOIL", "energy ICE/NYM", 6.0, 6.0),
    ("BRENT", "NATGAS", "energy ICE/NYM", 5.69, 6.0),
    ("NDX", "SPX", "US cash equity", 0.38, 0.38),
    ("NDX", "DJI", "US cash equity", 0.38, 0.5),
    ("SPX", "DJI", "US cash equity", 0.38, 0.5),
    ("UST2Y", "UST10Y", "US rates", 2.0, 2.0),
    ("UST10Y", "UST30Y", "US rates", 2.0, 2.0),
    ("BTC", "ETH", "crypto 24h", 1.13, 2.03),
    ("AUDUSD", "NZDUSD", "FX 24h", 1.5, 1.5),
    ("CORN", "WHEAT", "CBOT grains", 6.0, 6.0),
]
TREND_CORE = {
    "XAU":   ("data/XAUUSD_H4_long.csv", 252 * 6, 0.75, 1.0, 42),
    "BTC":   ("data/BTC_D1_long.csv", 252, 1.13, 1 / 6, 7),
    "NDX":   ("data/NDX_D1_long.csv", 252, 0.38, 1 / 6, 7),
    "BRENT": ("data/BRENT_D1_long.csv", 252, 5.69, 1 / 6, 7),
}


def ls_forecast(px: pd.Series) -> pd.Series:
    """Symmetric Carver blend at the champion's own horizons, rescaled to daily bars."""
    ewm = tuple((max(2, int(round(f * SCALE))), max(3, int(round(s * SCALE))))
                for f, s in EWMAC_MID)
    bko = tuple(max(3, int(round(w * SCALE))) for w in BKO_FAST)
    return (0.5 * ewmac_fc(px, ewm) + 0.5 * breakout_fc(px, bko)).clip(-2, 2)


def _buffer(p: np.ndarray, band: np.ndarray) -> np.ndarray:
    out, held = np.zeros(len(p)), 0.0
    for i in range(len(p)):
        if np.isfinite(p[i]):
            b = band[i] if np.isfinite(band[i]) else 0.0
            if abs(p[i] - held) > b:
                held = p[i] - np.sign(p[i] - held) * b
        out[i] = held
    return out


def pair_sleeve(a: str, b: str, bpa: float, bpb: float) -> pd.Series | None:
    """Trend the log ratio; hold +1 unit of A and -1 unit of B at equal risk, so the
    common factor cancels. Both legs pay cost on every rebalance."""
    da, db = load_sleeve(dict(path=f"data/{a}_D1_long.csv")), load_sleeve(dict(path=f"data/{b}_D1_long.csv"))
    if da is None or db is None:
        return None
    P = pd.concat({a: da["close"].astype(float), b: db["close"].astype(float)}, axis=1).dropna()
    if len(P) < 1500:
        return None
    ratio = P[a] / P[b]
    fc = ls_forecast(ratio)
    rr = ratio.pct_change()
    vol = rr.ewm(halflife=VOL_HL, min_periods=60).std() * np.sqrt(252)
    raw = (fc * (TARGET_VOL / vol.replace(0, np.nan))).clip(-MAX_LEV, MAX_LEV)
    band = (BUFFER * (TARGET_VOL / vol.replace(0, np.nan)).clip(0, MAX_LEV))
    pos = pd.Series(_buffer(raw.values, band.values), index=raw.index).shift(1).fillna(0.0)
    gross = pos * rr
    cost = pos.diff().abs().fillna(0.0) * ((bpa + bpb) * 1e-4)
    return (gross - cost).loc[EVAL_START:].dropna()


def core_sleeves() -> dict[str, pd.Series]:
    out = {}
    for name, (path, ann, bp, scale, hl) in TREND_CORE.items():
        df = load_sleeve(dict(path=path))
        fc = champion_signal(df["close"]) if scale == 1.0 else champion_recipe(df["close"], scale, 1.5)
        out[name] = engine_bp(df, fc, bp, ann, hl).loc[EVAL_START:]
    return out


def equal_risk(sleeves: dict[str, pd.Series]) -> pd.Series:
    """Causal equal-risk combination: expanding vol, shifted, so no future information."""
    R = pd.DataFrame(sleeves).dropna(how="all").fillna(0.0)
    v = R.expanding(min_periods=252).std().shift(1)
    W = ((0.10 / np.sqrt(252)) / v.replace(0, np.nan)).clip(upper=10)
    return (R * W).mean(axis=1).dropna()


def halves(r: pd.Series) -> list[float]:
    return [sharpe(r.loc[a:b]) for a, b, _ in SPLITS]


def gate(c: pd.Series, b: pd.Series) -> tuple[float, str, list[float]]:
    i = c.index.intersection(b.index)
    c, b = c.loc[i], b.loc[i]
    _, t, _ = paired(vol_match(c, b), b)
    yp, yn = per_year(vol_match(c, b), b)
    dh = [sharpe(vol_match(c.loc[a:x], b.loc[a:x])) - sharpe(b.loc[a:x]) for a, x, _ in SPLITS]
    return t, f"{yp}/{yn}", dh


def main() -> None:
    core = core_sleeves()
    base = pd.DataFrame(core).dropna(how="all").fillna(0.0).mean(axis=1)
    hb = halves(base)
    print(f"BASELINE deployed core-4 book: SR {sharpe(base):+.3f}  DD {dd_of(base):+.2f}%  "
          f"halves {hb[0]:+.2f}/{hb[1]:+.2f}")
    print("keeper bar: basket SR>=0.35, both halves>0, |corr|<0.30, book t@matched>=+1.50 both +\n")

    print("--- individual spread sleeves (reported in full; none is the claim) ---")
    sl = {}
    for a, b, sess, bpa, bpb in PAIRS:
        r = pair_sleeve(a, b, bpa, bpb)
        if r is None or len(r) < 400 or r.std() == 0:
            print(f"  {a}/{b:8s} unavailable")
            continue
        sl[f"{a}/{b}"] = r
        i = r.index.intersection(base.index)
        print(f"  {a+'/'+b:16s} {sess:16s} SR {sharpe(r):+.3f}  DD {dd_of(r):+7.2f}%  "
              f"halves {halves(r)[0]:+.2f}/{halves(r)[1]:+.2f}  corr(core) "
              f"{r.loc[i].corr(base.loc[i]):+.2f}")

    if not sl:
        print("no pairs available")
        return

    basket = equal_risk(sl)
    i = basket.index.intersection(base.index)
    corr = basket.loc[i].corr(base.loc[i])
    hbk = halves(basket)
    print(f"\n--- PRIMARY: equal-risk BASKET of {len(sl)} spread sleeves ---")
    print(f"  SR {sharpe(basket):+.3f}  DD {dd_of(basket):+.2f}%  "
          f"halves {hbk[0]:+.2f}/{hbk[1]:+.2f}  corr(core) {corr:+.2f}")

    aug = equal_risk({**core, "SPREADS": basket})
    ref = equal_risk(core)
    t, yy, dh = gate(aug, ref)
    keep = (sharpe(basket) >= 0.35 and min(hbk) > 0 and abs(corr) < 0.30 and t >= 1.50 and min(dh) > 0)
    print(f"  core-4 (equal risk) SR {sharpe(ref):+.3f} -> +spreads SR {sharpe(aug):+.3f}  "
          f"DD {dd_of(ref):+.2f}% -> {dd_of(aug):+.2f}%")
    print(f"  t@matched {t:+.2f} [{yy}]  dSR by half {dh[0]:+.2f}/{dh[1]:+.2f}   "
          f"{'** KEEPER **' if keep else 'FAIL'}")

    pd.DataFrame([dict(tag=k, sr=sharpe(v), dd=dd_of(v), h1=halves(v)[0], h2=halves(v)[1])
                  for k, v in {**sl, "BASKET": basket}.items()]).to_csv(
        ROOT / "data" / "v5_runs" / "spread_trend_sleeves.csv", index=False)
    print("-> data/v5_runs/spread_trend_sleeves.csv")


if __name__ == "__main__":
    main()
