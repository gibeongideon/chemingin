"""DAILY SIGNAL DIGEST — the five findings in BEST_FINDINGS.MD, as live numbers.

SCOPE, per the user's instruction: **only the ZigZag detector TRADES on FTMO** (that is
`scripts/v5_zigzag_ftmo.py`, hourly). This script computes all five findings' current state and
EMAILS them. It has no order-sending code path at all.

The five, and what each is honestly worth:

  1. CHAMPION TREND on gold — the deployed long-only trend follower. TRADABLE. On a swap-free
     instrument it nets +1.065 Sharpe; on FTMO it does not, because FTMO charges carry. The live
     figure is printed each day so the gap is visible rather than assumed.
  2. GOLD-TILTED CORE-4 BOOK — XAU 50% / BTC, US100, UKOIL 16.7% each. TRADABLE, net Sharpe
     +1.212, the best measured book. Correlations +0.04..+0.14.
  3. ZIGZAG BOTTOM DETECTOR — best classification numbers in the project, upgraded 2026-09-10
     to POOLED cross-instrument training (80% precision @ 5%
     recall) and DISPROVEN as a trade (walk-forward -0.76, 0/8 years, DSR 0.000). Traded on the
     demo as an observation harness only.
  4. REGIME DIRECTION EDGE — 51.97% vs a 49.85% persistence baseline, +2.12pp, 8/9 years,
     bootstrap CI [+0.51, +3.69pp]. REAL and NOT TRADABLE: five trade structures failed.
  5. M15 INTRABAR PATH — +0.0142 bits/event above its alignment null's p99, precision 0.600 vs
     a 0.427 base rate. REAL and NOT TRADABLE: 0/16 as a sleeve overlay, 0/32 at book level.

Every instrument is read from FTMO's own terminal, so one feed and no splice. The CSVs are
2,063h stale and are a different feed from the broker's.

    python scripts/v5_signals_digest.py           # print + write JSON
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from src.v5.xau_dual_signals import champion_signal  # noqa: E402
from scripts.v5_xau_champion_lifts import champion_recipe  # noqa: E402
from scripts.v5_zigzag_signal import AUX_SYMBOLS, bottom_probability  # noqa: E402
from scripts.v5_repr_ceiling import f_base, f_source, first_touch  # noqa: E402

PORT = 18814
STATE = ROOT / "data" / "v5_runs" / "signals_digest.json"
TF = {"M15": 15, "H1": 16385, "H4": 16388, "D1": 16408}
TARGET_VOL, MAX_LEV, BUFFER = 0.10, 8.0, 0.10

# engine symbol -> (FTMO name, bar clock, champion speed scale, vol halflife, ann bars)
BOOK = {
    "XAU":   ("XAUUSD",     "H4", 1.0,   42, 252 * 6),
    "BTC":   ("BTCUSD",     "D1", 1 / 6,  7, 252),
    "NDX":   ("US100.cash", "D1", 1 / 6,  7, 252),
    "BRENT": ("UKOIL.cash", "D1", 1 / 6,  7, 252),
}
TILT = {"XAU": 0.50, "BTC": 0.1667, "NDX": 0.1667, "BRENT": 0.1667}


def rates(mt5, symbol: str, tf: str, n: int) -> pd.DataFrame:
    """Closed bars only — `copy_rates_from_pos(..,0,n)` returns the in-progress bar at 0."""
    mt5.symbol_select(symbol, True)
    r = None
    for _ in range(5):
        r = mt5.copy_rates_from_pos(symbol, TF[tf], 0, n)
        if r is not None and len(r) > 200:
            break
        time.sleep(2)
    if r is None or len(r) < 200:
        return pd.DataFrame()
    a = np.array([(x[0], x[1], x[2], x[3], x[4]) for x in r], dtype=float)
    df = pd.DataFrame(a[:, 1:], columns=["open", "high", "low", "close"],
                      index=pd.to_datetime(a[:, 0], unit="s"))
    hours = {"M15": 0.25, "H1": 1, "H4": 4, "D1": 24}[tf]
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    open_bars = df.index[df.index + pd.Timedelta(hours=hours) > now]
    return df.drop(index=open_bars) if len(open_bars) else df


def target_leverage(close: pd.Series, fc: pd.Series, ann: int, hl: int) -> float:
    """The engine's own sizing: forecast x (target vol / realised vol), capped and banded."""
    vol = close.pct_change().ewm(halflife=hl, min_periods=20).std() * np.sqrt(ann)
    lev = float(np.clip(fc.iloc[-1], 0, 2) * (TARGET_VOL / vol.iloc[-1]))
    return float(np.clip(lev, 0, MAX_LEV))


def pct_yr_carry(info, price: float) -> float:
    """swap_long -> %/yr on notional. POINTS mode is the common case on FTMO."""
    sm = getattr(info, "swap_mode", 1)
    notional = price * info.trade_contract_size
    if notional <= 0:
        return float("nan")
    if sm in (5, 6):
        return float(info.swap_long)
    return float(info.swap_long) * info.point * info.trade_contract_size / notional * 365 * 100


def main() -> None:
    from mt5linux import MetaTrader5
    mt5 = MetaTrader5(host="localhost", port=PORT)
    if not mt5.initialize():
        raise SystemExit(f"bridge {PORT} init failed: {mt5.last_error()}")
    ai = mt5.account_info()
    eq = float(ai.equity)
    out: dict = {"computed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "login": ai.login, "server": ai.server, "equity": eq}
    L: list[str] = [f"account {ai.login} @ {ai.server}   equity ${eq:,.2f}", ""]

    # ---------- 1 + 2: champion, and the gold-tilted core-4 book ----------
    L.append("=" * 68)
    L.append("1+2. CHAMPION TREND and the GOLD-TILTED CORE-4 BOOK   [TRADABLE]")
    L.append("=" * 68)
    L.append(f"{'sleeve':8s} {'symbol':12s} {'forecast':>9s} {'target lev':>11s} "
             f"{'lots@100k':>10s} {'carry %/yr':>11s}")
    book: dict = {}
    for name, (sym, tf, scale, hl, ann) in BOOK.items():
        df = rates(mt5, sym, tf, 6000 if tf != "D1" else 3000)
        if df.empty:
            L.append(f"{name:8s} {sym:12s}  no data")
            continue
        c = df["close"]
        fc = champion_signal(c) if scale == 1.0 else champion_recipe(c, scale, 1.5)
        lev = target_leverage(c, fc, ann, hl)
        info = mt5.symbol_info(sym)
        px = float(c.iloc[-1])
        lots = lev * TILT[name] * eq / (px * info.trade_contract_size)
        step = info.volume_step or 0.01
        lots = round(lots / step) * step
        carry = pct_yr_carry(info, px)
        book[name] = dict(symbol=sym, forecast=round(float(fc.iloc[-1]), 3),
                          target_lev=round(lev, 3), weight=TILT[name],
                          lots=round(lots, 2), carry_pct_yr=round(carry, 2),
                          last_bar=str(df.index[-1]), price=px)
        L.append(f"{name:8s} {sym:12s} {float(fc.iloc[-1]):9.3f} {lev:11.3f} "
                 f"{lots:10.2f} {carry:+11.2f}")
    out["book"] = book
    if "XAU" in book:
        L.append("")
        L.append(f"  Gold carry here is {book['XAU']['carry_pct_yr']:+.2f}%/yr. A swap-free")
        L.append(f"  instrument would be 0.00%/yr and worth about +0.20 Sharpe on this sleeve —")
        L.append(f"  that is finding #1, and FTMO does not offer it. BRENT/UKOIL carry is")
        L.append(f"  POSITIVE here, which is why it was FTMO's best net sleeve.")
    L.append("")
    L.append("  Book weights: XAU 50%, BTC/US100/UKOIL 16.7% each. Net Sharpe +1.212,")
    L.append("  sleeve correlations +0.04..+0.14. EMAIL ONLY — not traded on this account.")

    # ---------- 3: zigzag ----------
    L.append("")
    L.append("=" * 68)
    L.append("3. ZIGZAG BOTTOM DETECTOR   [TRADED on this demo — observation harness]")
    L.append("=" * 68)
    h1 = rates(mt5, "XAUUSD", "H1", 30000)
    if h1.empty:
        L.append("  no H1 data")
    else:
        # POOLED, exactly as the executor runs it — the digest and the executor must never
        # disagree about what fired, which is why both call the same function the same way.
        aux = {}
        for sym in AUX_SYMBOLS:
            a = rates(mt5, sym, "H1", 30000)
            if not a.empty:
                aux[sym] = a
        z = bottom_probability(h1, 0.60, 0.006, 0.005, 48, aux=aux)
        out["zigzag"] = z.as_dict()
        L.append(f"  P(at a bottom)  {z.prob:.4f}   threshold 0.60   "
                 f"-> {'FIRES -> long' if z.fires else 'no fire'}")
        L.append(f"  last bar        {z.asof}  ({z.stale_hours:.1f}h ago)")
        L.append(f"  model           {z.arm}, {z.n_train:,} bars"
                 + (f" (+{', '.join(z.aux_used)})" if z.aux_used else ""))
        L.append(f"  if it fires     TP +0.60%  SL -0.50%  max hold 48h")
        L.append("  UPGRADED 2026-09-10 to POOLED training (gold + clean H1 majors):")
        L.append("  walk-forward AUC 0.7333 -> 0.7434 (+0.0101, 4.6x SE), 11/11 years,")
        L.append("  P@recall10 0.651 -> 0.693, P@recall05 0.671 -> 0.723 (V5_FINDINGS 3aw).")
        L.append("  Still walk-forward SR -0.76, DSR 0.000 as a TRADE: better DETECTION of")
        L.append("  the same target, and 3as showed a better detector of it is still worthless.")

    # ---------- 4: regime direction edge ----------
    L.append("")
    L.append("=" * 68)
    L.append("4. REGIME DIRECTION EDGE   [REAL, NOT TRADABLE]")
    L.append("=" * 68)
    h4 = rates(mt5, "XAUUSD", "H4", 12000)
    if h4.empty or len(h4) < 3000:
        L.append("  insufficient H4 data")
    else:
        X = f_base(h4)
        y = (np.log(h4["close"].shift(-6) / h4["close"]) > 0).astype(float)
        y[h4["close"].shift(-6).isna()] = np.nan
        ok = X.notna().all(axis=1) & y.notna()
        i = len(h4) - 1
        tr = ok.values.copy(); tr[max(0, i - 6 - 8):] = False
        if tr.sum() > 2000:
            m = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=8,
                                               l2_regularization=10.0, learning_rate=0.05,
                                               min_samples_leaf=40, random_state=0)
            m.fit(X.values[tr], y.values[tr])
            p_up = float(m.predict_proba(X.values[[i]])[0, 1])
            out["regime"] = dict(p_up=round(p_up, 4), n_train=int(tr.sum()),
                                 last_bar=str(h4.index[-1]))
            L.append(f"  P(next 6 H4 bars UP)  {p_up:.4f}   (~1 trading day)")
            L.append(f"  trained on {int(tr.sum()):,} bars, purged")
            L.append("  Measured edge 51.97% vs a 49.85% persistence baseline = +2.12pp,")
            L.append("  8/9 years, CI [+0.51, +3.69pp]. FIVE trade structures all failed:")
            L.append("  standalone SR -0.44, overlay, confidence gate, 5 horizons, and an")
            L.append("  80-cell SL/TP grid where 79 of 80 were negative. Watch it, do not size it.")
        else:
            L.append("  not enough purged training data")

    # ---------- 5: M15 intrabar path ----------
    L.append("")
    L.append("=" * 68)
    L.append("5. M15 INTRABAR-PATH DOWN-SIGNAL   [REAL, NOT TRADABLE]")
    L.append("=" * 68)
    m15 = rates(mt5, "XAUUSD", "M15", 40000)
    if m15.empty or h4.empty:
        L.append("  M15 feed unavailable from this terminal (needs M15 history)")
    else:
        Xs = f_source(h4, m15)
        yl = first_touch(h4, 30)
        ok = Xs.notna().all(axis=1) & yl.notna()
        i = len(h4) - 1
        tr = ok.values.copy(); tr[max(0, i - 30 - 8):] = False
        if tr.sum() > 1500 and Xs.notna().all(axis=1).iloc[i]:
            m = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=8,
                                               l2_regularization=10.0, learning_rate=0.05,
                                               min_samples_leaf=40, random_state=0)
            m.fit(Xs.values[tr], yl.values[tr])
            p_dn = float(m.predict_proba(Xs.values[[i]])[0, 1])
            out["m15_path"] = dict(p_down_first=round(p_dn, 4), n_train=int(tr.sum()))
            L.append(f"  P(-2% before +2% within 5 days)  {p_dn:.4f}   base rate 0.427")
            L.append(f"  trained on {int(tr.sum()):,} bars, purged")
            L.append("  +0.0142 bits/event above its alignment null's p99, 7/9 years,")
            L.append("  precision 0.600 at recall 0.20. As a trim overlay: 0/16 on the sleeve")
            L.append("  and 0/32 at book level — every cell negative in 2018-21. Do not size it.")
        else:
            L.append("  not enough purged training data")

    L.append("")
    L.append("-" * 68)
    L.append("ONLY the ZigZag harness places orders, on this DEMO account, and it has")
    L.append("negative expectancy by design. Findings 1, 2, 4 and 5 are EMAIL ONLY.")
    L.append("Findings 4 and 5 are real measurements that do NOT convert to P&L — that gap")
    L.append("is the project's central result, not a reason to trade them.")

    body = "\n".join(L)
    print(body)
    out["body"] = body
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(out, indent=2))
    print(f"\nstate -> {STATE.relative_to(ROOT)}")
    mt5.shutdown()


if __name__ == "__main__":
    main()
