#!/usr/bin/env python3
"""P&L of the XAU advisor's signals under the LIVE executor's own rules (magic 360602).

The advisor was validated as a first-touch PATH frequency (§3ba), never as money. This replays
the executor exactly — `decide()` is imported from `scripts/v5_xau_advisor_trader.py`, not
re-written — on Switch H1 bars:

  * signal re-scored on each closed H4 bar; acted on at the open of the H1 bar at its close
  * entry pays a fixed $0.25 spread (Switch quotes 0.21-0.28; its spread column is 0 pre-2025)
  * SL 1.0 x H4 ATR20 at entry, ratcheted 1.0 x ATR behind the best price, checked on H1 lows/
    highs BEFORE ratcheting on the same bar (no intrabar optimism)
  * on a flip the opposite position is closed if in profit (incl. swap), else kept as a hedge
  * one entry per episode, at most one position per side
  * swap: long -6%/yr, short +2.5%/yr of notional (repo carry assumptions)
  P&L in R (multiples of initial risk); at the live 0.5% risk, 1R = 0.5% of equity.

RULES
  BIAS        raw P(down-first) < 0.5 -> BUY, else SELL         (what trades live now)
  STRONG      the 3-state machine (step(): hysteresis + 9h dwell): UP->BUY, DOWN->SELL
  STRONG_LONG UP->BUY; DOWN closes a profitable BUY (the card's own "trim" advice), never sells

NULL. Each rule is re-run on its own signal series CIRCULARLY SHIFTED by a random offset (200
draws): same mechanics, same long/short mix, same episode lengths, timing destroyed. A rule whose
total R does not beat the shift p95 has no timing skill, whatever its sign.

OUT OF SAMPLE = after the artifact's `data.h4_last_bar` (the model never saw those bars).

    python scripts/v5_advisor_pnl.py --fetch      # on the VPS, under the bridge lock
    python scripts/v5_advisor_pnl.py              # simulate from the saved bars
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_xau_advisor_trader import BUY, SELL, Pos, decide  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "advisor_pnl"
SPREAD, SWAP_L, SWAP_S = 0.25, -0.06, 0.025
SL_ATR = TRAIL_ATR = 1.0
N_NULL, SEED = 200, 7


def fetch(port: int, sym: str) -> None:
    from mt5linux import MetaTrader5  # noqa: PLC0415
    m = MetaTrader5(host="localhost", port=port)
    if not m.initialize():
        sys.exit(f"init failed {m.last_error()}")
    m.symbol_select(sym, True)
    OUT.mkdir(parents=True, exist_ok=True)
    for tf, name, n in ((16388, "H4", 30000), (16385, "H1", 99999)):
        r = None
        while n >= 2000 and (r is None or len(r) == 0):
            r = m.copy_rates_from_pos(sym, tf, 1, n)       # pos 1: skip the forming bar
            n //= 2
        d = pd.DataFrame([tuple(x)[:7] for x in r],
                         columns=["ts", "open", "high", "low", "close", "tick_volume", "spread"])
        d["time"] = pd.to_datetime(d.ts, unit="s")
        d.drop(columns="ts").to_csv(OUT / f"{sym}_{name}.csv", index=False)
        print(name, len(d), d.time.iloc[0], "->", d.time.iloc[-1])
    # NO m.shutdown(): bridge 18815 is shared by every Switch job.


def score(h4: pd.DataFrame) -> pd.DataFrame:
    from scripts.v5_advisor_measure import features_for  # noqa: PLC0415
    from src.v5.xau_advisor_model import load  # noqa: PLC0415
    art = load()
    a, b = art["meta"]["adverse"], art["blobs"]["adverse"]
    X = features_for(a["arm"], h4, h4)[b["features"]].dropna()
    v = b["scaler"].transform(X.values) if b.get("scaler") is not None else X.values
    raw = b["model"].predict_proba(v)[:, 1]
    cal = b["calibrator"]
    p = (cal.predict(raw) if hasattr(cal, "predict") and not hasattr(cal, "predict_proba")
         else cal.predict_proba(raw.reshape(-1, 1))[:, 1])
    return pd.DataFrame({"raw": raw, "p": p}, index=X.index), art["meta"]


def signals(sc: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Per H4 bar: wanted 'state name' for each rule, and an episode id."""
    from src.v5.xau_advisor_state import AdvisorState, Thresholds, step  # noqa: PLC0415
    adv = meta["adverse"]
    th = Thresholds(down_at=float(adv["down_at"]), down_release=float(adv["down_release"]),
                    up_at=float(adv["up_at"]), up_release=float(adv["up_release"]),
                    base_rate=float(adv["measured"]["base_rate"]),
                    min_dwell_h=float(adv.get("hours", 9)))
    st, names = AdvisorState(), []
    for t, p in sc.p.items():
        now = (t + pd.Timedelta(hours=4)).to_pydatetime()
        st = step(st, float(p), t.to_pydatetime(), th, now)
        names.append(st.name)
    s = pd.DataFrame(index=sc.index)
    s["BIAS"] = np.where(sc.raw < 0.5, "UP", "DOWN")
    s["STRONG"] = names
    s["STRONG_LONG"] = names
    return s


def simulate(h1: pd.DataFrame, atr: pd.Series, sig: pd.Series, long_only: bool) -> list:
    """Replay the executor. Returns closed trades [(t_exit, side, R)]."""
    t1 = h1.index.to_numpy()
    o, hi, lo = (h1[c].to_numpy() for c in ("open", "high", "low"))
    # map each decision (H4 close) to the H1 bar that opens at it
    dec_t = (sig.index + pd.Timedelta(hours=4)).to_numpy()
    di = np.searchsorted(t1, dec_t)
    act = {}
    prev, ep = None, 0
    for k, (i, nm) in enumerate(zip(di, sig.to_numpy())):
        if nm != prev:
            ep += 1
            prev = nm
        if i < len(t1):
            act[int(i)] = (nm, ep, float(atr.iloc[k]))
    pos, trades, last_ep, cur_atr = [], [], None, np.nan
    for i in range(len(t1)):
        # 1) stops, on this bar's range, with the stop as it stood at the bar's open
        keep = []
        for p in pos:
            hit = lo[i] <= p["sl"] if p["side"] == BUY else hi[i] + SPREAD >= p["sl"]
            if hit:
                trades.append(_close(p, p["sl"], t1[i]))
            else:
                keep.append(p)
        pos = keep
        # 2) signal at this bar's open
        if i in act:
            nm, ep, cur_atr = act[i]
            if long_only and nm == "DOWN":
                rest = []
                for p in pos:
                    if _pnl(p, o[i], t1[i]) > 0:
                        trades.append(_close(p, o[i], t1[i]))
                    else:
                        rest.append(p)
                pos, nm = rest, "NEUTRAL"
            P = [Pos(ticket=j, type=p["side"], volume=1, price_open=p["entry"], sl=p["sl"],
                     pnl=_pnl(p, o[i], t1[i])) for j, p in enumerate(pos)]
            d = decide(nm, ep, last_ep, P)
            if d["close"]:
                for j in sorted(d["close"], reverse=True):
                    trades.append(_close(pos[j], o[i], t1[i]))
                pos = [p for j, p in enumerate(pos) if j not in d["close"]]
            if d["open"] is not None and np.isfinite(cur_atr) and cur_atr > 0:
                side = d["open"]
                entry = o[i] + SPREAD if side == BUY else o[i]
                risk = SL_ATR * cur_atr
                sl = entry - risk if side == BUY else entry + risk
                pos.append(dict(side=side, entry=entry, sl=sl, risk=risk, t0=t1[i],
                                best=entry))
                last_ep = ep
        # 3) ratchet on this bar's extreme (takes effect from the next bar)
        for p in pos:
            if p["side"] == BUY:
                p["best"] = max(p["best"], hi[i])
                p["sl"] = max(p["sl"], p["best"] - TRAIL_ATR * cur_atr)
            else:
                p["best"] = min(p["best"], lo[i])
                p["sl"] = min(p["sl"], p["best"] + TRAIL_ATR * cur_atr)
    for p in pos:
        trades.append(_close(p, o[-1], t1[-1]))
    return trades


def _swap(p, t) -> float:
    days = (t - p["t0"]) / np.timedelta64(1, "D")
    rate = SWAP_L if p["side"] == BUY else SWAP_S
    return p["entry"] * rate * days / 365


def _pnl(p, px, t) -> float:
    g = px - p["entry"] if p["side"] == BUY else p["entry"] - (px + SPREAD)
    return g + _swap(p, t)


def _close(p, px, t) -> tuple:
    if p["side"] == SELL and px == p["sl"]:
        g = p["entry"] - px                     # stop is on the ask; px already the ask level
    else:
        g = px - p["entry"] if p["side"] == BUY else p["entry"] - (px + SPREAD)
    return (pd.Timestamp(t), "BUY" if p["side"] == BUY else "SELL",
            (g + _swap(p, t)) / p["risk"])


def summarise(tr: list, start=None) -> dict:
    t = pd.DataFrame(tr, columns=["t", "side", "R"])
    if start is not None:
        t = t[t.t > start]
    if not len(t):
        return dict(n=0, R=0.0, avgR=np.nan, win=np.nan, t=np.nan, buyR=0.0, sellR=0.0)
    r = t.R
    return dict(n=len(t), R=round(r.sum(), 1), avgR=round(r.mean(), 3),
                win=round((r > 0).mean(), 3),
                t=round(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))), 2) if len(r) > 2 else np.nan,
                buyR=round(r[t.side == "BUY"].sum(), 1), sellR=round(r[t.side == "SELL"].sum(), 1),
                pct_at_05=round(r.sum() * 0.5, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--port", type=int, default=18815)
    ap.add_argument("--symbol", default="XAUUSDs")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.port, a.symbol)
        return
    h4 = pd.read_csv(OUT / f"{a.symbol}_H4.csv", parse_dates=["time"]).set_index("time")
    h1 = pd.read_csv(OUT / f"{a.symbol}_H1.csv", parse_dates=["time"]).set_index("time")
    sc, meta = score(h4)
    oos = pd.Timestamp(meta["data"]["h4_last_bar"])
    pc = h4.close.shift(1)
    tr = pd.concat([h4.high - h4.low, (h4.high - pc).abs(), (h4.low - pc).abs()], axis=1).max(1)
    atr = tr.rolling(20).mean().reindex(sc.index)
    sig = signals(sc, meta)
    h1 = h1[h1.index >= sc.index[0]]
    yrs_all = (h1.index[-1] - h1.index[0]).days / 365.25
    yrs_oos = (h1.index[-1] - oos).days / 365.25
    bh_all = h1.close.iloc[-1] / h1.close.iloc[0] - 1
    bh_oos = h1.close.iloc[-1] / h1.close[h1.index > oos].iloc[0] - 1
    lines = [f"Switch XAUUSDs H1 {h1.index[0]} -> {h1.index[-1]} ({yrs_all:.1f}y); "
             f"model {meta.get('model_id')} trained through {oos}  (OOS {yrs_oos:.2f}y)",
             f"buy & hold: full {100*bh_all:+.1f}%   OOS {100*bh_oos:+.1f}%",
             f"time in each 3-state: {sig.STRONG.value_counts(normalize=True).round(3).to_dict()}",
             ""]
    rng = np.random.default_rng(SEED)
    rows = []
    for rule in ("BIAS", "STRONG", "STRONG_LONG"):
        lo_ = rule == "STRONG_LONG"
        trd = simulate(h1, atr, sig[rule], lo_)
        f, o = summarise(trd), summarise(trd, oos)
        null_f, null_o = [], []
        for _ in range(N_NULL):
            k = int(rng.integers(50, len(sig) - 50))
            sh = pd.Series(np.roll(sig[rule].to_numpy(), k), index=sig.index)
            nt = simulate(h1, atr, sh, lo_)
            null_f.append(summarise(nt)["R"])
            null_o.append(summarise(nt, oos)["R"])
        pf = float((np.array(null_f) >= f["R"]).mean())
        po = float((np.array(null_o) >= o["R"]).mean())
        rows.append(dict(rule=rule, span="FULL", **f, null_p50=round(np.median(null_f), 1),
                         null_p95=round(np.percentile(null_f, 95), 1), p=round(pf, 3)))
        rows.append(dict(rule=rule, span="OOS", **o, null_p50=round(np.median(null_o), 1),
                         null_p95=round(np.percentile(null_o, 95), 1), p=round(po, 3)))
        print(rule, "done", flush=True)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    txt = "\n".join(lines) + pd.DataFrame(rows).to_string(index=False) + (
        "\n\nR = multiples of initial risk; pct_at_05 = % of equity at the live 0.5% risk."
        "\np = share of 200 timing-shuffled replays (same rule, same mechanics) with R >= this.\n")
    print(txt)
    (OUT / "report.txt").write_text(txt)


if __name__ == "__main__":
    sys.exit(main())
