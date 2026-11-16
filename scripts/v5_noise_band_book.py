#!/usr/bin/env python3
"""RANK #2 APPLIED TO THE INTRADAY EFFECT (V5_FINDINGS 3ex, part 2b). 3ef ran the noise-band
intraday momentum on gold only; the effect was published on equity indices (Zarattini, Aziz &
Barbon 2024). This runs 3ef's NY cell UNCHANGED (its own `positions`) on the instruments Switch
quotes cheaply and builds the equal-weight intraday book.

Cells (pre-registered, data/v5_runs/intraday_day_signal/PREREGISTRATION.md, commit 264ebb8):
{NQ, ES, BTC} x {LS, LO} + BOOK_LS + BOOK_LO, BOOK = mean of the 1/vol sleeves GC + NQ + BTC.
GC is 3ef's known answer (+0.25 / +0.19 net on CME) and must reproduce.
Cost round trip: GC $0.30 (3ef), NQ 1.0bp (Switch live 0.64), ES 2.0bp (1.94), BTC 5.0bp (4.08);
2x as the stress. Null = 3ef's day-shift null (positions rolled >= 20 sessions; ONE joint shift
for the book), max-stat over the 8 cells.

    python scripts/v5_noise_band_book.py cme        # CME H1 2010-2026, 60-min checks
    python scripts/v5_noise_band_book.py venue      # Switch USTECs M15 2024-26, Binance BTC M15 2017-26, gold M15 2015-26
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
from scripts.v5_noise_band_momentum import matrices, positions  # noqa: E402

OUT = ROOT / "data" / "v5_runs" / "intraday_day_signal"
NY, N_NULL = "America/New_York", 1000
COST_BP = {"NQ": 1.0, "ES": 2.0, "BTC": 5.0}                    # round trip; GC is $0.30
BOOK = ("GC", "NQ", "BTC")


def sharpe(x) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / x.std() * np.sqrt(252)) if len(x) > 2 and x.std() > 0 else 0.0


def load(src: str, sym: str) -> tuple[pd.DataFrame, int, int]:
    if src == "cme":
        from scripts.v5_zigzag_panel_probs import load_cme  # noqa: PLC0415
        d, step, check = load_cme(sym), 60, 60
        d.index = d.index.tz_localize("UTC").tz_convert(NY).tz_localize(None)
    elif sym == "GC":
        d = pd.read_csv(ROOT / "data/XAUUSD_M15_long.csv", parse_dates=["time"], index_col="time")
        d.index = d.index.tz_localize("UTC").tz_convert(NY).tz_localize(None)
        step, check = 15, 30
    elif sym == "NQ":
        d = pd.read_csv(ROOT / "data/sweep_switch/USTECs_M15.csv", parse_dates=["time"]).set_index("time").sort_index()
        d.index = d.index - pd.Timedelta(hours=7)                # Switch server = New York + 7h
        step, check = 15, 30
    else:
        d = pd.read_parquet(ROOT / "data/binance_1m/BTCUSDT_M15.parquet").rename(columns=dict(o="open", h="high", l="low", c="close"))
        d.index = d.index.tz_localize("UTC").tz_convert(NY).tz_localize(None)
        d = d[d.index.dayofweek < 5]                             # weekdays only: the NY session of the paper
        step, check = 15, 30
    d = d[~d.index.duplicated(keep="last")].sort_index()[["open", "high", "low", "close"]]
    return d, step, check


def sleeve(src: str, sym: str) -> dict:
    d, step, check = load(src, sym)
    O, C = matrices(d, step, 8 * 60, 8 * 60)
    c = C.ffill(axis=1).to_numpy()
    ret = np.zeros_like(c)
    ret[:, 1:] = np.nan_to_num(c[:, 1:] / c[:, :-1] - 1)
    sess = c[:, -1] / O[0].to_numpy() - 1
    vol = pd.Series(sess).ewm(span=20, min_periods=10).std().shift(1).to_numpy()
    w = np.minimum(np.nan_to_num(0.01 / vol, nan=0.0, posinf=0.0), 4.0)
    px = np.nanmedian(c, axis=1)
    frac = (0.30 / px) if sym == "GC" else np.full(len(px), COST_BP[sym] * 1e-4)   # round trip as a fraction
    return dict(sym=sym, dates=pd.DatetimeIndex(C.index), ret=ret, w=w, frac=frac, base=positions(O, C, step, check),
                sess=sess)


def pnl(s: dict, pos: np.ndarray, mult: float = 1.0) -> np.ndarray:
    gross = (pos[:, :-1] * s["ret"][:, 1:]).sum(axis=1)
    dpos = np.abs(np.diff(np.column_stack([np.zeros(len(pos)), pos]), axis=1)).sum(axis=1)
    return s["w"] * (gross - dpos * mult * s["frac"] / 2)


def book(series: dict[str, pd.Series]) -> pd.Series:
    """Equal-weight mean over the sleeves that exist on each date."""
    return pd.concat(series, axis=1).mean(axis=1, skipna=True).dropna()


def main() -> None:
    src = sys.argv[1]
    syms = ("GC", "NQ", "ES", "BTC") if src == "cme" else ("GC", "NQ", "BTC")
    S = {k: sleeve(src, k) for k in syms}
    rng = np.random.default_rng(3)
    rows, daily, null = [], {}, {}
    shifts = rng.integers(20, 2000, N_NULL)
    for mode in ("LS", "LO"):
        P = {k: (S[k]["base"] if mode == "LS" else np.maximum(S[k]["base"], 0.0)) for k in syms}
        for k in syms:
            s = S[k]
            net, gross, net2 = pnl(s, P[k]), pnl(s, P[k], 0.0), pnl(s, P[k], 2.0)
            hold = np.ones_like(P[k]); hold[:, -1] = 0.0
            hr = pnl(s, hold)
            name = f"{k}_{mode}"
            daily[name] = pd.Series(net, index=s["dates"])
            nl = np.array([sharpe(pnl(s, np.roll(P[k], int(j) % (len(P[k]) - 40) + 20, axis=0))) for j in shifts])
            null[name] = nl
            half, yrs = len(net) // 2, s["dates"].year.to_numpy()
            ntr = (np.abs(np.diff(np.column_stack([np.zeros(len(P[k])), P[k]]), axis=1)) > 0).sum() / 2
            dd = net * (hr.std() / net.std()) - hr if net.std() > 0 else net - hr
            rows.append(dict(cell=name, sessions=len(net), first=str(s["dates"][0].date()), trades_wk=round(ntr / (len(net) / 5), 1),
                             in_mkt=round(float(np.abs(P[k]).mean()), 2), SR_gross=round(sharpe(gross), 2),
                             SR=round(sharpe(net), 2), SR_2x=round(sharpe(net2), 2), h1=round(sharpe(net[:half]), 2),
                             h2=round(sharpe(net[half:]), 2),
                             yrs_pos=f"{sum(sharpe(net[yrs == y]) > 0 for y in np.unique(yrs))}/{len(np.unique(yrs))}",
                             null_med=round(float(np.median(nl)), 2), null95=round(float(np.percentile(nl, 95)), 2),
                             p=round(float((nl >= sharpe(net)).mean()), 3), SR_hold=round(sharpe(hr), 2),
                             t_vs_hold=round(float(dd.mean() / dd.std() * np.sqrt(len(dd))), 2) if mode == "LO" else np.nan))
        # the book: one joint shift for all sleeves
        bk = {m: book({k: pd.Series(pnl(S[k], P[k], m), index=S[k]["dates"]) for k in BOOK}) for m in (1.0, 0.0, 2.0)}
        name = f"BOOK_{mode}"
        daily[name] = bk[1.0]
        nl = np.array([sharpe(book({k: pd.Series(pnl(S[k], np.roll(P[k], int(j) % (len(P[k]) - 40) + 20, axis=0)), index=S[k]["dates"])
                                    for k in BOOK})) for j in shifts])
        null[name] = nl
        b = bk[1.0]
        half, yrs = len(b) // 2, b.index.year.to_numpy()
        eq = (1 + b).cumprod()
        rows.append(dict(cell=name, sessions=len(b), first=str(b.index[0].date()), trades_wk=np.nan, in_mkt=np.nan,
                         SR_gross=round(sharpe(bk[0.0]), 2), SR=round(sharpe(b), 2), SR_2x=round(sharpe(bk[2.0]), 2),
                         h1=round(sharpe(b[:half]), 2), h2=round(sharpe(b[half:]), 2),
                         yrs_pos=f"{sum(sharpe(b[yrs == y]) > 0 for y in np.unique(yrs))}/{len(np.unique(yrs))}",
                         null_med=round(float(np.median(nl)), 2), null95=round(float(np.percentile(nl, 95)), 2),
                         p=round(float((nl >= sharpe(b)).mean()), 3), SR_hold=np.nan, t_vs_hold=np.nan))
        if mode == "LS":
            cc = pd.concat({k: pd.Series(pnl(S[k], P[k]), index=S[k]["dates"]) for k in syms}, axis=1).corr().round(2)
    A = pd.DataFrame(rows)
    fam = [c for c in A.cell if not c.startswith("GC")]
    mx = np.vstack([null[c] for c in fam]).max(axis=0)
    best = A[A.cell.isin(fam)].sort_values("SR").iloc[-1]
    b = daily["BOOK_LS"]
    eq = (1 + b).cumprod()
    pd.DataFrame(daily).to_csv(OUT / f"nb_book_{src}_daily.csv")
    txt = "\n".join([
        f"NOISE-BAND INTRADAY MOMENTUM ACROSS INSTRUMENTS, source {src.upper()}, NY 08:00-16:00 ET, 3ef rule unchanged",
        f"cost round trip: GC $0.30, NQ {COST_BP['NQ']}bp, ES {COST_BP['ES']}bp, BTC {COST_BP['BTC']}bp; SR_2x = double cost",
        "", A.to_string(index=False), "",
        "sleeve correlation (LS, net daily):", cc.to_string(), "",
        f"BOOK_LS at the sleeves' 1% vol target: ann vol {100 * b.std() * np.sqrt(252):.1f}%, maxDD {100 * (eq / eq.cummax() - 1).min():.1f}%, "
        f"worst day {100 * b.min():+.2f}%",
        f"FAMILY ({len(fam)} new cells; GC excluded as the known answer): best {best.cell} SR {best.SR:+.2f} vs day-shift null max "
        f"median {np.median(mx):+.2f}, 95th {np.percentile(mx, 95):+.2f}, p {(mx >= best.SR).mean():.3f}"])
    print(txt)
    (OUT / f"nb_book_{src}_report.txt").write_text(txt + "\n")


if __name__ == "__main__":
    sys.exit(main())
