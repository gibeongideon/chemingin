"""Claude-as-analyst BACKTEST on XAUUSD H4 — does an LLM add tradeable edge?

Screenshot-style "AI Agent Strategy": feed Claude the market metrics for each
closed H4 bar, it reasons and returns a directional call, we simulate the trade
NET OF THE REAL SPREAD, and measure whether it beats buy&hold. This is the
MEASURE-FIRST harness the live ai_bot (src/bots/ai_bot.py) never had — no live
100K wiring until the number justifies it.

Honest prior (V5_FINDINGS §1/§3, this repo): intraday XAU direction dies net of
spread three times over. H4 is the least-doomed horizon (spread ~3% of a typical
bar, vs 15-25% on M15) and where LLM regime reasoning has a chance — hence H4.
The output is a hit-rate and a net-of-cost P&L vs buy&hold and always-long, over
a bounded, cost-capped window. Treat a marginal result as noise, not signal.

Cost control: claude-opus-5, response cache keyed by context hash (resumable,
re-runs are free), prompt-cached system prompt. ~150 calls ≈ a few dollars.

    python scripts/v5_claude_h4_backtest.py --n 12      # smoke test first
    python scripts/v5_claude_h4_backtest.py --n 150     # the real run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import anthropic  # noqa: E402

MODEL = "claude-opus-5"
XAU_SPREAD_USD = 0.45          # live FTMO gold spread, one-way is half of this
CACHE = ROOT / "data" / "v5_runs" / "claude_h4_cache.json"

SYSTEM = (
    "You are the analyst on a systematic gold (XAUUSD) trading desk. You are given "
    "the metrics of the just-closed H4 candle and recent price action. Decide the "
    "direction for the NEXT H4 bar: buy (expect up), sell (expect down), or hold "
    "(no clear edge). Judge trend, momentum, mean-reversion pressure, and volatility "
    "regime. Be decisive but honest: most bars have no edge — prefer 'hold' unless "
    "the setup is genuinely favourable. You see only past data; there is no lookahead. "
    "Reply strictly in the required JSON schema."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["buy", "sell", "hold"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reason": {"type": "string"},
    },
    "required": ["action", "confidence", "reason"],
    "additionalProperties": False,
}


def rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(close, prepend=close[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1 / n, adjust=False).mean().values
    rd = pd.Series(dn).ewm(alpha=1 / n, adjust=False).mean().values
    rs = ru / np.where(rd == 0, np.nan, rd)
    return 100 - 100 / (1 + rs)


def build_context(df: pd.DataFrame, i: int, lookback: int = 30) -> str:
    """Causal context ending at bar i (the just-closed bar)."""
    w = df.iloc[max(0, i - lookback + 1): i + 1]
    c = w["close"].values
    r = rsi(df["close"].values[: i + 1])[-1]
    sma20 = df["close"].values[max(0, i - 19): i + 1].mean()
    sma50 = df["close"].values[max(0, i - 49): i + 1].mean()
    trend = "BULLISH" if sma20 > sma50 else "BEARISH"
    atr = (w["high"] - w["low"]).tail(14).mean()
    t = df.index[i]
    recent = "\n".join(
        f"  {ts:%Y-%m-%d %H:%M}  O{row.open:.2f} H{row.high:.2f} L{row.low:.2f} C{row.close:.2f}"
        for ts, row in w.tail(10).iterrows()
    )
    return (
        f"XAUUSD H4  as of {t:%Y-%m-%d %H:%M} (bar just closed)\n"
        f"Close: {c[-1]:.2f}\n"
        f"RSI(14): {r:.1f}{' overbought' if r > 70 else ' oversold' if r < 30 else ''}\n"
        f"SMA20: {sma20:.2f}  SMA50: {sma50:.2f}  MA-trend: {trend}\n"
        f"ATR(14): {atr:.2f}  ({atr / c[-1] * 100:.2f}% of price)\n"
        f"30-bar range: {w['low'].min():.2f} — {w['high'].max():.2f}\n"
        f"5-bar change: {(c[-1] / c[-6] - 1) * 100:+.2f}%\n"
        f"day-of-week: {t:%A}\n"
        f"Last 10 H4 candles:\n{recent}"
    )


def load_cache() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def ask_claude(client, context: str, cache: dict) -> dict:
    key = hashlib.sha256((MODEL + "|" + context).encode()).hexdigest()
    if key in cache:
        return cache[key]
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            thinking={"type": "adaptive"},
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": context}],
        )
        if resp.stop_reason == "refusal":
            out = {"action": "hold", "confidence": "low", "reason": "refusal"}
        else:
            text = next((b.text for b in resp.content if b.type == "text"), "{}")
            out = json.loads(text)
    except Exception as e:  # noqa: BLE001
        out = {"action": "hold", "confidence": "low", "reason": f"error:{type(e).__name__}"}
    cache[key] = out
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=150, help="number of H4 decision bars")
    ap.add_argument("--min-conf", default="medium", choices=["high", "medium"],
                    help="act only at >= this confidence")
    args = ap.parse_args()

    df = pd.read_csv(f"{ROOT}/data/XAUUSD_H4_long.csv", parse_dates=["time"]).set_index("time")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    N = args.n
    start = len(df) - N - 1                       # decide at close of bars [start .. len-2]
    client = anthropic.Anthropic()
    cache = load_cache()

    print(f"Claude H4 backtest — model {MODEL}, {N} decision bars, "
          f"{df.index[start]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}, "
          f"spread ${XAU_SPREAD_USD}/oz, act>= {args.min_conf}\n")
    print(f"{'bar':>3} {'time':16} {'action':6} {'conf':6} {'next%':>7}  reason")

    rows = []
    for k in range(N):
        i = start + k                            # just-closed bar
        ctx = build_context(df, i)
        d = ask_claude(client, ctx, cache)
        act = d["action"]
        conf = d["confidence"]
        pos = (1 if act == "buy" else -1 if act == "sell" else 0)
        gate = conf == "high" or (conf == "medium" and args.min_conf == "medium")
        if not gate:
            pos = 0
        nxt = df["close"].iloc[i + 1] / df["close"].iloc[i] - 1     # bar i+1 close-to-close
        rows.append(dict(t=df.index[i], pos=pos, ret=nxt, act=act, conf=conf))
        if k < 20 or act != "hold":
            print(f"{k:>3} {df.index[i]:%Y-%m-%d %H:%M} {act:6} {conf:6} "
                  f"{nxt*100:+7.2f}  {d['reason'][:60]}")

    R = pd.DataFrame(rows)
    spread_frac = XAU_SPREAD_USD / df["close"].iloc[start:].mean()
    turn = R["pos"].diff().abs().fillna(R["pos"].abs())
    R["net"] = R["pos"] * R["ret"] - turn * spread_frac
    traded = R[R["pos"] != 0]
    hit = float((np.sign(traded["pos"]) == np.sign(traded["ret"])).mean() * 100) if len(traded) else float("nan")
    tot = float((1 + R["net"]).prod() - 1) * 100
    bh = float((1 + R["ret"]).prod() - 1) * 100
    always_long = float((1 + R["ret"] - (R["ret"].abs() * 0)).prod() - 1) * 100  # buy&hold ref
    gross = float((R["pos"] * R["ret"]).sum() * 100)

    print("\n" + "=" * 60)
    print(f"decisions: {N}   traded bars: {len(traded)}   "
          f"buys {int((R.act=='buy').sum())} sells {int((R.act=='sell').sum())} "
          f"holds {int((R.act=='hold').sum())}")
    print(f"directional HIT RATE (traded bars): {hit:.1f}%   (50% = coin flip)")
    print(f"gross directional sum: {gross:+.2f}%   turnover trades: {int((turn>0).sum())}")
    print(f"NET return (after ${XAU_SPREAD_USD} spread): {tot:+.2f}%")
    print(f"buy & hold same window:              {bh:+.2f}%")
    print("=" * 60)
    print("VERDICT: net must beat buy&hold AND hit-rate must clear ~52-53% "
          "(to overcome cost) for this to be real. Small N = wide error bars.")
    out = ROOT / "data" / "v5_runs" / "claude_h4_decisions.csv"
    R.to_csv(out, index=False)
    print(f"decisions -> {out}")


if __name__ == "__main__":
    main()
