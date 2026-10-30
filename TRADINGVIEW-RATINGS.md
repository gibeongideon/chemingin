# TradingView Technical Ratings — replicated, measured, and how to implement

Reference implementation: `scripts/v5_tv_ratings.py`. Measurement and verdict: `V5_FINDINGS.md`
§3cm. Live helper that uses it: `scripts/v5_trade_helper.py`.

---

## 1. What it is

TradingView's "Technical Ratings" box is a **headcount**. Twenty-six indicators each return
**Buy (+1), Sell (−1) or Neutral (0)**, and the summary is

```
rating = (count_buy − count_sell) / 26        # in [−1, +1]
```

mapped to five labels:

| rating | label |
|---|---|
| ≤ −0.5 | Strong Sell |
| −0.5 … −0.1 | Sell |
| −0.1 … +0.1 | Neutral |
| +0.1 … +0.5 | Buy |
| ≥ +0.5 | Strong Buy |

TradingView shows three sub-boxes — Oscillators (11), Moving Averages (15), and the Summary
over all 26.

---

## 2. ⚠️ Read this before you use it

Measured on XAUUSD at H1, H4 and D1 (§3cm), on 67,747 / 18,307 / 3,561 bars:

| what was measured | result |
|---|---|
| **effective number of independent opinions, all 26** | **4.81 – 5.00** |
| **effective opinions in the 15-MA block** | **2.71 – 2.77** |
| effective opinions in the 11 oscillators | 6.03 – 6.43 |
| correlation of the rating with **a single `close vs SMA50` test** | **+0.794 / +0.799 / +0.798** |
| correlation with this repo's champion forecast | +0.44 / +0.46 / +0.50 |
| best rating cell vs **holding gold**, H4 | +4.854bp — but the champion's top bucket is **+6.814bp** |

**"26 indicators agree" is worth about five opinions.** The 15 moving averages — EMA and SMA at
10/20/30/50/100/200 plus VWMA20, HMA9 and Ichimoku — share one input, *where price sits relative
to a trailing average*, and differ only in window length. They agree unanimously on ~12% of bars
at a mean pairwise correlation of 0.50. When price sits below all twelve EMAs/SMAs you do not have
twelve confirmations; you have **one fact counted twelve times**.

Three consequences for use:

1. **Never treat the vote count as evidence strength.** A 22-of-26 consensus is roughly a
   4-of-5 consensus. Any confidence computed from 26 is overstated by about **5×**.
2. **It is an SMA50 proxy.** At ~0.80 correlation, about 63% of its variance is one moving
   average. If you already look at a 50-period MA, the rating is mostly telling you that again.
3. **There is no interior "sweet spot" threshold.** The EV sweep is *monotonic* at H1 and H4 — it
   keeps improving to the highest threshold tested. So "the best setting" is just "the strongest
   consensus", which is "the strongest trend".

**What it is good for:** an objective, reproducible, emotion-free *description* of the current
tape. **What it is not:** an edge. It loses to the champion on the frame the champion is built
for, and its short side runs −7.6 to −8.2bp against simply holding gold.

---

## 3. The 26 components, exactly

All use the standard parameters TradingView ships. `sgn3(buy, sell)` returns +1 / −1 / 0.

### 3.1 The 11 oscillators

| # | indicator | Buy | Sell |
|---|---|---|---|
| 1 | **RSI (14)** | `RSI < 30` **and** rising | `RSI > 70` **and** falling |
| 2 | **Stochastic %K (14,3,3)** | `K < 20` and `D < 20` and `K > D` | `K > 80` and `D > 80` and `K < D` |
| 3 | **CCI (20)** | `CCI < −100` and rising | `CCI > 100` and falling |
| 4 | **ADX (14)** | `ADX > 20` and `+DI > −DI` and `+DI` rising | `ADX > 20` and `+DI < −DI` and `+DI` falling |
| 5 | **Awesome Oscillator** | `AO > 0` and (crossed up **or** rising) | `AO < 0` and (crossed down **or** falling) |
| 6 | **Momentum (10)** | `mom > mom[1]` | `mom < mom[1]` |
| 7 | **MACD (12,26,9)** | `macd > signal` | `macd < signal` |
| 8 | **Stochastic RSI (3,3,14,14)** | `K < 20` and `D < 20` and `K > D` | `K > 80` and `D > 80` and `K < D` |
| 9 | **Williams %R (14)** | `%R < −80` and rising | `%R > −20` and falling |
| 10 | **Bull Bear Power (13)** | `BBP > 0` and rising | `BBP < 0` and falling |
| 11 | **Ultimate Oscillator (7,14,28)** | `UO > 70` | `UO < 30` |

Definitions that are easy to get wrong:

- **Awesome Oscillator** = `SMA(hl2, 5) − SMA(hl2, 34)`, where `hl2 = (high + low) / 2`.
- **Bull Bear Power (13)** = `(high − EMA13) + (low − EMA13)`.
- **Ultimate Oscillator**: with `BP = close − min(low, prev_close)` and
  `TR = max(high, prev_close) − min(low, prev_close)`, and `avg(n) = ΣBP(n) / ΣTR(n)`, the value is
  `100 × (4·avg(7) + 2·avg(14) + avg(28)) / 7`.
- **Stochastic RSI** is a stochastic **of the RSI series**, not of price.
- **ADX** uses Wilder smoothing (`ewm(alpha=1/n, adjust=False)`), not a simple mean.

### 3.2 The 15 moving averages

Twelve of them vote on one rule — **Buy if `close > MA`, Sell if `close < MA`**:

> EMA 10, 20, 30, 50, 100, 200 · SMA 10, 20, 30, 50, 100, 200

Plus three with their own construction, same price-vs-line rule for the first two:

| # | indicator | definition |
|---|---|---|
| 13 | **VWMA (20)** | `Σ(close·volume, 20) / Σ(volume, 20)` |
| 14 | **HMA (9)** | `WMA(2·WMA(close, n/2) − WMA(close, n), √n)` with `n = 9` |
| 15 | **Ichimoku** | uses the **cloud** rule, below |

**Ichimoku (9, 26, 52, 26).** Conversion `= (HH9 + LL9)/2`, Base `= (HH26 + LL26)/2`,
LeadA `= ((Conversion + Base)/2)` shifted forward 26, LeadB `= ((HH52 + LL52)/2)` shifted forward 26.

- **Buy**: `LeadA > LeadB` and `close > LeadA` and `close > Base` and `Conversion > Base`
- **Sell**: the exact mirror

> Do **not** shortcut Ichimoku to a plain price-vs-Base-line test. It makes it behave like the
> other twelve MAs and pushes the measured effective rank *down* — i.e. it manufactures support
> for the "they're all the same" conclusion. `v5_tv_ratings.py` uses the cloud rule deliberately.

---

## 4. How to implement it

### 4.1 Data

Everything is OHLCV on one timeframe. Two rules that have each cost this repo a retracted finding:

- **Single provenance.** Build every frame from one feed. `src/v5/mtf_frames.py::aggregate`
  rebuilds M30/H1/H4 from M15; D1 is a `groupby(index.normalize())` with
  first-open / max-high / min-low / last-close. Never mix an MT5 intraday file with the Yahoo
  daily panel (`multi-broker dataset is a leak`).
- **`real_volume` is 0.0% populated** in this feed, so VWMA uses `tick_volume`. That makes it a
  tick-count weighting, not a volume weighting. Say so rather than implying otherwise.

### 4.2 Code

```python
from scripts.v5_tv_ratings import votes          # -> DataFrame, 26 columns of +1/-1/0
W      = votes(df)                               # df needs open/high/low/close[/tick_volume]
rating = W.mean(axis=1)                          # the TradingView summary, in [-1, +1]
buys   = (W ==  1).sum(axis=1)
sells  = (W == -1).sum(axis=1)
```

`votes()` drops leading NaNs, so it needs **at least 200 bars of warm-up** (SMA200 and the
Ichimoku 52-period lookback shifted 26 forward are the binding ones). Feed it ~300 bars minimum.

### 4.3 Reporting it honestly

Alongside any rating, print the two numbers that stop it being misread:

```python
from scripts.v5_tv_ratings import eff_rank, MAS
pr, n90, _ = eff_rank(W)                 # participation ratio: the effective opinion count
ma_unanimous = (W[MAS].mean(axis=1).abs() == 1)
```

Report **"18 of 26 buy — but these 26 carry ~5 independent opinions"**, never the raw count alone.

### 4.4 Verifying against TradingView

Compare on a liquid symbol and the same timeframe, and expect **component-level** agreement, not
identity of the summary: TradingView computes on its own feed, its bars close on its own clock, and
its gold is a different quote from any broker's (Switch and FundingPips disagreed by **10 cents**
on the same prior-day high on 2026-10-30). Check the 12 plain MAs first — they are unambiguous and
will match exactly if your data is aligned. If the MAs match and an oscillator does not, the bug is
in that oscillator's rule, not in the data.

---

## 5. Honest summary

The replication is faithful and cheap to run, and the measurement is the useful part: **this is a
~5-opinion indicator wearing 26 hats, about 80% explained by one moving average, with no interior
optimum and a short side that loses badly to holding.** Use it as a disciplined description of the
tape — which is exactly what `scripts/v5_trade_helper.py` does with it — and not as a reason to
take a trade.
