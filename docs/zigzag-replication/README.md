# ZigZag variant B: replicating the walk-forward Sharpe

This package rebuilds the ZigZag gold strategy from raw hourly prices and reproduces its walk-forward result:

- **Sharpe +1.58** on XAUUSD H1, January 2024 to November 2026.
- **510 trades**, which is 3.44 per week.

Everything needed is in this folder:

| File | What it is |
|---|---|
| `replicate.py` | The whole pipeline in one file (~280 lines). No other project code is needed. |
| `data/*.csv` | Four H1 price files from a retail MT5 broker feed. |
| `expected_output.txt` | The exact console output you should get. |
| `expected_trades.csv` | All 510 trades, for a line-by-line diff. |
| `expected_probs.csv` | The model probability for every gold bar, to check the model on its own. |
| `tests/test_replicate.py` | 13 pytest checks; see "Tests" below. |
| `repo_code/` | The original project code this was extracted from, including the live trading bot. See "Original code" below. |
| `requirements.txt` | Library versions the result was produced with. |

The section numbers below (§1 to §7) match the comments in `replicate.py`.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python replicate.py --null 300       # takes about 5-10 minutes
diff trades.csv expected_trades.csv  # should print nothing
diff probs.csv expected_probs.csv    # should print nothing
pytest -q tests/                     # 13 passed, in under a minute
```

The run writes two files:

- `probs.csv`: one model probability per gold bar.
- `trades.csv`: every trade.

Delete `probs.csv` to force the model to retrain.

---

## §1 The idea in one paragraph

A classifier reads 20 indicators on every closed hourly gold bar. It estimates the probability that this bar is at, or within 3 bars of, a **swing low**.

When that probability reaches **0.60**, the strategy:

- buys at the next bar's open;
- places a stop **3 × ATR** below the entry;
- moves that stop up behind the highest high, 3 ATR below it, so the stop only ever rises;
- takes profit at **4.3 times the initial risk**;
- closes after **120 hours** at most.

It is long only and holds one position at a time.

The model is retrained every month on data that ends before the month it predicts. Every trade in the backtest is therefore made by a model that had never seen that month.

---

## §2 Data

| File | Bars | From | To | Role |
|---|---|---|---|---|
| `XAUUSDs_H1.csv` | 60,000 | 2016-09-12 | 2026-11-05 | Traded, and used for training |
| `EURUSDs_H1.csv` | 60,000 | 2017-03-15 | 2026-11-05 | Training only |
| `GBPUSDs_H1.csv` | 30,000 | 2022-01-11 | 2026-11-05 | Training only |
| `USDJPYs_H1.csv` | 30,000 | 2022-01-11 | 2026-11-05 | Training only |

**Format.** Columns are `time,open,high,low,close`. Times are the broker's server clock (UTC+2 in winter, UTC+3 in summer). Each row is a **closed** bar stamped at its open time. Gold is quoted in USD per ounce.

**Why currency pairs.** They are used only as extra training examples, never traded. Training on gold plus these pairs measurably improved how well the model found gold's swing lows. The pairs work because every feature is scaled by price or ATR, so a "stretched, exhausted, long lower wick" bar looks the same on any market.

**Which bars are used where.** Bars before 2024 are used only for training. Results are measured from 2024-01-01 onwards.

### Leak screen

Some data feeds stamp a bar with the *next* bar's time. When that happens, features describing where the bar closed in its range secretly contain the next bar's return.

`intrabar_leak()` measures this as |corr(close position of bar t, return of bar t+1)|:

- clean feeds score about 0.02;
- broken ones score about 0.4.

Any series above 0.15 is refused. All four files here pass:

| Series | Leak score |
|---|---|
| Gold | 0.015 |
| EURUSD | 0.023 |
| GBPUSD | 0.013 |
| USDJPY | 0.006 |

---

## §3 Features (20 per bar, all causal)

"Causal" means every value at bar *t* uses only bars ≤ *t*. Below, `c`, `h`, `l` and `o` are close, high, low and open. `ATR` is the 14-bar Wilder ATR (true range, smoothed with α = 1/14).

| Name | Formula | What it captures |
|---|---|---|
| `z10`, `z20`, `z50` | (c − SMA_w) / stdev_w | How far price is stretched from its mean |
| `pctl10`, `pctl20`, `pctl50` | share of the last w closes below c | Rank of the close in its window |
| `rsi7`, `rsi14` | Wilder RSI | Oversold |
| `ema_dist` | (c − EMA50) / ATR | Gap from trend, in ATRs |
| `mom` | (c − c₋₅) / ATR | 5-bar momentum |
| `accel` | mom − (c₋₅ − c₋₁₀) / ATR | Is the fall slowing down? |
| `upwick`, `lowick` | wick length / bar range | Rejection wicks |
| `clpos` | (c − l) / (h − l) | Where the close sits in the bar |
| `atr_z` | z-score of ATR against its 100-bar mean | Volatility regime |
| `run_up`, `run_dn` | count of consecutive up or down closes | Streaks |
| `bull_div` | new 20-bar low **and** RSI14 > its 20-bar min + 3 | Bullish RSI divergence |
| `bear_div` | the mirror image at a 20-bar high | Bearish divergence |

Any bar with a missing value (the warm-up period) is skipped for both training and prediction.

---

## §4 The label: what counts as a "bottom"

`swing_lows()` builds a ZigZag in three steps:

1. **Candidates.** A bar is a candidate low if its low is ≤ every low within **±5 bars**. Candidate highs work the same way.
2. **Alternate.** Merge the candidates in time order. When two of the same type are adjacent, keep the more extreme one.
3. **Filter.** A new pivot is accepted only if its price is at least **1.5 × ATR** away from the previous accepted pivot. A same-type pivot that is more extreme replaces the previous one.

The label is **y = 1** for every bar within **±3 bars** of an accepted swing low, otherwise 0. About **33%** of bars are labelled 1, so that is the random base rate.

### The label uses future prices

To know that bar *t* is a swing low you need up to 5 bars after it. To label bar *t* as "near" a low you need 3 more. A later, deeper low can also replace an earlier one: in testing, labels up to 13 bars before a cut changed when more data arrived.

This is safe for training because of two rules, the first of which is the real guard:

1. **Truncate first, then label.** Each series is cut at the start of the test month *before* any swing is computed. Training labels are therefore always "what was knowable then", never revised with later prices. This is exactly how the live bot sees its training data.
2. **Purge the tail.** The last `TOL + 1 = 4` bars of every training window are dropped, because their labels are the most provisional.

`test_labels_need_the_future_so_training_is_purged` shows that labels near a cut really do change with more data. `test_walk_forward_predictions_are_causal` shows that predictions do not: deleting every bar after 13 Jan 2024 leaves the 2–12 Jan probabilities unchanged.

---

## §5 The model: walk-forward gradient boosting

**Model.** `sklearn.ensemble.HistGradientBoostingClassifier` with:

- `max_iter=300`
- `learning_rate=0.05`
- `max_leaf_nodes=31`
- `random_state=0`

**Maths.** The model is a sum of 300 small regression trees fitted to minimise log loss. Its score F(x) becomes a probability:

```
P(bottom | x) = 1 / (1 + exp(−F(x)))
```

Each new tree fits the gradient of the loss, which is y − P, left over by the trees before it. Its contribution is shrunk by 0.05.

**Walk-forward loop.** Starting 2024-01, for each calendar month M:

```
for each series (gold, EURUSD, GBPUSD, USDJPY):
    d   = bars strictly before the first hour of M
    F,y = features(d), labels(d)            # labels built on truncated data
    use rows [len(d) − 4 − 20,000, len(d) − 4) with no missing features
fit one model on all series stacked (~64k rows in 2024, ~80k from 2025)
predict P(bottom) for every gold bar in month M
```

**Expected size.** There are 35 refits. Gold fires (P ≥ 0.60) on **6.4%** of hourly bars in the test period.

**Live versus backtest.** The live bot retrains **every hour** on the same 20,000-bar windows. Refitting monthly can only make the backtest's model staler, so it is the conservative choice.

---

## §6 Trade rules (variant B)

Let *i* be the bar where P ≥ 0.60, and let ATRᵢ be its ATR. The constants are:

| Constant | Value |
|---|---|
| spread | $0.25/oz |
| swap | −6%/year of notional on longs |
| risk per trade | 0.25% of equity |

### Entry

- `entry = open[i+1] + spread`, so you pay the ask.
- `R = 3 × ATRᵢ`. This is the dollar risk per ounce, and it is "1 R".
- `stop = entry − R`
- `TP = entry + 4.3 × R`

### On each later bar j, checked in this order

1. If `low[j] ≤ stop`, exit at `min(stop, open[j])`. A gap through the stop fills at the open.
2. If `high[j] ≥ TP`, exit at `max(TP, open[j])`.
3. If `j − (i+1) ≥ 120`, exit at `open[j]`.
4. Otherwise: `best = max(best, high[j])`, then `stop = max(stop, best − 3 × ATRⱼ)`.

**Same-bar order.** The stop is checked before the take-profit. If one bar touches both, the backtest assumes the stop hit first, which is the pessimistic choice. A stop raised on bar j only applies from bar j+1, so the backtest never uses a high and a low from the same bar in the order that suits it.

### P&L in R

```
pnl_R = (exit − entry + entry × swap × days_held / 365) / R
```

### Position size

```
lots = 0.25% × equity / (3 × ATRᵢ × 100 oz)
```

Each trade therefore risks the same 0.25% of equity regardless of volatility. On a $100,000 account that is $250 per R.

**One position at a time.** After an exit on bar j, bar j's own probability may trigger the next entry at `open[j+1]`, as the live bot would.

### Worked example: trade 0 in `expected_trades.csv`

| Step | Value |
|---|---|
| Signal bar | 2024-01-02 18:00 fired; entry at the 19:00 open + $0.25 = **2066.38** |
| Risk | ATR was 5.64, so **R = $16.93/oz**; initial stop 2049.45 |
| Trail | Price rose first, and the trail lifted the stop to about 2055.9 |
| Exit | The stop was hit at 2024-01-03 11:00 → **−0.63 R** |

Many losses look like this one: smaller than −1 R, because the stop had already trailed up.

### Worked example: a take-profit (row 38)

| Step | Value |
|---|---|
| Entry | 2177.07, R = 18.41 |
| Target | 2177.07 + 4.3 × 18.41 = 2256.25, reached 51 bars later |
| Result | +79.18/oz minus 5 days of swap = **+4.20 R** |

---

## §7 From trades to Sharpe

### Daily returns

Each trade's equity return, 0.25% × R, is **booked on the day the trade exits**. Weekends are dropped, which gives 744 weekdays:

```
r_d = 0.0025 × Σ R of trades closed on day d        (0 on days with no exit)
```

### Sharpe

```
Sharpe = mean(r_d) / stdev(r_d) × √252
```

### Expectancy: why a 41% win rate makes money

With p = win rate, W = average win and L = average loss:

```
E = p·W − (1−p)·L = 0.412 × 1.07 − 0.588 × 0.54 = +0.125 R per trade
breakeven win rate p* = L / (W + L) = 0.54 / 1.61 = 33.5%
```

### Checks that `replicate.py` prints

| Check | How | Result |
|---|---|---|
| Bootstrap 95% CI | Resample 20-day blocks of daily returns, 2,000 times | [+0.54, +2.59] |
| Cost stress | Double the spread | 1.48 |
| TP dependence | Remove the take-profit | 1.51 |
| Benchmark | Hold gold over the same days, with swap | 0.93 |
| Timing null | Keep the same fire pattern but rotate it to a random offset inside 2024-26, 300 times, with the same exits | median 1.29, p 0.15 |

---

## Expected results

```
trades 510 = 3.44 per week; fire rate 6.4% of bars
win 41.2%  avg win +1.07R  avg loss -0.54R  expectancy +0.125R  total +63.8R
exits {'stop': 500, 'tp': 9, 'open': 1}  median hold 16h
equity at 0.25% risk +17.10%  maxDD 2.75%
SHARPE +1.58   block-bootstrap 95% CI [+0.54, +2.59]
per year {2024: 1.16, 2025: 2.65, 2026: 0.79}
2x spread +1.48   no TP +1.51
buy & hold gold, same days, with swap: Sharpe +0.93
random timing (300): median +1.29, 95th +1.75, p 0.150
```

**About the exit counts.** "stop" covers every trailing-stop exit, including profitable ones. 210 trades won and only 9 of them reached the TP. The one "open" trade was still running at the end of the data.

**If your numbers differ:**

- Check library versions first. Gradient-boosting results can shift slightly across scikit-learn releases; this was built with 1.6.1.
- If `probs.csv` matches but the trades don't, diff `trades.csv` against `expected_trades.csv` to find the first divergent row.

---

## How to read the result honestly

1. **The 1.58 is real but mostly not the detector.** Random entry times with the same exits average a Sharpe of 1.29. The model's timing beats random only 85% of the time (p 0.15), which is not statistically significant. Most of the return comes from being long gold, in a strong bull market, with a wide trailing stop that lets winners run.
2. **The 4.3 R take-profit was chosen on these same years.** That flatters the result slightly. Without a TP the Sharpe is 1.51.
3. **It is still a better way to hold gold.** Buy-and-hold had a Sharpe of 0.93 with a much deeper drawdown. ZigZag is in the market 63% of the time, its daily returns correlate only 0.26 with gold's, and its worst drawdown at 0.25% risk was 2.75%.
4. **It does not travel.** The same pipeline on 41 other futures and CFD markets showed no timing skill. It belongs on gold.

---

## Tests

Run `pytest -q tests/` from this folder. All 13 checks pass in under a minute, because the end-to-end check reads `expected_probs.csv` instead of retraining.

| Test | What it proves |
|---|---|
| `test_matches_repo_features_atr_labels` | Features, ATR and labels are identical to the original project code in `repo_code/`, which the live bot imports. |
| `test_matches_repo_leak_screen` | The leak score is identical to the original, and all four series pass it. |
| `test_features_are_causal` | Appending future bars leaves every past feature unchanged. |
| `test_labels_need_the_future_so_training_is_purged` | Labels near a cut *do* change with more data, which is why the walk-forward truncates before labelling. |
| `test_walk_forward_predictions_are_causal` | Retrains with every bar after 13 Jan 2024 deleted and gets the same 2–12 Jan probabilities. No look-ahead. |
| `test_initial_risk_and_trail_before_stop` | Entry = next open + spread; R = 3 ATR; the trail rises before the stop is hit. |
| `test_gap_through_stop_fills_at_open` | A gap below the stop exits at the open, not at the stop price. |
| `test_take_profit_at_4_3_R` | The TP sits at entry + 4.3 R. |
| `test_stop_checked_before_tp_on_the_same_bar` | When one bar touches both, the stop wins (pessimistic). |
| `test_trailing_stop_follows_highest_high` | The stop sits at the highest high minus 3 ATR and can exit at a profit. |
| `test_time_exit_after_120_bars` | A trade still open after 120 bars closes at that bar's open. |
| `test_no_entry_while_a_position_is_open` | Only one position at a time. |
| `test_reproduces_expected_trades_and_sharpe` | All 510 trades match, Sharpe is 1.58, and the rate is 3.44 trades a week. |

The exit-rule tests use a hand-built flat market (open = close = 100, high 100.5, low 99.5). That makes ATR exactly 1, so every stop and target can be checked by hand.

---

## Original code (`repo_code/`)

`replicate.py` is a faithful single-file extraction of these project files. The tests prove the two agree. All originals need only numpy, pandas, scipy and scikit-learn, except the live bot, which also needs `mt5linux` and a running MetaTrader 5 terminal.

| File | Role |
|---|---|
| `scripts/v5_xau_turning_ml.py` | The original ZigZag swing finder, features, ATR and RSI. |
| `scripts/v5_pooled_bottom_detector.py` | The `intrabar_leak` screen. |
| `scripts/v5_zigzag_signal.py` | The **live** signal: builds features and labels, fits on the trailing window, scores the last closed bar. |
| `scripts/v5_zigzag_live_wf.py` | The original walk-forward, which the probabilities here come from. Its `simulate()` is the older variant with a profit lock. |
| `scripts/v5_zigzag_champion_reproof.py` | The project's own run of this re-proof. |
| `scripts/v5_zigzag_lock_ladder.py`, `v5_zigzag_late_lock.py` | The exit studies that chose variant B: no profit lock, 3 ATR trail, TP at 4.3 R. |
| `scripts/v5_zigzag_executor.py` | The **live executor**, run hourly. It reads the broker's H1 feed, calls `bottom_probability()`, sizes the order as 0.25% equity / (3 ATR × contract size), and sends a BUY with the SL and TP attached. |
| `scripts/v5_zigzag_trail_watch.py` | The **live trail**, run every minute. It moves the stop up to the peak minus 3 ATR and never lowers it. |
| `configs/zigzag_variant_b.example.json` | Example live settings for variant B (`lock_steps: []`, `tp_r: 4.3`). Port, magic number and symbol names are placeholders for your own MT5 setup. |

**Live versus backtest.** The live bot retrains every hour, while the backtest retrains monthly. The live trail checks every minute, while the backtest checks once per H1 bar. Neither difference is in the backtest's favour.
