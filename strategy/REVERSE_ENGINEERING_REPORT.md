# Reverse Engineering "Happy Gold – OxS" — findings

**Source:** `happyforex_311_trades_773.csv` — 773 XAUUSD trades, 2025-10-15 → 2026-04-14.
Vendor forward test claims +507.88% on a $3,000 deposit (balance $18,251.80), 34.87%
monthly, "Drawdown 0.21%", M30 timeframe.

**Bottom line:** the log is genuine and the strategy is real, but **its edge lives in
sub-second entry timing that is invisible in bar data.** The direction rule was fully
recovered; the timing was not, because it is not there to be recovered from 15-minute bars.
A bar-based replica loses money in every one of 40 configurations tested over 10 years.

---

## 1. The log is authentic

- Cumulative profit reconciles to **$18,251.80 — the vendor's figure, to the cent.**
- Max **balance** drawdown −0.18% vs the vendor's stated 0.21%. The "0.21% drawdown" is
  real, and it is a *balance* measure: losses are tiny and streaks short (max 3 in a row).
- **Timestamp offset found:** the log runs **+2h** ahead of our XAUUSD feed. At −120 min,
  **85.3%** of fills land inside their own containing M15 bar with a **median miss of
  $0.00**; at every other offset tested (±13h) only 9–21% do. An initial run without this
  correction suggested 42.7% of trades were "physically impossible" — that was entirely the
  offset artifact, not fabrication.

## 2. What it is — and isn't

| property | measured |
|---|---|
| position sizing | **fixed 0.10 lots**, all 773 trades |
| concurrency | **0% overlap** — one position at a time |
| martingale / grid | **no** — lots identical after wins and losses |
| hold time | median **14 seconds**; 98.4% under 60s |
| win rate | **83.2%** |
| payoff ratio | **10.6:1** (wins $24.14, losses −$2.28) |
| largest loss ever | **−$5.40**, against a declared stop 2.4 wide |
| exits at declared TP | 2.6% | 
| exits at declared SL | **0.0% — never, in 773 trades** |
| exits at neither | **97.4%** |
| trade rate | 4.27/calendar-day, hours 03–18 vendor time, Fri ≈ half volume |

So: not a grid, not a martingale, not a hold-and-hope. A one-at-a-time scalper with a
declared bracket that essentially never binds.

## 3. The declared stop/target is not the real exit

Applying the vendor's **own** SL 2.4 / TP 10 to the vendor's **own** entries, checked
intrabar over the following 8 bars (stop-first, conservative):

> **7.9% win rate, −$19.02 per trade, −$14,702 total.**

It would have destroyed the account. The bracket is disaster insurance that never triggers
because something else closes the trade within seconds. Mechanically, the real exit:

- captures **27%** of the favourable excursion available inside the entry bar,
- while absorbing only **2%** of the adverse excursion,
- cutting losers **17.4× earlier** than the bar's own range implies.

That asymmetry *is* the 83% win rate and the 10.6 payoff.

## 4. The direction rule — fully recovered

Buy entries vs sell entries, features computed strictly pre-entry (t-stats, n=773):

| feature | buys | sells | t |
|---|---|---|---|
| 20-bar range position | 0.762 | 0.346 | **+30.2** |
| z-score (20) | +0.954 | −0.832 | **+30.0** |
| momentum (8 bars) | +0.953 | −0.880 | +24.4 |
| RSI(14) | 57.8 | 45.1 | +22.7 |
| momentum (4 / 16 / 48) | all positive | all negative | +19.3 / +17.9 / +7.1 |

Plus intra-bar fill position: **buys land at 0.74 of the bar range (near the high), sells at
0.31 (near the low).**

**It buys strength and sells weakness** — a momentum/breakout scalper entering *into* the
move, not fading it. This part is unambiguous.

## 5. The timing rule — provably not in the bars

Against **hour-and-date-matched control bars** (so time-of-day cannot masquerade as signal),
*nothing* distinguishes an entry moment:

| feature | entry | control | t |
|---|---|---|---|
| ATR percentile | 0.444 | 0.485 | −2.61 |
| bar range / ATR | 0.975 | 1.031 | −2.56 |
| z20 | 0.087 | 0.083 | +0.06 |
| range position | 0.560 | 0.561 | −0.05 |
| RSI(14) | 51.6 | 52.0 | −0.59 |
| momentum 4 / 8 / 16 | ~0 | ~0 | +0.17 / −0.16 / +0.30 |

The two features that come closest to separating point to *slightly calmer* conditions —
the opposite of a volatility trigger. Entry minutes use all 60 values of the clock and are
only mildly clustered at M15 boundaries (23.8% in the first 2 minutes vs 13.3% uniform).

**Conclusion: the entry trigger operates below bar resolution.**

## 6. Where the money actually comes from

The decisive measurement — the same entries under exits we can reproduce:

| exit applied to the VENDOR's own entries | win% | mean $ | total $ |
|---|---|---|---|
| +15m close | 46.1% | +12.21 | +9,438 |
| **+30m close** | 47.3% | **+14.99** | **+11,590** |
| +60m close | 46.3% | +4.04 | +3,120 |
| +120m close | 45.4% | −2.88 | −2,229 |
| declared SL2.4/TP10 | 7.9% | −19.02 | −14,702 |
| *vendor's own sub-second exit* | *83.2%* | *+19.73* | *+15,252* |

Two things follow.

**(a) The entry timing is worth a great deal.** Vendor entries held 30 minutes, at zero
cost, return **+$19.39/trade (t = +2.12)**; after a real $0.44 spread, **+$14.99**. Their
actual sub-second exit returns +$19.73. **The exit is not producing the edge — it is
harvesting the same ~$19 more smoothly**, converting a fat-tailed edge into a high-win-rate,
low-variance stream. Note they buy near the *top* of the entry bar and still profit over the
next 30 minutes, which is momentum continuation at tick scale, not a favourable-fill artifact.

**(b) The momentum direction rule alone is worth nothing.** Same rule, replica entries:

| window | signals | GROSS mean $ | t | NET mean $ |
|---|---|---|---|---|
| out-of-sample 2015–2025 | 106,708 | **+0.22** | +2.60 | **−4.18** |
| out-of-sample, z≥1.2 | 73,193 | +0.14 | +1.28 | −4.26 |
| vendor window | 4,652 | +0.91 | +0.39 | −3.49 |

Gross edge of **22 cents** against a **$4.40** round-trip spread. Statistically nonzero,
economically irrelevant.

## 7. The replica, backtested

40 configurations (4 signal thresholds × 5 bar-implementable exits), one position at a
time, entry at next-bar open, real $0.44 spread:

- **Vendor window (2025-10 → 2026-04): 19 of 20 configurations lose.** Best is +42.8% gain
  with an −81% drawdown — i.e. noise, not replication.
- **Out-of-sample (2015 → Oct 2025, ~14,000 trades per config): all 20 lose**, mean
  −$3.63 to −$5.20 per trade, gains −1,200% to −2,180%.

The mean loss per trade ≈ the spread. The replica is a machine for paying commission
14,000 times.

## 8. Verdict

**The strategy is real and not replicable from bar data.** Specifically:

1. The direction rule is momentum, recovered with certainty (t ≈ 30), and **carries no
   standalone edge** (+$0.22 gross vs $4.40 cost).
2. The entry *timing* carries the edge (~$19/trade at 30 minutes) and is **provably absent
   from 15-minute bars** (every feature t ≤ 2.6 against matched controls).
3. The exit converts that edge into a smooth equity curve but does not create it.
4. Therefore no indicator, threshold, or ML model fitted to M15/M30 data can reproduce this.
   The missing information is at tick level — consistent with latency/feed-timing or
   order-flow microstructure, neither of which exists in OHLC bars.

### What would be needed to go further

- **Tick data** for the exact test window, from a comparable feed, to test whether the
  entry trigger is a detectable tick-level event (burst, spread compression, feed lag).
- If it *is* latency-dependent, be aware such an edge is broker-specific and typically dies
  on execution-quality changes, and that many brokers explicitly prohibit it.

### Two caveats stated plainly

- The vendor's entry edge rests on **n=773, t=+2.12** (p ≈ 0.03) — real but not
  overwhelming, and tail-driven. It is also subject to selection bias: we are examining
  this EA *because* it performed, out of a population of many.
- The vendor's own dashboard is internally inconsistent — the header row reports **33%
  drawdown** for the forward test while the Stats panel shows **0.21%**. Those are almost
  certainly different accounts or periods. The 0.21% we verified is a *balance* drawdown on
  *this* log; a 33% equity drawdown elsewhere would describe a very different risk profile.

## 9. Stage 6 — the tick path, tested and eliminated

**The literal test is impossible.** Measured tick retention on this broker is ~1 month:
1,469 ticks in a 10-min probe yesterday, 1,522 at 1 week, 904 at 1 month, and **ZERO at 3,
6, 9 and 11 months** — using a download path proven to work on the dates that do have data.
The vendor window is 5-11 months old. Maven's M1 history reaches back only 5 days; the
older HFM/FTMO terminals remain unreachable (build-5836 IPC fault) and are subject to the
same server-side retention anyway. M1 would not have helped regardless: a **14-second**
median hold lives inside a single M1 bar.

**So the mechanism was tested on tick data that does exist** — 20 trading days,
**2,951,387 ticks** (2026-08-06 .. 2026-09-02). Trigger: mid moved >= k x sigma over the
last W seconds, direction = sign of the move (matching the recovered fingerprint), 15-min
cooldown, forward mid move measured at +30s/+5min/+30min.

| W(s) | k | trades/day | +30min gross | t | net of spread |
|---|---|---|---|---|---|
| 10 | 2.0 | 57.1 | -0.103 | -0.34 | -0.428 |
| 10 | 3.0 | 43.7 | +0.511 | +1.45 | +0.175 |
| 10 | 4.0 | 25.8 | +0.015 | +0.03 | -0.323 |
| 30 | 2.0 | 53.1 | -0.144 | -0.45 | -0.474 |
| **30** | **3.0** | **35.1** | **+0.713** | **+1.86** | **+0.378** |
| 30 | 4.0 | 13.9 | -0.048 | -0.07 | -0.397 |
| **30** | **5.0** | **4.85** | **-0.434** | **-0.30** | **-0.820** |
| 60 | 3.0 | 29.8 | +0.368 | +0.87 | +0.029 |
| 60 | 5.0 | 2.4 | -6.072 | -2.31 | -6.423 |

**Verdict: the burst trigger is not the mechanism.**

- **At the vendor's own frequency (~4.3 trades/day) it LOSES**: -0.434 gross, **-0.820 net**,
  t=-0.30. That is the only like-for-like comparison and it is negative.
- The best cell reaches **37%** of the vendor's +1.94, is **not significant** (t=+1.86 out of
  12 cells searched), and trades **8x more often** than the vendor.
- **No monotone structure**: k=2 negative, k=3 positive, k=4 ~zero, k=5 strongly negative.
  A real effect strengthens or weakens with threshold; this alternates, which is the
  signature of an isolated artifact, not a mechanism. The largest bursts actually
  MEAN-REVERT hard (W=60/k=5: -6.07 pts, t=-2.31, n=48).

## 10. Final verdict — four candidate mechanisms, all eliminated

| candidate | result |
|---|---|
| declared SL 2.4 / TP 10 | −$14,702, 7.9% win — would have blown the account |
| momentum direction rule on bars | all 40 configs lose; −$3.6 to −$5.2/trade over 10 years |
| bar-detectable entry timing | provably absent (every feature t ≤ 2.6 vs matched controls) |
| tick-level momentum burst | negative at matched frequency; best cell an unstable artifact |

**Genuine replication is not achievable from any data obtainable here.** What remains as
explanation for the vendor's record, in order of plausibility:

1. **Broker/feed-specific execution or latency effects** — an edge relative to *their*
   quote stream rather than to the market, which is unreproducible on a different feed by
   construction. Consistent with sub-second holds and with losers being cut at an average
   of **0.228 price points**, which is *smaller than a normal round-trip spread* (0.32-0.44
   measured live). That implies either a raw/ECN account with ~0.10 spread, **or reported
   P&L that excludes spread entirely** — in the latter case every trade carries an
   unreported ~$4 cost on 0.10 lots and the headline figures overstate by roughly 22%.
2. **Tail luck plus selection** — the entry edge rests on n=773, t=+2.12 (p ≈ 0.034),
   tail-driven, and we are examining this EA precisely *because* it performed, out of a
   vendor catalogue of many.
3. A tick-level trigger more sophisticated than a burst (order-flow imbalance, cross-venue
   lead-lag). Cannot be excluded, but is untestable without the vendor's own feed.

**Recommendation: do not attempt to trade a replica.** The one component that reproduces
cleanly (momentum direction) has no standalone edge, and the component carrying the edge is
either unavailable or not a market property. If the strategy is to be evaluated further, the
only honest route is a **live forward test on a small account against their actual broker**,
which tests the real hypothesis (execution-dependent edge) rather than a curve fit.

---

*Analysis scripts: `re_01_forensics.py` (log integrity, distributions), `re_02_consistency.py`
(timestamp alignment, physical plausibility), `re_03_decompose.py` (entry-vs-exit
decomposition), `re_04_entry_fingerprint.py` (matched-control fingerprinting),
`re_05_replicate.py` (replica + out-of-sample backtest), `re_06_tick_mechanism.py`
(tick retention probe + burst-mechanism test on 2.95M ticks).*
