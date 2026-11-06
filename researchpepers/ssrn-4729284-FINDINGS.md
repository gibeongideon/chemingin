# Findings: "A Profitable Day Trading Strategy For The U.S. Equity Market" (5-min ORB on Stocks in Play)

**Paper:** Zarattini, Barbon & Aziz (2024), SSRN 4729284 — `researchpepers/ssrn-4729284.pdf`.
**Tested:** 2026-11-06 with `scripts/v5_orb_stocks_in_play.py`.
**Verdict:** the direction effect replicates. It is not tradeable on MT5 share CFDs, because the spread is about 26× the paper's cost.
**Repo record:** `V5_FINDINGS.md` §3dh.

## What the paper claims

| Rule | Value |
|---|---|
| Signal | Direction of the first 5-minute candle (9:30–9:35 ET); a doji means no trade |
| Entry | Buy-stop at the opening-range high (bullish) / sell-stop at its low (bearish) |
| Stop | 10% of the 14-day ATR from the fill |
| Exit | At the close (no overnight) |
| Size | 1% of equity risked per trade, 4× leverage cap |
| Universe | Price > $5, ATR14 > $0.50, 14-day average volume ≥ 1M shares |
| "Stocks in Play" | Opening-range relative volume (first-5-min volume ÷ 14-day average) ≥ 1, top 20 |
| Data / cost | US stocks 2016–2023, $0.0035 per share commission |
| Claimed result | Sharpe 2.81 (Stocks in Play), 0.48 (base version) |

## How we tested it

- **Venue:** Switch Markets US-share CFDs on MT5 (the retail venue actually available). M5 bid bars with each bar's quoted spread, fetched by `scripts/v5_us_m5_fetch.py`.
- **Sample:**
  - 313 of 1,302 listed names returned history; 233 passed the filters.
  - 594 trading days, 2024-06-14 → 2026-11-05.
- **Execution:**
  - Long stop at the opening range's ask high, filled on the ask.
  - Short stop at the bid low, covered on the ask.
  - Spread is charged both ways.
  - GROSS = the same trades with the spread set to 0.
- **Substitutions:**
  - CFD tick volume replaces share volume, both for relative volume and for the 1M-share filter (the latter becomes "top half of the universe by 14-day opening tick volume").
- **Control:** random direction. Same names, days, entry levels and stops, but long/short is a coin flip (200 draws). If the first candle's direction means nothing, the strategy should look like these draws.
- **Variants:**
  - BASE (all eligible names);
  - SIP20 (the paper's top 20);
  - SIP20_CHEAP (SIP20 restricted to names whose stop is ≥ 6× the spread).

## Results

| Variant | Trades | Win % | Avg R | NET Sharpe | GROSS Sharpe | Random-direction median (gross) | p (gross) |
|---|---|---|---|---|---|---|---|
| BASE | 54,781 | 7.9 | −0.540 | −18.3 | −1.33 | −2.43 | 0.000 |
| SIP20 | 9,934 | 7.8 | −0.474 | −10.3 | −0.27 | −1.26 | 0.015 |
| SIP20_CHEAP | 2,067 | 16.6 | +0.011 | −0.12 | +0.35 | −0.31 | 0.055 |

- Win %, Avg R and NET Sharpe are net of the spread.
- Gross, SIP20 has Sharpe −0.46 in the first half and −0.12 in the second.
- Report: `data/v5_runs/orb_sip/report.txt`.
- Pessimistic-bound report: `report_pess.txt`.

## What it means

1. **The paper's core effect is real.** Trading the first candle's direction beats a random direction by about 1 Sharpe before costs, in both halves (p 0.000–0.015). The opening candle does carry information about the rest of the day.
2. **The venue's cost destroys it.**
   - A share CFD charges a flat ~$0.31 spread whatever the share price.
   - The stop is 10% of ATR, a median of about $0.46.
   - So the spread is **0.66R per side**, about 1.3R per round trip.
   - The edge is about +0.05R per trade.
   - The paper's per-share commission is about 0.01R.
   - Even the cheapest sixth of names nets slightly negative (−0.12).
3. **What would make it tradeable:** only a direct-market-access stock broker with per-share commissions. That is a different venue from the MT5 accounts. Wider stops would cut the spread per R, but they change the strategy away from the paper's.

## Method lesson (applies to any tight-stop intraday test)

A stop of 10% of ATR is about one M5 bar's range. So the bar that fills the order often also touches the stop, and OHLC bars cannot say which came first.

- **Always "stop second"** (pessimistic): even *cost-free random* trades lose −0.5R each. That cannot be right; the true value for a cost-free random trade is about 0.
- **OHLC path convention** (an up bar runs open→low→high→close, a down bar open→high→low→close): this is the main result. It still biases the absolute gross level low (random median −1.3 to −2.4).

Rule: judge a tight-stop strategy against its **random-direction null**, not against zero, and print the pessimistic bound beside it. M1 or tick data would settle the bar order, but it could not rescue 1.3R of cost.

## Reproduce

```bash
python scripts/v5_us_m5_fetch.py --port 18815        # on the VPS; data/us_m5/ is git-ignored
python scripts/v5_orb_stocks_in_play.py               # ~17 s, OHLC-path entry bar
ORB_PESS=1 python scripts/v5_orb_stocks_in_play.py    # pessimistic bound
```

## Follow-up: moving the working bit to gold, BTC and ETH, then into the champion book (§3di)

**Why try it:** the cost wall above was specific to share CFDs. On gold and BTC the spread is about 0.12–0.13R per side, versus 0.66R on shares.

| Test | Result |
|---|---|
| Paper's direction rule vs random direction (gold, BTC) | No difference (p ≈ 0.64): **the direction adds nothing on 24h assets** |
| Both stop orders live (buy-stop at the range high, sell-stop at the low) | Best version: BTC gross Sharpe +1.49, so the breakout pays in **both** directions |
| Gold, M15 bars | Undecidable (+1.17 / −1.26 by bar-order rule); M1 calibration puts it at ~0 net |
| BTC, M1 bars | Robust to bar order: net +0.3 to +0.6 |
| ETH, rules frozen (independent replication) | Net +0.82 to +0.99, both halves positive |
| Champion book XAU+BTC (Sharpe 1.26) + BTC/ETH breakout sleeve at 10% risk | Sharpe 1.32–1.35, drawdown −13.4% → −10.7%, t +2.0 to +2.8 at matched vol |
| **Same, at 2× cost (0.10% per side)** | **Sleeve −0.06, book 1.25: the gain disappears** |
| Real Switch spreads (2026-11-06) | BTC 0.045% (OK), **ETH 0.20% (4× too expensive)** |

**Conclusion:** the opening breakout effect is real and replicates, but it lives inside a cost margin of about 0.05% per side. On Switch, only BTC qualifies, and BTC alone lifts the book 1.26 → 1.32 (not significant). It is a lead for a forward test with real fills, not something to deploy yet.

```bash
python scripts/v5_orb_xau_btc.py              # gold/BTC standalone + XAU champion integration
python scripts/v5_orb_book_integration.py     # BTC sleeve inside the XAU+BTC book
ORB_COST=0.001 python scripts/v5_orb_book_integration.py   # the cost stress that kills it
```
