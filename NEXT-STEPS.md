# Next steps & realistic earnings — 2026-08-08

## Part 1 — What you would actually earn on $100,000

Monthly P&L distribution, measured from real daily returns 2018-2026, at **live FTMO
cost** and **net of financing/swap**. These are backtest figures, not guarantees.

### The deployed 4-asset book (XAU + BTC + NDX + BRENT) — the thing that works

| vol | mean/mo | **median/mo** | months positive | worst month | best month |
|---|---|---|---|---|---|
| 7% | $667 | $401 | 57% | −$4,245 | +$6,179 |
| **9% (deployed)** | **$919** | **$573** | **59%** | **−$5,369** | +$8,041 |
| 11% | $1,172 | $742 | 59% | −$6,485 | +$9,926 |

### After the profit split — what actually reaches you (book @9%)

| account | mean/mo | median/mo |
|---|---|---|
| FTMO 80% | $611 | $459 |
| FTMO 90% | $765 | $516 |
| FundingPips 95% | $842 | $545 |

### Single-XAU variants — read the MEDIAN, not the mean

| variant | mean/mo | **median/mo** | months positive | worst | best |
|---|---|---|---|---|---|
| full | $982 | **−$408** | **43%** | −$5,675 | +$17,794 |
| trim50 | $831 | −$281 | 43% | −$4,348 | +$13,866 |
| trim100 | $692 | −$41 | 50% | −$3,192 | +$10,588 |

**This is the most important table here.** Single-XAU has a *higher mean* than the
4-asset book but a **negative median** — you lose money in **57% of months** and make it
back in rare huge ones (best month +$17,794). The diversified book earns less on average
but pays you in **59% of months** with a **+$573 median**. For a prop account, where two
bad months in a row can breach the drawdown limit, the book is far more survivable.

### Honest framing for a prop account

- You do **not** own the $100k. You keep the split on profits; a drawdown breach loses
  the whole account, so the −$5,369 worst month (−5.4%) matters more than the mean.
- Realistic expectation on a **funded** 100K FTMO at 90%: **~$500-770/month**, with
  roughly 4-5 losing months a year and a bad month near −$5,000.
- Getting funded first: ~11 months at ~95% probability (book @9%, FTMO 2-step).

### The intraday strategy: expected gain = **$0**

Disproven today (see below). It is not deployed and the monitor is disarmed.

---

## Part 2 — What was settled today

**Intraday high-R:R on XAU — DISPROVEN (7th time), with a new diagnosis.**

The cost hurdle was genuinely beatable this time — only **1.3-2.5pp** of edge needed, vs
the old fade which faced a spread 10× its edge. The oracle gate passed with margin
(+1.29 to +2.48 R/trade, 12/12 years) and set a concrete target: **+7-10pp of precision
over the base rate**. It also proved expectancy is **invariant to recall** — selectivity
matters, coverage does not.

Then every entry failed:

| approach | expectancy R | verdict |
|---|---|---|
| Opening-range breakout | −0.051 | DEAD (fails 1.5× cost, 4/12 yrs) |
| Volatility-expansion breakout | −0.012 | DEAD (fails 1.5× cost, 4/12 yrs) |
| ATR-regime conditioning | −0.068 | hurdle reduction, **not profit** |
| H4-alignment | +0.087 | **see below** |

H4-alignment looked real — monotonic dose-response (+0.038 → +0.148 R as champion
strength rises), and a short-side control at **−0.204 R (t −4.69)** on the same bars,
which is the signature of genuine trend alignment rather than a cost artifact.

**The decisive paired test killed it:** holding passively from the same entry to the same
exit beat the stop/target version at every threshold — **−0.044 R, t −2.50** at champ≥1.70.
So the return was never intraday skill; it was **trend exposure**, and a 3×ATR stop on
M30 gets hit by noise at intrabar extremes before the move resolves. The unfiltered
breakout is −0.028 R: the entry carries no information at all.

---

## Part 3 — Next steps, in priority order

### 1. Close the intraday thread properly (~3h of compute, after the session limit resets)
Four approaches never ran: **session/hour conditioning**, **prior-day levels**,
**meta-labelling**, and the **green/red candle refit**. Session-of-day is the best
remaining bet (cheapest possible edge; gold has documented intraday drift). Completed
agents replay from cache, so only the 4 missing ones cost anything.
*If they also fail, write V5_FINDINGS §3q and close intraday permanently.*

### 2. Test conviction-scaling — the idea the disproof handed us
Today's result says *holding beats stopping* when the champion signal is strong.
`champ ≥ 1.70` was in-market only **22.7% of bars** and returned **+5.8%/yr gross**.
So test: **scale champion exposure UP when its own signal is very strong**, judged paired
against the unmodified champion with the block-shuffled-signal null. This is a conviction
overlay on a proven book, not a new strategy — and it is the direct opposite of the
champ-meta trim that failed earlier (which scaled *down*).

### 3. Re-price everything for swap
Financing was discovered only on 2026-08-07 and is **not** in most historical numbers:
gold longs pay ~6.6%/yr of notional, BTC ~30.4%, NDX ~6.9%, Brent **pays +9.6%**. Book
total ≈ **−2.59%/yr of equity**. Every CAGR/Sharpe quoted before that date is optimistic
by roughly this much. Add a swap term to `xau_lab.run` and the basket engine, then re-run
the FTMO pass-rate table.

### 4. Add a broker-health alert
FundingPips rejected every order with `AutoTrading disabled by server` for **four days**
before the account went invalid, and nothing alerted. FTMO has the same blind spot today.
Cheap fix: extend `challenge_daily_report.py` to page when order rejections spike or an
account stops accepting trades.

### 5. Decide the FundedNext slot
The VPS `.mt5` slot is stopped and free. FundedNext-Server3 is not in the MetaQuotes
address book, so it needs either the GUI "Open an Account" wizard (VNC is running on
`:99`, password `Vnc4Login!`, tunnel `ssh -N -L 5901:localhost:5900`) or their own MT5
build at `download.mql5.com/cdn/web/fundednext.ltd/mt5/fundednext5setup.exe`.

### Do NOT do
- Deploy the intraday monitor (`configs/v5_intraday.json` stays `"rule": "none"`).
- Build more XAU **bottom** detectors — oracle ceiling +0.017, and −0.675 standalone.
- Add martingale/recovery sizing anywhere — ruins 71-97% of paths even on a positive base.
- Chase RR ≥ 4 intraday — needs ~25,600 trades to prove a 0.5pp edge.
