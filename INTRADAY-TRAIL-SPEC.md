# Intraday trail-follower — the surviving spec

What is left of the 2026-10-30 request (*catch an intraday trend, tight SL, move to breakeven,
trail, no TP, monitor in real time*) after each element was measured. Every parameter below cites
the finding that fixes it. Measurements: `V5_FINDINGS.md` §3bl, §3bn, §3cj, §3ck, §3co.

> ⚠️ **This is a measurement-grade spec, not a validated money-maker.** The entry's edge is
> marginal and conditional — see §5. Build it to *measure*, not to size up.

---

## 1. What survived, and what was refuted

| element of the request | verdict | finding |
|---|---|---|
| long-only | **KEEP** | §3bn: long-only 1.091 vs long/short 0.954, both halves negative for shorts — the 6th refutation |
| **no take-profit** | **KEEP** | §3bl: a fixed bracket costs even a **perfect-foresight oracle** −0.94 to −1.43bp vs holding |
| wide **ratcheting** trail | **KEEP** | §3bn: peaks at 3.0 ATR with smooth decay both sides; ratchet worth **+0.077 Sharpe** over fixed SL/TP, +0.263 over signal-flip |
| brackets in **ATR**, not percent | **KEEP** | 2026 daily range ~$125 (2.73%); a 0.50% stop is **0.18 of one day** and is noise |
| **move SL to breakeven** | ❌ **REFUTED** | §3co: **−6.36bp/trade at t −3.55**; hit rate collapses 0.5345 → 0.4044 |
| real-time M1 monitoring | ❌ **UNNECESSARY** | §3co: M15/M30/H1 agree to **0.004 Sharpe**. Run on **H1** |

**Why the breakeven move fails.** With a ratcheting trail already running, the path it cuts is
*drawdown-then-recovery* — and those recoveries are what pay for the losers. §3bl's
*"barriers truncate the moves that pay for the losers"* applies to it, exactly as it applies to a
take-profit. The softer forms (stop to entry − 0.5 ATR, or triggering at +2.0 ATR) are neutral at
best; **no variant was positive.** If the goal is less risk, **size down** — `overnight-ban`
showed 0.89× sizing beating a +$200 TP at matched drawdown.

---

## 2. The spec

```
INSTRUMENT   XAUUSD  (XAUUSDs at Switch — every Switch symbol carries an `s` suffix)
FRAME        H1                      §3co: identical to M15, ¼ the polling and execution surface
DIRECTION    LONG ONLY               §3bn, and five prior refutations of the short leg

ENTRY        resting BUY STOP at the PRIOR DAY'S HIGH, one per day
             cancel it if the prior day's LOW trades first  (implementable, worth ~+2.3bp)
             skip the day if price is already above the level at decision time

INITIAL STOP entry − 1.5 × ATR(14) of the DAILY bars        (~$155 at a $103 daily ATR)
TRAIL        ratcheting, high-water − 3.0 × daily ATR, NEVER loosened
BREAKEVEN    none                                            §3co, t −3.55
TAKE PROFIT  none                                            §3bl
MAX HOLD     5 days, then exit at market
SIZE         equity × risk_frac ÷ (stop_distance × contract_size)
             risk_frac 0.5% under a 10% wall; never above 1.5%
```

**The stop is in DAILY ATR, not bar ATR.** That is what makes H1 sufficient: a 1.5-daily-ATR stop
sits ~20 M15 bar-ranges from entry, so the unresolvable intrabar ordering that inflated the NR7
result 19× (§`order-of-events-needs-intraday`) cannot bite here. Using *bar* ATR would reintroduce
it immediately.

---

## 3. Operational requirements

1. **The stop lives server-side, on the order.** Not in the bot's memory. A dead process, a
   dropped bridge or a closed laptop must not be able to orphan a position. This is already the
   repo's pattern (`configs/v5_zigzag_trail_fp.json`: *"server-side, attached to the order —
   does not need the bot alive"*).
2. **The bot only RATCHETS the stop.** It never places the protective stop as a decision; it
   tightens an existing one. That keeps the failure mode safe: if the bot dies, the last stop
   stands.
3. **Its own magic.** 360595 Switch book · 360597 zigzag · 360598 FundingPips book · 360599 fill
   probe are taken.
4. **Server clock for every time comparison.** MT5 stamps bars and ticks on the broker clock
   (UTC+3 at both brokers). Comparing to `utcnow()` has produced three dead gates here — and a
   gate that cannot fire *certifies* exactly what it was built to catch.
5. **Session boundary, measured not assumed.** XAUUSD trades server **01:00–23:59**; hour **00**
   is the daily break. Flat before the break means **no swap** — a real advantage against the
   book's −3.79%/yr financing drag.
6. **Crash-state guard.** A restart must re-read the account, not trust a stale state file; a
   stale `initial_balance` once put an 8% halt off by $400.
7. **Never on a hand-traded account.** Manual trading once drove equity −37%, which made the guard
   flatten the book and latch a **permanent** halt while it sat at +25.76%. Three of three fill-probe
   orders were also deleted from outside the terminal on such an account.

---

## 4. One-per-day polling, not 15-second

§3co measured M15, M30 and H1 at +2.071 / +2.067 / +2.068 Sharpe — a spread of 0.004. The frame
only decides how finely the stop resolves, and at 20 bar-ranges of distance that is immaterial.
**An hourly ratchet is enough.** A 15-second loop buys nothing measurable and costs a persistent
process, a watchdog, and a new class of failure.

---

## 5. ⚠️ What is NOT established

- **The entry's edge is marginal.** §3cj: a *placebo* level at the same ATR distance above the
  open earns ~60% of the excess and ~71% of the net; the prior-day high's advantage over an
  arbitrary level sits at **p 0.05–0.09**.
- **It loses to simply holding gold on return.** §3ck: +11.06% CAGR against holding's +11.96%,
  even at a perfect fill. It wins on *risk* — Sharpe +1.193 vs +0.784, drawdown −11.0% vs −26.6%,
  on 26.7% exposure. That makes it a **blend sleeve, not a standalone**.
- **The slippage budget is 3.5bp** = $1.40/oz at $4,000 gold = 6.7× the live spread. Past that it
  stops beating buy-and-hold on both Sharpe and Calmar. **This is still being measured** by
  `scripts/v5_fill_probe.py` (magic 360599).
- **Bar 1 (DSR ≥ 0.90, PBO < 0.30) has never been computed** for this candidate, and §3cj widened
  its trial set from one hypothesis to a level-placement family.
- **The Sharpe levels in §3co (~2.07) are not deployable numbers.** Holding the event set fixed
  stacks overlapping trades, hence its −39.5% drawdown. That run measured a *comparison*, not a
  strategy.

**Deploy to a demo to observe execution. Do not size it as a validated edge.**
