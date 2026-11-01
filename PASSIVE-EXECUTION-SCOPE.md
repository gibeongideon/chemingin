# Passive execution — ⛔ STOOD DOWN 2026-11-01, NOT RUN

> **CLOSED BEFORE PHASE 1 BY STANDING INSTRUCTION: "if something a retail trader cannot do, do
> not do it."** §0's feasibility check had already established that market making is structurally
> impossible on every account held — all venues are `trade_exemode = MARKET`, dealer execution
> with no book, no queue and no maker rebate. Phases 2–4 required an exchange account, colocation-
> adjacent latency and queue priority, none of which is retail-reachable. Running Phase 1 would
> have measured a quantity we could never harvest.
>
> **Kept as a record of the feasibility finding, which is durable:** the mechanism that
> demonstrably pays intraday (earning the spread) is unavailable on CFD/dealer venues by
> construction. That closes the whole liquidity-provision branch for this project, permanently,
> and it is a better reason than a failed backtest.

The one untested mechanism that the firms which demonstrably profit intraday actually use:
**earning the spread instead of paying it.** Everything this project has tested took liquidity.

> ⚠️ **This is a pivot, not an extension.** Different asset class, different venue type, different
> infrastructure, different skill. Read §0 before anything else.

---

## 0. Feasibility, measured 2026-11-01

| question | answer |
|---|---|
| Can we post passive on Switch / FundingPips / Maven / HFM? | **No.** All symbols report `trade_exemode = MARKET` (dealer execution) |
| Is there an order book to join? | **No.** CFD/dealer venues quote *to* you; there is no queue, no maker rebate |
| What does a limit order do there? | Waits for the dealer's price to reach it. You still cross their spread; the dealer controls the fill |
| Is exchange research data reachable from the VPS? | **Yes** — `data.binance.vision` returns HTTP 200 (free historical trades/quotes) |
| Is the exchange trading API reachable? | **No** — `api.binance.com` returns **403** from this host |

**Consequence.** The research is runnable today on free data. **Live execution is not possible on
any venue we currently hold** and would require an account on a real exchange with a central limit
order book. That is the decision this scope exists to inform.

---

## 1. The quantity that decides it: REALIZED SPREAD

A market maker does not earn the quoted spread. It earns the quoted spread **minus what informed
flow takes back**. Standard microstructure splits it:

```
effective spread = 2 · side · (P_trade − M_t)            what you capture at the fill
realized spread  = 2 · side · (P_trade − M_{t+Δ})        what you KEEP Δ later
price impact     = effective − realized                   what adverse selection took
```

**A maker's gross P&L is the realized spread.** So the whole question reduces to one measurement:

> ### Is the realized spread, net of maker fees, positive at a Δ we can actually manage?

We are ~100ms+ from any exchange and will sit at the **back of the queue**. If realized spread is
only positive at Δ < 10ms, the edge belongs to colocation and we cannot have it. If it is still
positive at Δ = 1–60s, there is something a slow maker can hold.

**This is CONTROL #5 applied before any build:** measure whether the harvested quantity is even
positive before constructing anything to harvest it. Exactly the discipline that closed the vol
oracle, the MAE oracle and the ZigZag on mechanism.

---

## 2. Phases and gates

| phase | what | gate to proceed |
|---|---|---|
| **0** ✅ | Feasibility (above) | done — research yes, live execution needs a new venue |
| **1** | **Realized spread vs Δ** on 2–3 liquid pairs from free Binance trade data, Δ ∈ {0.1s, 1s, 10s, 60s}, split by trade side and by hour | realized spread **net of the maker fee** > 0 at **Δ ≥ 1s**, with an episode-based SE and both sample halves positive |
| **2** | **Queue realism.** We are last in line, so we fill when the queue *clears* — which is when price is about to move through us. Measure the fill-weighted realized spread, not the quote-weighted one | the fills we would actually get must stay positive. ⚠️ This is `measure-the-skipped-sample` applied to quotes, and it is the most likely kill |
| **3** | **Inventory.** Spread capture minus inventory risk, with a cap and quote skew | Sharpe at matched inventory risk, vs simply holding (CONTROL #2) |
| **4** | Paper on an exchange testnet, then minimum size | fills match the simulation within the Phase 2 bound |

**Stop at the first failed gate and write it up.** Phases 2–4 are not started before 1 passes.

---

## 3. What will probably kill it, named in advance

1. **Adverse selection.** This repo has already measured passive fills on gold and found them
   adversely selected: **FILLED −22.8bp vs SKIPPED +17.9bp, t −5.23**, and "waiting for a dip
   selects losers" at t −8.60. That was *directional entry timing*, not market making — a genuinely
   different construction — but it is the same mechanism and it has bitten twice here.
2. **Queue position.** A back-of-queue maker is filled *preferentially* on the trades that hurt.
   Phase 2 exists solely to price this and is where I expect it to die.
3. **Latency.** The literature puts the latency-arbitrage tax at **0.42bp of volume**, roughly a
   third of the effective spread. That is taken from someone, and at 100ms the someone is us.
4. **Fees.** Maker fees are only negative at volume tiers we will not reach. At 0.00–0.02% the
   spread must cover it on every round turn.
5. **Capacity is the wrong way round.** Where a slow maker *can* earn, the edge is usually small
   enough that it funds a person, not a book.

**Honest prior: Phase 1 passes on the headline number and Phase 2 kills it.** The headline realized
spread is computed over all trades; the fills we would actually receive are the adversely selected
subset, and that gap is precisely what §3bc measured.

---

## 4. What it costs, and what it is worth

Phase 1 is **one script and free data** — a day's work, no new accounts, no capital. It answers a
question this project has never asked, and it answers it with the same controls as everything else
(oracle first, benchmark vs holding, episode-based SE, both halves).

Phases 2–4 require an exchange account and are a genuine change of project. **The decision point is
after Phase 1**, with a measured number rather than an argument.

**What it is NOT:** a replacement for the live gold book, which continues untouched. And it is not
a prediction strategy — the entire point is that the mechanism which demonstrably pays intraday is
*execution*, which is also where this project's own measurements kept landing (§3ci: 47% of the one
surviving lead was a fill advantage; financing beat 64 signal trials).
