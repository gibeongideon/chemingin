# Pre-registration — book WIDTH re-derived on the regime-U criterion, with carry

Committed BEFORE the first fit. Filed 2026-10-21.

## The question
The deployed Switch book (magic 360595) is 2 sleeves. §3br chose 2 sleeves using `pass` /
`fail_day` from an **FTMO phase-1 simulator**, and its stopping rule is explicitly a daily-breach
rule. Switch Markets has **no daily limit**. `RESEARCH-REGIMES.md:124-126` files that criterion as
**"meaningless in regime U"**, and regime U's rule is "net Sharpe, full stop" (L187).

§3br's own table (spread-only) says **XAU+BTC+NDX +1.696 / maxDD −9.5%** against **XAU+BTC
+1.511 / −11.2%**. So on the regime-U criterion the 3-sleeve book wins. **But §3bs proved carry is
worth 0.22-0.34 Sharpe and those rows charged spread only**, so the comparison must be re-run with
carry before it means anything.

## What is NOT claimed
The repo never writes this reversal down. §3br disclaims its own ranking (**DSR 0.000, PBO 0.381,
both failing CONTROL #6**) and width selection across N books is itself an N-way search. The
*wide* books (7-9 sleeves) stay rejected on carry grounds that hold in regime U (§3ah: GER30 net
Sharpe −0.003, "the added sleeves are net-negative after carry"). This tests **2 vs 3 vs 4**, not
a return to breadth. And the Sharpe-1.60 drift portfolio is **retracted to 1.561** by its own
causality audit and is not part of this.

## Books (pre-declared, K=5)
XAU · **XAU+BTC (incumbent)** · XAU+BTC+NDX · XAU+BTC+BRENT · XAU+BTC+NDX+BRENT

BRENT is included because its **measured Switch carry is −0.09%/yr**, i.e. effectively free to
hold. §3ae valued BRENT as *tail insurance* (fail_day 13.9%→0%) — a regime-P reason that no longer
applies; here it must earn its place on net Sharpe alone.

## Costs — MEASURED at Switch, not inherited from FTMO configs
Carry %/yr on a LONG, read from `symbol_info` with mode-aware conversion:
XAUUSDs **−5.76** · BTCUSDs **−15.00** · USTECs **−5.01** · BRENTs **−0.09**.
Spreads sampled live from ticks. One-way bp = 0.75 × spread_bp; round trip = 1.5 × spread_bp.

## Pre-registered decision rules
1. **Matched-vol paired-t ≥ +2.0** vs the incumbent (rule 1.3.1 — a de-risking candidate read as a
   loss on a raw paired-t once flipped a round 0/29 → 3/29).
2. **Positive in BOTH sample halves** (rule 1.3.8 — a pooled t of +2.25 with 7/9 positive years was
   still a post-2022 artifact).
3. **Net Sharpe higher AFTER carry**, not before.
4. Rule 1.3.6: compound onto a **common business-day index** before annualising — a day-count
   mismatch inverted a book ranking once.
5. Rule 1.3.5 as restated by §3bq: **the scale is per-timeframe.** XAU H4 → `scale=1.0`; D1 sleeves
   → `1/6`. Passing 1/6 to H4 gold once collapsed its Sharpe to 0.476 against a known 1.091.
6. **Report the ranking as a LEAD, not a verdict** (K=5 search).

## Dial frontier (user's explicit choice: "report the frontier, decide after")
vol ∈ {0.06, 0.09, 0.12, 0.15, 0.20} × each book → CAGR, maxDD, Calmar, worst month.
**Sharpe is leverage-invariant**, so the frontier is about loss tolerance, not edge
(`RESEARCH-REGIMES.md:217-219`: removing prop rules bought CAGR ~8.5%→~50%, not edge).

## Control cells with KNOWN answers (the bug tripwire)
XAU alone must reproduce **~+0.92** and XAU+BTC **~+1.29** at Switch costs (§3bs). If either is
off, the harness is wrong, not the finding. This is how the `scale=1/6` bug and the ZigZag
deadlock were both caught.

## Expected outcome, recorded so it cannot be rounded up
NDX's −5.01%/yr carry will eat part of the +0.185 spread-only gap. Honest prior: **+0.05 to +0.15
net, possibly inside noise.** BRENT at ~0% carry is the more interesting candidate precisely
because it is nearly free to hold.
