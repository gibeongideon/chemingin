# RULES.MD — how research is done in this repo

**Why this file exists.** ~130 pre-registered structures have been run here and the large majority
failed. The failures were not wasted — they produced the laws in [`RANK.MD`](RANK.MD) — but a
significant fraction of them were **re-runs of something already closed**, or died on a control
that should have been applied in the first hour. This file is the protocol that prevents both.

Three outcomes it is designed to produce:
1. **No idea is tested twice** unless something genuinely new makes it testable again.
2. **No result is believed** until it has survived the controls that have fooled us before.
3. **Nothing is learned and then lost** — every study ends in a written verdict, including the
   negative ones.

**Reading order for a new study:** `RANK.MD` → this file → `RESEARCH-REGIMES.md` → the relevant
`V5_FINDINGS.md` sections.

---

## 1. Before you write any code — the anti-repetition protocol

Run all four. They take ten minutes and have saved weeks.

```bash
# a. has this been done? search the full record, not just the summary
grep -niE "<your idea>|<its synonyms>" V5_FINDINGS.md | head -40
grep -niE "<your idea>" RANK.MD EXPLORATION-QUEUE.md BEST_FINDINGS.MD

# b. is it on the explicit do-not-repeat list? many closures say so in as many words
grep -niE "do not re-?(run|open|attempt)|closed on|don't retry" V5_FINDINGS.md | head -30

# c. has a script already been written for it?
ls scripts/ | grep -iE "<keyword>"
```

**d. Check the regime.** `RESEARCH-REGIMES.md` partitions every result into prop-firm-constrained
(P) and unconstrained (U). **Conclusions about book WIDTH, the risk dial and `risk_frac` FLIP
between them.** A result quoted without its regime is not a result.

### The executability gate — apply it before design, not after

> **If a retail trader cannot do it, do not design it.**

Market making is structurally impossible here (every venue is `trade_exemode = MARKET`), as are
latency arbitrage, L2 data and cross-venue execution. Check the instrument, the venue and the
account type **first**. A beautiful study of something you cannot trade is a dead study.

### When a closed idea MAY be reopened

Only one of these:
- a **new data type** arrives (the standing phrase in the closures is "needs a new data type");
- a **measurement bug** is found in the original closure — then say so explicitly and re-run;
- the **cost or venue changes** materially (e.g. swap-free gold reopened the financing question).

"I have a better model" is **not** a reason. §3dj/§3dk: the model class was never the lever.

---

## 2. Pre-registration — before the first fit

Write `data/v5_runs/<study>/PREREGISTRATION.md` and **commit it before fitting anything**. Record
the git hash in the results file. It must state:

| field | why |
|---|---|
| the hypothesis, in one sentence | if you cannot, you do not have one |
| the exact cells to be searched | the best-of-K floor depends on K, and K must not grow later |
| the pass bar, numerically | a bar chosen after seeing results is not a bar |
| the kill rule | when you stop, decided while you are still honest |
| the honest prior | written down, so hindsight cannot revise it |

**The bar is set before looking.** §3et failed its own 4/8 bar and was not deployed; §3es's
"recommended action" was then qualified rather than quietly kept. That is the standard.

---

## 3. The controls

**The canonical eight live in `V5_FINDINGS.md` → "MANDATORY CONTROLS".** Read them there; they are
not duplicated here. In short: 1 live cost · 2 vs buy-and-hold · 3 walk-forward the universe ·
4 standard error · 5 oracle ceiling first · 6 DSR ≥ 0.90 and PBO < 0.30 · 7 best-of-K floor ·
8 accuracy is payoff-blind.

**Nine more have been discovered since #8 and are equally mandatory.** They are listed here
because they currently exist only inside individual findings:

| # | control | what it caught |
|---|---|---|
| 9 | **Hold the EVENT SET fixed** in any exit-rule test | the breakeven stop flipped +0.20 → −3.55 on exposure alone (§3co) |
| 10 | **Score a FILTER against a random same-size drop**, never against the unfiltered rule | an SMA50 filter looked like +0.244 Sharpe, was a coin (§3er) |
| 11 | **Enter at CONFIRMATION**, never on the signal bar | win rate 1.000 → 0.48 on identical events (§3eq) |
| 12 | **Measure the SKIPPED sample** whenever you wait for a better price | FILLED −22.8bp vs SKIPPED +17.9bp (§3bc) |
| 13 | **One data source per study**; run a leak screen before any fit | `intrabar_leak() > 0.15`; FX D1 forward-stamps `ret[t+1]` (§3bd) |
| 14 | **Compute nulls GROSS**, and price the SIGNED maximum | a cost offset inside a zero-mean null manufactured p 0.0000 (§3bx-D) |
| 15 | **Require BOTH sources** (CME and Switch) and **BOTH halves** | a 1.58 Sharpe was 2024-26 only (§3dx) |
| 16 | **A PLACEBO at matched distance/geometry**, not a random bar | a matched placebo earned ~60% of the headline excess (§3cj) |
| 17 | **Print structural counts** beside every p-value | a "tercile" holding 116 of 873; an OVERLAP block of 983 days vs 873 |

Control 17 deserves its own line: **structural counts catch what p-values miss.** Every time a
cell looks surprisingly good, print n per side, the window width and the bucket sizes. Two of the
largest false positives here were found that way, not by a test.

---

## 4. Null design — the single most load-bearing piece of method

The default null is the **max-stat null**:

- **circular-shift the OUTCOME** by ≥ 20 days (never the features — the labels stay where they
  are, so the null is exactly "the structure carries nothing");
- **re-run the WHOLE search** on each draw and take the **maximum** statistic;
- the side is chosen **inside** the null (use `|t|`), because the search was free to choose it;
- compute it **GROSS** (see control 14);
- ≥ 2,000 draws.

**Record the trial array.** DSR is vacuous without it — agreeing grid cells give DSR 1.0000 at any
K, and §3bp's 0.743 is permanently unreproducible because its array was never saved.

**Max-stat qualifies the FAMILY, not the arg-max.** If the family clears, choose *within* it on
the statistic the product consumes — not on the biggest t. §3eq's family cleared while its arg-max
pointed the opposite way, in an untradeable cell.

---

## 5. The trap checklist

Every one of these has produced a wrong answer in this repo. Check them by name.

| trap | the rule |
|---|---|
| **magic-on-close** | a deal carries the magic of the ORDER that produced it; server SL/TP closes carry **magic 0**. Collect position ids whose OPENING deal has the magic, then take all their deals |
| **server clock ≠ host clock** | broker time is UTC+2/+3. Three dead gates came from one such mistake. Never use host UTC for a market-time test |
| **local-hour wrap** | comparing a bar's local session hour wraps at midnight and faked t = 14.5. Convert session opens INTO UTC instead |
| **cross-timeframe resample** | a bar stamped `t` is not knowable until `t + duration`. Stamp at the close and lag one bar. Faked AUC 0.901 |
| **multi-broker splice** | FTMO was spliced into the H4 files from 2023-06-28 (25 of 38 pairs). Provenance is NOT testable by timestamp checks |
| **recency weighting** | destroys power — corr +0.364 → +0.140 — by cutting N_eff to 33, and manufactures patterns |
| **window extremes** | "at the N-day high" is a tautology that inverts with the window's return sign. Split by outcome first |
| **dry runs write live state** | a dry `v5_zigzag_ftmo.py` run still writes the state file; a notifier then announced an order that was never sent |
| **PEP 701 f-strings** | the VPS serves Python **3.10**. Nested same-quote f-strings parse locally and crash there. `ast.parse(..., feature_version=(3,10))` |
| **shared bridge** | one MT5 session per port. **Never call `shutdown()`**; yield to peers via their `/tmp/*.lock`; retry 3× on `-6 Authorization failed` |

---

## 6. The cheapest order to kill an idea

Do them in this order. Most ideas die at step 1 or 2, for almost no cost.

1. **Oracle ceiling** (control 5). Trade the label with perfect hindsight. If the oracle is flat,
   nothing downstream can work — stop here.
2. **The cost wall** (control 1 + `RANK.MD` #16). Compute the breakeven hit rate for the
   timeframe: M15 needs 0.587, H1 0.549, D1 0.510. If the best honest accuracy here (~0.59) cannot
   clear it, stop.
3. **Versus buy-and-hold** (control 2), not versus zero.
4. **The placebo / random control** (controls 10 and 16).
5. **Only now** fit a model, and apply the max-stat null to the whole search.

---

## 7. Deployment protocol

| rule | why |
|---|---|
| **Demo only.** The executor must refuse a non-demo account | every live-order script here checks `trade_mode == 0` |
| **New magic, own state file, own log** | shared state silently mixes two accounts and corrupts dedupe cursors |
| **An A/B changes exactly ONE variable** | otherwise the live record cannot attribute the difference |
| **Every gate FAILS OPEN** | a data problem must not silently shrink the book to nothing |
| **Diff VPS against HEAD before `scp`** | the VPS has been edited out-of-band; blind copying has reverted live fixes |
| **Run the wrapper once by hand after editing it** | a three-word `--bots` argument made argparse exit 2 on **477 consecutive runs** and no alert reached the phone |
| **Pick an unused timer second on that bridge** | `:00` hourly, `:27`, `:40`, `:45` are taken on 18814. Collisions starve other jobs |
| **Never `systemctl stop mt5-terminal.service`** | its `ExecStop` carries a bare `pkill -f terminal64.exe` + `wineserver -k` |
| **Never hand-trade a bot account** | manual trades once drove equity −37%, the guard flattened the bot and latched a permanent halt |
| **A pre-registered kill rule ships with the deployment** | not decided later, when it is losing |

---

## 8. Documentation — what every study owes the next reader

A study is not finished when the number is computed. It is finished when it is written down.

### 8.1 Always — a `V5_FINDINGS.md` section

Next letter in sequence (`§3ev`, `§3ew`, …). It must contain:

- **the question**, in the user's own words where they asked it;
- **the data**: source, span, bar count, cost, swap;
- **the controls applied**, by number;
- **the numbers, including the nulls** — a verdict without its null is an opinion;
- **the verdict**, stated plainly, negative results first-class;
- **the kill rule or the action**, and whether anything was deployed;
- **what would change the answer** — the condition under which it may be reopened.

> **Negative results are the main product.** ~130 structures failed and that is what bought the
> laws in `RANK.MD`. Write them up with the same care as a win.

### 8.2 Always — a memory entry

One durable fact per file, with the measured number in the description so it is findable without
opening it. Link related entries. Update the index line.

### 8.3 If load-bearing — a `RANK.MD` row

Only if it changed a deployment or closed a family. Everything else stays in `V5_FINDINGS.md`.

### 8.4 If deployed — a `README.MD` magic-table row

Magic, system, venue/bridge, status, and the § that justifies it.

### 8.5 Housekeeping

- **New `data/` folders go into `.gitignore` at creation.** Research output is regenerable; it is
  not source. Untrack with `git rm -r --cached`.
- **When a finding is refuted, do not delete it.** Move it down and say what killed it — the way
  §3er refuted §3es's recommended action, and §3et qualified §3eu's headline.
- **Update the stale pointer.** If a summary file no longer covers the record, mark it superseded
  rather than leaving it to mislead.

---

## 9. When to stop

Stop and write it up when any of these is true:

- the **oracle is flat** — no detector can rescue it;
- the result **fails in one of the two sources**, or in one of the two halves;
- the effect **does not survive the cost wall** at the live quote;
- you are choosing between cells whose differences are **inside the base strategy's own CI**;
- you have started **adding parameters to rescue a dying result** — that is the point at which a
  study becomes a search, and the best-of-K floor rises faster than the result does.

**The honest prior in this repo is that the next idea will fail.** That is not pessimism, it is
the base rate — and it is why the cheap kill order in §6 and the write-up in §8 matter more than
any individual hypothesis.
