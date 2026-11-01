# Futures and carry — the practical reference

Measurements: `V5_FINDINGS.md` §3by, §3cr. Script: `scripts/v5_futures_lab.py` (re-run it any
time: `python scripts/v5_futures_lab.py --port 18815`). Data: `data/v5_runs/futures_lab.csv`,
`futures_decision.csv`.

**Why this matters at all:** financing costs the live book **−3.79%/yr = −0.29 Sharpe**, larger
than any signal result in this project's history. Swap-free instruments are the only measured way
to move it.

---

## 1. The decision table — measured 2026-11-01, Switch Markets

| sleeve | spot | spot swap | future | future carry | **saving** | rollable? | verdict |
|---|---|---|---|---|---|---|---|
| silver | XAGUSDs | −9.29%/yr | SI.Z6s | +0.34%/yr | **+9.63pp** | **NO** | best economics in the project, unusable |
| gold | XAUUSDs | −6.13% | GC.Z6s | −1.59% | **+4.54pp** | **NO** | §3by's lead, not deployable past Dec 2026 |
| dow | US30s | −6.13% | US30.Z6s | −2.28% | +3.85pp | **NO** | |
| nasdaq | USTECs | −5.24% | NQ100.Z6s | −3.63% | +1.60pp | **NO** | |
| sp500 | US500s | −4.35% | ES500.Z6s | −2.77% | +1.59pp | **NO** | |
| **brent** | BRENTs | −0.08% | BRN.F7s | −3.07% | **−2.98pp** | **yes** | **swap-free COSTS here** |
| **wti** | WTIs | −0.06% | CL.Z6s | −5.08% | **−5.02pp** | **yes** | **swap-free COSTS here** |

> **5 of 7 futures beat their spot. 0 of those 5 can be rolled. Both that can be rolled cost more.**
>
> The inversion is not coincidence. The rollable contracts are energy and softs, whose **spot**
> swap at this venue is already ~0, so swap-free has nothing to save — while the basis still
> charges 3–5%/yr.

**Tracking quality** (for when this reopens): financials corr **0.9987–0.9995**, TE **0.53–1.84bp**.
Energy is materially worse: corr 0.9945, TE **5.6bp**.

---

## 2. Roll status — who has a successor

| rollable | contracts listed |
|---|---|
| BRN | Q6 (2026-08), F7 (2027-01), G7 (2027-02) |
| LCC | Z6 (2026-12), H7 (2027-03), K7 (2027-05) |
| CC | Z6, H7 |
| CL | Z6 (2026-12), F7 (2027-01) |
| KC | Z6, H7 |

**Orphans — one contract, nothing to roll into:**
`CT · DE40 · DX · ES500 · EU50 · FRA40 · GC · HG · JP225 · LRC · NG · NQ100 · SB · SI · UK100 · US30`

⚠️ `expiration_time` reads **0** on every contract here, so expiry is inferred from the month
code (Z=Dec, F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun, N=Jul, Q=Aug, U=Sep, V=Oct, X=Nov) and
assumed at the 25th. That assumption only scales the forward-carry number; it does not change any
verdict above.

---

## 3. How to read a basis without fooling yourself

**The financing in a future is in the BASIS, not in a swap line.** A future above spot decays
toward spot by expiry, and that decay *is* the carry — merely invisible. Two estimators, and you
need both:

| estimator | formula | what it measures | when it lies |
|---|---|---|---|
| **realised** | observed basis decay, annualised | what the basis actually did | **on short windows it measures the CURVE, not convergence** |
| **forward** | −basis_now × 365 / days-to-expiry | what holding from here costs | needs an expiry date; insensitive to curve moves |

**The worked example that proves the point.** Brent over a 30-day window gave a *realised* carry
of **+32.43%/yr** and WTI **+34.31%** — nonsense as carry, and entirely the oil curve steepening.
*Forward* carry on the same data: **−3.07%** and **−5.08%**. Gold, on a 101-day window where the
realised estimator has room to work, gives **−4.61% realised / −1.59% forward**, reproducing
§3by's independent **−4.13% / −2.60%**.

> **Rule: below ~60 days of overlap, quote the forward estimator only.**

### ⚠️ The comparison bug — do not repeat it

The first version of this analysis compared each spot swap against **zero** and concluded
*"spot pays carry → swap-free is a saving"* for **all seven sleeves**, Brent and WTI included.

**The alternative to paying a swap is not paying nothing — it is paying the basis.** The
comparison is always `spot_swap` vs `futures_forward_carry`, pairwise. Making that correction
reverses the verdict on exactly the two sleeves that are actually available.

---

## 4. The one action, and it is not research

Everything above is blocked on a question only the broker can answer. Suggested wording:

> Do you list successor contracts for your financial futures — specifically **GC (gold)**,
> **SI (silver)** and the index futures (**NQ100, US30, ES500**)? At present only the Z6
> (December 2026) contract appears in the symbol list for each, with no later month, whereas your
> energy and softs contracts (BRN, CL, CC, KC, LCC) do list forward months. Will the March/June
> 2027 financial contracts be added, and if so when do they normally appear?

**If the answer is yes**, this reopens immediately and **silver joins gold** — at +9.63pp, the
largest single carry saving measured anywhere in this project.
**If the answer is no**, the futures branch is finished at Switch, and the remaining question is
whether any other retail broker lists continuous financial futures.

---

## 5. Re-measure when

- **Monthly**, and always before acting on any number here — the basis moves, and it has already
  gone **+1.532% → +0.491% → +0.239%** on gold across three measurements as December approaches.
- **Whenever a new contract appears** in the symbol list — that single event unblocks section 4.
- **After any broker/account change.** Carry conclusions are **per-venue**: Brent carried
  **+9.6%/yr (positive, it earned)** at FTMO and **−0.08%** at Switch. The same instrument name
  tells you nothing about its carry at a different firm.

`futures-roll-check.timer` already runs daily on the VPS and is the natural place to wire the
"has a successor appeared?" alert.
