# PRE-REGISTRATION — XAUUSD direction ADVISOR (calibrated decision support)

**Committed before the first model fit.** Nothing in this file may be changed after a result is
seen. If something here turns out to be wrong, the correction is recorded as a dated amendment
BELOW the original text, never by editing it — the §3as/§3av precedent.

Product: an advisor that states a direction and an adverse-move risk for XAUUSD at a 4–6 hour
horizon, refreshed for the user every 5 minutes, with probabilities that mean what they say.
**It has no order path and never will.** Trading gates (MANDATORY CONTROL #8's EV/fire, the
breakeven table) are therefore declared INAPPLICABLE here, and the reason is recorded in §7 so a
future reader does not think they were forgotten.

---

## 1. Why the baseline choice is the whole experiment

§3r's headline is **51.97% vs a 49.85% PERSISTENCE baseline = +2.12pp, 8/9 years**. Measured
read-only on `data/XAUUSD_H4_long.csv` (18,450 bars, 2015-01-01 → 2026-07-16), evaluated 2018+,
with the same expanding yearly walk-forward and a `h+8` purge, the harness reproduces it and
adds the number §3r never printed:

| horizon | model | PERSISTENCE | **DRIFT ("always up")** | Δ vs persistence | years |
|---|---|---|---|---|---|
| 4h (fwd1) | 51.68% | 49.32% | 51.39% | +2.37pp (t +3.56) | 9/9 |
| 8h (fwd2) | 51.63% | 48.84% | **52.05%** | +2.79pp (t +4.31) | 9/9 |
| 12h (fwd3) | 51.59% | 48.90% | 52.52% | +2.69pp | 7/9 |
| 24h (fwd6) | 52.23% | 49.85% | **53.56%** | +2.38pp | 8/9 |

**The model beats persistence at every horizon and LOSES to "assume gold rises" at every
horizon.** For an advisor this is the decisive comparison, because the user's default is not
"repeat the last move" — it is "gold goes up". §3ay's own rule ("always print the BASE RATE
beside the hit rate; `skill = hit − base`") applied here is damning, and it is why DRIFT is a
**co-primary baseline** below rather than a footnote.

Calibration, measured the same way at fwd2 with the repo's purged-isotonic construction:

| calibrator | Brier | **skill vs base rate** | p5 | p50 | p95 |
|---|---|---|---|---|---|
| raw HistGB | 0.2530 | **−1.37%** | 0.387 | 0.506 | 0.613 |
| isotonic | 0.2502 | **−0.26%** | 0.475 | 0.517 | 0.561 |
| Platt | 0.2500 | **−0.15%** | 0.493 | 0.514 | 0.541 |
| 5-bin OOS | 0.2505 | **−0.34%** | 0.477 | 0.513 | 0.555 |

**No calibrator reaches positive Brier skill.** This is expected, not a bug: if the true
probability only moves 0.52 → 0.54, the achievable skill is ≈ σ²/(p̄(1−p̄)) ≈ 0.04%, so
estimation noise makes it negative. The *ranking* is informative (isotonic top decile 53.40%,
consistent with §3r's 55.6% under confidence gating); the *per-bar number* is not. Hence the
product decision: **print the measured hit rate of the state, not a model probability.**

---

## 2. The declared grid — 16 cells, fixed

No cell may be added, removed or re-parameterised after any result is read.

**Family D — DIRECTION.** Decision index = H4 closes, `data/XAUUSD_H4_long.csv`.
- horizons `h ∈ {1, 2, 3, 6}` H4 bars (4h, 8h, 12h, 24h)
- arms `{BASE, SOURCE}` = `f_base(df)` / `f_source(df, m15)`, imported verbatim from
  `scripts/v5_repr_ceiling.py:94` and `:154`
- **8 cells.** `h=6 / BASE` is the **replication control (§4), not a candidate.**
- label `y = 1[close[t+h] > close[t]]`; exact ties dropped

**Family A — ADVERSE MOVE.** Decision index = H4 closes; barriers ±`k`·ATR; the path is resolved
on **M15 bars strictly after the H4 close**, because an H4 bar cannot say which barrier came
first (§3ax's own lesson).
- `k ∈ {0.5, 0.75, 1.0}` × horizon `∈ {4h, 6h, 9h}`, minus the three cells whose resolved
  fraction is below the admissibility floor (§6) → **8 cells**
- **primary: `k = 0.5 ATR`, horizon 6h** (24 M15 bars). Pre-measured: 81.8% of windows resolve,
  **11,175 decided events**, base rate **0.5068**
- arms `{BASE, SOURCE}`
- label `1` = the −k·ATR barrier is touched first, `0` = +k·ATR first, `NaN` = neither or both in
  the same M15 bar (dropped, never guessed — identical semantics to `first_touch`)
- ATR is `atr_series(d, w=20)` from `scripts/v5_range_multitf.py:74` (gap-aware true range, the
  estimator §3ai found beat all four range estimators)

**Why ATR-scaled and not §3ao's fixed 2%:** ATR14/price median is **0.348% in 2018 and 0.936% in
2026**, so `BARRIER = 0.02` is 5.7 ATRs at the start of the sample and 2.1 ATRs at the end — a
different label in each era. §3ax's "any fixed-percentage bracket on gold must be re-derived in
ATR or range units" applies to the label, not only to brackets.

**Why symmetric barriers:** the distance/geometry null is then exactly 0.5 by construction, which
is §3ay's fix for the §3ax failure where geometry alone scored AUC 0.8659. And at a 6-hour
horizon drift cannot inflate the base rate — measured 0.5068, within 0.7pp of 0.5 — so §3ay's
"positive EV with negative skill" trap (D1 base rates 0.578 → 0.669) is structurally absent.

**Declared secondary, flagged unreplicated if it alone passes:** Family H, the same direction
label on an H1 decision index at `h ∈ {4, 5, 6}` H1 bars (exactly 4h/5h/6h), BASE only, 3 cells.
**Blocked until** `f_base` gains `d1_lag` (§5.1) and `probe_lookahead` passes at `freq="1h"`.

---

## 3. Baselines — three, reported on every cell, on the identical OOS event set

| name | definition | role |
|---|---|---|
| PERSISTENCE | `1[close[t] > close[t−h]]` — the expression at `v5_xau_intermarket_accuracy.py:670` | comparability to §3r |
| **DRIFT** | constant `1` iff the TRAIN-slice up-rate > 0.5 (train-only, no peek) | **co-primary** (§1) |
| CLIMATOLOGY | the constant *probability* at the train-slice up-rate | the Brier/log-loss reference |

---

## 4. TIER C — harness replication, runs first and blocks everything

Read before any candidate cell:

- Family D: `h=6 / BASE` must land **Δ_persistence ∈ [+1.10, +3.10] pp** (§3r's +2.12pp ± 1σ of
  the measured block SE ≈ 0.96pp).
- Family A: `k=12 H4 bars, BARRIER=0.02, SOURCE` must reproduce §3ao's **dI +0.0142 ± 0.0100**
  (one alignment-null sd).

**If either fails: STOP. Do not read the other cells.** A null result elsewhere would be
uninterpretable — it could be the pipeline. This is the control that makes the rest falsifiable.

---

## 5. Leakage controls, declared before any fit

### 5.1 A latent lookahead in `f_base` on any sub-H4 frame

`scripts/v5_repr_ceiling.py:124-128` builds the daily features as
`resample("D").last()` → `reindex(method="ffill")` → `.shift(6)`. The ffill puts day *D*'s final
close on every bar inside day *D*; `.shift(6)` moves it back 6 bars, which on **H4 is exactly 24
hours and fully clears**. On an **H1** frame `.shift(6)` clears only 6 hours and **leaks ~19
hours of the same day.**

Fix: `f_base(df, d1_lag: int = 6)` with `assert d1_lag >= bars_per_day`. The default preserves
every recorded H4 result byte-for-byte. Family H must run with `d1_lag=24` and pass
`probe_lookahead` (`src/evaluation/lookahead_probe.py:84`) at `freq="1h"` before any fit.

### 5.2 Probes that run before any model is fitted

- `probe_lookahead` at `n=3500, warmup_buffer=2200` on every feature function used, plus a 400
  offset, and on the new ATR first-touch label builder.
- A per-feature `|corr(feature[t], next-bar return)|` screen. §3av's retraction was caught by
  implausibility, not by a control: a shuffled-LABEL test passed at 0.5109 while a weekly-bar
  `reindex(ffill)` was leaking Friday's high into Tuesday. **A shuffled-label control cannot see
  feature lookahead, and a truncation probe cannot see timeframe misalignment.** This screen can.
- `intrabar_leak()` (`scripts/v5_pooled_bottom_detector.py:86`) on every input series, threshold
  0.15 — XAUUSD H1 measures 0.019 and is clean, but nothing in the code enforces it.

### 5.3 Calibration must not reuse the slice that chose the model

`oos_prob_all_bars` (`v5_m15_trim_overlay.py:57`) selects between HistGB and logistic on the last
400 training rows — exactly where an isotonic calibrator would otherwise be fitted. Fitting the
calibrator on the slice that chose the model is a double-use. The declared split is:

```
train[0 : 75%]            fit the classifier
gap (h + PURGE_EXTRA)
train[~75% : ~87%]        SELECT the model class by log-loss
gap (h + PURGE_EXTRA)
train[~87% : end]         fit the calibrator
test year                 predict, then map
```

Calibrator by calibration-slice size, recorded per fold: `n ≥ 500` → isotonic; `200 ≤ n < 500` →
Platt; `n < 200` → none, and the cell is **inadmissible for Tier A**.

---

## 6. Admissibility floors (MANDATORY CONTROL #7)

A cell is inadmissible — not reported as a candidate at all — unless:
- **decided events ≥ 1,000** and **non-overlapping events ≥ 400**
- resolved fraction ≥ 0.20 for Family A
- every displayed reliability bucket has **n ≥ 200** after the merge rule (§8)
- the calibration slice reaches at least the Platt threshold (§5.3)

---

## 7. Gates — what each verdict requires

**TIER A — "usable signal".** All seven:
1. `Δ_persistence` lower 90% CI bound **> 0**
2. `Δ_drift` point estimate **> 0** and lower 90% CI bound **> −0.50pp**
3. **≥ 7/9** years positive vs persistence **and ≥ 6/9** vs drift
4. **Brier skill vs CLIMATOLOGY > 0 with its lower 90% CI bound > 0**
5. reliability monotone across displayed buckets, top-minus-bottom observed gap **≥ 4.0pp**,
   every bucket n ≥ 200, **and the top bucket above the base rate in BOTH halves** (2018-21 and
   2022-26). This is §3ap's killer imported into the calibration domain and is expected to be
   the gate that fires: §3ap had a pooled t +2.39 with *every cell negative in 2018-21*.
6. clears the **max-statistic** null at p95, and for SOURCE the **alignment** null at p99
7. the **synthetic** control returns Δ ≤ 0 and AUC ≤ 0.52 over ≥ 200 draws

**TIER B — "no measured edge". A shippable outcome, and the expected one for Family D.**
Triggered by any of: `Δ_persistence` CI includes 0; Brier-skill CI includes 0; reliability gap
< 4.0pp; `Δ_drift` lower bound < −1.0pp; the half-split in (5) fails; any null not cleared.
The artifact ships `usable: false`, the displayed number becomes the calibrated climatology, the
state machine collapses to a single NEUTRAL state, and the card states the verdict in plain
words. **Reporting Tier B takes the same effort as reporting a win.**

**TIER B-plus — "tails only".** Declared now so it cannot be a post-hoc rescue. If the pooled
statistics fail but the extreme buckets are individually informative — top bucket ≥ base + 4pp
and bottom ≤ base − 4pp, each with n ≥ 200 and above/below base in **both halves** — ship
`usable: "tails_only"`: silent in the middle, speaks only at the extremes. This is §3r's
confidence gating in the one form an advisor is allowed to use, and its failure there was an
opportunity cost of *trading* (flat 98% of the time forfeits drift), which does not apply here.

**Declared inapplicable, with reasons:** MANDATORY CONTROL #8's EV/fire and the §3ay breakeven
table (M15 0.5868 / H1 0.5494 / D1 0.5095) are *trading* controls. An advisor pays no spread and
displaces no drift. The breakeven table is quoted in the report as the **bound on usefulness**,
not as a gate. DSR/PBO are Sharpe-grid tools and §3ap already recorded DSR as vacuous for
overlay-style grids.

---

## 8. Reliability table — edges and merge rule, declared now

Bucket edges: `[0.00, 0.45) [0.45, 0.50) [0.50, 0.525) [0.525, 0.55) [0.55, 0.60) [0.60, 0.65)
[0.65, 1.00]`

Merge rule: **any bucket with n < 200 is merged into its neighbour in the direction of the base
rate, iteratively, until every displayed bucket has n ≥ 200.** Fixed in advance so the table
cannot be re-cut after the counts are seen.

Columns: bucket · mean forecast · n · share of bars · **observed frequency** · 95% block-bootstrap
CI · calibration error (observed − mean forecast) · **observed 2018-21** · **observed 2022-26**.

Also reported: Brier, Brier skill vs climatology with CI, Murphy decomposition
(reliability / resolution / uncertainty — which separates *dishonest* from merely
*uninformative*), ECE, MCE.

---

## 9. Nulls — decision table

| comparison | arms differ in capacity? | null | bar |
|---|---|---|---|
| SOURCE vs BASE, same cell | no (shuffle 8 cols, count fixed at 29) | **ALIGNMENT**: `block_shuffle` the 8 `ib_*` columns only, block 60 **and** 120, independent RNG per column per draw, 200 draws | **p99** |
| model vs PERSISTENCE / DRIFT | n/a (baseline is a fixed rule) | block-bootstrap CI on the paired per-event difference, block 60, 2000 draws | 90% CI excludes 0 |
| best cell of a declared family | yes (search) | **MAX-STATISTIC**: block-shuffle the LABEL, refit **every** cell in the family, record the maximum | **p95** |
| any Tier A candidate | — | **SYNTHETIC**: i.i.d. Gaussian, **≥ 200 draws**, **with a synthetic M15 series so SOURCE is exercised** | Δ ≤ 0, AUC ≤ 0.52 |

Two facts that make these first-class rather than ad hoc:
- the p99 = +0.0112 alignment null behind §3ao's headline **was run once in a REPL and never
  committed**; there is no artifact;
- the only null in code (`--synthetic`) **fired**: CEILING beat BASE by **+0.0319 bits with 7/9
  years on pure Gaussian noise**, which *would have passed* the pre-registered bar. §3ao's
  diagnosis was that dI is biased toward the higher-capacity arm, which is why one draw is not a
  null. It also **drops the SOURCE arm entirely**, so the repo's best AUC has never faced it.

Every draw is appended to CSV so the nulls are cached, resumable and auditable. The best-of-two
model selection stays **inside** the null, because §3ao identified it as part of the bias.

---

## 10. Other measurements declared in advance

- **Partial-bar skew.** ~20% of H4 groups have fewer than 16 M15 legs, so `ib_rv`/`ib_n`/
  `ib_hi_t`/`ib_lo_t` are partial-bar values on a fifth of training rows while live will be
  complete — a train/serve skew. Re-run the primary cell on `ib_n == 16` and report the delta.
- **Feed transfer.** `data/XAUUSD_*_long.csv` and `data/v5_runs/range_study/*` are different
  quote series: median |Δclose| **$5.84, rising to $13.33 in 2026**. Never mixed in one pipeline;
  the frozen model is re-scored across the overlap and the Brier delta reported.
- **Staleness curve.** Brier and accuracy by minutes-since-H4-close, so the card's age stamp is
  honest.
- **Panel redundancy.** `corr(p_dir, p_adverse)`; if |corr| > 0.90 the second panel is clutter
  and the report says to drop one. Plus a joint decile cross-tab to quantify how often the two
  panels contradict.
- **Operating points.** For τ ∈ {0.52, 0.55, 0.58, 0.60, 0.65}: coverage, n, observed frequency,
  95% CI, crossings/month and median dwell — with and without a ±0.02 hysteresis band. This is
  what tells the service whether "P ≥ 0.60 → UPTREND" fires twice a year or forty times a month.

---

## 11. Expected outcome, stated before the run

Family D is expected to reach **Tier B**: the edge over persistence is real and consistent
(9/9 years at 4h and 8h) but it does not beat "assume gold rises", and no calibrator has positive
Brier skill. Family A at the primary cell is the better prospect — §3ao's AUC 0.603 was measured
on this label family and the intrabar path of the last 4 hours is more plausibly about the next 6
hours than about the next 2 days — and it has **11,175 decided events against §3ao's 251**, an
SE(AUC) ≈ 0.008 versus 0.0367, a 4.5× tighter measurement.

Recording this in advance so that a Tier B verdict cannot be reframed afterwards as a surprise,
and a Tier A verdict cannot be reframed as having been obvious.

_Committed 2026-09-19. Amendments, if any, are appended below with dates._
