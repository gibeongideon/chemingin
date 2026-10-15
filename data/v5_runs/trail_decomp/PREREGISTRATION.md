# PRE-REGISTRATION — trailing-stop decomposition + the lookback-window question

Written and committed **before the first fit**. Gate A was run first because it is pure
arithmetic on already-recorded numbers and cannot be influenced by a result.

## The two questions

1. **Does ratcheting beat not-ratcheting on a signal that works?** Never asked here. §3aa's 64
   exit cells contained **no ratchet at all** and resolved on closes, never the path. The only
   apparent XAU trail sweep (§3f, `V5_FINDINGS.md:394`) is **confounded** —
   `scripts/v5_multi_tf_trend.py:117-118` passes `sl_atr=trail_atr, trail_atr=trail_atr`, so the
   protective stop and the trail distance moved together.
2. **Which lookback window best identifies a top/bottom?** Never run as an arm: every prior study
   fed windows 5/10/20/60 into ONE model simultaneously.

**The confounded number is LIVE.** `configs/v5_xau_maven.json` ships
`bots.champ.params = {sl_atr: 3.0, trail_atr: 3.0}` on Maven magic 360571, with
`trail_activation_atr` unset so it defaults to **1.0** (`src/v5/xau_trend.py:70`).

## GATE A RESULTS — run first, binding on everything below

### A3 — the cost convention was wrong, and it is fixed before any trail cell

`V5_FINDINGS.md:2207` states "$0.448 (= **1.94bp one-way** on median gold $1,733)", and
`RESEARCH-REGIMES.md:79` gives one-way bp = 0.75 × spread_bp. So **$0.448 is the SPREAD**:

| | |
|---|---|
| spread | $0.448/oz = **2.592 bp** at median $1,728 |
| one-way (×0.75) | **1.944 bp** ✓ matches the filed 1.94 |
| **round trip** | **3.888 bp = $0.672/oz** |

`scripts/v5_zigzag_atr_bracket.py:122` subtracts $0.448 once and calls it a round trip —
**understated by 1.5×**. §3bl ran at two-thirds of its own stated honest cost. It changed nothing
there (everything failed anyway) but **a ratchet has materially higher turnover than a fixed
bracket**, so this is fixed first.

Cost in bp **fell** as gold tripled: round trip 5.75 bp (2015) → 3.72 (2022) → **1.43 (2026)**.
Reported to remove an excuse, never as a licence — delivered skill on this family is 0.000, so
cost was never the binding constraint.

### A1 — answerability, and it KILLS two windows before they are written

Honest **decided, non-overlapping** counts on the detector's live window (2018-05-15+):

| window | horizon | non-overlapping | SE(AUC) | best-of-5 floor |
|---|---|---|---|---|
| §3ay's (16 bars) | 16 | 2,985 | 0.0105 | +0.0122 |
| same-day | 24 | 1,990 | 0.0129 | +0.0150 |
| 2 days | 48 | 995 | 0.0182 | +0.0212 |
| 5 days | 120 | 398 | 0.0288 | +0.0335 |
| **10 days** | 240 | 199 | 0.0408 | **+0.0474** |
| **20 days** | 480 | 99 | 0.0580 | **+0.0674** |

> **§3ay's filed SE of 0.0035 was computed on ALL labelled bars at 16-fold overlap — a CONTROL #7
> violation. The honest value is 0.0105, exactly 3.0× the filed one.** This makes §3ay's verdict
> *more* secure (0.5003 is even more firmly zero) and inflates every budget here.

**PRE-REGISTERED KILL:** the **10-day and 20-day** windows are **inadmissible** — a best-of-5
noise floor of +0.047/+0.067 AUC exceeds any effect this repo has ever delivered on gold.
C2 runs on **same-day / 2-day / 5-day only, K=3**, floor **+0.0335** at the widest.

### A2 — the exit ceiling on the detector family

From §3bl's own oracle table (`data/v5_runs/zigzag_atr/z1_z3_cells.csv`): oracle EV/fire +20.49
to +33.79 bp; `ev_minus_hold` −0.94 to −1.43 bp; oracle-to-honest gap **33.29 bp**.

> **CEILING = 1.43 / 33.29 = 4.3%.** The most any exit change can recover on the bottom detector.
> A trail may be worth testing on the CHAMPION; it cannot revive the detector. Gate D is a
> **closure**, not a candidate, and this number is quoted beside its result.

## Declared arms — fixed, no additions

**Gate B — three 1-D profiles through the live incumbent, K = 12 (incumbent counted once):**

| axis | cells | held fixed |
|---|---|---|
| `sl_atr` | 2.0, **3.0**, 4.0 | trail 3.0, activation 1.0 |
| `trail_atr` | 1.5, 2.0, **3.0**, 4.0, 6.0 | sl 3.0, activation 1.0 |
| `trail_activation_atr` | 0.0, **1.0**, 2.0, 3.0 | sl 3.0, trail 3.0 |

Secondary: `exit_mode ∈ {trail, flip, sltp}` + a time-exit control.
Side arm: **S1** long-only+trail · **S2** S1 plus a standalone short-on-top closure · **S3**
long/short+trail.

**Gate C1** — outcome-split drift test, w ∈ {same-day, 2d, 5d, 10d, 20d}, descriptive only.
**Gate C2** — day-scale boxes, **K=3** (24/48/120 bars) after the A1 kill, conditional on C1.
**Gate D** — oracle-trail / oracle-bracket / oracle-hold, then the same on the honest detector.
**Gate E** — the shootout: every arm on one table, ranked.

## Pass gates

**Gate B — a SHAPE test, not an argmax:**

> The incumbent is retained unless a cell beats it at **t@matched-vol ≥ +2.5**, AND **both
> neighbouring cells on the same axis** also exceed the incumbent, AND it holds in **both sample
> halves**, AND **PBO < 0.30**. A single-cell peak with neighbours below the incumbent is
> recorded as noise and the incumbent stands.

A sharp peak is a **failure** signal here: §3f killed H4/med precisely because it peaked
(0.82/1.02/**1.27**/1.14/1.02), and §3ab found parameter-cloud averaging *hurts* the champion.
**A plateau qualifies.**

**Gate C1 KILL:** if P(the window's extreme sits at the current bar) **inverts** when split by
the sign of that window's return, at every window, the question is answered descriptively and no
classifier is warranted. Three-for-three so far (Mon/Fri; close-position U-shape; session timing,
which inverted 0.547 → 0.045).
**Gate C2:** a window is a result only at **ΔAUC ≥ +0.025** over *both* 0.50 and §3ay's pooled
H1 model (0.5003).
**Gate E verdict rule:** the winner is the arm with the highest matched-vol paired-t vs the
incumbent that *also* clears both halves and PBO < 0.30 and whose axis-neighbours beat the
incumbent. **If none clears, the incumbent wins** — a decision-grade result, not a failure.

## Honest expectations, recorded so they cannot be rounded up

- **Most likely Gate E winner: row 0, the incumbent**, on a flat trail surface.
- **S3 (short on detected tops): standalone Sharpe −0.1 to −0.6, under 2% chance of clearing
  +0.20.** §3an: 15 causal bear switches, **0 positive**, best −0.106 vs +0.465 required, inside a
  4.3-year bear where an unconditional short earned +0.475. §3am: short-only H4 **gross** −0.211,
  so cost is 6.6% of the gap — not a cost problem. Run because the user asked for the comparison;
  a measured refutation beats a cited one.
- **C1 inverts** (3-for-3 prior) and C2 never runs.
- Realistic prize: **three permanent closures and one validated live parameter**, not a new
  strategy.

## Method rules that bind

M15 path resolution with `dec_bar_minutes` (§3bj — convention choice was worth 0.65 Sharpe on
daily bars; **re-measure trail ambiguity, do not inherit the bracket's 0.1–0.4%**) · ATR frozen at
entry · ratchet AFTER the exit check, never before · never seed the peak from a pre-entry bar ·
matched-vol before any paired-t (rule 1.3.1) · bootstrap GROSS and price the SIGNED maximum
(§3bf) · base rate beside hit rate, EV/fire against the displaced drift (#8) · common
business-day index before annualising (rule 1.3.6) · H4 rebuilt from M15 only, never the spliced
`XAUUSD_H4_long.csv` (agreement 0.04 from 2023-06-28) or `XAUUSD_M30_long.csv` (0.31 in 2026).

**Scope:** clean frames end **2026-06-15, 122 days stale**. The window ends there by choice.
