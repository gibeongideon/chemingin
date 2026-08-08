# Improving the 4-asset book — research plan & rationale (2026-08-08)

## The baseline we must beat

Deployed FTMO book: champion trend recipe on **XAU + BTC + NDX + BRENT**, equal-class
risk, vol-targeted to 9%, with the drawdown scaler.

| metric | value |
|---|---|
| Sharpe (2018+) | **1.4641** |
| CAGR | +14.20% |
| maxDD | −12.69% |
| realised vol | 9.04% |
| Calmar | 1.12 |
| **mean pairwise sleeve correlation** | **0.071** |

Per-year Sharpe: 2018 **−0.11** · 2019 +2.15 · 2020 +2.48 · 2021 +1.22 · 2022 **−0.26**
· 2023 +2.20 · 2024 +1.40 · 2025 +2.03 · 2026 +1.97

**Only two losing years: 2018 (−1%) and 2022 (−2%).** Both were trend-hostile. That is
where the available Sharpe actually lives — the good years are already excellent.

## The central thesis of this study

The book is **4 assets × 1 strategy**. Prior work established that adding more *assets*
to the same strategy fails — breadth 4→8→14→22 gave Sharpe 1.67→1.44→1.31→**1.04**,
monotone down in both half-samples, because extra markets have weaker trend-IC and
equal-risk weighting waters down the strong sleeves.

The unexplored axis is therefore **more strategies on the same assets**. This is what the
literature supports: Asness/Moskowitz/Pedersen (*Value and Momentum Everywhere*, JF 2013)
show cross-sectional and time-series momentum are complementary; Koijen/Moskowitz/
Pedersen/Vrugt (*Carry*, JFE 2018) show carry is a distinct, positively-rewarded factor in
every asset class with low correlation to momentum. A CTA book running one signal family
on four markets is leaving style diversification on the table.

Supporting fact: at 0.071 mean pairwise correlation the *asset* diversification is already
near-exhausted. Further gains have to come from *signal* diversification, better risk
allocation, or rescuing the two bad years.

## The eight approaches under test

Each is scored the same way: **paired daily-delta vs the baseline book**, never an
unpaired Sharpe comparison (Lo SE at SR 1.46 over 8.6y is ~0.5 — an unpaired test cannot
tell 1.46 from 1.9).

1. **Carry sleeve** — we now have *measured* carry from live FTMO swaps: BRENT **+9.6%/yr**
   (backwardation pays you to be long), XAU −6.6%, NDX −6.9%, BTC **−30.4%**. Two
   hypotheses: BRENT's positive carry is unexploited, and BTC's brutal negative carry means
   the crypto sleeve should be *smaller* than equal-class gives it.
2. **Cross-sectional momentum sleeve** — the best diversifier lead this project has ever
   found (Sharpe +0.74, half-samples +0.72/+0.76, correlation to this book **+0.15**), but
   never properly walk-forwarded. Re-select lookback/topN/hold each January on trailing
   data only, then add as a 5th sleeve at 5-25% of book risk.
3. **Trend-speed diversification** — the champion is mid-speed EWMAC + fast breakout;
   there is **no slow sleeve**. Practitioner CTAs run fast+medium+slow because different
   speeds pay in different regimes (Moskowitz/Ooi/Pedersen 2012; Baltas & Kosowski). Tests
   specifically whether speed diversity rescues 2018/2022.
4. **Correlation-aware risk allocation** — the book uses *static* equal-class risk, which
   assumes constant correlation. It isn't: in crises everything correlates and the book
   carries more risk than intended. Test causal EWMA correlation estimation, diversification-
   ratio scaling, and Ledoit-Wolf-shrunk risk parity, all walk-forward.
5. **Profit capture & re-entry** (the requested idea) — reframed as a **rebalancing /
   volatility-harvesting** overlay rather than market timing, since naive profit-take was
   already disproven (§3e). Systematic rebalancing of volatile, low-correlation assets
   harvests a real premium (Fernholz, stochastic portfolio theory). Tests trim-on-stretch
   with pullback re-entry, fixed-band rebalancing, and per-sleeve vol-scaled profit-takes —
   with full round-trip cost *and* swap on every adjustment, because overlays like this
   usually die on turnover.
6. **New diversifiers by marginal contribution** — breadth failed because assets were added
   indiscriminately. Instead rank every panel candidate by *marginal* dSharpe and correlation
   to the existing book, add greedily, and walk-forward the ranking. Special attention to
   **fixed income (UST 2/5/10/30Y)** — a trend book with no bond sleeve is unusual for a CTA
   and bonds are the classic crisis diversifier.
7. **Defensive regime overlay aimed at 2018/2022** — cut exposure when trend quality decays
   (trailing hit-rate, cross-sectional trend agreement, realised-vs-forecast vol divergence,
   equity/credit stress proxies).
8. **Sizing & signal-shape** — no new assets. Forecast-to-position mapping (conc^1.5 vs
   linear/binary/conc^2), better vol estimators (HAR-RV, Parkinson/Garman-Klass range-based,
   which are more efficient than close-to-close), buffer width, and the drawdown-scaler shape.

## Controls (any result skipping one is invalid)

- **Paired t vs baseline**, plus per-year deltas, plus explicit 2018/2022 deltas.
- **Block-shuffled null** (100 shuffles, ~5-day blocks): preserves the overlay's marginal
  distribution and autocorrelation while destroying its timing. Must beat the 95th pct.
  A matched-*constant* control is useless — the book is vol-targeted, so scaling exposure
  is Sharpe-neutral.
- **Swap/financing charged** wherever holding period or exposure changes (−2.59%/yr on the
  book; discovered only 2026-08-07 and absent from most historical numbers).
- **Walk-forward all selection** — re-choose sleeves/weights/params each January on trailing
  data only. Naive Sharpe-weighting of sleeves was a measured lookahead illusion here.
- **Trials tried + median trial result**, not just the best cell. Deflated Sharpe and PBO
  across the whole study at the end.

Promotion gate: dSharpe ≥ +0.15 **and** paired t ≥ 2.0 **and** better in ≥6 of 9 years
**and** does not worsen 2018 or 2022. Then three adversarial verifiers (reproduce /
lookahead+selection / robustness) must fail to refute it, 2 of 3.

## Honest prior

The book is already strong and near the asset-diversification frontier, so most of these
will fail — that is the expected outcome and the controls exist to make failure visible
fast. The two I would bet on are **cross-sectional momentum** (a genuinely different,
already-measured, near-uncorrelated signal) and **carry** (newly measurable from the swap
discovery, and the BTC −30.4%/BRENT +9.6% asymmetry is large enough to matter on its own).
Correlation-aware sizing is the most likely *risk* improvement even if it adds no return.
Profit-capture/re-entry most likely fails on turnover — but the rebalancing-premium framing
is materially different from the disproven version, so it earns one honest test.
