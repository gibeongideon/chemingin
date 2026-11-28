# Gold-sleeve META TRIM: live A/B (pre-registered 2026-11-28, before deployment)

**Origin.** §3fa #1 / Phase D: the §3p champion meta-label trim at b = 0.5.
- Matched-vol t +2.73 (M15→H4 2018+), +0.89 (OANDA 2012+).
- Inside the XAU+BTC book: SR 1.118 → 1.183, maxDD −12.5% → −10.7%, t +1.33.
- Positive everywhere, never significant in the book, hence a forward A/B.

**Design.** Two gold-only book arms on FTMO-Demo 1515427365 (bridge 18814), run back to back in ONE job on the same pass and the same data.
- **CONTROL (magic 360619):** the deployed XAU champion sleeve (`configs/v5_trim_ab_control.json`).
- **TRIM (magic 360620):** identical, plus `meta_trim` {XAUCHAMP, b 0.5} (`configs/v5_trim_ab_trim.json`).
- **Shared settings:**
  - signal file `data/XAUUSDs_H4_switch.csv` (the Switch book's feed), the same staleness gate as the GFT job
  - vol dial 0.20, chosen for lot resolution on a demo: at 9% a 100K gold sleeve is 0.02-0.03 lots and the trim would be lost in 0.01-lot rounding
  - guards loose (never bind): daily 10%, halt 20%, no phase target; compliance SL 3%
- **Model.** `scripts/v5_trim_model.py`, ONE yearly fold of §3p's walk-forward. Fitted on the VPS (its scikit-learn version) on Switch H4 through 2025-12-31, used through 2026-12-31, refitted each January. Fail-open: an error leaves the trim arm untrimmed (= control).

**Readouts (recorded now so they cannot be chosen later):**
1. **FIDELITY (monthly).** The live multiplier must equal an offline re-score of the same bars (|diff| < 0.01). Mean multiplier is expected near 0.73 (1 − 0.5 × mean p_bad). Lot ratio trim/control should track the multiplier.
2. **PERFORMANCE (at 6 and 12 months).** Matched-vol paired t of the daily P&L (trim vs control), maxDD both arms. The backtest effect is about +0.6%/yr at equal vol, so 12 months cannot reach t 2. The pre-registered decision is therefore:
   - **(a) promote the trim to the live XAU+BTC books** only if, after 12 months, the matched-vol t is ≥ 0 AND trim maxDD ≤ control maxDD AND fidelity held;
   - **(b) abandon** if the matched-vol t is ≤ −1.5 at any review, or fidelity fails twice.
