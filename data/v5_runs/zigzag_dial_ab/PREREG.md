# ZigZag Y2UP + CHAMP size dial: forward test on fresh months (pre-registered 2026-11-28, before deployment)

**Origin.** §3fa #4. As half-size dials on ZigZag variant B + breaker (causal signal bar, event set fixed):
- Y2UP: dSR +0.113 (CME, 99.5th pctile) / +0.117 (Switch, 94.5th)
- CHAMP: +0.107 (97.5th) / +0.089 (94.0th)
- Both positive in both halves in both sources, but each misses Switch's 95% bar; family p 0.135 / 0.325
- So: a lead, to be decided on months the study never saw.

**Rule (fixed now, BEFORE any combined number is computed).**
- **Multiplier.** The trade size multiplier is the PRODUCT of the two dials:
  - CHAMP: × 0.5 when the champion gold forecast (last CLOSED H4 bar of `data/XAUUSDs_H4_switch.csv`) < 0.5
  - Y2UP: × 0.5 when FRED DGS2 (latest published value, at least t−1) > its value 63 business days earlier
  - so 1.0 / 0.5 / 0.25
- **Read time.** Both are read at the signal bar's close.
- **Fail-open.** Each dial fails open at × 1.0 (stale or missing data).

**Arms.** FTMO-Demo 1515427365, the same H1 fires:
- B (360608, flat size), the control
- V (360613, vol taper)
- **D (360622)** = B's config + the two dials above (no vol taper), so D vs B isolates the dials

**Readout.** Matched-vol daily paired t, D vs B, plus the event-set match check (D must take exactly B's trades).
- **Reviews:** at 50 and 100 closed trades.
- **Promote** the dials to the live ZigZag configs only if D ≥ B at matched vol at 100 trades AND the event sets matched.
- **Abandon** if the matched-vol t ≤ −1.5 at either review.

**In-sample note (descriptive only).** The combined dial is measured once on the cached 2012-26 / 2018-26 data after this file is written. It is NOT a test and cannot change the rule above.

**In-sample result (computed after this file was written; descriptive, NOT a test):** combined product dial on the cached data:
- CME 2012-26: dSR +0.181, 99.0th pctile vs shuffled multipliers, halves +0.054/+0.259, mean size 0.57
- Switch 2018-26: +0.173, 96.0th pctile, halves +0.211/+0.115, mean size 0.62
This is the expected effect the forward arm will be judged against. It is not evidence by itself: the two gates were picked from a family of six.

## Venue change (2026-11-28, before any live pass)
- **What moved.** At the user's choice, the arms run on the Goat Pay Later account 315531834 (bridge 18820, `XAUUSD.x`, aux `EURUSD.x`/`GBPUSD.x`/`USDJPY.x`):
  - B-goat 360624 (control, flat size), `configs/v5_zigzag_goat_b.json`
  - D-goat 360625 (B + dials), `configs/v5_zigzag_goat_d.json`
- **Trailing.** Both are in the every-minute trail watcher.
- **Allowance.** `allow_real_login` 315531834 allows the real-flagged prop account.
- **FTMO arm D.** The FTMO arm D files were written but never enabled. FTMO arm V was added to the watcher (it had only been trailed hourly). Readouts and decision rules unchanged.
