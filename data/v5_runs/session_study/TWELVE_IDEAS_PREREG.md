# Pre-registration: the user's twelve setups

Declared 2026-09-30 before any cell was scored.

## Already closed — not re-run, and why

| # | idea | closed by |
|---|---|---|
| 2 | Fade the spike after US data | §3bf. Exactly this test: 08:30/10:00/14:00 ET anchors, 15/30min initial window, FADE rule, big-reaction filter. Best gross of 72 cells +1.769bp against a 2.34bp round trip; 0 of 72 cleared cost. **"First move is the fake move" is the WRONG SIGN** — continuation beat reversal. |
| 4 | Overnight effect (long Asia/Europe, flat NY) | §3be measured the session drifts directly: ASIA **+1.68bp (t +1.96)**, LONDON +0.47, NY −0.17. The claimed pattern IS there in sign — but Asia's edge is below the 2.34bp round trip, so it does not pay. The index/ETF version was separately **verified dead on the broker's own symbols (−4.65)** when the cash-index close→open window turned out not to exist on CFDs. |
| 8 | Month-end / quarter-end flows | Seasonality was one of five directions in a 64-trial round; 0 keepers. |
| 12 | Gold/silver ratio as filter | `gold-silver-spread-disproven`: SILVER correlates 0.79 but the rolling z-spread edge is pre-2015 only and dead out-of-sample from 2017. |

## Tested here — eight cells, one per idea, declared

All on the provenance-clean M15 series (§3bb: XAUUSD's consistent core is M15+H1; H4 and M30 are
spliced). All DST-correct via `zoneinfo`. One rule per idea, no tuning, no parameter sweep — the
point is to test the ideas as stated, not to search around them.

| # | idea | rule as tested | fires/yr (est) |
|---|---|---|---|
| 1 | Asian range sweep/reversal | mark 00:00–07:00 UTC range; London (07:00–12:00) trades past one side then closes back inside → enter the reversal, target the opposite side, exit London close | ~120 |
| 3 | London PM fix | LBMA auction 15:00 London. Measure the 14:00–15:00 drift, trade the OPPOSITE way 15:00–15:30 | ~250 |
| 5 | Gold vs dollar + real yields divergence | gold up while BOTH DXY and 10y yield up (or all three down) over 5 days → trade gold back toward them, 5-day hold | ~90 |
| 6 | Round-number stop runs | price trades 3–8 USD past a level ending in 00/50/25 then the M15 closes back through it → trade the reversal, 4h hold | ~400 |
| 7 | Monday gap fill | Sunday-open gap vs Friday close → enter toward the gap at the open, exit on fill or after 4h | ~50 |
| 9 | COMEX expiry pinning | the 4 sessions before the monthly options expiry (4th-to-last business day of the prior month) → RANGE: fade moves >1 ATR from the period's open | ~12 |
| 10 | Anchored-VWAP reversion | VWAP anchored at 07:00 UTC using tick_volume; ≥2σ away and not trending → revert, exit at VWAP touch or session close | ~150 |
| 11 | NR7 / inside-day breakout | narrowest daily range of the last 7 → next day trade the break of that day's high/low **in the weekly-trend direction** | ~35 |

## Gates, applied mechanically

1. **Benchmark is HOLDING GOLD, not zero** (§3bg/§3bh). Gold's drift over this sample runs
   t +5.97 at 5 days and **t +12.00 at 20 days**. The reported statistic is excess over the
   unconditional return at the matched horizon.
2. **Null on GROSS, SIGNED maximum, over exactly these 8** (§3bf: a net-of-cost series under a
   zero-mean null manufactures significance in proportion to n, always negatively).
3. **Cost is 2.34bp round trip**, the live Maven XAUUSD spread, floored — CSV spreads understate
   20–50x.
4. **Skipped sample** reported for every cell (§3bc): what a passive long earned when the trigger
   did NOT fire. A trigger underperforming the days it sat out is a filter pointed backwards.
5. **Mirror check** where the idea has a natural opposite.

## Expected outcome, stated in advance

Poor. The price-side repertoire is closed (16 ICT concepts, 21,300 triggers, sessions, releases)
and the external one too (positioning, implied vol, real yields, dollar). Ideas 3, 6, 9 and 10
are the genuinely novel mechanisms — a specific auction time, a price-level effect, an
expiry-calendar effect and a volume-weighted anchor — and none has been tried here. They are
cheap to test and that is the case for testing them, not a prediction that they work.
