# ADDENDUM 1 to the pre-registration — Kronos tested the way its authors test it

Written 2026-11-16 ~13:40, after reading the Kronos paper the user supplied
(`researchpepers/Kronos: A Foundation Model for the Language of Financial Markets.pdf`, arXiv
2508.02739v1) and **before any foundation-model outcome was scored** — `scripts/v5_fm_day_signal.py`
had been written and not yet run; the feature files existed, the outcomes had not been joined.
Study 2 (gold M1 family) had been run; nothing here depends on it.

## What the paper says that the original pre-registration did not know

| paper | consequence |
|---|---|
| App. D: "The pre-training data for Kronos extends up to June 2024 ... our test period for all tasks begins in July 2024." | Windows added: **PRE** < 2020-02-01, **TRAIN** 2020-02-01..2024-06-30, **POST** ≥ 2024-07-01. The pre-registered **CLEAN** (> 2025-09-09, last change to the released weights) is kept as the strictest one. |
| Table 13: "Foreign Exchange, 5T/15T/30T/H/D/W, 1,023 assets, from 2020/1/31"; "China, Future, 75 assets, from 2010" | Gold spot is very likely IN the corpus for 2020-02..2024-06 (and SHFE gold futures from 2010). Skill that exists only in TRAIN is memory, not forecasting. |
| App. D: "For cryptocurrency and forex assets, we intentionally exclude volume and amount" | OHLC-only was the right input. No change. |
| Table 6: forecasting uses **T = 0.6, top-p = 0.90, N = 10** | The original cells used the README defaults (T = 1.0, S = 32). Kept as registered; the paper's settings are added below. |
| Table 8: look-back / horizon **1-hour 80 / 12, daily 40 / 12**; investment simulation: daily, 90-day look-back, H = 10, signal = mean forecast path / last close − 1 (Eq. 13) | The original cells were one-step with a 256-bar look-back. The paper's multi-step protocol is added below. |
| Tables 16-17, Forex, return forecasting: Kronos-small IC 0.0398, **Kronos-base 0.0555**, large 0.0441 (not released); Time-MoE 0.0334, TimesFM 0.0353, Chronos-small 0.0289, DLinear 0.0254 | Base is the strongest RELEASED model on forex, so it is added. Note what the yardstick is: an IC pooled over 1,023 pairs, where a generic model already reaches 0.033-0.036. |
| Table 4 / 13: trained frequencies are 1-min .. weekly; no 8-hour bar | The S8 frame is out-of-distribution for Kronos. Kept, flagged; the NY-session decision is re-done below on H1 bars, which it was trained on. |
| Table 10: the only trading result is a cross-sectional top-k stock book on Chinese A-shares, July 2024 - June 2025 | The paper makes no single-asset timing claim. This study is the first such test of it that I know of on gold. |

## Added cells — paper protocol ("KP"), Kronos-small AND Kronos-base

All: OHLC only, per-window z-score, clip ±5, **T = 0.6, top-p = 0.9, N = 10 sampled paths**,
forecast = mean of the decoded paths (the reference `auto_regressive_inference` behaviour).

| id | bars / look-back / horizon | decisions | signal | outcome |
|---|---|---|---|---|
| KPh | H1 / 80 / 12 | every H1 close, OANDA, POST window (≥ 2024-07-01) | `p̂[t+12]/p[t] − 1` (Eq. 11) | realised 12-bar return |
| KPn | H1 / 80 / 12 | the 08:00 ET bar open, every day: OANDA 2007-2026, CME GC 2010-2026 | `p̂[t+8]/p[t] − 1` | NY session 08:00 → 16:00 ET |
| KPd | D1 / 90 / 10 | every day's open: OANDA 2007-2026, CME 2010-2026 | `p̂[t+1]/p[t] − 1` | that day's open → close |
| KPd10 | same paths | same | `mean(p̂[t+1..t+10])/p[t] − 1` (Eq. 13) | 10-day forward return — informational, not intraday |

Same position rule, statistic and null as T1 (sign of the signal minus its trailing-250 median,
signed t by days, circular-shift max-stat). Overlapping horizons (KPh, KPd10): the t is taken on
non-overlapping decisions (every 12th / every 10th) and the IC is printed for the full set.
**Addendum family = {small, base} × {KPh, KPn, KPd} = 6 cells on OANDA.** PASS and KILL are T1's,
unchanged. The yardstick line for KPh is the paper's own forex IC (0.040 small / 0.055 base).

## Added cell — the embedding as a feature ("E"), the literal reading of the user's sentence

The final hidden state at the last context position of Kronos-small on the KPh contexts (512
numbers per H1 bar) IS the model's representation of the last 80 hours. Ridge regression
(features standardised on the training rows, α = 1000, fixed — not tuned) of the 12-bar forward
return, scaled by trailing 20-day H1 vol, on that vector. Walk-forward INSIDE the POST window
only: first fit on the first 40% of POST, refit every 20 trading days on everything before a
12-bar purge; test = the remaining 60%. Statistic: out-of-sample Spearman IC and the signed t of
`sign(prediction) × outcome` on non-overlapping decisions. Null: the same pipeline with the
outcome circularly shifted by ≥ 20 days, 200 draws.
**PASS:** t ≥ +2.0 and null p < 0.05 → a lead worth a proper study. Otherwise closed.

## Engineering note (not a degree of freedom)

Multi-step sampling on CPU needs a key/value cache in the transformer; the reference code
re-runs the whole window for every generated bar. The cached path must reproduce the vendored
model's next-token logits (max |diff| < 1e-4) on every position of a test sequence before it is
used, exactly as the one-step fast path was verified (max diff 4.8e-7).

## Honest prior

~5% for any cell. The paper's forex IC is pooled across pairs, a linear model gets 60% of it,
and the repo's own 5.9-sigma AUC lift (§3as) produced −0.70 Sharpe.
