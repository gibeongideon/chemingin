#!/usr/bin/env python3
"""Goat Funded Traders 100K 2-step ($99, refundable) vs LABS001 (user screenshots 2026-11-27).

GFT:  P1 10%, P2 5%, daily loss 3%, max loss 8%, consistency 40% (FUNDED only), 1:100,
      bi-weekly rewards, split 80%. Max-loss convention not stated -> STATIC and TRAILING both run.
Same engine and book as scripts/v5_lab001_buffer.py: XAU+BTC deployed book, block bootstrap,
intraday daily GUARD g (flatten at g, realised -(g + slip)), optional CUSHION sizing on the floor.

FUNDED (12 months): withdraw every 10 business days everything above 1 + buffer, paid at 80%,
but only if the CONSISTENCY rule holds for that period (best day <= 40% of the period's profit);
otherwise the profit stays in the account and rolls into the next period.

    python scripts/v5_gft_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from v5_account_chooser_2026_10 import xau_btc_book  # noqa: E402
from v5_lab001_check import MAXD, paths  # noqa: E402

SAFETY = 1.5
GFT = dict(targets=(0.10, 0.05), daily=0.03, maxloss=0.08, split=0.80, consistency=0.40)


def day_ret(d, k, eq, guard, slip):
    x = k * d
    if guard is not None and -x * SAFETY * eq >= guard:
        return -(guard + slip) / eq
    return x


def size(dial, sd, eq, peak, ml, trailing, cushion):
    k = dial / sd
    if cushion:
        floor = (peak if trailing else 1.0) - ml
        k *= min(2.0, max(0.0, (eq - floor) / ml))
    return k


def challenge(r, sd, dial, guard, slip, trailing, cushion, rules=GFT):
    res = []
    for seed in range(3):
        P = paths(r, seed, MAXD)
        for p in P:
            t, out = 0, None
            for tgt in rules["targets"]:
                eq = peak = 1.0                      # each phase is a fresh 100K
                while out is None:
                    if t >= MAXD:
                        out = ("time", t); break
                    k = size(dial, sd, eq, peak, rules["maxloss"], trailing, cushion)
                    x = day_ret(p[t], k, eq, guard, slip); t += 1
                    if -x * eq > rules["daily"] or (guard is None and -k * p[t - 1] * SAFETY * eq > rules["daily"]):
                        out = ("day", t); break
                    eq *= 1 + x; peak = max(peak, eq)
                    if eq < (peak if trailing else 1.0) - rules["maxloss"]:
                        out = ("dd", t); break
                    if eq >= 1 + tgt:
                        break
            res.append(out or ("pass", t))
    c = np.array([o for o, _ in res])
    days = np.array([t for o, t in res if o == "pass"])
    return (100 * (c == "pass").mean(), np.median(days) / 21 if len(days) else np.nan,
            100 * (c == "day").mean(), 100 * (c == "dd").mean(), 100 * (c == "time").mean())


def funded(r, sd, dial, buf, guard, slip, trailing, cushion, rules=GFT):
    pays, surv = [], []
    for seed in range(3):
        for p in paths(r, 100 + seed, 252):
            eq = peak = 1.0
            paid, alive, per_start, best = 0.0, True, 1.0, 0.0
            for t, d in enumerate(p):
                k = size(dial, sd, eq, peak, rules["maxloss"], trailing, cushion)
                x = day_ret(d, k, eq, guard, slip)
                if -x * eq > rules["daily"]:
                    alive = False; break
                prev = eq; eq *= 1 + x; peak = max(peak, eq)
                best = max(best, eq - prev)
                if eq < (peak if trailing else 1.0) - rules["maxloss"]:
                    alive = False; break
                if (t + 1) % 10 == 0:                 # bi-weekly payout window
                    prof = eq - per_start
                    if eq > 1 + buf and prof > 0 and best <= rules["consistency"] * prof:
                        paid += eq - (1 + buf); eq = 1 + buf
                    per_start, best = eq, 0.0
            pays.append(paid * 100_000 * rules["split"]); surv.append(alive)
    return 100 * np.mean(surv), np.mean(pays)


def main() -> int:
    r = xau_btc_book().to_numpy()
    sd = r.std() * np.sqrt(252)
    print(f"XAU+BTC book vol {100 * sd:.1f}%.  GFT 100K: P1 10% / P2 5%, daily 3%, max 8%, 80%, consistency 40% funded")
    for trailing in (False, True):
        print(f"\n=== max loss {'TRAILING' if trailing else 'STATIC'} ===")
        print(f"  {'setup':<28}{'dial':>6}{'pass%':>7}{'med mo':>8}{'day%':>6}{'dd%':>6}{'time%':>7}"
              f"  funded buf0: surv% $/yr   buf3%: surv% $/yr")
        for name, g, s, cs in (("no guard", None, 0.0, False), ("guard 2.0%", 0.02, 0.002, False),
                               ("guard 2.0% + cushion", 0.02, 0.002, True)):
            for dial in (0.05, 0.07, 0.09, 0.12):
                pr, mo, dy, dd, tm = challenge(r, sd, dial, g, s, trailing, cs)
                s0, p0 = funded(r, sd, dial, 0.0, g, s, trailing, cs)
                s3, p3 = funded(r, sd, dial, 0.03, g, s, trailing, cs)
                print(f"  {name if dial == 0.05 else '':<28}{100 * dial:>5.0f}%{pr:>7.1f}{mo:>8.1f}{dy:>6.1f}{dd:>6.1f}"
                      f"{tm:>7.1f}  {s0:>15.1f} {p0:>6,.0f} {s3:>12.1f} {p3:>6,.0f}")
    return 0


if __name__ == "__main__" and not (len(sys.argv) > 1 and sys.argv[1] == "promo"):
    raise SystemExit(main())


# ---------------------------------------------------------------------------------------------
# PROMO FUNDED STAGE with the full rule table (user screenshot 2026-11-27, last column):
#   daily 3%, max loss 8% static (until first payout), consistency 40%, FLOATING LOSS LIMIT 1% of all
#   open trades combined, max daily profit $3,000, payout cap $2,000 per payout, split 80%.
# Floating loss is modelled as equity minus the equity at the last (re)entry: the book holds one
# continuous position, so its floating P&L is the move since it was last opened. A FLOATING GUARD
# closes everything at -gf (+ slip) and re-opens next day at a fresh entry (cost RESET_COST), which
# resets the floating reference. Daily-profit cap: a day above +3% is a breach (conservative).
RESET_COST = 0.0002


def funded_promo(r, sd, dial, buf, gd, gf, slip, cushion, n_days=252, seed0=100):
    pays, surv, resets = [], [], []
    for seed in range(3):
        for p in paths(r, seed0 + seed, n_days):
            eq = 1.0; ref = 1.0
            paid, alive, per_start, best, nres = 0.0, True, 1.0, 0.0, 0
            for t, d in enumerate(p):
                k = size(dial, sd, eq, 1.0, 0.08, False, cushion)
                x = k * d
                low = eq * (1 + min(x, 0.0) * SAFETY)
                prev = eq
                if eq - (ref - gf) > 0 and low <= ref - gf and (ref - gf) - eq > -gd:
                    eq = ref - gf - slip - RESET_COST; ref = eq; nres += 1     # floating guard
                elif -x * SAFETY * eq >= gd:
                    eq = eq - gd - slip - RESET_COST; ref = eq; nres += 1      # daily guard
                else:
                    eq *= 1 + x
                if prev - eq > 0.03 or ref - eq > 0.01 or eq - prev > 0.03 or eq < 0.92:
                    alive = False; break
                best = max(best, eq - prev)
                if (t + 1) % 10 == 0:
                    prof = eq - per_start
                    if eq > 1 + buf and prof > 0 and best <= 0.40 * prof:
                        w = min(eq - (1 + buf), 0.02 / 0.80)                      # $2,000 cap (after split)
                        paid += w; eq -= w; ref -= w
                    per_start, best = eq, 0.0
            pays.append(paid * 100_000 * 0.80); surv.append(alive); resets.append(nres)
    return 100 * np.mean(surv), np.mean(pays), np.mean(resets)


def main_promo() -> int:
    r = xau_btc_book().to_numpy()
    sd = r.std() * np.sqrt(252)
    print("PROMO FUNDED 12 months: daily guard 2.0%, floating guard gf (limit 1%), static 8%, $3k day cap, $2k payout cap")
    print(f"  {'gf':>5}{'slip':>6}{'cushion':>8}{'dial':>6}   buf0: surv%  $/yr  resets/yr   buf3%: surv%  $/yr")
    for gf, slip in ((0.007, 0.001), (0.007, 0.002), (0.006, 0.003)):
        for cs in (False, True):
            for dial in (0.03, 0.05, 0.07, 0.09):
                s0, p0, n0 = funded_promo(r, sd, dial, 0.0, 0.02, gf, slip, cs)
                s3, p3, _ = funded_promo(r, sd, dial, 0.03, 0.02, gf, slip, cs)
                print(f"  {100 * gf:>4.1f}%{100 * slip:>5.1f}%{('yes' if cs else 'no'):>8}{100 * dial:>5.0f}%"
                      f"   {s0:>10.1f} {p0:>6,.0f} {n0:>9.1f}   {s3:>11.1f} {p3:>6,.0f}")
    return 0


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "promo":
    raise SystemExit(main_promo())
