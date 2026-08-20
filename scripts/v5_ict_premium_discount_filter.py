"""ICT/SMC concept sweep, Phase 4 (Bucket C — filter, not a standalone bet):
Premium/Discount as a directional filter on the LIVE CHAMPION. XAU H4.

Plan: `~/.claude/plans/i-wanttt-you-to-playful-widget.md` — this test runs
REGARDLESS of Phase 1-3 outcomes (unlike everything else in Bucket C, which
was gated on a Phase 1-3 survivor to apply to). Explicitly flagged in the plan
as mechanistically close to the already-disproven ADX regime-gate
(`v5_xau_regime_gate.py`, `regime-gate-champion-disproven`): both cut the
champion's OWN forecast with a discrete gate rather than adding new
information. The mechanism differs — ADX gates on TREND STRENGTH (a lagging
volatility-of-direction read); this gates on PRICE POSITION within the
current swing range (premium = "already extended, don't chase," discount =
"room to run") — different enough to warrant its own clean test, priced-in
skepticism given the precedent.

  Gate: reduce the champion's forecast (by a swept `reduction` factor) when
  its OWN direction disagrees with premium/discount — i.e. a long forecast
  while price sits in PREMIUM (buying high, against the ICT read), or a short
  forecast while price sits in DISCOUNT (selling low). Full size when aligned
  or at equilibrium.

    python scripts/v5_ict_premium_discount_filter.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_xau_turn_prob import load_h4, engine, paired, per_year  # noqa: E402
from src.v5.xau_dual_signals import champion_signal  # noqa: E402
from src.features.ict_primitives import swing_levels, premium_discount, REGIME_BOUNDARIES  # noqa: E402
from src.evaluation.lookahead_probe import probe_lookahead, print_report  # noqa: E402
from src.evaluation.dsr_pbo import deflated_sharpe_ratio, pbo_cscv  # noqa: E402
from src.evaluation.walk_forward_grid import run_concept_pipeline, print_summary_table  # noqa: E402

PNL_EVAL_START = "2018-01-01"
YEARS = list(range(2018, 2027))
SELECT_WINDOW_BARS = 3 * 252 * 6

ORDERS = (3, 5, 8)
REDUCTIONS = (0.0, 0.25, 0.5, 0.75)


def pd_gated_champion_fc(df: pd.DataFrame, order: int, reduction: float) -> pd.Series:
    champ_fc = champion_signal(df["close"])
    sh, sl = swing_levels(df, order=order)
    _, zone = premium_discount(df, sh, sl)
    c, z = champ_fc.values, zone.values
    gate = np.ones(len(df))
    against_long = (c > 0) & (z > 0)   # long forecast, price in premium -> buying high
    against_short = (c < 0) & (z < 0)  # short forecast, price in discount -> selling low
    gate[against_long | against_short] = reduction
    return champ_fc * pd.Series(gate, index=df.index)


def main():
    print("=== STAGE 0: lookahead probe ===")
    results = {
        "pd_gated_champion(5,0.25)": probe_lookahead(
            lambda d: pd_gated_champion_fc(d, 5, 0.25), n=4200, offsets=(300, 200, 100, 50), warmup_buffer=1700
        ),
    }
    print_report(results)
    if not all(v.ok for v in results.values()):
        print("!! lookahead probe FAILED — refusing to trust the backtest below.")
        return

    df = load_h4(0.448)
    champ_fc = champion_signal(df["close"])
    bh_m, bh_d = engine(df, pd.Series(1.0, index=df.index), eval_start=PNL_EVAL_START)
    ch_m, ch_d = engine(df, champ_fc, eval_start=PNL_EVAL_START)
    print(f"buy&hold-vt SR={bh_m['sharpe']:+.3f}   champion SR={ch_m['sharpe']:+.3f}  DD={ch_m['dd']:+.1f}%")

    grid = {(o, r): (lambda o=o, r=r: pd_gated_champion_fc(df, o, r)) for o in ORDERS for r in REDUCTIONS}
    result = run_concept_pipeline(
        "Premium/Discount-gated champion", df, grid,
        engine=engine, paired=paired, per_year=per_year,
        deflated_sharpe_ratio=deflated_sharpe_ratio, pbo_cscv=pbo_cscv,
        bh_m=bh_m, bh_d=bh_d, ch_m=ch_m, ch_d=ch_d,
        regime_boundaries=REGIME_BOUNDARIES["XAUUSD"],
        pnl_eval_start=PNL_EVAL_START, years=YEARS, select_window_bars=SELECT_WINDOW_BARS,
    )

    print(f"\n{'='*70}\nSUMMARY (premium/discount filter)\n{'='*70}")
    print_summary_table([result])

    if not np.isnan(result.get("sr_wf", float("nan"))):
        import scripts.v5_basket_challenge as vbc
        m = vbc.MODELS["ftmo"]
        j = pd.concat([ch_d.rename("champ")], axis=1)

        def fp(series):
            r10 = (series * (0.10 / (series.std() * np.sqrt(252)))).values
            return vbc.fp_sim(r10, m["vol"] / vbc.TARGET_VOL, day_safety=1.5, p1=m["p1"], p2=m["p2"],
                              dayloss=m["daily"], maxloss=m["maxloss"])
        sf = fp(ch_d)
        print(f"\nFTMO pass-sim   champion alone: {sf['passpct']:.1f}%  median {sf['med_mo']:.1f}mo")

    out = ROOT / "data" / "v5_runs" / "ict_premium_discount_filter_summary.csv"
    pd.DataFrame([{k: v for k, v in result.items() if k != "regimes"}]).to_csv(out, index=False)
    print(f"\nsummary -> {out}")


if __name__ == "__main__":
    main()
