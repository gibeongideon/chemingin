"""Composite fitness score for ranking backtest candidates — MT5 adaptation.

Forven's `strategies/fitness.py` blends Sharpe/win-rate/profit-factor/drawdown/
trade-count into one 0-100 score for the same reason this repo keeps needing
one: a screen (grid search, instrument sweep) produces many candidates scored
on several metrics at once (SR, drawdown, DSR, correlation to the existing
book), and eyeballing a table of all of them each time is slow and inconsistent
about which axis wins ties.

This repo's engines are continuous vol-targeted positions, not discrete trades,
so there is no natural win-rate/profit-factor/trade-count to reuse — those three
of Forven's five weights don't apply here. Adapted to what this repo actually
produces every screen:

  Sharpe (50%)        capped at 3.0, the same cap Forven uses
  drawdown (20%)      100 at 0% DD, 0 at -30% DD or worse
  DSR (20%)           already a legitimate real-edge probability (0-1) — the
                      selection-bias-corrected replacement for a raw p-value
  |correlation| to the existing book (10%)  100 at corr=0, 0 at |corr|>=1 —
                      a diversification bonus, since a candidate identical to
                      something already held is worth less even at the same SR

Pass `corr=None` when there's no existing book to diversify (renormalizes the
other three weights so they still sum to 1).
"""
from __future__ import annotations


def fitness_score(
    sharpe: float,
    max_drawdown_pct: float,
    dsr: float | None = None,
    corr_to_book: float | None = None,
    *,
    sharpe_cap: float = 3.0,
    dd_floor_pct: float = 30.0,
) -> float:
    """0-100 composite score. Any missing optional input renormalizes the
    remaining weights to still sum to 1 rather than silently zeroing them out."""
    weights = {"sharpe": 0.5, "dd": 0.2, "dsr": 0.2, "corr": 0.1}
    if dsr is None:
        weights.pop("dsr")
    if corr_to_book is None:
        weights.pop("corr")
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    sr_component = max(0.0, min(1.0, sharpe / sharpe_cap)) * 100.0
    dd_component = max(0.0, min(1.0, 1.0 - abs(max_drawdown_pct) / dd_floor_pct)) * 100.0

    score = weights["sharpe"] * sr_component + weights["dd"] * dd_component
    if dsr is not None:
        score += weights["dsr"] * max(0.0, min(1.0, dsr)) * 100.0
    if corr_to_book is not None:
        score += weights["corr"] * max(0.0, min(1.0, 1.0 - abs(corr_to_book))) * 100.0
    return round(score, 1)


def rank_candidates(rows: list[dict], **kwargs) -> list[dict]:
    """Attach a `fitness` field to each row (dicts with sharpe/max_drawdown_pct/
    dsr/corr_to_book keys, matching fitness_score's params where present) and
    return them sorted best-first. Does not mutate the input rows."""
    out = []
    for r in rows:
        r2 = dict(r)
        r2["fitness"] = fitness_score(
            r2.get("sharpe", 0.0),
            r2.get("max_drawdown_pct", 0.0),
            r2.get("dsr"),
            r2.get("corr_to_book"),
            **kwargs,
        )
        out.append(r2)
    return sorted(out, key=lambda r: -r["fitness"])
