"""MT5 SYMBOL_SWAP_MODE -> name, and the mode-aware carry conversion.

*** WHY THIS FILE EXISTS. *** Before 2026-10-21 there was NO swap-mode mapping anywhere in this
codebase. The interpretation was split across three places that did not agree:
  * `scripts/v5_signals_digest.py:93-101` -- numeric, handles only modes 5/6 and silently treats
    EVERYTHING ELSE as POINTS, including a missing attribute (`getattr(info, "swap_mode", 1)`).
  * `scripts/v5_manual_alert.py:138` -- `swap_mode == 0` as the swap-free guard.
  * `configs/v5_xau_btc_switch.json:72` -- prose, the only place stating the number<->name pairs.

Two traps that mapping hid, both of which this module makes explicit:

  1. MODES 2/3/4 (CURRENCY_*) AND 7/8 (REOPEN_*) WERE MIS-CONVERTED AS POINTS. Their `swap_long`
     is denominated in a currency or is a reopen rule, so running it through the points formula
     produces a confident, wrong number. `carry_pct_yr` returns None for these -- report UNKNOWN,
     never a figure.

  2. MODE 0 (DISABLED) CAN STILL CARRY POPULATED FIELDS. Maven's GoldEternal showed
     `swap_mode = 0` with `swap_long = swap_short = -30.0` (3bd). DISABLED means the fields are
     dormant, but a broker can flip the mode without clearing them -- so the raw fields must be
     reported BESIDE the mode, never collapsed into one number. That is why `carry_pct_yr` returns
     0.0 for mode 0 while callers are expected to also print swap_long/swap_short.
"""
from __future__ import annotations

# MT5 ENUM_SYMBOL_SWAP_MODE
SWAP_MODE = {
    0: "DISABLED",          # no swap charged (fields may still be populated -- see trap 2)
    1: "POINTS",            # swap_long is in POINTS of the symbol
    2: "CURRENCY_SYMBOL",   # in the symbol's base currency
    3: "CURRENCY_MARGIN",   # in the margin currency
    4: "CURRENCY_DEPOSIT",  # in the deposit currency
    5: "INTEREST_CURRENT",  # annual %, current price
    6: "INTEREST_OPEN",     # annual %, open price
    7: "REOPEN_CURRENT",    # position reopened at close price
    8: "REOPEN_BID",        # position reopened at Bid
}

# Modes whose swap_long can be converted to %/yr with only (point, contract_size, price).
CONVERTIBLE = {0, 1, 5, 6}


def mode_name(mode) -> str:
    try:
        return SWAP_MODE.get(int(mode), "MODE_%s" % mode)
    except (TypeError, ValueError):
        return "MISSING"


def carry_pct_yr(info, price: float):
    """swap_long as %/yr of notional, or None when the mode makes that unknowable.

    Returns None (not a number) for modes 2/3/4/7/8 -- the currency and reopen modes. Callers must
    render None as UNKNOWN. Guessing here is how a wrong carry gets into a backtest.
    """
    mode = getattr(info, "swap_mode", None)
    if mode is None:
        return None
    mode = int(mode)
    if mode == 0:
        return 0.0                      # DISABLED; caller still prints the raw fields
    if mode in (5, 6):
        return float(info.swap_long)    # already an annual percentage
    if mode == 1:
        notional = price * info.trade_contract_size
        if notional <= 0:
            return None
        return (float(info.swap_long) * info.point * info.trade_contract_size
                / notional * 365 * 100)
    return None                         # 2/3/4/7/8 -- UNKNOWN, deliberately
