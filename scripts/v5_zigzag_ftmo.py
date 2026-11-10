"""ZIGZAG bottom-detector OBSERVATION HARNESS — FTMO demo, bridge 18814.

*** THIS STRATEGY HAS NEGATIVE EXPECTANCY AND IS NOT A RECOMMENDATION. ***
V5_FINDINGS §3x: all 48 grid cells negative full-sample (best SR -0.44), walk-forward
**SR -0.76 / DD -35.7%** against buy-and-hold's +0.79, paired t **-3.72**, **0 of 8 years
better**, DSR **0.000**. Deployed at the user's explicit request to watch a documented
detector operate live on a DEMO account ("I just want to see it in action, no profit needed").
Sized small for that reason. Do not point this at a funded account.

WHAT IT TRADES. The walk-forward-selected cell from §3x, not the full-sample best:
    threshold 0.60  ·  TP +0.60%  ·  SL -0.50%  ·  max hold 48 H1 bars
(2023/24/25 all selected 0.6/0.6%/0.5%; 2026 selected the 48-bar hold. The full-sample best
was 0.5/0.6%/0.5%/48 at SR -0.441 and is deliberately not used.)

DETECTOR UPGRADE (2026-09-10). The model is now POOLED: it trains on gold plus the clean H1
majors (EURUSD/GBPUSD/USDJPY) from this same broker feed and predicts gold. Validated in
`v5_zigzag_pooled_h1.py`: walk-forward AUC 0.7333 -> 0.7434 (+0.0101 = 4.6x SE 0.0022), 11 of
11 test years, sign-test p 0.0005, with precision up where the 0.60 threshold operates
(P@recall10 0.651 -> 0.693, P@recall05 0.671 -> 0.723). Each auxiliary series is screened for
intrabar leakage at runtime (§3av) and refused above 0.15; if all are refused or unavailable the
detector falls back to GOLD-ONLY, the incumbent behaviour. **This improves DETECTION ONLY — the
strategy's expectancy is unchanged and still negative.**

DESIGN CHOICES THAT MATTER
  * Rates come from the BROKER'S OWN FEED via the bridge, never from the CSVs. The H1 CSV is
    currently 2,063 hours stale, and two of this repo's data files are known to be different
    feeds from the same instrument (differing by $405 and $177 at the extremes). One feed, no
    splice.
  * TP and SL are attached to the order SERVER-SIDE, so the two profitable exits do not depend
    on this process being alive. Only the max-hold exit needs the bot, which is why the timer
    runs hourly.
  * Double safety lock: `--live --execute` both required to send anything. Default is a dry
    plan. `--execute` is ignored on a non-demo account unless `--live` is also passed.
  * Writes state to `data/v5_runs/zigzag_ftmo_state.json` for the notifier, so the manual
    channel and the executor never disagree about what fired.

    python scripts/v5_zigzag_ftmo.py                      # dry plan
    python scripts/v5_zigzag_ftmo.py --live --execute      # actually trade the demo
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_zigzag_signal import AUX_SYMBOLS, bottom_probability, TOL  # noqa: E402

CONFIG = ROOT / "configs" / "v5_zigzag_ftmo.json"
STATE = ROOT / "data" / "v5_runs" / "zigzag_ftmo_state.json"
TF_H1 = 16385
FETCH_BARS = 30_000


def fetch_aux(mt5, symbols, now: pd.Timestamp | None = None) -> tuple[dict, list]:
    """Auxiliary H1 series for POOLED training, from the SAME broker feed as gold.

    Deliberately NOT from the CSVs: the H1 CSV is >2,000 hours stale and two data files in this
    repo are known to be different feeds for the same instrument (differing by $405 at the
    extremes). One feed, no splice — the rule the gold leg already follows.

    A missing or unfetchable auxiliary symbol is skipped, never fatal: the detector falls back
    to GOLD-ONLY, which is exactly the incumbent behaviour.
    """
    out, missing = {}, []
    for sym in symbols:
        try:
            d = fetch_h1_closed(mt5, sym, now)
            if d is None or len(d) < 6000:
                missing.append((sym, f"only {0 if d is None else len(d)} bars"))
                continue
            out[sym] = d
        except Exception as e:
            missing.append((sym, f"{type(e).__name__}"))
    return out, missing


def server_now(mt5, symbol: str) -> pd.Timestamp | None:
    """'Now' on the BROKER clock, from the newest tick. MT5 stamps bars and ticks at server time
    (Switch/FTMO: NY+7, UTC+2/+3), so any 'is this bar closed / how old is it' test must be
    written in that frame — never against the host's UTC (memory: mt5-server-clock-vs-host-clock).
    With the market shut the last tick is Friday's, which is still the right frame: every bar
    before it is closed. None if no tick could be read."""
    for i in range(3):
        if i:
            time.sleep(2 * i)
        mt5.symbol_select(symbol, True)
        t = mt5.symbol_info_tick(symbol)
        if t is not None and getattr(t, "time", 0):
            return pd.to_datetime(int(t.time), unit="s")
    return None


def fetch_h1_closed(mt5, symbol: str, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Broker's own H1 history, CLOSED BARS ONLY. `copy_rates_from_pos(..,0,n)` returns the
    in-progress bar at position 0; scoring an unfinished bar would evaluate a signal that can
    still change (the same bug caught in the manual-alert build).

    *** BUG FIXED 2026-11-03: `now` WAS THE HOST'S UTC against a SERVER-stamped index. ***
    The broker runs 2-3h ahead of UTC, so `index + 1h > utcnow` threw away the newest 2-3
    CLOSED bars as well as the forming one: at 18:19 UTC (21:19 server) the detector scored the
    16:00 bar although 20:00 had closed. Every P(bottom), ATR and trail distance since this
    harness was built has been computed 2-3 hours late. `now` is now the server's own tick
    time; without one, drop only position 0 (the forming bar), which errs by one bar at most."""
    mt5.symbol_select(symbol, True)
    r = None
    for _ in range(6):
        r = mt5.copy_rates_from_pos(symbol, TF_H1, 0, FETCH_BARS)
        if r is not None and len(r) > 5000:
            break
        time.sleep(2)
    if r is None or len(r) < 5000:
        raise SystemExit(f"could not fetch enough {symbol} H1 history: {mt5.last_error()}")
    a = np.array([(x[0], x[1], x[2], x[3], x[4]) for x in r], dtype=float)
    df = pd.DataFrame(a[:, 1:], columns=["open", "high", "low", "close"],
                      index=pd.to_datetime(a[:, 0], unit="s"))
    if now is None:
        return df.iloc[:-1]
    open_bars = df.index[df.index + pd.Timedelta(hours=1) > now]
    return df.drop(index=open_bars) if len(open_bars) else df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--config", default=None,

                    help="config JSON; defaults to the module CONFIG_FILE")
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text() if args.config else CONFIG.read_text())
    port = args.port or cfg["port"]
    symbol, magic = cfg["symbol"], cfg["magic"]
    thr, tp_pct, sl_pct = cfg["threshold"], cfg["tp_pct"], cfg["sl_pct"]
    max_hold, risk_frac = cfg["max_hold_bars"], cfg["risk_frac"]
    # EXIT MODE. "pct_bracket" is the original fixed +0.60%/-0.50% structure. "atr_trail" is the
    # structure V5_FINDINGS 3ay named as the one open actionable item ("re-derive in ATR, not
    # percent") and 3bo measured as the only non-negative cell on this detector: at a 3.0 ATR
    # trail, EV/fire is +2.22bp net against -4.15bp at 1.0 ATR. It is also the only exit that
    # does not TRUNCATE a winner -- 3bl found a fixed bracket costs the ORACLE itself -0.94 to
    # -1.43bp against simply holding, because the barriers cut the moves that pay for the losers.
    exit_mode = cfg.get("exit_mode", "pct_bracket")
    trail_atr = float(cfg.get("trail_atr", 3.0))
    tp_r = float(cfg.get("tp_r", 0.0) or 0.0)       # take-profit in units of the 3-ATR risk; 0 = none
    entry_limit_atr = float(cfg.get("entry_limit_atr", 0.0) or 0.0)   # 0 = market entry

    from mt5linux import MetaTrader5
    # RETRY: 18815 is shared by seven timers, two of them polling every 5 minutes since
    # 2026-11-02. A terminal already serving another client returns (-6, 'Authorization failed')
    # on initialize(), indistinguishable from a real credential fault.
    mt5, _last = None, None
    for _i in range(3):
        if _i:
            time.sleep(3 * _i)
        mt5 = MetaTrader5(host="localhost", port=port)
        if mt5.initialize():
            break
        _last = mt5.last_error()
    else:
        raise SystemExit(f"bridge {port} init failed after 3 tries: {_last}")
    ai = mt5.account_info()
    is_demo = "demo" in str(getattr(ai, "server", "")).lower() or getattr(ai, "trade_mode", 0) == 0
    print(f"account {ai.login} @ {ai.server}   equity ${ai.equity:,.2f}   "
          f"{'DEMO' if is_demo else '*** NOT FLAGGED DEMO ***'}")

    # One server-clock 'now' for the whole pass: closed-bar filter, staleness and max-hold age.
    srv_now = server_now(mt5, symbol)
    print(f"server clock {srv_now}" if srv_now is not None
          else "server clock UNAVAILABLE — dropping only the forming bar")
    df = fetch_h1_closed(mt5, symbol, srv_now)
    aux_syms = cfg.get("aux_symbols", list(AUX_SYMBOLS))
    aux, aux_missing = fetch_aux(mt5, aux_syms, srv_now)
    st = bottom_probability(df, thr, tp_pct, sl_pct, max_hold, aux=aux, now=srv_now)
    # *** symbol_info CAN RETURN None ON A CONTENDED BRIDGE, AND DID — TWICE, 2026-11-02. ***
    # The model trains for ~13s between the bridge calls above and these, and a second client
    # arriving in that window made both of these return None. The old line then raised
    # `AttributeError: 'NoneType' object has no attribute 'ask'` and the pass died. Harmless on
    # a no-fire pass; on a FIRING pass it is the difference between an order and no order, which
    # is the one failure here that costs a trade. Retry, then fail with a sentence that says
    # what happened rather than an AttributeError.
    info = tick = None
    for _i in range(3):
        if _i:
            time.sleep(2 * _i)
        mt5.symbol_select(symbol, True)
        info, tick = mt5.symbol_info(symbol), mt5.symbol_info_tick(symbol)
        if info is not None and tick is not None:
            break
    if info is None or tick is None:
        raise SystemExit(f"{symbol}: symbol_info/tick still None after 3 tries on bridge {port} "
                         f"— terminal busy or symbol not in Market Watch; no order attempted")
    ask = getattr(tick, "ask", 0.0) or info.ask
    # *** BUG FIXED 2026-10-28: `bid` was used by the atr_trail ratchet but NEVER DEFINED. ***
    # The ratchet block raised `NameError: name 'bid' is not defined` on EVERY pass that had an
    # open position -- 291 passes, 0 ratchet decisions ever logged, the process exiting 1 each
    # time. So the trailing stop never moved once since deployment on 2026-10-19, and the max-hold
    # exit and entry logic below it never ran either, because the exception aborted the pass.
    # The live tell was three closed trades at -248.94 / -243.36 / -246.35: a spread of $5.58 on
    # losses that should vary if a trail had ratcheted at all. Identical losses = the stop was hit
    # at its INITIAL distance every time. A fourth position then sat +42 points (4x ATR) in profit
    # with its stop still 45 points BELOW entry.
    # A LONG trails the BID, not the ask -- the bid is the price the position would actually be
    # closed at, so trailing the ask would place the stop one spread too high.
    bid = getattr(tick, "bid", 0.0) or info.bid
    contract = info.trade_contract_size
    print(f"feed: {len(df):,} closed H1 bars, last {st.asof} ({st.stale_hours:.1f}h ago)")
    print(f"model: {st.arm}, trained on {st.n_train:,} bars"
          + (f" (+{', '.join(st.aux_used)})" if st.aux_used else ""))
    if st.aux_rejected:
        print(f"  aux REJECTED: " + "; ".join(f"{a} ({b})" for a, b in st.aux_rejected))
    if aux_missing:
        print(f"  aux unavailable: " + "; ".join(f"{a} ({b})" for a, b in aux_missing))
    print(f"P(bottom) {st.prob:.4f}  threshold {thr:.2f}  -> "
          f"{'FIRES' if st.fires else 'no fire'}")

    # ---- TREND GATE (§3dt, live A/B on Switch from 2026-11-09) ----
    # Skip a fire when the champion's gold forecast (last CLOSED H4 bar of the Book's own Switch file)
    # is below `min_forecast`: dip-buys made money with the trend up (+0.30/+0.44 SR) and lost with it
    # down (-0.15/-0.24) in both panel models — consistent sign, NOT significant (null p 0.35-0.59).
    # FAIL-OPEN: an unreadable or stale file leaves the fire untouched (behaves as before the gate).
    fires, trend_fc, gate_blocked = st.fires, None, False
    gate = cfg.get("trend_gate")
    if gate and fires:
        try:
            from src.v5.xau_dual_signals import champion_signal  # noqa: PLC0415
            h4 = pd.read_csv(ROOT / gate["h4_csv"], usecols=["time", "close"], parse_dates=["time"])
            h4 = h4.drop_duplicates("time").set_index("time").sort_index()
            h4 = h4[h4.index + pd.Timedelta(hours=4) <= srv_now] if srv_now is not None else h4
            lag_h = (df.index[-1] - h4.index[-1]).total_seconds() / 3600
            if lag_h > 12:
                print(f"  trend gate: H4 file {lag_h:.0f}h behind the H1 feed -> FAIL-OPEN (gate ignored)")
            else:
                trend_fc = float(champion_signal(h4["close"]).iloc[-1])
                gate_blocked = trend_fc < float(gate.get("min_forecast", 0.5))
                print(f"  trend gate: champion forecast {trend_fc:.2f} (bar {h4.index[-1]}) "
                      f"{'< ' if gate_blocked else '>= '}{gate.get('min_forecast', 0.5)} -> "
                      f"{'SKIP this fire (gold trend down)' if gate_blocked else 'allowed'}")
        except Exception as exc:  # noqa: BLE001
            print(f"  trend gate: error {exc!r} -> FAIL-OPEN (gate ignored)")
        fires = fires and not gate_blocked

    # ---- §3dz INTRADAY CIRCUIT BREAKER (config `circuit_breaker_atr`, absent = off) ----------
    # "If the broker day is already down more than k x YESTERDAY's D1 ATR(14) from today's first
    # open, take no new ZigZag buys until the next broker day." Recommended k = 1.0.
    #
    # WHY IT EXISTS: ZigZag fires AFTER falls, so a fire on a day already down hard is a falling
    # knife. §3dz measured the blocked fires traded ALONE at -0.024 / -0.068 / -0.134 R for
    # k 0.5 / 0.75 / 1.0 against +0.04 R for the kept ones — losses deepening monotonically with
    # k, which is the dose-response that makes it a mechanism rather than a fitted threshold.
    # It is the FIRST ZigZag lift to clear a family max-stat null in two independent sources
    # (CME p 0.010, Switch p 0.030), improves both halves in every cell, and blocks only 1-2% of
    # days at k 1.0. It REPAIRS ZigZag (CME 0.21 -> 0.56); it does not make it a new edge.
    #
    # The latch is reproduced exactly: §3dz takes a cummax of the per-bar condition within the
    # day, so once the breaker trips it stays on. Taking the day's LOWEST CLOSE against the
    # first open is the same test, done in one line.
    #
    # FAIL-OPEN, like the trend gate: any error leaves the fire untouched. A gate that blocks on
    # its own bugs is worse than no gate.
    cb_k = cfg.get("circuit_breaker_atr")
    cb_blocked, cb_move, cb_atr = False, None, None
    if cb_k and fires:
        try:
            dayix = df.index.normalize()
            d1 = df.groupby(dayix).agg(high=("high", "max"), low=("low", "min"),
                                       close=("close", "last"), open=("open", "first"))
            pc = d1["close"].shift(1)
            tr_ = pd.concat([d1["high"] - d1["low"], (d1["high"] - pc).abs(),
                             (d1["low"] - pc).abs()], axis=1).max(axis=1)
            atr_d1 = tr_.ewm(alpha=1 / 14, adjust=False).mean().shift(1)   # YESTERDAY's ATR
            today = dayix[-1]
            cb_atr = float(atr_d1.loc[today])
            todays = df[dayix == today]
            cb_move = float(todays["close"].min() - float(todays["open"].iloc[0]))
            cb_blocked = bool(np.isfinite(cb_atr) and cb_move < -float(cb_k) * cb_atr)
            print(f"  circuit breaker: day move {cb_move:+.2f} vs trigger "
                  f"{-float(cb_k) * cb_atr:+.2f} ({cb_k} x prior D1 ATR {cb_atr:.2f}) -> "
                  f"{'BLOCK new buys today (falling knife)' if cb_blocked else 'allowed'}")
        except Exception as exc:  # noqa: BLE001
            print(f"  circuit breaker: error {exc!r} -> FAIL-OPEN (gate ignored)")
        fires = fires and not cb_blocked

    # PROSPECTIVE order — computed every run, fired or not, so the alert can always show
    # the pair and the exact levels you would place if the threshold were met.
    # ATR of COMPLETED H1 bars (Wilder), the frame the trade is HELD on -- 3bl's method rule:
    # scale a stop against the ATR of the timeframe the trade is held on, never a coarser one.
    _pc = df["close"].shift(1)
    _tr = pd.concat([df["high"] - df["low"], (df["high"] - _pc).abs(),
                     (df["low"] - _pc).abs()], axis=1).max(axis=1)
    atr_h1 = float(_tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    print(f"ATR(14) H1 = {atr_h1:.2f}  ({100*atr_h1/ask:.3f}% of price); "
          f"exit_mode={exit_mode}" + (f" trail {trail_atr}xATR = {trail_atr*atr_h1:.2f}"
                                      if exit_mode == "atr_trail" else ""))

    risk_usd_p = ai.equity * risk_frac
    if exit_mode == "atr_trail":
        stop_dist_p = trail_atr * atr_h1
        lots_p = risk_usd_p / (stop_dist_p * contract)
    else:
        lots_p = risk_usd_p / (sl_pct * ask * contract)
    step_p = info.volume_step or 0.01
    lots_p = max(info.volume_min, round(lots_p / step_p) * step_p)
    prospective = dict(symbol=symbol, price=round(float(ask), info.digits),
                       entry=round(float(ask), info.digits),
                       sl=round(ask - trail_atr * atr_h1, info.digits) if exit_mode == "atr_trail"
                          else round(ask * (1 - sl_pct), info.digits),
                       # Trail mode has NO take-profit unless `tp_r` is set. §3dr: a HIGH TP
                       # (4.3 x the 3-ATR risk, ~$1,500 at 0.06 lots) was the one consistent lift
                       # (both halves); low TPs ($700) truncate the winners and lose.
                       tp=(round(ask + tp_r * trail_atr * atr_h1, info.digits) if tp_r > 0 else 0.0)
                          if exit_mode == "atr_trail"
                          else round(ask * (1 + tp_pct), info.digits),
                       lots=round(lots_p, 2),
                       sl_pct=sl_pct, tp_pct=tp_pct, max_hold=max_hold)
    print(f"prospective: BUY {prospective['lots']} {symbol} @ {prospective['entry']}  "
          f"SL {prospective['sl']}  TP {prospective['tp']}")

    held = [p for p in (mt5.positions_get(symbol=symbol) or []) if p.magic == magic]
    # A RESTING LIMIT COUNTS AS BEING IN THE TRADE. Without this the limit arm would place a
    # fresh order every hour the signal stayed above threshold, ending up with a stack of
    # pending buys at descending prices — a scale-in ladder nobody asked for, and §3bs already
    # measured scaling in as leverage rather than edge.
    pending = ([o for o in (mt5.orders_get(symbol=symbol) or []) if o.magic == magic]
               if entry_limit_atr else [])
    print(f"open under magic {magic}: {len(held)}"
          + (f", resting limit orders: {len(pending)}" if entry_limit_atr else ""))
    actions = []

    # ---- 0. RATCHET the trailing stop (atr_trail mode only) ----
    # Peak is taken from COMPLETED H1 bars at or after the entry, never before it: seeding the
    # peak from a pre-entry bar produced phantom instant trail activation once
    # (src/v5/xau_m15_exec.py:227-229 and its permanent regression test). The stop is raised
    # ONE-WAY only, and it lives server-side so the broker enforces it intrabar -- this process
    # only ever moves it up between passes.
    if exit_mode == "atr_trail":
        for p_ in held:
            entry_t = pd.to_datetime(p_.time, unit="s", utc=True).tz_localize(None)
            since = df[df.index >= entry_t]
            peak = float(max(since["high"].max() if len(since) else p_.price_open,
                             float(p_.price_open), float(bid)))
            new_sl = round(peak - trail_atr * atr_h1, info.digits)
            if new_sl > float(p_.sl) + 1e-9:
                actions.append(("MODIFY", p_.ticket, p_.volume,
                                f"trail {trail_atr}xATR: peak {peak:.2f} -> SL "
                                f"{float(p_.sl):.2f} -> {new_sl:.2f}",
                                dict(entry=float(p_.price_open), sl=new_sl, tp=None,
                                     close_by=None, age_h=None)))
            else:
                print(f"  ticket {p_.ticket}: peak {peak:.2f}, SL {float(p_.sl):.2f} "
                      f"already at or above the trail ({new_sl:.2f}) — no change")

    # ---- 1. max-hold exits (the only exit this process is responsible for) ----
    # p.time is a SERVER epoch, so age it against the server clock. Against host UTC every
    # position read 2-3h younger than it was and max-hold closed it 2-3h late.
    now = (srv_now.timestamp() if srv_now is not None
           else datetime.now(timezone.utc).timestamp())
    for p in held:
        age_h = (now - p.time) / 3600.0
        if age_h >= max_hold:
            actions.append(("CLOSE", p.ticket, p.volume,
                            f"max-hold {age_h:.0f}h >= {max_hold}h",
                            dict(entry=float(p.price_open), sl=None, tp=None,
                                 close_by=None, age_h=round(age_h, 1))))
        else:
            print(f"  ticket {p.ticket}: {p.volume} lots, age {age_h:.1f}h "
                  f"(closes at {max_hold}h; TP/SL are server-side)")

    # ---- 2. entry ----
    if fires and not held and not pending:
        if st.stale_hours > 3:
            print(f"  SKIP entry: feed is {st.stale_hours:.1f}h stale")
        else:
            lots = prospective["lots"]
            sl, tp = prospective["sl"], prospective["tp"]
            close_by = (datetime.now(timezone.utc)
                        + pd.Timedelta(hours=max_hold)).strftime("%Y-%m-%d %H:%M UTC")
            actions.append(("BUY", None, round(lots, 2),
                            f"prob {st.prob:.3f} >= {thr}",
                            dict(entry=round(float(ask), info.digits), sl=sl, tp=tp,
                                 close_by=close_by, age_h=None)))
    elif fires and held:
        print("  fires, but already in a position — no pyramiding")
    elif fires and pending:
        print(f"  fires, but a limit order is already resting at "
              f"{pending[0].price_open} — no stacking")

    if not actions:
        print("  ACTION: none")
    for kind, ticket, vol, why, px in actions:
        extra = (f" @ {px['entry']}  SL {px['sl']}  TP {px['tp']}" if kind == "BUY"
                 else (f"  SL -> {px['sl']}" if kind == "MODIFY" else ""))
        print(f"  ACTION: {kind} {vol} lots{extra}  ({why})")

    sent = []
    if args.live and args.execute and actions:
        if not is_demo:
            print("  ** REFUSING: account is not flagged demo. "
                  "This harness is demo-only by design. **")
        else:
            for kind, ticket, vol, why, px in actions:
                if kind == "MODIFY":
                    # Raise the server-side stop. TRADE_ACTION_SLTP only -- it never touches
                    # volume, so a ratchet can never accidentally close or resize a position.
                    # KEEP the position's own TP: sending tp=0.0 here would silently delete it.
                    cur = next((p for p in held if p.ticket == ticket), None)
                    req = dict(action=mt5.TRADE_ACTION_SLTP, symbol=symbol,
                               position=ticket, sl=px["sl"], tp=float(getattr(cur, "tp", 0.0) or 0.0))
                    r = mt5.order_send(req)
                    rc = getattr(r, "retcode", r)
                    print(f"    SL RAISED on #{ticket} -> {px['sl']}: retcode={rc}")
                    sent.append((kind, ticket, vol, rc))
                    continue
                if kind == "CLOSE":
                    pos = next((p for p in held if p.ticket == ticket), None)
                    req = dict(action=mt5.TRADE_ACTION_DEAL, symbol=symbol, volume=vol,
                               type=mt5.ORDER_TYPE_SELL, position=ticket,
                               price=getattr(tick, "bid", 0.0) or info.bid,
                               magic=magic, comment=cfg["run_id"][:31],
                               type_filling=mt5.ORDER_FILLING_IOC)
                else:
                    # LIMIT ARM (config `entry_limit_atr`, absent = market as before). Rests a
                    # BUY LIMIT `entry_limit_atr` x ATR below the ask instead of paying the
                    # offer. Brackets are measured from the LIMIT price, not the ask, so the arm
                    # keeps the same R multiple rather than a wider stop.
                    #
                    # *** THE FILL IS NOT FREE AND THIS REPO HAS MEASURED THE BILL TWICE. ***
                    # §3bc: FILLED -22.8bp vs SKIPPED +17.9bp, t -5.23. Waiting for a dip SELECTS
                    # LOSERS — the fires that come back to you are the ones that keep going. So
                    # this arm is expected to fill on the worse subset of signals, and the
                    # report must compare the arms on the FILLED subset as well as overall or it
                    # will credit the exit rule for a selection effect.
                    px_ref = ask
                    if entry_limit_atr:
                        px_ref = round(ask - entry_limit_atr * atr_h1, info.digits)
                    if exit_mode == "atr_trail":
                        sl = round(px_ref - trail_atr * atr_h1, info.digits)
                        tp = round(px_ref + tp_r * trail_atr * atr_h1, info.digits) if tp_r > 0 else 0.0
                    else:
                        sl = round(px_ref * (1 - sl_pct), info.digits)
                        tp = round(px_ref * (1 + tp_pct), info.digits)
                    if entry_limit_atr:
                        # Expire with the signal. An unfilled limit left resting would fill days
                        # later on a signal that no longer exists, which is a different strategy.
                        exp = int(time.time()) + int(float(cfg.get("limit_expire_h", 24)) * 3600)
                        req = dict(action=mt5.TRADE_ACTION_PENDING, symbol=symbol, volume=vol,
                                   type=mt5.ORDER_TYPE_BUY_LIMIT, price=px_ref, sl=sl, tp=tp,
                                   magic=magic, comment=cfg["run_id"][:31],
                                   type_time=mt5.ORDER_TIME_SPECIFIED, expiration=exp,
                                   type_filling=mt5.ORDER_FILLING_RETURN)
                    else:
                        req = dict(action=mt5.TRADE_ACTION_DEAL, symbol=symbol, volume=vol,
                                   type=mt5.ORDER_TYPE_BUY, price=ask, sl=sl, tp=tp,
                                   magic=magic, comment=cfg["run_id"][:31],
                                   type_filling=mt5.ORDER_FILLING_IOC)
                r = mt5.order_send(req)
                rc = getattr(r, "retcode", r)
                print(f"    sent {kind}: retcode {rc} "
                      f"{'OK' if rc == 10009 else getattr(r, 'comment', '')}")
                sent.append(dict(kind=kind, vol=vol, retcode=int(rc) if rc else None))
    elif actions:
        print("  (dry plan — rerun with --live --execute to send)")

    # Re-query AFTER sending: `held` was captured before the entry, so writing it would report
    # open_positions 0 on the very run that opened a position — and the notifier renders that
    # field as "Holding N open position(s)". Verified against ticket 539723326.
    try:
        held_now = [p for p in (mt5.positions_get(symbol=symbol) or []) if p.magic == magic]
    except Exception:
        held_now = held

    # Each config gets its OWN state file. The notifiers read
    # data/v5_runs/zigzag_ftmo_state.json; a second harness writing the same path would feed
    # them a mixture of two accounts' states and silently corrupt the dedupe cursor.
    # BUG FIXED 2026-10-19: the previous form was
    #     state_path = ROOT / cfg.get("state_file", STATE.name)
    #     if not state_path.is_absolute(): state_path = ROOT / "data/v5_runs" / ...
    # ROOT is absolute, so `ROOT / name` is ALWAYS absolute and the fallback never ran. Both
    # harnesses wrote to the repo ROOT instead of data/v5_runs/, so the notifier kept reading a
    # stale file and exiting 1 for three days. Resolve the config value on its own first.
    _sf = cfg.get("state_file") or STATE.name
    state_path = Path(_sf) if Path(_sf).is_absolute() else (ROOT / "data" / "v5_runs" / _sf)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(dict(
        computed_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        login=ai.login, server=ai.server, equity=float(ai.equity), symbol=symbol,
        bars=len(df), last_bar=st.asof, stale_hours=st.stale_hours,
        prob=st.prob, threshold=thr, fires=fires, model_fires=st.fires, trend_fc=trend_fc,
        trend_gate_blocked=gate_blocked, prospective=prospective,
        cb_blocked=cb_blocked, cb_move=cb_move, cb_atr=cb_atr, cb_k=cb_k,
        arm=st.arm, n_train=st.n_train, aux_used=list(st.aux_used),
        aux_rejected=[list(x) for x in st.aux_rejected],
        aux_missing=[list(x) for x in aux_missing],
        open_positions=len(held_now), actions=[dict(kind=k, vol=v, why=w, **px)
                                           for k, _, v, w, px in actions],
        sent=sent, executed=bool(args.live and args.execute and is_demo),
        warning="NEGATIVE EXPECTANCY — walk-forward SR -0.76, 0/8 years, DSR 0.000 (V5_FINDINGS 3x)"
    ), indent=2))
    print(f"\nstate -> {state_path.relative_to(ROOT)}")
    pass  # NO mt5.shutdown(): shared bridge (2026-11-06)


if __name__ == "__main__":
    main()
