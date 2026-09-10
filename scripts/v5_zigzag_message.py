"""SINGLE SOURCE OF TRUTH for the ZigZag harness message — bands, subject, body, dedupe.

WHY THIS FILE EXISTS. The harness now has two senders: `v5_zigzag_mail_relay.py` on the desktop
(Gmail SMTP) and `v5_zigzag_notify.py` on the VPS (HTTPS provider). If each built its own text
they would eventually disagree about what fired, which is the exact failure the executor's state
file was introduced to prevent ("the manual channel and the executor never disagree"). Both
senders import from here and neither formats anything itself.

THE BANDS (user request 2026-09-10):
    prob >= threshold (0.60)   TRADABLE      — the demo executor has already placed it
    NOTIFY_MIN (0.55) .. 0.60  NOT TRADABLE  — notification only, no order exists anywhere
    below 0.55                 silent

The 0.60 line is the WALK-FORWARD-SELECTED threshold (§3x: 2023/24/25 all chose 0.6) and lives
in the executor's config. Nothing here can change what gets traded — this module only decides
what gets *said*.

ANTI-SPAM. Both senders run hourly, so a probability parked at 0.57 would mail every hour.
`should_send` fires a watch notice only when the band is ENTERED; leaving and re-entering
notifies again, and a TRADABLE fire always notifies.

*** NEGATIVE EXPECTANCY. *** §3x: walk-forward SR -0.76, 0/8 years, DSR 0.000. §3as: EV/fire
+0.05% against +0.15% for simply holding, mean fire offset +0.771 — a confirmation detector.
Every message carries the disclaimer. A notifier that quietly hands over trade instructions from
a disproven model is how a research artifact becomes a live loss.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

NOTIFY_MIN = 0.55
DISCLAIMER = "Demo test only — this strategy is expected to lose money (walk-forward -0.76)."


def band(prob, thr: float) -> str:
    """Which notification band a probability sits in. NaN-safe."""
    if prob is None or not (prob == prob):
        return "none"
    if prob >= thr:
        return "tradable"
    if prob >= NOTIFY_MIN:
        return "watch"
    return "none"


def load_band(path: Path) -> str:
    try:
        return json.loads(Path(path).read_text()).get("band", "none")
    except Exception:
        return "none"


def save_band(path: Path, b: str, prob) -> None:
    """Never fatal: losing the dedupe file costs one duplicate email, not the timer."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(
            band=b, prob=(float(prob) if prob is not None else None),
            at=datetime.now(timezone.utc).isoformat(timespec="seconds"))))
    except Exception:
        pass


def build(s: dict, prev_band: str = "none") -> dict:
    """Turn an executor state dict into {band, should_send, subject, body}."""
    prob = s.get("prob")
    thr = s.get("threshold", 0.60)
    cur = band(prob, thr)
    sym = s.get("symbol", "XAUUSD")
    arm_line = None
    if s.get("arm"):
        arm_line = (f"Model: {s['arm']}"
                    + (f" (+{', '.join(s.get('aux_used', []))})" if s.get("aux_used") else ""))

    pp = s.get("prospective") or {}
    slp, tpp = pp.get("sl_pct", 0.005), pp.get("tp_pct", 0.006)
    if s.get("actions"):
        L = ["TRADABLE", ""]
        for a in s["actions"]:
            if a["kind"] == "BUY":
                L += [f"BUY {sym}   {a['vol']} lots", "",
                      f"  Entry   {a['entry']}      (market)",
                      f"  SL      {a['sl']}      (-{slp:.2%})",
                      f"  TP      {a['tp']}      (+{tpp:.2%})",
                      f"  Close   {a['close_by']} if neither is hit"]
            else:
                L += [f"CLOSE {sym}   {a['vol']} lots", "",
                      f"  Opened at {a['entry']}, held {a.get('age_h')}h — 48h limit reached."]
            L.append("")
        L.append("The demo bot has already placed this. Mirror it by hand if you want to.")
        a0 = s["actions"][0]
        # subject carries the order itself, so it is actionable from a phone lock screen
        subj = (f"[zigzag] TRADABLE — BUY {a0['vol']} lots {sym} @ {a0['entry']}"
                if a0["kind"] == "BUY" else f"[zigzag] CLOSE {a0['vol']} lots {sym}")
    elif cur == "watch":
        L = ["NOT TRADABLE", "",
             f"{sym}   P(bottom) {prob:.3f}   needs {thr:.2f}", "",
             f"  Would be   BUY {pp.get('lots')} lots @ {pp.get('entry')}",
             f"  SL {pp.get('sl')}   TP {pp.get('tp')}", "",
             "No order has been placed. Notification only."]
        subj = f"[zigzag] NOT TRADABLE — {sym} P {prob:.2f} (needs {thr:.2f})"
    else:
        pstr = f"{prob:.2f}" if (prob is not None and prob == prob) else "n/a"
        L = [f"{sym}   no trade.   P(bottom) {pstr}  vs  {thr:.2f} threshold."]
        if s.get("open_positions"):
            L.append(f"Holding {s['open_positions']} open position(s); "
                     f"TP/SL are on the broker.")
        subj = f"[zigzag] {sym} no trade (P {pstr}/{thr:.2f})"

    if arm_line:
        L.append(arm_line)
    # a silently degraded model is the failure mode that matters, so surface it
    if s.get("aux_rejected") or s.get("aux_missing"):
        bad = [a for a, _ in (s.get("aux_rejected") or [])] + \
              [a for a, _ in (s.get("aux_missing") or [])]
        L.append(f"(reduced model — unavailable: {', '.join(bad)})")
    L += ["", DISCLAIMER]

    return dict(band=cur,
                should_send=bool(s.get("actions")) or (cur == "watch" and prev_band != "watch"),
                subject=subj, body="\n".join(L))
