"""Send the advisor card from the VPS over HTTPS. Reads state from disk; computes nothing.

The split is deliberate: `v5_xau_advisor.py` decides, this file delivers. A sender that can also
compute will eventually compute something different from what was stored, and the stored state is
what the dwell timer and thrash guard are counted against.

The VPS cannot reach smtp.gmail.com on any port (re-verified 2026-09-10), so delivery goes over
HTTPS via `v5_notify.notify`, which picks the first configured channel from
`/home/trader/MT5/.env.notify`. Until a credential exists it prints what it would have sent, so
the pipeline is verifiable before any account is.

`seen` advances ONLY after a successful send, so a failed delivery retries next run instead of
being silently swallowed (`v5_trend_signal.py:185-193`).

    python scripts/v5_xau_advisor_notify.py
    python scripts/v5_xau_advisor_notify.py --dry
    python scripts/v5_xau_advisor_notify.py --always      # delivery test
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v5_notify import channel, env, notify  # noqa: E402
from scripts.v5_xau_advisor_message import build, load_seen, save_seen  # noqa: E402
from src.v5.xau_advisor_model import load  # noqa: E402

STATE = ROOT / "data" / "v5_runs" / "xau_advisor_state.json"
SEEN = ROOT / "data" / "v5_runs" / "xau_advisor_seen.json"
STALE_ALERT_H = 12          # gold's H4 grid has real gaps; see health_of's reasoning


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--always", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    ch = channel(env())
    print(f"channel: {ch}")
    if ch == "smtp":
        print("  WARNING: only SMTP is configured and this host cannot reach SMTP. "
              "Add an HTTPS provider key to .env.notify.")
    send = (lambda s, b: print(f"[dry] {s}\n\n{b}")) if a.dry else notify

    meta = load()["meta"]
    seen = load_seen(SEEN)
    now = datetime.now(timezone.utc)

    if not STATE.exists():
        m = build({}, None, meta, seen, now=now,
                  broken=f"No state file at {STATE}. The compute step has never run, or has "
                         f"never succeeded.")
    else:
        s = json.loads(STATE.read_text())
        age_h = (now - datetime.fromisoformat(s["computed_utc"])).total_seconds() / 3600
        reading = None
        if s.get("p") is not None and not s.get("broken"):
            adv = meta["adverse"]
            reading = {"asof": s.get("state", {}).get("asof"),
                       "decision_time": s.get("state", {}).get("asof"),
                       "adverse": {"p": s["p"], "observed": s.get("observed"),
                                   "observed_n": None, "observed_ci": None,
                                   "base_rate": adv["measured"]["base_rate"],
                                   "label": ("probability that gold touches "
                                             f"-{adv['k_atr']}xATR before +{adv['k_atr']}xATR "
                                             f"within {adv['hours']}h"),
                                   "advice": (adv["advice_warn"]
                                              if s["p"] >= adv["warn_at"]
                                              else adv["advice_clear"])},
                       "direction": {"p_up": None}}
            # fill n / CI from the frozen bucket this reading lands in
            from src.v5.xau_advisor_model import bucket_for
            b = bucket_for(s["p"], adv["measured"]["reliability"])
            if b:
                reading["adverse"].update(observed=float(b["observed"]),
                                          observed_n=int(b["n"]),
                                          observed_ci=[b.get("ci_lo"), b.get("ci_hi")])
        broken = s.get("broken")
        if age_h > STALE_ALERT_H and not broken and not s.get("market_closed"):
            broken = (f"The newest advisor state is {age_h:.0f}h old (computed "
                      f"{s['computed_utc']}). A CLEAR reading this old is not evidence of a "
                      f"calm market.")
        m = build(s.get("state", {}), reading, meta, seen, now=now,
                  market=s.get("market"), broken=broken,
                  market_closed=bool(s.get("market_closed")))

    print(m["body"])
    print(f"\ntrigger={m['trigger']}  should_send={m['should_send']}")

    if m["should_send"] or a.always:
        ok = send(m["subject"], m["body"])
        if ok is False:
            sys.exit("send FAILED — seen NOT advanced, will retry next run")
        if not a.dry:
            save_seen(SEEN, m["seen"])
    else:
        print("(nothing sent — no state change and this hour's heartbeat already went)")


if __name__ == "__main__":
    main()
