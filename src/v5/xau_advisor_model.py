"""Load the frozen advisor artifact and turn two price frames into one advisory reading.

THE CONTRACT. `advise(h4, m15)` returns a dict or raises. It NEVER returns a made-up probability:
every failure path sets `health` to something other than "ok" and leaves the probability None, so
`xau_advisor_state.step` freezes rather than moving on a fabricated number. A silently degraded
model is the failure mode `v5_zigzag_message.py:115-119` already exists to prevent, and here it
would be worse, because the output of this file is advice a person acts on.

WHAT IS QUOTED TO THE USER, AND WHY IT IS NOT `p`. §3az found no calibrator with positive Brier
skill on any of 24 cells: the RANKING carries information, the per-bar number does not. So the
reading is mapped through the artifact's frozen reliability table to the MEASURED frequency of
the bucket it lands in — "bars that scored this high went on to move adversely 64.6% of the
time, 212 of them, CI [57.5, 72.6]" — which is a fact about the sample rather than a claim about
this bar. `p` is still returned, labelled as a ranking score, so nothing is hidden.

THE DIRECTION PANEL IS RETURNED WITH `usable: False`. It would be easy, and wrong, to omit it:
the user asked for a direction call and is owed the measurement, not silence. It comes back with
its own refutation attached (the max-stat null p-values and the drift comparison) so it can be
judged rather than acted on.

FEED FAMILY. Features must be built from the SAME quote series the artifact was trained on
(`data/XAUUSD_{H4,M15}_long.csv`, refreshed from bridge 18814). Mixing feeds moves gold's close
by a median $5.84, rising to $13.33 in 2026. `advise` does not fetch anything itself precisely so
the caller has to be explicit about where the frames came from.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACT = ROOT / "data" / "models" / "xau_advisor"

_CACHE: dict = {}


class AdvisorUnavailable(RuntimeError):
    """Raised for every condition that must freeze the state rather than produce a reading."""


def load(path: Path | str = ARTIFACT) -> dict:
    """Load and cache the artifact. Cached by mtime so a retrain is picked up without a restart."""
    p = Path(path)
    jf, bf = p.with_suffix(".json"), p.with_suffix(".joblib")
    if not jf.exists() or not bf.exists():
        raise AdvisorUnavailable(f"artifact missing: {jf.name} / {bf.name} — run "
                                 "scripts/v5_train_advisor.py")
    key = (str(p), jf.stat().st_mtime_ns, bf.stat().st_mtime_ns)
    if _CACHE.get("key") != key:
        import joblib
        _CACHE.clear()
        _CACHE.update(key=key, meta=json.loads(jf.read_text()), blobs=joblib.load(bf))
    return _CACHE


def _predict(blob: dict, X: pd.DataFrame) -> float:
    """One row through the frozen estimator and its frozen calibrator."""
    cols = blob["features"]
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise AdvisorUnavailable(f"feature columns missing at serve time: {missing[:5]}")
    x = X[cols].iloc[[-1]]
    if not np.isfinite(x.values).all():
        bad = [c for c, v in zip(cols, x.values[0]) if not np.isfinite(v)]
        raise AdvisorUnavailable(f"non-finite features on the decision bar: {bad[:5]}")
    v = blob["scaler"].transform(x.values) if blob.get("scaler") is not None else x.values
    p = float(blob["model"].predict_proba(v)[:, 1][0])
    cal = blob.get("calibrator")
    if cal is not None:
        p = float(cal.predict(np.array([p]))[0]) if hasattr(cal, "predict") and not hasattr(
            cal, "predict_proba") else float(cal.predict_proba(np.array([[p]]))[:, 1][0])
    if not (0.0 <= p <= 1.0):
        raise AdvisorUnavailable(f"model returned {p}, outside [0,1]")
    return p


def bucket_for(p: float, reliability: list) -> dict | None:
    """The frozen reliability row this reading falls in — the number that gets quoted.

    Returns None when the reading falls in no displayed bucket, which is not an error: thin
    buckets were merged away at measurement time precisely because n < 200 could not support a
    stated frequency. The caller must then say it has no measured frequency for this reading
    rather than interpolate one.
    """
    for row in reliability:
        lo, hi = float(row["lo"]), float(row["hi"])
        if (lo <= p < hi) or (hi >= 1.0 and p >= lo):
            return row
    return None


def advise(h4: pd.DataFrame, m15: pd.DataFrame, path: Path | str = ARTIFACT) -> dict:
    """The full reading for the newest CLOSED H4 bar. Raises AdvisorUnavailable, never guesses."""
    from scripts.v5_advisor_measure import features_for      # noqa: PLC0415  (heavy import)

    art = load(path)
    meta, blobs = art["meta"], art["blobs"]
    warnings: list[str] = []

    for name, df in (("h4", h4), ("m15", m15)):
        if df is None or not len(df):
            raise AdvisorUnavailable(f"{name} frame is empty")
    asof = pd.Timestamp(h4.index[-1])
    # The decision is made at the bar's CLOSE, not its stamp (§3az: the stamp-vs-close error
    # produced AUC 0.901). Everything downstream timestamps the reading at the close.
    decision_time = asof + pd.Timedelta(hours=4)

    # The M15 path must reach the decision bar, or the SOURCE features describe an older bar.
    m15_last = pd.Timestamp(m15.index[-1])
    if m15_last < asof:
        raise AdvisorUnavailable(
            f"M15 ends {m15_last} but the decision bar is {asof} — the intrabar features would "
            "describe an earlier bar than the one being scored")

    out = {"asof": asof, "decision_time": decision_time,
           "model_id": meta.get("model_id"), "trained_on": meta.get("trained_on"),
           "findings": meta.get("findings"), "warnings": warnings, "health": "ok"}

    # ---------------------------------------------------------------- adverse (the shipped one)
    a = meta["adverse"]
    Xa = features_for(a["arm"], h4, m15)
    if pd.Timestamp(Xa.index[-1]) != asof:
        raise AdvisorUnavailable(f"feature frame ends {Xa.index[-1]}, decision bar is {asof}")
    p_adv = _predict(blobs["adverse"], Xa)
    b = bucket_for(p_adv, a["measured"]["reliability"])
    out["adverse"] = {
        "p": p_adv,
        "usable": a["usable"], "tier": a["tier"],
        "cell": a["cell"], "k_atr": a["k_atr"], "hours": a["hours"],
        "warn_at": a["warn_at"], "clear_at": a["clear_at"],
        "base_rate": a["measured"]["base_rate"],
        "resolved_frac": a["resolved_frac"],
        "observed": None if b is None else float(b["observed"]),
        "observed_n": None if b is None else int(b["n"]),
        "observed_ci": None if b is None else [b.get("ci_lo"), b.get("ci_hi")],
        "bucket": None if b is None else [float(b["lo"]), float(b["hi"])],
        "label": ("probability that gold touches "
                  f"-{a['k_atr']}xATR before +{a['k_atr']}xATR within {a['hours']}h"),
        "advice": a["advice_warn"] if p_adv >= a["warn_at"] else a["advice_clear"],
        "limits": a["limits"],
        "auc": a["measured"].get("auc"), "bss": a["measured"].get("bss"),
    }
    if b is None:
        warnings.append(f"adverse reading {p_adv:.3f} falls outside every displayed reliability "
                        "bucket — no measured frequency can be quoted for it")

    # ---------------------------------------------------------------- direction (usable: false)
    d = meta["direction"]
    try:
        Xd = features_for(d["arm"], h4, m15)
        p_up = _predict(blobs["direction"], Xd)
    except AdvisorUnavailable as e:                      # never fatal: this panel is advisory
        p_up, _ = None, warnings.append(f"direction panel unavailable: {e}")
    out["direction"] = {
        "p_up": p_up, "usable": d["usable"], "tier": d["tier"],
        "horizon_hours": d["horizon_hours"],
        "base_rate": d["measured"]["base_rate"],
        "edge_note": d["edge_note"], "drift_note": d["drift_note"],
        "print_instead": d["print_instead"],
    }
    return out


def health_of(reading: dict | None, now: pd.Timestamp, max_age_h: float = 9.0) -> str:
    """Translate a reading into the state machine's health field.

    9 hours, not 4: gold's H4 grid has real gaps (the daily broker break, the weekend), so a
    4-hour rule would call a normal Friday evening stale. Two H4 periods plus slack is late
    enough to mean something is actually wrong on a weekday.
    """
    if reading is None:
        return "model_unavailable"
    age = (pd.Timestamp(now) - pd.Timestamp(reading["decision_time"])).total_seconds() / 3600
    if age > max_age_h:
        return "stale_feed"
    if reading.get("adverse", {}).get("p") is None:
        return "model_unavailable"
    return "ok"
