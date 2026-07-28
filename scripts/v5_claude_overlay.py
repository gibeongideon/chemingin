"""Claude as a VETO/DOWNSIZE overlay on the H4 XAU champion (no new API calls).

Option-3 test: the LLM never opens its own trade. It can only cut the champion's
position when it disagrees. Reuses the 150 cached Claude H4 decisions
(data/v5_runs/claude_h4_decisions.csv) + the champion long-only signal on the same
bars, so this is FREE and fast. The crisp question: on the bars where Claude
vetoed (said SELL while champion was long), was the champion actually about to
lose? If yes, the veto adds value; if not, it's noise that clips winners.
"""
import sys
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from src.v5.xau_dual_signals import SIGNALS
SPREAD = 0.45

d = pd.read_csv(f"{ROOT}/data/XAUUSD_H4_long.csv", parse_dates=["time"]).set_index("time")
d = d[~d.index.duplicated(keep="last")].sort_index()
fc = SIGNALS["champ"](d["close"])                        # long-only champion forecast (>=0)
base_pos = (fc.clip(lower=0) > 0.15).astype(float)       # champion "long on" per bar

C = pd.read_csv(f"{ROOT}/data/v5_runs/claude_h4_decisions.csv", parse_dates=["t"]).set_index("t")
C["nret"] = d["close"].reindex(C.index).pct_change().shift(-1).reindex(C.index)  # bar i -> i+1
C["ret"] = d["close"].pct_change().shift(-1).reindex(C.index)
C["base"] = base_pos.reindex(C.index).fillna(0.0)
C = C.dropna(subset=["ret"])
spread_frac = SPREAD / d["close"].reindex(C.index).mean()

def sim(pos):
    turn = pos.diff().abs().fillna(pos.abs())
    net = pos * C["ret"] - turn * spread_frac
    cum = float((1 + net).prod() - 1) * 100
    return cum

opp = (C["act"] == "sell") & C["conf"].isin(["high", "medium"])   # Claude opposes the long
opp_lo = (C["act"] == "sell")                                     # incl. low conf
base = C["base"]
variants = {
    "champion base (no overlay)": base,
    "VETO on Claude sell (med+)": base.where(~opp, 0.0),
    "DOWNSIZE 0.5 on sell (med+)": base * np.where(opp, 0.5, 1.0),
    "VETO on ANY Claude sell": base.where(~opp_lo, 0.0),
}
print(f"H4 champion + Claude veto overlay — {len(C)} bars "
      f"{C.index[0]:%Y-%m-%d} -> {C.index[-1]:%Y-%m-%d}, spread ${SPREAD}\n")
print(f"{'variant':30s} {'net%':>7} {'#bars long':>10}")
for name, pos in variants.items():
    print(f"{name:30s} {sim(pos):+7.2f} {int((pos>0).sum()):>10}")

# the crisp diagnostic: on the veto bars, what was the champion base P&L?
vb = C[opp & (base > 0)]
if len(vb):
    saved = -float((vb["base"] * vb["ret"]).sum() * 100)
    wr = float((vb["ret"] < 0).mean() * 100)
    print(f"\nveto bars (Claude sell med+ WHILE champion long): {len(vb)}")
    print(f"  champion base return on those bars: {float((vb['base']*vb['ret']).sum()*100):+.2f}%"
          f"  -> vetoing them {'SAVED' if saved>0 else 'COST'} {abs(saved):.2f}%")
    print(f"  of those bars, {wr:.0f}% were actually down bars (veto is right if >50%)")
else:
    print("\nno veto bars (Claude never opposed a champion long at med+ conf)")
