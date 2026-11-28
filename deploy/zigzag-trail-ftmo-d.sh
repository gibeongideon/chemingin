#!/bin/bash
# ZigZag arm D on FTMO-Demo (bridge 18814) — magic 360622 (Y2UP x CHAMP size dials, data/v5_runs/zigzag_dial_ab/PREREG.md).
#
# THE A/B TWIN OF ARM B (360608). Identical in every respect except the §3eu DOWNSIDE-VOL SIZE
# DIAL: when downside_vol(500 H1 bars) >= its own rolling median(2000), the fire is still taken
# but at HALF SIZE. One variable differs, so the live gap is attributable to the dial.
#
# WHY A DIAL AND NOT A VETO. §3et tested the veto and it failed the incremental half-sample bar
# 4/8 — weak in every second half. §3eu's taper passes 16/16 cells and improves BOTH halves in
# BOTH sources. The difference is the EVENT SET: skipping a fire frees this harness's single slot
# for a later, different fire, and that reshuffling destroys the information. Sizing does not.
#
# NOT A VALIDATED EDGE: the base rule's own Sharpe is [-0.30,+0.70] CME / [-0.26,+1.08] Switch
# (§3dg/§3dx). This changes the SHAPE of something undistinguishable from zero. Demo only.
# Report: python scripts/v5_zigzag_ab_vol_taper.py
# Disarm: systemctl --user disable --now zigzag-trail-ftmo-d.timer
set -uo pipefail
cd /home/trader/MT5 || exit 1
export NOTIFY_NO_TELEGRAM=1   # phone gets these via trade-events-ftmo; this job -> email
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/zigzag-trail-ftmo-d.log
mkdir -p data/v5_runs
exec 9>/tmp/zigzag-trail-ftmo-d.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) previous run still going, skipping" >> "$LOG"; exit 0; }
echo "===== $(date -u +%FT%TZ) =====" >> "$LOG"
$PY scripts/v5_zigzag_ftmo.py --config configs/v5_zigzag_trail_ftmo_d.json --live --execute \
    >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
if [ "$(wc -l < "$LOG")" -gt 30000 ]; then tail -10000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"; fi
