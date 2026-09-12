#!/usr/bin/env bash
# Overnight Mac chain for the own-voice axis: RLVR capture -> style/role span scores on matrix payloads -> style steering vs CoT forgery.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; LOG=artifacts/runs/mac_voice_chain.log
log() { echo "=== $(date '+%F %T') $*" >> "$LOG"; }
log "start"
log "capture voice rlvr"
$PY scripts/capture_voice.py --stage rlvr --n 60 --batch-size 6 >> artifacts/runs/voice01_rlvr.log 2>&1
$PY -c "import numpy as np; [np.save(f'results/generated/voice01/dir_style_neg_rlvr_b{b}.npy', -np.load(f'results/generated/voice01/dir_style_rlvr_b{b}.npy')) for b in (16,24)]"
for st in sft rlvr; do
  log "span scores $st (role b16, style b16, style b24)"
  $PY scripts/score_spans.py --run-id inj_matrix --stage $st --roles-run roles_gpu --block 16 --batch-size 8 --tag role >> artifacts/runs/score_spans_mac_$st.log 2>&1
  $PY scripts/score_spans.py --run-id inj_matrix --stage $st --block 16 --batch-size 8 --direction results/generated/voice01/dir_style_${st}_b16.npy --tag style >> artifacts/runs/score_spans_mac_$st.log 2>&1
  $PY scripts/score_spans.py --run-id inj_matrix --stage $st --block 24 --batch-size 8 --direction results/generated/voice01/dir_style_${st}_b24.npy --tag style >> artifacts/runs/score_spans_mac_$st.log 2>&1
done
log "style steering vs cot forgery / fake turn, SFT"
COMMON=(--run-id inj_style_def --stage sft --types tc2_cot_forgery tc2_fake_turn --n-injected 50 --n-user-control 0 --n-clean 4 --batch-size 4 --max-new-tokens 2000 --max-items-per-process 12)
inj() { local i rc; for i in $(seq 1 15); do $PY scripts/run_injection.py "$@"; rc=$?; [ $rc -eq 0 ] && return 0; [ $rc -ne 3 ] && log "exit $rc retry $i"; sleep 90; done; }
inj "${COMMON[@]}" >> artifacts/runs/inj_style_def_sft.log 2>&1
inj "${COMMON[@]}" --defense steer --steer-vector results/generated/voice01/dir_style_neg_sft_b16.npy --steer-block 16 --steer-alpha 3.8 >> artifacts/runs/inj_style_def_sft.log 2>&1
inj "${COMMON[@]}" --defense steer --steer-vector results/generated/voice01/dir_style_neg_sft_b16.npy --steer-block 16 --steer-alpha 7.6 >> artifacts/runs/inj_style_def_sft.log 2>&1
inj "${COMMON[@]}" --defense random --steer-block 16 --steer-alpha 7.6 --random-seed 0 >> artifacts/runs/inj_style_def_sft.log 2>&1
$PY scripts/summarize_injection.py --run-id inj_style_def --effective-cap 2000 >> artifacts/runs/inj_style_def_summary.log 2>&1
log "complete"
