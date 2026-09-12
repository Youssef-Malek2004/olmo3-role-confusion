#!/usr/bin/env bash
# Mac diagnostics for Think's context gap (SFT, HF, scripted call, fake turn + forgery, 9 items each + 2 clean, cap 2000):
#  M1 webpage, user turn = question only, payload nouns = page      (removes the "fetch it for context" sentence)
#  M2 webpage, user turn = question only, payload nouns = record    (M1 + record wording)
#  M3 record via the runner, 9+9 items                              (pipeline check, small)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; LOG=artifacts/runs/mac_ctxdiag.log; log() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
C=(--engine hf --stage sft --run-id agentic_diag --scripted --rounds 3 --types tc2_fake_turn tc2_cot_forgery --n-injected 18 --n-clean 2 --batch-size 4 --max-new-tokens 2000)
log "M1 webpage ask=none nouns=context"; $PY scripts/run_agentic.py "${C[@]}" --context webpage --ask none --tag M1_web_asknone >> artifacts/runs/agentic_diag.log 2>&1
log "M2 webpage ask=none nouns=record";  $PY scripts/run_agentic.py "${C[@]}" --context webpage --ask none --payload-nouns record --tag M2_web_asknone_recnoun >> artifacts/runs/agentic_diag.log 2>&1
log "M3 record via runner";              $PY scripts/run_agentic.py "${C[@]}" --context record --tag M3_record >> artifacts/runs/agentic_diag.log 2>&1
log "mac ctxdiag complete"
