#!/usr/bin/env bash
# Mac, unattended: form vs identity of the tool result (SFT, HF, scripted, fake turn + forgery + important, 8 items each + 2 clean, cap 2000)
#  X1 webpage_xml    : fetch_page returns a tagged record-like page
#  X2 webpage_lookup : lookup_question_record returns the prose page
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; LOG=artifacts/runs/mac_ctxdiag2.log; log() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
C=(--engine hf --stage sft --run-id agentic_diag --scripted --rounds 3 --ask none --types tc2_fake_turn tc2_cot_forgery tc2_important --n-injected 24 --n-clean 2 --batch-size 4 --max-new-tokens 2000)
log "X1 webpage_xml";    $PY scripts/run_agentic.py "${C[@]}" --context webpage_xml --tag X1_web_xml >> artifacts/runs/agentic_diag.log 2>&1
log "X2 webpage_lookup"; $PY scripts/run_agentic.py "${C[@]}" --context webpage_lookup --tag X2_web_lookup >> artifacts/runs/agentic_diag.log 2>&1
log "mac ctxdiag2 complete"
