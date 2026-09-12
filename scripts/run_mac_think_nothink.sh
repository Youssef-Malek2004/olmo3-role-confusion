#!/usr/bin/env bash
# Think SFT with its reasoning removed (empty <think></think> prefill), scripted call, record + web page. Gated on the Instruct reason battery.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; LOG=artifacts/runs/mac_think_nothink.log; log() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
while ! grep -q "reason battery complete" artifacts/runs/mac_reason.log 2>/dev/null; do sleep 60; done
T=(--engine hf --stage sft --run-id agentic_nothink --types tc2_fake_turn tc2_cot_forgery tc2_important answer --n-injected 48 --n-clean 4 --rounds 3 --batch-size 4 --scripted --ask none --no-think --max-new-tokens 1200)
log "think sft record no-think";  $PY scripts/run_agentic.py "${T[@]}" --context record  --tag rec_nothink >> artifacts/runs/agentic_nothink.log 2>&1
log "think sft webpage no-think"; $PY scripts/run_agentic.py "${T[@]}" --context webpage --tag web_nothink >> artifacts/runs/agentic_nothink.log 2>&1
log "think nothink complete"
