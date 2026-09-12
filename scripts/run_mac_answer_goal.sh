#!/usr/bin/env bash
# Mac: does a tool-trained model accept planted tool *facts* more than Think? Answer-manipulation goal inside the fetched page/email.
# Instruct SFT/DPO/RL agentic (both contexts) + Think SFT scripted (web page), 12 planted + 12 length-control + 4 clean each.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; LOG=artifacts/runs/mac_answer_goal.log; log() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
C=(--run-id agentic_answer --rounds 3 --types answer tc3_length_control --n-injected 24 --n-clean 4 --batch-size 4)
for st in instruct_sft instruct_dpo instruct; do for ctx in webpage email; do
  log "$st $ctx"; $PY scripts/run_agentic.py --engine hf --stage $st --context $ctx --ask soft --max-new-tokens 1500 "${C[@]}" >> artifacts/runs/agentic_answer.log 2>&1
done; done
log "think sft webpage scripted"; $PY scripts/run_agentic.py --engine hf --stage sft --context webpage --scripted --ask none --max-new-tokens 2500 "${C[@]}" >> artifacts/runs/agentic_answer.log 2>&1
log "think sft email scripted"; $PY scripts/run_agentic.py --engine hf --stage sft --context email --scripted --ask none --max-new-tokens 2500 "${C[@]}" >> artifacts/runs/agentic_answer.log 2>&1
log "answer goal complete"
