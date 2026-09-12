#!/usr/bin/env bash
# Mac: is the Instruct/Think gap about the reasoning block? (gated on the answer-goal chain)
#  I-A  Instruct RL, record context, agentic, baseline                (like-for-like with Think's record numbers)
#  I-B  Instruct RL, record + webpage, reason=think                   (asked to reason before acting)
#  I-C  Instruct RL, record + webpage, reason=think_example           (with a worked example of reasoning)
#  T-A  Think SFT, record, scripted, --no-think                       (reasoning removed)
#  T-B  Think SFT, webpage, scripted, --no-think
# 5 payloads x 8 items + 4 clean per run.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; LOG=artifacts/runs/mac_cot_confound.log; log() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
while ! grep -q "answer goal complete" artifacts/runs/mac_answer_goal.log 2>/dev/null; do sleep 60; done
T=(--types tc2_fake_turn tc2_cot_forgery tc2_important answer tc3_length_control --n-injected 40 --n-clean 4 --batch-size 4 --rounds 3)
log "I-A instruct record baseline";      $PY scripts/run_agentic.py --engine hf --stage instruct --run-id agentic_cot --context record --ask none --max-new-tokens 1500 --tag IA_record_base "${T[@]}" >> artifacts/runs/agentic_cot.log 2>&1
for ctx in record webpage; do
  log "I-B instruct $ctx reason=think";   $PY scripts/run_agentic.py --engine hf --stage instruct --run-id agentic_cot --context $ctx --ask $([ $ctx = record ] && echo none || echo soft) --reason think --max-new-tokens 2500 --tag IB_${ctx}_think "${T[@]}" >> artifacts/runs/agentic_cot.log 2>&1
  log "I-C instruct $ctx reason=think_example"; $PY scripts/run_agentic.py --engine hf --stage instruct --run-id agentic_cot --context $ctx --ask $([ $ctx = record ] && echo none || echo soft) --reason think_example --max-new-tokens 2500 --tag IC_${ctx}_thinkex "${T[@]}" >> artifacts/runs/agentic_cot.log 2>&1
done
for ctx in record webpage; do
  log "T Think SFT $ctx no-think";        $PY scripts/run_agentic.py --engine hf --stage sft --run-id agentic_cot --context $ctx --scripted --ask none --no-think --max-new-tokens 1200 --tag T_${ctx}_nothink "${T[@]}" >> artifacts/runs/agentic_cot.log 2>&1
done
log "cot confound complete"
