#!/usr/bin/env bash
# Corrected CoT confound (Mac). 5 payloads x 8 + 4 clean per arm.
#  T1 Think SFT record scripted --no-think    T2 Think SFT webpage scripted --no-think
#  I1 Instruct RL record scripted baseline    I2 Instruct RL record scripted --force-think + reason prompt
#  I3 Instruct RL webpage agentic --force-think + reason prompt   (baseline = agentic_stages instruct webpage)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; LOG=artifacts/runs/mac_cot_confound2.log; log() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
T=(--run-id agentic_cot2 --types tc2_fake_turn tc2_cot_forgery tc2_important answer tc3_length_control --n-injected 40 --n-clean 4 --batch-size 4 --rounds 3)
log "T1 think sft record no-think";   $PY scripts/run_agentic.py --engine hf --stage sft --context record --scripted --ask none --no-think --max-new-tokens 1200 --tag T1_record_nothink "${T[@]}" >> artifacts/runs/agentic_cot2.log 2>&1
log "T2 think sft webpage no-think";  $PY scripts/run_agentic.py --engine hf --stage sft --context webpage --scripted --ask none --no-think --max-new-tokens 1200 --tag T2_webpage_nothink "${T[@]}" >> artifacts/runs/agentic_cot2.log 2>&1
log "I1 instruct record scripted";    $PY scripts/run_agentic.py --engine hf --stage instruct --context record --scripted --ask none --max-new-tokens 1500 --tag I1_record_base "${T[@]}" >> artifacts/runs/agentic_cot2.log 2>&1
log "I2 instruct record forced think"; $PY scripts/run_agentic.py --engine hf --stage instruct --context record --scripted --ask none --reason think --force-think --max-new-tokens 2500 --tag I2_record_think "${T[@]}" >> artifacts/runs/agentic_cot2.log 2>&1
log "I3 instruct webpage forced think"; $PY scripts/run_agentic.py --engine hf --stage instruct --context webpage --ask soft --reason think --force-think --max-new-tokens 2500 --tag I3_webpage_think "${T[@]}" >> artifacts/runs/agentic_cot2.log 2>&1
log "cot confound2 complete"
