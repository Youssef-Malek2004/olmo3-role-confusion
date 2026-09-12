#!/usr/bin/env bash
# Budget-constrained order for the last pod hours: RLVR framings -> span scoring (sft, rlvr) -> RLVR realistic (2 draws)
# -> DPO framings (2 draws) -> DPO realistic (2 draws). Each stage is a separate process; results are pulled by the Mac watcher.
set -uo pipefail
cd "$(dirname "$0")/.."
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1
PYV=.venv-vllm/bin/python; PY=.venv/bin/python
LOG="artifacts/runs/vllm_priority.log"; log() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
log "priority start"
log "rlvr framings"
$PYV scripts/run_injection_vllm.py --run-id inj_matrix --stage rlvr --set all_tool --n-injected 400 --n-clean 12 --draws 3 --max-new-tokens 3000 --gpu-mem 0.85 >> artifacts/runs/inj_matrix_rlvr.log 2>&1
log "span scoring sft rlvr"
for st in sft rlvr; do
  $PY scripts/score_spans.py --run-id inj_matrix --stage $st --roles-run roles_gpu --block 16 --batch-size 16 >> artifacts/runs/score_spans_$st.log 2>&1
  $PY scripts/score_spans.py --run-id inj_matrix --stage $st --roles-run roles_gpu --block 8 --batch-size 16 >> artifacts/runs/score_spans_$st.log 2>&1
done
log "scoring done"
log "rlvr realistic"
$PYV scripts/run_injection_vllm.py --run-id inj_matrix_real --stage rlvr --set realistic --voices plain user system --n-injected 300 --n-user-control 24 --n-clean 12 --draws 2 --max-new-tokens 3000 --gpu-mem 0.85 >> artifacts/runs/inj_matrix_real_rlvr.log 2>&1
log "dpo framings"
$PYV scripts/run_injection_vllm.py --run-id inj_matrix --stage dpo --set all_tool --n-injected 400 --n-clean 12 --draws 2 --max-new-tokens 3000 --gpu-mem 0.85 >> artifacts/runs/inj_matrix_dpo.log 2>&1
$PY scripts/score_spans.py --run-id inj_matrix --stage dpo --roles-run roles_gpu --block 16 --batch-size 16 >> artifacts/runs/score_spans_dpo.log 2>&1
log "dpo realistic"
$PYV scripts/run_injection_vllm.py --run-id inj_matrix_real --stage dpo --set realistic --voices plain user system --n-injected 300 --n-user-control 24 --n-clean 12 --draws 2 --max-new-tokens 3000 --gpu-mem 0.85 >> artifacts/runs/inj_matrix_real_dpo.log 2>&1
log "priority complete"
