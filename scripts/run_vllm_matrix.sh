#!/usr/bin/env bash
# Compliance matrix on vLLM. Waits for the HF chain to finish (gpu_chain.log 'chain complete'), then runs:
#   all tool framings + ablations (16 types, plain voice), 25 items each, 3 draws, SFT/DPO/RLVR, cap 3000
#   realistic goals (answer, link, deny, toolcall), 3 voices, 25 per type x voice, 3 draws, 3 stages
# FlashInfer JIT paths disabled (no nvcc on the pod).
set -uo pipefail
cd "$(dirname "$0")/.."
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1
PY=.venv-vllm/bin/python
LOG="artifacts/runs/vllm_matrix.log"; mkdir -p artifacts/runs
log() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
# Sharing the GPU with the HF chain caused repeated OOMs in the chain; run only after it completes, at full memory.
GPU_MEM="${GPU_MEM:-0.85}"
until grep -q 'chain complete' artifacts/runs/gpu_chain.log 2>/dev/null; do sleep 120; done
log "matrix start"
for stage in sft rlvr dpo; do
  log "tool framings+ablations $stage"
  $PY scripts/run_injection_vllm.py --run-id inj_matrix --stage "$stage" --set all_tool --n-injected 400 --n-clean 12 --draws 3 \
      --max-new-tokens 3000 --gpu-mem "$GPU_MEM" >> "artifacts/runs/inj_matrix_${stage}.log" 2>&1
  log "realistic goals $stage"
  $PY scripts/run_injection_vllm.py --run-id inj_matrix_real --stage "$stage" --set realistic --voices plain user system \
      --n-injected 300 --n-user-control 24 --n-clean 12 --draws 3 --max-new-tokens 3000 --gpu-mem "$GPU_MEM" >> "artifacts/runs/inj_matrix_real_${stage}.log" 2>&1
done
log "summaries"
.venv/bin/python scripts/summarize_injection.py --run-id inj_matrix --effective-cap 3000 >> artifacts/runs/inj_matrix_summary.log 2>&1
.venv/bin/python scripts/summarize_injection.py --run-id inj_matrix_real --effective-cap 3000 >> artifacts/runs/inj_matrix_real_summary.log 2>&1
log "matrix complete"
