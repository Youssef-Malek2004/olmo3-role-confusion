#!/usr/bin/env bash
# Day-2 Step 0 queue (vLLM at reduced memory so the HF defense chain can share the A100):
#   DPO framings (2 draws) -> RLVR framings draw 3 -> RLVR realistic (2 draws) -> DPO realistic (2 draws)
#   -> DPO span scores (role b16/b8, HF prefill) -> DPO own-voice capture + style scores -> summaries.
set -uo pipefail
cd "$(dirname "$0")/.."
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYV=.venv-vllm/bin/python; PY=.venv/bin/python; GM="${GPU_MEM:-0.35}"
LOG="artifacts/runs/day2_vllm.log"; log() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
vl() { local i rc; for i in $(seq 1 4); do $PYV scripts/run_injection_vllm.py "$@" --gpu-mem "$GM"; rc=$?; [ $rc -eq 0 ] && return 0; log "vllm exit $rc, retry $i"; sleep 20; done; return 1; }
log "day2 vllm start GPU_MEM=$GM"
log "dpo framings 2 draws"
vl --run-id inj_matrix --stage dpo --set all_tool --n-injected 400 --n-clean 12 --draws 2 --max-new-tokens 3000 >> artifacts/runs/inj_matrix_dpo.log 2>&1
log "rlvr framings draw 3"
vl --run-id inj_matrix --stage rlvr --set all_tool --n-injected 400 --n-clean 12 --draws 3 --max-new-tokens 3000 >> artifacts/runs/inj_matrix_rlvr.log 2>&1
log "rlvr realistic 2 draws"
vl --run-id inj_matrix_real --stage rlvr --set realistic --voices plain user system --n-injected 300 --n-user-control 24 --n-clean 12 --draws 2 --max-new-tokens 3000 >> artifacts/runs/inj_matrix_real_rlvr.log 2>&1
log "dpo realistic 2 draws"
vl --run-id inj_matrix_real --stage dpo --set realistic --voices plain user system --n-injected 300 --n-user-control 24 --n-clean 12 --draws 2 --max-new-tokens 3000 >> artifacts/runs/inj_matrix_real_dpo.log 2>&1
log "vllm done; dpo span scores (role b16, b8 via --direction; raw role acts not on this pod)"
for b in 16 8; do
  $PY scripts/score_spans.py --run-id inj_matrix --stage dpo --block $b --batch-size 16 --direction results/generated/roles_gpu/dir_dpo_b$b.npy --tag role >> artifacts/runs/score_spans_dpo.log 2>&1
done
log "dpo own-voice capture + style scores"
$PY scripts/capture_voice.py --stage dpo --batch-size 8 >> artifacts/runs/voice_dpo.log 2>&1
for b in 16 24; do
  $PY scripts/score_spans.py --run-id inj_matrix --stage dpo --block $b --batch-size 16 --direction results/generated/voice01/dir_style_dpo_b$b.npy --tag style >> artifacts/runs/score_spans_dpo.log 2>&1
done
log "summaries"
$PY scripts/summarize_injection.py --run-id inj_matrix --effective-cap 3000 >> artifacts/runs/inj_matrix_summary.log 2>&1
$PY scripts/summarize_injection.py --run-id inj_matrix_real --effective-cap 3000 >> artifacts/runs/inj_matrix_real_summary.log 2>&1
log "day2 vllm complete"
