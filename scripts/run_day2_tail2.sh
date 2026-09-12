#!/usr/bin/env bash
# Second tail (gated on "tail complete"): Think scripted web/email contexts in vLLM, then a second draw of the role-4x arms.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
PY=.venv/bin/python; PYV=.venv-vllm/bin/python
LOG="artifacts/runs/day2_tail2.log"; log() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
inj() { local i rc; for i in $(seq 1 6); do $PY scripts/run_injection.py "$@"; rc=$?; [ $rc -eq 0 ] && return 0; log "run_injection exit $rc, retry $i"; sleep 15; done; return 1; }
ag() { local i rc; for i in $(seq 1 3); do $PYV scripts/run_agentic.py "$@"; rc=$?; [ $rc -eq 0 ] && return 0; log "run_agentic exit $rc, retry $i"; sleep 20; done; return 1; }
while ! grep -q "tail complete" artifacts/runs/day2_tail.log 2>/dev/null; do sleep 60; done
log "tail2 start"
log "5 think scripted contexts (vllm)"
for st in sft rlvr; do for ctx in webpage email; do
  ag --engine vllm --stage $st --run-id agentic01 --context $ctx --scripted --rounds 3 --n-injected 200 --n-clean 12 --draws 1 --max-new-tokens 3000 --gpu-mem 0.85 >> artifacts/runs/agentic01_${st}_$ctx.log 2>&1
done; done
log "6 second draw of role 4x arms"
TYPES=(tc2_fake_turn tc2_fake_completion tc2_important tc2_cot_forgery)
COMMON=(--types "${TYPES[@]}" --n-injected 60 --n-user-control 0 --n-clean 4 --batch-size 10 --max-new-tokens 3000 --max-items-per-process 100000 --no-acts --draw 2)
for st in rlvr sft; do
  a=$($PY -c "print({'sft':11.12,'rlvr':11.44}['$st'])")
  inj "${COMMON[@]}" --run-id inj_frame_def --stage $st --defense none >> artifacts/runs/inj_frame_def_$st.log 2>&1
  inj "${COMMON[@]}" --run-id inj_frame_def --stage $st --defense steer --steer-vector results/generated/roles_gpu/dir_${st}_b16.npy --steer-block 16 --steer-alpha $a >> artifacts/runs/inj_frame_def_$st.log 2>&1
done
log "tail2 complete"
