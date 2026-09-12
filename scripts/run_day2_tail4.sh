#!/usr/bin/env bash
# Reordered end of day 2: wait for the running Think-email vLLM job, then the context-gap diagnostic (was tail3),
# then the second draw of the no-defense and role-4x arms (was tail2 step 6). Credit-aware ordering: diagnostic first.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
PY=.venv/bin/python; PYV=.venv-vllm/bin/python
LOG="artifacts/runs/day2_tail4.log"; log() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
inj() { local i rc; for i in $(seq 1 6); do $PY scripts/run_injection.py "$@"; rc=$?; [ $rc -eq 0 ] && return 0; log "run_injection exit $rc, retry $i"; sleep 15; done; return 1; }
ag() { local i rc; for i in $(seq 1 3); do $PYV scripts/run_agentic.py "$@"; rc=$?; [ $rc -eq 0 ] && return 0; log "run_agentic exit $rc, retry $i"; sleep 30; done; return 1; }
while pgrep -f "run_agenti[c].py" >/dev/null; do sleep 30; done
log "tail4 start"
T4=(--types tc2_fake_turn tc2_cot_forgery tc2_fake_completion tc2_important --n-injected 100 --n-clean 8 --draws 1 --max-new-tokens 3000 --gpu-mem 0.85)
for st in sft rlvr; do
  log "record context via runner $st"
  ag --engine vllm --stage $st --run-id agentic01 --context record --scripted --rounds 3 "${T4[@]}" >> artifacts/runs/agentic01_${st}_record.log 2>&1
  log "webpage with record nouns $st"
  ag --engine vllm --stage $st --run-id agentic01 --context webpage --tag webpage_recnoun --payload-nouns record --scripted --rounds 3 "${T4[@]}" >> artifacts/runs/agentic01_${st}_webpage_recnoun.log 2>&1
done
log "second draw of none and role 4x arms"
TYPES=(tc2_fake_turn tc2_fake_completion tc2_important tc2_cot_forgery)
COMMON=(--types "${TYPES[@]}" --n-injected 60 --n-user-control 0 --n-clean 4 --batch-size 10 --max-new-tokens 3000 --max-items-per-process 100000 --no-acts --draw 2)
for st in rlvr sft; do
  a=$($PY -c "print({'sft':11.12,'rlvr':11.44}['$st'])")
  inj "${COMMON[@]}" --run-id inj_frame_def --stage $st --defense none >> artifacts/runs/inj_frame_def_$st.log 2>&1
  inj "${COMMON[@]}" --run-id inj_frame_def --stage $st --defense steer --steer-vector results/generated/roles_gpu/dir_${st}_b16.npy --steer-block 16 --steer-alpha $a >> artifacts/runs/inj_frame_def_$st.log 2>&1
done
log "tail4 complete"
