#!/usr/bin/env bash
# Post-Step-0 queue (gated on "day2 vllm complete"), ordered by value per GPU-hour:
#  1 HF  random vector at norm 16.6, RLVR, same items as inj_frame_def (decides whether the style-4x fake-turn effect is norm or direction)
#  2 vLLM Instruct agentic runs: web page + email, 7 payload types x 25 items + 12 clean, 2 draws, cap 1500
#  3 HF  legitimate-tool-use utility: 30 legit + 4 clean items, arms none / role 4x, SFT and RLVR
#  4 HF  role 2x arms on the inj_frame_def items, RLVR and SFT (additivity baseline for the sum arm)
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
PY=.venv/bin/python; PYV=.venv-vllm/bin/python
LOG="artifacts/runs/day2_tail.log"; log() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
inj() { local i rc; for i in $(seq 1 6); do $PY scripts/run_injection.py "$@"; rc=$?; [ $rc -eq 0 ] && return 0; log "run_injection exit $rc, retry $i"; sleep 15; done; return 1; }
ag() { local i rc; for i in $(seq 1 3); do $PYV scripts/run_agentic.py "$@"; rc=$?; [ $rc -eq 0 ] && return 0; log "run_agentic exit $rc, retry $i"; sleep 20; done; return 1; }
TYPES=(tc2_fake_turn tc2_fake_completion tc2_important tc2_cot_forgery)
COMMON=(--types "${TYPES[@]}" --n-injected 60 --n-user-control 0 --n-clean 4 --batch-size 10 --max-new-tokens 3000 --max-items-per-process 100000 --no-acts)
while ! grep -q "day2 vllm complete" artifacts/runs/day2_vllm.log 2>/dev/null; do sleep 60; done
log "tail start"
log "1 random norm 16.6 rlvr"
inj "${COMMON[@]}" --run-id inj_frame_def --stage rlvr --defense random --steer-block 16 --steer-alpha 16.6 --random-seed 1 >> artifacts/runs/inj_frame_def_rlvr.log 2>&1
log "2 instruct agentic (vllm)"
for ctx in webpage email; do
  ag --engine vllm --stage instruct --run-id agentic01 --context $ctx --ask soft --rounds 3 --n-injected 200 --n-clean 12 --draws 2 --max-new-tokens 1500 --gpu-mem 0.85 >> artifacts/runs/agentic01_instruct_$ctx.log 2>&1
done
log "3 legit utility"
for st in sft rlvr; do
  a=$($PY -c "print({'sft':11.12,'rlvr':11.44}['$st'])")
  inj --n-injected 0 --n-user-control 0 --n-clean 4 --n-legit 30 --realistic --batch-size 10 --max-new-tokens 3000 --max-items-per-process 100000 --no-acts --run-id inj_legit --stage $st >> artifacts/runs/inj_legit_$st.log 2>&1
  inj --n-injected 0 --n-user-control 0 --n-clean 4 --n-legit 30 --realistic --batch-size 10 --max-new-tokens 3000 --max-items-per-process 100000 --no-acts --run-id inj_legit --stage $st --defense steer --steer-vector results/generated/roles_gpu/dir_${st}_b16.npy --steer-block 16 --steer-alpha $a >> artifacts/runs/inj_legit_$st.log 2>&1
done
log "4 role 2x arms"
for st in rlvr sft; do
  a=$($PY -c "print({'sft':5.56,'rlvr':5.72}['$st'])")
  inj "${COMMON[@]}" --run-id inj_frame_def --stage $st --defense steer --steer-vector results/generated/roles_gpu/dir_${st}_b16.npy --steer-block 16 --steer-alpha $a >> artifacts/runs/inj_frame_def_$st.log 2>&1
done
log "5 think scripted contexts (vllm)"
for st in sft rlvr; do for ctx in webpage email; do
  ag --engine vllm --stage $st --run-id agentic01 --context $ctx --scripted --rounds 3 --n-injected 200 --n-clean 12 --draws 1 --max-new-tokens 3000 --gpu-mem 0.85 >> artifacts/runs/agentic01_${st}_$ctx.log 2>&1
done; done
log "summaries"
$PY scripts/summarize_injection.py --run-id inj_legit --effective-cap 3000 >> artifacts/runs/inj_legit_summary.log 2>&1
log "tail complete"
