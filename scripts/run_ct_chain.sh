#!/usr/bin/env bash
# Copy-vs-trust (docs/COPY_VS_TRUST.md, section 11): can a role-steered model still copy tool content
# when the USER asks for it? Four conditions x 12 questions per arm; arms none / steer 4x / random 4x; SFT then RLVR.
# Vectors and alphas are the ones behind the headline defense arms (roles_gpu block 16; a11.12 sft, a11.44 rlvr).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; LOG=artifacts/runs/ct_chain.log; log() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
RUN="${RUN:-ct01}"; N="${N:-12}"; CAP="${CAP:-2000}"; BS="${BS:-4}"; ROLES="${ROLES:-roles_gpu}"
alpha_for() { case "$1" in sft) echo 11.12;; dpo) echo 11.29;; rlvr) echo 11.44;; esac; }
for stage in ${STAGES:-sft rlvr}; do
  A=$(alpha_for "$stage"); VEC="results/generated/${ROLES}/dir_${stage}_b16.npy"
  C=(scripts/run_copy_vs_trust.py --run-id "$RUN" --stage "$stage" --n "$N" --max-new-tokens "$CAP" --batch-size "$BS")
  log "$stage none";   $PY "${C[@]}" --arm none                                                   >> "artifacts/runs/ct_${stage}.log" 2>&1
  log "$stage steer";  $PY "${C[@]}" --arm steer --steer-vector "$VEC" --steer-block 16 --steer-alpha "$A" >> "artifacts/runs/ct_${stage}.log" 2>&1
  log "$stage random"; $PY "${C[@]}" --arm random --steer-block 16 --steer-alpha "$A" --random-seed 0 >> "artifacts/runs/ct_${stage}.log" 2>&1
  log "$stage report"; $PY scripts/run_copy_vs_trust.py --run-id "$RUN" --stage "$stage" --report | tee -a "$LOG"
done
log "ct chain complete"
