#!/usr/bin/env bash
# Copy-vs-trust dose response, reordered 2026-09-12 08:00 for time: SFT 2x, SFT 1x, then RLVR none + 2x as an optional tail.
# Same 12 questions x 4 conditions as ct01. Alphas are exact fractions of the headline 4x values (11.12 sft, 11.44 rlvr).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; LOG=artifacts/runs/ct_chain.log; log() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
RUN="${RUN:-ct01}"; N="${N:-12}"; CAP="${CAP:-2000}"; BS="${BS:-4}"; ROLES="${ROLES:-roles_gpu}"
run() { local stage=$1 arm=$2 alpha=${3:-0}
  local C=(scripts/run_copy_vs_trust.py --run-id "$RUN" --stage "$stage" --n "$N" --max-new-tokens "$CAP" --batch-size "$BS")
  if [ "$arm" = none ]; then $PY "${C[@]}" --arm none >> "artifacts/runs/ct_${stage}.log" 2>&1
  else $PY "${C[@]}" --arm steer --steer-vector "results/generated/${ROLES}/dir_${stage}_b16.npy" --steer-block 16 --steer-alpha "$alpha" >> "artifacts/runs/ct_${stage}.log" 2>&1; fi; }
log "sft steer 2x (alpha 5.56)";  run sft steer 5.56
log "sft report";                 $PY scripts/run_copy_vs_trust.py --run-id "$RUN" --stage sft --rescore | tee -a "$LOG"
log "sft steer 1x (alpha 2.78)";  run sft steer 2.78
log "sft report";                 $PY scripts/run_copy_vs_trust.py --run-id "$RUN" --stage sft --rescore | tee -a "$LOG"
log "sft dose complete"
log "rlvr none (optional tail)";  run rlvr none
log "rlvr steer 2x (alpha 5.72)"; run rlvr steer 5.72
log "rlvr report";                $PY scripts/run_copy_vs_trust.py --run-id "$RUN" --stage rlvr --rescore | tee -a "$LOG"
log "ct dose chain complete"
