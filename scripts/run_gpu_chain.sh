#!/usr/bin/env bash
# Full experiment chain for an 80 GB CUDA GPU. Run inside tmux on the pod:
#   tmux new -d -s rc 'bash scripts/run_gpu_chain.sh'
# Steps (each restartable; rerunning skips finished rows):
#   1 role captures + probes (sft, dpo, rlvr)
#   2 injection eval, no defense, 3 draws, all three stages
#   3 defenses on marker + clean items (1 draw): delimiter, steer x{1,2,4} class gap, random matched norm — rlvr, sft
#   4 dpo defenses (delimiter, steer 2x, random)
#   5 summaries (uniform effective cap = CAP)
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
ROLES="${ROLES:-roles_gpu}"; MAIN="${MAIN:-inj_gpu}"
N_INJ="${N_INJ:-90}"; N_UC="${N_UC:-12}"; N_CLEAN="${N_CLEAN:-8}"; DRAWS="${DRAWS:-3}"
BS="${BS:-12}"; CAP="${CAP:-5000}"; BLOCK="${BLOCK:-16}"
LOG="artifacts/runs/gpu_chain.log"; mkdir -p artifacts/runs
log() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
# retry until run_injection exits 0 (it resumes from finished rows); guards against transient OOM
inj() { local i rc; for i in $(seq 1 6); do $PY scripts/run_injection.py "$@"; rc=$?; [ $rc -eq 0 ] && return 0; log "run_injection exit $rc, retry $i"; sleep 10; done; return 1; }
COMMON=(--run-id "$MAIN" --n-injected "$N_INJ" --n-user-control "$N_UC" --n-clean "$N_CLEAN" --batch-size "$BS" --max-new-tokens "$CAP" --max-items-per-process 100000)
alpha_for() { $PY -c "import json;print(round(json.load(open('results/generated/$ROLES/role_probe.json'))['stages']['$1']['blocks']['$BLOCK']['diffmean_norm']*$2,2))"; }

log "chain start: N_INJ=$N_INJ DRAWS=$DRAWS BS=$BS CAP=$CAP"
for stage in sft dpo rlvr; do
  log "roles $stage"
  $PY scripts/capture_roles.py --run-id "$ROLES" --stage "$stage" --n-questions 150 --batch-size 16 >> "artifacts/runs/${ROLES}_${stage}.log" 2>&1
done
log "fit role probe"
$PY scripts/fit_role_probe.py --run-id "$ROLES" --stages sft dpo rlvr >> "artifacts/runs/${ROLES}_fit.log" 2>&1

for stage in sft rlvr dpo; do
  for d in $(seq 1 "$DRAWS"); do
    log "injection none $stage draw $d"
    inj "${COMMON[@]}" --stage "$stage" --draw "$d" >> "artifacts/runs/${MAIN}_${stage}.log" 2>&1
  done
  $PY scripts/summarize_injection.py --run-id "$MAIN" --effective-cap "$CAP" >> "artifacts/runs/${MAIN}_summary.log" 2>&1
done

for stage in rlvr sft; do
  SDIR="results/generated/${ROLES}/dir_${stage}_b${BLOCK}.npy"
  log "defenses $stage"
  inj "${COMMON[@]}" --stage "$stage" --conditions injected clean --itypes marker --defense delimiter >> "artifacts/runs/${MAIN}_${stage}.log" 2>&1
  MULTS="2"; [ "$stage" = "rlvr" ] && MULTS="1 2 4"
  for mult in $MULTS; do
    A=$(alpha_for $stage $mult)
    inj "${COMMON[@]}" --stage "$stage" --conditions injected clean --itypes marker --defense steer --steer-vector "$SDIR" --steer-block "$BLOCK" --steer-alpha "$A" >> "artifacts/runs/${MAIN}_${stage}.log" 2>&1
  done
  A=$(alpha_for $stage 2)
  inj "${COMMON[@]}" --stage "$stage" --conditions injected clean --itypes marker --defense random --steer-block "$BLOCK" --steer-alpha "$A" --random-seed 0 >> "artifacts/runs/${MAIN}_${stage}.log" 2>&1
  $PY scripts/summarize_injection.py --run-id "$MAIN" --effective-cap "$CAP" >> "artifacts/runs/${MAIN}_summary.log" 2>&1
done

log "dpo defenses"
SDIR="results/generated/${ROLES}/dir_dpo_b${BLOCK}.npy"; A=$(alpha_for dpo 2)
inj "${COMMON[@]}" --stage dpo --conditions injected clean --itypes marker --defense delimiter >> "artifacts/runs/${MAIN}_dpo.log" 2>&1
inj "${COMMON[@]}" --stage dpo --conditions injected clean --itypes marker --defense steer --steer-vector "$SDIR" --steer-block "$BLOCK" --steer-alpha "$A" >> "artifacts/runs/${MAIN}_dpo.log" 2>&1
inj "${COMMON[@]}" --stage dpo --conditions injected clean --itypes marker --defense random --steer-block "$BLOCK" --steer-alpha "$A" --random-seed 0 >> "artifacts/runs/${MAIN}_dpo.log" 2>&1

log "final summaries"
$PY scripts/summarize_injection.py --run-id "$MAIN" --effective-cap "$CAP" >> "artifacts/runs/${MAIN}_summary.log" 2>&1
$PY scripts/fit_role_probe.py --run-id "$ROLES" --stages sft dpo rlvr --inj-run "$MAIN" >> "artifacts/runs/${ROLES}_fit_main.log" 2>&1
log "chain complete"
