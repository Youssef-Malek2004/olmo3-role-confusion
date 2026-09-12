#!/usr/bin/env bash
# Role-confusion main chain. One model in memory at a time; batch sizes chosen for ~22 GB peak.
#   1. capture role activations (sft, dpo, rlvr)          ~10 min each
#   2. fit role probes + directions
#   3. main injection eval, no defense (sft, dpo, rlvr)    ~1 h each
#   4. steering alpha sweep on a dev subset (rlvr)         ~15 min per alpha
#   5. defenses on injected items: delimiter, steer, random (sft, rlvr)
#   6. summaries
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ROLES="${ROLES:-roles01}"; MAIN="${MAIN:-inj_main}"; SWEEP="${SWEEP:-inj_sweep}"
N_INJ="${N_INJ:-54}"; N_UC="${N_UC:-12}"; N_CLEAN="${N_CLEAN:-8}"; BS="${BS:-4}"; CAP="${CAP:-3600}"
BLOCK="${BLOCK:-16}"; ALPHAS="${ALPHAS:-4 8 16}"
LOG="artifacts/runs/rc_chain.log"; mkdir -p artifacts/runs
log() { echo "=== $(date '+%F %T') $*" >> "$LOG"; }
# run_injection exits 3 when it stopped early to reset the MPS graph cache; loop until it exits 0 (max 12 passes)
inj() { local i; for i in $(seq 1 12); do $PY scripts/run_injection.py "$@"; local rc=$?; [ $rc -ne 3 ] && return $rc; done; return 1; }

log "chain start"
for stage in sft dpo rlvr; do
  log "capture roles $stage"
  $PY scripts/capture_roles.py --run-id "$ROLES" --stage "$stage" --n-questions 150 --batch-size 8 >> "artifacts/runs/${ROLES}_${stage}.log" 2>&1
done
log "fit role probe"
$PY scripts/fit_role_probe.py --run-id "$ROLES" --stages sft dpo rlvr --inj-run inj_pilot >> "artifacts/runs/${ROLES}_fit.log" 2>&1

for stage in sft rlvr dpo; do
  log "main injection none $stage"
  inj --run-id "$MAIN" --stage "$stage" --n-injected "$N_INJ" --n-user-control "$N_UC" --n-clean "$N_CLEAN" \
      --batch-size "$BS" --max-new-tokens "$CAP" >> "artifacts/runs/${MAIN}_${stage}.log" 2>&1
done

# steering sweep on a dev subset: first 18 injected items, RLVR's own direction
DIR="results/generated/${ROLES}/dir_rlvr_b${BLOCK}.npy"
log "steer sweep rlvr"
inj --run-id "$SWEEP" --stage rlvr --n-injected 18 --n-user-control 0 --n-clean 0 --conditions injected \
    --batch-size "$BS" --max-new-tokens "$CAP" >> "artifacts/runs/${SWEEP}.log" 2>&1
for a in $ALPHAS; do
  inj --run-id "$SWEEP" --stage rlvr --n-injected 18 --n-user-control 0 --n-clean 0 --conditions injected \
      --defense steer --steer-vector "$DIR" --steer-block "$BLOCK" --steer-alpha "$a" --batch-size "$BS" --max-new-tokens "$CAP" >> "artifacts/runs/${SWEEP}.log" 2>&1
done
ALPHA=$($PY scripts/select_alpha.py --run-id "$SWEEP" --stage rlvr --block "$BLOCK" --alphas $ALPHAS)
log "chosen alpha: $ALPHA"
[ "$ALPHA" = "none" ] && ALPHA=8

for stage in rlvr sft; do
  SDIR="results/generated/${ROLES}/dir_${stage}_b${BLOCK}.npy"
  log "defenses $stage (alpha $ALPHA)"
  inj --run-id "$MAIN" --stage "$stage" --n-injected "$N_INJ" --n-user-control "$N_UC" --n-clean "$N_CLEAN" --conditions injected clean \
      --defense delimiter --batch-size "$BS" --max-new-tokens "$CAP" >> "artifacts/runs/${MAIN}_${stage}.log" 2>&1
  inj --run-id "$MAIN" --stage "$stage" --n-injected "$N_INJ" --n-user-control "$N_UC" --n-clean "$N_CLEAN" --conditions injected clean \
      --defense steer --steer-vector "$SDIR" --steer-block "$BLOCK" --steer-alpha "$ALPHA" --batch-size "$BS" --max-new-tokens "$CAP" >> "artifacts/runs/${MAIN}_${stage}.log" 2>&1
  inj --run-id "$MAIN" --stage "$stage" --n-injected "$N_INJ" --n-user-control "$N_UC" --n-clean "$N_CLEAN" --conditions injected clean \
      --defense random --steer-block "$BLOCK" --steer-alpha "$ALPHA" --random-seed 0 --batch-size "$BS" --max-new-tokens "$CAP" >> "artifacts/runs/${MAIN}_${stage}.log" 2>&1
done

log "summaries"
$PY scripts/summarize_injection.py --run-id "$MAIN" >> "artifacts/runs/${MAIN}_summary.log" 2>&1
$PY scripts/summarize_injection.py --run-id "$SWEEP" >> "artifacts/runs/${SWEEP}_summary.log" 2>&1
$PY scripts/fit_role_probe.py --run-id "$ROLES" --stages sft dpo rlvr --inj-run "$MAIN" >> "artifacts/runs/${ROLES}_fit_main.log" 2>&1
log "chain complete"
