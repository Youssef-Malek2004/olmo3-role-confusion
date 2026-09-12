#!/usr/bin/env bash
# Reduced role-confusion chain for a throttled GPU (~2x slower than the morning benchmarks).
# Priorities: finish RLVR injected items -> DPO marker curve -> RLVR defenses on marker items -> SFT defenses if time.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ROLES="${ROLES:-roles01}"; MAIN="${MAIN:-inj_main}"
BS="${BS:-4}"; CAP="${CAP:-3600}"; BLOCK="${BLOCK:-16}"
LOG="artifacts/runs/rc_chain2.log"
log() { echo "=== $(date '+%F %T') $*" >> "$LOG"; }
inj() { local i; for i in $(seq 1 12); do $PY scripts/run_injection.py "$@"; local rc=$?; [ $rc -ne 3 ] && return $rc; done; return 1; }
COMMON=(--run-id "$MAIN" --n-injected 54 --n-user-control 12 --n-clean 8 --batch-size "$BS" --max-new-tokens "$CAP" --max-items-per-process 12)

# steering scale = the class gap (norm of env-minus-user mean difference at the chosen block) for each stage
alpha_for() { $PY -c "import json;print(round(json.load(open('results/generated/$ROLES/role_probe.json'))['stages']['$1']['blocks']['$BLOCK']['diffmean_norm']*2,2))"; }

log "chain2 start"
log "rlvr remaining injected"
inj "${COMMON[@]}" --stage rlvr --conditions injected >> "artifacts/runs/${MAIN}_rlvr.log" 2>&1

log "dpo marker + clean"
inj "${COMMON[@]}" --stage dpo --conditions injected clean --itypes marker >> "artifacts/runs/${MAIN}_dpo.log" 2>&1

for stage in rlvr sft; do
  A=$(alpha_for $stage); SDIR="results/generated/${ROLES}/dir_${stage}_b${BLOCK}.npy"
  log "defenses $stage marker items (alpha $A)"
  inj "${COMMON[@]}" --stage "$stage" --conditions injected --itypes marker --defense delimiter >> "artifacts/runs/${MAIN}_${stage}.log" 2>&1
  inj "${COMMON[@]}" --stage "$stage" --conditions injected --itypes marker --defense steer --steer-vector "$SDIR" --steer-block "$BLOCK" --steer-alpha "$A" >> "artifacts/runs/${MAIN}_${stage}.log" 2>&1
  inj "${COMMON[@]}" --stage "$stage" --conditions injected --itypes marker --defense random --steer-block "$BLOCK" --steer-alpha "$A" --random-seed 0 >> "artifacts/runs/${MAIN}_${stage}.log" 2>&1
  log "summaries after $stage"
  $PY scripts/summarize_injection.py --run-id "$MAIN" >> "artifacts/runs/${MAIN}_summary.log" 2>&1
done

log "dpo remaining injected (format, exfil) + user controls"
inj "${COMMON[@]}" --stage dpo >> "artifacts/runs/${MAIN}_dpo.log" 2>&1
log "final summaries"
$PY scripts/summarize_injection.py --run-id "$MAIN" >> "artifacts/runs/${MAIN}_summary.log" 2>&1
$PY scripts/fit_role_probe.py --run-id "$ROLES" --stages sft dpo rlvr --inj-run "$MAIN" >> "artifacts/runs/${ROLES}_fit_main.log" 2>&1
log "chain2 complete"
