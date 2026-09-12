#!/usr/bin/env bash
# Heavily throttled GPU: cap 2,000 for the remaining defense/DPO runs; compare at a uniform effective cap of 2,000.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ROLES="${ROLES:-roles01}"; MAIN="${MAIN:-inj_main}"
BS="${BS:-4}"; CAP="${CAP:-2000}"; BLOCK="${BLOCK:-16}"; REST="${REST:-180}"; PER_PROC="${PER_PROC:-8}"
LOG="artifacts/runs/rc_chain4.log"
log() { echo "=== $(date '+%F %T') $*" >> "$LOG"; }
inj() { local i; for i in $(seq 1 12); do $PY scripts/run_injection.py "$@"; local rc=$?; [ $rc -ne 3 ] && return $rc; log "rest ${REST}s"; sleep "$REST"; done; return 1; }
COMMON=(--run-id "$MAIN" --n-injected 54 --n-user-control 12 --n-clean 8 --batch-size "$BS" --max-new-tokens "$CAP" --max-items-per-process "$PER_PROC")
alpha_for() { $PY -c "import json;print(round(json.load(open('results/generated/$ROLES/role_probe.json'))['stages']['$1']['blocks']['$BLOCK']['diffmean_norm']*2,2))"; }
MARKER12=$(cat data/processed/marker12_item_ids.txt)
log "chain4 start (cap $CAP)"
for stage in rlvr sft; do
  A=$(alpha_for $stage); SDIR="results/generated/${ROLES}/dir_${stage}_b${BLOCK}.npy"
  log "defenses $stage (alpha $A)"
  inj "${COMMON[@]}" --stage "$stage" --item-ids $MARKER12 --defense delimiter >> "artifacts/runs/${MAIN}_${stage}.log" 2>&1; sleep "$REST"
  inj "${COMMON[@]}" --stage "$stage" --item-ids $MARKER12 --defense steer --steer-vector "$SDIR" --steer-block "$BLOCK" --steer-alpha "$A" >> "artifacts/runs/${MAIN}_${stage}.log" 2>&1; sleep "$REST"
  inj "${COMMON[@]}" --stage "$stage" --item-ids $MARKER12 --defense random --steer-block "$BLOCK" --steer-alpha "$A" --random-seed 0 >> "artifacts/runs/${MAIN}_${stage}.log" 2>&1; sleep "$REST"
  $PY scripts/summarize_injection.py --run-id "$MAIN" --effective-cap 2000 >> "artifacts/runs/${MAIN}_summary.log" 2>&1
  if [ "$stage" = "rlvr" ]; then
    log "dpo marker + clean"
    inj "${COMMON[@]}" --stage dpo --conditions injected clean --itypes marker >> "artifacts/runs/${MAIN}_dpo.log" 2>&1; sleep "$REST"
  fi
done
log "dpo remaining"
inj "${COMMON[@]}" --stage dpo >> "artifacts/runs/${MAIN}_dpo.log" 2>&1
log "final summaries"
$PY scripts/summarize_injection.py --run-id "$MAIN" --effective-cap 2000 >> "artifacts/runs/${MAIN}_summary.log" 2>&1
$PY scripts/fit_role_probe.py --run-id "$ROLES" --stages sft dpo rlvr --inj-run "$MAIN" >> "artifacts/runs/${ROLES}_fit_main.log" 2>&1
log "chain4 complete"
