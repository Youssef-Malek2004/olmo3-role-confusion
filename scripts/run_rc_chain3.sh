#!/usr/bin/env bash
# Throttled-GPU chain: small processes with cooling rests, defenses first.
#   1. RLVR defenses on the first 12 marker items: delimiter, steer (2x class gap), random
#   2. DPO marker + clean items (three-stage marker curve)
#   3. SFT defenses on the same 12 marker items
#   4. DPO remaining items
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ROLES="${ROLES:-roles01}"; MAIN="${MAIN:-inj_main}"
BS="${BS:-4}"; CAP="${CAP:-3600}"; BLOCK="${BLOCK:-16}"; REST="${REST:-240}"; PER_PROC="${PER_PROC:-8}"
LOG="artifacts/runs/rc_chain3.log"
log() { echo "=== $(date '+%F %T') $*" >> "$LOG"; }
inj() { local i; for i in $(seq 1 12); do $PY scripts/run_injection.py "$@"; local rc=$?; [ $rc -ne 3 ] && return $rc; log "rest ${REST}s"; sleep "$REST"; done; return 1; }
COMMON=(--run-id "$MAIN" --n-injected 54 --n-user-control 12 --n-clean 8 --batch-size "$BS" --max-new-tokens "$CAP" --max-items-per-process "$PER_PROC")
alpha_for() { $PY -c "import json;print(round(json.load(open('results/generated/$ROLES/role_probe.json'))['stages']['$1']['blocks']['$BLOCK']['diffmean_norm']*2,2))"; }
# first 12 marker items by item index (deterministic across stages/defenses)
MARKER12=$(cat data/processed/marker12_item_ids.txt)
log "chain3 start; marker12: $MARKER12"

for stage in rlvr sft; do
  A=$(alpha_for $stage); SDIR="results/generated/${ROLES}/dir_${stage}_b${BLOCK}.npy"
  log "defenses $stage (alpha $A)"
  inj "${COMMON[@]}" --stage "$stage" --item-ids $MARKER12 --defense delimiter >> "artifacts/runs/${MAIN}_${stage}.log" 2>&1; log "rest"; sleep "$REST"
  inj "${COMMON[@]}" --stage "$stage" --item-ids $MARKER12 --defense steer --steer-vector "$SDIR" --steer-block "$BLOCK" --steer-alpha "$A" >> "artifacts/runs/${MAIN}_${stage}.log" 2>&1; log "rest"; sleep "$REST"
  inj "${COMMON[@]}" --stage "$stage" --item-ids $MARKER12 --defense random --steer-block "$BLOCK" --steer-alpha "$A" --random-seed 0 >> "artifacts/runs/${MAIN}_${stage}.log" 2>&1; log "rest"; sleep "$REST"
  $PY scripts/summarize_injection.py --run-id "$MAIN" >> "artifacts/runs/${MAIN}_summary.log" 2>&1
  if [ "$stage" = "rlvr" ]; then
    log "dpo marker + clean"
    inj "${COMMON[@]}" --stage dpo --conditions injected clean --itypes marker >> "artifacts/runs/${MAIN}_dpo.log" 2>&1; log "rest"; sleep "$REST"
    $PY scripts/summarize_injection.py --run-id "$MAIN" >> "artifacts/runs/${MAIN}_summary.log" 2>&1
  fi
done
log "dpo remaining"
inj "${COMMON[@]}" --stage dpo >> "artifacts/runs/${MAIN}_dpo.log" 2>&1
log "final summaries"
$PY scripts/summarize_injection.py --run-id "$MAIN" >> "artifacts/runs/${MAIN}_summary.log" 2>&1
$PY scripts/fit_role_probe.py --run-id "$ROLES" --stages sft dpo rlvr --inj-run "$MAIN" >> "artifacts/runs/${ROLES}_fit_main.log" 2>&1
log "chain3 complete"
