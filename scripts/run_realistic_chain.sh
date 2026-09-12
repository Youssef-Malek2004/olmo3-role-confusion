#!/usr/bin/env bash
# Mac pilot of the realistic goal set (toolcall, answer, link, deny): SFT, RLVR, then DPO if time.
# 60 injected (5 per goal x voice) + 8 user-control (2 per goal) + 6 clean per stage, one draw, cap 2,000.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUN="${RUN:-inj_real}"; BS="${BS:-4}"; CAP="${CAP:-2000}"; REST="${REST:-120}"; PER_PROC="${PER_PROC:-8}"
LOG="artifacts/runs/real_chain.log"; mkdir -p artifacts/runs
log() { echo "=== $(date '+%F %T') $*" >> "$LOG"; }
inj() { local i rc; for i in $(seq 1 15); do $PY scripts/run_injection.py "$@"; rc=$?; [ $rc -eq 0 ] && return 0; [ $rc -ne 3 ] && { log "exit $rc, retry $i"; }; sleep "$REST"; done; return 1; }
log "realistic chain start"
for stage in sft rlvr dpo; do
  log "realistic none $stage"
  inj --run-id "$RUN" --stage "$stage" --realistic --n-injected 60 --n-user-control 8 --n-clean 6 \
      --batch-size "$BS" --max-new-tokens "$CAP" --max-items-per-process "$PER_PROC" >> "artifacts/runs/${RUN}_${stage}.log" 2>&1
  $PY scripts/summarize_injection.py --run-id "$RUN" --effective-cap "$CAP" >> "artifacts/runs/${RUN}_summary.log" 2>&1
done
log "realistic chain complete"
