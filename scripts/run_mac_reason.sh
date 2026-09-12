#!/usr/bin/env bash
# Does letting Instruct reason (reason() tool after the fetch) reduce its gullibility? Instruct RL, record_final for answers in both arms.
# reason=none vs reason=tool, x {webpage agentic, record scripted}, 4 payloads x 12 + 4 clean. Then Think SFT no-think mirror arms.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; LOG=artifacts/runs/mac_reason.log; log() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
T=(--engine hf --stage instruct --run-id agentic_reason --final-tool --types tc2_fake_turn tc2_cot_forgery tc2_important answer --n-injected 48 --n-clean 4 --rounds 3 --batch-size 4)
log "webpage reason=none"; $PY scripts/run_agentic.py "${T[@]}" --context webpage --ask soft --reason none --max-new-tokens 1500 --tag web_noreason >> artifacts/runs/agentic_reason.log 2>&1
log "webpage reason=tool"; $PY scripts/run_agentic.py "${T[@]}" --context webpage --ask soft --reason tool --max-new-tokens 2200 --tag web_reason >> artifacts/runs/agentic_reason.log 2>&1
log "record reason=none";  $PY scripts/run_agentic.py "${T[@]}" --context record --scripted --ask none --reason none --max-new-tokens 1500 --tag rec_noreason >> artifacts/runs/agentic_reason.log 2>&1
log "record reason=tool";  $PY scripts/run_agentic.py "${T[@]}" --context record --scripted --ask none --reason tool --max-new-tokens 2200 --tag rec_reason >> artifacts/runs/agentic_reason.log 2>&1
K=(--engine hf --stage sft --run-id agentic_nothink --types tc2_fake_turn tc2_cot_forgery tc2_important answer --n-injected 48 --n-clean 4 --rounds 3 --batch-size 4 --scripted --ask none --no-think --max-new-tokens 1200)
log "think sft record no-think";  $PY scripts/run_agentic.py "${K[@]}" --context record  --tag rec_nothink >> artifacts/runs/agentic_nothink.log 2>&1
log "think sft webpage no-think"; $PY scripts/run_agentic.py "${K[@]}" --context webpage --tag web_nothink >> artifacts/runs/agentic_nothink.log 2>&1
log "reason battery complete"
