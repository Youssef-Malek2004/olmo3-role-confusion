#!/usr/bin/env bash
# Mac: Instruct role probe + agentic web-page run + span scoring on the Instruct and Think role directions.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; LOG=artifacts/runs/mac_instruct_chain.log; log() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
log "agentic instruct webpage, 7 types x 6 + 4 clean"
$PY scripts/run_agentic.py --engine hf --stage instruct --run-id agentic_pilot --context webpage --tag webpage7 --ask soft --rounds 3 --n-injected 42 --n-clean 4 --batch-size 4 --max-new-tokens 1500 >> artifacts/runs/agentic_pilot_instruct7.log 2>&1
log "role capture instruct"
$PY scripts/capture_roles.py --run-id roles_mac --stage instruct --n-questions 60 --batch-size 6 >> artifacts/runs/roles_mac_instruct.log 2>&1
log "fit probe"
$PY scripts/fit_role_probe.py --run-id roles_mac --stages instruct >> artifacts/runs/roles_mac_fit.log 2>&1
log "span scores: instruct dir, think sft dir"
$PY scripts/score_agentic_spans.py --run-id agentic_pilot --stage instruct --tag webpage7 --direction results/generated/roles_mac/dir_instruct_b16.npy --name instruct_role >> artifacts/runs/agentic_spans_instruct.log 2>&1
$PY scripts/score_agentic_spans.py --run-id agentic_pilot --stage instruct --tag webpage7 --direction results/generated/roles_gpu/dir_sft_b16.npy --name think_sft_role >> artifacts/runs/agentic_spans_instruct.log 2>&1
$PY - <<'PYEOF' >> artifacts/runs/agentic_spans_instruct.log 2>&1
import numpy as np
for b in (8,16,24):
    try:
        di=np.load(f"results/generated/roles_mac/dir_instruct_b{b}.npy"); 
        for st in ("sft","dpo","rlvr"):
            dt=np.load(f"results/generated/roles_gpu/dir_{st}_b{b}.npy"); print(f"block {b}: cos(instruct, think-{st}) = {float(di@dt/np.linalg.norm(di)/np.linalg.norm(dt)):.3f}")
    except FileNotFoundError as e: print("missing", e)
PYEOF
log "mac instruct chain complete"
