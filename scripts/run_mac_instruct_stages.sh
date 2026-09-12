#!/usr/bin/env bash
# Mac unattended: Instruct stage comparison (SFT, DPO, released RL) on the agentic web-page and email contexts, plus role probes.
# Waits for the form-vs-identity diagnostic and for the checkpoint downloads to finish.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; LOG=artifacts/runs/mac_instruct_stages.log; log() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
while ! grep -q "mac ctxdiag2 complete" artifacts/runs/mac_ctxdiag2.log 2>/dev/null; do sleep 60; done
while pgrep -f "pin_and_downloa[d].py" >/dev/null; do sleep 60; done
log "instruct stages start"
for st in instruct_sft instruct_dpo instruct; do
  for ctx in webpage email; do
    log "agentic $st $ctx"
    $PY scripts/run_agentic.py --engine hf --stage $st --run-id agentic_stages --context $ctx --ask soft --rounds 3 --n-injected 96 --n-clean 4 --batch-size 4 --max-new-tokens 1500 >> artifacts/runs/agentic_stages_$st.log 2>&1
  done
done
for st in instruct_sft instruct_dpo; do
  log "roles $st"
  $PY scripts/capture_roles.py --run-id roles_mac --stage $st --n-questions 60 --batch-size 6 >> artifacts/runs/roles_mac_$st.log 2>&1
done
log "fit probes"
$PY scripts/fit_role_probe.py --run-id roles_mac --stages instruct_sft instruct_dpo instruct >> artifacts/runs/roles_mac_fit.log 2>&1
$PY - <<'PYEOF' >> artifacts/runs/roles_mac_fit.log 2>&1
import numpy as np, itertools
L={"i_sft":"roles_mac/dir_instruct_sft","i_dpo":"roles_mac/dir_instruct_dpo","i_rl":"roles_mac/dir_instruct","t_sft":"roles_gpu/dir_sft","t_rlvr":"roles_gpu/dir_rlvr"}
for b in (8,16,24):
    v={k:np.load(f"results/generated/{p}_b{b}.npy") for k,p in L.items()}
    v={k:x/np.linalg.norm(x) for k,x in v.items()}
    print(f"block {b}: " + ", ".join(f"cos({a},{c})={float(v[a]@v[c]):.3f}" for a,c in itertools.combinations(v,2)))
PYEOF
log "instruct stages complete"
